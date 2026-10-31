"""字体注册表：内置字体 + 用户字体目录；按 (字体, 字号) 缓存 FreeTypeFont；缺字检测；排版度量。

要点：
- 内置的思源黑体（Noto Sans SC/TC）是可变字体，默认实例是 Thin(100)，必须用
  ``set_variation_by_axes`` 设字重；每个 字号×字重 组合缓存一个 FreeTypeFont。
- Windows 上 Pillow 遇到非 ASCII 路径会把整个字体读进内存（Python bytes + C 拷贝各一份，
  17MB 的思源黑体每个字号约 34MB）。所以加载前先换成 ASCII 路径：8.3 短路径 →
  复制到 ASCII 缓存目录 → 实在不行才按 bytes 加载并收紧缓存数量。
- 缺字检测直接解析字体的 cmap 表（不依赖 fontTools），用户字体的中文显示名解析 name 表。
- FreeType 渲染不释放 GIL（已实测），FreeTypeFont 对象可以在多个线程间共享；
  这里的缓存字典用锁保护。
"""
from __future__ import annotations

import ctypes
import hashlib
import logging
import os
import shutil
import struct
import tempfile
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

from PIL import ImageFont

log = logging.getLogger("hanhua.typeset")

DEFAULT_FONT = "noto-sans-sc-bold"
FONT_EXTS = (".ttf", ".otf", ".ttc", ".otc")
REF_SIZE = 128                      # 排版度量用的参考字号（advance / ink 框按比例缩放）
_MAX_CACHED_FONTS = 160             # 路径加载的 FreeTypeFont 缓存上限（每个只占几百 KB）
_MAX_CACHED_BYTES_FONTS = 6         # 退化为 bytes 加载时的上限（每个约 2× 文件大小）
_ZH_LANGS = (0x0804, 0x1004, 0x0404, 0x0C04, 0x1404)   # zh-CN / zh-SG / zh-TW / zh-HK / zh-MO
_STYLE_ZH = {
    "regular": "", "normal": "", "book": "", "roman": "",
    "thin": "极细", "extralight": "特细", "ultralight": "特细", "light": "细",
    "medium": "中", "semibold": "半粗", "demibold": "半粗", "bold": "粗",
    "extrabold": "特粗", "ultrabold": "特粗", "black": "特粗", "heavy": "特粗",
}


@dataclass(frozen=True)
class _Builtin:
    id: str
    file: str
    name: str
    weight: Optional[int]
    tags: tuple[str, ...]


BUILTIN_FONTS: tuple[_Builtin, ...] = (
    _Builtin("noto-sans-sc-medium", "NotoSansSC-VF.ttf", "思源黑体 中", 500, ("builtin", "sans", "sc", "dialog")),
    _Builtin("noto-sans-sc-bold", "NotoSansSC-VF.ttf", "思源黑体 粗", 700, ("builtin", "sans", "sc", "dialog")),
    _Builtin("noto-sans-sc-black", "NotoSansSC-VF.ttf", "思源黑体 特粗", 900, ("builtin", "sans", "sc", "free")),
    _Builtin("noto-sans-tc-medium", "NotoSansTC-VF.ttf", "思源黑体 繁 中", 500, ("builtin", "sans", "tc", "dialog")),
    _Builtin("noto-sans-tc-bold", "NotoSansTC-VF.ttf", "思源黑体 繁 粗", 700, ("builtin", "sans", "tc", "dialog")),
    _Builtin("noto-sans-tc-black", "NotoSansTC-VF.ttf", "思源黑体 繁 特粗", 900, ("builtin", "sans", "tc", "free")),
    _Builtin("lxgw-wenkai", "LXGWWenKai-Regular.ttf", "霞鹜文楷", None, ("builtin", "kai", "sc", "tc")),
    _Builtin("zcool-kuaile", "ZCOOLKuaiLe-Regular.ttf", "站酷快乐体", None, ("builtin", "display", "sc", "sfx")),
    _Builtin("ma-shan-zheng", "MaShanZheng-Regular.ttf", "马善政毛笔", None, ("builtin", "brush", "sc", "sfx")),
)
_BUILTIN_IDS = {b.id for b in BUILTIN_FONTS}


# --------------------------------------------------------------------------------------
# sfnt 解析（cmap / name / fvar），只读需要的表，不依赖第三方库
# --------------------------------------------------------------------------------------
def _face_offsets(f) -> list[int]:
    """单字体返回 [0]；TTC/OTC 返回每个 face 的表目录偏移。"""
    f.seek(0)
    head = f.read(12)
    if len(head) < 12:
        raise ValueError("字体文件太短")
    if head[:4] == b"ttcf":
        num = struct.unpack(">I", head[8:12])[0]
        if not 0 < num <= 256:
            raise ValueError("TTC 字体数量异常")
        return list(struct.unpack(f">{num}I", f.read(4 * num)))
    return [0]


def _table_dir(f, offset: int) -> dict[str, tuple[int, int]]:
    f.seek(offset)
    hdr = f.read(12)
    if len(hdr) < 12:
        raise ValueError("字体表目录损坏")
    num = struct.unpack(">H", hdr[4:6])[0]
    raw = f.read(16 * num)
    tables: dict[str, tuple[int, int]] = {}
    for i in range(min(num, len(raw) // 16)):
        tag, _ck, off, length = struct.unpack(">4sIII", raw[i * 16:(i + 1) * 16])
        tables[tag.decode("latin-1")] = (off, length)
    return tables


def _read_table(f, tables: dict[str, tuple[int, int]], tag: str, limit: int = 32 << 20) -> bytes:
    if tag not in tables:
        return b""
    off, length = tables[tag]
    if length <= 0 or length > limit:
        return b""
    f.seek(off)
    return f.read(length)


def _cmap_fmt4(d: bytes, out: set[int]) -> None:
    seg2 = struct.unpack(">H", d[6:8])[0]
    n = seg2 // 2
    ends = struct.unpack(f">{n}H", d[14:14 + seg2])
    p = 16 + seg2
    starts = struct.unpack(f">{n}H", d[p:p + seg2])
    deltas = struct.unpack(f">{n}h", d[p + seg2:p + 2 * seg2])
    ro_pos = p + 2 * seg2
    roffs = struct.unpack(f">{n}H", d[ro_pos:ro_pos + seg2])
    for i in range(n):
        s, e = starts[i], ends[i]
        if s == 0xFFFF or e < s:
            continue
        if roffs[i] == 0:
            for c in range(s, e + 1):
                if (c + deltas[i]) & 0xFFFF:
                    out.add(c)
            continue
        base = ro_pos + 2 * i + roffs[i]
        for c in range(s, e + 1):
            q = base + 2 * (c - s)
            if q + 2 > len(d):
                break
            g = struct.unpack(">H", d[q:q + 2])[0]
            if g and (g + deltas[i]) & 0xFFFF:
                out.add(c)


def _cmap_fmt12(d: bytes, out: set[int]) -> None:
    ngroups = struct.unpack(">I", d[12:16])[0]
    for i in range(min(ngroups, (len(d) - 16) // 12)):
        s, e, g = struct.unpack(">III", d[16 + 12 * i:28 + 12 * i])
        if e < s or e - s > 0x10FFFF:
            continue
        out.update(range(s if g else s + 1, e + 1))


def _parse_cmap(data: bytes) -> set[int]:
    """返回字体实际有字形（glyph != 0）的码位集合。"""
    out: set[int] = set()
    if len(data) < 4:
        return out
    num = struct.unpack(">H", data[2:4])[0]
    subs: list[tuple[int, int, int, int]] = []
    for i in range(num):
        pid, eid, off = struct.unpack(">HHI", data[4 + 8 * i:12 + 8 * i])
        if off + 2 > len(data):
            continue
        fmt = struct.unpack(">H", data[off:off + 2])[0]
        subs.append((pid, eid, off, fmt))
    # 优先完整 Unicode（format 12），其次 BMP（format 4）
    for want_fmt, ok in ((12, lambda p, e: (p, e) in ((3, 10), (0, 4), (0, 6))),
                         (4, lambda p, e: (p, e) in ((3, 1), (0, 3), (0, 1), (0, 2), (0, 0)))):
        for pid, eid, off, fmt in subs:
            if fmt == want_fmt and ok(pid, eid):
                try:
                    (_cmap_fmt12 if fmt == 12 else _cmap_fmt4)(data[off:], out)
                except struct.error:
                    continue
                if out:
                    return out
    return out


def _decode_name(pid: int, raw: bytes) -> str:
    try:
        if pid in (0, 3):
            return raw.decode("utf-16-be", errors="ignore").strip("\x00 ").strip()
        return raw.decode("mac_roman", errors="ignore").strip("\x00 ").strip()
    except Exception:  # noqa: BLE001
        return ""


def _parse_names(data: bytes) -> dict[tuple[int, int], str]:
    """{(nameID, 语言优先级): 字符串}；语言优先级 0 = 中文，1 = 英文，2 = 其它。"""
    names: dict[tuple[int, int], str] = {}
    if len(data) < 6:
        return names
    count, str_off = struct.unpack(">HH", data[2:6])
    for i in range(count):
        rec = data[6 + 12 * i:18 + 12 * i]
        if len(rec) < 12:
            break
        pid, _eid, lang, nid, length, off = struct.unpack(">6H", rec)
        if nid not in (1, 2, 16, 17) or pid not in (0, 1, 3):
            continue
        if pid == 3 and lang in _ZH_LANGS:
            pri = 0
        elif (pid == 3 and lang == 0x0409) or (pid == 1 and lang == 0):
            pri = 1
        else:
            pri = 2
        s = _decode_name(pid, data[str_off + off:str_off + off + length])
        if s and (nid, pri) not in names:
            names[(nid, pri)] = s
    return names


def _display_name(names: dict[tuple[int, int], str], fallback: str) -> str:
    def pick(*nids: int) -> tuple[str, int]:
        for pri in (0, 1, 2):
            for nid in nids:
                s = names.get((nid, pri))
                if s:
                    return s, pri
        return "", 9

    family, fpri = pick(16, 1)
    style, spri = pick(17, 2)
    if not family:
        return fallback
    st = style.strip()
    if st and spri != 0:  # 英文字重名 → 中文
        key = st.lower().replace(" ", "").replace("-", "")
        st = _STYLE_ZH.get(key, st)
    if st and st.lower() not in family.lower():
        return f"{family} {st}"
    return family


# --------------------------------------------------------------------------------------
# ASCII 安全路径
# --------------------------------------------------------------------------------------
def _short_path(path: str) -> Optional[str]:
    if os.name != "nt":
        return None
    try:
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        fn = k32.GetShortPathNameW
        fn.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
        fn.restype = wintypes.DWORD
        buf = ctypes.create_unicode_buffer(1024)
        n = fn(path, buf, len(buf))
        if 0 < n < len(buf):
            return buf.value
    except Exception as e:  # noqa: BLE001
        log.debug("GetShortPathNameW 失败: %s", e)
    return None


def _ascii_cache_roots() -> list[str]:
    roots: list[str] = []
    try:
        from ..runtime import data_dir

        roots.append(os.path.join(data_dir(), "fontcache"))
    except Exception:  # noqa: BLE001
        pass
    for env in ("PROGRAMDATA", "PUBLIC"):
        v = os.environ.get(env)
        if v:
            roots.append(os.path.join(v, "HanhuaManga", "fontcache"))
    roots.append(os.path.join(tempfile.gettempdir(), "HanhuaManga_fontcache"))
    return [r for r in roots if r.isascii()]


def ascii_font_path(path: str) -> Optional[str]:
    """给 FreeType 用的纯 ASCII 路径；拿不到返回 None（调用方退回 bytes 加载）。"""
    if path.isascii():
        return path
    sp = _short_path(path)
    if sp and sp.isascii() and os.path.exists(sp):
        return sp
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = hashlib.sha1(f"{os.path.abspath(path)}|{st.st_size}|{int(st.st_mtime)}".encode("utf-8")).hexdigest()[:16]
    ext = os.path.splitext(path)[1].lower() or ".ttf"
    for root in _ascii_cache_roots():
        dst = os.path.join(root, key + ext)
        try:
            if os.path.exists(dst) and os.path.getsize(dst) == st.st_size:
                return dst
            os.makedirs(root, exist_ok=True)
            fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=root)
            os.close(fd)
            try:
                shutil.copyfile(path, tmp)
                os.replace(tmp, dst)
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)
            log.info("字体路径含非 ASCII 字符，已缓存到 %s", dst)
            return dst
        except OSError as e:
            log.debug("字体缓存目录 %s 不可用: %s", root, e)
    return None


# --------------------------------------------------------------------------------------
# 数据结构
# --------------------------------------------------------------------------------------
@dataclass
class _FontFile:
    """一个字体文件中的一个 face（cmap 在同一文件的多个字重之间共享）。"""

    path: str
    index: int = 0
    load_path: Optional[str] = None
    load_checked: bool = False
    cmap: Optional[frozenset[int]] = None
    is_variable: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass
class _FontEntry:
    id: str
    name: str
    file: _FontFile
    weight: Optional[int]
    tags: tuple[str, ...]
    broken: bool = False


class FontMetrics:
    """参考字号下的度量（单位：em，按字号线性缩放）。排版二分字号时不用为每个字号建字体。"""

    def __init__(self, font: ImageFont.FreeTypeFont) -> None:
        self._font = font
        self._adv: dict[str, float] = {}
        self._ink: dict[str, Optional[tuple[float, float, float, float]]] = {}
        self._lock = threading.Lock()
        # 汉字墨迹的竖直中心（相对基线，向下为正），用于把字放在行 / 格子正中
        ys: list[float] = []
        for ch in "国田回中永":
            b = self.ink(ch)
            if b is not None:
                ys.append((b[1] + b[3]) / 2.0)
        self.cjk_center_y = sum(ys) / len(ys) if ys else -0.38

    def advance(self, ch: str) -> float:
        v = self._adv.get(ch)
        if v is None:
            with self._lock:
                try:
                    v = float(self._font.getlength(ch)) / REF_SIZE
                except Exception:  # noqa: BLE001
                    v = 1.0
                self._adv[ch] = v
        return v

    def ink(self, text: str) -> Optional[tuple[float, float, float, float]]:
        """墨迹框 (x0, y0, x1, y1)，相对原点 / 基线（anchor "ls"），单位 em；无墨迹返回 None。"""
        if text in self._ink:
            return self._ink[text]
        with self._lock:
            r = ink_box(self._font, text)
            if r is not None:
                r = (r[0] / REF_SIZE, r[1] / REF_SIZE, r[2] / REF_SIZE, r[3] / REF_SIZE)
            self._ink[text] = r
        return r


def ink_box(font: ImageFont.FreeTypeFont, text: str) -> Optional[tuple[int, int, int, int]]:
    """真实墨迹框（像素，相对原点 / 基线，anchor "ls"）。

    Pillow 的 getbbox 横向给的是 0..advance，不是墨迹；这里渲染位图后取非零像素的范围。
    """
    try:
        im, (ox, oy) = font.getmask2(text, mode="L", anchor="ls")
        bb = im.getbbox()
    except Exception:  # noqa: BLE001
        return None
    if not bb:
        return None
    return (ox + bb[0], oy + bb[1], ox + bb[2], oy + bb[3])


# --------------------------------------------------------------------------------------
# 注册表
# --------------------------------------------------------------------------------------
class FontRegistry:
    """所有可用字体。线程安全：get / has_glyph / metrics 可在多个渲染线程里同时调用。"""

    def __init__(self, fonts_dir: str, user_dirs: list[str] | None = None) -> None:
        self.fonts_dir = fonts_dir
        self.user_dirs = [d for d in (user_dirs or []) if d]
        self._lock = threading.RLock()
        self._entries: dict[str, _FontEntry] = {}
        self._files: dict[tuple[str, int], _FontFile] = {}
        self._cache: OrderedDict[tuple[str, int], ImageFont.FreeTypeFont] = OrderedDict()
        self._metrics: dict[str, FontMetrics] = {}
        self._bytes_mode_files: set[str] = set()
        self._scan()

    # -- 扫描 ---------------------------------------------------------------------------
    def _file(self, path: str, index: int = 0) -> _FontFile:
        key = (os.path.normcase(os.path.abspath(path)), index)
        ff = self._files.get(key)
        if ff is None:
            ff = _FontFile(path=os.path.abspath(path), index=index)
            self._files[key] = ff
        return ff

    def _scan(self) -> None:
        entries: dict[str, _FontEntry] = {}
        for b in BUILTIN_FONTS:
            p = os.path.join(self.fonts_dir, b.file)
            if os.path.isfile(p):
                ff = self._file(p)
                ff.is_variable = b.weight is not None
                entries[b.id] = _FontEntry(b.id, b.name, ff, b.weight, b.tags)
            else:
                log.warning("缺少内置字体文件：%s", p)
        for d in self.user_dirs:
            for e in self._scan_user_dir(d):
                if e.id not in entries:
                    entries[e.id] = e
        with self._lock:
            self._entries = entries
        if DEFAULT_FONT not in entries:
            log.error("默认字体 %s 不可用，嵌字将退回系统默认字体", DEFAULT_FONT)

    def _scan_user_dir(self, d: str) -> list[_FontEntry]:
        out: list[_FontEntry] = []
        try:
            names = sorted(os.listdir(d))
        except OSError:
            return out
        for fn in names:
            if not fn.lower().endswith(FONT_EXTS) or fn.startswith("."):
                continue
            p = os.path.join(d, fn)
            if not os.path.isfile(p):
                continue
            try:
                out.extend(self._user_entries(p, fn))
            except Exception as e:  # noqa: BLE001
                log.warning("用户字体 %s 无法识别，已跳过：%s", fn, e)
        return out

    def _user_entries(self, path: str, fn: str) -> list[_FontEntry]:
        out: list[_FontEntry] = []
        stem = os.path.splitext(fn)[0]
        with open(path, "rb") as f:
            offsets = _face_offsets(f)
            for idx, off in enumerate(offsets):
                tables = _table_dir(f, off)
                if "cmap" not in tables:
                    continue
                names = _parse_names(_read_table(f, tables, "name", 4 << 20))
                name = _display_name(names, stem if len(offsets) == 1 else f"{stem} #{idx}")
                ff = self._file(path, idx)
                ff.is_variable = "fvar" in tables
                fid = f"user:{fn}" + (f"#{idx}" if len(offsets) > 1 else "")
                out.append(_FontEntry(fid, name, ff, None, ("user",)))
        return out

    def refresh(self) -> None:
        """重新扫描用户字体目录（界面添加字体后调用）。已缓存的字体对象保留。"""
        self._scan()

    def install(self, path: str) -> list[dict]:
        """把一个字体文件复制进第一个用户字体目录并注册，返回新增的字体条目。"""
        if not self.user_dirs:
            raise ValueError("没有可写的用户字体目录")
        if not path.lower().endswith(FONT_EXTS) or not os.path.isfile(path):
            raise ValueError("只支持 ttf / otf / ttc 字体文件")
        d = self.user_dirs[0]
        os.makedirs(d, exist_ok=True)
        dst = os.path.join(d, os.path.basename(path))
        if os.path.normcase(os.path.abspath(dst)) != os.path.normcase(os.path.abspath(path)):
            shutil.copyfile(path, dst)
        fn = os.path.basename(dst)
        try:
            new = self._user_entries(dst, fn)
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"字体文件无法识别：{e}") from e
        with self._lock:
            for e in new:
                self._entries[e.id] = e
        return [self._public(e) for e in new]

    # -- 查询 ---------------------------------------------------------------------------
    @staticmethod
    def _public(e: _FontEntry) -> dict:
        return {"id": e.id, "name": e.name, "path": e.file.path, "tags": list(e.tags)}

    def list(self) -> list[dict]:
        with self._lock:
            ents = [e for e in self._entries.values() if not e.broken]
        order = {b.id: i for i, b in enumerate(BUILTIN_FONTS)}
        ents.sort(key=lambda e: (0, order[e.id], "") if e.id in order else (1, 0, e.name.lower()))
        return [self._public(e) for e in ents]

    def exists(self, font_id: Optional[str]) -> bool:
        with self._lock:
            e = self._entries.get(font_id or "")
        return e is not None and not e.broken

    def name_of(self, font_id: str) -> str:
        with self._lock:
            e = self._entries.get(font_id)
        return e.name if e else font_id

    def resolve(self, font_id: Optional[str], target: str = "zh-Hans") -> str:
        """字体 id → 实际可用的 id：未知 / 损坏 → 默认字体；繁体目标时 noto sc → tc。"""
        fid = font_id or DEFAULT_FONT
        if not self.exists(fid):
            fid = DEFAULT_FONT
        if target == "zh-Hant" and fid.startswith("noto-sans-sc-"):
            tc = "noto-sans-tc-" + fid[len("noto-sans-sc-"):]
            if self.exists(tc):
                fid = tc
        if not self.exists(fid):
            for b in BUILTIN_FONTS:
                if self.exists(b.id):
                    return b.id
        return fid

    def fallback_chain(self, font_id: str, target: str = "zh-Hans") -> list[str]:
        """缺字时逐字回退的顺序：本字体 → 同字重的思源黑体（按简繁）→ 其它内置字体。"""
        weight = "bold"
        for w in ("medium", "black"):
            if font_id.endswith("-" + w):
                weight = w
        first = "tc" if target == "zh-Hant" else "sc"
        second = "sc" if first == "tc" else "tc"
        chain = [font_id, f"noto-sans-{first}-{weight}", f"noto-sans-{second}-{weight}", "lxgw-wenkai"]
        out: list[str] = []
        for c in chain:
            if c not in out and self.exists(c):
                out.append(c)
        return out

    def _entry(self, font_id: str) -> Optional[_FontEntry]:
        with self._lock:
            e = self._entries.get(font_id)
            if e is None or e.broken:
                e = self._entries.get(DEFAULT_FONT)
                if e is not None and e.broken:
                    e = None
            if e is None:
                e = next((x for x in self._entries.values() if not x.broken), None)
            return e

    # -- 加载 ---------------------------------------------------------------------------
    def _load(self, e: _FontEntry, size: int) -> ImageFont.FreeTypeFont:
        ff = e.file
        with ff.lock:
            if not ff.load_checked:
                ff.load_path = ascii_font_path(ff.path)
                ff.load_checked = True
                if ff.load_path is None:
                    log.warning("字体 %s 只能整份载入内存（路径含特殊字符且无可用缓存目录）", ff.path)
        if ff.load_path is not None:
            font = ImageFont.truetype(ff.load_path, size=size, index=ff.index,
                                      layout_engine=ImageFont.Layout.BASIC)
        else:
            with open(ff.path, "rb") as fh:
                font = ImageFont.truetype(fh, size=size, index=ff.index, layout_engine=ImageFont.Layout.BASIC)
            self._bytes_mode_files.add(ff.path)
        if ff.is_variable:
            self._apply_weight(font, e.weight)
        return font

    @staticmethod
    def _apply_weight(font: ImageFont.FreeTypeFont, weight: Optional[int]) -> None:
        try:
            axes = font.get_variation_axes()
        except Exception:  # noqa: BLE001
            return
        vals: list[float] = []
        for ax in axes:
            nm = ax.get("name", b"")
            nm = nm.decode("latin-1", "ignore") if isinstance(nm, bytes) else str(nm)
            lo, hi, df = ax.get("minimum", 0), ax.get("maximum", 0), ax.get("default", 0)
            if nm.lower() in ("weight", "wght"):
                w = weight if weight is not None else max(df, 400)
                vals.append(float(min(max(w, lo), hi)))
            else:
                vals.append(float(df))
        if vals:
            try:
                font.set_variation_by_axes(vals)
            except Exception as ex:  # noqa: BLE001
                log.debug("设置可变字体轴失败: %s", ex)

    def get(self, font_id: str, size: int) -> ImageFont.FreeTypeFont:
        """(字体, 字号) → FreeTypeFont（LRU 缓存）。未知 id 退回默认字体；全部不可用时退回 Pillow 内置字体。"""
        size = max(1, min(int(round(size)), 2048))
        e = self._entry(font_id)
        if e is None:
            return ImageFont.load_default(size)  # type: ignore[return-value]
        key = (e.id, size)
        with self._lock:
            f = self._cache.get(key)
            if f is not None:
                self._cache.move_to_end(key)
                return f
        try:
            f = self._load(e, size)
        except Exception as ex:  # noqa: BLE001
            log.warning("字体 %s 加载失败，已停用：%s", e.name, ex)
            with self._lock:
                e.broken = True
            if e.id == DEFAULT_FONT or self._entry(DEFAULT_FONT) is None:
                return ImageFont.load_default(size)  # type: ignore[return-value]
            return self.get(DEFAULT_FONT, size)
        with self._lock:
            self._cache[key] = f
            limit = _MAX_CACHED_BYTES_FONTS if self._bytes_mode_files else _MAX_CACHED_FONTS
            while len(self._cache) > limit:
                self._cache.popitem(last=False)
        return f

    def metrics(self, font_id: str) -> FontMetrics:
        e = self._entry(font_id)
        key = e.id if e is not None else "__default__"
        m = self._metrics.get(key)
        if m is None:
            font = self.get(key, REF_SIZE) if e is not None else ImageFont.load_default(REF_SIZE)
            m = FontMetrics(font)  # type: ignore[arg-type]
            with self._lock:
                m = self._metrics.setdefault(key, m)
        return m

    # -- 缺字 ---------------------------------------------------------------------------
    def _cmap(self, e: _FontEntry) -> frozenset[int]:
        ff = e.file
        if ff.cmap is None:
            with ff.lock:
                if ff.cmap is None:
                    try:
                        with open(ff.path, "rb") as f:
                            offs = _face_offsets(f)
                            tables = _table_dir(f, offs[min(ff.index, len(offs) - 1)])
                            ff.cmap = frozenset(_parse_cmap(_read_table(f, tables, "cmap")))
                    except Exception as ex:  # noqa: BLE001
                        log.warning("读取字体 %s 的字符表失败：%s", ff.path, ex)
                        ff.cmap = frozenset()
        return ff.cmap

    def has_glyph(self, font_id: str, ch: str) -> bool:
        if not ch:
            return True
        e = self._entry(font_id)
        if e is None:
            return False
        cmap = self._cmap(e)
        if not cmap:  # 字符表读不出来时不做回退（交给 FreeType）
            return True
        return all(ord(c) in cmap for c in ch)
