"""bookio 公共底层：格式识别、自然排序、压缩包文件名乱码修复、垃圾条目过滤、
Book 基类，以及「文件夹」「散图」两种书。

依赖方向：formats ← archive / pdf / epub ← export ← ``hanhua.bookio``（对外入口 open_book / scan_inputs）。
这里不 import 其它 bookio 子模块，避免循环导入。
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import stat
import struct
import tempfile
import threading
import unicodedata
import zipfile
import zlib
from typing import Iterator

import numpy as np

from .. import imageio as hio
from ..types import UnsupportedFormat

log = logging.getLogger("hanhua.bookio")

# ---- 扩展名与格式 ---------------------------------------------------------------------------------
IMAGE_EXTS: frozenset[str] = frozenset(hio.IMAGE_EXTS)

# 扩展名 → (容器家族, 是否漫画专用扩展名 cbz/cbr/cb7/cbt)
_EXT_FAMILY: dict[str, tuple[str, bool]] = {
    ".zip": ("zip", False), ".cbz": ("zip", True),
    ".7z": ("7z", False), ".cb7": ("7z", True),
    ".rar": ("rar", False), ".cbr": ("rar", True),
    ".tar": ("tar", False), ".cbt": ("tar", True),
    ".tar.gz": ("tar", False), ".tgz": ("tar", False),
    ".tar.bz2": ("tar", False), ".tbz2": ("tar", False), ".tbz": ("tar", False),
    ".tar.xz": ("tar", False), ".txz": ("tar", False), ".tar.zst": ("tar", False),
    ".pdf": ("pdf", False), ".epub": ("epub", False),
}
_DOUBLE_EXTS = (".tar.gz", ".tar.bz2", ".tar.xz", ".tar.zst")
_KIND_OF: dict[tuple[str, bool], str] = {
    ("zip", False): "zip", ("zip", True): "cbz",
    ("7z", False): "7z", ("7z", True): "cb7",
    ("rar", False): "rar", ("rar", True): "cbr",
    ("tar", False): "tar", ("tar", True): "cbt",
}
_COMPRESSED_MAGIC = (b"\x1f\x8b", b"BZh", b"\xfd7zXZ\x00", b"\x28\xb5\x2f\xfd")  # gzip / bzip2 / xz / zstd

BOOK_FILE_EXTS: frozenset[str] = frozenset(_EXT_FAMILY)
INPUT_EXTS: tuple[str, ...] = tuple(sorted(BOOK_FILE_EXTS | IMAGE_EXTS))   # 给文件对话框用
ZIP_KINDS = frozenset({"zip", "cbz"})
EXTRACT_KINDS = frozenset({"7z", "cb7", "rar", "cbr", "tar", "cbt"})
ARCHIVE_KINDS = ZIP_KINDS | EXTRACT_KINDS
BOOK_KINDS = ARCHIVE_KINDS | {"pdf", "epub", "folder", "images"}

META_EXTS = frozenset({".xml", ".txt", ".nfo", ".json", ".opf"})   # 文件夹书要原样带走的元数据
MAX_META_BYTES = 8 << 20
LIST_SUFFIX = ".hanhua-images.json"        # 散图清单（scan_inputs 生成，open_book 识别）
ORIGIN_MARKER = ".hanhua_origin.json"      # 套娃展开目录里记录最外层压缩包的位置

SUPPORTED_TEXT = "文件夹、图片、zip/cbz、7z/cb7、rar/cbr、tar/cbt、pdf、epub"


def book_ext(name: str) -> str:
    """小写扩展名；.tar.gz 这类双扩展名整体返回。"""
    low = name.lower()
    for d in _DOUBLE_EXTS:
        if low.endswith(d):
            return d
    return os.path.splitext(low)[1]


def is_image_name(name: str) -> bool:
    return os.path.splitext(name)[1].lower() in IMAGE_EXTS


def is_book_name(name: str) -> bool:
    """按扩展名判断是不是一本书（压缩包 / pdf / epub）。"""
    return book_ext(name) in BOOK_FILE_EXTS


def title_from_name(name: str) -> str:
    base = os.path.basename(name.replace("\\", "/").rstrip("/"))
    ext = book_ext(base)
    if ext and (ext in BOOK_FILE_EXTS or ext in IMAGE_EXTS):
        base = base[: len(base) - len(ext)]
    return base.strip() or "漫画"


def sniff_family(path: str) -> str | None:
    """按文件头魔数识别容器：zip | rar | 7z | pdf | tar | compressed（gzip 等）| None。"""
    try:
        with open(path, "rb") as f:
            head = f.read(1024)
    except OSError:
        return None
    if head[:4] in (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"):
        return "zip"
    if head.startswith(b"Rar!\x1a\x07"):
        return "rar"
    if head.startswith(b"7z\xbc\xaf\x27\x1c"):
        return "7z"
    if b"%PDF-" in head:
        return "pdf"
    if len(head) >= 262 and head[257:262] == b"ustar":
        return "tar"
    if head.startswith(_COMPRESSED_MAGIC):
        return "compressed"
    return None


def resolve_kind(path: str) -> str | None:
    """书的种类（zip cbz 7z cb7 rar cbr tar cbt pdf epub），以魔数为准、扩展名兜底。

    .cbr 实际是 zip、.cbz 实际是 rar 这种改错扩展名的情况很常见：按真实格式读，
    但保留「漫画专用扩展名」这一属性（输出 same 时仍是 cbz）。
    """
    ext = book_ext(path)
    fam_ext, comic = _EXT_FAMILY.get(ext, (None, False))
    fam = sniff_family(path)
    if fam == "pdf":
        return "pdf"
    if fam == "zip":
        return "epub" if fam_ext == "epub" else _KIND_OF[("zip", comic)]
    if fam in ("rar", "7z", "tar"):
        return _KIND_OF[(fam, comic)]
    if fam == "compressed" and fam_ext == "tar":
        return _KIND_OF[("tar", comic)]
    # 魔数认不出（自解压头、文件头损坏、老式 tar）：按扩展名交给具体格式，由它给出中文错误
    if fam_ext in ("zip", "7z", "rar", "tar"):
        return _KIND_OF[(fam_ext, comic)]
    if fam_ext in ("pdf", "epub"):
        return fam_ext
    return None


# ---- 自然排序 / 垃圾条目 / 文件名清洗 ------------------------------------------------------------
_DIGITS = re.compile(r"(\d+)")


def natural_key(name: str) -> tuple:
    """自然排序 key：数字按数值比较（2 < 10）、不分大小写、全角数字归一，逐级目录比较。

    re.split 带捕获组时结果严格是 [文本, 数字, 文本, ...]，同一位置的类型总是一致，不会出现 str 与 int 比较。
    """
    parts: list[tuple] = []
    comps = name.replace("\\", "/").split("/")
    for k, comp in enumerate(comps):
        chunks = _DIGITS.split(unicodedata.normalize("NFKC", comp).casefold())
        # 同一层里文件排在子目录前面（封面等根目录页面先读，再进各话子目录），zip / 7z / 文件夹排序一致
        is_dir = 1 if k < len(comps) - 1 else 0
        parts.append((is_dir,) + tuple((int(c), len(c)) if i % 2 else c for i, c in enumerate(chunks)))
    return (tuple(parts), name)


_JUNK_FILES = frozenset({"thumbs.db", "desktop.ini", "ehthumbs.db", "ehthumbs_vista.db", ".ds_store"})
_JUNK_DIRS = frozenset({"__macosx", "@eadir", "$recycle.bin", "system volume information", ".trashes"})


def is_junk_path(name: str) -> bool:
    """__MACOSX、Thumbs.db、.DS_Store、隐藏文件（含 ._xxx AppleDouble）、群晖 @eaDir 等。"""
    comps = [c for c in name.replace("\\", "/").split("/") if c not in ("", ".")]
    if not comps:
        return True
    for c in comps:
        low = c.lower()
        if low.startswith((".", "~$")) or low in _JUNK_DIRS:
            return True
    last = comps[-1].lower()
    return last in _JUNK_FILES or last.endswith(":zone.identifier")


_DRIVE = re.compile(r"^[A-Za-z]:$")


def clean_entry_name(name: str) -> str:
    """统一成 posix 相对路径：反斜杠转正斜杠，去掉 . / .. / 盘符 / 开头的 /（防路径穿越）。"""
    out: list[str] = []
    for c in name.replace("\x00", "").replace("\\", "/").split("/"):
        if c in ("", ".", ".."):
            continue
        out.append(c)
    if out and _DRIVE.match(out[0]):
        out.pop(0)
    return "/".join(out)


_BAD_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED = frozenset({"con", "prn", "aux", "nul"} | {f"com{i}" for i in range(1, 10)} | {f"lpt{i}" for i in range(1, 10)})


def safe_component(name: str, fallback: str = "_", max_len: int = 150) -> str:
    """一级路径名在 Windows 上可用：替换非法字符、去掉结尾的点和空格、避开 CON/NUL 等保留名、限长。"""
    s = _BAD_CHARS.sub("_", name).strip().rstrip(". ")
    if not s:
        s = fallback
    if s.split(".", 1)[0].lower() in _RESERVED:
        s = "_" + s
    if len(s) > max_len:
        stem, ext = os.path.splitext(s)
        s = (stem[: max(1, max_len - len(ext))] + ext) if len(ext) <= 16 else s[:max_len]
    return s


def safe_relpath(name: str) -> str:
    return "/".join(safe_component(c) for c in clean_entry_name(name).split("/") if c) or "_"


def sanitize_filename(name: str, fallback: str = "漫画") -> str:
    return safe_component(name, fallback, 120)


# ---- zip 文件名乱码修复 ----------------------------------------------------------------------------
# 没有 UTF-8 标志的条目被 zipfile 按 cp437 解码；编码回 cp437 拿到原始字节，再在下面几种编码里挑最像人话的。
LEGACY_ENCODINGS: tuple[str, ...] = ("cp932", "gbk", "cp950", "cp949")   # 日 / 简中 / 繁中 / 韩（平局按此顺序）


def _char_score(ch: str, enc: str) -> float:
    """单个非 ASCII 字符在某编码下“像正常文字”的程度：+1 常用字，0 次常用，负数 = 典型乱码特征。"""
    o = ord(ch)
    if 0xFF61 <= o <= 0xFF9F or 0xE000 <= o <= 0xF8FF:   # 半角片假名（GBK 字节被当 SJIS 的典型乱码）/ 私用区
        return -1.0
    if unicodedata.category(ch) in ("Cc", "Cf", "Cn", "Co", "Cs"):
        return -1.0
    try:
        b = ch.encode(enc)
    except UnicodeEncodeError:
        return -1.0
    if len(b) != 2:
        return -0.5
    b1, b2 = b[0], b[1]
    if enc == "cp932":
        if b1 in (0x82, 0x83) or 0x88 <= b1 <= 0x98:     # 假名 / 全角英数 / JIS 第一水准汉字
            return 1.0
        if b1 == 0x81:                                   # 全角标点、长音符
            return 0.3
        if 0x99 <= b1 <= 0x9F or 0xE0 <= b1 <= 0xEA:     # 第二水准汉字
            return 0.0
        return -1.0                                      # 机种依存字符 / 外字区
    if enc == "gbk":
        if b2 < 0xA1:                                    # GB2312 以外的 GBK 扩展区（生僻字）
            return -1.0
        if 0xB0 <= b1 <= 0xD7:                           # 一级汉字
            return 1.0
        if 0xA1 <= b1 <= 0xA9:                           # 全角符号 / 假名
            return 0.3
        if 0xD8 <= b1 <= 0xF7:                           # 二级汉字
            return 0.0
        return -1.0
    if enc == "cp950":
        if 0xA4 <= b1 <= 0xC6:                           # 常用字
            return 1.0
        if 0xA1 <= b1 <= 0xA3:                           # 符号
            return 0.3
        if 0xC9 <= b1 <= 0xF9:                           # 次常用字
            return 0.0
        return -1.0
    if enc == "cp949":
        if 0xB0 <= b1 <= 0xC8 and b2 >= 0xA1:            # KS X 1001 常用谚文
            return 1.0
        if 0xA1 <= b1 <= 0xAC and b2 >= 0xA1:            # 符号
            return 0.3
        if 0xCA <= b1 <= 0xFD and b2 >= 0xA1:            # 汉字
            return 0.0
        return -0.3                                      # cp949 扩展谚文
    return 0.0


def guess_legacy_encoding(raws: list[bytes]) -> str | None:
    """一个压缩包里的文件名通常同一种编码：把所有非 ASCII 名字合起来打分，选平均分最高的编码。"""
    best: str | None = None
    best_score = float("-inf")
    for enc in LEGACY_ENCODINGS:
        total, count = 0.0, 0
        try:
            for raw in raws:
                for ch in raw.decode(enc):
                    if ord(ch) >= 0x80:
                        total += _char_score(ch, enc)
                        count += 1
        except UnicodeDecodeError:
            continue
        score = total / count if count else 0.0
        if score > best_score + 1e-9:
            best, best_score = enc, score
    return best


def _raw_name(info: zipfile.ZipInfo) -> bytes | None:
    try:
        return info.orig_filename.encode("cp437")    # orig_filename 是“反斜杠转斜杠”之前的原值
    except UnicodeEncodeError:
        return None


def _is_utf8(raw: bytes) -> bool:
    try:
        raw.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def _unicode_path_extra(info: zipfile.ZipInfo, raw: bytes) -> str | None:
    """Info-ZIP Unicode Path 扩展字段（0x7075）：7-Zip / WinRAR 会把 UTF-8 名字放这里（CRC 对得上才用）。"""
    extra = info.extra or b""
    i = 0
    while i + 4 <= len(extra):
        hid, size = struct.unpack_from("<HH", extra, i)
        body = extra[i + 4: i + 4 + size]
        if hid == 0x7075 and len(body) >= 5 and body[0] == 1:
            if struct.unpack_from("<I", body, 1)[0] == (zlib.crc32(raw) & 0xFFFFFFFF):
                try:
                    return body[5:].decode("utf-8")
                except UnicodeDecodeError:
                    return None
        i += 4 + size
    return None


def decode_zip_name(info: zipfile.ZipInfo, enc: str | None) -> str:
    """修复后的条目名（尚未清洗路径）。"""
    if info.flag_bits & 0x800:
        return info.orig_filename
    raw = _raw_name(info)
    if raw is None:
        return info.orig_filename
    if raw.isascii():
        return raw.decode("ascii")
    uni = _unicode_path_extra(info, raw)
    if uni:
        return uni
    if _is_utf8(raw):
        return raw.decode("utf-8")
    for e in ((enc,) if enc else ()) + LEGACY_ENCODINGS:
        try:
            return raw.decode(e)
        except UnicodeDecodeError:
            continue
    return info.orig_filename


def zip_entries(zf: zipfile.ZipFile) -> list[tuple[str, zipfile.ZipInfo]]:
    """[(修复并清洗后的条目名, ZipInfo)]，保持压缩包原顺序；跳过目录、垃圾条目、重复条目。"""
    infos = zf.infolist()
    legacy: list[bytes] = []
    for info in infos:
        if info.flag_bits & 0x800:
            continue
        raw = _raw_name(info)
        if raw and not raw.isascii() and not _is_utf8(raw) and _unicode_path_extra(info, raw) is None:
            legacy.append(raw)
    enc = guess_legacy_encoding(legacy) if legacy else None
    if enc:
        log.debug("压缩包文件名编码判定为 %s（%d 个非 UTF-8 名字）", enc, len(legacy))
    out: list[tuple[str, zipfile.ZipInfo]] = []
    seen: set[str] = set()
    for info in infos:
        decoded = decode_zip_name(info, enc)
        if decoded.endswith(("/", "\\")) or (info.external_attr & 0x10 and info.file_size == 0):
            continue                                    # 目录条目
        name = clean_entry_name(decoded)
        if not name or is_junk_path(name):
            continue
        if name in seen:
            log.debug("压缩包里有重复条目，只取第一个：%s", name)
            continue
        seen.add(name)
        out.append((name, info))
    return out


# ---- 文件系统小工具 -------------------------------------------------------------------------------
def file_signature(path: str) -> str:
    """路径 + 大小 + 修改时间：源文件变了缓存就失效。"""
    p = os.path.normcase(os.path.abspath(path))
    try:
        st = os.stat(path)
        return f"{p}|{st.st_size}|{st.st_mtime_ns}"
    except OSError:
        return p


def short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "surrogatepass")).hexdigest()[:12]


def read_file(path: str) -> bytes:
    try:
        with open(path, "rb") as f:
            return f.read()
    except OSError as e:
        raise UnsupportedFormat(f"读取文件失败：{path}（{e.strerror or e}）") from e


def read_small_text(path: str) -> str | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except (OSError, UnicodeDecodeError):
        return None


def decode_text(data: bytes) -> str:
    """XML / XHTML 字节 → str：BOM > XML 声明里的 encoding > UTF-8（坏字节替换，不抛异常）。"""
    if data.startswith(b"\xef\xbb\xbf"):
        return data[3:].decode("utf-8", "replace")
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16", "replace")
    m = re.match(rb"""\s*<\?xml[^>]*encoding\s*=\s*["']([A-Za-z0-9_.-]+)["']""", data[:200])
    if m:
        try:
            return data.decode(m.group(1).decode("ascii"), "replace")
        except LookupError:
            pass
    return data.decode("utf-8", "replace")


def _is_hidden(entry: os.DirEntry) -> bool:
    if entry.name.startswith(".") or is_junk_path(entry.name):
        return True
    try:
        attrs = getattr(entry.stat(follow_symlinks=False), "st_file_attributes", 0)
    except OSError:
        return True
    return bool(attrs & (stat.FILE_ATTRIBUTE_HIDDEN | stat.FILE_ATTRIBUTE_SYSTEM))


def list_dir(d: str) -> tuple[list[str], list[str]]:
    """(文件, 子文件夹) 完整路径，自然排序；跳过隐藏 / 系统 / 垃圾条目和符号链接 / junction（防死循环）。"""
    files: list[str] = []
    dirs: list[str] = []
    try:
        with os.scandir(d) as it:
            for e in it:
                try:
                    if e.is_symlink() or (hasattr(e, "is_junction") and e.is_junction()):
                        continue
                    if _is_hidden(e):
                        continue
                    if e.is_dir(follow_symlinks=False):
                        dirs.append(e.path)
                    elif e.is_file(follow_symlinks=False):
                        files.append(e.path)
                except OSError:
                    continue
    except OSError as err:
        log.warning("无法读取文件夹「%s」：%s", d, err)
    files.sort(key=lambda p: natural_key(os.path.basename(p)))
    dirs.sort(key=lambda p: natural_key(os.path.basename(p)))
    return files, dirs


def iter_dir_files(root: str, recursive: bool = True, max_depth: int = 16) -> Iterator[tuple[str, str]]:
    """遍历文件：(posix 相对路径, 完整路径)。"""
    stack: list[tuple[str, str, int]] = [(root, "", 0)]
    while stack:
        d, prefix, depth = stack.pop()
        files, dirs = list_dir(d)
        for f in files:
            yield prefix + os.path.basename(f), f
        if recursive and depth < max_depth:
            for sub in reversed(dirs):
                stack.append((sub, prefix + os.path.basename(sub) + "/", depth + 1))


def find_origin(path: str) -> tuple[str, str] | None:
    """path 在套娃展开目录里时返回 (最外层压缩包路径, 它的标题)，用于决定默认输出位置和书名。"""
    d = os.path.dirname(os.path.abspath(path))
    for _ in range(8):
        text = read_small_text(os.path.join(d, ORIGIN_MARKER))
        if text:
            try:
                meta = json.loads(text)
                origin = str(meta.get("origin") or "")
                if origin:
                    return origin, str(meta.get("title") or title_from_name(origin))
            except (ValueError, AttributeError):
                pass
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return None


# ---- 图片头信息（不解码整张图） -------------------------------------------------------------------
_SOF_OK = frozenset({0xC0, 0xC1, 0xC2})                                     # 基线 / 扩展 / 渐进：PDF 和阅读器都支持
_SOF_ALL = _SOF_OK | {0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}


def _jpeg_info(data: bytes) -> tuple[str, int, int, int]:
    i, n = 2, len(data)
    while i + 4 <= n:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker == 0xFF:
            i += 1
            continue
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        if marker in (0xD9, 0xDA):
            break
        seg = struct.unpack(">H", data[i + 2: i + 4])[0]
        if marker in _SOF_ALL and i + 10 <= n:
            precision = data[i + 4]
            h, w = struct.unpack(">HH", data[i + 5: i + 9])
            comps = data[i + 9]
            # 12 位精度 / 无损 / 算术编码的 JPEG 很多阅读器不认：通道数记 0，调用方会重新编码
            return "jpg", w, h, comps if (marker in _SOF_OK and precision == 8) else 0
        i += 2 + seg
    return "jpg", 0, 0, 0


def image_info(data: bytes) -> tuple[str, int, int, int] | None:
    """(格式 jpg|png|webp|gif|bmp|tif|avif, 宽, 高, 通道数)；宽高读不到时为 0；认不出返回 None。"""
    if data[:3] == b"\xff\xd8\xff":
        return _jpeg_info(data)
    if data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) >= 26:
        w, h = struct.unpack(">II", data[16:24])
        return "png", w, h, {0: 1, 2: 3, 3: 3, 4: 2, 6: 4}.get(data[25], 3)
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        chunk = data[12:16]
        w = h = 0
        if chunk == b"VP8X" and len(data) >= 30:
            w = 1 + int.from_bytes(data[24:27], "little")
            h = 1 + int.from_bytes(data[27:30], "little")
        elif chunk == b"VP8 " and len(data) >= 30:
            w, h = (v & 0x3FFF for v in struct.unpack("<HH", data[26:30]))
        elif chunk == b"VP8L" and len(data) >= 25:
            bits = int.from_bytes(data[21:25], "little")
            w, h = (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
        return "webp", w, h, 3
    if data[:6] in (b"GIF87a", b"GIF89a") and len(data) >= 10:
        w, h = struct.unpack("<HH", data[6:10])
        return "gif", w, h, 3
    if data[:2] == b"BM" and len(data) >= 26:
        w, h = struct.unpack("<ii", data[18:26])
        return "bmp", abs(w), abs(h), 3
    if data[:4] in (b"II*\x00", b"MM\x00*"):
        return "tif", 0, 0, 0
    if data[4:12] in (b"ftypavif", b"ftypavis"):
        return "avif", 0, 0, 0
    return None


# ---- Book 基类 -------------------------------------------------------------------------------------
class Book:
    """一本书。pages = 页面名（相对路径，阅读顺序）；extras = 需要原样带走的非图片条目。

    读取接口线程安全（各子类自己加锁）；出错一律抛 UnsupportedFormat（中文说明）。
    """

    def __init__(self, path: str, kind: str, title: str, origin_dir: str) -> None:
        self.path = path
        self.kind = kind
        self.title = title
        # 默认输出目录：源文件所在目录；套娃卷 = 最外层压缩包所在目录；散图 = 图片所在目录
        self.origin_dir = origin_dir
        self.pages: list[str] = []
        self.extras: list[str] = []
        self._page_set: frozenset[str] = frozenset()
        self._extra_set: frozenset[str] = frozenset()

    def _set_entries(self, pages: list[str], extras: list[str]) -> None:
        self.pages = list(pages)
        self.extras = list(extras)
        self._page_set = frozenset(self.pages)
        self._extra_set = frozenset(self.extras)

    def _need_page(self, name: str) -> None:
        if name not in self._page_set:
            raise UnsupportedFormat(f"书里没有这一页：{name}")

    def _need_extra(self, name: str) -> None:
        if name not in self._extra_set:
            raise UnsupportedFormat(f"书里没有这个附加文件：{name}")

    def read_page(self, name: str) -> np.ndarray:
        """BGR uint8 (H, W, 3)。"""
        data = self.read_page_bytes(name)
        try:
            return hio.decode(data)
        except ValueError as e:
            raise UnsupportedFormat(f"页面图片已损坏，无法解码：{name}") from e

    def read_page_bytes(self, name: str) -> bytes:
        raise NotImplementedError

    def read_extra(self, name: str) -> bytes:
        raise NotImplementedError

    def close(self) -> None:
        """释放文件句柄（可重复调用）。"""

    def __enter__(self) -> "Book":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"<{type(self).__name__} kind={self.kind} pages={len(self.pages)} path={self.path!r}>"


class FolderBook(Book):
    """文件夹：直接含图片 → 只取这一层（子文件夹由 scan_inputs 另算一本）；否则递归取所有子文件夹里的图。"""

    def __init__(self, path: str) -> None:
        root = os.path.abspath(path)
        super().__init__(root, "folder", os.path.basename(root.rstrip("\\/")) or "漫画", os.path.dirname(root) or root)
        self._root = root
        rels = [rel for rel, _ in iter_dir_files(root, recursive=False)]
        if not any(is_image_name(r) for r in rels):
            rels = [rel for rel, _ in iter_dir_files(root, recursive=True)]
        pages = sorted((r for r in rels if is_image_name(r)), key=natural_key)
        if not pages:
            raise UnsupportedFormat(f"文件夹里没有找到图片：{root}")
        extras = sorted((r for r in rels if not is_image_name(r) and self._is_meta(r)), key=natural_key)
        self._set_entries(pages, extras)

    def _full(self, name: str) -> str:
        return os.path.join(self._root, *name.split("/"))

    def _is_meta(self, rel: str) -> bool:
        if os.path.splitext(rel)[1].lower() not in META_EXTS:
            return False
        try:
            return os.path.getsize(self._full(rel)) <= MAX_META_BYTES
        except OSError:
            return False

    def read_page_bytes(self, name: str) -> bytes:
        self._need_page(name)
        return read_file(self._full(name))

    def read_extra(self, name: str) -> bytes:
        self._need_extra(name)
        return read_file(self._full(name))


class ImageListBook(Book):
    """散图：用户拖进来的同目录若干张图（或单张图）。页面名 = 文件名。"""

    def __init__(self, files: list[str], title: str, origin_dir: str, path: str) -> None:
        super().__init__(path, "images", title, origin_dir)
        self._files: dict[str, str] = {}
        for f in files:
            name = os.path.basename(f)
            if name in self._files:                     # 不同目录的同名图（手工清单才会出现）
                name = f"{os.path.basename(os.path.dirname(f))}/{name}"
            self._files.setdefault(name, f)
        self._set_entries(sorted(self._files, key=natural_key), [])

    @classmethod
    def single(cls, image_path: str) -> "ImageListBook":
        p = os.path.abspath(image_path)
        return cls([p], title_from_name(p), os.path.dirname(p), p)

    @classmethod
    def from_list_file(cls, path: str) -> "ImageListBook":
        try:
            meta = json.loads(read_file(path).decode("utf-8"))
            images = [p for p in meta.get("images", []) if isinstance(p, str)]
        except (ValueError, UnicodeDecodeError, AttributeError) as e:
            raise UnsupportedFormat(f"散图清单文件已损坏：{path}") from e
        alive = [p for p in images if os.path.isfile(p)]
        if len(alive) < len(images):
            log.warning("散图清单里有 %d 张图片已经不存在：%s", len(images) - len(alive), path)
        if not alive:
            raise UnsupportedFormat("这些图片已经被移动或删除了，请重新拖进来。")
        base = str(meta.get("dir") or os.path.dirname(alive[0]))
        title = str(meta.get("title") or os.path.basename(base.rstrip("\\/")) or "散图")
        return cls(alive, title, base, os.path.abspath(path))

    def read_page_bytes(self, name: str) -> bytes:
        self._need_page(name)
        return read_file(self._files[name])

    def read_extra(self, name: str) -> bytes:
        self._need_extra(name)
        return b""


def make_image_list(files: list[str], tmp_root: str) -> str:
    """同目录散图 → 一本书：在 tmp_root 写清单文件并返回其路径（写不了就退回整个文件夹）。

    清单内容确定、修改时间设为图片里最新的那张 → 同一批图再拖进来，Engine 算出的 job_id 不变（可续跑）。
    """
    uniq: dict[str, str] = {}
    for f in files:
        p = os.path.abspath(f)
        uniq.setdefault(os.path.normcase(p), p)
    images = sorted(uniq.values(), key=lambda p: natural_key(os.path.basename(p)))
    base = os.path.dirname(images[0])
    key = short_hash("\n".join(os.path.normcase(p) for p in images))
    root = os.path.abspath(tmp_root) if tmp_root else os.path.join(tempfile.gettempdir(), "hanhua")
    path = os.path.join(root, f"loose_{key}{LIST_SUFFIX}")
    payload = json.dumps(
        {"version": 1, "dir": base, "title": os.path.basename(base.rstrip("\\/")) or "散图", "images": images},
        ensure_ascii=False, indent=1,
    ).encode("utf-8")
    try:
        try:
            with open(path, "rb") as f:
                same = f.read() == payload
        except OSError:
            same = False
        if not same:
            hio.write_bytes_atomic(path, payload)
        newest = max(os.path.getmtime(p) for p in images)
        os.utime(path, (newest, newest))
    except OSError as e:
        log.warning("散图清单写入失败（%s），改为整个文件夹算一本：%s", e, base)
        return base
    return path


# ---- 其它子模块共用的小工具 ------------------------------------------------------------------------
def is_meta_name(name: str) -> bool:
    """需要原样带走的元数据条目（ComicInfo.xml、说明 txt 等），只看扩展名。"""
    return os.path.splitext(name)[1].lower() in META_EXTS


def tmp_dir_or_default(tmp_dir: str | None) -> str:
    """调用方没给临时目录时退回系统临时目录下的 hanhua/。"""
    if isinstance(tmp_dir, (str, os.PathLike)) and str(tmp_dir).strip():
        return os.path.abspath(os.fspath(tmp_dir))
    return os.path.join(tempfile.gettempdir(), "hanhua")


class ZipUnreadable(UnsupportedFormat):
    """zipfile 打不开（中央目录损坏、其实不是 zip …）：调用方可以退回「用 tar.exe 解压」再试一次。"""


class ZipReader:
    """线程安全地按条目读 zip（zip / cbz / epub 共用）：句柄懒打开、close 后再读会自动重开，异常统一成中文。"""

    def __init__(self, path: str) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._zf: zipfile.ZipFile | None = None

    def _handle(self) -> zipfile.ZipFile:
        if self._zf is None:
            name = os.path.basename(self.path)
            try:
                self._zf = zipfile.ZipFile(self.path)
            except FileNotFoundError as e:
                raise UnsupportedFormat(f"找不到文件：{self.path}") from e
            except PermissionError as e:
                raise UnsupportedFormat(f"没有权限读取「{name}」（文件可能正被其它程序占用）。") from e
            except Exception as e:  # noqa: BLE001 - BadZipFile / 截断 / 偏移错乱等都交给调用方决定是否退回解压
                raise ZipUnreadable(f"「{name}」不是有效的 zip 文件，或者已经损坏。") from e
        return self._zf

    def entries(self) -> list[tuple[str, zipfile.ZipInfo]]:
        with self._lock:
            return zip_entries(self._handle())

    def read(self, info: zipfile.ZipInfo, label: str) -> bytes:
        name = os.path.basename(self.path)
        with self._lock:
            try:
                return self._handle().read(info)
            except UnsupportedFormat:
                raise
            except RuntimeError as e:          # 加密条目：zipfile 要求密码
                raise UnsupportedFormat(
                    f"压缩包「{name}」有密码，暂不支持加密压缩包。请先用解压软件输入密码解压，再把解压出来的文件夹拖进来。"
                ) from e
            except NotImplementedError as e:   # Deflate64 / PPMd 等 zipfile 不支持的压缩方式
                raise UnsupportedFormat(f"压缩包「{name}」用了暂不支持的压缩方式，无法读取「{label}」。") from e
            except Exception as e:  # noqa: BLE001 - CRC 错 / 截断 / zlib 错误 …
                raise UnsupportedFormat(f"压缩包「{name}」里的「{label}」已损坏，无法读取。") from e

    def close(self) -> None:
        with self._lock:
            if self._zf is not None:
                try:
                    self._zf.close()
                except Exception:  # noqa: BLE001 - 关闭失败不影响后续
                    pass
                self._zf = None
