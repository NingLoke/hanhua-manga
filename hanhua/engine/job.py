"""任务（一本书）：数据结构、工作目录布局、原子落盘、去字补丁编解码、文字策略、进度统计。

工作目录 ``<work_root>/<job_id>/``（合同 5.7）::

    job.json          任务元数据 + 设置快照（原子写）
    pages/0001.json   PageData + 引擎附加信息 "_engine"（每完成一个阶段原子写一次；缺文件 = 还没开始）
    clean/0001.png    去字补丁：BGRA，alpha=255 处是去字后的像素，其余全 0（读取时合成回原图）
    out/0001.<ext>    成品页
    thumbs/0001.jpg   360px 缩略图
    src_cache/        原图缓存（压缩包 / pdf 里的页给界面看时才解出来）
    view/             去字图预览（原图 + 补丁合成，给界面看）
    tmp/              bookio 的解压缓存（rar / 7z / pdf）

文件序号从 1 开始（0001 = 第 0 页）。所有写盘都是「先写临时文件再替换」，强杀进程也不会留下半个文件。
"""
from __future__ import annotations

import collections
import hashlib
import json
import logging
import os
import threading
import time
from typing import Any, Iterable, Optional

import cv2
import numpy as np

from .. import imageio as hio
from ..config import DEFAULTS, deep_merge, validate, column_order
from ..types import STAGES, Box, HanhuaError, PageData, TextBlock, clamp_box

log = logging.getLogger("hanhua.engine")

JOB_SCHEMA = 1
JOB_FILE = "job.json"
THUMB_SIDE = 360
STATUSES: tuple[str, ...] = ("queued", "running", "paused", "done", "error", "canceled")
ACTIVE: tuple[str, ...] = ("queued", "running")
FINAL_STAGES: tuple[str, ...] = ("rendered", "error")
SPEED_WINDOW_S = 90.0
_SAME_EXT = {".jpg": "jpg", ".jpeg": "jpg", ".jfif": "jpg", ".png": "png", ".webp": "webp"}
_OUT_EXTS = ("jpg", "png", "webp")


# --------------------------------------------------------------------------------------
# 小工具
# --------------------------------------------------------------------------------------
def page_stem(index: int) -> str:
    """第 index 页（0 起）的文件名主干：0 → "0001"。"""
    return f"{int(index) + 1:04d}"


def make_job_id(source: str) -> str:
    """源路径与文件属性签名，用于单次导入去重；独立任务 ID 另附 UUID。"""
    path = os.path.normcase(os.path.abspath(source))
    try:
        st = os.stat(source)
        size, mtime = int(st.st_size), int(st.st_mtime)
    except OSError:
        size, mtime = 0, 0
    raw = f"{path}|{size}|{mtime}".encode("utf-8", "surrogatepass")
    return hashlib.sha1(raw).hexdigest()[:12]


def write_atomic(path: str, data: bytes, attempts: int = 8) -> None:
    """原子写（hanhua.imageio.write_bytes_atomic）。

    Windows 上目标文件正被别的线程 / 进程打开（界面取图、杀毒软件扫描）时 os.replace 会
    PermissionError，稍等重试几次。"""
    delay = 0.02
    for i in range(attempts):
        try:
            hio.write_bytes_atomic(path, data)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2.0, 0.3)


def _json_default(o: Any) -> Any:
    """json 兜底：numpy 标量 / 数组、tuple、set 都转成普通类型（各模块回写的值可能是 numpy 类型）。"""
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    return str(o)


def write_json(path: str, obj: Any) -> None:
    data = json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=_json_default)
    write_atomic(path, data.encode("utf-8"))


def read_json(path: str) -> Optional[dict]:
    """读 json；文件缺失 / 损坏返回 None（不抛）。"""
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except FileNotFoundError:
        return None
    except OSError as e:
        log.warning("读取 %s 失败：%s", path, e)
        return None
    try:
        obj = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as e:
        log.warning("文件损坏，已忽略：%s（%s）", path, e)
        return None
    return obj if isinstance(obj, dict) else None


def remove_quietly(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def complete_settings(raw: Any) -> dict:
    """设置快照补齐默认值（老版本任务的快照缺新字段时不至于 KeyError）。"""
    try:
        return validate(deep_merge(DEFAULTS, raw if isinstance(raw, dict) else {}))
    except Exception:  # noqa: BLE001 - 快照坏了就用默认
        log.warning("任务设置快照无效，改用默认设置")
        return validate(deep_merge(DEFAULTS, {}))


def layout_label(settings: dict) -> str:
    ts = settings.get("typeset") or {}
    if ts.get("direction") == "h":
        return "横排"
    columns = "右到左" if column_order(settings) == "rtl" else "左到右"
    return ("竖排" if ts.get("direction") == "v" else "自动排版_列") + columns


def reading_order(settings: dict) -> str:
    """阅读顺序：设置指定 rtl / ltr 就用它；auto → 日文 rtl，其它 ltr。"""
    src = settings.get("source") or {}
    ro = str(src.get("reading_order") or "auto")
    if ro in ("rtl", "ltr"):
        return ro
    return "rtl" if str(src.get("lang") or "ja") == "ja" else "ltr"


def out_ext_for(page_name: str, image_format: str) -> str:
    """成品图片扩展名：jpg / png / webp；same 沿用原扩展名（原图是 bmp / tif 等时用无损 png）。"""
    fmt = str(image_format or "same").lower().lstrip(".")
    if fmt == "jpeg":
        fmt = "jpg"
    if fmt in _OUT_EXTS:
        return fmt
    return _SAME_EXT.get(os.path.splitext(page_name)[1].lower(), "png")


def encode_quality(ext: str, settings: dict) -> int:
    out = settings.get("output") or {}
    try:
        if ext == "webp":
            return int(out.get("webp_quality", 90))
        return int(out.get("jpeg_quality", 92))
    except (TypeError, ValueError):
        return 92


# --------------------------------------------------------------------------------------
# 文字块：规范化 / 策略 / 签名
# --------------------------------------------------------------------------------------
def _int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _pos_int_or_none(v: Any) -> Optional[int]:
    if v is None or isinstance(v, bool):
        return None
    try:
        n = int(round(float(v)))
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def _float(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if np.isfinite(f) else default


def _float_or_none(v: Any) -> Optional[float]:
    if v is None or isinstance(v, bool):
        return None
    f = _float(v, -1.0)
    return f if f >= 0 else None


def _box(v: Any, w: int, h: int) -> Optional[Box]:
    try:
        vals = [int(round(float(x))) for x in v]
    except (TypeError, ValueError):
        return None
    if len(vals) < 4:
        return None
    box: Box = (vals[0], vals[1], vals[2], vals[3])
    if w > 0 and h > 0:
        box = clamp_box(box, w, h)
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def _rgb(v: Any) -> Optional[tuple[int, int, int]]:
    if v is None:
        return None
    try:
        vals = [max(0, min(255, int(round(float(x))))) for x in v]
    except (TypeError, ValueError):
        return None
    return (vals[0], vals[1], vals[2]) if len(vals) >= 3 else None


def block_from_dict(d: Any, pos: int) -> TextBlock:
    """编辑器传来的 dict → TextBlock；数据不对抛 HanhuaError（中文）。"""
    if not isinstance(d, dict):
        raise HanhuaError(f"第 {pos + 1} 个文字块数据格式不对")
    dd = dict(d)
    if dd.get("id") is None:
        dd["id"] = pos
    if dd.get("box") is None:
        raise HanhuaError(f"第 {pos + 1} 个文字块缺少位置（box）")
    try:
        return TextBlock.from_dict(dd)
    except (TypeError, ValueError) as e:
        raise HanhuaError(f"第 {pos + 1} 个文字块数据有误：{e}") from e


def sanitize_blocks(blocks: Iterable[Any], w: int, h: int) -> list[TextBlock]:
    """把检测 / OCR / 排版 / 编辑器给的块规范成干净的 TextBlock：

    类型转换（numpy → int / float）、框裁进页面（退化成空框的丢掉）、枚举值兜底、id 唯一。"""
    out: list[TextBlock] = []
    used: set[int] = set()
    for pos, b in enumerate(list(blocks or [])):
        if isinstance(b, dict):
            try:
                b = block_from_dict(b, pos)
            except HanhuaError as e:
                log.debug("丢弃无效文字块：%s", e)
                continue
        if not isinstance(b, TextBlock):
            continue
        box = _box(b.box, w, h)
        if box is None:
            continue
        b.box = box
        b.bubble_box = _box(b.bubble_box, w, h) if b.bubble_box is not None else None
        b.render_box = _box(b.render_box, w, h) if b.render_box is not None else None
        b.text_color, b.stroke_color = _rgb(b.text_color), _rgb(b.stroke_color)
        b.fill, b.stroke = _rgb(b.fill), _rgb(b.stroke)
        b.kind = "free" if b.kind == "free" else "bubble"
        b.direction = "h" if b.direction == "h" else "v"
        b.src_text = "" if b.src_text is None else str(b.src_text)
        b.translation = "" if b.translation is None else str(b.translation)
        b.score, b.ocr_conf = _float(b.score), _float(b.ocr_conf)
        b.is_sfx, b.skip, b.overflow = bool(b.is_sfx), bool(b.skip), bool(b.overflow)
        b.font = str(b.font) if b.font else None
        b.font_size = _pos_int_or_none(b.font_size)
        b.render_font_size = _pos_int_or_none(b.render_font_size)
        b.out_direction = b.out_direction if b.out_direction in ("v", "h") else None
        b.align = b.align if b.align in ("center", "start", "end") else None
        b.stroke_width = _float_or_none(b.stroke_width)
        bid = _int(b.id, pos)
        if bid < 0 or bid in used:
            bid = max(used) + 1 if used else 0
        used.add(bid)
        b.id = bid
        out.append(b)
    return out


def apply_text_policy(blocks: Iterable[TextBlock], settings: dict) -> None:
    """拟声词 / 气泡外文字策略（OCR 之后执行一次；之后编辑器里的人工修改不会被它覆盖）。

    - 拟声词：sfx_mode=keep → skip；erase → 擦除但不嵌字（翻译阶段跳过它）；translate → 照常翻译。
    - translate_free_text=False → 气泡外的非拟声词块 skip。
    - 没认出字的块 skip：宁可保留原样，也不能把画面擦掉却什么都不填。"""
    tr = settings.get("translate") or {}
    sfx_mode = str(tr.get("sfx_mode") or "translate")
    free_ok = bool(tr.get("translate_free_text", True))
    for b in blocks:
        text = (b.src_text or "").strip()
        if b.is_sfx:
            if sfx_mode == "keep" or (sfx_mode != "erase" and not text):
                b.skip = True
        elif not text:
            b.skip = True
        elif b.kind == "free" and not free_ok:
            b.skip = True


def want_translate(b: TextBlock, settings: dict, only: Optional[set[int]] = None) -> bool:
    """这个块要不要送去翻译（只填空译文，绝不覆盖人工译文）。"""
    if b.skip or not (b.src_text or "").strip() or (b.translation or "").strip():
        return False
    if b.is_sfx and str((settings.get("translate") or {}).get("sfx_mode") or "translate") == "erase":
        return False
    return only is None or b.id in only


def clean_signature(blocks: Iterable[TextBlock]) -> str:
    """影响去字结果的块属性的签名：签名不变 → 现有去字补丁仍然有效，只需重新排版。"""
    items = []
    for b in blocks:
        if b.skip:
            continue
        items.append([
            [int(v) for v in b.box], b.kind,
            [int(v) for v in b.bubble_box] if b.bubble_box else None,
            b.direction, bool(b.is_sfx),
            [int(v) for v in b.text_color] if b.text_color else None,
            [int(v) for v in b.stroke_color] if b.stroke_color else None,
        ])
    items.sort(key=json.dumps)
    return hashlib.sha1(json.dumps(items).encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------------------
# 去字补丁
# --------------------------------------------------------------------------------------
def make_patch(src: np.ndarray, cleaned: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    """BGRA 补丁：alpha=255 的像素取去字图，其余全 0。

    alpha = mask ∪ (cleaned != src)：即使去字模块违反「mask 外不变」，合成回去也和去字图逐位一致。"""
    if cleaned.shape != src.shape:
        raise HanhuaError(f"去字结果尺寸 {cleaned.shape} 和原图 {src.shape} 不一致")
    alpha = np.any(cleaned != src, axis=2)
    if mask is not None:
        m = np.asarray(mask)
        if m.ndim == 3:
            m = m[:, :, 0]
        if m.shape == alpha.shape:
            alpha |= m > 0
    patch = np.zeros((src.shape[0], src.shape[1], 4), np.uint8)
    patch[alpha, :3] = cleaned[alpha]
    patch[alpha, 3] = 255
    return patch


def encode_patch(patch: np.ndarray) -> bytes:
    return hio.encode(patch, "png")


def decode_patch(data: bytes) -> Optional[np.ndarray]:
    """补丁 PNG 字节 → BGRA；坏数据返回 None。

    hanhua.imageio.decode 会把透明合成到白底（丢掉 alpha），所以这里直接解内存字节
    （不经过文件路径，中文路径安全）。"""
    if not data:
        return None
    try:
        arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    except cv2.error:
        return None
    if arr is None or arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] != 4:
        return None
    return arr


def apply_patch(src: np.ndarray, patch: np.ndarray) -> np.ndarray:
    """原图 + 补丁 → 去字图（新数组，不改 src）。"""
    if patch.shape[:2] != src.shape[:2]:
        raise HanhuaError("去字补丁和原图尺寸不一致")
    out = src.copy()
    m = patch[:, :, 3] > 0
    out[m] = patch[:, :, :3][m]
    return out


def as_bgr(img: Any) -> np.ndarray:
    """书本读出来的页 → BGR uint8 (H, W, 3)；读不出来抛 HanhuaError。"""
    if not isinstance(img, np.ndarray) or img.size == 0:
        raise HanhuaError("这一页图片读不出来")
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if img.ndim == 3 and img.shape[2] == 3:
        return np.ascontiguousarray(img)
    raise HanhuaError(f"这一页图片格式不对：{img.shape}")


# --------------------------------------------------------------------------------------
# Job
# --------------------------------------------------------------------------------------
class Job:
    """一本书的任务。字段名与合同 5.7 一致；除持久化字段外的运行时字段只由 Engine 在锁内修改。"""

    def __init__(self, id: str, source: str, title: str, kind: str, workdir: str, settings: dict,
                 pages: list[PageData], status: str = "queued", message: str = "", out_path: str = "",
                 created: Optional[float] = None, persistent: bool = True) -> None:
        self.id = id
        self.source = source
        self.title = title
        self.kind = kind
        self.workdir = workdir
        self.out_path = out_path
        self.status = status if status in STATUSES else "paused"
        self.message = message
        self.settings = settings
        self.pages = pages
        self.created = float(created or time.time())
        self.updated = time.time()
        self.persistent = persistent          # 打不开的文件生成的「错误任务」只在内存里
        self.removed = False
        self.purged = False                   # keep_workdir=False：导出后中间文件已清理
        self.dirty = False                    # 导出后又改过页面（成品需要重新导出）
        self.pause_reason = ""                # user | translate（翻译全挂自动暂停，会定时重试）
        n = len(pages)
        self.meta: list[dict] = [{} for _ in range(n)]   # 每页引擎附加信息（clean_sig / failed_stage），随页落盘
        # ---- 运行时（不落盘）----
        self.gen: list[int] = [0] * n         # 每页代号：人工修改 +1，旧结果提交时被丢弃
        self.inflight: set[int] = set()       # 正在某个阶段处理中的页
        self.manual: set[int] = set()         # 人工请求的页：不管任务状态都优先处理
        self.tr_ops: dict[int, Any] = {}      # 待处理的翻译请求："all" 全部重译 | set[int] 只翻这些块
        self.hint = 0                         # 第一个还没完成（rendered / error）的页（扫描加速）
        self.retry_count = 0
        self.retry_at: Optional[float] = None
        self.exporting = False
        self.run_started = 0.0
        self.book: Any = None
        self.book_lock = threading.RLock()
        self.export_lock = threading.Lock()
        self.io_lock = threading.Lock()       # 写补丁 / 成品 / 缩略图时持有；remove(delete_files) 删目录前也要拿到它
        self.comps: Any = None
        self.tr_cancel: Optional[threading.Event] = None
        self._done_times: collections.deque[float] = collections.deque(maxlen=256)

    # ---- 路径 ------------------------------------------------------------------------
    def path(self, *parts: str) -> str:
        return os.path.join(self.workdir, *parts)

    def page_json(self, i: int) -> str:
        return self.path("pages", page_stem(i) + ".json")

    def clean_file(self, i: int) -> str:
        return self.path("clean", page_stem(i) + ".png")

    def out_ext(self, i: int) -> str:
        return out_ext_for(self.pages[i].name, str((self.settings.get("output") or {}).get("image_format") or "same"))

    def out_file(self, i: int) -> str:
        return self.path("out", f"{page_stem(i)}.{self.out_ext(i)}")

    def out_variants(self, i: int) -> list[str]:
        return [self.path("out", f"{page_stem(i)}.{e}") for e in _OUT_EXTS]

    def thumb_file(self, i: int) -> str:
        return self.path("thumbs", page_stem(i) + ".jpg")

    def src_cache_file(self, i: int) -> str:
        ext = os.path.splitext(self.pages[i].name)[1].lower()
        if not hio.is_image_name("x" + ext):
            ext = ".png"
        return self.path("src_cache", page_stem(i) + ext)

    def view_file(self, i: int) -> str:
        return self.path("view", page_stem(i) + ".jpg")

    # ---- 持久化 ----------------------------------------------------------------------
    def to_meta(self) -> dict:
        return {
            "schema": JOB_SCHEMA, "id": self.id, "source": self.source, "title": self.title,
            "kind": self.kind, "out_path": self.out_path, "status": self.status, "message": self.message,
            "settings": self.settings, "created": self.created, "updated": self.updated,
            "page_names": [p.name for p in self.pages], "removed": self.removed, "purged": self.purged,
            "dirty": self.dirty, "pause_reason": self.pause_reason,
        }

    def save_meta(self) -> None:
        if not self.persistent:
            return
        self.updated = time.time()
        write_json(self.path(JOB_FILE), self.to_meta())

    def save_page(self, i: int) -> None:
        if not self.persistent or self.removed:
            return
        d = self.pages[i].to_dict()
        d["_engine"] = self.meta[i]
        write_json(self.page_json(i), d)
        self.updated = time.time()

    @classmethod
    def load(cls, workdir: str) -> Optional["Job"]:
        """从工作目录恢复；job.json 缺失 / 损坏返回 None。单页 json 缺失 / 损坏 → 该页从头来。"""
        meta = read_json(os.path.join(workdir, JOB_FILE))
        if not meta or not meta.get("id") or not isinstance(meta.get("page_names"), list):
            return None
        names = [str(n) for n in meta["page_names"]]
        pages: list[PageData] = []
        metas: list[dict] = []
        for i, name in enumerate(names):
            d = read_json(os.path.join(workdir, "pages", page_stem(i) + ".json"))
            page: Optional[PageData] = None
            if d is not None:
                try:
                    page = PageData.from_dict(d)
                    if page.index != i or page.name != name:
                        page = None
                except Exception:  # noqa: BLE001 - 单页数据坏了：这一页从头处理
                    page = None
            if page is None:
                pages.append(PageData(index=i, name=name))
                metas.append({})
                continue
            if page.stage not in STAGES and page.stage != "error":
                page.stage = "pending"
            page.blocks = sanitize_blocks(page.blocks, page.width, page.height)
            em = d.get("_engine") if d is not None else None
            metas.append(dict(em) if isinstance(em, dict) else {})
            pages.append(page)
        job = cls(
            id=str(meta["id"]), source=str(meta.get("source", "")), title=str(meta.get("title", "")),
            kind=str(meta.get("kind", "")), workdir=workdir, settings=complete_settings(meta.get("settings")),
            pages=pages, status=str(meta.get("status", "paused")), message=str(meta.get("message", "")),
            out_path=str(meta.get("out_path", "") or ""), created=_float(meta.get("created"), time.time()),
        )
        job.meta = metas
        job.removed = bool(meta.get("removed", False))
        job.purged = bool(meta.get("purged", False))
        job.dirty = bool(meta.get("dirty", False))
        job.pause_reason = str(meta.get("pause_reason", "") or "")
        job.updated = _float(meta.get("updated"), job.created)
        return job

    def clean_temp_files(self) -> None:
        """删掉强杀时可能残留的 .tmp_ 临时文件（write_bytes_atomic 的半成品）。"""
        for sub in ("", "pages", "clean", "out", "thumbs", "src_cache", "view"):
            d = self.path(sub) if sub else self.workdir
            try:
                names = os.listdir(d)
            except OSError:
                continue
            for n in names:
                if n.startswith(".tmp_"):
                    remove_quietly(os.path.join(d, n))

    # ---- 统计 ------------------------------------------------------------------------
    def advance_hint(self) -> int:
        n = len(self.pages)
        while self.hint < n and self.pages[self.hint].stage in FINAL_STAGES:
            self.hint += 1
        return self.hint

    def record_done(self, now: float) -> None:
        self._done_times.append(now)

    def speed_ppm(self, now: Optional[float] = None) -> float:
        """滑动窗口速度（页 / 分钟）：最近 90 秒内完成的页数 ÷ 窗口时长。"""
        if self.status != "running":
            return 0.0
        now = time.time() if now is None else now
        recent = [t for t in self._done_times if now - t <= SPEED_WINDOW_S]
        if not recent:
            return 0.0
        start = max(now - SPEED_WINDOW_S, self.run_started or recent[0])
        span = max(3.0, now - start)
        return round(60.0 * len(recent) / span, 1)

    def stage_counts(self) -> dict[str, int]:
        counts = {s: 0 for s in STAGES}
        counts["error"] = 0
        for p in self.pages:
            counts[p.stage if p.stage in counts else "pending"] += 1
        return counts

    def summary(self, now: Optional[float] = None) -> dict:
        now = time.time() if now is None else now
        counts = self.stage_counts()
        total, done, failed = len(self.pages), counts["rendered"], counts["error"]
        remaining = max(0, total - done - failed)
        speed = self.speed_ppm(now)
        eta: Optional[int]
        if remaining == 0:
            eta = 0
        elif speed > 0:
            eta = int(round(remaining * 60.0 / speed))
        else:
            eta = None
        cover = self.thumb_file(0) if (self.persistent and self.pages) else ""
        return {
            "id": self.id, "title": self.title, "source": self.source, "kind": self.kind,
            "status": self.status, "message": self.message, "total": total, "done": done, "failed": failed,
            "stage_counts": counts, "speed_ppm": speed, "eta_s": eta, "out_path": self.out_path or "",
            "cover": cover if cover and os.path.exists(cover) else None, "updated": self.updated,
            # ---- 合同之外的附加信息（界面可选用）----
            "failed_pages": [p.index for p in self.pages if p.stage == "error"][:500],
            "dirty": self.dirty,
            "layout_label": layout_label(self.settings),
            "retry_in": max(0, int(round(self.retry_at - now))) if self.retry_at else None,
        }
