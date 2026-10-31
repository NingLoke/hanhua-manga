"""任务引擎：多本书排队、流水线并行、每页每阶段原子落盘、断点续跑、人工修改后只重做必要的阶段。

线程模型（合同 5.7）::

    视觉 ×1（检测 + OCR，GPU）→ 翻译协调 ×1（攒 1~4 页一批调 translate_blocks，网络并发由 TranslationService 负责）
      → 去字 ×1（GPU）→ 渲染池 ×profile.cpu_workers（排版 + 编码 + 缩略图）
    导出 ×1（整本都到终态 → bookio.export_book）      节拍 ×1（事件合并节流 / 翻译重试定时器 / 速度刷新）

调度是「拉取式」的：页面 stage 是唯一事实来源。各工作线程在引擎锁内挑出下一张该自己处理的页（标记 inflight），
锁外干活，再回到锁内用页代号 gen 校验后提交并原子落盘。所以暂停 / 继续 / 取消 / 人工修改 / 强杀重启都只是改
stage 或任务状态，不会有「队列里残留的旧任务」。阶段之间的有界队列由背压实现：视觉最多领先下游 max_ahead 页，
去字最多领先渲染 2×线程数+2 页；页面图片只经按字节限额的 LRU 缓存在阶段间传递（超限即丢，下游按需从书 /
去字补丁重建），内存始终有上界。多本书按队列顺序处理，最多 max_open_jobs 本同时在流水线里（前一本的尾巴和
下一本的视觉阶段重叠）。

锁顺序：引擎锁内不做耗时 IO（只写几 KB 的页 json）；job.book_lock / job.io_lock 内绝不再拿引擎锁；
界面事件回调只在节拍线程里、不持任何锁时调用。
"""
from __future__ import annotations

import copy
import importlib
import json
import logging
import os
import shutil
import stat
import sys
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np

from .. import imageio as hio
from ..config import SettingsStore
from ..runtime import data_dir, fonts_dir
from ..types import (
    STAGES, Box, Cancelled, HanhuaError, PageData, TextBlock, TranslationUnavailable, UnsupportedFormat, stage_at_least,
)
from .job import (
    ACTIVE, FINAL_STAGES, JOB_FILE, THUMB_SIDE, Job, _box, apply_patch, apply_text_policy, as_bgr, block_from_dict,
    clean_signature, complete_settings, decode_patch, encode_patch, encode_quality, make_job_id, make_patch,
    reading_order, remove_quietly, sanitize_blocks, want_translate, write_atomic, layout_label,
)

log = logging.getLogger("hanhua.engine")

__all__ = ["ComponentFactory", "Engine"]

EventCallback = Callable[[dict], None]

MAX_RENDER_THREADS = 4
TRANSLATE_BATCH_PAGES = 4
SRC_CACHE_MB, SRC_CACHE_ITEMS = 384, 16
CLEAN_CACHE_MB, CLEAN_CACHE_ITEMS = 256, 8
SHUTDOWN_BUDGET_S = 1.9
REMOVE_WAIT_S = 3.0
MAX_RECLEAN = 2

STEP_CN: dict[str, str] = {
    "read": "读取图片", "detect": "文字检测", "ocr": "文字识别", "translate": "翻译", "clean": "去字",
    "render": "嵌字排版", "save": "保存结果", "load": "加载组件", "internal": "处理",
}
COMP_CN: dict[str, str] = {
    "detector": "文字检测", "ocr": "文字识别", "translator": "翻译", "inpainter": "去字", "typesetter": "嵌字排版",
}
_MAGIC: dict[str, bytes] = {"jpg": b"\xff\xd8\xff", "png": b"\x89PNG\r\n\x1a\n", "webp": b"RIFF"}


# --------------------------------------------------------------------------------------
# 小工具
# --------------------------------------------------------------------------------------
def _err_text(e: BaseException, limit: int = 300) -> str:
    """异常 → 给客户看的一句话（HanhuaError 本身就是中文；其它异常带上类型名，方便客户反馈）。"""
    if isinstance(e, HanhuaError) and str(e).strip():
        text = str(e).strip()
    else:
        detail = str(e).strip()
        text = f"{type(e).__name__}: {detail}" if detail else type(e).__name__
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _fmt_delay(seconds: float) -> str:
    s = max(0, int(round(seconds)))
    if s >= 60:
        return f"{s // 60} 分钟" if s % 60 == 0 else f"{s // 60} 分 {s % 60} 秒"
    return f"{s} 秒"


def _title_of(path: str) -> str:
    base = os.path.basename(os.path.normpath(path)) or path
    stem, ext = os.path.splitext(base)
    return stem if ext and len(ext) <= 6 else base


def _dispose(obj: Any) -> None:
    """有 close() 就调用（关书 / 关翻译缓存 / 释放模型）；出错只记日志。"""
    close = getattr(obj, "close", None)
    if callable(close):
        try:
            close()
        except Exception:  # noqa: BLE001
            log.debug("关闭 %r 出错", obj, exc_info=True)


def _on_rm_error(func: Callable[[str], Any], path: str, _exc: Any) -> None:
    """rmtree 遇到只读文件：去掉只读属性再删一次。"""
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except OSError:
        pass


def _rmtree(path: str, attempts: int = 4) -> bool:
    """删除目录（Windows 上文件被界面 / 杀毒占用时稍等重试）。返回是否删干净。"""
    for k in range(attempts):
        if not os.path.exists(path):
            return True
        try:
            if sys.version_info >= (3, 12):
                shutil.rmtree(path, onexc=_on_rm_error)
            else:  # pragma: no cover - 3.12 以下
                shutil.rmtree(path, onerror=_on_rm_error)
        except FileNotFoundError:
            return True
        except OSError as e:
            if k == attempts - 1:
                log.warning("删除 %s 失败：%s", path, e)
        if not os.path.exists(path):
            return True
        time.sleep(0.1 * (k + 1))
    return not os.path.exists(path)


class _ImageCache:
    """线程安全、按字节限额的 LRU：在阶段之间传递页面图片（超限丢最旧的，下游按需重建）。

    缓存里的数组只读共享：取出来要改的人（去字）自己先 copy。"""

    def __init__(self, budget_mb: int, max_items: int) -> None:
        self._budget = int(budget_mb) * 1024 * 1024
        self._max = max(1, int(max_items))
        self._lock = threading.Lock()
        self._items: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self._bytes = 0

    def put(self, key: tuple, img: Optional[np.ndarray]) -> None:
        if img is None:
            return
        n = int(img.nbytes)
        if n > self._budget:
            return
        with self._lock:
            old = self._items.pop(key, None)
            if old is not None:
                self._bytes -= int(old.nbytes)
            self._items[key] = img
            self._bytes += n
            while self._items and (self._bytes > self._budget or len(self._items) > self._max):
                _, ev = self._items.popitem(last=False)
                self._bytes -= int(ev.nbytes)

    def get(self, key: tuple) -> Optional[np.ndarray]:
        with self._lock:
            img = self._items.get(key)
            if img is not None:
                self._items.move_to_end(key)
            return img

    def pop(self, key: tuple) -> Optional[np.ndarray]:
        with self._lock:
            img = self._items.pop(key, None)
            if img is not None:
                self._bytes -= int(img.nbytes)
            return img

    def drop(self, pred: Callable[[tuple], bool]) -> None:
        with self._lock:
            for k in [k for k in self._items if pred(k)]:
                self._bytes -= int(self._items.pop(k).nbytes)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._bytes = 0


@dataclass
class _Task:
    kind: str                 # vision | translate | clean | render | export
    job: Job
    indices: list[int]
    gens: list[int]
    settings: dict            # 任务快照 + 实时硬件设置
    manual: bool = False      # 人工请求（编辑器里的修改）：不管任务状态都处理


class _JobFailure(HanhuaError):
    """整本书都做不下去的错误（源文件打不开 / 组件加载失败）：任务标 error，页面进度保留，可「继续」重试。"""


class _NeedReclean(Exception):
    """去字补丁缺失 / 损坏：页面退回去字阶段重做。"""


# --------------------------------------------------------------------------------------
# 组件工厂
# --------------------------------------------------------------------------------------
class ComponentFactory:
    """组件工厂：真实组件懒加载 + 按设置缓存（线程安全）。

    Engine 只通过这里拿组件、读写书本；测试继承本类覆盖方法即可注入假组件（不需要 GPU / 网络）。
    组件的缓存 key = 决定它的那部分设置：硬件（设备 / CPU 线程 / 显卡序号）、画质档位、源语言，以及去字 /
    排版 / 翻译各自的设置段。设置一变，下次取组件时按新 key 自动重建；被挤出缓存的旧实例可能还有线程在用，
    先「退休」，到 close() 时统一关闭。首次取组件（加载模型）发生在 Engine 的后台工作线程里。
    """

    CACHE_SIZE = 3
    _KINDS: tuple[str, ...] = ("profile", "detector", "ocr", "translator", "inpainter", "typesetter")

    def __init__(self) -> None:
        self._rt_lock = threading.Lock()
        self._fonts_lock = threading.Lock()
        self._locks: dict[str, threading.Lock] = {k: threading.Lock() for k in self._KINDS}
        self._caches: dict[str, OrderedDict[Any, Any]] = {k: OrderedDict() for k in self._KINDS}
        self._retired: list[Any] = []
        self._rt: Any = None
        self._rt_key: Optional[tuple[str, int, int]] = None
        self._fonts: Any = None
        self._label = ""

    # ---- key / 缓存 --------------------------------------------------------------------
    @staticmethod
    def hardware_key(settings: dict) -> tuple[str, int, int]:
        hw = settings.get("hardware") or {}

        def _i(v: Any) -> int:
            try:
                return max(0, int(v or 0))
            except (TypeError, ValueError):
                return 0

        return (str(hw.get("device") or "auto"), _i(hw.get("cpu_threads")), _i(hw.get("gpu_id")))

    @staticmethod
    def _skey(settings: dict, *paths: str) -> str:
        """若干设置段 / 设置项（点路径，如 "translate.target"）→ 缓存 key 字符串。"""
        vals: list[Any] = []
        for p in paths:
            cur: Any = settings
            for part in p.split("."):
                cur = cur.get(part) if isinstance(cur, dict) else None
            vals.append(cur)
        return json.dumps(vals, sort_keys=True, ensure_ascii=False, default=str)

    def _cached(self, kind: str, key: Any, build: Callable[[], Any]) -> Any:
        with self._locks[kind]:
            cache = self._caches[kind]
            if key in cache:
                cache.move_to_end(key)
                return cache[key]
            obj = build()
            cache[key] = obj
            while len(cache) > self.CACHE_SIZE:
                _, old = cache.popitem(last=False)
                self._retired.append(old)
            return obj

    @staticmethod
    def _import(module: str, name: str, what: str) -> Any:
        try:
            mod = importlib.import_module(module, __package__)
            return getattr(mod, name)
        except Exception as e:  # noqa: BLE001 - 半成品模块的 SyntaxError 等也要给中文说明
            raise HanhuaError(f"{what}模块（hanhua{module.lstrip('.') and '.' + module.lstrip('.')}）"
                              f"没有装好或已损坏：{_err_text(e)}") from e

    # ---- 推理组件 ----------------------------------------------------------------------
    def runtime(self, settings: dict) -> Any:
        key = self.hardware_key(settings)
        old = None
        with self._rt_lock:
            if self._rt is None or self._rt_key != key:
                runtime_cls = self._import("..runtime", "Runtime", "推理后端")
                try:
                    rt = runtime_cls(key[0], key[1], key[2])
                except Exception as e:  # noqa: BLE001
                    raise HanhuaError(f"推理后端初始化失败：{_err_text(e)}") from e
                old, self._rt, self._rt_key = self._rt, rt, key
                try:
                    self._label = str(rt.label)
                except Exception:  # noqa: BLE001
                    self._label = str(getattr(rt, "device", ""))
                log.info("推理后端就绪：%s", self._label)
            rt = self._rt
        if old is not None:
            try:
                old.release()
            except Exception:  # noqa: BLE001
                log.debug("释放旧推理后端出错", exc_info=True)
        return rt

    def profile(self, settings: dict) -> Any:
        rt = self.runtime(settings)
        name = str((settings.get("hardware") or {}).get("profile") or "balanced")

        def build() -> Any:
            resolve = self._import("..runtime", "resolve_profile", "推理后端")
            return resolve(name, rt.device, getattr(rt, "hw", None))

        return self._cached("profile", (self.hardware_key(settings), name, str(rt.device)), build)

    def detector(self, settings: dict) -> Any:
        rt, prof = self.runtime(settings), self.profile(settings)

        def build() -> Any:
            return self._import("..detect", "TextDetector", "文字检测")(rt, prof)

        return self._cached("detector", (self.hardware_key(settings), prof.name), build)

    def ocr(self, settings: dict) -> Any:
        rt, prof = self.runtime(settings), self.profile(settings)
        lang = str((settings.get("source") or {}).get("lang") or "ja")

        def build() -> Any:
            return self._import("..ocr", "OCREngine", "文字识别")(rt, prof, lang)

        return self._cached("ocr", (self.hardware_key(settings), prof.name, lang), build)

    def inpainter(self, settings: dict) -> Any:
        rt, prof = self.runtime(settings), self.profile(settings)

        def build() -> Any:
            return self._import("..inpaint", "Inpainter", "去字")(rt, prof, settings)

        return self._cached("inpainter", (self.hardware_key(settings), prof.name, self._skey(settings, "inpaint")), build)

    def translator(self, settings: dict) -> Any:
        def build() -> Any:
            cls = self._import("..translate", "TranslationService", "翻译")
            cache_dir = os.path.join(data_dir(), "cache")
            os.makedirs(cache_dir, exist_ok=True)
            return cls(settings, os.path.join(cache_dir, "translate.sqlite3"))

        return self._cached("translator", self._skey(settings, "translate", "source"), build)

    def fonts(self) -> Any:
        with self._fonts_lock:
            if self._fonts is None:
                cls = self._import("..typeset", "FontRegistry", "嵌字排版")
                user = os.path.join(data_dir(), "fonts")
                try:
                    os.makedirs(user, exist_ok=True)
                except OSError:
                    pass
                self._fonts = cls(fonts_dir(), [user])
            return self._fonts

    def typesetter(self, settings: dict) -> Any:
        fonts = self.fonts()

        def build() -> Any:
            return self._import("..typeset", "Typesetter", "嵌字排版")(settings, fonts)

        key = self._skey(settings, "typeset", "translate.target", "translate.sfx_mode")
        return self._cached("typesetter", key, build)

    def cpu_workers(self, settings: dict) -> int:
        return int(getattr(self.profile(settings), "cpu_workers", 1) or 1)

    def device_label(self) -> str:
        """当前推理设备的显示名（还没加载模型时为空）。不加锁，节拍线程随时可调。"""
        return self._label

    # ---- 书本读写（hanhua.bookio）--------------------------------------------------------
    def _bookio(self) -> Any:
        try:
            return importlib.import_module("..bookio", __package__)
        except Exception as e:  # noqa: BLE001
            raise HanhuaError(f"文件读写模块（hanhua.bookio）没有装好或已损坏：{_err_text(e)}") from e

    def scan_inputs(self, paths: list[str], tmp_root: str) -> list[str]:
        return [str(p) for p in self._bookio().scan_inputs(list(paths), tmp_root)]

    def open_book(self, path: str, tmp_dir: str) -> Any:
        return self._bookio().open_book(path, tmp_dir)

    def default_output_path(self, book: Any, settings: dict) -> str:
        return str(self._bookio().default_output_path(book, settings))

    def export_book(self, book: Any, pages: list[tuple[str, str]], out_path: str, settings: dict,
                    progress: Optional[Callable[[int, int], None]] = None) -> str:
        return str(self._bookio().export_book(book, pages, out_path, settings, progress=progress))

    def close(self) -> None:
        objs = list(self._retired)
        self._retired.clear()
        for kind in self._KINDS:
            with self._locks[kind]:
                objs.extend(self._caches[kind].values())
                self._caches[kind].clear()
        for o in objs:
            _dispose(o)
        with self._rt_lock:
            rt, self._rt, self._rt_key = self._rt, None, None
        if rt is not None:
            try:
                rt.release()
            except Exception:  # noqa: BLE001
                log.debug("释放推理后端出错", exc_info=True)


# --------------------------------------------------------------------------------------
# 引擎
# --------------------------------------------------------------------------------------
class Engine:
    """任务引擎（合同 5.7）。所有公开方法线程安全；耗时的事（加载模型 / 推理 / 联网）都在后台线程里做。

    合同之外的关键字参数（都有默认值；界面不用管，测试用来提速）：
    - retry_delays：翻译全挂后自动重试的间隔（秒），默认 1 / 2 / 5 分钟，之后一直按最后一个间隔重试；
    - batch_wait：翻译攒批时最多等后续页面多久（秒）；
    - max_ahead：视觉阶段最多领先下游多少页（背压，控制内存）；
    - max_open_jobs：同时在流水线里的书最多几本；
    - tick：事件合并 / 定时器节拍（秒）；同一个 key 的事件每节拍最多发一次（默认 ≤10 次/秒）。
    """

    def __init__(self, store: SettingsStore, on_event: Optional[EventCallback] = None,
                 work_root: Optional[str] = None, components: Optional[ComponentFactory] = None, *,
                 retry_delays: Sequence[float] = (60.0, 120.0, 300.0), batch_wait: float = 0.6,
                 max_ahead: int = 12, max_open_jobs: int = 2, tick: float = 0.1) -> None:
        self.store = store
        self._on_event = on_event
        self._factory: ComponentFactory = components if components is not None else ComponentFactory()
        delays = []
        for x in retry_delays or ():
            try:
                delays.append(max(0.0, float(x)))
            except (TypeError, ValueError):
                continue
        self._retry_delays: tuple[float, ...] = tuple(delays) or (60.0,)
        self._batch_wait = max(0.0, float(batch_wait))
        self._max_ahead = max(1, int(max_ahead))
        self._max_open = max(1, int(max_open_jobs))
        self._tick = min(1.0, max(0.02, float(tick)))

        self._lock = threading.RLock()
        self._cv = threading.Condition(self._lock)
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._started = False
        self._stopping = False
        self._closed = False
        self._threads: list[threading.Thread] = []
        self._workers_started = False
        self._render_limit = max(1, min(MAX_RENDER_THREADS, (os.cpu_count() or 2) // 3))
        self._render_key: Optional[str] = None
        self._tr_wait: dict[str, float] = {}
        self._announced_load = False
        self._src_cache = _ImageCache(SRC_CACHE_MB, SRC_CACHE_ITEMS)
        self._clean_cache = _ImageCache(CLEAN_CACHE_MB, CLEAN_CACHE_ITEMS)
        # 事件（节拍线程合并后发送）
        self._ev_jobs: set[str] = set()
        self._ev_gone: dict[str, dict] = {}
        self._ev_pages: dict[tuple[str, int], dict] = {}
        self._ev_logs: list[dict] = []
        self._ev_engine_last: Optional[tuple[bool, str]] = None
        self._last_periodic = 0.0
        self._cb_errors = 0
        self._wake = threading.Event()

        settings = self._settings_snapshot()
        self._hw: dict = dict(settings.get("hardware") or {})
        self._output_lock = threading.Lock()
        root = work_root or str((settings.get("output") or {}).get("work_dir") or "").strip() \
            or os.path.join(data_dir(), "jobs")
        self.work_root = os.path.abspath(root)
        try:
            os.makedirs(self.work_root, exist_ok=True)
        except OSError as e:
            raise HanhuaError(f"工作目录无法创建：{self.work_root}（{e}）") from e
        self._inputs_root = os.path.join(self.work_root, "_inputs")
        try:
            store.on_change(self._on_settings)
        except Exception:  # noqa: BLE001
            log.debug("注册设置监听失败", exc_info=True)
        self._restore(bool((settings.get("general") or {}).get("auto_resume", True)))
        self._ticker = threading.Thread(target=self._tick_loop, name="hanhua-engine-tick", daemon=True)
        self._ticker.start()
        with self._cv:
            if self._started:
                self._ensure_workers()

    # ================================================================== 公开接口：任务
    def add_paths(self, paths: Iterable[Any] | str) -> list[dict]:
        """每次导入创建独立任务；只在本次导入内按实际源路径去重。"""
        self._check_alive()
        items = self._normalize_paths(paths)
        if not items:
            return []
        try:
            os.makedirs(self._inputs_root, exist_ok=True)
        except OSError as e:
            raise HanhuaError(f"工作目录无法写入：{e}") from e
        books, failures = self._scan(items)
        jobs: list[Job] = [self._error_job(p, f"无法读取：{msg}") for p, msg in failures]
        seen: set[str] = set()
        for path in books:
            source_key = make_job_id(path)
            if source_key in seen:
                continue
            seen.add(source_key)
            jid = source_key + "-" + uuid.uuid4().hex
            try:
                jobs.append(self._add_book(path, jid))
            except Exception as e:  # noqa: BLE001 - 单本打不开不影响其它
                log.warning("打开 %s 失败：%s", path, e, exc_info=not isinstance(e, HanhuaError))
                jobs.append(self._error_job(path, f"打不开：{_err_text(e)}"))
        with self._cv:
            if not jobs:
                self._log_event("warning", "没有找到可以处理的漫画（支持文件夹、图片、压缩包、PDF、EPUB）")
            out = [j.summary() for j in jobs]
            self._cv.notify_all()
        return out

    def list_jobs(self) -> list[dict]:
        with self._lock:
            return [self._jobs[j].summary() for j in self._order if j in self._jobs]

    def start(self) -> None:
        """开始处理队列（排队中的任务）。"""
        self._check_alive()
        with self._cv:
            self._started = True
            self._ensure_workers()
            self._cv.notify_all()

    def pause(self, job_id: Optional[str] = None) -> None:
        """暂停一本（job_id）或全部（None：之后新加的任务也先不跑，直到 start / resume）。正在算的那一页算完再停。"""
        with self._cv:
            jobs = [self._job(job_id)] if job_id else [self._jobs[j] for j in self._order if j in self._jobs]
            for job in jobs:
                if not job.persistent:
                    continue
                if job.status in ACTIVE or (job.status == "paused" and job.pause_reason == "translate"):
                    job.status, job.pause_reason, job.retry_at, job.message = "paused", "user", None, "已暂停"
                    if job.tr_cancel is not None:
                        job.tr_cancel.set()
                    self._save_meta_safe(job)
                    self._mark_job(job)
            if not job_id:
                self._started = False
            self._cv.notify_all()

    def resume(self, job_id: Optional[str] = None) -> None:
        """继续一本（暂停 / 取消 / 出错的都可以）或全部暂停中的任务。"""
        self._check_alive()
        with self._cv:
            if job_id:
                job = self._job(job_id)
                if job.persistent and job.status in ("paused", "canceled", "error"):
                    self._revive(job)
            else:
                for jid in self._order:
                    job = self._jobs.get(jid)
                    if job is not None and job.persistent and job.status == "paused":
                        self._revive(job)
            self._started = True
            self._ensure_workers()
            self._cv.notify_all()

    def cancel(self, job_id: str) -> None:
        with self._cv:
            job = self._job(job_id)
            if not job.persistent or job.status == "canceled":
                return
            job.manual.clear()
            if job.status != "done":
                job.status, job.pause_reason, job.retry_at, job.message = "canceled", "", None, "已取消"
                self._log_event("info", f"已取消《{job.title}》")
            if job.tr_cancel is not None:
                job.tr_cancel.set()
            self._save_meta_safe(job)
            self._mark_job(job)
            self._cv.notify_all()
        self._drop_caches(job)
        self._release_book(job)

    def remove(self, job_id: str, delete_files: bool = True) -> None:
        """删除任务，默认清除其中间结果；原漫画和已导出成品保留。"""
        with self._cv:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.removed = True
            job.manual.clear()
            if job.tr_cancel is not None:
                job.tr_cancel.set()
            self._jobs.pop(job.id, None)
            if job.id in self._order:
                self._order.remove(job.id)
            self._tr_wait.pop(job.id, None)
            self._ev_jobs.discard(job.id)
            self._ev_pages = {k: v for k, v in self._ev_pages.items() if k[0] != job.id}
            gone = job.summary()
            gone["removed"] = True
            self._ev_gone[job.id] = gone
            deadline = time.monotonic() + REMOVE_WAIT_S
            while (job.inflight or job.exporting) and time.monotonic() < deadline:
                self._cv.wait(timeout=0.05)
            self._cv.notify_all()
        self._drop_caches(job)
        self._release_book(job)
        if job.persistent:
            try:
                job.save_meta()
            except OSError as e:
                log.warning("保存删除标记 %s 失败：%s", job.id, e)
            if delete_files:
                with job.io_lock:
                    ok = _rmtree(job.workdir)
                if not ok:
                    self._log_event("warning", f"《{job.title}》的工作目录没能完全删除（文件被占用）：{job.workdir}")
            else:
                try:
                    job.save_meta()      # removed=True：下次启动不再列出
                except OSError as e:
                    log.warning("保存任务 %s 失败：%s", job.id, e)
        self._log_event("info", f"已移除《{job.title}》")

    def retry_failed(self, job_id: str) -> None:
        """出错的页从出错的那个阶段重新来；任务本身出错（导出失败 / 打不开）也会重新排队。"""
        self._check_alive()
        with self._cv:
            job = self._editable(job_id)
            n = 0
            for i, p in enumerate(job.pages):
                if p.stage != "error":
                    continue
                last = str(job.meta[i].get("last_stage") or "pending")
                if last not in STAGES or last == "rendered":
                    last = "pending"
                if last == "cleaned" and not os.path.isfile(job.clean_file(i)):
                    last = "translated"
                p.stage, p.error = last, ""
                for k in ("failed_stage", "reclean"):
                    job.meta[i].pop(k, None)
                job.gen[i] += 1
                job.hint = min(job.hint, i)
                self._save_page_safe(job, i)
                self._mark_page(job, i)
                n += 1
            if n or job.status in ("error", "canceled", "paused"):
                if job.status not in ACTIVE:
                    job.status, job.pause_reason, job.retry_at, job.retry_count = "queued", "", None, 0
                job.message = f"重新处理 {n} 个出错的页面" if n else "排队中"
                self._save_meta_safe(job)
                self._mark_job(job)
                self._started = True
                self._ensure_workers()
            self._cv.notify_all()

    # ================================================================== 公开接口：页面
    def get_page(self, job_id: str, index: int) -> dict:
        with self._lock:
            job = self._job(job_id)
            return self._page_dict(job, self._check_index(job, index))

    def update_page(self, job_id: str, index: int, blocks: list[dict]) -> dict:
        """保存编辑器里的人工修改并重新排版。只有影响去字的属性（skip / 框 / 增删块）变了才重新去字；
        不重新检测、不覆盖人工译文。立即返回，排好后发 page 事件（version +1）。"""
        self._check_alive()
        if not isinstance(blocks, (list, tuple)):
            raise HanhuaError("文字块数据格式不对（应该是列表）")
        parsed: list[TextBlock] = []
        for pos, b in enumerate(blocks):
            tb = block_from_dict(b, pos) if isinstance(b, dict) else b
            if not isinstance(tb, TextBlock):
                raise HanhuaError(f"第 {pos + 1} 个文字块数据格式不对")
            parsed.append(copy.deepcopy(tb))
        with self._cv:
            job = self._editable(job_id)
            i = self._check_index(job, index)
            p = job.pages[i]
            new_blocks = sanitize_blocks(parsed, p.width, p.height)
            patch_ok = job.meta[i].get("clean_sig") == clean_signature(new_blocks) and os.path.isfile(job.clean_file(i))
            if p.stage in ("pending", "detected", "recognized"):
                stage = "recognized"          # 还没翻译：翻译阶段只补空译文，人工写的保留
            elif patch_ok and p.stage in ("cleaned", "rendered", "error"):
                stage = "cleaned"             # 去字补丁仍然有效：只重排
            else:
                stage = "translated"          # skip / 框变了：重新去字再排
            auto = set(job.meta[i].get("auto_skip") or [])
            if auto:
                job.meta[i]["auto_skip"] = sorted(b.id for b in new_blocks if b.id in auto and b.skip)
            p.blocks, p.stage, p.error = new_blocks, stage, ""
            job.meta[i].pop("failed_stage", None)
            self._touch_page(job, i)
            return self._page_dict(job, i)

    def retranslate_page(self, job_id: str, index: int) -> dict:
        """清空这一页（非保留原文块）的译文重新翻译；翻译设置（引擎 / 术语表）用当前设置。"""
        self._check_alive()
        cur = self._settings_snapshot()
        with self._cv:
            job = self._editable(job_id)
            i = self._check_index(job, index)
            p = job.pages[i]
            failed_early = p.stage == "error" and job.meta[i].get("failed_stage") in ("read", "detect", "ocr", "load")
            if failed_early:
                return self._reset_to_pending(job, i)
            if p.stage in ("pending", "detected"):
                self._touch_page(job, i)          # 还没识别：照常走流程，优先处理
                return self._page_dict(job, i)
            auto = set(job.meta[i].get("auto_skip") or [])
            for b in p.blocks:
                if b.id in auto:
                    b.skip = False
                if not b.skip:
                    b.translation = ""
            job.meta[i]["auto_skip"] = []
            job.meta[i].pop("failed_stage", None)
            p.stage, p.error = "recognized", ""
            job.settings = self._refreshed(job.settings, cur, "translate", keep=("target", "sfx_mode", "translate_free_text"))
            self._save_meta_safe(job)
            self._touch_page(job, i)
            return self._page_dict(job, i)

    def redetect_page(self, job_id: str, index: int) -> dict:
        """丢掉这一页的全部结果，从检测开始重做。"""
        self._check_alive()
        with self._cv:
            job = self._editable(job_id)
            i = self._check_index(job, index)
            d = self._reset_to_pending(job, i)
        with job.io_lock:
            if not job.removed:
                remove_quietly(job.clean_file(i))
                remove_quietly(job.view_file(i))
        return d

    def rerender_all(self, job_id: str) -> None:
        """按当前排版设置（字体 / 字号等）把整本重新嵌字；已完成的书排完会自动重新导出。"""
        self._check_alive()
        cur = self._settings_snapshot()
        with self._cv:
            job = self._editable(job_id)
            old_label = layout_label(job.settings)
            current = copy.deepcopy(job.settings)
            current["source"]["reading_order"] = cur["source"]["reading_order"]
            job.settings = self._refreshed(current, cur, "typeset")
            if layout_label(job.settings) != old_label:
                job.out_path = ""
            for i, p in enumerate(job.pages):
                if p.stage in ("rendered", "cleaned"):
                    p.stage = "cleaned"
                    job.gen[i] += 1
                    self._save_page_safe(job, i)
                    self._mark_page(job, i)
            self._clean_cache.drop(lambda k: k[0] == job.id)
            job.hint = 0
            if job.status in ("done", "error", "canceled"):
                job.status, job.pause_reason, job.retry_at, job.message = "queued", "", None, "重新排版中…"
                self._started = True
            self._save_meta_safe(job)
            self._mark_job(job)
            self._ensure_workers()
            self._cv.notify_all()

    def export(self, job_id: str) -> str:
        """导出已处理完的书（同步，返回输出路径）；出错的页用原图。"""
        self._check_alive()
        with self._cv:
            job = self._job(job_id)
            if not job.persistent:
                raise HanhuaError(job.message or "这个任务没有可以导出的内容")
            if job.purged:
                raise HanhuaError("这本书的中间文件已按设置清理，成品就在输出位置：" + (job.out_path or "（未知）"))
            if job.exporting:
                raise HanhuaError("正在导出，请稍等")
            pending = sum(p.stage not in ("rendered", "error") for p in job.pages)
            if pending:
                raise HanhuaError(f"还有 {pending} 页尚未完成汉化，请等待处理完成后再导出。")
            if not any(p.stage == "rendered" for p in job.pages):
                raise HanhuaError("还没有处理好的页面，暂时不能导出")
            job.exporting = True
            prev_msg = job.message
            settings = self._comp_settings(job)
        try:
            path = self._export_job(job, settings)
        except _JobFailure as e:
            raise HanhuaError(str(e)) from None
        except HanhuaError:
            raise
        except Exception as e:  # noqa: BLE001
            log.warning("《%s》导出失败", job.title, exc_info=True)
            raise HanhuaError(f"导出失败：{_err_text(e)}") from e
        finally:
            with self._cv:
                job.exporting = False
                if job.message.startswith("正在导出"):
                    job.message = prev_msg
                self._cv.notify_all()
        with self._cv:
            finished = job.advance_hint() >= len(job.pages) and not job.inflight and not job.manual
            if job.status in ACTIVE and finished:
                job.status, job.pause_reason = "done", ""
            if job.status == "done":
                job.message = self._done_message(job)
            self._save_meta_safe(job)
            self._mark_job(job)
            self._log_event("info", f"《{job.title}》已导出 → {path}")
        return path

    def page_image_path(self, job_id: str, index: int, kind: str) -> Optional[str]:
        """页面图片文件路径（界面取图用）：src 原图（压缩包里的页缓存到 src_cache/）、clean 去字图、out 成品、thumb 缩略图。"""
        with self._lock:
            job = self._job(job_id)
            i = self._check_index(job, index)
            if not job.persistent:
                return None
        k = str(kind or "").lower()
        if k == "thumb":
            p = job.thumb_file(i)
            return p if os.path.isfile(p) else None
        if k == "out":
            p = job.out_file(i)
            return p if os.path.isfile(p) else None
        if k == "src":
            return self._src_file(job, i)
        if k == "clean":
            return self._view_file(job, i)
        raise HanhuaError("图片类型只能是 src / clean / out / thumb")

    def shutdown(self) -> None:
        """2 秒内优雅停止：通知所有线程收工，没提交的半成品直接丢弃（磁盘上只有完整阶段），保存任务状态。

        进行中的任务在 job.json 里保持 running / queued：下次启动按 general.auto_resume 自动续跑或显示为暂停。"""
        t0 = time.monotonic()
        with self._cv:
            if self._stopping:
                return
            self._stopping = True
            jobs = list(self._jobs.values())
            for job in jobs:
                if job.tr_cancel is not None:
                    job.tr_cancel.set()
                self._save_meta_safe(job)
            self._cv.notify_all()
        for t in list(self._threads):
            t.join(timeout=max(0.0, t0 + 1.3 - time.monotonic()))
        alive = [t.name for t in self._threads if t.is_alive()]
        if alive:
            log.warning("%d 个工作线程没在时限内结束（它们的结果会被丢弃）：%s", len(alive), ", ".join(alive))

        def _close() -> None:
            for j in jobs:
                self._release_book(j)
            _dispose(self._factory)

        closer = threading.Thread(target=_close, name="hanhua-engine-close", daemon=True)
        closer.start()
        closer.join(timeout=max(0.05, t0 + 1.7 - time.monotonic()))
        self._src_cache.clear()
        self._clean_cache.clear()
        with self._lock:
            self._closed = True
        self._wake.set()
        self._ticker.join(timeout=max(0.05, t0 + SHUTDOWN_BUDGET_S - time.monotonic()))
        log.info("引擎已停止（用时 %.2f 秒）", time.monotonic() - t0)

    # ================================================================== 任务创建 / 恢复
    @staticmethod
    def _normalize_paths(paths: Any) -> list[str]:
        if isinstance(paths, (str, bytes, os.PathLike)):
            paths = [paths]
        try:
            raw = list(paths or [])
        except TypeError:
            return []
        out: list[str] = []
        for p in raw:
            try:
                s = os.fsdecode(os.fspath(p))
            except TypeError:
                continue
            s = s.strip().strip('"').strip()
            if s:
                a = os.path.abspath(s)
                if a not in out:
                    out.append(a)
        return out

    def _scan(self, items: list[str]) -> tuple[list[str], list[tuple[str, str]]]:
        try:
            return [str(b) for b in self._factory.scan_inputs(items, self._inputs_root) if b], []
        except Exception as e:  # noqa: BLE001
            log.warning("扫描输入失败，改为逐个扫描：%s", e)
        books: list[str] = []
        failures: list[tuple[str, str]] = []
        for p in items:
            try:
                books.extend(str(b) for b in self._factory.scan_inputs([p], self._inputs_root) if b)
            except Exception as e:  # noqa: BLE001
                failures.append((p, _err_text(e)))
        return books, failures

    def _add_book(self, path: str, jid: str) -> Job:
        workdir = os.path.join(self.work_root, jid)
        existed = os.path.isdir(workdir)
        try:
            book = self._factory.open_book(path, os.path.join(workdir, "tmp"))
        except Exception:
            if not existed:
                _rmtree(workdir)
            raise
        try:
            names = [str(n) for n in (getattr(book, "pages", None) or [])]
            if not names:
                raise UnsupportedFormat("这本书里没有找到图片")
            job = Job(
                id=jid, source=path, title=str(getattr(book, "title", "") or _title_of(path)),
                kind=str(getattr(book, "kind", "") or ""), workdir=workdir,
                settings=self._settings_snapshot(), pages=[PageData(index=i, name=n) for i, n in enumerate(names)],
                status="queued", message="排队中",
            )
            os.makedirs(job.path("pages"), exist_ok=True)
            job.save_meta()
        except Exception:
            _dispose(book)
            if not existed:
                _rmtree(workdir)
            raise
        job.book = book
        with self._cv:
            cur = self._jobs.get(jid)
            if cur is None or not cur.persistent:
                self._insert(job)
                self._log_event("info", f"已添加《{job.title}》（{len(job.pages)} 页）")
                self._cv.notify_all()
                return job
        job.book = None                       # 并发重复添加：用先到的那个
        _dispose(book)
        with self._cv:
            self._revive(cur)
        return cur

    def _error_job(self, path: str, msg: str) -> Job:
        """打不开的输入也列出来（只在内存里），让客户知道为什么。"""
        jid = make_job_id(path)
        with self._cv:
            old = self._jobs.get(jid)
            if old is not None and old.persistent:
                self._log_event("error", f"{os.path.basename(path)}：{msg}")
                return old
            job = Job(id=jid, source=path, title=_title_of(path), kind="", workdir=os.path.join(self.work_root, jid),
                      settings=complete_settings({}), pages=[], status="error", message=msg, persistent=False)
            self._insert(job)
            self._log_event("error", f"《{job.title}》{msg}")
            return job

    def _insert(self, job: Job) -> None:
        """持锁调用。同 id 的旧对象（打不开时留下的错误任务）原位替换，保持队列顺序。"""
        if job.id not in self._jobs:
            self._order.append(job.id)
        self._jobs[job.id] = job
        self._ev_gone.pop(job.id, None)
        self._mark_job(job)

    def _revive(self, job: Job) -> None:
        """持锁调用：暂停 / 取消 / 出错 → 排队（已完成的保持不变）。"""
        if job.status in ("paused", "canceled", "error"):
            job.status, job.pause_reason, job.retry_at, job.retry_count = "queued", "", None, 0
            job.message = "排队中"
            self._save_meta_safe(job)
        self._mark_job(job)
        self._cv.notify_all()

    def _restore(self, auto_resume: bool) -> None:
        """启动时扫描 work_root：未完成的任务恢复为暂停（auto_resume 时直接排队续跑）。"""
        try:
            names = sorted(os.listdir(self.work_root))
        except OSError as e:
            log.warning("读取工作目录失败：%s", e)
            return
        loaded: list[Job] = []
        ids: set[str] = set()
        for n in names:
            d = os.path.join(self.work_root, n)
            if n.startswith(("_", ".")) or not os.path.isfile(os.path.join(d, JOB_FILE)):
                continue
            try:
                job = Job.load(d)
                if job is None:
                    log.warning("任务目录 %s 的 job.json 损坏，已跳过", d)
                    continue
                if job.removed or job.id in ids:
                    continue
                self._prepare_restored(job, auto_resume)
            except Exception:  # noqa: BLE001 - 一个任务坏了不影响其它任务恢复
                log.exception("恢复任务 %s 失败，已跳过", d)
                continue
            ids.add(job.id)
            loaded.append(job)
        loaded.sort(key=lambda j: j.created)
        with self._cv:
            for job in loaded:
                self._insert(job)
                if job.status in ACTIVE:
                    self._started = True
        if loaded:
            log.info("恢复了 %d 个任务", len(loaded))

    def _prepare_restored(self, job: Job, auto_resume: bool) -> None:
        job.clean_temp_files()
        was_active = job.status in ACTIVE or (job.status == "paused" and job.pause_reason == "translate")
        if job.status != "done" and not job.purged:
            for i, p in enumerate(job.pages):          # 文件丢了的阶段退回去重做
                if p.stage == "rendered" and not os.path.isfile(job.out_file(i)):
                    p.stage = "cleaned"
                if p.stage == "cleaned" and not os.path.isfile(job.clean_file(i)):
                    p.stage = "translated"
                    job.meta[i].pop("clean_sig", None)
        elif job.status == "done" and not job.purged:
            for i, p in enumerate(job.pages):          # 人工修改做到一半就退出了：静默补完
                if p.stage not in FINAL_STAGES:
                    job.manual.add(i)
        if was_active:
            if auto_resume:
                job.status, job.message, job.pause_reason = "queued", "继续上次没做完的部分", ""
            else:
                job.status, job.message, job.pause_reason = "paused", "上次没做完，点「继续」接着处理", "user"
        job.retry_at, job.retry_count = None, 0
        job.hint = 0
        job.advance_hint()
        try:
            job.save_meta()
        except OSError as e:
            log.warning("保存任务 %s 失败：%s", job.id, e)

    # ================================================================== 内部：通用
    def _check_alive(self) -> None:
        if self._stopping:
            raise HanhuaError("程序正在退出，操作没有执行")

    def _job(self, job_id: Any) -> Job:
        job = self._jobs.get(str(job_id)) if job_id is not None else None
        if job is None:
            raise HanhuaError("找不到这个任务（可能已经移除）")
        return job

    def _editable(self, job_id: Any) -> Job:
        job = self._job(job_id)
        if not job.persistent:
            raise HanhuaError(job.message or "这个任务打不开，无法编辑")
        if job.purged:
            raise HanhuaError("这本书的中间文件已按设置在导出后清理，不能再编辑；需要修改请移除后重新处理")
        return job

    @staticmethod
    def _check_index(job: Job, index: Any) -> int:
        try:
            i = int(index)
        except (TypeError, ValueError):
            raise HanhuaError("页码格式不对") from None
        if not 0 <= i < len(job.pages):
            raise HanhuaError(f"页码超出范围（这本共 {len(job.pages)} 页）")
        return i

    def _settings_snapshot(self) -> dict:
        try:
            return complete_settings(self.store.get())
        except Exception:  # noqa: BLE001
            log.warning("读取设置失败，改用默认设置", exc_info=True)
            return complete_settings({})

    def _comp_settings(self, job: Job) -> dict:
        """组件用的设置 = 任务快照 + 实时硬件设置（换显卡 / 档位立即生效；语言等跟着任务走）。持锁调用。"""
        s = dict(job.settings)
        s["hardware"] = dict(self._hw)
        return s

    @staticmethod
    def _refreshed(old: dict, cur: dict, section: str, keep: Sequence[str] = ()) -> dict:
        """任务快照里的某一段换成当前设置（keep 里的项保留任务原值）。返回新 dict（快照只整体替换，不原地改）。"""
        new = copy.deepcopy(old)
        sec = cur.get(section)
        if not isinstance(sec, dict):
            return new
        sec = copy.deepcopy(sec)
        prev = new.get(section) if isinstance(new.get(section), dict) else {}
        for k in keep:
            if k in prev:
                sec[k] = prev[k]
        new[section] = sec
        return complete_settings(new)

    def _on_settings(self, snap: dict) -> None:
        if self._closed or not isinstance(snap, dict):
            return
        hw = snap.get("hardware")
        if not isinstance(hw, dict):
            return
        with self._cv:
            if hw != self._hw:
                self._hw = dict(hw)
                self._render_key = None
                self._log_event("info", "硬件 / 画质档位设置已更新，接下来处理的页面生效")
            self._cv.notify_all()

    def _page_dict(self, job: Job, i: int) -> dict:
        p = job.pages[i]
        d = p.to_dict()
        m = job.meta[i]
        d["job_id"] = job.id
        d["busy"] = i in job.inflight or i in job.manual
        d["failed_stage"] = str(m.get("failed_stage") or "") if p.stage == "error" else ""
        d["auto_skip"] = list(m.get("auto_skip") or [])
        return d

    def _touch_page(self, job: Job, i: int) -> None:
        """持锁调用：人工改了第 i 页 → 代号 +1（在途的旧结果作废）、优先处理、落盘、通知。"""
        job.gen[i] += 1
        job.manual.add(i)
        job.hint = min(job.hint, i)
        if job.status == "done" or job.exporting:
            job.dirty = True
            self._save_meta_safe(job)
        self._clean_cache.drop(lambda k: k[0] == job.id and k[1] == i)
        self._save_page_safe(job, i)
        self._mark_page(job, i)
        self._ensure_workers()
        self._cv.notify_all()

    def _reset_to_pending(self, job: Job, i: int) -> dict:
        p = job.pages[i]
        p.blocks, p.bubbles, p.stage, p.error = [], [], "pending", ""
        job.meta[i] = {}
        self._touch_page(job, i)
        return self._page_dict(job, i)

    def _save_meta_safe(self, job: Job) -> bool:
        if not job.persistent or job.removed:
            return True
        try:
            job.save_meta()
            return True
        except OSError as e:
            log.error("保存任务 %s 失败：%s", job.id, e)
            self._log_event("error", f"《{job.title}》保存进度失败：{e}")
            return False

    def _save_page_safe(self, job: Job, i: int) -> bool:
        try:
            job.save_page(i)
            return True
        except OSError as e:
            log.error("保存第 %d 页进度失败：%s", i + 1, e)
            return False

    def _set_status(self, job: Job, status: str, message: Optional[str] = None) -> None:
        job.status = status
        if message is not None:
            job.message = message
        if status == "running":
            job.run_started = time.time()
        self._save_meta_safe(job)
        self._mark_job(job)

    @staticmethod
    def _done_message(job: Job) -> str:
        failed = sum(1 for p in job.pages if p.stage == "error")
        return f"已完成（{failed} 页出错，已保留原图，可点「重试失败页」）" if failed else "已完成"

    def _drop_caches(self, job: Job) -> None:
        self._src_cache.drop(lambda k: k[0] == job.id)
        self._clean_cache.drop(lambda k: k[0] == job.id)

    # ================================================================== 内部：书本 / 文件
    def _open_book_locked(self, job: Job) -> Any:
        """持 job.book_lock 调用：懒打开源文件，并核对页名和任务一致。"""
        if job.book is not None:
            return job.book
        if job.removed or not job.persistent:
            raise _JobFailure("任务已移除")
        try:
            book = self._factory.open_book(job.source, job.path("tmp"))
        except Exception as e:  # noqa: BLE001
            raise _JobFailure(f"打不开源文件：{_err_text(e)}") from e
        got = [str(n) for n in (getattr(book, "pages", None) or [])]
        if got != [p.name for p in job.pages]:
            _dispose(book)
            raise _JobFailure("源文件和上次处理时不一样了（页数或页名变了），请移除这个任务后重新拖入")
        job.book = book
        return book

    def _read_src(self, job: Job, i: int, raw_bytes: bool = False) -> Any:
        name = job.pages[i].name
        with job.book_lock:
            book = self._open_book_locked(job)
            return book.read_page_bytes(name) if raw_bytes else book.read_page(name)

    def _src_image(self, job: Job, i: int) -> np.ndarray:
        img = self._src_cache.get((job.id, i))
        if img is not None:
            return img
        return as_bgr(self._read_src(job, i))

    def _release_book(self, job: Job) -> None:
        with job.book_lock:
            book, job.book = job.book, None
            if book is not None:
                _dispose(book)

    def _write_files(self, job: Job, items: list[tuple[str, bytes]]) -> bool:
        """原子写一组文件；任务已移除 / 引擎在退出就不写（返回 False）。"""
        with job.io_lock:
            if job.removed or self._stopping:
                return False
            for path, data in items:
                write_atomic(path, data)
        return True

    def _src_file(self, job: Job, i: int) -> Optional[str]:
        path = job.src_cache_file(i)
        if os.path.isfile(path):
            return path
        try:
            data = self._read_src(job, i, raw_bytes=True)
        except Exception as e:  # noqa: BLE001
            log.warning("读取《%s》第 %d 页原图失败：%s", job.title, i + 1, e)
            return None
        if not data:
            return None
        try:
            return path if self._write_files(job, [(path, bytes(data))]) else None
        except OSError as e:
            log.warning("缓存原图失败：%s", e)
            return None

    def _view_file(self, job: Job, i: int) -> Optional[str]:
        """去字图预览 = 原图 + 去字补丁（JPEG，补丁更新后自动重新生成）。"""
        with self._lock:
            has = bool(job.meta[i].get("clean_sig"))
        patch_path, view = job.clean_file(i), job.view_file(i)
        if not has or not os.path.isfile(patch_path):
            return None
        try:
            if os.path.isfile(view) and os.stat(view).st_mtime_ns >= os.stat(patch_path).st_mtime_ns:
                return view
        except OSError:
            pass
        try:
            src = self._src_image(job, i)
            with open(patch_path, "rb") as f:
                patch = decode_patch(f.read())
            if patch is None or patch.shape[:2] != src.shape[:2]:
                return None
            data = hio.encode(apply_patch(src, patch), "jpg", 90)
            return view if self._write_files(job, [(view, data)]) else None
        except Exception as e:  # noqa: BLE001
            log.warning("生成《%s》第 %d 页去字预览失败：%s", job.title, i + 1, e)
            return None

    def _original_bytes(self, job: Job, i: int, ext: str) -> Optional[bytes]:
        """没有任何改动的页：扩展名一致就直接用原文件字节（不二次压缩，画质零损失）。"""
        own = os.path.splitext(job.pages[i].name)[1].lower().lstrip(".")
        own = {"jpeg": "jpg", "jfif": "jpg"}.get(own, own)
        if own != ext or ext not in _MAGIC:
            return None
        try:
            data = bytes(self._read_src(job, i, raw_bytes=True) or b"")
        except Exception:  # noqa: BLE001
            return None
        if not data.startswith(_MAGIC[ext]) or (ext == "webp" and data[8:12] != b"WEBP"):
            return None
        return data

    def _purge(self, job: Job) -> None:
        """keep_workdir=False：导出成功后删掉大文件（去字补丁 / 成品 / 缓存），只留 json 和缩略图。"""
        self._release_book(job)
        with job.io_lock:
            if job.removed:
                return
            for sub in ("clean", "out", "src_cache", "view", "tmp"):
                _rmtree(job.path(sub))
        with self._cv:
            if not job.removed:
                job.purged = True
                self._save_meta_safe(job)
                self._mark_job(job)

    # ================================================================== 内部：组件
    def _component(self, kind: str, s: dict) -> Any:
        try:
            obj = getattr(self._factory, kind)(s)
        except Exception as e:  # noqa: BLE001
            log.warning("%s组件加载失败：%s", COMP_CN.get(kind, kind), e, exc_info=not isinstance(e, HanhuaError))
            raise _JobFailure(f"{COMP_CN.get(kind, kind)}组件加载失败：{_err_text(e)}") from e
        if obj is None:
            raise _JobFailure(f"{COMP_CN.get(kind, kind)}组件不可用")
        return obj

    def _update_render_limit(self, s: dict) -> None:
        key = json.dumps(s.get("hardware") or {}, sort_keys=True, default=str)
        if key == self._render_key:
            return
        try:
            n = int(self._factory.cpu_workers(s))
        except Exception:  # noqa: BLE001
            n = self._render_limit
        with self._cv:
            self._render_key = key
            self._render_limit = max(1, min(MAX_RENDER_THREADS, n))
            self._cv.notify_all()

    # ================================================================== 内部：调度
    def _ensure_workers(self) -> None:
        """持锁调用：第一次需要时启动全部工作线程（之后常驻，空闲时睡在条件变量上）。"""
        if self._workers_started or self._stopping:
            return
        self._workers_started = True
        specs = [("vision", 0), ("translate", 0), ("clean", 0), ("export", 0)]
        specs += [("render", k) for k in range(MAX_RENDER_THREADS)]
        for kind, slot in specs:
            t = threading.Thread(target=self._worker, args=(kind, slot), name=f"hanhua-{kind}-{slot}", daemon=True)
            self._threads.append(t)
            t.start()

    def _worker(self, kind: str, slot: int) -> None:
        pick = getattr(self, f"_pick_{kind}")
        run = getattr(self, f"_run_{kind}")
        while True:
            task: Optional[_Task] = None
            with self._cv:
                while task is None and not self._stopping:
                    try:
                        res = pick(slot)
                    except Exception:  # noqa: BLE001
                        log.exception("调度出错（%s）", kind)
                        res = None
                    if isinstance(res, _Task):
                        task = res
                        break
                    wait = float(res) if isinstance(res, (int, float)) and not isinstance(res, bool) else 0.5
                    self._cv.wait(timeout=max(0.01, min(0.5, wait)))
                if task is None:
                    return
            try:
                run(task)
            except Exception as e:  # noqa: BLE001 - 兜底：程序 bug 也不能让线程死掉或同一页无限重试
                log.exception("%s 线程出错", kind)
                self._internal_error(task, e)
            finally:
                with self._cv:
                    for i in task.indices:
                        task.job.inflight.discard(i)
                    if task.kind == "export":
                        task.job.exporting = False
                    self._cv.notify_all()

    def _take(self, kind: str, job: Job, idxs: list[int], manual: bool = False) -> _Task:
        for i in idxs:
            job.inflight.add(i)
        return _Task(kind, job, list(idxs), [job.gen[i] for i in idxs], self._comp_settings(job), manual)

    def _window(self) -> list[Job]:
        """持锁调用：参与流水线的书（队列顺序，最多 max_open_jobs 本还有页没做完的）。顺带维护 running / queued 状态。"""
        if not self._started:
            return []
        win: list[Job] = []
        for jid in self._order:
            job = self._jobs.get(jid)
            if job is None or job.removed or not job.persistent or job.status not in ACTIVE:
                continue
            if job.advance_hint() >= len(job.pages):
                continue
            if len(win) < self._max_open:
                if job.status == "queued":
                    self._set_status(job, "running", "处理中")
                    self._log_event("info", f"开始处理《{job.title}》")
                win.append(job)
            elif job.status == "running" and not job.inflight:
                self._set_status(job, "queued", "排队中")
        return win

    def _manual_jobs(self) -> list[Job]:
        out: list[Job] = []
        for jid in self._order:
            job = self._jobs.get(jid)
            if job is not None and job.manual and job.persistent and not job.removed and job.status != "canceled":
                out.append(job)
        return out

    def _first_manual(self, stages: tuple[str, ...], limit: int = 1) -> Optional[tuple[Job, list[int]]]:
        now = time.time()
        for job in self._manual_jobs():
            if stages == ("recognized",) and job.retry_at and now < job.retry_at:
                continue
            idxs = [i for i in sorted(job.manual) if i not in job.inflight and job.pages[i].stage in stages][:limit]
            if idxs:
                return job, idxs
        return None

    @staticmethod
    def _first_in(win: list[Job], stages: tuple[str, ...]) -> Optional[tuple[Job, int]]:
        for job in win:
            for i in range(job.hint, len(job.pages)):
                if job.pages[i].stage in stages and i not in job.inflight:
                    return job, i
        return None

    @staticmethod
    def _count_in(win: list[Job], stages: tuple[str, ...]) -> int:
        return sum(1 for job in win for i in range(job.hint, len(job.pages)) if job.pages[i].stage in stages)

    def _pick_vision(self, slot: int) -> Any:
        m = self._first_manual(("pending", "detected"))
        if m:
            return self._take("vision", m[0], m[1], True)
        win = self._window()
        if not win or self._count_in(win, ("recognized", "translated", "cleaned")) >= self._max_ahead:
            return None
        hit = self._first_in(win, ("pending", "detected"))
        return self._take("vision", hit[0], [hit[1]]) if hit else None

    def _pick_translate(self, slot: int) -> Any:
        m = self._first_manual(("recognized",), TRANSLATE_BATCH_PAGES)
        if m:
            return self._take("translate", m[0], m[1], True)
        now = time.time()
        for job in self._window():
            if job.retry_at and now < job.retry_at:
                continue
            idxs: list[int] = []
            for i in range(job.hint, len(job.pages)):
                if job.pages[i].stage == "recognized" and i not in job.inflight:
                    idxs.append(i)
                    if len(idxs) >= TRANSLATE_BATCH_PAGES:
                        break
            if not idxs:
                continue
            if len(idxs) < TRANSLATE_BATCH_PAGES and self._batch_wait > 0 and self._vision_pending(job):
                first = self._tr_wait.setdefault(job.id, time.monotonic())
                remain = self._batch_wait - (time.monotonic() - first)
                if remain > 0:
                    return remain          # 再等等后面的页，凑成一批（更少请求、更多上下文）
            self._tr_wait.pop(job.id, None)
            return self._take("translate", job, idxs)
        return None

    @staticmethod
    def _vision_pending(job: Job) -> bool:
        return any(job.pages[i].stage in ("pending", "detected") for i in range(job.hint, len(job.pages)))

    def _pick_clean(self, slot: int) -> Any:
        m = self._first_manual(("translated",))
        if m:
            return self._take("clean", m[0], m[1], True)
        win = self._window()
        if not win or self._count_in(win, ("cleaned",)) >= 2 * self._render_limit + 2:
            return None
        hit = self._first_in(win, ("translated",))
        return self._take("clean", hit[0], [hit[1]]) if hit else None

    def _pick_render(self, slot: int) -> Any:
        if slot >= self._render_limit:
            return None
        m = self._first_manual(("cleaned",))
        if m:
            return self._take("render", m[0], m[1], True)
        hit = self._first_in(self._window(), ("cleaned",))
        return self._take("render", hit[0], [hit[1]]) if hit else None

    def _pick_export(self, slot: int) -> Any:
        if not self._started:
            return None
        for jid in self._order:
            job = self._jobs.get(jid)
            if job is None or job.removed or not job.persistent or job.status not in ACTIVE:
                continue
            if job.exporting or job.inflight or job.manual or job.advance_hint() < len(job.pages):
                continue
            job.exporting = True
            if job.status == "queued":
                self._set_status(job, "running", "处理中")
            return _Task("export", job, [], [], self._comp_settings(job))
        return None

    # ================================================================== 内部：提交 / 失败
    def _commit(self, job: Job, i: int, gen: int, stage: str, *, blocks: Optional[list[TextBlock]] = None,
                bubbles: Optional[list[Box]] = None, size: Optional[tuple[int, int]] = None,
                timing: Optional[tuple[str, float]] = None, meta: Optional[dict] = None,
                rendered: bool = False) -> bool:
        """锁内提交一个阶段的结果并原子落盘；页面已被人工改过（gen 变了）/ 任务移除 / 引擎退出 → 丢弃。"""
        with self._cv:
            if self._stopping or job.removed or job.gen[i] != gen or self._jobs.get(job.id) is not job:
                return False
            p = job.pages[i]
            if size:
                p.width, p.height = int(size[0]), int(size[1])
            if blocks is not None:
                p.blocks = blocks
            if bubbles is not None:
                p.bubbles = bubbles
            if timing:
                p.timings[timing[0]] = round(float(timing[1]), 4)
            if meta:
                job.meta[i].update(meta)
            job.meta[i].pop("failed_stage", None)
            p.stage, p.error = stage, ""
            if rendered:
                p.version += 1
                job.meta[i].pop("reclean", None)
                job.record_done(time.time())
                if job.status == "done":
                    job.dirty = True
                    self._save_meta_safe(job)
            if stage in FINAL_STAGES:
                job.manual.discard(i)
            ok = self._save_page_safe(job, i)
            self._mark_page(job, i)
            self._cv.notify_all()
            if not ok:
                self._fail_job(job, "保存进度失败（磁盘满了或没有写入权限）")
            return ok

    def _page_failed(self, job: Job, i: int, gen: int, step: str, exc: BaseException) -> None:
        """单页出错：标 error 继续处理别的页（导出时这一页用原图）。"""
        if self._stopping:
            return
        detail = _err_text(exc)
        msg = detail if isinstance(exc, _JobFailure) else f"{STEP_CN.get(step, step)}失败：{detail}"
        if isinstance(exc, HanhuaError):
            log.warning("《%s》第 %d 页%s", job.title, i + 1, msg)
        else:
            log.warning("《%s》第 %d 页%s", job.title, i + 1, msg, exc_info=exc)
        with self._cv:
            if job.removed or job.gen[i] != gen or self._jobs.get(job.id) is not job:
                return
            p = job.pages[i]
            m = job.meta[i]
            m["failed_stage"] = step
            if p.stage != "error":
                m["last_stage"] = p.stage
            p.stage, p.error = "error", msg[:500]
            job.manual.discard(i)
            if job.status == "done":
                job.dirty = True
                self._save_meta_safe(job)
            self._save_page_safe(job, i)
            self._mark_page(job, i)
            self._log_event("warning", f"《{job.title}》第 {i + 1} 页{msg}")
            self._cv.notify_all()

    def _fail_job(self, job: Job, msg: str) -> None:
        """整本书做不下去：任务标 error（页面进度保留，「继续」/「重试」可接着做）。"""
        with self._cv:
            if self._stopping or job.removed:
                return
            if job.status in ("done", "canceled"):
                self._log_event("error", f"《{job.title}》{msg}")
                return
            job.status, job.message, job.pause_reason, job.retry_at = "error", msg, "", None
            job.manual.clear()
            if job.tr_cancel is not None:
                job.tr_cancel.set()
            self._save_meta_safe(job)
            self._mark_job(job)
            self._log_event("error", f"《{job.title}》{msg}")
            self._cv.notify_all()

    def _job_failure(self, t: _Task, exc: _JobFailure) -> None:
        if t.manual or t.job.status not in ACTIVE:
            for k, i in enumerate(t.indices):      # 人工请求失败只标这一页，免得任务被反复拉起
                self._page_failed(t.job, i, t.gens[k], "load", exc)
        else:
            self._fail_job(t.job, str(exc))

    def _internal_error(self, t: _Task, exc: BaseException) -> None:
        if t.kind == "export":
            self._fail_job(t.job, f"导出出错：{_err_text(exc)}")
            return
        for k, i in enumerate(t.indices):
            self._page_failed(t.job, i, t.gens[k], "internal", exc)

    # ================================================================== 内部：各阶段
    @staticmethod
    def _parse_detect(res: Any, w: int, h: int) -> tuple[list[TextBlock], list[Box]]:
        if isinstance(res, tuple) and len(res) == 2:
            raw_blocks, raw_bubbles = res
        elif isinstance(res, list):
            raw_blocks, raw_bubbles = res, []
        else:
            raise HanhuaError("检测结果格式不对")
        blocks = sanitize_blocks(list(raw_blocks or []), w, h)
        bubbles: list[Box] = []
        for b in list(raw_bubbles or []):
            box = _box(b, w, h)
            if box is not None:
                bubbles.append(box)
        return blocks, bubbles

    def _run_vision(self, t: _Task) -> None:
        job, i, gen, s = t.job, t.indices[0], t.gens[0], t.settings
        with self._lock:
            stage = job.pages[i].stage
            blocks = copy.deepcopy(job.pages[i].blocks)
        try:
            img = self._src_image(job, i)
        except _JobFailure as e:
            self._job_failure(t, e)
            return
        except Exception as e:  # noqa: BLE001
            self._page_failed(job, i, gen, "read", e)
            return
        h, w = img.shape[:2]
        if not self._announced_load:
            self._announced_load = True
            self._log_event("info", "正在加载识别模型（第一次需要几秒钟）…")
        if stage == "pending":
            try:
                det = self._component("detector", s)
            except _JobFailure as e:
                self._job_failure(t, e)
                return
            self._update_render_limit(s)
            t0 = time.perf_counter()
            try:
                blocks, bubbles = self._parse_detect(det.detect(img, reading_order(s)), w, h)
            except Cancelled:
                return
            except Exception as e:  # noqa: BLE001
                self._page_failed(job, i, gen, "detect", e)
                return
            if not self._commit(job, i, gen, "detected", blocks=blocks, bubbles=bubbles, size=(w, h),
                                timing=("detect", time.perf_counter() - t0)):
                return
            blocks = copy.deepcopy(blocks)        # 提交出去的对象归页面所有，后面用副本
        try:
            ocr = self._component("ocr", s)
        except _JobFailure as e:
            self._job_failure(t, e)
            return
        t0 = time.perf_counter()
        try:
            if blocks:
                ocr.recognize(img, blocks)
            blocks = sanitize_blocks(blocks, w, h)
            apply_text_policy(blocks, s)
        except Cancelled:
            return
        except Exception as e:  # noqa: BLE001
            self._page_failed(job, i, gen, "ocr", e)
            return
        if self._commit(job, i, gen, "recognized", blocks=blocks, size=(w, h), timing=("ocr", time.perf_counter() - t0)):
            self._src_cache.put((job.id, i), img)

    def _translate_batch(self, svc: Any, todo: list[list[TextBlock]], src_lang: str,
                         cancel: threading.Event) -> dict[int, BaseException]:
        """整批翻译；整批出了「非网络」错误就逐页重试，把坏页隔离出来。返回 {批内序号: 异常}。"""
        nonempty = [k for k, blocks in enumerate(todo) if blocks]
        try:
            svc.translate_blocks([todo[k] for k in nonempty], src_lang, cancel)
            return {}
        except (TranslationUnavailable, Cancelled):
            raise
        except Exception as e:  # noqa: BLE001
            if len(nonempty) <= 1:
                return {k: e for k in nonempty}
            log.warning("整批翻译出错，改为逐页重试：%s", e)
        failed: dict[int, BaseException] = {}
        for k in nonempty:
            if cancel.is_set():
                raise Cancelled("已取消")
            try:
                svc.translate_blocks([todo[k]], src_lang, cancel)
            except (TranslationUnavailable, Cancelled):
                raise
            except Exception as e2:  # noqa: BLE001
                failed[k] = e2
        return failed

    def _run_translate(self, t: _Task) -> None:
        job, s = t.job, t.settings
        cancel = threading.Event()
        with self._lock:
            pages = [copy.deepcopy(job.pages[i].blocks) for i in t.indices]
            job.tr_cancel = cancel
        try:
            todo = [[b for b in blocks if want_translate(b, s)] for blocks in pages]
            failed: dict[int, BaseException] = {}
            t0 = time.perf_counter()
            if any(todo):
                try:
                    svc = self._component("translator", s)
                except _JobFailure as e:
                    self._job_failure(t, e)
                    return
                src_lang = str((s.get("source") or {}).get("lang") or "ja")
                try:
                    failed = self._translate_batch(svc, todo, src_lang, cancel)
                except TranslationUnavailable as e:
                    self._translation_down(job, e)
                    return
                except Cancelled:
                    return
            dt = (time.perf_counter() - t0) / max(1, len(t.indices))
            committed = 0
            for k, i in enumerate(t.indices):
                if k in failed:
                    self._page_failed(job, i, t.gens[k], "translate", failed[k])
                    continue
                empty = {b.id for b in todo[k] if not (b.translation or "").strip()}
                meta = None
                if empty:                       # 没拿到译文的块保留原文：宁可不动，也不能擦掉不填
                    for b in todo[k]:
                        if b.id in empty:
                            b.skip = True
                    with self._lock:
                        prev = set(job.meta[i].get("auto_skip") or [])
                    meta = {"auto_skip": sorted(prev | empty)}
                    log.info("《%s》第 %d 页有 %d 个块没拿到译文，保留原文", job.title, i + 1, len(empty))
                if self._commit(job, i, t.gens[k], "translated", blocks=pages[k], timing=("translate", dt), meta=meta):
                    committed += 1
            if committed:
                self._translation_ok(job)
        finally:
            with self._lock:
                if job.tr_cancel is cancel:
                    job.tr_cancel = None

    def _translation_down(self, job: Job, exc: BaseException) -> None:
        """所有翻译引擎都挂了：任务暂停（不产出没翻译的页），按 1 / 2 / 5 分钟自动重试。"""
        with self._cv:
            if self._stopping or job.removed:
                return
            delay = self._retry_delays[min(job.retry_count, len(self._retry_delays) - 1)]
            job.retry_count += 1
            job.retry_at = time.time() + delay
            text = f"翻译服务暂时不可用（{_err_text(exc, 120)}），{_fmt_delay(delay)}后自动重试"
            if job.status in ACTIVE:
                job.status, job.pause_reason = "paused", "translate"
            job.message = text
            self._save_meta_safe(job)
            self._mark_job(job)
            self._log_event("warning", f"《{job.title}》{text}")
            self._cv.notify_all()

    def _translation_ok(self, job: Job) -> None:
        with self._cv:
            if not (job.retry_count or job.retry_at):
                return
            job.retry_count, job.retry_at = 0, None
            if job.status in ACTIVE:
                job.message = "处理中"
            elif job.status == "done":
                job.message = self._done_message(job)
            self._save_meta_safe(job)
            self._mark_job(job)

    @staticmethod
    def _parse_clean(res: Any, img: np.ndarray) -> tuple[np.ndarray, Optional[np.ndarray]]:
        mask = None
        if isinstance(res, tuple):
            cleaned = res[0] if res else None
            mask = res[1] if len(res) > 1 else None
        else:
            cleaned = res
        cleaned = as_bgr(cleaned)
        if cleaned.shape != img.shape:
            raise HanhuaError(f"去字结果尺寸 {cleaned.shape} 和原图 {img.shape} 不一致")
        if mask is not None:
            mask = np.asarray(mask)
            if mask.shape[:2] != img.shape[:2]:
                mask = None
        return cleaned, mask

    def _run_clean(self, t: _Task) -> None:
        job, i, gen, s = t.job, t.indices[0], t.gens[0], t.settings
        with self._lock:
            blocks = copy.deepcopy(job.pages[i].blocks)
            old_sig = job.meta[i].get("clean_sig")
        sig = clean_signature(blocks)
        patch_path = job.clean_file(i)
        if old_sig == sig and os.path.isfile(patch_path):
            self._commit(job, i, gen, "cleaned")         # 去字补丁仍然有效（例如只是重新翻译）
            return
        try:
            img = self._src_image(job, i)
        except _JobFailure as e:
            self._job_failure(t, e)
            return
        except Exception as e:  # noqa: BLE001
            self._page_failed(job, i, gen, "read", e)
            return
        h, w = img.shape[:2]
        t0 = time.perf_counter()
        try:
            if any(not b.skip for b in blocks):
                inp = self._component("inpainter", s)
                cleaned, mask = self._parse_clean(inp.clean(img.copy(), blocks), img)
            else:
                cleaned, mask = img, None
            data = encode_patch(make_patch(img, cleaned, mask))
        except _JobFailure as e:
            self._job_failure(t, e)
            return
        except Cancelled:
            return
        except Exception as e:  # noqa: BLE001
            self._page_failed(job, i, gen, "clean", e)
            return
        try:
            if not self._write_files(job, [(patch_path, data)]):
                return
        except OSError as e:
            self._page_failed(job, i, gen, "save", e)
            return
        if self._commit(job, i, gen, "cleaned", meta={"clean_sig": sig}, size=(w, h),
                        timing=("clean", time.perf_counter() - t0)):
            self._clean_cache.put((job.id, i, gen), cleaned)
        self._src_cache.pop((job.id, i))

    def _rebuild_clean(self, job: Job, i: int) -> np.ndarray:
        """去字图不在缓存里：原图 + 去字补丁重建。补丁缺失 / 损坏 → _NeedReclean。"""
        src = self._src_image(job, i)
        try:
            with open(job.clean_file(i), "rb") as f:
                patch = decode_patch(f.read())
        except OSError:
            raise _NeedReclean() from None
        if patch is None or patch.shape[:2] != src.shape[:2]:
            raise _NeedReclean()
        return apply_patch(src, patch)

    def _send_back_to_clean(self, job: Job, i: int, gen: int) -> None:
        with self._cv:
            if self._stopping or job.removed or job.gen[i] != gen:
                return
            n = int(job.meta[i].get("reclean", 0) or 0) + 1
            if n > MAX_RECLEAN:
                bad = True
            else:
                bad = False
                job.meta[i]["reclean"] = n
                job.meta[i].pop("clean_sig", None)
                job.pages[i].stage = "translated"
                self._save_page_safe(job, i)
                self._mark_page(job, i)
                self._cv.notify_all()
        if bad:
            self._page_failed(job, i, gen, "render", HanhuaError("去字结果反复读不出来（磁盘可能有问题）"))

    def _run_render(self, t: _Task) -> None:
        job, i, gen, s = t.job, t.indices[0], t.gens[0], t.settings
        with self._lock:
            p = job.pages[i]
            blocks = copy.deepcopy(p.blocks)
            sig_ok = job.meta[i].get("clean_sig") == clean_signature(p.blocks)
            ext, out_path, thumb_path = job.out_ext(i), job.out_file(i), job.thumb_file(i)
        cleaned = self._clean_cache.pop((job.id, i, gen))
        if cleaned is None:
            if not sig_ok:
                self._send_back_to_clean(job, i, gen)
                return
            try:
                cleaned = self._rebuild_clean(job, i)
            except _NeedReclean:
                self._send_back_to_clean(job, i, gen)
                return
            except _JobFailure as e:
                self._job_failure(t, e)
                return
            except Exception as e:  # noqa: BLE001
                self._page_failed(job, i, gen, "read", e)
                return
        t0 = time.perf_counter()
        try:
            untouched = all(b.skip for b in blocks)
            if untouched:
                out, data = cleaned, self._original_bytes(job, i, ext)
            else:
                ts = self._component("typesetter", s)
                shape = cleaned.shape
                out = as_bgr(ts.render(cleaned, blocks))
                if out.shape != shape:
                    raise HanhuaError(f"排版结果尺寸 {out.shape} 和原图 {shape} 不一致")
                blocks = sanitize_blocks(blocks, out.shape[1], out.shape[0])
                data = None
            if data is None:
                data = hio.encode(out, ext, encode_quality(ext, s))
            thumb = hio.encode(hio.thumbnail(out, THUMB_SIDE), "jpg", 85)
        except _JobFailure as e:
            self._job_failure(t, e)
            return
        except Cancelled:
            return
        except Exception as e:  # noqa: BLE001
            self._page_failed(job, i, gen, "render", e)
            return
        try:
            if not self._write_files(job, [(out_path, data), (thumb_path, thumb)]):
                return
        except OSError as e:
            self._page_failed(job, i, gen, "save", e)
            return
        self._commit(job, i, gen, "rendered", blocks=blocks, timing=("render", time.perf_counter() - t0), rendered=True)
        self._src_cache.pop((job.id, i))

    def _run_export(self, t: _Task) -> None:
        job = t.job
        with self._lock:
            total = len(job.pages)
            errors = [p for p in job.pages if p.stage == "error"]
        if total == 0:
            self._fail_job(job, "这本书没有页面")
            return
        if len(errors) == total:
            self._fail_job(job, f"所有页面都处理失败（例如第 {errors[0].index + 1} 页：{errors[0].error}）")
            return
        try:
            path = self._export_job(job, t.settings)
        except _JobFailure as e:
            self._fail_job(job, str(e))
            return
        except Exception as e:  # noqa: BLE001
            log.warning("《%s》导出失败：%s", job.title, e, exc_info=not isinstance(e, HanhuaError))
            self._fail_job(job, f"导出失败：{_err_text(e)}")
            return
        purge = False
        with self._cv:
            if job.removed or self._stopping:
                return
            if job.status in ACTIVE or job.status == "paused":
                job.status, job.pause_reason, job.retry_at, job.retry_count = "done", "", None, 0
                purge = not bool((job.settings.get("output") or {}).get("keep_workdir", True)) and not job.dirty
            if job.status == "done":
                job.message = self._done_message(job)
            self._save_meta_safe(job)
            self._mark_job(job)
            self._log_event("info", f"《{job.title}》已完成 → {path}")
        if purge:
            self._purge(job)
        self._release_book(job)

    def _export_job(self, job: Job, settings: dict) -> str:
        """出错 / 没渲染的页用原图（缓存到 src_cache/），按设置写出成品。返回输出路径。"""
        with self._lock:
            gens = list(job.gen)
            plan = [(i, p.name, p.stage) for i, p in enumerate(job.pages)]
            out_path = job.out_path
        pages: list[tuple[str, str]] = []
        skipped = 0
        for i, name, stage in plan:
            out = job.out_file(i)
            if stage != "error" and os.path.isfile(out):
                pages.append((name, out))
                continue
            src = self._src_file(job, i)
            if src:
                pages.append((name, src))
            else:
                skipped += 1
        if not pages:
            raise HanhuaError("没有可以导出的页面")
        with self._output_lock, job.book_lock:
            book = self._open_book_locked(job)
            if not out_path:
                try:
                    named = copy.deepcopy(settings)
                    named["output"]["suffix"] = str(named["output"].get("suffix", "_汉化")) + "_" + layout_label(settings)
                    out_path = str(self._factory.default_output_path(book, named))
                except Exception as e:  # noqa: BLE001
                    raise HanhuaError(f"无法确定输出位置：{_err_text(e)}") from e
            with job.export_lock:
                result = self._factory.export_book(book, pages, out_path, settings,
                                                   lambda d, n: self._export_progress(job, d, n))
        result = str(result or out_path)
        with self._cv:
            if not job.removed:
                job.out_path = result
                job.dirty = job.gen != gens or bool(job.manual)
                self._save_meta_safe(job)
                self._mark_job(job)
        if skipped:
            self._log_event("warning", f"《{job.title}》有 {skipped} 页连原图都读不出来，导出时已跳过")
        return result

    def _export_progress(self, job: Job, done: Any, total: Any) -> None:
        try:
            with self._lock:
                if not job.removed:
                    job.message = f"正在导出 {int(done)}/{int(total)}"
                    self._mark_job(job)
        except Exception:  # noqa: BLE001 - 进度回调绝不能打断导出
            pass

    # ================================================================== 内部：事件 / 定时器
    def _mark_job(self, job: Job) -> None:
        self._ev_jobs.add(job.id)

    def _mark_page(self, job: Job, i: int) -> None:
        p = job.pages[i]
        self._ev_pages[(job.id, i)] = {"type": "page", "job_id": job.id, "index": i, "stage": p.stage, "version": p.version}
        self._ev_jobs.add(job.id)

    def _log_event(self, level: str, message: str) -> None:
        lv = level if level in ("debug", "info", "warning", "error") else "info"
        if lv == "info":
            log.info("%s", message)
        with self._lock:
            self._ev_logs.append({"type": "log", "level": lv, "message": str(message)})
            if len(self._ev_logs) > 200:
                del self._ev_logs[:-200]

    def _engine_running(self) -> bool:
        return any(j.status == "running" or j.inflight or j.manual or j.exporting for j in self._jobs.values())

    def _check_retries(self, now: float) -> None:
        for job in self._jobs.values():
            if job.retry_at is None or now < job.retry_at:
                continue
            job.retry_at = None
            if job.status == "paused" and job.pause_reason == "translate":
                job.status, job.pause_reason, job.message = "queued", "", "正在重新尝试翻译…"
                self._save_meta_safe(job)
                self._mark_job(job)
                self._log_event("info", f"《{job.title}》重新尝试翻译")
            self._cv.notify_all()

    def _tick_loop(self) -> None:
        while True:
            closing = self._closed
            try:
                self._tick_once()
            except Exception:  # noqa: BLE001
                log.exception("引擎节拍线程出错")
            if closing:
                return
            self._wake.wait(self._tick)

    def _tick_once(self) -> None:
        with self._cv:
            now = time.time()
            if not self._stopping:
                self._check_retries(now)
            if now - self._last_periodic >= 1.0:          # 每秒刷新一次运行中任务的速度 / 剩余时间
                self._last_periodic = now
                for job in self._jobs.values():
                    if job.status == "running":
                        self._ev_jobs.add(job.id)
            running = (not self._stopping) and self._engine_running()
        try:
            label = str(self._factory.device_label() or "")
        except Exception:  # noqa: BLE001
            label = ""
        with self._lock:
            evs: list[dict] = list(self._ev_logs)
            self._ev_logs.clear()
            evs.extend(self._ev_pages.values())
            self._ev_pages.clear()
            for jid in list(self._ev_jobs):
                job = self._jobs.get(jid)
                if job is not None:
                    evs.append({"type": "job", "job": job.summary()})
            self._ev_jobs.clear()
            evs.extend({"type": "removed", "job_id": g["id"]} for g in self._ev_gone.values())
            self._ev_gone.clear()
            state = (running, label)
            if state != self._ev_engine_last:
                self._ev_engine_last = state
                evs.append({"type": "engine", "running": running, "device": label})
        cb = self._on_event
        if cb is None:
            return
        for ev in evs:
            try:
                cb(ev)
            except Exception:  # noqa: BLE001 - 界面回调出错不能影响引擎
                self._cb_errors += 1
                if self._cb_errors <= 5:
                    log.warning("界面事件回调出错", exc_info=True)
