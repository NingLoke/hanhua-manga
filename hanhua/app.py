"""桌面程序入口与界面后端 Api。

- ``Api``：纯 Python（不依赖 pywebview），网页通过 ``POST /api/<方法>`` 调用；浏览器调试 / 测试都能直接用。
  每个公开方法都包了 ``_guard``：可预期错误抛 HanhuaError（中文），意外异常记日志并转成中文 ApiError，
  由 AppServer 包装成 ``{"ok": false, "error": "..."}``。长操作（导出 / 全书重排）放后台线程立即返回，
  完成后通过事件总线推 ``{"type": "op", ...}``。
- ``main()``：日志轮转 → SettingsStore → Engine（--demo 用假引擎）→ AppServer → pywebview 窗口。
  ``--demo --no-window`` 只起服务器并打印 URL（浏览器调界面 / 无头截图用）。
"""
from __future__ import annotations

import argparse
import collections
import copy
import csv
import functools
import io
import logging
import logging.handlers
import os
import re
import subprocess
import sys
import threading
import time
import webbrowser
from typing import Any, Callable, Optional, Protocol, Sequence

from .config import DEFAULTS, ENUMS, SettingsStore
from .runtime import DEVICE_LABELS, VERSION, build_info, data_dir, detect_hardware, fonts_dir, resource_path
from .server import AppServer, EventBus
from .types import Cancelled, HanhuaError

log = logging.getLogger("hanhua.app")

APP_TITLE = f"一键汉化漫画 {VERSION}"
WINDOW_SIZE = (1280, 820)
WINDOW_MIN = (1080, 700)
WEBVIEW2_URL = "https://developer.microsoft.com/zh-cn/microsoft-edge/webview2/"

BOOK_EXTS = (".zip", ".cbz", ".7z", ".cb7", ".rar", ".cbr", ".tar", ".cbt", ".pdf", ".epub")
PAGE_EXTS = (".jpg", ".jpeg", ".jfif", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".avif")
FONT_EXTS = (".ttf", ".otf", ".ttc", ".otc")
# pywebview 的 file_types 描述只能是文字和空格（正则 [\w ]+），不能带标点
_BOOK_FILTER = "漫画文件 (" + ";".join("*" + e for e in BOOK_EXTS + PAGE_EXTS) + ")"
_ALL_FILTER = "所有文件 (*.*)"
_FONT_FILTER = "字体文件 (" + ";".join("*" + e for e in FONT_EXTS) + ")"
_CSV_FILTER = "CSV 表格 (*.csv)"

# 安全：open_path 拒绝直接运行程序 / 脚本类文件（纵深防御；/api 本身已有 token 保护）
_DANGEROUS_EXTS = frozenset({
    ".exe", ".bat", ".cmd", ".com", ".ps1", ".psm1", ".vbs", ".vbe", ".js", ".jse", ".wsf", ".wsh", ".msi",
    ".msp", ".scr", ".lnk", ".url", ".hta", ".cpl", ".jar", ".reg", ".pif", ".application", ".appref-ms",
    ".dll", ".sys", ".inf", ".py", ".pyw",
})
MAX_PATHS = 5000          # 一次最多添加多少个路径
MAX_BLOCKS = 2000         # 一页最多多少个文字块
MAX_TEXT = 4000           # 单块原文 / 译文最长字符
MAX_GLOSSARY = 20000      # 术语表最多条数
MAX_UI_ERRORS = 60        # 每次运行最多记录多少条界面脚本错误
_JOB_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")
_ENGINE_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,32}$")

# 翻译模块未就绪时的引擎目录（名称给界面显示）
_ENGINE_FALLBACK: tuple[dict, ...] = (
    {"id": "local", "name": "本地大模型", "kind": "local", "batch": True},
    {"id": "api", "name": "API 服务", "kind": "api", "batch": True},
    {"id": "google", "name": "谷歌翻译", "kind": "free", "batch": True},
    {"id": "edge", "name": "微软翻译", "kind": "free", "batch": True},
    {"id": "tencent", "name": "腾讯翻译君", "kind": "free", "batch": True},
)
_ENGINE_NAMES = {"google": "谷歌翻译", "edge": "微软翻译", "tencent": "腾讯翻译君", "llm": "自装大模型（Ollama）",
                 "local": "本地大模型", "api": "API 服务"}
_GGUF_FILTER = "GGUF 模型 (*.gguf)"
_URL_RE = re.compile(r"^https?://[^\s/?#]+[^\s]*$", re.I)
LLM_EVENT_INTERVAL = 0.25   # 下载进度事件最快 4 次 / 秒
# 字体模块未就绪时的内置字体（id 与合同 5.4 一致）
_FONT_FALLBACK: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("noto-sans-sc-medium", "思源黑体 中", ("builtin", "sans", "sc", "dialog")),
    ("noto-sans-sc-bold", "思源黑体 粗", ("builtin", "sans", "sc", "dialog")),
    ("noto-sans-sc-black", "思源黑体 特粗", ("builtin", "sans", "sc", "free")),
    ("noto-sans-tc-medium", "思源黑体 繁 中", ("builtin", "sans", "tc", "dialog")),
    ("noto-sans-tc-bold", "思源黑体 繁 粗", ("builtin", "sans", "tc", "dialog")),
    ("noto-sans-tc-black", "思源黑体 繁 特粗", ("builtin", "sans", "tc", "free")),
    ("lxgw-wenkai", "霞鹜文楷", ("builtin", "kai", "sc", "tc")),
    ("zcool-kuaile", "站酷快乐体", ("builtin", "display", "sc", "sfx")),
    ("ma-shan-zheng", "马善政毛笔", ("builtin", "brush", "sc", "sfx")),
)
VARIANT_LABELS = {"nvidia": "NVIDIA 显卡版", "gpu": "通用显卡版", "cpu": "CPU 版", "dev": "开发版（源码运行）"}
_DEVICE_SHORT = {"cuda": "CUDA", "dml": "DirectML", "cpu": "CPU"}


# --------------------------------------------------------------------------------------
# 错误处理
# --------------------------------------------------------------------------------------
class ApiError(HanhuaError):
    """Api 层给客户看的中文错误（message 直接显示在界面上）。"""


def brief_error(e: BaseException) -> str:
    """把意外异常压成一句给客户看的中文（细节进日志）。"""
    if isinstance(e, FileNotFoundError):
        return "找不到文件或文件夹" + (f"：{e.filename}" if getattr(e, "filename", None) else "")
    if isinstance(e, PermissionError):
        return "没有访问权限（文件可能正被其它程序占用，或需要管理员权限）"
    if isinstance(e, MemoryError):
        return "内存不足，请关闭其它程序后重试"
    if isinstance(e, TimeoutError):
        return "操作超时，请稍后重试"
    if isinstance(e, OSError) and getattr(e, "errno", None) == 28:
        return "磁盘空间不足"
    msg = str(e).strip().splitlines()[0] if str(e).strip() else ""
    return msg[:160] if msg else type(e).__name__


def _guard(action: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Api 方法装饰器：HanhuaError 原样抛出；其它异常记日志 → ApiError(中文)。保留原签名供参数校验。"""

    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(self: "Api", *args: Any, **kwargs: Any) -> Any:
            try:
                return fn(self, *args, **kwargs)
            except HanhuaError:
                raise
            except Exception as e:  # noqa: BLE001
                log.exception("%s出错", action)
                raise ApiError(f"{action}失败：{brief_error(e)}") from e

        return wrapper

    return deco


# --------------------------------------------------------------------------------------
# 文件对话框钩子
# --------------------------------------------------------------------------------------
class Dialogs(Protocol):
    def open_files(self, file_types: Sequence[str], multiple: bool = True) -> list[str]: ...
    def open_folder(self) -> Optional[str]: ...
    def save_file(self, filename: str, file_types: Sequence[str]) -> Optional[str]: ...


class WebviewDialogs:
    """用 pywebview 窗口实现的系统文件对话框（可在任意线程调用）。"""

    def __init__(self, window: Any) -> None:
        self.window = window

    @staticmethod
    def _flat(res: Any) -> list[str]:
        if not res:
            return []
        if isinstance(res, str):
            return [res]
        return [str(p) for p in res if p]

    def open_files(self, file_types: Sequence[str], multiple: bool = True) -> list[str]:
        import webview

        return self._flat(self.window.create_file_dialog(
            webview.FileDialog.OPEN, allow_multiple=multiple, file_types=tuple(file_types)))

    def open_folder(self) -> Optional[str]:
        import webview

        res = self._flat(self.window.create_file_dialog(webview.FileDialog.FOLDER))
        return res[0] if res else None

    def save_file(self, filename: str, file_types: Sequence[str]) -> Optional[str]:
        import webview

        res = self._flat(self.window.create_file_dialog(
            webview.FileDialog.SAVE, save_filename=filename, file_types=tuple(file_types)))
        return res[0] if res else None


def system_open(path: str, reveal: bool = False) -> None:
    """用系统默认程序打开；reveal=True 时在资源管理器里选中它。"""
    path = os.path.normpath(path)
    if sys.platform == "win32":
        if reveal and os.path.isfile(path):
            subprocess.Popen(["explorer", "/select,", path])  # noqa: S603,S607 - 固定程序 + 列表参数
        else:
            os.startfile(path if not reveal else (path if os.path.isdir(path) else os.path.dirname(path)))  # type: ignore[attr-defined]  # noqa: S606
        return
    target = path if not reveal or os.path.isdir(path) else os.path.dirname(path)
    opener = "open" if sys.platform == "darwin" else "xdg-open"
    subprocess.Popen([opener, target])  # noqa: S603


# --------------------------------------------------------------------------------------
# 引擎不可用时的替身：界面照常打开，操作时给出中文原因
# --------------------------------------------------------------------------------------
class UnavailableEngine:
    label = "不可用"
    running = False

    def __init__(self, reason: str) -> None:
        self.reason = reason or "处理引擎加载失败"

    def _fail(self, *_a: Any, **_k: Any) -> Any:
        raise HanhuaError(f"{self.reason}（请重新安装软件，或把日志发给客服）")

    add_paths = start = pause = resume = cancel = remove = retry_failed = _fail
    get_page = update_page = retranslate_page = redetect_page = rerender_all = export = _fail

    def list_jobs(self) -> list[dict]:
        return []

    def page_image_path(self, *_a: Any) -> None:
        return None

    def shutdown(self) -> None:
        return None


# --------------------------------------------------------------------------------------
# 术语表 CSV
# --------------------------------------------------------------------------------------
_HEAD_SRC = {"原文", "src", "source", "原词", "日文", "韩文", "英文", "term", "原名"}
_HEAD_DST = {"译文", "dst", "target", "translation", "中文", "译名", "译词"}
_HEAD_NOTE = {"备注", "note", "notes", "说明", "comment"}


def clean_glossary(items: Any) -> list[dict]:
    """规范化术语表：只保留 src 非空的条目，去首尾空白，同一原文只留最后一条。"""
    if not isinstance(items, list):
        raise ApiError("术语表数据格式不对")
    out: dict[str, dict] = {}
    for it in items[:MAX_GLOSSARY]:
        if not isinstance(it, dict):
            continue
        src = str(it.get("src") or "").strip()[:200]
        if not src:
            continue
        out.pop(src, None)
        out[src] = {"src": src, "dst": str(it.get("dst") or "").strip()[:200], "note": str(it.get("note") or "").strip()[:500]}
    return list(out.values())


def parse_glossary_csv(text: str) -> list[dict]:
    """CSV / TSV 文本 → 术语条目。自动识别分隔符和表头（原文 / 译文 / 备注），无表头时按前三列。"""
    text = (text or "").lstrip("﻿")
    if not text.strip():
        return []
    try:
        dialect: Any = csv.Sniffer().sniff(text[:4096], delimiters=",\t;")
    except csv.Error:
        dialect = csv.excel
    rows = [[c.strip() for c in r] for r in csv.reader(io.StringIO(text), dialect)]
    rows = [r for r in rows if any(r)]
    if not rows:
        return []
    si, di, ni = 0, 1, 2
    head = [c.lower() for c in rows[0]]
    if any(c in _HEAD_SRC or c in _HEAD_DST for c in head):
        def col(names: set[str], default: int) -> int:
            return next((i for i, c in enumerate(head) if c in names), default)
        si, di, ni = col(_HEAD_SRC, 0), col(_HEAD_DST, 1), col(_HEAD_NOTE, -1)
        rows = rows[1:]
    items = []
    for r in rows:
        src = r[si] if 0 <= si < len(r) else ""
        dst = r[di] if 0 <= di < len(r) else ""
        note = r[ni] if 0 <= ni < len(r) else ""
        if src and dst:
            items.append({"src": src, "dst": dst, "note": note})
    return clean_glossary(items)


def dump_glossary_csv(items: list[dict]) -> str:
    """术语表 → 带 BOM 的 CSV 文本（Excel 双击打开不乱码）。"""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\r\n")
    w.writerow(["原文", "译文", "备注"])
    for it in items:
        w.writerow([it.get("src", ""), it.get("dst", ""), it.get("note", "")])
    return "﻿" + buf.getvalue()


# --------------------------------------------------------------------------------------
# 硬件 → 标签 / 推荐
# --------------------------------------------------------------------------------------
def short_gpu_name(name: str) -> str:
    """'NVIDIA GeForce RTX 5060 Ti' → 'RTX 5060 Ti'；'Intel(R) Arc(TM) A770' → 'Arc A770'。"""
    n = re.sub(r"\((?:R|TM|C)\)|[®™]", "", name or "", flags=re.I)
    n = re.sub(r"\s+", " ", n).strip()
    for p in ("NVIDIA GeForce ", "NVIDIA ", "GeForce ", "AMD ", "Intel "):
        if n.lower().startswith(p.lower()):
            n = n[len(p):]
            break
    return n.strip() or (name or "显卡")


def resolve_device(setting: str, hw: Any) -> str:
    providers = set(getattr(hw, "providers", []) or [])
    gpus = list(getattr(hw, "gpus", []) or [])
    cuda_ok = "CUDAExecutionProvider" in providers and any(g.vendor == "nvidia" for g in gpus)
    dml_ok = "DmlExecutionProvider" in providers and bool(getattr(hw, "best_gpu", None))
    if setting == "cpu":
        return "cpu"
    if setting == "cuda" and cuda_ok:
        return "cuda"
    if setting == "dml" and dml_ok:
        return "dml"
    return "cuda" if cuda_ok else ("dml" if dml_ok else "cpu")


def hardware_chip(device: str, hw: Any) -> str:
    """顶栏硬件标签，例如 'RTX 5060 Ti · CUDA'。"""
    gpus = list(getattr(hw, "gpus", []) or [])
    if device == "cuda":
        g = next((g for g in sorted(gpus, key=lambda g: -g.vram_mb) if g.vendor == "nvidia"), None)
        return f"{short_gpu_name(g.name) if g else 'NVIDIA'} · CUDA"
    if device == "dml":
        g = getattr(hw, "best_gpu", None)
        return f"{short_gpu_name(g.name) if g else '显卡'} · DirectML"
    return f"CPU · {int(getattr(hw, 'cpu_threads', 1) or 1)} 线程"


def recommend(hw: Any, variant: str) -> dict:
    """按硬件推荐发行版和画质档位。"""
    g = getattr(hw, "best_gpu", None)
    vram = int(getattr(g, "vram_mb", 0) or 0)
    if g is not None and g.vendor == "nvidia":
        rec, why = "nvidia", f"检测到 NVIDIA 显卡（{short_gpu_name(g.name)}），NVIDIA 版使用 CUDA 加速，速度最快"
    elif g is not None and g.vendor in ("amd", "intel"):
        rec, why = "gpu", f"检测到 {short_gpu_name(g.name)}，通用显卡版使用 DirectML 加速，比纯 CPU 快很多"
    else:
        rec, why = "cpu", "没有检测到可加速的显卡，CPU 版体积最小、兼容性最好"
    if rec == "cpu":
        profile = "fast" if int(getattr(hw, "cpu_threads", 1) or 1) < 8 else "balanced"
    else:
        profile = "quality" if vram >= 8000 else ("balanced" if vram >= 4000 else "fast")
    current_ok = variant in ("dev", rec) or (variant == "gpu" and rec == "nvidia")
    return {
        "variant": rec, "variant_label": VARIANT_LABELS.get(rec, rec), "reason": why,
        "profile": profile, "current_ok": current_ok,
    }


# --------------------------------------------------------------------------------------
# 本地大模型（真实实现，包一层 hanhua.llm；演示引擎用 ui/demo.py 的 DemoModels 顶替，接口相同）
# --------------------------------------------------------------------------------------
def check_model_id(model_id: Any) -> str:
    """模型 id：内置清单 id，或 custom:<文件名.gguf>（拦路径穿越 / 控制字符）。"""
    mid = str(model_id or "").strip()
    if mid.startswith("custom:"):
        fn = mid[len("custom:"):]
        if (not fn or len(fn) > 240 or fn != os.path.basename(fn) or "/" in fn or "\\" in fn or fn.startswith(".")
                or not fn.lower().endswith(".gguf") or any(ord(c) < 32 for c in fn)):
            raise ApiError("模型名称不对")
        return mid
    if not re.fullmatch(r"[A-Za-z0-9._\-]{1,64}", mid):
        raise ApiError("模型名称不对")
    return mid


class LocalModels:
    """Api 用到的本地模型操作（都可能抛中文 HanhuaError）。"""

    def status(self) -> dict:
        from .llm import available_backends, current_recommendation, list_models

        rec = current_recommendation()
        return {"hardware": self.hardware(), "recommend": rec.to_dict() if rec else None,
                "backends": list(available_backends()), "models": list_models(rec)}

    @staticmethod
    def hardware() -> dict:
        hw = detect_hardware()
        g = hw.best_gpu
        return {
            "gpu": short_gpu_name(g.name) if g else "", "gpu_full": g.name if g else "", "vendor": g.vendor if g else "",
            "vram_gb": round(g.vram_mb / 1024, 1) if g else 0.0, "ram_gb": round(float(hw.ram_gb or 0.0), 1),
            "cpu": hw.cpu_name, "cpu_threads": int(hw.cpu_threads or 1),
        }

    @staticmethod
    def info(model_id: str) -> Optional[dict]:
        from .llm import BY_ID, list_models

        if model_id not in BY_ID:
            return None
        return next((m for m in list_models(None) if m.get("id") == model_id), None)

    @staticmethod
    def download(model_id: str, progress: Callable[[int, int, float], None], cancel: threading.Event) -> str:
        from .llm import BY_ID, download

        m = BY_ID.get(model_id)
        if m is None:
            raise ApiError("没有这个模型，请刷新后重试")
        return download(m, progress, cancel)

    @staticmethod
    def delete(model_id: str) -> None:
        from .llm import BY_ID, delete_model
        from .llm.manager import model_path

        delete_model(model_id)
        m = BY_ID.get(model_id)
        if m is not None:                       # 没下完的 .part 也一起清掉
            try:
                os.remove(model_path(m) + ".part")
            except OSError:
                pass

    @staticmethod
    def import_file(path: str) -> dict:
        from .llm import import_gguf

        return import_gguf(path)

    @staticmethod
    def presets() -> list[dict]:
        from .llm.engine import API_PRESETS

        return [{"id": k, **v} for k, v in API_PRESETS.items()]

    @staticmethod
    def api_models(base_url: str, api_key: str) -> list[str]:
        from .llm.engine import list_api_models

        return list_api_models(base_url, api_key)


# --------------------------------------------------------------------------------------
# Api
# --------------------------------------------------------------------------------------
class Api:
    """界面调用的全部后端方法。线程安全（HTTP 服务器在多个线程里并发调用）。"""

    def __init__(
        self,
        store: SettingsStore,
        engine: Any,
        bus: Optional[EventBus] = None,
        *,
        demo: bool = False,
        dialogs: Optional[Dialogs] = None,
        engine_error: str = "",
        opener: Optional[Callable[[str, bool], None]] = None,
        translator_factory: Optional[Callable[[dict], Any]] = None,
        fonts: Any = None,
        local_models: Any = None,
    ) -> None:
        self.store = store
        self.engine = engine
        self.bus = bus if bus is not None else EventBus()
        self.demo = bool(demo)
        self.dialogs = dialogs
        self.engine_error = engine_error
        self._opener = opener or system_open
        self._translator_factory = translator_factory
        self._fonts = fonts
        self._lock = threading.RLock()
        self._ops: dict[str, threading.Thread] = {}
        self._preview_cache: collections.OrderedDict[tuple[str, str], bytes] = collections.OrderedDict()
        self._ui_errors = 0
        self._engine_state: dict[str, Any] = {}
        # 本地模型：None = 首次用到时取 engine.local_models（演示）或真实 LocalModels
        self._local_models = local_models
        self._dl: dict[str, Any] = {}                      # 当前 / 最近一次下载的状态（llm 事件同结构）
        self._dl_cancel: Optional[threading.Event] = None
        self._dl_thread: Optional[threading.Thread] = None
        self.bus.add_listener(self._watch_events)

    # ---- 内部工具 --------------------------------------------------------------------
    def _watch_events(self, ev: dict) -> None:
        if ev.get("type") == "engine":
            self._engine_state = {"running": bool(ev.get("running")), "device": str(ev.get("device") or "")}

    def _publish(self, ev: dict) -> None:
        try:
            self.bus.publish(ev)
        except Exception:  # noqa: BLE001
            log.debug("事件发布失败", exc_info=True)

    @staticmethod
    def _job_id(job_id: Any) -> str:
        if not isinstance(job_id, str) or not _JOB_ID_RE.match(job_id):
            raise ApiError("任务编号不对，请刷新后重试")
        return job_id

    @staticmethod
    def _index(index: Any) -> int:
        if isinstance(index, bool) or not isinstance(index, (int, float, str)):
            raise ApiError("页码不对")
        try:
            i = int(index)
        except (TypeError, ValueError):
            raise ApiError("页码不对") from None
        if i < 0 or i > 10_000_000:
            raise ApiError("页码不对")
        return i

    def _need_dialogs(self) -> Dialogs:
        if self.dialogs is None:
            raise ApiError("当前在浏览器调试模式下运行，无法打开系统文件对话框；请把文件直接拖进窗口，或在桌面程序里操作")
        return self.dialogs

    def _settings(self) -> dict:
        try:
            return self.store.get()
        except Exception:  # noqa: BLE001
            log.exception("读取设置失败")
            return copy.deepcopy(DEFAULTS)

    def _engine_running(self) -> bool:
        if "running" in self._engine_state:
            return bool(self._engine_state["running"])
        for attr in ("running", "_running"):
            v = getattr(self.engine, attr, None)
            if isinstance(v, bool):
                return v
        return False

    def _spawn(self, op: str, job_id: str, fn: Callable[[], Any], action: str, ok_text: str) -> dict:
        """后台执行长操作：立即返回；开始 / 完成 / 出错都推 op 事件。同一任务同一操作不重复进行。"""
        key = f"{op}:{job_id}"
        with self._lock:
            t = self._ops.get(key)
            if t is not None and t.is_alive():
                raise ApiError(f"正在{action}，请稍候")
            t = threading.Thread(target=self._run_op, args=(op, job_id, fn, action, ok_text, key),
                                 name=f"hanhua-op-{op}", daemon=True)
            self._ops[key] = t
        self._publish({"type": "op", "op": op, "job_id": job_id, "state": "start", "message": f"正在{action}…"})
        t.start()
        return {"started": True, "op": op, "job_id": job_id}

    def _run_op(self, op: str, job_id: str, fn: Callable[[], Any], action: str, ok_text: str, key: str) -> None:
        ev: dict[str, Any] = {"type": "op", "op": op, "job_id": job_id}
        try:
            result = fn()
            ev.update(state="done", message=ok_text)
            if isinstance(result, str) and result:
                ev["path"] = result
        except HanhuaError as e:
            ev.update(state="error", message=str(e) or f"{action}失败")
        except Exception as e:  # noqa: BLE001
            log.exception("%s出错", action)
            ev.update(state="error", message=f"{action}失败：{brief_error(e)}")
        finally:
            with self._lock:
                self._ops.pop(key, None)
        self._publish(ev)

    def wait_ops(self, timeout: float = 10.0) -> bool:
        """等后台操作全部结束（测试 / 退出前用）。"""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            with self._lock:
                alive = [t for t in self._ops.values() if t.is_alive()]
            if not alive:
                return True
            alive[0].join(max(0.01, min(0.2, end - time.monotonic())))
        return False

    # ---- 启动数据 --------------------------------------------------------------------
    @_guard("加载界面数据")
    def bootstrap(self) -> dict:
        info = build_info()
        variant = str(info.get("variant", "dev"))
        return {
            "app": {
                "title": APP_TITLE, "version": str(info.get("version", VERSION)), "variant": variant,
                "variant_label": VARIANT_LABELS.get(variant, variant), "demo": self.demo,
                "window": self.dialogs is not None, "platform": sys.platform,
            },
            "settings": self._settings(),
            "defaults": copy.deepcopy(DEFAULTS),
            "enums": {f"{s}.{k}": list(v) for (s, k), v in ENUMS.items()},
            "jobs": self.list_jobs(),
            "engine": {"running": self._engine_running(), "error": self.engine_error,
                       "label": str(self._engine_state.get("device") or getattr(self.engine, "label", "") or "")},
            "hardware": self._hardware_brief(),
            "engines": self.list_engines(),
            "fonts": self.list_fonts(),
            "formats": {"books": [e[1:] for e in BOOK_EXTS], "pages": [e[1:] for e in PAGE_EXTS]},
            "paths": {"data": data_dir(), "logs": os.path.join(data_dir(), "logs")},
        }

    def _hardware_brief(self) -> dict:
        try:
            hw = detect_hardware()
            device = resolve_device(self._settings()["hardware"]["device"], hw)
            return {"device": device, "chip": hardware_chip(device, hw), "device_label": DEVICE_LABELS.get(device, device)}
        except Exception:  # noqa: BLE001
            log.warning("硬件识别失败", exc_info=True)
            return {"device": "cpu", "chip": "CPU", "device_label": "CPU"}

    @_guard("读取硬件信息")
    def hardware(self) -> dict:
        hw = detect_hardware()
        info = build_info()
        variant = str(info.get("variant", "dev"))
        device = resolve_device(self._settings()["hardware"]["device"], hw)
        d = hw.to_dict()
        d.update(
            device=device, device_label=DEVICE_LABELS.get(device, device), chip=hardware_chip(device, hw),
            variant=variant, variant_label=VARIANT_LABELS.get(variant, variant), version=str(info.get("version", VERSION)),
            recommend=recommend(hw, variant), engine_label=str(self._engine_state.get("device") or getattr(self.engine, "label", "") or ""),
            python=sys.version.split()[0],
        )
        return d

    @_guard("检查连接")
    def ping(self) -> dict:
        return {"ts": time.time()}

    # ---- 任务 ------------------------------------------------------------------------
    @_guard("读取任务列表")
    def list_jobs(self) -> list[dict]:
        jobs = self.engine.list_jobs()
        return [j for j in (jobs or []) if isinstance(j, dict)]

    @_guard("读取页面列表")
    def list_pages(self, job_id: str) -> list[dict]:
        jid = self._job_id(job_id)
        fn = getattr(self.engine, "list_pages", None)
        if callable(fn):
            return list(fn(jid))
        get_job = getattr(self.engine, "get_job", None)
        if callable(get_job):
            job = get_job(jid)
            pages = getattr(job, "pages", None)
            if pages is not None:
                return [{"index": p.index, "name": p.name, "stage": p.stage, "version": p.version, "error": p.error}
                        for p in pages]
        # 退路：按 summary 的总页数逐页读（慢，但对任何 Engine 都成立）
        total = next((int(j.get("total") or 0) for j in self.list_jobs() if j.get("id") == jid), None)
        if total is None:
            raise ApiError("找不到这个任务（可能已被删除）")
        out = []
        for i in range(min(total, 20000)):
            try:
                d = self.engine.get_page(jid, i)
            except Exception:  # noqa: BLE001
                d = {}
            out.append({"index": i, "name": d.get("name", ""), "stage": d.get("stage", "pending"),
                        "version": d.get("version", 0), "error": d.get("error", "")})
        return out

    @_guard("添加文件")
    def add_paths(self, paths: list, autostart: bool = True) -> dict:
        if isinstance(paths, str):
            paths = [paths]
        if not isinstance(paths, list):
            raise ApiError("文件列表格式不对")
        clean: list[str] = []
        skipped: list[dict] = []
        seen: set[str] = set()
        for raw in paths[:MAX_PATHS]:
            if not isinstance(raw, str) or not raw.strip() or "\x00" in raw:
                continue
            p = raw.strip().strip('"')
            key = os.path.normcase(os.path.abspath(p)) if os.path.isabs(p) else p
            if key in seen:
                continue
            seen.add(key)
            if not self.demo and not os.path.exists(p):
                skipped.append({"path": p, "reason": "文件不存在或无法访问"})
                continue
            clean.append(p)
        if len(paths) > MAX_PATHS:
            skipped.append({"path": "", "reason": f"一次最多添加 {MAX_PATHS} 个，其余已忽略"})
        jobs = list(self.engine.add_paths(clean) or []) if clean else []
        if jobs and autostart:
            try:
                self.engine.start()
            except HanhuaError:
                raise
            except Exception:  # noqa: BLE001
                log.exception("自动开始失败")
        return {"jobs": [j for j in jobs if isinstance(j, dict)], "skipped": skipped}

    @_guard("选择文件")
    def pick_files(self, add: bool = True) -> dict:
        paths = self._need_dialogs().open_files((_BOOK_FILTER, _ALL_FILTER), True)
        if not paths:
            return {"paths": [], "jobs": [], "skipped": []}
        res = self.add_paths(paths) if add else {"jobs": [], "skipped": []}
        return {"paths": paths, **res}

    @_guard("选择文件夹")
    def pick_folder(self, add: bool = True) -> dict:
        p = self._need_dialogs().open_folder()
        if not p:
            return {"paths": [], "jobs": [], "skipped": []}
        res = self.add_paths([p]) if add else {"jobs": [], "skipped": []}
        return {"paths": [p], **res}

    @_guard("选择输出目录")
    def pick_output_dir(self, field: str = "dir") -> dict:
        if field not in ("dir", "work_dir"):
            raise ApiError("不能设置这个目录")
        p = self._need_dialogs().open_folder()
        if not p:
            return {"path": "", "settings": self._settings()}
        snap = self.store.update({"output": {field: p}})
        self._publish({"type": "settings", "settings": snap})
        return {"path": p, "settings": snap}

    @_guard("开始")
    def start(self) -> dict:
        self.engine.start()
        return {"running": True}

    @_guard("暂停")
    def pause(self, job_id: Optional[str] = None) -> dict:
        if job_id:
            self.engine.pause(self._job_id(job_id))
        else:
            self.engine.pause()
        return {"paused": True}

    @_guard("继续")
    def resume(self, job_id: Optional[str] = None) -> dict:
        if job_id:
            self.engine.resume(self._job_id(job_id))
        else:
            self.engine.resume()
        return {"resumed": True}

    @_guard("取消任务")
    def cancel(self, job_id: str) -> dict:
        self.engine.cancel(self._job_id(job_id))
        return {"canceled": True}

    @_guard("删除任务")
    def remove(self, job_id: str, delete_files: bool = True) -> dict:
        jid = self._job_id(job_id)
        self.engine.remove(jid, delete_files=bool(delete_files))
        self._publish({"type": "removed", "job_id": jid})
        return {"removed": jid}

    @_guard("重试失败页")
    def retry(self, job_id: str) -> dict:
        self.engine.retry_failed(self._job_id(job_id))
        return {"retrying": True}

    # ---- 页面 / 校对 -----------------------------------------------------------------
    @_guard("读取页面")
    def get_page(self, job_id: str, index: int) -> dict:
        return self.engine.get_page(self._job_id(job_id), self._index(index))

    @staticmethod
    def _clean_blocks(blocks: Any) -> list[dict]:
        """校验编辑器提交的块：类型 / 范围不对的字段改成 None（自动），绝不让脏数据进引擎。"""
        if not isinstance(blocks, list):
            raise ApiError("文字块数据格式不对")
        if len(blocks) > MAX_BLOCKS:
            raise ApiError("这一页的文字块太多了")

        def box(v: Any) -> Optional[list[int]]:
            if isinstance(v, (list, tuple)) and len(v) == 4 and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v):
                return [int(x) for x in v]
            return None

        def color(v: Any) -> Optional[list[int]]:
            if isinstance(v, str) and re.fullmatch(r"#?[0-9a-fA-F]{6}", v.strip()):
                h = v.strip().lstrip("#")
                return [int(h[i:i + 2], 16) for i in (0, 2, 4)]
            if isinstance(v, (list, tuple)) and len(v) == 3 and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v):
                return [max(0, min(255, int(x))) for x in v]
            return None

        out: list[dict] = []
        for i, b in enumerate(blocks):
            if not isinstance(b, dict):
                continue
            d = dict(b)
            bx = box(d.get("box"))
            if bx is None:
                raise ApiError(f"第 {i + 1} 个文字块缺少位置信息")
            d["box"] = bx
            d["id"] = int(d["id"]) if isinstance(d.get("id"), int) and not isinstance(d.get("id"), bool) else i
            for k in ("bubble_box", "render_box"):
                if k in d:
                    d[k] = box(d[k])
            for k in ("src_text", "translation"):
                d[k] = str(d.get(k) or "")[:MAX_TEXT]
            for k in ("text_color", "stroke_color", "fill", "stroke"):
                if k in d:
                    d[k] = color(d[k])
            d["skip"] = bool(d.get("skip", False))
            d["is_sfx"] = bool(d.get("is_sfx", False))
            fs = d.get("font_size")
            d["font_size"] = int(fs) if isinstance(fs, (int, float)) and not isinstance(fs, bool) and 4 <= fs <= 2000 else None
            sw = d.get("stroke_width")
            d["stroke_width"] = float(sw) if isinstance(sw, (int, float)) and not isinstance(sw, bool) and 0 <= sw <= 1 else None
            if d.get("out_direction") not in ("v", "h"):
                d["out_direction"] = None
            if d.get("align") not in ("center", "start", "end"):
                d["align"] = None
            f = d.get("font")
            d["font"] = f.strip()[:200] if isinstance(f, str) and f.strip() else None
            if d.get("kind") not in ("bubble", "free"):
                d["kind"] = "bubble" if d.get("bubble_box") else "free"
            if d.get("direction") not in ("v", "h"):
                d["direction"] = "h"
            out.append(d)
        return out

    @_guard("保存修改")
    def update_page(self, job_id: str, index: int, blocks: list) -> dict:
        return self.engine.update_page(self._job_id(job_id), self._index(index), self._clean_blocks(blocks))

    @_guard("重新翻译本页")
    def retranslate_page(self, job_id: str, index: int) -> dict:
        return self.engine.retranslate_page(self._job_id(job_id), self._index(index))

    @_guard("重新检测本页")
    def redetect_page(self, job_id: str, index: int) -> dict:
        return self.engine.redetect_page(self._job_id(job_id), self._index(index))

    @_guard("重新排版全书")
    def rerender_all(self, job_id: str) -> dict:
        jid = self._job_id(job_id)
        return self._spawn("rerender", jid, lambda: self.engine.rerender_all(jid), "重新排版全书", "全书已按新设置重新排版")

    @_guard("导出")
    def export(self, job_id: str) -> dict:
        jid = self._job_id(job_id)
        return self._spawn("export", jid, lambda: self.engine.export(jid), "导出", "导出完成")

    def _image_path(self, job_id: str, index: int, kind: str) -> Optional[str]:
        """供 /img 路由使用（不是公开 API）。"""
        return self.engine.page_image_path(job_id, int(index), kind)

    # ---- 打开文件 --------------------------------------------------------------------
    @staticmethod
    def _existing(path: Any) -> str:
        if not isinstance(path, str) or not path.strip() or "\x00" in path:
            raise ApiError("路径不对")
        p = os.path.abspath(path.strip())
        if not os.path.exists(p):
            raise ApiError("文件或文件夹不存在（可能已被移动或删除）")
        return p

    @_guard("打开")
    def open_path(self, path: str) -> dict:
        p = self._existing(path)
        if os.path.isfile(p) and os.path.splitext(p)[1].lower() in _DANGEROUS_EXTS:
            raise ApiError("出于安全考虑，不能从这里打开程序或脚本文件")
        self._opener(p, False)
        return {"path": p}

    @_guard("在文件夹中显示")
    def reveal(self, path: str) -> dict:
        try:
            p = self._existing(path)
        except ApiError:
            parent = os.path.dirname(os.path.abspath(str(path or "")))
            if not path or not os.path.isdir(parent):
                raise
            p = parent
        self._opener(p, True)
        return {"path": p}

    @_guard("打开日志文件夹")
    def open_logs(self) -> dict:
        d = os.path.join(data_dir(), "logs")
        os.makedirs(d, exist_ok=True)
        self._opener(d, False)
        return {"path": d}

    @_guard("打开数据目录")
    def open_data_dir(self) -> dict:
        d = data_dir()
        self._opener(d, False)
        return {"path": d}

    # ---- 设置 ------------------------------------------------------------------------
    @_guard("读取设置")
    def get_settings(self) -> dict:
        return self._settings()

    @_guard("保存设置")
    def save_settings(self, patch: dict) -> dict:
        if not isinstance(patch, dict):
            raise ApiError("设置数据格式不对")
        if isinstance(patch.get("translate"), dict) and "glossary" in patch["translate"]:
            patch = copy.deepcopy(patch)
            patch["translate"]["glossary"] = clean_glossary(patch["translate"]["glossary"])
        snap = self.store.update(patch)
        self._publish({"type": "settings", "settings": snap})
        return snap

    @_guard("恢复默认设置")
    def reset_settings(self, section: Optional[str] = None) -> dict:
        """恢复默认（整体或某一组）；术语表是用户数据，任何情况下都保留。"""
        glossary = self._settings()["translate"].get("glossary", [])
        if section:
            if section not in DEFAULTS or not isinstance(DEFAULTS[section], dict):
                raise ApiError("没有这个设置分组")
            patch = {section: copy.deepcopy(DEFAULTS[section])}
            if section == "translate":
                patch["translate"]["glossary"] = glossary
            snap = self.store.update(patch)
        else:
            self.store.reset()
            snap = self.store.update({"translate": {"glossary": glossary}})
        self._publish({"type": "settings", "settings": snap})
        return snap

    # ---- 术语表 ----------------------------------------------------------------------
    @_guard("读取术语表")
    def get_glossary(self) -> list[dict]:
        return clean_glossary(self._settings()["translate"].get("glossary", []))

    @_guard("保存术语表")
    def save_glossary(self, items: list) -> list[dict]:
        cleaned = clean_glossary(items)
        snap = self.store.update({"translate": {"glossary": cleaned}})
        self._publish({"type": "settings", "settings": snap})
        return snap["translate"]["glossary"]

    @_guard("导入术语表")
    def import_glossary(self, text: str, replace: bool = False) -> dict:
        if not isinstance(text, str):
            raise ApiError("文件内容不对")
        new = parse_glossary_csv(text)
        if not new:
            raise ApiError("没有读到有效的词条（需要至少两列：原文、译文）")
        cur = [] if replace else self.get_glossary()
        index = {it["src"]: i for i, it in enumerate(cur)}
        added = updated = 0
        for it in new:
            if it["src"] in index:
                cur[index[it["src"]]] = it
                updated += 1
            else:
                index[it["src"]] = len(cur)
                cur.append(it)
                added += 1
        items = self.save_glossary(cur)
        return {"items": items, "added": added, "updated": updated}

    @_guard("导出术语表")
    def export_glossary(self) -> dict:
        items = self.get_glossary()
        text = dump_glossary_csv(items)
        path = ""
        if self.dialogs is not None:
            path = self.dialogs.save_file("术语表.csv", (_CSV_FILTER,)) or ""
            if path:
                if not path.lower().endswith(".csv"):
                    path += ".csv"
                tmp = path + ".tmp"
                with open(tmp, "w", encoding="utf-8", newline="") as f:
                    f.write(text)
                os.replace(tmp, path)
        return {"path": path, "csv": text, "count": len(items)}

    # ---- 翻译引擎 --------------------------------------------------------------------
    @_guard("读取翻译引擎")
    def list_engines(self) -> list[dict]:
        try:
            from .translate import TranslationService

            engines = [dict(e) for e in TranslationService.engines() if isinstance(e, dict) and e.get("id")]
        except Exception:  # noqa: BLE001
            log.debug("翻译模块未就绪，使用内置引擎目录", exc_info=True)
            engines = [dict(e) for e in _ENGINE_FALLBACK]
        for e in engines:
            e.setdefault("name", _ENGINE_NAMES.get(e["id"], e["id"]))
            e.setdefault("kind", "free")
        return engines

    @_guard("测试翻译引擎")
    def test_engine(self, engine_id: str) -> dict:
        eid = str(engine_id or "").strip()
        if not _ENGINE_ID_RE.match(eid):
            raise ApiError("引擎名称不对")
        t0 = time.perf_counter()
        fn = getattr(self.engine, "test_engine", None)
        if callable(fn):
            ok, msg = fn(eid)
        else:
            svc = self._make_translator()
            try:
                ok, msg = svc.test_engine(eid)
            finally:
                close = getattr(svc, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:  # noqa: BLE001
                        log.debug("关闭翻译服务出错", exc_info=True)
        return {"id": eid, "ok": bool(ok), "message": str(msg or ("可以使用" if ok else "不可用")),
                "ms": int((time.perf_counter() - t0) * 1000)}

    def _make_translator(self) -> Any:
        settings = self._settings()
        if self._translator_factory is not None:
            return self._translator_factory(settings)
        try:
            from .translate import TranslationService
        except Exception as e:  # noqa: BLE001
            log.warning("翻译模块加载失败", exc_info=True)
            raise ApiError("翻译模块还没有准备好，暂时无法测试") from e
        return TranslationService(settings, os.path.join(data_dir(), "translate_cache.sqlite"))

    # ---- 本地大模型 / API 服务 -------------------------------------------------------
    def _models(self) -> Any:
        with self._lock:
            if self._local_models is None:
                self._local_models = getattr(self.engine, "local_models", None) or LocalModels()
            return self._local_models

    @staticmethod
    def _active_model(sel: str, rec: Optional[dict], models: list[dict]) -> str:
        """实际会用的模型（规则同 hanhua.llm.engine.LocalLLMEngine.resolve）；空 = 还没有可用的。"""
        installed = {m.get("id"): m for m in models if m.get("installed")}
        if sel != "auto":
            return sel if sel in installed else ""
        rm = (rec or {}).get("model") or {}
        rid = str(rm.get("id") or "")
        if rid in installed:
            return rid
        limit = float(rm.get("need_gb") or 0) * 1.35 if rm else float("inf")
        ok = [m for m in installed.values() if not m.get("custom") and float(m.get("need_gb") or 0) <= limit]
        return str(max(ok, key=lambda m: int(m.get("quality") or 0)).get("id")) if ok else ""

    @_guard("读取本地模型")
    def llm_status(self) -> dict:
        """硬件摘要 + 推荐 + 可用后端 + 模型列表 + 当前选择 / 实际使用 + 下载状态。"""
        st = dict(self._models().status() or {})
        local = dict(self._settings()["translate"].get("local") or {})
        models = [dict(m) for m in st.get("models") or [] if isinstance(m, dict) and m.get("id")]
        sel = str(local.get("model") or "auto")
        with self._lock:
            dl = dict(self._dl) if self._dl else None
        st.update(models=models, local=local, selected=sel, active=self._active_model(sel, st.get("recommend"), models),
                  installed=[m["id"] for m in models if m.get("installed")], download=dl)
        return st

    @_guard("下载模型")
    def llm_download(self, model_id: str) -> dict:
        mid = check_model_id(model_id)
        info = self._models().info(mid)
        if not info:
            raise ApiError("没有这个模型，请刷新后重试")
        if info.get("installed"):
            raise ApiError(f"{info.get('name') or mid} 已经下载好了")
        with self._lock:
            if self._dl_thread is not None and self._dl_thread.is_alive():
                raise ApiError(f"正在下载 {self._dl.get('name') or '模型'}，请等它完成或先取消")
            cancel = threading.Event()
            self._dl_cancel = cancel
            self._dl = {"model_id": mid, "name": str(info.get("name") or mid), "state": "downloading",
                        "done": int(info.get("partial_bytes") or 0), "total": int(info.get("size") or 0),
                        "speed": 0.0, "message": "正在选择最快的下载源…"}
            snap = dict(self._dl)
            t = threading.Thread(target=self._run_download, args=(mid, cancel), name="hanhua-llm-download", daemon=True)
            self._dl_thread = t
        self._publish({"type": "llm", **snap})
        t.start()
        return {"started": True, "model_id": mid}

    def _run_download(self, mid: str, cancel: threading.Event) -> None:
        last = [0.0]

        def progress(done: int, total: int, speed: float) -> None:
            with self._lock:
                self._dl.update(done=int(done), total=int(total or self._dl.get("total") or 0),
                                speed=float(speed or 0.0), message="")
                snap = dict(self._dl)
            now = time.monotonic()
            if now - last[0] >= LLM_EVENT_INTERVAL:
                last[0] = now
                self._publish({"type": "llm", **snap})

        with self._lock:
            name = str(self._dl.get("name") or mid)
        try:
            self._models().download(mid, progress, cancel)
            final: dict[str, Any] = {"state": "done", "speed": 0.0, "message": f"{name} 下载完成，翻译会自动使用本地模型"}
        except Cancelled:
            final = {"state": "canceled", "speed": 0.0, "message": "已取消下载（已下载的部分会保留，下次接着下）"}
        except HanhuaError as e:
            final = {"state": "error", "speed": 0.0, "message": str(e) or "模型下载失败"}
        except Exception as e:  # noqa: BLE001
            log.exception("下载模型出错")
            final = {"state": "error", "speed": 0.0, "message": f"模型下载失败：{brief_error(e)}"}
        with self._lock:
            self._dl.update(final)
            if final["state"] == "done" and self._dl.get("total"):
                self._dl["done"] = self._dl["total"]
            snap = dict(self._dl)
        self._publish({"type": "llm", **snap})

    def wait_download(self, timeout: float = 10.0) -> bool:
        """等下载线程结束（测试 / 退出前用）。"""
        with self._lock:
            t = self._dl_thread
        if t is not None:
            t.join(timeout)
            return not t.is_alive()
        return True

    @_guard("取消下载")
    def llm_cancel(self) -> dict:
        with self._lock:
            t, ev = self._dl_thread, self._dl_cancel
            active = t is not None and t.is_alive()
            if active and ev is not None:
                ev.set()
                self._dl["message"] = "正在取消…"
        return {"canceling": active}

    @_guard("删除模型")
    def llm_delete(self, model_id: str) -> dict:
        mid = check_model_id(model_id)
        with self._lock:
            if self._dl_thread is not None and self._dl_thread.is_alive() and self._dl.get("model_id") == mid:
                raise ApiError("这个模型正在下载，请先取消下载")
            if self._dl.get("model_id") == mid:
                self._dl = {}
        self._models().delete(mid)
        if (self._settings()["translate"].get("local") or {}).get("model") == mid:
            snap = self.store.update({"translate": {"local": {"model": "auto"}}})
            self._publish({"type": "settings", "settings": snap})
        return self.llm_status()

    @_guard("导入模型")
    def llm_import(self, path: Optional[str] = None, select: bool = True) -> dict:
        """导入自己的 .gguf（不传 path 时弹文件对话框）；select=True 时设为当前模型。"""
        if isinstance(path, str) and path.strip():
            src = path.strip().strip('"')
        else:
            picked = self._need_dialogs().open_files((_GGUF_FILTER, _ALL_FILTER), False)
            if not picked:
                return {"imported": None, "status": self.llm_status()}
            src = picked[0]
        info = dict(self._models().import_file(src) or {})
        if select and info.get("id"):
            snap = self.store.update({"translate": {"local": {"model": str(info["id"])}}})
            self._publish({"type": "settings", "settings": snap})
        return {"imported": info, "status": self.llm_status()}

    @_guard("读取 API 服务商")
    def api_presets(self) -> list[dict]:
        return [dict(p) for p in self._models().presets() if isinstance(p, dict) and p.get("id")]

    @_guard("获取模型列表")
    def api_models(self, base_url: str = "", api_key: str = "", provider: str = "") -> dict:
        """GET {base_url}/models。地址留空 = 用服务商预设；Key 留空 = 用已保存的 Key。"""
        cfg = self._settings()["translate"].get("api") or {}
        url = str(base_url or "").strip()
        if not url:
            pid = str(provider or cfg.get("provider") or "")
            url = next((str(p.get("base_url") or "") for p in self.api_presets() if p.get("id") == pid), "")
        if not _URL_RE.match(url):
            raise ApiError("请先填写正确的服务地址（以 http:// 或 https:// 开头）")
        key = str(api_key or "").strip() or str(cfg.get("api_key") or "")
        return {"base_url": url, "models": [str(m) for m in self._models().api_models(url, key) or []]}

    # ---- 字体 ------------------------------------------------------------------------
    def _font_registry(self) -> Any:
        with self._lock:
            if self._fonts is None:
                from .typeset.fonts import FontRegistry

                self._fonts = FontRegistry(fonts_dir(), [os.path.join(data_dir(), "fonts")])
            return self._fonts

    @_guard("读取字体")
    def list_fonts(self) -> list[dict]:
        try:
            return list(self._font_registry().list())
        except Exception:  # noqa: BLE001
            log.warning("字体库不可用，使用内置字体目录", exc_info=True)
            return [{"id": i, "name": n, "path": "", "tags": list(t)} for i, n, t in _FONT_FALLBACK]

    @_guard("添加字体")
    def add_font(self, path: Optional[str] = None) -> dict:
        paths = [path] if isinstance(path, str) and path.strip() else self._need_dialogs().open_files((_FONT_FILTER,), True)
        if not paths:
            return {"added": [], "fonts": self.list_fonts()}
        reg = self._font_registry()
        added: list[dict] = []
        errors: list[str] = []
        for p in paths:
            try:
                added.extend(reg.install(p))
            except ValueError as e:
                errors.append(f"{os.path.basename(p)}：{e}")
        if not added and errors:
            raise ApiError("；".join(errors)[:300])
        # 引擎自己的字体库也刷新（新字体马上能在排版里用）：engine.refresh_fonts() 或 engine.fonts.refresh()
        engine_fonts = getattr(self.engine, "fonts", None)
        for fn in (getattr(self.engine, "refresh_fonts", None),
                   getattr(engine_fonts, "refresh", None) if engine_fonts is not reg else None):
            if callable(fn):
                try:
                    fn()
                except Exception:  # noqa: BLE001
                    log.debug("刷新引擎字体失败", exc_info=True)
        with self._lock:
            self._preview_cache.clear()
        return {"added": added, "fonts": self.list_fonts(), "errors": errors}

    def _font_preview(self, font_id: str, text: str = "") -> Optional[bytes]:
        """字体预览 PNG（黑字透明底）。供 /font-preview 路由使用。"""
        text = (text or "").strip()[:40] or "汉化漫画 你好世界 Aa123"
        key = (font_id, text)
        with self._lock:
            hit = self._preview_cache.get(key)
            if hit is not None:
                self._preview_cache.move_to_end(key)
                return hit
        from PIL import Image, ImageDraw

        font = self._font_registry().get(font_id, 40)
        probe = ImageDraw.Draw(Image.new("L", (1, 1)))
        x1, y1, x2, y2 = probe.textbbox((0, 0), text, font=font)
        w, h = min(1600, max(1, x2 - x1) + 8), min(160, max(1, y2 - y1) + 8)
        im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ImageDraw.Draw(im).text((4 - x1, 4 - y1), text, font=font, fill=(0, 0, 0, 255))
        buf = io.BytesIO()
        im.save(buf, "PNG", optimize=True)
        data = buf.getvalue()
        with self._lock:
            self._preview_cache[key] = data
            while len(self._preview_cache) > 96:
                self._preview_cache.popitem(last=False)
        return data

    # ---- 诊断 ------------------------------------------------------------------------
    @_guard("记录界面错误")
    def ui_error(self, message: str = "", source: str = "", line: int = 0, stack: str = "") -> dict:
        with self._lock:
            self._ui_errors += 1
            n = self._ui_errors
        if n <= MAX_UI_ERRORS:
            log.warning("界面脚本错误：%s（%s:%s）\n%s", str(message)[:500], str(source)[-200:], line, str(stack)[:2000])
        return {"logged": n <= MAX_UI_ERRORS}


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
class _SafeStreamHandler(logging.StreamHandler):
    """控制台是 cp1252 时中文会编码失败：退化成转义输出，不刷 Logging error。"""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            stream = self.stream
            try:
                stream.write(msg + self.terminator)
            except UnicodeEncodeError:
                enc = getattr(stream, "encoding", None) or "ascii"
                stream.write(msg.encode(enc, "backslashreplace").decode(enc, "replace") + self.terminator)
            self.flush()
        except Exception:  # noqa: BLE001
            self.handleError(record)


def setup_logging(debug: bool = False, console: bool = False) -> str:
    """日志写到 data_dir()/logs/app.log（2MB × 5 轮转）；重复调用不会重复加 handler。"""
    log_dir = os.path.join(data_dir(), "logs")
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, "app.log")
    root = logging.getLogger()
    if not any(getattr(h, "_hanhua", False) for h in root.handlers):
        fh = logging.handlers.RotatingFileHandler(path, maxBytes=2 * 1024 * 1024, backupCount=5, encoding="utf-8", delay=True)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s [%(threadName)s] %(message)s"))
        fh._hanhua = True  # type: ignore[attr-defined]
        root.addHandler(fh)
        if console and sys.stderr is not None:
            sh = _SafeStreamHandler(sys.stderr)
            sh.setLevel(logging.DEBUG if debug else logging.INFO)
            sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))
            sh._hanhua = True  # type: ignore[attr-defined]
            root.addHandler(sh)
    root.setLevel(logging.DEBUG if debug else logging.INFO)
    for noisy in ("pywebview", "urllib3", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return path


def _install_excepthooks() -> None:
    def hook(exc_type: Any, exc: Any, tb: Any) -> None:
        log.critical("未处理的异常", exc_info=(exc_type, exc, tb))

    def thread_hook(args: Any) -> None:
        if args.exc_type is SystemExit:
            return
        log.critical("线程 %s 未处理的异常", getattr(args.thread, "name", "?"),
                     exc_info=(args.exc_type, args.exc_value, args.exc_traceback))

    sys.excepthook = hook
    threading.excepthook = thread_hook


def ui_dir() -> str:
    """界面文件目录（源码运行 / PyInstaller 打包都能找到）。"""
    cands = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui")]
    for parts in (("hanhua", "ui"), ("ui",)):
        try:
            cands.append(resource_path(*parts))
        except Exception:  # noqa: BLE001
            pass
    for c in cands:
        if os.path.isfile(os.path.join(c, "index.html")):
            return c
    return cands[0]


def make_engine(store: SettingsStore, bus: EventBus, demo: bool) -> tuple[Any, str]:
    """创建引擎：--demo 用假引擎；正式引擎加载失败时返回替身 + 中文原因（界面照常打开）。"""
    if demo:
        from .ui.demo import DemoEngine

        return DemoEngine(store, bus.publish), ""
    try:
        from .engine import Engine
    except Exception as e:  # noqa: BLE001
        log.exception("处理引擎模块加载失败")
        reason = f"处理引擎加载失败：{brief_error(e)}"
        return UnavailableEngine(reason), reason
    try:
        work_root = store.get()["output"].get("work_dir") or None
        return Engine(store, bus.publish, work_root=work_root), ""
    except Exception as e:  # noqa: BLE001
        log.exception("处理引擎初始化失败")
        reason = f"处理引擎初始化失败：{brief_error(e)}"
        return UnavailableEngine(reason), reason


def webview2_available() -> bool:
    """WebView2 运行库是否已安装（缺失时 pywebview 会退回 IE 内核，新界面无法运行）。"""
    if sys.platform != "win32":
        return True
    try:
        import platform
        import winreg
    except ImportError:
        return True
    keys = ("{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}", "{2CD8A007-E189-409D-A2C8-9AF4EF3C72AA}",
            "{0D50BFEC-CD6A-4F9A-964C-C7416E3ACB10}", "{65C35B14-6C1D-4122-AC46-7148CC9D6497}")
    for key in keys:
        for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
            sub = (rf"SOFTWARE\Microsoft\EdgeUpdate\Clients\{key}" if hive == winreg.HKEY_CURRENT_USER or platform.machine() == "x86"
                   else rf"SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{key}")
            try:
                with winreg.OpenKey(hive, sub) as k:
                    pv, _ = winreg.QueryValueEx(k, "pv")
            except OSError:
                continue
            try:
                if tuple(int(x) for x in str(pv).split(".")[:3]) >= (86, 0, 622):
                    return True
            except ValueError:
                continue
    return False


def _message_box(text: str, title: str = APP_TITLE, error: bool = True) -> None:
    log.error("提示框：%s", text) if error else log.info("提示框：%s", text)
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.user32.MessageBoxW(None, text, title, 0x10 if error else 0x40)  # type: ignore[attr-defined]
            return
        except Exception:  # noqa: BLE001
            pass
    _console(f"{title}: {text}")


def _console(text: str) -> None:
    out = sys.stdout
    if out is None:
        return
    try:
        out.write(text + "\n")
    except UnicodeEncodeError:
        enc = getattr(out, "encoding", None) or "ascii"
        out.write(text.encode(enc, "replace").decode(enc, "replace") + "\n")
    try:
        out.flush()
    except Exception:  # noqa: BLE001
        pass


def _system_dark() -> bool:
    if sys.platform != "win32":
        return False
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize") as k:
            v, _ = winreg.QueryValueEx(k, "AppsUseLightTheme")
            return int(v) == 0
    except OSError:
        return False


def _on_drop(api: Api, event: Any) -> None:
    """pywebview DOM drop 事件：取 pywebviewFullPath 加入任务，结果用 toast 事件告诉界面。"""
    try:
        files = ((event or {}).get("dataTransfer") or {}).get("files") or []
        paths = [f["pywebviewFullPath"] for f in files if isinstance(f, dict) and f.get("pywebviewFullPath")]
        if not paths:
            if files:
                api.bus.publish({"type": "toast", "level": "warning", "message": "没有拿到文件的完整路径，请改用“添加文件”按钮"})
            return
        res = api.add_paths(paths)
        n, skipped = len(res["jobs"]), len(res["skipped"])
        msg = f"已添加 {n} 项，开始汉化" if n else "没有可以添加的文件"
        if skipped:
            msg += f"（{skipped} 项无法读取，已跳过）"
        api.bus.publish({"type": "toast", "level": "success" if n else "warning", "message": msg})
    except HanhuaError as e:
        api.bus.publish({"type": "toast", "level": "error", "message": str(e)})
    except Exception as e:  # noqa: BLE001
        log.exception("处理拖放出错")
        api.bus.publish({"type": "toast", "level": "error", "message": f"添加文件失败：{brief_error(e)}"})


def _bind_drop(window: Any, api: Api) -> None:
    """每次页面加载后重新绑定（pywebview 在 loaded 时会清空 DOM 元素表）。"""
    try:
        from webview.dom import DOMEventHandler

        window.dom.document.events.drop += DOMEventHandler(lambda e: _on_drop(api, e), prevent_default=True)
    except Exception:  # noqa: BLE001
        log.exception("绑定拖放事件失败")


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="hanhua", description=APP_TITLE)
    p.add_argument("paths", nargs="*", help="启动后自动加入的漫画文件 / 文件夹")
    p.add_argument("--demo", action="store_true", help="演示模式（假引擎，不加载模型）")
    p.add_argument("--no-window", action="store_true", help="只启动本地服务并打印地址（浏览器调试）")
    p.add_argument("--port", type=int, default=0, help="固定端口（默认随机）")
    p.add_argument("--debug", action="store_true", help="详细日志 + 网页开发者工具")
    args, unknown = p.parse_known_args(argv)
    if unknown:
        log.debug("忽略未知参数：%s", unknown)
    return args


def _shutdown(engine: Any, server: Optional[AppServer], api: Optional[Api]) -> None:
    if api is not None:
        api.wait_ops(timeout=5.0)
    done = threading.Event()

    def run() -> None:
        try:
            engine.shutdown()
        except Exception:  # noqa: BLE001
            log.exception("关闭引擎出错")
        finally:
            done.set()

    threading.Thread(target=run, name="hanhua-shutdown", daemon=True).start()
    if not done.wait(20):
        log.warning("引擎 20 秒内没有关闭完成，强制退出")
    if server is not None:
        server.stop()


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    log_path = setup_logging(args.debug, console=bool(args.no_window or args.debug))
    _install_excepthooks()
    log.info("%s 启动（版本 %s，%s，日志 %s）", APP_TITLE, VERSION, "演示模式" if args.demo else "正式模式", log_path)

    if not args.no_window and not webview2_available():
        _message_box("没有检测到 Microsoft Edge WebView2 运行库，界面无法显示。\n\n"
                     "点“确定”后会打开微软官方下载页面，请安装“常青版引导程序（Evergreen Bootstrapper）”，"
                     "装好后重新打开本软件。")
        try:
            webbrowser.open(WEBVIEW2_URL)
        except Exception:  # noqa: BLE001
            pass
        return 2

    store = SettingsStore()
    bus = EventBus()
    engine, engine_error = make_engine(store, bus, args.demo)
    api = Api(store, engine, bus, demo=args.demo, engine_error=engine_error)
    try:
        server = AppServer(api, ui_dir(), bus=bus, port=max(0, int(args.port))).start()
    except OSError as e:
        log.exception("界面服务启动失败")
        _message_box(f"界面服务启动失败：{brief_error(e)}\n请检查是否被安全软件拦截。")
        _shutdown(engine, None, api)
        return 3

    if args.paths:
        def add_initial() -> None:
            try:
                res = api.add_paths(list(args.paths))
                if res["jobs"]:
                    bus.publish({"type": "toast", "level": "success", "message": f"已添加 {len(res['jobs'])} 项，开始汉化"})
            except Exception as e:  # noqa: BLE001
                log.warning("启动时添加文件失败：%s", e)

        threading.Thread(target=add_initial, name="hanhua-initial", daemon=True).start()

    code = 0
    try:
        if args.no_window:
            code = _serve_forever(server)
        else:
            code = _run_window(api, server, store, args.debug)
    finally:
        _shutdown(engine, server, api)
        log.info("%s 已退出", APP_TITLE)
    return code


def _serve_forever(server: AppServer) -> int:
    _console(server.url)
    _console("（浏览器打开上面的地址；按 Ctrl+C 退出）")
    stop = threading.Event()
    try:
        while not stop.wait(0.5):
            pass
    except KeyboardInterrupt:
        pass
    return 0


def _run_window(api: Api, server: AppServer, store: SettingsStore, debug: bool) -> int:
    try:
        import webview
    except Exception as e:  # noqa: BLE001
        log.exception("pywebview 加载失败")
        _message_box(f"窗口组件加载失败：{brief_error(e)}\n请重新安装软件。")
        return 4
    theme = store.get()["general"]["theme"]
    dark = theme == "dark" or (theme == "auto" and _system_dark())
    window = webview.create_window(
        APP_TITLE, server.url, width=WINDOW_SIZE[0], height=WINDOW_SIZE[1], min_size=WINDOW_MIN,
        background_color="#0f0f12" if dark else "#f6f6f8", text_select=True, zoomable=False,
    )
    if window is None:
        _message_box("窗口创建失败，请重新安装软件。")
        return 4
    api.dialogs = WebviewDialogs(window)
    window.events.loaded += lambda: _bind_drop(window, api)
    try:
        kwargs: dict[str, Any] = {"debug": debug, "private_mode": False,
                                  "storage_path": os.path.join(data_dir(), "webview")}
        if sys.platform == "win32":
            kwargs["gui"] = "edgechromium"
        webview.start(**kwargs)
    except Exception as e:  # noqa: BLE001
        log.exception("窗口运行出错")
        _message_box(f"窗口运行出错：{brief_error(e)}\n详情已写入日志。")
        return 5
    finally:
        api.dialogs = None
    return 0


if __name__ == "__main__":
    sys.exit(main())
