"""本地大模型管理：模型下载 / 校验 / 导入 + llama-server 进程生命周期。

llama.cpp 运行库放在 <app>/runtimes/llama/<cuda|vulkan|cpu>/llama-server.exe（每个发行版只带自己那一个）；
模型文件放在用户数据目录 <data>/models/llm/（体积大，按需下载，不进安装包）。
"""
from __future__ import annotations

import atexit
import concurrent.futures as cf
import hashlib
import logging
import os
import shutil
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from typing import Callable, Optional

from ..runtime import app_root, data_dir, detect_hardware
from ..types import Cancelled, HanhuaError
from .catalog import BY_ID, CATALOG, LLMModel, Recommendation, recommend

log = logging.getLogger("hanhua.llm")

BACKENDS = ("cuda", "vulkan", "cpu")
SOURCE_URLS = {
    "hf": "https://huggingface.co/{repo}/resolve/main/{file}",
    "ms": "https://modelscope.cn/models/{repo}/resolve/master/{file}",
}
SOURCE_NAMES = {"hf": "Hugging Face", "ms": "ModelScope（国内镜像）"}
_UA = {"User-Agent": "hanhua-manga/4.0"}
Progress = Callable[[int, int, float], None]   # (已下载字节, 总字节, 速度 B/s)


# --------------------------------------------------------------------------------------
# 运行库与模型文件
# --------------------------------------------------------------------------------------
def runtimes_dir() -> str:
    return os.path.join(app_root(), "runtimes", "llama")


def server_exe(backend: str) -> Optional[str]:
    p = os.path.join(runtimes_dir(), backend, "llama-server.exe")
    return p if os.path.exists(p) else None


def available_backends() -> list[str]:
    return [b for b in BACKENDS if server_exe(b)]


def llm_models_dir() -> str:
    # HANHUA_LLM_DIR：模型放在别的盘（体积大）或测试时共用已下载的模型
    d = os.environ.get("HANHUA_LLM_DIR") or os.path.join(data_dir(), "models", "llm")
    os.makedirs(d, exist_ok=True)
    return d


def model_path(m: LLMModel) -> str:
    return os.path.join(llm_models_dir(), m.file)


def is_installed(m: LLMModel) -> bool:
    p = model_path(m)
    return os.path.exists(p) and os.path.getsize(p) == m.size


def custom_models() -> list[dict]:
    """用户自己导入的 GGUF（不在内置清单里的）。"""
    known = {m.file for m in CATALOG}
    out = []
    for fn in sorted(os.listdir(llm_models_dir())):
        if fn.lower().endswith(".gguf") and fn not in known:
            p = os.path.join(llm_models_dir(), fn)
            out.append({"id": "custom:" + fn, "name": fn, "path": p, "size_gb": round(os.path.getsize(p) / 1024**3, 2),
                        "installed": True, "custom": True})
    return out


def resolve_model_path(model_id: str) -> Optional[str]:
    if model_id.startswith("custom:"):
        p = os.path.join(llm_models_dir(), model_id[len("custom:"):])
        return p if os.path.exists(p) else None
    m = BY_ID.get(model_id)
    return model_path(m) if m and is_installed(m) else None


def import_gguf(src: str) -> dict:
    if not src.lower().endswith(".gguf") or not os.path.isfile(src):
        raise HanhuaError("请选择 .gguf 格式的模型文件")
    with open(src, "rb") as f:
        if f.read(4) != b"GGUF":
            raise HanhuaError("这个文件不是有效的 GGUF 模型")
    dst = os.path.join(llm_models_dir(), os.path.basename(src))
    if os.path.abspath(src) != os.path.abspath(dst):
        try:
            os.link(src, dst)                 # 同盘硬链接：秒完成、不占双份空间
        except OSError:
            shutil.copy2(src, dst)
    return {"id": "custom:" + os.path.basename(dst), "name": os.path.basename(dst), "path": dst}


def list_models(rec: Optional[Recommendation] = None) -> list[dict]:
    rows = []
    for m in CATALOG:
        d = m.to_dict()
        d["installed"] = is_installed(m)
        d["recommended"] = bool(rec and rec.model.id == m.id)
        part = model_path(m) + ".part"
        d["partial_bytes"] = os.path.getsize(part) if os.path.exists(part) else 0
        rows.append(d)
    return rows + custom_models()


def current_recommendation(profile: str = "balanced") -> Optional[Recommendation]:
    """profile 为画质档位：fast（极速）只推荐 4B 及以下的模型，其余按配置推荐最好的。"""
    return recommend(detect_hardware(), available_backends(), max_params=4 if profile == "fast" else None)


# --------------------------------------------------------------------------------------
# 下载（断点续传 + 多源 + SHA-256 校验）
# --------------------------------------------------------------------------------------
def _probe(src: str, m: LLMModel, timeout: float) -> float:
    repo = m.hf_repo if src == "hf" else m.ms_repo
    if not repo:
        return float("inf")
    url = SOURCE_URLS[src].format(repo=repo, file=m.file)
    t = time.perf_counter()
    try:
        req = urllib.request.Request(url, headers={**_UA, "Range": "bytes=0-65535"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            r.read()
        return time.perf_counter() - t
    except Exception:  # noqa: BLE001
        return float("inf")


def pick_sources(m: LLMModel, timeout: float = 6.0) -> list[str]:
    """同时试探两个下载源，按响应速度排序（国内访问不了 HF 时自动用 ModelScope）。"""
    with cf.ThreadPoolExecutor(2) as ex:
        t = dict(zip(("hf", "ms"), ex.map(lambda s: _probe(s, m, timeout), ("hf", "ms"))))
    order = sorted((s for s in t if t[s] != float("inf")), key=lambda s: t[s])
    return order or ["ms", "hf"]


def _fetch(url: str, part: str, total: int, progress: Optional[Progress], cancel: Optional[threading.Event]) -> None:
    have = os.path.getsize(part) if os.path.exists(part) else 0
    if total and have >= total:
        return
    headers = dict(_UA)
    if have:
        headers["Range"] = f"bytes={have}-"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as r:
        if have and r.status != 206:
            have = 0
        t0, done0, last = time.perf_counter(), have, 0.0
        with open(part, "ab" if have else "wb") as f:
            done = have
            while True:
                if cancel is not None and cancel.is_set():
                    raise Cancelled("已取消下载")
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                now = time.perf_counter()
                if progress and now - last > 0.25:
                    last = now
                    progress(done, total, (done - done0) / max(1e-6, now - t0))
    if total and os.path.getsize(part) < total:
        raise HanhuaError("下载中断")


def _sha256(path: str, cancel: Optional[threading.Event]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            if cancel is not None and cancel.is_set():
                raise Cancelled("已取消")
            b = f.read(8 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def download(m: LLMModel, progress: Optional[Progress] = None, cancel: Optional[threading.Event] = None,
             sources: Optional[list[str]] = None) -> str:
    dest = model_path(m)
    if is_installed(m):
        return dest
    part = dest + ".part"
    last: Optional[Exception] = None
    for src in sources or pick_sources(m):
        repo = m.hf_repo if src == "hf" else m.ms_repo
        if not repo:
            continue
        url = SOURCE_URLS[src].format(repo=repo, file=m.file)
        for attempt in range(3):
            try:
                log.info("下载 %s ← %s（第 %d 次）", m.file, SOURCE_NAMES[src], attempt + 1)
                _fetch(url, part, m.size, progress, cancel)
                if m.sha256 and _sha256(part, cancel) != m.sha256:
                    os.remove(part)
                    raise HanhuaError("模型文件校验失败（下载不完整或被篡改），已删除，请重试")
                os.replace(part, dest)
                return dest
            except Cancelled:
                raise
            except Exception as e:  # noqa: BLE001
                last = e
                log.warning("下载 %s 失败：%s", m.file, e)
                time.sleep(min(10, 2 * (attempt + 1)))
    raise HanhuaError(f"模型下载失败：{last}（可在设置里改用「导入模型」手动导入 .gguf 文件）")


def delete_model(model_id: str) -> None:
    p = resolve_model_path(model_id)
    for path in filter(None, [p, (p + ".part") if p else None]):
        try:
            os.remove(path)
        except OSError:
            pass


# --------------------------------------------------------------------------------------
# llama-server 进程
# --------------------------------------------------------------------------------------
def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _cuda_dll_dirs() -> list[str]:
    """llama.cpp 的 CUDA 版依赖 cudart / cuBLAS：打包后与 onnxruntime 共用 <app>/cuda，开发时用 nvidia pip 包。"""
    dirs = [os.path.join(app_root(), "cuda")]
    try:
        import nvidia  # type: ignore[import-not-found]

        for base in getattr(nvidia, "__path__", []):
            dirs += [os.path.join(base, "cu13", "bin", "x86_64"), os.path.join(base, "cudnn", "bin")]
    except Exception:  # noqa: BLE001
        pass
    return [d for d in dirs if os.path.isdir(d)]


class LlamaServer:
    """单实例 llama-server。换模型 / 后端时自动重启；进程崩溃后下次调用自动拉起。"""

    def __init__(self) -> None:
        self._proc: Optional[subprocess.Popen] = None
        self._key: Optional[tuple] = None
        self._port = 0
        self._lock = threading.Lock()
        self._log_path = os.path.join(data_dir(), "logs", "llama-server.log")
        atexit.register(self.stop)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._port}/v1"

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def ensure(self, model_file: str, backend: str, gpu_layers: int = 999, ctx: int = 8192,
               threads: int = 0, timeout: float = 240.0) -> str:
        key = (os.path.abspath(model_file), backend, gpu_layers, ctx, threads)
        with self._lock:
            if self.alive and self._key == key:
                return self.base_url
            self._stop_locked()
            exe = server_exe(backend)
            if exe is None:
                raise HanhuaError(f"缺少本地翻译引擎（{backend}），请重新安装软件")
            self._port = _free_port()
            args = [exe, "-m", model_file, "--host", "127.0.0.1", "--port", str(self._port),
                    "-c", str(ctx), "-ngl", str(gpu_layers), "-np", "1", "--jinja",
                    "--reasoning-budget", "0", "--no-webui"]
            if threads:
                args += ["-t", str(threads)]
            env = dict(os.environ)
            if backend == "cuda":
                env["PATH"] = os.pathsep.join(_cuda_dll_dirs() + [env.get("PATH", "")])
            os.makedirs(os.path.dirname(self._log_path), exist_ok=True)
            logf = open(self._log_path, "w", encoding="utf-8", errors="replace")
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            log.info("启动本地翻译引擎：%s", " ".join(args[1:]))
            self._proc = subprocess.Popen(args, stdout=logf, stderr=subprocess.STDOUT, env=env,
                                          cwd=os.path.dirname(exe), creationflags=flags)
            self._key = key
            self._wait_ready(timeout)
            return self.base_url

    def _wait_ready(self, timeout: float) -> None:
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self._proc is None or self._proc.poll() is not None:
                tail = self._log_tail()
                hint = "显存不足，请在设置里换一个更小的本地模型" if ("out of memory" in tail.lower() or "alloc" in tail.lower()) else "请查看日志"
                self._proc = None
                raise HanhuaError(f"本地翻译引擎启动失败：{hint}\n{tail[-600:]}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self._port}/health", timeout=2) as r:
                    if r.status == 200:
                        log.info("本地翻译引擎就绪（%.1fs）", time.time() - t0)
                        return
            except (urllib.error.URLError, OSError, ConnectionError):
                pass
            time.sleep(0.4)
        self._stop_locked()
        raise HanhuaError("本地翻译引擎启动超时（模型可能太大），请换小一点的模型")

    def _log_tail(self) -> str:
        try:
            with open(self._log_path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()[-4000:]
        except OSError:
            return ""

    def stop(self) -> None:
        with self._lock:
            self._stop_locked()

    def _stop_locked(self) -> None:
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=5)
            except Exception:  # noqa: BLE001
                try:
                    self._proc.kill()
                except Exception:  # noqa: BLE001
                    pass
        self._proc = None
        self._key = None
