"""硬件识别 + ONNX Runtime 会话管理 + 画质档位。

同一套代码，三个发行版：
  - nvidia 版：onnxruntime-gpu（CUDA）      → device "cuda"
  - gpu 版   ：onnxruntime-directml（DX12）  → device "dml"（AMD / Intel / NVIDIA 都能用）
  - cpu 版   ：onnxruntime（CPU）            → device "cpu"
运行时自动探测可用后端，GPU 初始化失败时逐级降级，绝不因为显卡问题直接崩掉。

所有模型调用都必须走 Runtime.model(key).run(...)，不要自己 new InferenceSession。
"""
from __future__ import annotations

import ctypes
import json
import logging
import os
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import numpy as np

log = logging.getLogger("hanhua.runtime")

try:  # onnxruntime 是唯一推理依赖；三个发行版分别装 gpu / directml / cpu 包
    import onnxruntime as ort
except Exception as _e:  # pragma: no cover - 打包错误时给出明确提示
    ort = None  # type: ignore[assignment]
    _ORT_IMPORT_ERROR: Optional[Exception] = _e
else:
    _ORT_IMPORT_ERROR = None
    try:
        ort.set_default_logger_severity(3)  # 只保留 error，屏蔽 Memcpy 之类的刷屏警告
    except Exception:  # noqa: BLE001
        pass

VERSION = "4.0.5"

# 模型 key → 相对 models/ 的路径
MODEL_FILES: dict[str, str] = {
    "detector": "detector/detector.onnx",                 # RT-DETR-v2 fp32，动态尺寸（GPU 用）
    "detector_int8": "detector/detector_int8.onnx",       # 动态量化，CPU 用
    "detector_s_int8": "detector/detector-v4-s_int8.onnx",  # 小模型，固定 640×640，CPU 极速档
    "ocr_encoder": "manga-ocr/encoder_model.onnx",
    "ocr_decoder": "manga-ocr/decoder_model.onnx",
    "ocr_encoder_int8": "manga-ocr/encoder_model_int8.onnx",
    "ocr_decoder_int8": "manga-ocr/decoder_model_int8.onnx",
    "lama": "lama/lama-manga-dynamic.onnx",
    "ppocr_det": "ppocr/det.onnx",
    "ppocr_rec_multi": "ppocr/rec_multi.onnx",            # 简中 / 繁中 / 英 / 日（横排）
    "ppocr_rec_korean": "ppocr/rec_korean.onnx",
    "ppocr_rec_en": "ppocr/rec_en.onnx",
}

DEVICE_LABELS = {"cuda": "NVIDIA CUDA", "dml": "DirectML 显卡加速", "cpu": "CPU"}


# --------------------------------------------------------------------------------------
# 路径
# --------------------------------------------------------------------------------------
def _source_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def app_root() -> str:
    """包含 models/ fonts/ hanhua/ui 的根目录。

    查找顺序：环境变量 HANHUA_ROOT → exe 所在目录（打包后 models 放在 exe 旁边）
    → PyInstaller 的 _MEIPASS → 源码目录（4.0/）。
    """
    cands: list[str] = []
    env = os.environ.get("HANHUA_ROOT")
    if env:
        cands.append(env)
    if getattr(sys, "frozen", False):
        cands.append(os.path.dirname(sys.executable))
        mp = getattr(sys, "_MEIPASS", None)
        if mp:
            cands.append(mp)
    cands.append(_source_root())
    for c in cands:
        if os.path.isdir(os.path.join(c, "models")):
            return c
    return cands[0]


def resource_path(*parts: str) -> str:
    """打包后 ui 等资源在 _MEIPASS 里，models/fonts 在 exe 旁边；两边都找。"""
    rel = os.path.join(*parts)
    roots = [app_root()]
    mp = getattr(sys, "_MEIPASS", None)
    if mp:
        roots.append(mp)
    roots.append(_source_root())
    for r in roots:
        p = os.path.join(r, rel)
        if os.path.exists(p):
            return p
    return os.path.join(roots[0], rel)


def models_dir() -> str:
    return os.path.join(app_root(), "models")


def fonts_dir() -> str:
    return os.path.join(app_root(), "fonts")


def data_dir() -> str:
    """用户数据目录：设置、翻译缓存、任务断点、日志。可用 HANHUA_DATA 覆盖。"""
    base = os.environ.get("HANHUA_DATA")
    if not base:
        appdata = os.environ.get("APPDATA") or os.path.expanduser("~")
        base = os.path.join(appdata, "HanhuaManga")
    os.makedirs(base, exist_ok=True)
    return base


def build_info() -> dict[str, Any]:
    """打包时写入 build_info.json：{"variant": "nvidia"|"gpu"|"cpu", "version": "..."}。"""
    p = os.path.join(app_root(), "build_info.json")
    info: dict[str, Any] = {"variant": "dev", "version": VERSION}
    try:
        with open(p, "r", encoding="utf-8") as f:
            info.update(json.load(f))
    except FileNotFoundError:
        pass
    except Exception as e:  # noqa: BLE001
        log.warning("build_info.json 读取失败: %s", e)
    return info


# --------------------------------------------------------------------------------------
# 硬件识别（只用标准库 + 注册表，毫秒级，不调外部命令）
# --------------------------------------------------------------------------------------
@dataclass
class GpuInfo:
    name: str
    vram_mb: int
    vendor: str  # "nvidia" | "amd" | "intel" | "other"


@dataclass
class HardwareInfo:
    variant: str
    ort_version: str
    providers: list[str]
    gpus: list[GpuInfo] = field(default_factory=list)
    cpu_name: str = ""
    cpu_threads: int = 1
    ram_gb: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def best_gpu(self) -> Optional[GpuInfo]:
        real = [g for g in self.gpus if g.vendor in ("nvidia", "amd", "intel")]
        return max(real, key=lambda g: g.vram_mb) if real else None


def _vendor_of(name: str) -> str:
    n = name.lower()
    if "nvidia" in n or "geforce" in n or "rtx" in n or "quadro" in n:
        return "nvidia"
    if "amd" in n or "radeon" in n:
        return "amd"
    if "intel" in n or "arc" in n or "iris" in n or "uhd" in n:
        return "intel"
    return "other"


def _gpus_from_registry() -> list[GpuInfo]:
    gpus: list[GpuInfo] = []
    if os.name != "nt":
        return gpus
    try:
        import winreg

        cls = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, cls) as root:
            i = 0
            while True:
                try:
                    sub = winreg.EnumKey(root, i)
                except OSError:
                    break
                i += 1
                if not sub.isdigit():
                    continue
                try:
                    with winreg.OpenKey(root, sub) as k:
                        name = str(winreg.QueryValueEx(k, "DriverDesc")[0])
                        vram = 0
                        for val in ("HardwareInformation.qwMemorySize", "HardwareInformation.MemorySize"):
                            try:
                                raw = winreg.QueryValueEx(k, val)[0]
                                vram = int.from_bytes(raw, "little") if isinstance(raw, bytes) else int(raw)
                                if vram:
                                    break
                            except OSError:
                                continue
                        gpus.append(GpuInfo(name=name, vram_mb=int(vram // (1024 * 1024)), vendor=_vendor_of(name)))
                except OSError:
                    continue
    except Exception as e:  # noqa: BLE001
        log.debug("读取显卡注册表失败: %s", e)
    return gpus


def _cpu_name() -> str:
    if os.name == "nt":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                return str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
        except Exception:  # noqa: BLE001
            pass
    import platform

    return platform.processor() or "CPU"


def _ram_gb() -> float:
    if os.name == "nt":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        st = MEMORYSTATUSEX()
        st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return round(st.ullTotalPhys / 1024**3, 1)
    return 0.0


_HW_CACHE: Optional[HardwareInfo] = None


def detect_hardware(refresh: bool = False) -> HardwareInfo:
    global _HW_CACHE
    if _HW_CACHE is not None and not refresh:
        return _HW_CACHE
    providers = list(ort.get_available_providers()) if ort is not None else []
    _HW_CACHE = HardwareInfo(
        variant=str(build_info().get("variant", "dev")),
        ort_version=getattr(ort, "__version__", "missing"),
        providers=providers,
        gpus=[g for g in _gpus_from_registry() if "virtual" not in g.name.lower() and "basic" not in g.name.lower()],
        cpu_name=_cpu_name(),
        cpu_threads=os.cpu_count() or 1,
        ram_gb=_ram_gb(),
    )
    return _HW_CACHE


# --------------------------------------------------------------------------------------
# 画质档位
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Profile:
    name: str                    # "fast" | "balanced" | "quality"
    device: str                  # "cuda" | "dml" | "cpu"
    detector: str                # MODEL_FILES key
    detector_size: int           # 检测输入边长（RT-DETR 训练尺寸 640）
    ocr_encoder: str
    ocr_decoder: str
    ocr_batch: int               # OCR 每批裁剪数
    inpaint_max_side: int        # LaMa 裁剪最长边（超出先缩小再修复）
    inpaint_bucket: int          # LaMa 输入尺寸分桶粒度（避免 GPU 对每种新尺寸重复预热）
    cpu_workers: int             # 排版 / 编码等 CPU 阶段并行页数

    def to_dict(self) -> dict:
        return asdict(self)


def resolve_profile(name: str, device: str, hw: Optional[HardwareInfo] = None) -> Profile:
    """按档位 + 实际设备 + 实际存在的模型，选出具体参数。缺模型时自动退到可用版本。"""
    hw = hw or detect_hardware()
    name = name if name in ("fast", "balanced", "quality") else "balanced"
    threads = max(1, hw.cpu_threads)
    low_ram = 0 < hw.ram_gb < 7.5

    if device in ("cuda", "dml"):
        det, enc, dec = "detector", "ocr_encoder", "ocr_decoder"
        inpaint = {"fast": 512, "balanced": 768, "quality": 1024}[name]
        batch = 16
    else:
        det = {"fast": "detector_s_int8", "balanced": "detector_int8", "quality": "detector"}[name]
        enc, dec = ("ocr_encoder", "ocr_decoder") if name == "quality" else ("ocr_encoder_int8", "ocr_decoder_int8")
        inpaint = {"fast": 384, "balanced": 512, "quality": 768}[name]
        batch = 8
    if low_ram:
        inpaint = min(inpaint, 512)
        batch = min(batch, 4)

    def pick(*keys: str) -> str:
        for k in keys:
            if os.path.exists(os.path.join(models_dir(), MODEL_FILES[k])):
                return k
        return keys[0]

    det = pick(det, "detector", "detector_int8", "detector_s_int8")
    enc = pick(enc, "ocr_encoder", "ocr_encoder_int8")
    dec = pick(dec, "ocr_decoder", "ocr_decoder_int8")
    return Profile(
        name=name,
        device=device,
        detector=det,
        detector_size=640,
        ocr_encoder=enc,
        ocr_decoder=dec,
        ocr_batch=batch,
        inpaint_max_side=inpaint,
        inpaint_bucket=128 if device != "cpu" else 64,
        cpu_workers=max(1, min(4, threads // 3)) if not low_ram else 1,
    )


# --------------------------------------------------------------------------------------
# 会话管理
# --------------------------------------------------------------------------------------
class Model:
    """线程安全的 InferenceSession 包装。GPU 后端串行执行（DirectML 不支持并发 Run）。"""

    def __init__(self, key: str, sess: "ort.InferenceSession", device: str):
        self.key = key
        self.sess = sess
        self.device = device  # 实际使用的设备（可能因降级变成 cpu）
        self.provider = sess.get_providers()[0]
        self.input_names = [i.name for i in sess.get_inputs()]
        self.output_names = [o.name for o in sess.get_outputs()]
        self.input_shapes = {i.name: i.shape for i in sess.get_inputs()}
        self._lock = threading.Lock() if device != "cpu" else None
        self.calls = 0
        self.total_s = 0.0

    def run(self, feeds: dict[str, np.ndarray], outputs: Optional[list[str]] = None) -> list[np.ndarray]:
        t = time.perf_counter()
        if self._lock is not None:
            with self._lock:
                out = self.sess.run(outputs, feeds)
        else:
            out = self.sess.run(outputs, feeds)
        self.calls += 1
        self.total_s += time.perf_counter() - t
        return out


class Runtime:
    """设备选择 + 模型懒加载缓存。一个进程一个实例（Engine 持有）。"""

    def __init__(self, device: str = "auto", cpu_threads: int = 0, gpu_id: int = 0):
        if ort is None:
            raise RuntimeError(f"onnxruntime 加载失败：{_ORT_IMPORT_ERROR}")
        self.hw = detect_hardware()
        self.gpu_id = int(gpu_id)
        self.cpu_threads = int(cpu_threads) if cpu_threads else max(1, (os.cpu_count() or 2) // 2)
        self.requested = device
        self.device = self._choose_device(device)
        self._models: dict[str, Model] = {}
        self._lock = threading.Lock()
        self._dlls_loaded = False
        log.info("推理后端：%s（请求 %s，可用 %s）", self.device, device, self.hw.providers)

    # -- 设备 ----------------------------------------------------------------------------
    def _choose_device(self, device: str) -> str:
        avail = self.hw.providers
        has_cuda = "CUDAExecutionProvider" in avail and any(g.vendor == "nvidia" for g in self.hw.gpus)
        has_dml = "DmlExecutionProvider" in avail and any(g.vendor in ("nvidia", "amd", "intel") for g in self.hw.gpus)
        if device == "cuda" and has_cuda:
            return "cuda"
        if device == "dml" and has_dml:
            return "dml"
        if device == "cpu":
            return "cpu"
        if device not in ("auto", "cuda", "dml", "cpu"):
            log.warning("未知设备 %s，改用自动", device)
        # auto 或请求的后端不可用：按 cuda → dml → cpu 降级
        if has_cuda:
            return "cuda"
        if has_dml:
            return "dml"
        return "cpu"

    @property
    def label(self) -> str:
        gpu = self.hw.best_gpu
        if self.device == "cpu" or gpu is None:
            return f"CPU · {self.hw.cpu_name}"
        return f"{gpu.name} · {DEVICE_LABELS[self.device]}"

    def _providers(self, device: str) -> list:
        if device == "cuda":
            return [
                ("CUDAExecutionProvider", {
                    "device_id": self.gpu_id,
                    "cudnn_conv_algo_search": "HEURISTIC",   # 动态尺寸下 EXHAUSTIVE 会每种尺寸重搜一次
                    "arena_extend_strategy": "kSameAsRequested",
                }),
                "CPUExecutionProvider",
            ]
        if device == "dml":
            # 核显 + 独显的机器上，0 号适配器可能是核显：默认按"高性能"挑独显，手动指定 gpu_id 时才用编号
            opts: dict[str, Any] = (
                {"device_id": self.gpu_id} if self.gpu_id
                else {"performance_preference": "high_performance", "device_filter": "gpu"}
            )
            return [("DmlExecutionProvider", opts), "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def _session_options(self, device: str) -> "ort.SessionOptions":
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.log_severity_level = 3
        if device == "dml":
            so.enable_mem_pattern = False
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        else:
            so.intra_op_num_threads = self.cpu_threads
        return so

    def _preload_cuda(self) -> None:
        if self._dlls_loaded:
            return
        self._dlls_loaded = True
        pre = getattr(ort, "preload_dlls", None)
        if pre is not None:
            # 打包后 CUDA / cuDNN 的 DLL 平铺在 <app>/cuda/ 下；开发环境从 nvidia 的 pip 包里找
            flat = os.path.join(app_root(), "cuda")
            try:
                if os.path.isdir(flat):
                    pre(directory=flat)
                else:
                    pre()
            except Exception as e:  # noqa: BLE001
                log.warning("CUDA DLL 预加载失败（将尝试直接创建会话）: %s", e)

    # -- 模型 ----------------------------------------------------------------------------
    def model_path(self, key: str) -> str:
        return os.path.join(models_dir(), MODEL_FILES[key])

    def has_model(self, key: str) -> bool:
        return key in MODEL_FILES and os.path.exists(self.model_path(key))

    def model(self, key: str) -> Model:
        """懒加载并缓存。GPU 创建失败自动退到 CPU（记日志，不抛）。"""
        with self._lock:
            m = self._models.get(key)
            if m is not None:
                return m
            path = self.model_path(key)
            if not os.path.exists(path):
                raise FileNotFoundError(f"缺少模型文件：{path}（请重新安装软件）")
            chain = [self.device] + (["cpu"] if self.device != "cpu" else [])
            last: Optional[Exception] = None
            for dev in chain:
                try:
                    if dev == "cuda":
                        self._preload_cuda()
                    t = time.perf_counter()
                    sess = ort.InferenceSession(path, self._session_options(dev), providers=self._providers(dev))
                    m = Model(key, sess, dev)
                    log.info("加载模型 %s → %s（%.1fs）", key, m.provider, time.perf_counter() - t)
                    if dev != self.device:
                        log.warning("模型 %s 在 %s 上加载失败，已降级为 CPU", key, self.device)
                    self._models[key] = m
                    return m
                except Exception as e:  # noqa: BLE001
                    last = e
                    log.warning("模型 %s 在 %s 上创建失败: %s", key, dev, e)
            raise RuntimeError(f"模型 {key} 无法加载：{last}")

    def stats(self) -> dict[str, dict]:
        return {k: {"provider": m.provider, "calls": m.calls, "total_s": round(m.total_s, 3)} for k, m in self._models.items()}

    def release(self) -> None:
        with self._lock:
            self._models.clear()
