"""本地翻译大模型清单 + 按电脑配置自动推荐。

全部为 Apache-2.0 授权（可商用）的 Qwen 模型 GGUF 量化版，由 llama.cpp（MIT）在本机运行：
NVIDIA 版用 CUDA 后端，通用显卡版用 Vulkan 后端（AMD / Intel / NVIDIA 都能跑），CPU 版用 CPU 后端。
下载源：Hugging Face（unsloth）与国内镜像 ModelScope（同名仓库，SHA-256 一致）。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

from ..runtime import HardwareInfo

# GPU 版里检测 / OCR / 去字模型本身要占的显存，以及系统和显示输出的余量（GB）
VISION_VRAM_GB = 2.5
SYSTEM_VRAM_GB = 0.6
# CPU 跑大模型时给系统和本软件其余部分留的内存（GB）
SYSTEM_RAM_GB = 6.0


@dataclass(frozen=True)
class LLMModel:
    id: str
    name: str                 # 界面显示名
    params_b: float           # 参数量（十亿）
    quant: str
    file: str                 # GGUF 文件名
    size: int                 # 字节
    sha256: str               # 空 = 不校验（自定义导入的模型）
    hf_repo: str
    ms_repo: str              # ModelScope 仓库（国内镜像），空 = 无
    quality: int              # 翻译质量档（越大越好）
    ctx_gb: float             # 8K 上下文 KV cache 等额外显存 / 内存（GB）
    license: str = "Apache-2.0"

    @property
    def size_gb(self) -> float:
        return self.size / 1024**3

    @property
    def need_gb(self) -> float:
        return self.size_gb + self.ctx_gb

    def to_dict(self) -> dict:
        d = asdict(self)
        d["size_gb"] = round(self.size_gb, 2)
        d["need_gb"] = round(self.need_gb, 2)
        return d


CATALOG: tuple[LLMModel, ...] = (
    LLMModel("qwen3.5-2b-q4", "Qwen3.5 2B（Q4，轻量）", 2, "Q4_K_M", "Qwen3.5-2B-Q4_K_M.gguf", 1280835840,
             "aaf42c8b7c3cab2bf3d69c355048d4a0ee9973d48f16c731c0520ee914699223",
             "unsloth/Qwen3.5-2B-GGUF", "unsloth/Qwen3.5-2B-GGUF", 1, 0.4),
    LLMModel("qwen3.5-2b-q8", "Qwen3.5 2B（Q8）", 2, "Q8_0", "Qwen3.5-2B-Q8_0.gguf", 2012012800,
             "1b04acba824817554f4ce23639bc8495ff70453b8fcb047900c731521021f2c1",
             "unsloth/Qwen3.5-2B-GGUF", "unsloth/Qwen3.5-2B-GGUF", 2, 0.4),
    LLMModel("qwen3.5-4b-q4", "Qwen3.5 4B（Q4，均衡）", 4, "Q4_K_M", "Qwen3.5-4B-Q4_K_M.gguf", 2740937888,
             "00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4",
             "unsloth/Qwen3.5-4B-GGUF", "unsloth/Qwen3.5-4B-GGUF", 3, 0.6),
    LLMModel("qwen3.5-4b-q8", "Qwen3.5 4B（Q8）", 4, "Q8_0", "Qwen3.5-4B-Q8_0.gguf", 4482403488,
             "10cc391b403021dd11c614679d2fd92f611c3681d29e29651b717316965d61e1",
             "unsloth/Qwen3.5-4B-GGUF", "unsloth/Qwen3.5-4B-GGUF", 4, 0.6),
    LLMModel("qwen3.5-9b-q4", "Qwen3.5 9B（Q4）", 9, "Q4_K_M", "Qwen3.5-9B-Q4_K_M.gguf", 5680522464,
             "03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8",
             "unsloth/Qwen3.5-9B-GGUF", "unsloth/Qwen3.5-9B-GGUF", 5, 0.9),
    # 9B：Q6 与 Q8 翻译质量几乎相同，Q6 小 2GB、快约 25%，所以默认推荐 Q6；Q8 留给追求极致的用户手动选
    LLMModel("qwen3.5-9b-q6", "Qwen3.5 9B（Q6，高质量）", 9, "Q6_K", "Qwen3.5-9B-Q6_K.gguf", 7458301152,
             "91898433cf5ce0a8f45516a4cc3e9343b6e01d052d01f684309098c66a326c59",
             "unsloth/Qwen3.5-9B-GGUF", "unsloth/Qwen3.5-9B-GGUF", 7, 0.9),
    LLMModel("qwen3.5-9b-q8", "Qwen3.5 9B（Q8，体积更大）", 9, "Q8_0", "Qwen3.5-9B-Q8_0.gguf", 9527502048,
             "809626574d0cb43d4becfa56169980da2bb448f2299270f7be443cb89d0a6ae4",
             "unsloth/Qwen3.5-9B-GGUF", "unsloth/Qwen3.5-9B-GGUF", 6, 0.9),
    LLMModel("qwen3.8-27b-q4", "Qwen3.8 27B（Q4，旗舰，需 24GB 显存）", 27, "Q4_K_M", "Qwen3.8-27B-Q4_K_M.gguf",
             int(18.97e9), "", "ggml-org/Qwen3.8-27B-GGUF", "", 9, 1.8),
)

BY_ID = {m.id: m for m in CATALOG}


@dataclass(frozen=True)
class Recommendation:
    model: LLMModel
    backend: str              # cuda | vulkan | cpu
    gpu_layers: int           # -ngl：999 = 全部放显卡
    reason: str               # 给客户看的中文说明

    def to_dict(self) -> dict:
        return {"model": self.model.to_dict(), "backend": self.backend,
                "gpu_layers": self.gpu_layers, "reason": self.reason}


def _gpu_budget(hw: HardwareInfo) -> tuple[Optional[str], float]:
    gpu = hw.best_gpu
    if gpu is None or gpu.vram_mb < 1500:
        return None, 0.0
    return gpu.name, gpu.vram_mb / 1024 - VISION_VRAM_GB - SYSTEM_VRAM_GB


def recommend(hw: HardwareInfo, backends: list[str], allow_flagship: bool = True,
              max_params: Optional[float] = None) -> Optional[Recommendation]:
    """按本机硬件和可用后端，挑能流畅跑的质量最高的模型。backends 为本发行版自带的 llama.cpp 后端。
    max_params：画质档位为「极速」时限制模型大小（4B 比 9B 快约 2.5 倍）。"""
    pool = [m for m in CATALOG if (allow_flagship or m.params_b < 20) and (max_params is None or m.params_b <= max_params)]
    gpu_name, budget = _gpu_budget(hw)
    gpu_backend = "cuda" if ("cuda" in backends and hw.best_gpu and hw.best_gpu.vendor == "nvidia") else (
        "vulkan" if "vulkan" in backends else None)
    if gpu_backend and gpu_name and budget > 0:
        fit = [m for m in pool if m.need_gb <= budget]
        if fit:
            best = max(fit, key=lambda m: m.quality)
            return Recommendation(best, gpu_backend, 999,
                                  f"检测到 {gpu_name}（可用约 {budget:.1f}GB 显存），"
                                  f"推荐 {best.name}，整模型放在显卡上运行，翻译最快。")
    if "cpu" in backends or not backends:
        ram_budget = (hw.ram_gb or 8.0) - SYSTEM_RAM_GB
        cores = max(1, hw.cpu_threads // 2)
        cpu_pool = [m for m in pool if m.need_gb <= ram_budget and m.params_b <= (4 if cores >= 6 else 2)]
        best = max(cpu_pool, key=lambda m: m.quality) if cpu_pool else BY_ID["qwen3.5-2b-q4"]
        why = "没有可用的独立显卡" if not gpu_name else f"{gpu_name} 显存不足以同时运行识别模型和翻译模型"
        return Recommendation(best, "cpu", 0,
                              f"{why}，推荐用 CPU 运行 {best.name}（约 {best.size_gb:.1f}GB）。"
                              "速度较慢；追求速度可改用「API 服务」。")
    return None
