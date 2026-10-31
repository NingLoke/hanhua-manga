"""翻译服务用的两个引擎适配器：本地模型（llama.cpp）和 API 服务（填 Key，OpenAI 兼容）。

统一接口：translate_batch(texts, src, tgt, context=None, glossary=None) -> list[str]
"""
from __future__ import annotations

import logging
import threading
from typing import Optional

from ..runtime import detect_hardware
from ..types import HanhuaError
from .catalog import BY_ID, CATALOG
from .client import ChatTranslator
from .manager import LlamaServer, available_backends, current_recommendation, is_installed, resolve_model_path

log = logging.getLogger("hanhua.llm")

# API 服务预设（模型名会随厂商更新，界面提供「获取模型列表」按钮按 /models 实时拉取）
API_PRESETS: dict[str, dict] = {
    "deepseek": {"name": "DeepSeek", "base_url": "https://api.deepseek.com/v1", "model": "deepseek-chat"},
    "siliconflow": {"name": "硅基流动", "base_url": "https://api.siliconflow.cn/v1", "model": ""},
    "openai": {"name": "OpenAI", "base_url": "https://api.openai.com/v1", "model": ""},
    "gemini": {"name": "Google Gemini", "base_url": "https://generativelanguage.googleapis.com/v1beta/openai", "model": ""},
    "openrouter": {"name": "OpenRouter", "base_url": "https://openrouter.ai/api/v1", "model": ""},
    "custom": {"name": "自定义（OpenAI 兼容）", "base_url": "", "model": ""},
}

_SERVER = LlamaServer()          # 进程内唯一的本地引擎实例


def shared_server() -> LlamaServer:
    return _SERVER


class LocalLLMEngine:
    """本地模型。settings["translate"]["local"] = {model: "auto"|id, backend: "auto"|cuda|vulkan|cpu,
    gpu_layers: -1(自动)|n, ctx: 8192}。"""

    id = "local"
    name = "本地大模型"

    def __init__(self, settings: dict):
        cfg = (settings.get("translate") or {}).get("local") or {}
        self.profile = str((settings.get("hardware") or {}).get("profile") or "balanced")
        self.model_id = str(cfg.get("model") or "auto")
        self.backend = str(cfg.get("backend") or "auto")
        self.gpu_layers = int(cfg.get("gpu_layers", -1))
        self.ctx = int(cfg.get("ctx") or 8192)
        self._lock = threading.Lock()
        self._client: Optional[ChatTranslator] = None

    def resolve(self) -> tuple[str, str, int]:
        """→ (模型文件路径, 后端, gpu_layers)。缺模型 / 缺后端时抛 HanhuaError（服务会换下一个引擎）。"""
        rec = current_recommendation(self.profile)
        mid = self.model_id
        if mid == "auto":
            mid = rec.model.id if rec else ""
            if not (mid and resolve_model_path(mid)):
                # 推荐的还没下载：用已经装好的、本机跑得动的质量最好的那个（例如推荐 Q6、本机已有 Q8）
                limit = rec.model.need_gb * 1.35 if rec else float("inf")
                ok = [m for m in CATALOG if is_installed(m) and m.need_gb <= limit]
                if ok:
                    mid = max(ok, key=lambda m: m.quality).id
        path = resolve_model_path(mid)
        if not path:
            name = BY_ID[mid].name if mid in BY_ID else mid
            raise HanhuaError(f"本地翻译模型「{name}」还没有下载，请在 设置 → 翻译 里下载")
        backends = available_backends()
        backend = self.backend if self.backend in backends else (rec.backend if rec and rec.backend in backends else
                                                                  (backends[0] if backends else ""))
        if not backend:
            raise HanhuaError("缺少本地翻译引擎，请重新安装软件")
        layers = self.gpu_layers if self.gpu_layers >= 0 else (0 if backend == "cpu" else 999)
        return path, backend, layers

    def available(self) -> bool:
        try:
            self.resolve()
            return True
        except HanhuaError:
            return False

    def translate_batch(self, texts: list[str], src: str, tgt: str, context: Optional[list[str]] = None,
                        glossary: Optional[list[dict]] = None) -> list[str]:
        with self._lock:
            path, backend, layers = self.resolve()
            threads = max(1, (detect_hardware().cpu_threads or 2) // 2) if backend == "cpu" else 0
            url = _SERVER.ensure(path, backend, gpu_layers=layers, ctx=self.ctx, threads=threads)
            if self._client is None or self._client.base_url != url.rstrip("/"):
                self._client = ChatTranslator(url, timeout=300.0)
            client = self._client
        return client.translate(texts, src, tgt, glossary, context)


class ApiLLMEngine:
    """填 Key 的 API 服务。settings["translate"]["api"] = {provider, base_url, api_key, model}。"""

    id = "api"
    name = "API 服务"

    def __init__(self, settings: dict):
        cfg = (settings.get("translate") or {}).get("api") or {}
        preset = API_PRESETS.get(str(cfg.get("provider") or "custom"), API_PRESETS["custom"])
        self.base_url = str(cfg.get("base_url") or preset["base_url"])
        self.model = str(cfg.get("model") or preset["model"])
        self.api_key = str(cfg.get("api_key") or "")
        self._client = ChatTranslator(self.base_url, self.model, self.api_key, timeout=120.0) if self.available() else None

    def available(self) -> bool:
        local = self.base_url.startswith(("http://127.0.0.1", "http://localhost"))   # 自装的 Ollama / LM Studio 不需要 Key
        return bool(self.base_url and self.model and (self.api_key or local))

    def translate_batch(self, texts: list[str], src: str, tgt: str, context: Optional[list[str]] = None,
                        glossary: Optional[list[dict]] = None) -> list[str]:
        if self._client is None:
            raise HanhuaError("还没有填写 API 服务的地址、模型和 Key")
        return self._client.translate(texts, src, tgt, glossary, context)


def list_api_models(base_url: str, api_key: str, timeout: float = 15.0) -> list[str]:
    """GET {base_url}/models，给设置界面的「获取模型列表」按钮用。"""
    import json
    import urllib.request

    req = urllib.request.Request(base_url.rstrip("/") + "/models", headers={"Authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        raise HanhuaError(f"获取模型列表失败：{e}") from e
    return sorted(str(m.get("id")) for m in data.get("data", []) if m.get("id"))
