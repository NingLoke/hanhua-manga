"""本地大模型翻译：按配置推荐模型、下载、运行 llama-server、整页翻译。"""
from .catalog import BY_ID, CATALOG, LLMModel, Recommendation, recommend
from .client import ChatTranslator
from .manager import (LlamaServer, available_backends, current_recommendation, delete_model, download,
                      import_gguf, list_models, resolve_model_path)

__all__ = [
    "BY_ID", "CATALOG", "ChatTranslator", "LLMModel", "LlamaServer", "Recommendation", "available_backends",
    "current_recommendation", "delete_model", "download", "import_gguf", "list_models", "recommend",
    "resolve_model_path",
]
