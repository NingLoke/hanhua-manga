"""翻译模块：``from hanhua.translate import TranslationService``。

- service.TranslationService：整页批量 + 多引擎切换 + 限速 / 熔断 / 退避 + 缓存 + 文本处理；
- engines：免费在线引擎（google 可用；edge / tencent 2026-10 实测失效，保留类）+ 本地大模型（OpenAI 兼容）；
- cache.TranslationCache：sqlite（WAL）翻译缓存；
- sfx_dict.lookup：内置拟声词词典；textproc：原文规范化 / 中文后处理 / 术语表等纯函数。
"""
from . import sfx_dict
from .cache import TranslationCache
from .engines import ENGINE_INFO, EngineError, SplitNeeded
from .service import TranslationService

__all__ = ["TranslationService", "TranslationCache", "EngineError", "SplitNeeded", "ENGINE_INFO", "sfx_dict"]
