"""嵌字排版：FontRegistry（字体）+ Typesetter（按气泡形状排版并渲染译文）。"""
from .fonts import DEFAULT_FONT, FontRegistry
from .render import BlockRender, Typesetter

__all__ = ["DEFAULT_FONT", "BlockRender", "FontRegistry", "Typesetter"]
