"""核心数据结构：所有模块之间只通过这些类型交换数据。

坐标约定：Box = (x1, y1, x2, y2)，页面像素坐标，x2/y2 为开区间（与 numpy 切片一致）。
颜色约定：tuple 一律是 RGB；图片数组一律是 BGR uint8（OpenCV 习惯）。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Optional

Box = tuple[int, int, int, int]
RGB = tuple[int, int, int]

# 页面处理阶段（严格按顺序推进；断点续跑时从下一个阶段继续）
STAGES: tuple[str, ...] = ("pending", "detected", "recognized", "translated", "cleaned", "rendered")


def stage_at_least(stage: str, target: str) -> bool:
    """stage 是否已经完成到 target（含）。未知阶段（如 error）视为 pending。"""
    try:
        return STAGES.index(stage) >= STAGES.index(target)
    except ValueError:
        return False


def clamp_box(box: Box, w: int, h: int) -> Box:
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(x1), w))
    y1 = max(0, min(int(y1), h))
    x2 = max(x1, min(int(x2), w))
    y2 = max(y1, min(int(y2), h))
    return (x1, y1, x2, y2)


def expand_box(box: Box, pad: int | float, w: int, h: int) -> Box:
    p = int(round(pad))
    return clamp_box((box[0] - p, box[1] - p, box[2] + p, box[3] + p), w, h)


def box_area(box: Box) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def box_iou(a: Box, b: Box) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def box_contain_ratio(inner: Box, outer: Box) -> float:
    """inner 有多大比例落在 outer 里（0..1）。"""
    ix = max(0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    iy = max(0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    a = box_area(inner)
    return (ix * iy) / a if a > 0 else 0.0


def _tuple_or_none(v: Any) -> Optional[tuple]:
    return tuple(int(x) for x in v) if v is not None else None


@dataclass
class TextBlock:
    """一个文字块（一个气泡里的一段对白，或一段气泡外的旁白 / 招牌 / 拟声词）。"""

    id: int                                   # 页内阅读顺序序号 0..n-1
    box: Box                                  # 检测器给出的紧贴文字框
    kind: str = "bubble"                      # "bubble" 气泡内文字 | "free" 气泡外文字
    score: float = 0.0                        # 检测置信度 0..1
    bubble_box: Optional[Box] = None          # 所属气泡框（kind == "bubble" 时）
    direction: str = "v"                      # 原文方向 "v" 竖排 | "h" 横排
    src_text: str = ""                        # OCR 原文
    ocr_conf: float = 0.0                     # OCR 置信度 0..1
    is_sfx: bool = False                      # 是否为拟声词 / 效果字
    translation: str = ""                     # 译文
    text_color: Optional[RGB] = None          # 原文字形颜色（估计）
    stroke_color: Optional[RGB] = None        # 原文描边颜色（有描边时）
    # ---- 编辑器里的人工覆盖项（None = 自动）----
    skip: bool = False                        # 保留原文：不擦除、不嵌字
    font: Optional[str] = None                # 字体 id（FontRegistry 的 key）
    font_size: Optional[int] = None           # 指定字号（px）
    out_direction: Optional[str] = None       # 译文排版方向 "v" | "h" | None 自动
    align: Optional[str] = None               # "center" | "start" | "end" | None 自动
    fill: Optional[RGB] = None                # 译文颜色
    stroke: Optional[RGB] = None              # 译文描边颜色
    stroke_width: Optional[float] = None      # 描边宽度（相对字号，如 0.12）
    # ---- 排版结果（Typesetter 回写，给编辑器显示）----
    render_box: Optional[Box] = None
    render_font_size: Optional[int] = None
    overflow: bool = False                    # 最小字号仍放不下（已强制缩放进气泡）

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, tuple):
                d[k] = list(v)
        return d

    @staticmethod
    def from_dict(d: dict) -> "TextBlock":
        names = {f.name for f in fields(TextBlock)}
        kw = {k: v for k, v in d.items() if k in names}
        for k in ("box", "bubble_box", "text_color", "stroke_color", "fill", "stroke", "render_box"):
            if k in kw:
                kw[k] = _tuple_or_none(kw[k])
        return TextBlock(**kw)


@dataclass
class PageData:
    """一页的全部中间结果（断点续跑的最小单位）。"""

    index: int                                # 页序号（0 起）
    name: str                                 # 在源里的相对路径 / 条目名
    width: int = 0
    height: int = 0
    blocks: list[TextBlock] = field(default_factory=list)
    bubbles: list[Box] = field(default_factory=list)   # 所有检测到的气泡框
    stage: str = "pending"                    # 见 STAGES；出错时为 "error"
    error: str = ""
    timings: dict[str, float] = field(default_factory=dict)
    version: int = 0                          # 每次重新渲染 +1（界面刷新缓存用）

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "name": self.name,
            "width": self.width,
            "height": self.height,
            "blocks": [b.to_dict() for b in self.blocks],
            "bubbles": [list(b) for b in self.bubbles],
            "stage": self.stage,
            "error": self.error,
            "timings": dict(self.timings),
            "version": self.version,
        }

    @staticmethod
    def from_dict(d: dict) -> "PageData":
        return PageData(
            index=int(d["index"]),
            name=str(d["name"]),
            width=int(d.get("width", 0)),
            height=int(d.get("height", 0)),
            blocks=[TextBlock.from_dict(b) for b in d.get("blocks", [])],
            bubbles=[tuple(int(x) for x in b) for b in d.get("bubbles", [])],  # type: ignore[misc]
            stage=str(d.get("stage", "pending")),
            error=str(d.get("error", "")),
            timings={k: float(v) for k, v in d.get("timings", {}).items()},
            version=int(d.get("version", 0)),
        )


class HanhuaError(Exception):
    """所有可预期错误的基类（界面会把 message 直接显示给客户，必须是中文、说人话）。"""


class TranslationUnavailable(HanhuaError):
    """所有翻译引擎都不可用（断网 / 全部被限流）。任务应暂停而不是产出未翻译的页面。"""


class UnsupportedFormat(HanhuaError):
    """不支持或损坏的输入文件。"""


class Cancelled(HanhuaError):
    """用户取消 / 暂停。"""
