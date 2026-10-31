"""嵌字渲染：Typesetter。

流程（每页）：
1. 在去字后的图上为每个要嵌字的块求区域（气泡内部 mask / 气泡外矩形），相互重叠的区域按 Voronoi 切开；
2. 原文字号估计（按行方向推算，再用整页中位数校正）→ 字号范围 [lo, hi]（页面尺寸相对化，可被设置覆盖）；
3. 选方向 / 字体（繁体目标自动换 tc）/ 颜色（保证对比度）/ 描边；
4. layout.fit_block 二分字号并按 mask 逐行 / 逐列可用长度排版，放不下逐级退让；
5. 每个字画成小的 alpha stamp（填充 + 描边两张），拼到区域大小的图层上，乘以区域 mask 后合成回 BGR。

不变量：画出的每个像素（含描边、抗锯齿边缘）都在该块的区域 mask 之内 —— 字永远不会画出气泡。
"""
from __future__ import annotations

import logging
import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..config import DEFAULTS, column_order
from ..types import Box, HanhuaError, TextBlock
from .fonts import FontRegistry
from .layout import (
    Layout,
    LayoutParams,
    Region,
    _sane_text_box,
    build_tokens,
    choose_direction,
    compute_region,
    estimate_src_font,
    fit_block,
    font_range,
    normalize_text,
    page_ref,
    partition_regions,
)

log = logging.getLogger("hanhua.typeset")

RGB = tuple[int, int, int]
BLACK: RGB = (0, 0, 0)
WHITE: RGB = (255, 255, 255)
_MIN_TEXT_CONTRAST = 3.0          # 气泡字与底色的最低对比度（WCAG 大字标准）
_MIN_STROKE_CONTRAST = 1.8        # 描边色与字色的最低对比度


# ======================================================================================
# 颜色工具
# ======================================================================================
def _rgb(c: Any) -> Optional[RGB]:
    """任意 [r, g, b(, a)] → 合法 RGB tuple；非法返回 None。"""
    if c is None:
        return None
    try:
        vals = [int(round(float(x))) for x in list(c)[:3]]
    except (TypeError, ValueError):
        return None
    if len(vals) != 3:
        return None
    return (min(max(vals[0], 0), 255), min(max(vals[1], 0), 255), min(max(vals[2], 0), 255))


def _lum(c: RGB) -> float:
    """WCAG 相对亮度 0..1。"""
    def ch(v: int) -> float:
        x = v / 255.0
        return x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4

    return 0.2126 * ch(c[0]) + 0.7152 * ch(c[1]) + 0.0722 * ch(c[2])


def contrast(a: RGB, b: RGB) -> float:
    """WCAG 对比度 1..21。"""
    la, lb = _lum(a), _lum(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


def best_bw(bg: RGB) -> RGB:
    """在底色上对比度更高的黑 / 白。"""
    return BLACK if contrast(BLACK, bg) >= contrast(WHITE, bg) else WHITE


# ======================================================================================
# 数据结构
# ======================================================================================
@dataclass
class _Style:
    font: str                     # 实际字体 id（已按简繁换好）
    fill: RGB
    stroke: Optional[RGB]
    stroke_ratio: float           # 描边宽度 / 字号（0 = 无描边）

    def stroke_px(self, fs: int) -> int:
        if self.stroke is None or self.stroke_ratio <= 0:
            return 0
        return max(1, int(round(self.stroke_ratio * fs)))


@dataclass
class BlockRender:
    """一个块的排版明细（调试 / 测试 / 编辑器用）。"""

    block_id: int
    region: Region
    layout: Optional[Layout]
    stage: str                    # normal | shrink | relaxed | overflow | forced | failed
    direction: str
    font: str
    font_range: tuple[int, int]
    clipped: int = 0              # 被区域 mask 裁掉的像素数（正常排版应为 0）
    render_box: Optional[Box] = None


@dataclass
class _Stamp:
    fill: np.ndarray                      # uint8 (h, w)：填充 alpha
    stroke: Optional[np.ndarray]          # uint8 (h, w)：描边（含填充）alpha
    ox: float                             # 文字原点（基线左端）在 stamp 里的位置
    oy: float
    cx: float = 0.0                       # 填充墨迹中心
    cy: float = 0.0
    total: int = field(default=0)         # 非零像素数（统计裁剪用）


def _ink_center(a: np.ndarray) -> Optional[tuple[float, float]]:
    ys = np.flatnonzero(a.any(axis=1))
    xs = np.flatnonzero(a.any(axis=0))
    if ys.size == 0:
        return None
    return ((xs[0] + xs[-1] + 1) / 2.0, (ys[0] + ys[-1] + 1) / 2.0)


def _finish_stamp(fill: np.ndarray, stroke: Optional[np.ndarray], ox: float, oy: float) -> Optional[_Stamp]:
    c = _ink_center(fill)
    if c is None:
        return None
    cover = stroke if stroke is not None else fill
    return _Stamp(fill, stroke, ox, oy, c[0], c[1], int(np.count_nonzero(cover)))


def _make_stamp(font: ImageFont.FreeTypeFont, text: str, sw: int) -> Optional[_Stamp]:
    """把一段文字画成小 alpha 图（填充 / 描边各一张，同尺寸同原点）。"""
    try:
        x0, y0, x1, y1 = font.getbbox(text, anchor="ls", stroke_width=sw)
    except Exception:  # noqa: BLE001 - 个别字体 / 字符量不出来就跳过这个字
        return None
    size = int(getattr(font, "size", 16) or 16)
    pad = 2 + sw + int(0.15 * size)        # 墨迹可能超出 advance 框（斜体 / 负侧距），多留一点
    w, h = int(math.ceil(x1 - x0)) + 2 * pad, int(math.ceil(y1 - y0)) + 2 * pad
    if w <= 2 * pad or h <= 2 * pad or w * h > 64_000_000:
        return None
    ox, oy = pad - x0, pad - y0
    im = Image.new("L", (w, h), 0)
    ImageDraw.Draw(im).text((ox, oy), text, font=font, fill=255, anchor="ls")
    fill = np.asarray(im)
    stroke = None
    if sw > 0:
        im2 = Image.new("L", (w, h), 0)
        ImageDraw.Draw(im2).text((ox, oy), text, font=font, fill=255, anchor="ls",
                                 stroke_width=sw, stroke_fill=255)
        stroke = np.asarray(im2)
    return _finish_stamp(fill, stroke, float(ox), float(oy))


def _rotate_cw(st: _Stamp) -> Optional[_Stamp]:
    """顺时针旋转 90°（竖排里的英文单词 / 长音符号）。"""
    f = np.ascontiguousarray(np.rot90(st.fill, k=-1))
    s = np.ascontiguousarray(np.rot90(st.stroke, k=-1)) if st.stroke is not None else None
    return _finish_stamp(f, s, 0.0, 0.0)


def _stretch_x(st: _Stamp, length: float) -> Optional[_Stamp]:
    """沿横向把墨迹拉伸 / 压缩到 length 像素（破折号连成一条线）。"""
    cover = st.stroke if st.stroke is not None else st.fill
    xs = np.flatnonzero(cover.any(axis=0))
    if xs.size == 0:
        return None
    a, b = int(xs[0]), int(xs[-1]) + 1
    ink = float(b - a)
    new_w = max(1, int(round(length)))
    if abs(new_w - ink) < 1:
        return st
    h = cover.shape[0]
    f = cv2.resize(np.ascontiguousarray(st.fill[:, a:b]), (new_w, h), interpolation=cv2.INTER_LINEAR)
    s = cv2.resize(np.ascontiguousarray(st.stroke[:, a:b]), (new_w, h), interpolation=cv2.INTER_LINEAR) \
        if st.stroke is not None else None
    return _finish_stamp(f, s, 0.0, 0.0)


class _Canvas:
    """区域大小的两张 alpha 图层（填充 / 描边）；stamp 用 max 合并（抗锯齿边缘不会互相变暗）。"""

    def __init__(self, h: int, w: int, with_stroke: bool) -> None:
        self.fill = np.zeros((h, w), np.uint8)
        self.stroke = np.zeros((h, w), np.uint8) if with_stroke else None
        self.lost = 0                       # 落在图层外的像素（计入裁剪统计）

    def paste(self, st: _Stamp, px: float, py: float) -> None:
        """把 stamp 的 (0, 0) 放到图层的 (px, py)。"""
        x, y = int(round(px)), int(round(py))
        sh, sw_ = st.fill.shape
        H, W = self.fill.shape
        x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + sw_), min(H, y + sh)
        if x1 <= x0 or y1 <= y0:
            self.lost += st.total
            return
        sub = (slice(y0 - y, y1 - y), slice(x0 - x, x1 - x))
        dst = (slice(y0, y1), slice(x0, x1))
        np.maximum(self.fill[dst], st.fill[sub], out=self.fill[dst])
        cover = st.stroke if st.stroke is not None else st.fill
        if self.stroke is not None:
            np.maximum(self.stroke[dst], cover[sub], out=self.stroke[dst])
        if x0 > x or y0 > y or x1 < x + sw_ or y1 < y + sh:
            self.lost += st.total - int(np.count_nonzero(cover[sub]))

    def center(self, st: _Stamp, cx: float, cy: float) -> None:
        self.paste(st, cx - st.cx, cy - st.cy)


def _as_bgr(img: Any) -> np.ndarray:
    """任意输入图 → BGR uint8 (H, W, 3) 的新数组。"""
    if not isinstance(img, np.ndarray) or img.ndim not in (2, 3) or img.size == 0:
        raise HanhuaError("嵌字失败：输入的页面图片无效")
    a = img
    if a.dtype != np.uint8:
        a = np.clip(a.astype(np.float32), 0, 255).astype(np.uint8)
    if a.ndim == 2:
        return cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
    if a.shape[2] == 1:
        return cv2.cvtColor(a[:, :, 0], cv2.COLOR_GRAY2BGR)
    if a.shape[2] >= 4:
        return np.ascontiguousarray(a[:, :, :3]).copy()
    return np.ascontiguousarray(a).copy()


def _num(d: dict, key: str, default: float, lo: float, hi: float) -> float:
    try:
        v = float(d.get(key, default))
    except (TypeError, ValueError):
        v = default
    if not math.isfinite(v):
        v = default
    return min(max(v, lo), hi)


# ======================================================================================
# Typesetter
# ======================================================================================
class Typesetter:
    """按气泡形状排版并渲染译文。构造一次、可在多个线程里反复调用（无可变共享状态）。"""

    def __init__(self, settings: dict, fonts: FontRegistry) -> None:
        settings = settings if isinstance(settings, dict) else {}
        ts = dict(DEFAULTS["typeset"])
        if isinstance(settings.get("typeset"), dict):
            ts.update(settings["typeset"])
        self.ts = ts
        tr = settings.get("translate") if isinstance(settings.get("translate"), dict) else {}
        self.target = "zh-Hant" if tr.get("target") == "zh-Hant" else "zh-Hans"
        self.fonts = fonts
        self.line_spacing = _num(ts, "line_spacing", 1.12, 0.8, 3.0)
        self.letter_spacing = _num(ts, "letter_spacing", 0.04, -0.2, 1.0)
        self.stroke_free = _num(ts, "stroke_free", 0.14, 0.0, 0.5)
        self.margin = _num(ts, "bubble_margin", 0.12, 0.0, 0.35)
        self.keep_color = bool(ts.get("keep_original_color", True))
        d = ts.get("direction", "auto")
        self.direction = d if d in ("auto", "h", "v") else "auto"
        self.column_order = column_order(settings)

    # -- 对外接口 -------------------------------------------------------------------------
    def render(self, clean: np.ndarray, blocks: list[TextBlock]) -> np.ndarray:
        """把每个 skip=False 且 translation 非空的块排进去；回写 render_box / render_font_size / overflow。"""
        out, _ = self.render_detailed(clean, blocks)
        return out

    def render_detailed(self, clean: np.ndarray, blocks: list[TextBlock]
                        ) -> tuple[np.ndarray, list[BlockRender]]:
        """同 render，另外返回每个块的排版明细。"""
        out = _as_bgr(clean)
        H, W = out.shape[:2]
        items = [b for b in (blocks or []) if isinstance(b, TextBlock)]
        if len(items) != len(blocks or []):
            log.warning("嵌字：忽略了 %d 个格式不对的文字块", len(blocks or []) - len(items))
        jobs: list[TextBlock] = []
        for b in items:
            text = normalize_text(b.translation if isinstance(b.translation, str) else "")
            if b.skip or not text:
                b.render_box, b.render_font_size, b.overflow = None, None, False
            else:
                jobs.append(b)
        if not jobs or H < 4 or W < 4:
            return out, []

        ref = page_ref(W, H)
        regions = [self._region(out, b) for b in jobs]          # 全部在画字之前求（只看去字图）
        try:
            partition_regions(regions, [_sane_text_box(b, W, H) for b in jobs])
        except Exception:  # noqa: BLE001 - 分区失败不影响单块排版
            log.exception("嵌字：区域分区失败，按各自区域排版")
        ests = self._font_estimates(items, jobs, ref)
        cache: dict[tuple, Optional[_Stamp]] = {}
        details: list[BlockRender] = []
        for b, reg, est in zip(jobs, regions, ests):
            try:
                details.append(self._render_block(out, b, reg, est, ref, cache))
            except Exception:  # noqa: BLE001 - 单块失败不能让整页失败
                log.exception("嵌字：第 %s 块排版失败，已跳过", b.id)
                b.render_box, b.render_font_size, b.overflow = None, None, True
                details.append(BlockRender(b.id, reg, None, "failed", "h", "", (0, 0)))
        return out, details

    # -- 区域 / 字号 ---------------------------------------------------------------------
    def _region(self, img: np.ndarray, b: TextBlock) -> Region:
        try:
            return compute_region(img, b, self.margin)
        except Exception:  # noqa: BLE001
            log.exception("嵌字：第 %s 块求气泡区域失败，退回文字框", b.id)
            free = TextBlock(id=b.id, box=b.box, kind="free")
            return compute_region(img, free, 0.0)

    @staticmethod
    def _font_estimates(all_blocks: list[TextBlock], jobs: list[TextBlock], ref: float) -> list[float]:
        """每块的原文字号估计；对白块用整页对白字号的中位数校正（同一页对白字号基本一致）。"""
        def est(b: TextBlock) -> Optional[float]:
            try:
                return estimate_src_font(b, ref)
            except Exception:  # noqa: BLE001
                return None

        dialog = [e for b in all_blocks if b.kind != "free" and not b.is_sfx and (e := est(b))]
        if not dialog:
            dialog = [e for b in all_blocks if not b.is_sfx and (e := est(b))]
        page = statistics.median(dialog) if dialog else 0.026 * ref
        out: list[float] = []
        for b in jobs:
            e = est(b)
            if b.is_sfx:
                out.append(e if e else 1.5 * page)
            elif e is None:
                out.append(page)
            else:
                out.append(min(max(e, 0.7 * page), 1.45 * page))
        return out

    # -- 样式 -----------------------------------------------------------------------------
    def _style(self, b: TextBlock, region: Region) -> _Style:
        ts = self.ts
        if b.font and self.fonts.exists(b.font):
            fid = b.font
        elif b.is_sfx:
            fid = ts.get("font_sfx")
        elif b.kind == "free":
            fid = ts.get("font_free")
        else:
            fid = ts.get("font_dialog")
        fid = self.fonts.resolve(fid if isinstance(fid, str) else None, self.target)
        bg = _rgb(region.bg) or WHITE
        loud = b.kind == "free" or b.is_sfx

        # 字色：人工指定 > 原文颜色（保证对比度）> 底色上对比度高的黑 / 白
        fill = _rgb(b.fill)
        src = _rgb(b.text_color) if self.keep_color else None
        if fill is None:
            if src is not None and (loud or contrast(src, bg) >= _MIN_TEXT_CONTRAST):
                fill = src
            else:
                fill = best_bw(bg)

        # 描边：气泡外 / 拟声词默认有；气泡内只有原文带描边（且保持原色）时才加细描边
        stroke: Optional[RGB] = None
        ratio = 0.0
        src_stroke = _rgb(b.stroke_color) if self.keep_color else None
        if loud:
            ratio = self.stroke_free * (1.25 if b.is_sfx else 1.0)
            stroke = src_stroke if src_stroke is not None and contrast(src_stroke, fill) >= _MIN_STROKE_CONTRAST \
                else best_bw(fill)
        elif src is not None and fill == src and src_stroke is not None \
                and contrast(src_stroke, fill) >= _MIN_STROKE_CONTRAST:
            ratio = 0.5 * self.stroke_free
            stroke = src_stroke
        # 人工覆盖
        user_stroke = _rgb(b.stroke)
        if user_stroke is not None:
            stroke = user_stroke
            if ratio <= 0:
                ratio = self.stroke_free or 0.12
        if b.stroke_width is not None:
            try:
                ratio = min(max(float(b.stroke_width), 0.0), 0.5)
            except (TypeError, ValueError):
                pass
            if ratio > 0 and stroke is None:
                stroke = best_bw(fill)
        if ratio <= 0:
            stroke = None
        return _Style(fid, fill, stroke, ratio)

    # -- 单块 -----------------------------------------------------------------------------
    def _render_block(self, out: np.ndarray, b: TextBlock, region: Region, est: float, ref: float,
                      cache: dict) -> BlockRender:
        style = self._style(b, region)
        direction = choose_direction(b, region, self.direction)
        text = normalize_text(b.translation)
        toks = build_tokens(text, direction, style.font, self.fonts, self.target, self.letter_spacing)
        align = b.align if b.align in ("center", "start", "end") else ("start" if direction == "v" else "center")
        ls = self.line_spacing + (0.1 if direction == "v" else 0.0)     # 竖排列距略宽
        params = LayoutParams(direction, ls, align,
                              reserve=lambda fs: style.stroke_px(fs) + 0.05 * fs + 0.5,
                              column_order=self.column_order)
        lo, hi = font_range(est, ref, self.ts, bool(b.is_sfx))
        fixed = None
        if b.font_size is not None:
            try:
                fixed = int(round(float(b.font_size))) or None
            except (TypeError, ValueError):
                fixed = None
        min_abs = max(5, int(round(0.004 * ref)))
        res = fit_block(region, toks, params, lo, hi, min_abs=min_abs, fixed=fixed)
        lay = res.layout
        sw = style.stroke_px(lay.fs)
        canvas = _Canvas(region.shape[0], region.shape[1], sw > 0)
        self._draw(canvas, lay, sw, cache)
        box, clipped = self._composite(out, region, canvas, style)
        b.render_box = box
        b.render_font_size = int(lay.fs)
        b.overflow = bool(res.overflow)
        if res.overflow:
            log.info("嵌字：第 %s 块译文过长，已缩小到 %dpx（下限 %dpx）", b.id, lay.fs, lo)
        return BlockRender(b.id, region, lay, res.stage, direction, style.font, (lo, hi), clipped, box)

    # -- 画字 -----------------------------------------------------------------------------
    def _stamp(self, cache: dict, font_id: str, size: int, text: str, sw: int, op: str = "",
               length: float = 0.0) -> Optional[_Stamp]:
        key = (font_id, size, text, sw, op, int(round(length)))
        if key in cache:
            return cache[key]
        st = _make_stamp(self.fonts.get(font_id, size), text, sw)
        if st is not None and op in ("stretch", "stretch_rot"):
            st = _stretch_x(st, length)
        if st is not None and op in ("rot", "stretch_rot"):
            st = _rotate_cw(st)
        cache[key] = st
        return st

    def _draw(self, cv: _Canvas, lay: Layout, sw: int, cache: dict) -> None:
        fs = int(lay.fs)
        v = lay.direction == "v"
        for pl in lay.placed:
            u = pl.unit
            k = u.kind
            if k == "sp":
                continue
            m = self.fonts.metrics(u.font)
            cx, cy = pl.x + pl.w / 2.0, pl.y + pl.h / 2.0
            if k == "c":
                st = self._stamp(cache, u.font, fs, u.text, sw)
                if st is None:
                    continue
                x = pl.x + (pl.w - m.advance(u.text) * fs) / 2.0 if v else pl.x
                base = cy - m.cjk_center_y * fs           # 汉字墨迹中心对齐格子中心
                cv.paste(st, x - st.ox, base - st.oy)
            elif k == "ci":
                st = self._stamp(cache, u.font, fs, u.text, sw)
                if st is not None:
                    cv.center(st, cx, cy)
            elif k == "tr":                                 # 竖排时没有竖排形的 ，。 放到格子右上
                st = self._stamp(cache, u.font, fs, u.text, sw)
                if st is not None:
                    cv.center(st, pl.x + 0.70 * pl.w, pl.y + 0.30 * min(pl.h, fs))
            elif k == "tcy":                                # 縦中横：缩到一格宽内横着写
                adv = sum(m.advance(c) for c in u.text)
                size = max(4, int(fs * min(1.0, 0.94 / max(adv, 1e-3))))
                st = self._stamp(cache, u.font, size, u.text, sw)
                if st is not None:
                    cv.center(st, cx, cy)
            elif k == "rot":
                st = self._stamp(cache, u.font, fs, u.text, sw, "rot")
                if st is not None:
                    cv.center(st, cx, cy)
            elif k == "ell":                                # 每个 … 占一格，墨迹居中（竖排旋转成竖点）
                st = self._stamp(cache, u.font, fs, "…", sw, "rot" if v else "")
                if st is None:
                    continue
                for i in range(len(u.text)):
                    if v:
                        cv.center(st, cx, pl.y + (i + 0.5) * fs)
                    else:
                        cv.center(st, pl.x + (i + 0.5) * fs, cy)
            elif k == "dash":                               # 破折号：拉成一条连续的线
                length = max(1.0, (pl.h if v else pl.w) - 0.1 * fs)
                st = self._stamp(cache, u.font, fs, "—", sw, "stretch_rot" if v else "stretch", length)
                if st is not None:
                    cv.center(st, cx, cy)
            else:                                           # 未知类型：按墨迹居中兜底
                st = self._stamp(cache, u.font, fs, u.text, sw)
                if st is not None:
                    cv.center(st, cx, cy)

    @staticmethod
    def _composite(out: np.ndarray, region: Region, cv: _Canvas, style: _Style) -> tuple[Optional[Box], int]:
        """图层 × 区域 mask → alpha 合成到页面。返回 (页面上的实际墨迹框, 被裁掉的像素数)。"""
        hard = region.hard
        cover = cv.stroke if cv.stroke is not None else cv.fill
        clipped = int(np.count_nonzero((cover > 0) & ~hard)) + cv.lost
        fill = np.where(hard, cv.fill, 0).astype(np.uint8)
        stroke = np.where(hard, cv.stroke, 0).astype(np.uint8) if cv.stroke is not None else None
        cov = stroke if stroke is not None else fill
        rows = np.flatnonzero(cov.any(axis=1))
        cols = np.flatnonzero(cov.any(axis=0))
        if rows.size == 0:
            return None, clipped
        y0, y1, x0, x1 = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
        rx, ry = region.roi[0], region.roi[1]
        dst = out[ry + y0:ry + y1, rx + x0:rx + x1]
        acc = dst.astype(np.float32)
        if stroke is not None and style.stroke is not None:
            a = stroke[y0:y1, x0:x1, None].astype(np.float32) * (1.0 / 255.0)
            col = np.array(style.stroke[::-1], np.float32)          # RGB → BGR
            acc = acc * (1.0 - a) + col * a
        a = fill[y0:y1, x0:x1, None].astype(np.float32) * (1.0 / 255.0)
        col = np.array(style.fill[::-1], np.float32)
        acc = acc * (1.0 - a) + col * a
        dst[...] = np.clip(acc + 0.5, 0, 255).astype(np.uint8)
        return (rx + x0, ry + y0, rx + x1, ry + y1), clipped
