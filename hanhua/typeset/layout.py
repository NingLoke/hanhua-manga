"""排版几何：求气泡内部区域（mask）→ 切分文字（禁则）→ 按 mask 逐行 / 逐列可用长度排版 → 二分字号。

坐标：本模块内部全部用 ROI 坐标（区域裁剪框左上角为原点），Region.roi 给出它在页面上的位置。
单位：Unit / Token 的长度是 em（乘字号得到像素），这样二分字号时不用为每个字号重新量字。
"""
from __future__ import annotations

import logging
import math
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Callable, Optional

import cv2
import numpy as np

from ..types import Box, TextBlock, box_area, box_contain_ratio, clamp_box
from .fonts import FontRegistry

log = logging.getLogger("hanhua.typeset")

RGB = tuple[int, int, int]

# ======================================================================================
# 区域（气泡内部 mask）
# ======================================================================================
_FIXED_TOL = 56       # 与底色的最大色差（超过即视为屏障：描边 / 背景 / 人物）
_FLOAT_TOL = 9        # flood fill 相邻像素的色差（跟随渐变，遇到描边的陡变就停）
_SEED_TOL = 30        # 种子像素与底色的最大色差


@dataclass
class Region:
    roi: Box                       # 区域在页面上的裁剪框
    hard: np.ndarray               # bool (h, w)：字（含描边）绝不能越出的范围
    dist: np.ndarray               # float32：hard 内每点到边界的距离（px）
    center: tuple[float, float]    # 排版中心 (x, y)，ROI 坐标
    margin: float                  # 留白（px），排版时从边界再内缩这么多
    method: str                    # fill | fill_ellipse | ellipse | fill_nobox | rect
    bg: RGB                        # 区域底色（RGB）

    @property
    def shape(self) -> tuple[int, int]:
        return self.hard.shape[:2]

    @property
    def short_side(self) -> float:
        return 2.0 * float(self.dist.max()) if self.dist.size else 0.0

    def restrict(self, keep: np.ndarray) -> None:
        """把区域限制在 keep 内（多个块共用一个气泡时分区用），重算距离和中心。"""
        hard = self.hard & keep
        if hard.sum() < 16:
            return
        self.hard = hard
        self.dist = _dist(hard)
        self.center = _center(hard, self.dist)


def _dist(hard: np.ndarray) -> np.ndarray:
    """到区域边界的距离；ROI 的外框也当作边界（区域在 ROI 外没有像素）。"""
    m = np.zeros((hard.shape[0] + 2, hard.shape[1] + 2), np.uint8)
    m[1:-1, 1:-1] = hard
    return cv2.distanceTransform(m, cv2.DIST_L2, 3)[1:-1, 1:-1]


def _center(hard: np.ndarray, dist: np.ndarray) -> tuple[float, float]:
    """距离变换最大值附近（≥0.85×max 的脊线 / 平台）里离质心最近的点。"""
    mx = float(dist.max())
    if mx <= 0:
        h, w = hard.shape
        return (w / 2.0, h / 2.0)
    mom = cv2.moments(hard.astype(np.uint8), binaryImage=True)
    if mom["m00"] > 0:
        gx, gy = mom["m10"] / mom["m00"], mom["m01"] / mom["m00"]
    else:
        gx, gy = hard.shape[1] / 2.0, hard.shape[0] / 2.0
    ys, xs = np.nonzero(dist >= 0.85 * mx)
    i = int(np.argmin((xs - gx) ** 2 + (ys - gy) ** 2))
    return (float(xs[i]) + 0.5, float(ys[i]) + 0.5)


def _base_color(crop: np.ndarray, box: Box) -> np.ndarray:
    """文字框内（去字后应是气泡底色）的中位色，BGR int16。"""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    sx, sy = int(w * 0.15), int(h * 0.15)
    patch = crop[y1 + sy:max(y1 + sy + 1, y2 - sy), x1 + sx:max(x1 + sx + 1, x2 - sx)].reshape(-1, 3)
    if patch.size == 0:
        patch = crop.reshape(-1, 3)
    if len(patch) > 4000:
        patch = patch[:: len(patch) // 4000 + 1]
    return np.median(patch, axis=0).astype(np.int16)


def _flood(crop: np.ndarray, sbox: Box, base: np.ndarray) -> np.ndarray:
    """从文字框内接近底色的像素出发做浮动范围 flood fill（跟随渐变），并用固定色差做屏障防漂移。"""
    h, w = crop.shape[:2]
    diff = np.abs(crop.astype(np.int16) - base[None, None, :]).max(axis=2)
    mask = np.zeros((h + 2, w + 2), np.uint8)
    mask[1:-1, 1:-1][diff > _FIXED_TOL] = 1
    flags = 4 | cv2.FLOODFILL_MASK_ONLY | (2 << 8)
    img = np.ascontiguousarray(crop)
    tol = (_FLOAT_TOL, _FLOAT_TOL, _FLOAT_TOL)
    x1, y1, x2, y2 = sbox
    for yf in np.linspace(y1, y2 - 1, 7):
        for xf in np.linspace(x1, x2 - 1, 7):
            xi, yi = int(round(xf)), int(round(yf))
            if not (0 <= xi < w and 0 <= yi < h) or mask[yi + 1, xi + 1] != 0 or diff[yi, xi] > _SEED_TOL:
                continue
            cv2.floodFill(img, mask, (xi, yi), 0, tol, tol, flags)
    return mask[1:-1, 1:-1] == 2


def _clean_fill(fill: np.ndarray, sbox: Box) -> np.ndarray:
    """补洞（残留笔画 / 装饰）、断开细缝漏出，只保留与文字框重叠最多的连通块。"""
    h, w = fill.shape
    f = cv2.morphologyEx(fill.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(f, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    f = np.zeros_like(f)
    if cnts:
        cv2.drawContours(f, cnts, -1, 1, thickness=-1)
    k = max(3, int(round(0.02 * min(h, w))) | 1)
    f = cv2.morphologyEx(f, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, lab = cv2.connectedComponents(f, connectivity=4)
    if n <= 2:
        return f.astype(bool)
    x1, y1, x2, y2 = sbox
    sub = lab[y1:y2, x1:x2]
    counts = np.bincount(sub.ravel(), minlength=n)
    counts[0] = 0
    best = int(np.argmax(counts)) if counts.max() > 0 else int(np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    return lab == best


def _ellipse(h: int, w: int, box: Box, inset: float) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    x1, y1, x2, y2 = box
    ax = max(1, int((x2 - x1) / 2 - inset))
    ay = max(1, int((y2 - y1) / 2 - inset))
    cv2.ellipse(m, (int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2))), (ax, ay), 0, 0, 360, 1, -1)
    return m.astype(bool)


def _border_touch(mask: np.ndarray, roi: Box, page_w: int, page_h: int) -> float:
    """区域贴着 ROI 外框的比例（与页面边缘重合的边不算：气泡被页面截断是正常的）。"""
    h, w = mask.shape
    sides = []
    if roi[1] > 0:
        sides.append(mask[0, :])
    if roi[3] < page_h:
        sides.append(mask[-1, :])
    if roi[0] > 0:
        sides.append(mask[:, 0])
    if roi[2] < page_w:
        sides.append(mask[:, -1])
    if not sides:
        return 0.0
    edge = np.concatenate(sides)
    return float(edge.mean()) if edge.size else 0.0


def _local(box: Box, roi: Box) -> Box:
    w, h = roi[2] - roi[0], roi[3] - roi[1]
    return clamp_box((box[0] - roi[0], box[1] - roi[1], box[2] - roi[0], box[3] - roi[1]), w, h)


def _try_fill(img: np.ndarray, tb: Box, roi: Box) -> tuple[Optional[np.ndarray], RGB, float]:
    """在 roi 内 flood fill，返回 (区域 mask 或 None, 底色 RGB, 贴边比例)。"""
    H, W = img.shape[:2]
    crop = img[roi[1]:roi[3], roi[0]:roi[2]]
    lt = _local(tb, roi)
    if box_area(lt) <= 0 or crop.size == 0:
        return None, (255, 255, 255), 1.0
    base = _base_color(crop, lt)
    bg: RGB = (int(base[2]), int(base[1]), int(base[0]))
    fill = _flood(crop, lt, base)
    if fill.sum() < 16:
        return None, bg, 1.0
    fill = _clean_fill(fill, lt)
    return fill, bg, _border_touch(fill, roi, W, H)


def _xywh_expand(box: Box, fx: float, fy: float, w: int, h: int) -> Box:
    bw, bh = box[2] - box[0], box[3] - box[1]
    px, py = int(round(bw * fx)), int(round(bh * fy))
    return clamp_box((box[0] - px, box[1] - py, box[2] + px, box[3] + py), w, h)


def _sane_text_box(block: TextBlock, w: int, h: int) -> Box:
    tb = clamp_box(block.box, w, h)
    if box_area(tb) > 0:
        return tb
    # 框退化或完全在页外：以夹到页内的中心点为准给一个最小框
    cx = min(max((block.box[0] + block.box[2]) / 2.0, 0), w - 1)
    cy = min(max((block.box[1] + block.box[3]) / 2.0, 0), h - 1)
    r = max(4.0, 0.01 * max(w, h))
    return clamp_box((int(cx - r), int(cy - r), int(cx + r) + 1, int(cy + r) + 1), w, h)


def compute_region(img: np.ndarray, block: TextBlock, margin_ratio: float) -> Region:
    """块 → 可排字区域。气泡块在去字后的图上求气泡内部；气泡外块用文字框略放大的矩形。"""
    H, W = img.shape[:2]
    tb = _sane_text_box(block, W, H)
    margin_ratio = min(max(float(margin_ratio), 0.0), 0.35)

    if block.kind != "free":
        bb = clamp_box(block.bubble_box, W, H) if block.bubble_box else None
        if bb is not None and (box_area(bb) <= 0 or box_contain_ratio(tb, bb) < 0.3):
            bb = None
        reg = _bubble_region(img, tb, bb, W, H) if bb is not None else _nobox_region(img, tb, W, H)
        if reg is not None:
            hard, roi, method, bg = reg
            dist = _dist(hard)
            margin = margin_ratio * 2.0 * float(dist.max()) if method != "rect" else 0.0
            return Region(roi, hard, dist, _center(hard, dist), margin, method, bg)

    # free 块（或气泡块的所有办法都失败）：文字框外扩约 8% 的矩形
    roi = _xywh_expand(tb, 0.08, 0.08, W, H)
    hard = np.ones((roi[3] - roi[1], roi[2] - roi[0]), bool)
    crop = img[roi[1]:roi[3], roi[0]:roi[2]].reshape(-1, 3)
    med = np.median(crop[:: max(1, len(crop) // 4000)], axis=0) if crop.size else np.array([255, 255, 255])
    dist = _dist(hard)
    return Region(roi, hard, dist, _center(hard, dist), 0.0, "rect", (int(med[2]), int(med[1]), int(med[0])))


def _bubble_region(img: np.ndarray, tb: Box, bb: Box, W: int, H: int
                   ) -> Optional[tuple[np.ndarray, Box, str, RGB]]:
    bw, bh = bb[2] - bb[0], bb[3] - bb[1]
    roi = _xywh_expand(bb, 0.04, 0.04, W, H)
    fill, bg, touch = _try_fill(img, tb, roi)
    rh, rw = roi[3] - roi[1], roi[2] - roi[0]
    inset = max(2.0, 0.035 * min(bw, bh))
    ell = _ellipse(rh, rw, _local(bb, roi), inset)
    box_area_ = float(bw * bh)

    if fill is not None and touch > 0.01:
        # 可能是检测框偏紧：放大搜索范围再试一次，闭合且面积合理就接受
        roi2 = _xywh_expand(bb, 0.25, 0.25, W, H)
        fill2, bg2, touch2 = _try_fill(img, tb, roi2)
        if fill2 is not None and touch2 <= 0.01 and fill2.sum() <= 1.6 * box_area_:
            ell2 = _ellipse(roi2[3] - roi2[1], roi2[2] - roi2[0], _local(bb, roi2), inset)
            if fill2.sum() >= 0.3 * ell2.sum():
                return fill2, roi2, "fill", bg2
        # 漏出：限制在内切椭圆里（相似色区域 ∩ 椭圆）
        inter = fill & ell
        if inter.sum() >= 0.5 * ell.sum():
            return inter, roi, "fill_ellipse", bg
        return ell, roi, "ellipse", bg

    if fill is not None:
        area = float(fill.sum())
        lt = _local(tb, roi)
        tsub = fill[lt[1]:lt[3], lt[0]:lt[2]]
        contain = float(tsub.mean()) if tsub.size else 0.0
        if area > 1.05 * box_area_:
            inter = fill & ell
            return (inter, roi, "fill_ellipse", bg) if inter.sum() >= 0.5 * ell.sum() else (ell, roi, "ellipse", bg)
        if area >= 0.3 * float(ell.sum()) and contain >= 0.5:
            return fill, roi, "fill", bg
    return ell, roi, "ellipse", bg


def _nobox_region(img: np.ndarray, tb: Box, W: int, H: int
                  ) -> Optional[tuple[np.ndarray, Box, str, RGB]]:
    """检测器没给气泡框：在文字框周围找闭合的气泡；找不到就用文字框外扩的矩形。"""
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    pad = int(round(0.6 * max(tw, th))) + 4
    roi = clamp_box((tb[0] - pad, tb[1] - pad, tb[2] + pad, tb[3] + pad), W, H)
    fill, bg, touch = _try_fill(img, tb, roi)
    if fill is not None and touch <= 0.01 and fill.sum() >= 1.3 * box_area(tb):
        return fill, roi, "fill_nobox", bg
    rect = _xywh_expand(tb, 0.10, 0.10, W, H)
    hard = np.ones((rect[3] - rect[1], rect[2] - rect[0]), bool)
    return hard, rect, "rect", bg


# ======================================================================================
# 文字 → Unit / Token（禁则）
# ======================================================================================
# 行首禁则（不能出现在行首 / 列首）
NO_START = set("，。、．,.；;：:！!？?）)］]｝}」』】〕〗〙〛〉》”’…‥⋯—―～〜ー・·•%‰℃°′″々〻"
               "ぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮヵヶ")
# 行尾禁则（不能出现在行尾 / 列尾）
NO_END = set("（(［[｛{「『【〔〖〘〚〈《“‘")

# 竖排标点 → 竖排形（presentation forms）
VFORMS: dict[str, str] = {
    "，": "︐", ",": "︐", "、": "︑", "。": "︒", ".": "︒", "．": "︒",
    "：": "︓", ":": "︓", "；": "︔", ";": "︔",
    "「": "﹁", "」": "﹂", "『": "﹃", "』": "﹄", "“": "﹁", "”": "﹂", "‘": "﹃", "’": "﹄",
    "（": "︵", "）": "︶", "(": "︵", ")": "︶", "｛": "︷", "｝": "︸", "{": "︷", "}": "︸",
    "〔": "︹", "〕": "︺", "【": "︻", "】": "︼", "《": "︽", "》": "︾", "〈": "︿", "〉": "﹀",
    "［": "﹇", "］": "﹈", "[": "﹇", "]": "﹈",
}
_TOP_RIGHT = set("，,、。.．")              # 没有竖排形时：横排字形平移到格子右上
_ROTATE = set("～〜ー-–‐_/|｜")            # 竖排时旋转 90°
_EXCL = set("!?！？")
_ELLIPSIS = set("…⋯‥")
_DASH = set("—―─")
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[.,'’\-:/][A-Za-z0-9]+)*")
_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿­"), None)
_HALF_EXCL = str.maketrans("！？", "!?")


@dataclass
class Unit:
    """一个绘制单元。kind：
    c 普通字（横排按 advance 摆，竖排在格子里水平居中）；ci 墨迹居中；tr 竖排右上标点；
    rot 旋转 90°；tcy 纵中横；ell 省略号；dash 破折号线；sp 空白。"""

    text: str
    kind: str
    font: str
    adv: float        # 沿行方向的长度（em）
    gap: float        # 之后的字距（em）


@dataclass
class Token:
    units: list[Unit]
    length: float     # em（含内部字距）
    gap: float        # 最后一个 unit 之后的字距（em）
    space: bool = False
    brk: bool = False
    splittable: bool = False

    @property
    def visible(self) -> int:
        return sum(1 for u in self.units if u.kind in ("c", "tcy", "rot") and u.text not in NO_START)


def normalize_text(s: str) -> str:
    """去零宽 / 控制字符，统一换行，... → …，压缩空白，去掉空行。"""
    s = (s or "").replace("\r\n", "\n").replace("\r", "\n").translate(_ZERO_WIDTH)
    s = "".join(ch for ch in s if ch == "\n" or not unicodedata.category(ch).startswith("C"))
    s = re.sub(r"\.{3,}", "…", s)
    s = re.sub(r"[ \t　 ]+", " ", s)
    return "\n".join(ln.strip() for ln in s.split("\n") if ln.strip())


def _clusters(text: str) -> list[tuple[str, str]]:
    """切成 (类型, 文本)：brk / sp / word / ell / dash / excl / ch。"""
    out: list[tuple[str, str]] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch == "\n":
            out.append(("brk", ch))
            i += 1
            continue
        if ch.isspace():
            j = i
            while j < n and text[j].isspace() and text[j] != "\n":
                j += 1
            out.append(("sp", " "))
            i = j
            continue
        m = _WORD_RE.match(text, i)
        if m and ch.isascii():
            out.append(("word", m.group()))
            i = m.end()
            continue
        for kind, chars in (("ell", _ELLIPSIS), ("dash", _DASH), ("excl", _EXCL)):
            if ch in chars:
                j = i
                while j < n and text[j] in chars:
                    j += 1
                out.append((kind, text[i:j]))
                i = j
                break
        else:
            out.append(("ch", ch))
            i += 1
    return out


class _UnitBuilder:
    """把 cluster 变成 Unit（选字体、缺字回退、竖排标点形 / 旋转 / 纵中横）。"""

    def __init__(self, fonts: FontRegistry, font_id: str, target: str, direction: str,
                 letter_spacing: float) -> None:
        self.fonts = fonts
        self.font = font_id
        self.chain = fonts.fallback_chain(font_id, target)
        self.v = direction == "v"
        self.ls = letter_spacing

    def pick(self, text: str) -> Optional[str]:
        for f in self.chain:
            if self.fonts.has_glyph(f, text):
                return f
        return None

    def font_for(self, text: str) -> str:
        return self.pick(text) or self.font

    def adv(self, font: str, text: str) -> float:
        m = self.fonts.metrics(font)
        return sum(m.advance(c) for c in text)

    def units(self, kind: str, text: str, split_words: bool) -> list[Unit]:
        ls = self.ls
        if kind == "sp":
            return [Unit(" ", "sp", self.font, 0.5 if self.v else max(0.25, self.adv(self.font, " ")), 0.0)]
        if kind == "ell":
            n = max(1, len(text))
            f = self.font_for("…")
            return [Unit("…" * n, "ell", f, float(n), ls)]
        if kind == "dash":
            n = max(1, len(text))
            return [Unit("—" * n, "dash", self.font_for("—"), float(n), ls)]
        if kind == "excl":
            return self._excl(text)
        if kind == "word":
            return self._word(text, split_words)
        return [self._char(text)]

    def _char(self, ch: str) -> Unit:
        ls = self.ls
        if not self.v:
            f = self.font_for(ch)
            return Unit(ch, "c", f, self.adv(f, ch), ls)
        vf = VFORMS.get(ch)
        if vf is not None:
            if self.fonts.has_glyph(self.font, vf):
                return Unit(vf, "c", self.font, 1.0, ls)
            if self.fonts.has_glyph(self.font, ch):
                return Unit(ch, "tr" if ch in _TOP_RIGHT else "rot", self.font, 1.0, ls)
            f = self.pick(vf)
            if f is not None:
                return Unit(vf, "c", f, 1.0, ls)
            return Unit(ch, "tr" if ch in _TOP_RIGHT else "rot", self.font_for(ch), 1.0, ls)
        if ch in _ROTATE:
            return Unit(ch, "rot", self.font_for(ch), 1.0, ls)
        f = self.font_for(ch)
        eaw = unicodedata.east_asian_width(ch)
        cat = unicodedata.category(ch)
        if eaw in ("W", "F") and cat.startswith("L"):
            return Unit(ch, "c", f, 1.0, ls)          # 汉字 / 假名 / 全角字母：正常直立
        return Unit(ch, "ci", f, 1.0, ls)             # 其它符号（！ ♡ · 等）：按墨迹居中

    def _excl(self, text: str) -> list[Unit]:
        ls = self.ls
        if not self.v:
            out = []
            for ch in text:
                f = self.font_for(ch)
                out.append(Unit(ch, "c", f, self.adv(f, ch), ls))
            return out
        out = []
        i = 0
        while i < len(text):
            pair = text[i:i + 2]
            if len(pair) == 2:      # 「！？」「！！」合成一格（纵中横）
                half = pair.translate(_HALF_EXCL)
                out.append(Unit(half, "tcy", self.font_for(half), 1.0, ls))
                i += 2
            else:
                out.append(Unit(pair, "ci", self.font_for(pair), 1.0, ls))
                i += 1
        return out

    def _word(self, word: str, split: bool) -> list[Unit]:
        ls = self.ls
        if not self.v:
            out = []
            for k, ch in enumerate(word):
                f = self.font_for(ch)
                out.append(Unit(ch, "c", f, self.adv(f, ch), ls if k == len(word) - 1 else 0.0))
            return out
        if len(word) <= 2:
            return [Unit(word, "tcy", self.font_for(word), 1.0, ls)]
        if split:
            return [Unit(ch, "tcy", self.font_for(ch), 1.0, ls) for ch in word]
        f = self.font_for(word)
        return [Unit(word, "rot", f, self.adv(f, word) + 0.1, ls)]


def _finish(units: list[Unit], **flags: bool) -> Token:
    length = sum(u.adv for u in units) + sum(u.gap for u in units[:-1])
    return Token(units, length, units[-1].gap if units else 0.0, **flags)


def build_tokens(text: str, direction: str, font_id: str, fonts: FontRegistry, target: str,
                 letter_spacing: float, split_words: bool = False) -> list[Token]:
    """按禁则把文字切成不可再分的 token：行首禁则字粘前一个，行尾禁则字粘后一个，……/—— 不拆。"""
    ub = _UnitBuilder(fonts, font_id, target, direction, letter_spacing)
    tokens: list[Token] = []
    cur: list[Unit] = []
    cur_split = False
    glue_next = False
    prev_kind = ""
    for kind, s in _clusters(text):
        if kind == "brk":
            if cur:
                tokens.append(_finish(cur, splittable=cur_split))
                cur = []
            tokens.append(Token([], 0.0, 0.0, brk=True))
            glue_next, prev_kind = False, kind
            continue
        if kind == "sp":
            if cur:
                tokens.append(_finish(cur, splittable=cur_split))
                cur = []
            tokens.append(_finish(ub.units(kind, s, split_words), space=True))
            glue_next, prev_kind = False, kind
            continue
        first, last = s[0], s[-1]
        if first in ('"', "'"):        # 直引号：按位置判断开 / 闭
            opening = prev_kind in ("", "sp", "brk") or glue_next
            first = last = "“" if opening else "”"
        units = ub.units(kind, s, split_words)
        if not units:
            continue
        if cur and (glue_next or first in NO_START):
            cur.extend(units)
        elif not cur and first in NO_START and len(tokens) >= 2 and tokens[-1].space and not tokens[-2].brk \
                and not tokens[-2].space:
            tokens.pop()                 # 「你好 。」：去掉空格，标点粘回前一个 token
            cur = tokens.pop().units + units
        else:
            if cur:
                tokens.append(_finish(cur, splittable=cur_split))
            cur = list(units)
            cur_split = False
        if kind == "word" and not ub.v and len(s) > 1:
            cur_split = True
        glue_next = last in NO_END
        prev_kind = kind
    if cur:
        tokens.append(_finish(cur, splittable=cur_split))
    return tokens


def split_long_words(tokens: list[Token]) -> list[Token]:
    """超长英文单词拆成单字 token（横排兜底用）。"""
    out: list[Token] = []
    for t in tokens:
        if t.splittable and len(t.units) > 1:
            out.extend(_finish([u]) for u in t.units)
        else:
            out.append(t)
    return out


# ======================================================================================
# 行 / 列排布
# ======================================================================================
@dataclass
class Placed:
    unit: Unit
    x: float          # 单元框左上角（ROI 坐标）
    y: float
    w: float          # 单元框大小（px）
    h: float


@dataclass
class Layout:
    fs: int
    direction: str
    placed: list[Placed]
    lines: int
    bbox: tuple[float, float, float, float]
    overflow: bool = False
    forced: bool = False


@dataclass
class LayoutParams:
    direction: str = "h"
    line_spacing: float = 1.12
    align: str = "center"
    reserve: Callable[[int], float] = field(default=lambda fs: 0.0)   # 额外内缩（如描边宽度）
    column_order: str = "rtl"


def _greedy(toks: list[Token], caps: list[float]) -> Optional[list[tuple[int, int]]]:
    """按每行容量（em）贪心填 token。返回每行 (起, 止) token 下标；放不下返回 None。"""
    n = len(toks)
    i = 0
    lines: list[tuple[int, int]] = []
    li = 0
    while i < n:
        while i < n and (toks[i].space or toks[i].brk):
            i += 1
        if i >= n:
            break
        if li >= len(caps):
            return None
        cap = caps[li] + 1e-6
        start = i
        used = 0.0
        prev_gap = 0.0
        end = i
        while i < n:
            t = toks[i]
            if t.brk:
                i += 1
                break
            add = t.length + (prev_gap if i > start else 0.0)
            if used + add <= cap:
                used += add
                prev_gap = t.gap
                i += 1
                if not t.space:
                    end = i
                continue
            if i == start:
                return None
            break
        lines.append((start, end))
        li += 1
    return lines


def _line_len(toks: list[Token], a: int, b: int) -> float:
    total = 0.0
    prev_gap = 0.0
    for k in range(a, b):
        t = toks[k]
        total += t.length + (prev_gap if k > a else 0.0)
        prev_gap = t.gap
    return total


class _Geometry:
    """某个留白阈值下的可用区域，按排版方向转成「行带」查询。"""

    def __init__(self, region: Region, thr: float, direction: str) -> None:
        S = region.dist > thr
        cx, cy = region.center
        if direction == "v":
            S = S.T                      # 行 = x（列带），沿行方向 = y
            self.cm, self.cc = cy, cx    # 沿行方向的中心、垂直方向的中心
        else:
            self.cm, self.cc = cx, cy
        self.ok = bool(S.any())
        self.rows = S.shape[0]
        if not self.ok:
            return
        c = int(min(max(round(self.cm - 0.5), 0), S.shape[1] - 1))
        if not S[:, c].any():            # 中心线不在可用区内：换到可用像素最多的一列
            c = int(np.argmax(S.sum(axis=0)))
        self.c = c
        left = S[:, c::-1]
        right = S[:, c:]
        lf = np.argmin(left, axis=1)
        lf[left.all(axis=1)] = left.shape[1]
        rf = np.argmin(right, axis=1)
        rf[right.all(axis=1)] = right.shape[1]
        L = (c + 1 - lf).astype(np.int32)
        R = (c + rf).astype(np.int32)
        bad = ~S[:, c]
        L[bad] = c
        R[bad] = c
        self.L, self.R = L, R
        usable = np.nonzero(~bad)[0]
        self.lo = int(usable[0]) if usable.size else 0
        self.hi = int(usable[-1]) + 1 if usable.size else 0
        self._bands: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def bands(self, t: int) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """厚度 t 的所有行带：起点 r 的带可用区间 [Lb[r], Rb[r])。"""
        if t > self.rows or t <= 0:
            return None
        b = self._bands.get(t)
        if b is None:
            win = np.lib.stride_tricks.sliding_window_view
            b = (win(self.L, t).max(axis=1), win(self.R, t).min(axis=1))
            self._bands[t] = b
        return b


def _fit_at(toks: list[Token], geo: _Geometry, fs: int, p: LayoutParams) -> Optional[Layout]:
    if not geo.ok or not toks:
        return None
    t = max(1, int(fs))
    bands = geo.bands(t)
    if bands is None:
        return None
    Lb, Rb = bands
    caps_all = (Rb - Lb).clip(min=0)
    maxcap = float(caps_all.max()) / fs if caps_all.size else 0.0
    if maxcap <= 0:
        return None
    total = sum(tk.length for tk in toks if not tk.brk)
    hard_breaks = sum(1 for tk in toks if tk.brk)
    pitch = fs * p.line_spacing
    span = geo.hi - geo.lo
    n_max = int((span - t) / pitch) + 1 if span >= t else 0
    n_min = max(1, int(math.ceil(total / maxcap - 1e-9)), hard_breaks + 1)
    nb = len(Rb)
    rev = p.direction == "v" and p.column_order != "ltr"
    for n in range(n_min, n_max + 1):
        extent = (n - 1) * pitch + t
        top0 = geo.cc - extent / 2.0
        for off in (0.0, -0.25, 0.25, -0.5, 0.5):
            tops = [int(round(top0 + off * pitch + i * pitch)) for i in range(n)]
            if tops[0] < 0 or tops[-1] >= nb:
                continue
            caps_px = [float(caps_all[r]) for r in tops]
            if min(caps_px) <= 0:
                continue
            order = tops[::-1] if rev else tops
            caps = [c / fs for c in (caps_px[::-1] if rev else caps_px)]
            lines = _greedy(toks, caps)
            if lines is None:
                continue
            lines = _balance(toks, caps, lines)
            if len(lines) < n:      # 实际用的行数比预留的少：按实际行数重新居中
                order = _recenter(toks, lines, caps_all, geo.cc, pitch, t, fs, rev) or order
            return _place(toks, lines, order, Lb, Rb, geo.cm, fs, p)
    return None


def _recenter(toks: list[Token], lines: list[tuple[int, int]], caps_all: np.ndarray, cc: float,
              pitch: float, t: int, fs: int, rev: bool) -> Optional[list[int]]:
    """按实际行数 m 重新把整段文字放到垂直方向的中心；新位置放不下就返回 None（保持原位）。"""
    m = len(lines)
    if m <= 0:
        return None
    top0 = cc - ((m - 1) * pitch + t) / 2.0
    tops = [int(round(top0 + i * pitch)) for i in range(m)]
    if tops[0] < 0 or tops[-1] >= len(caps_all):
        return None
    order = tops[::-1] if rev else tops
    for k, (a, b) in enumerate(lines):
        if _line_len(toks, a, b) * fs > float(caps_all[order[k]]) + 1e-6:
            return None
    return order


def _balance(toks: list[Token], caps: list[float], lines: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """在行数不变的前提下按比例收紧每行容量，让各行长度跟随气泡形状均匀分布；再处理孤字。"""
    n = len(caps)
    if n <= 1:
        return lines
    lo, hi = 0.0, 1.0
    best = lines
    for _ in range(9):
        mid = (lo + hi) / 2.0
        r = _greedy(toks, [c * mid for c in caps])
        if r is not None and len(r) <= n:
            hi, best = mid, r
        else:
            lo = mid
    return _fix_orphan(toks, caps, best)


def _fix_orphan(toks: list[Token], caps: list[float], lines: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """最后一行 / 列只有 1 个字时，从上一行挪一个 token 下来（放得下的话）。"""
    if len(lines) < 2:
        return lines
    a, b = lines[-1]
    vis = sum(toks[k].visible for k in range(a, b))
    pa, pb = lines[-2]
    if vis > 1 or pb - pa < 2:
        return lines
    prev_vis = sum(toks[k].visible for k in range(pa, pb))
    if prev_vis < 3:
        return lines
    k = pb - 1
    while k > pa and toks[k].space:
        k -= 1
    if k <= pa:
        return lines
    new_last = (k, b)
    new_prev_end = k
    while new_prev_end > pa and toks[new_prev_end - 1].space:
        new_prev_end -= 1
    if _line_len(toks, *new_last) <= caps[len(lines) - 1] + 1e-6:
        return lines[:-2] + [(pa, new_prev_end), new_last]
    return lines


def _line_starts(lengths: list[float], bands: list[tuple[float, float]], cm: float, align: str) -> list[float]:
    """每行（列）的起点（沿行方向，px）。

    center：每行各自以区域中心居中；start / end：所有行共用起边 / 止边（整段仍以中心为准）；
    没有对所有行都可行的共用边时，以最长一行居中的位置为共用边，各行再夹回自己的可用区间 [lb, rb)。
    """
    def clamp(s: float, lb: float, rb: float, ln: float) -> float:
        return min(max(s, lb), max(lb, rb - ln))

    if lengths and align in ("start", "end"):
        mx = max(lengths)
        if align == "start":
            lo_f = max(lb for lb, _ in bands)
            hi_f = min(rb - ln for (_, rb), ln in zip(bands, lengths))
            s0 = cm - mx / 2.0
            if lo_f <= hi_f + 1e-6:
                s0 = min(max(s0, lo_f), max(lo_f, hi_f))
            return [clamp(s0, lb, rb, ln) for (lb, rb), ln in zip(bands, lengths)]
        lo_f = max(lb + ln for (lb, _), ln in zip(bands, lengths))
        hi_f = min(rb for _, rb in bands)
        e0 = cm + mx / 2.0
        if lo_f <= hi_f + 1e-6:
            e0 = min(max(e0, lo_f), max(lo_f, hi_f))
        return [clamp(e0 - ln, lb, rb, ln) for (lb, rb), ln in zip(bands, lengths)]
    return [clamp(cm - ln / 2.0, lb, rb, ln) for (lb, rb), ln in zip(bands, lengths)]


def _place(toks: list[Token], lines: list[tuple[int, int]], order: list[int], Lb: np.ndarray, Rb: np.ndarray,
           cm: float, fs: int, p: LayoutParams) -> Layout:
    placed: list[Placed] = []
    v = p.direction == "v"
    x0 = y0 = math.inf
    x1 = y1 = -math.inf
    rows = [order[k] if k < len(order) else order[-1] for k in range(len(lines))]
    lengths = [_line_len(toks, a, b) * fs for a, b in lines]
    starts = _line_starts(lengths, [(float(Lb[r]), float(Rb[r])) for r in rows], cm, p.align)
    for k, (a, b) in enumerate(lines):
        r = rows[k]
        pos = starts[k]
        prev_gap = 0.0
        for ti in range(a, b):
            tk = toks[ti]
            if ti > a:
                pos += prev_gap * fs
            for ui, u in enumerate(tk.units):
                ext = u.adv * fs
                if v:
                    pl = Placed(u, float(r), pos, float(fs), ext)
                else:
                    pl = Placed(u, pos, float(r), ext, float(fs))
                if u.kind != "sp":
                    placed.append(pl)
                    x0, y0 = min(x0, pl.x), min(y0, pl.y)
                    x1, y1 = max(x1, pl.x + pl.w), max(y1, pl.y + pl.h)
                pos += ext + (u.gap * fs if ui < len(tk.units) - 1 else 0.0)
            prev_gap = tk.gap
    if not placed:
        x0 = y0 = x1 = y1 = 0.0
    return Layout(fs, p.direction, placed, len(lines), (x0, y0, x1, y1))


class Fitter:
    """对一个区域 + 一段 token 二分字号；留白阈值相同的几何结果复用。"""

    def __init__(self, region: Region, toks: list[Token], params: LayoutParams) -> None:
        self.region = region
        self.toks = toks
        self.p = params
        self._geo: dict[int, _Geometry] = {}
        self._memo: dict[int, Optional[Layout]] = {}

    def geometry(self, fs: int) -> _Geometry:
        thr = self.region.margin + max(0.0, float(self.p.reserve(fs)))
        key = int(round(thr * 2))
        g = self._geo.get(key)
        if g is None:
            g = _Geometry(self.region, key / 2.0, self.p.direction)
            self._geo[key] = g
        return g

    def fit(self, fs: int) -> Optional[Layout]:
        fs = int(fs)
        if fs in self._memo:
            return self._memo[fs]
        lay = _fit_at(self.toks, self.geometry(fs), fs, self.p) if fs >= 1 else None
        self._memo[fs] = lay
        return lay

    def largest(self, lo: int, hi: int) -> Optional[Layout]:
        """[lo, hi] 内能完整放下的最大字号（fit 近似单调，二分后再向上探两档）。"""
        lo, hi = int(lo), int(hi)
        if hi < lo:
            return None
        top = self.fit(hi)
        if top is not None:
            return top
        best = self.fit(lo)
        if best is None:
            return None
        a, b = lo, hi            # 不变量：a 可行，b 不可行
        while b - a > 1:
            m = (a + b) // 2
            lay = self.fit(m)
            if lay is not None:
                a, best = m, lay
            else:
                b = m
        for up in (a + 2, a + 3):
            if up < hi:
                lay = self.fit(up)
                if lay is not None:
                    best = lay
        return best


def forced_layout(region: Region, toks: list[Token], fs: int, p: LayoutParams) -> Layout:
    """最小字号仍放不下时的兜底：按区域外接框逐行硬排（渲染时会裁剪到区域内）。"""
    hard = region.hard
    ys, xs = np.nonzero(hard)
    if ys.size == 0:
        h, w = hard.shape
        x0, y0, x1, y1 = 0, 0, w, h
    else:
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    v = p.direction == "v"
    main = (y1 - y0) if v else (x1 - x0)
    cap = max(1.0, main / fs)
    toks2 = split_long_words(toks)
    for t in toks2:          # 单个 token 也比一行长时，容量至少要能放下它
        cap = max(cap, t.length)
    lines = _greedy(toks2, [cap] * 10000) or []
    pitch = fs * p.line_spacing
    n = max(1, len(lines))
    extent = (n - 1) * pitch + fs
    cc = region.center[0] if v else region.center[1]
    cm = region.center[1] if v else region.center[0]
    order = [int(round(cc - extent / 2.0 + i * pitch)) for i in range(n)]
    if v and p.column_order != "ltr":
        order = order[::-1]
    # 用区域外接框作为每行的可用区间（行号可能为负，统一平移到非负下标）
    shift = -min(order) if order and min(order) < 0 else 0
    nb = (max(order) + shift + 1) if order else 1
    lo_edge, hi_edge = (y0, y1) if v else (x0, x1)
    Lb = np.full(max(nb, 1), lo_edge, np.int64)
    Rb = np.full(max(nb, 1), hi_edge, np.int64)
    lay = _place(toks2, lines, [r + shift for r in order], Lb, Rb, cm, fs,
                 LayoutParams(p.direction, p.line_spacing, "center", column_order=p.column_order))
    if shift:
        for pl in lay.placed:
            if v:
                pl.x -= shift
            else:
                pl.y -= shift
        bx0, by0, bx1, by1 = lay.bbox
        lay.bbox = (bx0 - shift, by0, bx1 - shift, by1) if v else (bx0, by0 - shift, bx1, by1 - shift)
    lay.overflow = True
    lay.forced = True
    return lay


# ======================================================================================
# 字号范围 / 方向
# ======================================================================================
def count_cells(src: str) -> float:
    """原文占多少个字格（估计原文字号用）：… / !? 算一格，半角字母算半格。"""
    s = re.sub(r"\s+", "", src or "")
    if not s:
        return 0.0
    s = re.sub(r"(\.{2,}|[…⋯‥]+)", "…", s)
    s = re.sub(r"[!?！？]{2}", "!", s)
    n = 0.0
    for ch in s:
        n += 0.55 if ch.isascii() and ch.isalnum() else 1.0
    return n


def page_ref(w: int, h: int) -> float:
    """字号相对化的参考长度：普通页 = 页高；韩漫长条 / 超高页按宽度折算（避免长条页字号爆炸）。"""
    return float(max(64.0, min(float(h), 1.6 * float(w))))


def estimate_src_font(block: TextBlock, ref: float) -> Optional[float]:
    """由文字框和原文字数估计原文字号（px）；没有原文时返回 None。

    检测框在垂直于行的方向上常常很松（单列竖排字的框可以是字宽的 3 倍），所以不用面积估计，
    而是沿行方向推算：假设分成 k 行 / 列，每行 ceil(n/k) 个字格 → 字号 = 主轴长度 / 每行字格数；
    取 k 列并排仍放得进框宽（列距约 0.2 字）的最大 k。
    """
    x1, y1, x2, y2 = block.box
    w, h = float(x2 - x1), float(y2 - y1)
    n = count_cells(block.src_text)
    if n <= 0 or w < 2 or h < 2:
        return None
    main, cross = (h, w) if block.direction == "v" else (w, h)
    best: Optional[float] = None
    for k in range(1, int(min(math.ceil(n), 60)) + 1):
        fs = main / math.ceil(n / k - 1e-9)
        if fs * (k + 0.2 * (k - 1)) > cross * 1.05:
            break
        best = fs
    if best is None:                 # 一列都放不下（框比字还窄）：受框宽限制
        best = min(cross, main / max(n, 1.0))
    return float(min(max(best * 0.96, ref * 0.006, 4.0), ref * 0.12))


def font_range(est: float, ref: float, ts: dict, sfx: bool = False) -> tuple[int, int]:
    """字号范围 [lo, hi]（px）：以原文字号为中心，再用页面参考长度夹住；设置里的 min / max（>0）优先。"""
    lo = min(max(0.62 * est, 0.0115 * ref), 0.021 * ref)
    hi = min(max(1.22 * est, lo + 1.0), 0.05 * ref)
    if sfx:                          # 拟声词允许更大
        hi = max(hi, min(1.8 * est, 0.12 * ref))
    try:
        mn = float(ts.get("min_font_px", 0) or 0)
        mx = float(ts.get("max_font_px", 0) or 0)
    except (TypeError, ValueError):
        mn = mx = 0.0
    if mn > 0:
        lo = mn
    if mx > 0:
        hi = mx
    lo = max(4.0, lo)
    hi = max(hi, lo)
    return int(round(lo)), int(round(hi))


# ======================================================================================
# 适配策略（放不下时逐级退让，但绝不画出区域）
# ======================================================================================
@dataclass
class FitResult:
    layout: Layout
    overflow: bool
    stage: str          # normal | shrink | relaxed | overflow | forced


def fit_block(region: Region, toks: list[Token], params: LayoutParams, lo: int, hi: int,
              min_abs: int = 6, fixed: Optional[int] = None) -> FitResult:
    """按顺序尝试：[lo, hi] → 降到 0.75×lo → 留白缩小到 35% → 继续缩到 min_abs（overflow）→ 硬排（overflow）。

    fixed：人工指定字号；放不下时同样逐级缩小并标 overflow。英文长单词放不下时拆字再试。
    """
    min_abs = max(1, int(min_abs))
    variants = [toks]
    if any(t.splittable for t in toks):
        variants.append(split_long_words(toks))
    relaxed = Region(region.roi, region.hard, region.dist, region.center, region.margin * 0.35,
                     region.method, region.bg)

    def first(reg: Region, a: int, b: int) -> Optional[Layout]:
        a, b = max(min_abs, int(a)), int(b)
        if b < a:
            return None
        for tv in variants:
            lay = Fitter(reg, tv, params).largest(a, b)
            if lay is not None:
                return lay
        return None

    if fixed is not None and fixed > 0:
        fs = max(min_abs, int(fixed))
        for reg, stage in ((region, "normal"), (relaxed, "relaxed")):
            lay = first(reg, fs, fs)
            if lay is not None:
                return FitResult(lay, False, stage)
        lay = first(relaxed, min_abs, fs - 1)
        if lay is not None:
            return FitResult(lay, True, "overflow")
        return FitResult(forced_layout(region, toks, min_abs, params), True, "forced")

    lo = max(min_abs, int(lo))
    hi = max(lo, int(hi))
    lo2 = max(min_abs, int(math.floor(lo * 0.75)))
    for reg, a, b, stage in ((region, lo, hi, "normal"), (region, lo2, lo - 1, "shrink"),
                             (relaxed, lo2, hi, "relaxed")):
        lay = first(reg, a, b)
        if lay is not None:
            return FitResult(lay, False, stage)
    lay = first(relaxed, min_abs, lo2 - 1)
    if lay is not None:
        return FitResult(lay, True, "overflow")
    return FitResult(forced_layout(relaxed, toks, min_abs, params), True, "forced")


# ======================================================================================
# 多个块共用 / 相连的区域：按离各自文字框的距离切开
# ======================================================================================
def _box_distance(roi: Box, shape: tuple[int, int], box: Box) -> np.ndarray:
    """ROI 内每个像素中心到 box（页面坐标）的欧氏距离（box 内为 0）。"""
    h, w = shape
    xs = np.arange(w, dtype=np.float32) + (roi[0] + 0.5)
    ys = np.arange(h, dtype=np.float32) + (roi[1] + 0.5)
    dx = np.maximum(np.maximum(box[0] - xs, xs - box[2]), 0.0)
    dy = np.maximum(np.maximum(box[1] - ys, ys - box[3]), 0.0)
    return np.sqrt(dy[:, None] ** 2 + dx[None, :] ** 2)


def partition_regions(regions: list[Region], boxes: list[Box]) -> None:
    """区域两两重叠（同一气泡里两段话 / 连体气泡 / 挨着的旁白框）时按 Voronoi 切开，原地修改。"""
    n = len(regions)
    for i in range(n):
        for j in range(i + 1, n):
            ri, rj = regions[i], regions[j]
            ix1, iy1 = max(ri.roi[0], rj.roi[0]), max(ri.roi[1], rj.roi[1])
            ix2, iy2 = min(ri.roi[2], rj.roi[2]), min(ri.roi[3], rj.roi[3])
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            a = ri.hard[iy1 - ri.roi[1]:iy2 - ri.roi[1], ix1 - ri.roi[0]:ix2 - ri.roi[0]]
            b = rj.hard[iy1 - rj.roi[1]:iy2 - rj.roi[1], ix1 - rj.roi[0]:ix2 - rj.roi[0]]
            if int(np.count_nonzero(a & b)) < 16:
                continue
            ri.restrict(_box_distance(ri.roi, ri.shape, boxes[i]) <= _box_distance(ri.roi, ri.shape, boxes[j]))
            rj.restrict(_box_distance(rj.roi, rj.shape, boxes[j]) < _box_distance(rj.roi, rj.shape, boxes[i]))


def choose_direction(block: TextBlock, region: Region, setting: str) -> str:
    """out_direction > 设置 typeset.direction > 自动（原文竖排且区域高宽比 ≥ 1.1 → 竖排）。"""
    if block.out_direction in ("h", "v"):
        return block.out_direction
    if setting in ("h", "v"):
        return setting
    if block.direction != "v":
        return "h"
    safe = region.dist > region.margin
    if not safe.any():
        safe = region.hard
    ys, xs = np.nonzero(safe)
    if ys.size == 0:
        return "h"
    h = float(ys.max() - ys.min() + 1)
    w = float(xs.max() - xs.min() + 1)
    return "v" if h / max(w, 1.0) >= 1.1 else "h"
