"""文字 / 气泡检测（RT-DETR-v2，ONNX）。

流程：
  任意尺寸页面 →（长条韩漫竖切 / 超宽跨页横切，25% 重叠）→ 检测器（直接 resize 成正方形，不 letterbox）
  → 平移回原图 → 切缝碎片合并 → 高分辨率普通页分块补检 → 按类别阈值 + NMS → 相对尺寸过滤
  → 文字框归属气泡（containment ≥ 0.6）→ 同一气泡内相邻文字框合并 → 竖排 / 横排判断 → 阅读顺序排序。

设计要点：
  - 所有尺寸阈值都按页面 / 文字框尺寸的比例计算，不写死像素（3.0 适配差的根源）。
  - 检测器只负责“哪里有字”；衣服 / 招牌上的拉丁字母 logo 是否需要翻译要看内容，
    由 OCR 按源语言判定后标 skip（见 ocr.py），检测阶段不做依赖语言的猜测。
  - 模块对象构造一次、多线程反复调用；本模块没有可变共享状态（模型调用由 Runtime 加锁）。
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np

from .runtime import Profile, Runtime
from .types import Box, HanhuaError, TextBlock, box_area, box_contain_ratio, box_iou, clamp_box

log = logging.getLogger("hanhua.detect")

CLS_BUBBLE = 0        # 气泡
CLS_TEXT_BUBBLE = 1   # 气泡内文字
CLS_TEXT_FREE = 2     # 气泡外文字（旁白 / 拟声词 / 招牌）

FBox = tuple[float, float, float, float]


@dataclass
class _Det:
    """一个原始检测结果（页面坐标，浮点）。"""

    box: FBox
    score: float
    cls: int
    tile: int = 0
    cut_lo: bool = False   # 被切块的“起始”内部切缝截断（竖切 = 上边，横切 = 左边）
    cut_hi: bool = False   # 被切块的“结束”内部切缝截断（竖切 = 下边，横切 = 右边）


@dataclass(frozen=True)
class _Tile:
    x0: int
    y0: int
    x1: int
    y1: int


# --------------------------------------------------------------------------------------
# 小工具
# --------------------------------------------------------------------------------------
def _to_int_box(b: Sequence[float], w: int, h: int) -> Box:
    """浮点 xyxy → 整数开区间 Box（向外取整后裁到页面内）。"""
    return clamp_box((math.floor(b[0]), math.floor(b[1]), math.ceil(b[2]), math.ceil(b[3])), w, h)


def _fiou(a: FBox, b: FBox) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1]) + max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _fcontain(inner: FBox, outer: FBox) -> float:
    ix = max(0.0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    iy = max(0.0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    a = max(0.0, inner[2] - inner[0]) * max(0.0, inner[3] - inner[1])
    return (ix * iy) / a if a > 0 else 0.0


def _funion(a: FBox, b: FBox) -> FBox:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _overlap_1d(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _box_gap(a: Sequence[float], b: Sequence[float]) -> float:
    """两个框之间的最短距离（相交为 0）。"""
    dx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    return math.hypot(dx, dy)


def ensure_bgr(img: np.ndarray) -> Optional[np.ndarray]:
    """把各种奇怪输入规整成 BGR uint8 (H, W, 3)；无法处理时返回 None（不抛）。"""
    if img is None or not isinstance(img, np.ndarray) or img.size == 0:
        return None
    a = img
    if a.dtype != np.uint8:
        if a.dtype == np.bool_:
            a = a.astype(np.uint8) * 255
        elif np.issubdtype(a.dtype, np.floating):
            mx = float(np.nanmax(a)) if a.size else 1.0
            a = np.nan_to_num(a.astype(np.float32))
            a = np.clip(a * (255.0 if mx <= 1.0 else 1.0), 0, 255).astype(np.uint8)
        else:
            a = np.clip(a, 0, 255).astype(np.uint8)
    if a.ndim == 2:
        a = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
    elif a.ndim == 3 and a.shape[2] == 1:
        a = cv2.cvtColor(a[:, :, 0], cv2.COLOR_GRAY2BGR)
    elif a.ndim == 3 and a.shape[2] == 4:
        a = cv2.cvtColor(a, cv2.COLOR_BGRA2BGR)
    elif a.ndim != 3 or a.shape[2] != 3:
        return None
    return np.ascontiguousarray(a)


# --------------------------------------------------------------------------------------
# 竖排 / 横排判断（检测、OCR 共用）
# --------------------------------------------------------------------------------------
def text_fg_mask(gray: np.ndarray) -> np.ndarray:
    """Otsu 二值化，取像素少的一类当作文字（bool 数组）。"""
    if gray.size == 0:
        return np.zeros(gray.shape, bool)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg = th > 0
    if fg.mean() > 0.5:
        fg = ~fg
    return fg


def _runs(on: np.ndarray) -> list[tuple[int, int]]:
    idx = np.flatnonzero(np.diff(np.concatenate(([0], on.astype(np.int8), [0]))))
    return [(int(idx[i]), int(idx[i + 1])) for i in range(0, len(idx), 2)]


def _profile_runs(mask: np.ndarray, axis: int) -> tuple[list[tuple[int, int]], float, np.ndarray]:
    """投影原始分段（不合并空隙）。返回 (段列表, 估计的列宽 / 行高 s, 投影计数)。

    两遍：先用相对强阈值估计 s，再按“每列文字像素数 ≥ 0.12·s”分段，
    这样只有一个字的短列（如说话人署名“希”）也不会被当成噪声；面积太小的段当噪声丢掉。
    """
    cnt = mask.sum(axis=axis).astype(np.float32)
    if cnt.size == 0 or float(cnt.max()) <= 0:
        return [], 1.0, cnt
    strong = _runs(cnt > 0.06 * float(cnt.max()))
    if not strong:
        return [], 1.0, cnt
    s = max(1.0, float(np.median([e - st for st, e in strong])))
    runs = [(st, e) for st, e in _runs(cnt >= max(1.5, 0.12 * s)) if float(cnt[st:e].sum()) >= 0.1 * s * s]
    return runs, s, cnt


def _axis_segments(mask: np.ndarray, axis: int) -> list[tuple[int, int]]:
    """沿某个方向的投影分段：axis=0 → 按列投影（x 方向的段），axis=1 → 按行投影（y 方向的段）。
    小于 0.25·s 的空隙视为字间距并入同一段。"""
    runs, s, _ = _profile_runs(mask, axis)
    if not runs:
        return []
    merged: list[tuple[int, int]] = [runs[0]]
    for st, e in runs[1:]:
        if st - merged[-1][1] < max(1.0, 0.25 * s):
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((st, e))
    return merged


def split_paragraphs(crop_bgr: np.ndarray, direction: str) -> list[tuple[int, int, int, int]]:
    """把一个气泡外文字块按“明显大于正常列距 / 行距”的空隙拆成几段（返回 crop 内坐标）。

    p003 这类页面：两段旁白（各 3 列）被检测器框成一个大块，段间距约是段内列距的 8 倍。
    以原始的列（竖排）/ 行（横排）为单位：空隙 ≥ max(2×其余空隙中位数, 中位数 + 0.5·列宽, 0.6·列宽)
    才拆；只有两列时要求空隙 ≥ 1.2 倍列宽。正常排版不会被误拆。
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return []
    h, w = crop_bgr.shape[:2]
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if crop_bgr.ndim == 3 else crop_bgr
    s = min(1.0, 480.0 / max(h, w))
    if s < 1.0:
        gray = cv2.resize(gray, (max(4, int(w * s)), max(4, int(h * s))), interpolation=cv2.INTER_AREA)
    fg = text_fg_mask(gray)
    axis = 0 if direction == "v" else 1          # 竖排按列分段，横排按行分段
    segs, med_w, _ = _profile_runs(fg, axis)
    whole = [(0, 0, w, h)]
    if len(segs) < 2:
        return whole
    gaps = [segs[i + 1][0] - segs[i][1] for i in range(len(segs) - 1)]
    cuts: list[int] = []
    for i, g in enumerate(gaps):
        others = gaps[:i] + gaps[i + 1:]
        if others:
            mo = float(np.median(others))
            ok = g >= max(2.0 * mo, mo + 0.5 * med_w, 0.6 * med_w)
        else:
            ok = g >= 1.2 * med_w
        if ok:
            cuts.append(i)
    if not cuts:
        return whole
    parts: list[tuple[int, int, int, int]] = []
    bounds = [0] + [c + 1 for c in cuts] + [len(segs)]
    for k in range(len(bounds) - 1):
        lo, hi = segs[bounds[k]][0], segs[bounds[k + 1] - 1][1]
        sub = fg[:, lo:hi] if axis == 0 else fg[lo:hi, :]
        ys, xs = np.nonzero(sub)
        if xs.size == 0:
            continue
        if axis == 0:
            bx1, bx2, by1, by2 = lo + xs.min(), lo + xs.max() + 1, ys.min(), ys.max() + 1
        else:
            bx1, bx2, by1, by2 = xs.min(), xs.max() + 1, lo + ys.min(), lo + ys.max() + 1
        pad = 0.15 * med_w
        parts.append((
            max(0, int(math.floor((bx1 - pad) / s))), max(0, int(math.floor((by1 - pad) / s))),
            min(w, int(math.ceil((bx2 + pad) / s))), min(h, int(math.ceil((by2 + pad) / s))),
        ))
    return parts if len(parts) >= 2 else whole


def estimate_direction(crop_bgr: np.ndarray, default: str = "v") -> str:
    """按框长宽比 + 行 / 列投影判断竖排 "v" / 横排 "h"。

    - 多列且每列明显细长 → 竖排；多行且每行明显扁长 → 横排；
    - 单行 / 单列（投影看不出）→ 看长宽比；接近正方形 → 用阅读方向的默认值。
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return default
    h, w = crop_bgr.shape[:2]
    if h < 4 or w < 4:
        return "v" if h >= w * 1.15 else ("h" if w >= h * 1.15 else default)
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if crop_bgr.ndim == 3 else crop_bgr
    # 太大的块先缩小（只为速度，比例不变）
    s = 160.0 / max(h, w)
    if s < 1.0:
        gray = cv2.resize(gray, (max(4, int(w * s)), max(4, int(h * s))), interpolation=cv2.INTER_AREA)
    fg = text_fg_mask(gray)
    ys, xs = np.nonzero(fg)
    if xs.size >= 8:
        fg = fg[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]
    th, tw = fg.shape
    xseg = _axis_segments(fg, 0)
    yseg = _axis_segments(fg, 1)
    v_elong = th / max(1.0, float(np.median([e - s for s, e in xseg]))) if xseg else 0.0
    h_elong = tw / max(1.0, float(np.median([e - s for s, e in yseg]))) if yseg else 0.0
    v_ok = len(xseg) >= 2 and v_elong >= 1.5
    h_ok = len(yseg) >= 2 and h_elong >= 1.5
    if v_ok and not h_ok:
        return "v"
    if h_ok and not v_ok:
        return "h"
    if v_ok and h_ok:
        return "v" if v_elong >= h_elong else "h"
    if h >= w * 1.15:
        return "v"
    if w >= h * 1.15:
        return "h"
    return default


# --------------------------------------------------------------------------------------
# 检测器
# --------------------------------------------------------------------------------------
class TextDetector:
    """RT-DETR 文字 / 气泡检测。构造一次，多线程反复调用 detect()。"""

    # 按类别的最终阈值（已在 9 张真实页上按原始输出校准，见 tests/test_detect.py）
    THRESH: dict[int, float] = {CLS_BUBBLE: 0.40, CLS_TEXT_BUBBLE: 0.30, CLS_TEXT_FREE: 0.30}
    # 切块时先用较低阈值收集碎片，合并后再按最终阈值过滤（被切断的框分数会偏低）
    PRE_FACTOR = 0.6
    # 高置信气泡里没找到任何文字时，对气泡内部的文字候选放宽到这个分数
    RESCUE_THRESH = 0.15
    RESCUE_BUBBLE = 0.60
    NMS_IOU = 0.5
    CONTAIN_BUBBLE = 0.6          # 文字框落在气泡里的比例 ≥ 此值 → 属于该气泡
    TALL_RATIO = 2.2              # 长宽比超过此值就切块
    TILE_LONG = 1.6               # 切块边长 = 1.6 × 短边
    TILE_OVERLAP = 0.25
    MAX_BATCH = 4                 # 动态 batch 模型一次最多送几块

    def __init__(self, rt: Runtime, profile: Profile) -> None:
        self.rt = rt
        self.profile = profile
        self.model_key = profile.detector
        self.size = int(profile.detector_size) if profile.detector_size else 640
        self._fixed_batch1 = False
        self._names: Optional[tuple[str, str, list[str]]] = None

    # -- 模型 ----------------------------------------------------------------------------
    def _model(self):
        m = self.rt.model(self.model_key)
        if self._names is None:
            ins = list(m.input_names)
            img_name = "images" if "images" in ins else ins[0]
            size_name = "orig_target_sizes" if "orig_target_sizes" in ins else (ins[1] if len(ins) > 1 else "")
            shape = m.input_shapes.get(img_name) or []
            # detector_s_int8 这类固定 640×640 的模型只能 batch=1，并按模型的固定尺寸送入
            if len(shape) == 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
                self.size = int(shape[2])
                self._fixed_batch1 = True
            self._names = (img_name, size_name, list(m.output_names))
        return m

    def _run_batch(self, m, crops: list[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        img_name, size_name, out_names = self._names  # type: ignore[misc]
        size = self.size
        xs = []
        sizes = []
        for c in crops:
            ch, cw = c.shape[:2]
            rgb = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
            rs = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
            xs.append(rs.transpose(2, 0, 1))
            sizes.append((cw, ch))
        x = np.ascontiguousarray(np.stack(xs).astype(np.float32) * (1.0 / 255.0))
        feeds = {img_name: x}
        if size_name:
            feeds[size_name] = np.asarray(sizes, dtype=np.int64)
        outs = m.run(feeds)
        by_name = dict(zip(out_names, outs))
        labels = by_name.get("labels", outs[0])
        boxes = by_name.get("boxes", outs[1])
        scores = by_name.get("scores", outs[2])
        return [(np.asarray(labels[i]), np.asarray(boxes[i]), np.asarray(scores[i])) for i in range(len(crops))]

    def _infer_tiles(self, img: np.ndarray, tiles: list[_Tile], axis: int) -> list[_Det]:
        """逐块推理（能批量就批量），结果平移回原图并标记切缝截断。"""
        m = self._model()
        H, W = img.shape[:2]
        bs = 1 if self._fixed_batch1 else self.MAX_BATCH
        pre = {c: t * self.PRE_FACTOR for c, t in self.THRESH.items()}
        pre_floor = min(self.RESCUE_THRESH, min(pre.values()))
        dets: list[_Det] = []
        i = 0
        while i < len(tiles):
            group = tiles[i: i + bs]
            crops = [img[t.y0: t.y1, t.x0: t.x1] for t in group]
            try:
                results = self._run_batch(m, crops)
            except Exception as e:  # noqa: BLE001
                if len(group) == 1:
                    raise HanhuaError(f"文字检测失败：{e}") from e
                log.warning("批量检测失败，改为逐块: %s", e)
                results = []
                for c in crops:
                    try:
                        results.extend(self._run_batch(m, [c]))
                    except Exception as e2:  # noqa: BLE001
                        raise HanhuaError(f"文字检测失败：{e2}") from e2
            for k, (t, (labels, boxes, scores)) in enumerate(zip(group, results)):
                ti = i + k
                tw, th = t.x1 - t.x0, t.y1 - t.y0
                tile_len = th if axis == 1 else tw
                eps = max(2.0, 0.015 * tile_len)
                for lab, b, s in zip(labels.reshape(-1), boxes.reshape(-1, 4), scores.reshape(-1)):
                    c = int(lab)
                    s = float(s)
                    if c not in self.THRESH or not np.isfinite(s) or s < pre_floor:
                        continue
                    if not np.all(np.isfinite(b)):
                        continue
                    x1, y1, x2, y2 = (float(v) for v in b)
                    x1, x2 = sorted((min(max(x1, 0.0), tw), min(max(x2, 0.0), tw)))
                    y1, y2 = sorted((min(max(y1, 0.0), th), min(max(y2, 0.0), th)))
                    if x2 - x1 < 1 or y2 - y1 < 1:
                        continue
                    if axis == 1:
                        cut_lo = t.y0 > 0 and y1 <= eps
                        cut_hi = t.y1 < H and y2 >= th - eps
                    elif axis == 0:
                        cut_lo = t.x0 > 0 and x1 <= eps
                        cut_hi = t.x1 < W and x2 >= tw - eps
                    else:
                        cut_lo = cut_hi = False
                    dets.append(_Det((x1 + t.x0, y1 + t.y0, x2 + t.x0, y2 + t.y0), s, c, ti, cut_lo, cut_hi))
            i += len(group)
        return dets

    # -- 切块 ----------------------------------------------------------------------------
    @classmethod
    def make_tiles(cls, w: int, h: int) -> tuple[list[_Tile], int]:
        """返回 (切块列表, 切分方向)：方向 1 = 竖切（长条），0 = 横切（超宽），-1 = 不切。"""
        def starts(total: int, size: int) -> list[int]:
            if size >= total:
                return [0]
            stride = size * (1.0 - cls.TILE_OVERLAP)
            n = int(math.ceil((total - size) / stride)) + 1
            step = (total - size) / (n - 1)
            return [int(round(k * step)) for k in range(n)]

        if h > w * cls.TALL_RATIO:
            th = min(h, max(1, int(round(cls.TILE_LONG * w))))
            return [_Tile(0, y, w, min(h, y + th)) for y in starts(h, th)], 1
        if w > h * cls.TALL_RATIO:
            tw = min(w, max(1, int(round(cls.TILE_LONG * h))))
            return [_Tile(x, 0, min(w, x + tw), h) for x in starts(w, tw)], 0
        return [_Tile(0, 0, w, h)], -1

    # -- 切缝合并 + NMS ------------------------------------------------------------------
    @staticmethod
    def _merge_fragments(dets: list[_Det], axis: int) -> list[_Det]:
        """把被切缝截断、分散在相邻两块里的同一个框拼回去；完整框里的残片直接丢掉。"""
        if axis < 0 or len(dets) < 2:
            return dets
        n = len(dets)
        parent = list(range(n))
        drop = [False] * n

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        by_tile: dict[int, list[int]] = {}
        for k, d in enumerate(dets):
            by_tile.setdefault(d.tile, []).append(k)
        pairs = [(i, j) for t, idx in by_tile.items() for i in idx for j in by_tile.get(t + 1, [])]
        for i, j in pairs:
            a, b = dets[i], dets[j]   # a 在前一块、b 在后一块
            if a.cls != b.cls:
                continue
            # cross = 垂直于切分方向的重叠；along = 沿切分方向的重叠
            if axis == 1:
                cross = _overlap_1d(a.box[0], a.box[2], b.box[0], b.box[2])
                cmin = min(a.box[2] - a.box[0], b.box[2] - b.box[0])
                along = _overlap_1d(a.box[1], a.box[3], b.box[1], b.box[3])
            else:
                cross = _overlap_1d(a.box[1], a.box[3], b.box[1], b.box[3])
                cmin = min(a.box[3] - a.box[1], b.box[3] - b.box[1])
                along = _overlap_1d(a.box[0], a.box[2], b.box[0], b.box[2])
            if cmin <= 0 or cross < 0.5 * cmin or along <= 0:
                continue
            if a.cut_hi and b.cut_lo:
                parent[find(i)] = find(j)            # 两段都被截断 → 拼接
            elif a.cut_hi and _fcontain(a.box, b.box) >= 0.5:
                drop[i] = True                       # a 是 b（完整框）的残片
            elif b.cut_lo and _fcontain(b.box, a.box) >= 0.5:
                drop[j] = True                       # b 是 a（完整框）的残片
        groups: dict[int, list[int]] = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)
        out: list[_Det] = []
        for idxs in groups.values():
            keep = [k for k in idxs if not drop[k]] if len(idxs) == 1 else idxs
            if not keep:
                continue
            box = dets[keep[0]].box
            for k in keep[1:]:
                box = _funion(box, dets[k].box)
            best = max(keep, key=lambda k: dets[k].score)
            d = dets[best]
            out.append(_Det(box, max(dets[k].score for k in keep), d.cls, d.tile,
                            any(dets[k].cut_lo for k in keep), any(dets[k].cut_hi for k in keep)))
        return out

    @staticmethod
    def _nms(dets: list[_Det], iou: float, contain: float = 1.01) -> list[_Det]:
        """标准 NMS（按分数）；contain < 1 时，被高分框包含 ≥ contain 的低分框也去掉。"""
        order = sorted(dets, key=lambda d: -d.score)
        keep: list[_Det] = []
        for d in order:
            if any(_fiou(d.box, k.box) >= iou or _fcontain(d.box, k.box) >= contain for k in keep):
                continue
            keep.append(d)
        return keep

    # -- 主流程 --------------------------------------------------------------------------
    def detect(self, img: np.ndarray, reading: str = "rtl") -> tuple[list[TextBlock], list[Box]]:
        """返回 (按阅读顺序排好、id=0..n-1 的文字块, 全部气泡框)。"""
        img = ensure_bgr(img)
        if img is None:
            log.warning("检测输入为空或格式不支持，跳过")
            return [], []
        H, W = img.shape[:2]
        if H < 8 or W < 8:
            return [], []
        reading = reading if reading in ("rtl", "ltr") else "rtl"
        tiles, axis = self.make_tiles(W, H)
        raw = self._infer_tiles(img, tiles, axis)
        merged = self._merge_fragments(raw, axis)
        if self.profile.name != "fast":
            detail_tiles = self.make_detail_tiles(W, H)
            if detail_tiles:
                detail = self._infer_tiles(img, detail_tiles, -1)
                merged.extend(d for d in detail if self._inside_detail_tile(d, detail_tiles[d.tile], W, H))

        ref = float(min(W, H))
        bubbles_d: list[_Det] = []
        texts_d: list[_Det] = []
        weak_texts: list[_Det] = []
        for d in merged:
            bw, bh = d.box[2] - d.box[0], d.box[3] - d.box[1]
            if d.cls == CLS_BUBBLE:
                if d.score >= self.THRESH[CLS_BUBBLE] and min(bw, bh) >= 0.02 * ref:
                    bubbles_d.append(d)
                continue
            if max(bw, bh) < 0.012 * ref or min(bw, bh) < 0.005 * ref:
                continue
            if d.score >= self.THRESH[d.cls]:
                texts_d.append(d)
            elif d.score >= self.RESCUE_THRESH:
                weak_texts.append(d)

        bubbles_d = self._nms(bubbles_d, self.NMS_IOU)
        # 同一处文字常被同时报成 text_bubble / text_free：跨类别 NMS，留高分的类别
        texts_d = self._nms(texts_d, self.NMS_IOU, contain=0.9)
        texts_d += self._rescue_texts(bubbles_d, texts_d, weak_texts)

        bubble_boxes = [_to_int_box(d.box, W, H) for d in bubbles_d]
        blocks = self._build_blocks(img, texts_d, bubble_boxes, reading)
        blocks = sort_reading_order(blocks, reading)
        for i, b in enumerate(blocks):
            b.id = i
        bubble_boxes = [b.box for b in sort_reading_order(
            [TextBlock(id=0, box=bb) for bb in bubble_boxes], reading)]
        log.debug("检测：%dx%d 切块 %d → 文字块 %d，气泡 %d", W, H, len(tiles), len(blocks), len(bubble_boxes))
        return blocks, bubble_boxes

    @staticmethod
    def make_detail_tiles(w: int, h: int) -> list[_Tile]:
        """高分辨率普通页补检；整页检测仍负责跨越切缝的大气泡。"""
        side = 1536
        if min(w, h) <= 1.25 * side or max(w, h) > 2.2 * min(w, h):
            return []

        def starts(total: int) -> list[int]:
            count = int(math.ceil((total - side) / (0.65 * side)))
            return [int(round(k * (total - side) / count)) for k in range(count + 1)]

        return [_Tile(x, y, x + side, y + side) for y in starts(h) for x in starts(w)]

    @staticmethod
    def _inside_detail_tile(det: _Det, tile: _Tile, w: int, h: int) -> bool:
        # 切缝上的半个字 / 半个气泡不能覆盖整页检测的完整结果。
        gap = 0.015 * min(tile.x1 - tile.x0, tile.y1 - tile.y0)
        x1, y1, x2, y2 = det.box
        return not ((tile.x0 > 0 and x1 <= tile.x0 + gap)
                    or (tile.y0 > 0 and y1 <= tile.y0 + gap)
                    or (tile.x1 < w and x2 >= tile.x1 - gap)
                    or (tile.y1 < h and y2 >= tile.y1 - gap))

    def _rescue_texts(self, bubbles: list[_Det], texts: list[_Det], weak: list[_Det]) -> list[_Det]:
        """高置信气泡里一个文字框都没有时，接受气泡内部的低分文字候选（提高召回）。"""
        out: list[_Det] = []
        for bd in bubbles:
            if bd.score < self.RESCUE_BUBBLE:
                continue
            if any(_fcontain(t.box, bd.box) >= self.CONTAIN_BUBBLE for t in texts):
                continue
            cands = [t for t in weak if _fcontain(t.box, bd.box) >= 0.8]
            if cands:
                best = max(cands, key=lambda t: t.score)
                out.append(_Det(best.box, best.score, CLS_TEXT_BUBBLE, best.tile))
        return out

    def _build_blocks(self, img: np.ndarray, texts: list[_Det], bubbles: list[Box], reading: str) -> list[TextBlock]:
        H, W = img.shape[:2]
        # 1) 每个文字框找包含它最多的气泡
        assigned: list[tuple[_Det, Optional[int]]] = []
        for t in texts:
            tb = _to_int_box(t.box, W, H)
            best, best_c = None, 0.0
            for bi, bb in enumerate(bubbles):
                c = box_contain_ratio(tb, bb)
                if c >= self.CONTAIN_BUBBLE and (c > best_c + 1e-6 or (
                        abs(c - best_c) <= 1e-6 and best is not None and box_area(bb) < box_area(bubbles[best]))):
                    best, best_c = bi, c
            assigned.append((t, best))

        # 2) 分组：同一气泡内相邻的文字框合并；气泡外 / 无气泡的框只合并明显重叠的
        groups: list[dict] = []
        for t, bi in assigned:
            kind = "bubble" if (bi is not None or t.cls == CLS_TEXT_BUBBLE) else "free"
            groups.append({"box": t.box, "score": t.score, "kind": kind, "bubble": bi, "n": 1})
        changed = True
        while changed:
            changed = False
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    a, b = groups[i], groups[j]
                    if self._should_merge(a, b):
                        if a["kind"] != b["kind"]:
                            if a["bubble"] is not None or b["bubble"] is not None:
                                a["kind"] = "bubble"
                            elif b["score"] > a["score"]:
                                a["kind"] = b["kind"]
                        a["box"] = _funion(a["box"], b["box"])
                        a["score"] = max(a["score"], b["score"])
                        a["n"] += b["n"]
                        if a["bubble"] is None:
                            a["bubble"] = b["bubble"]
                        groups.pop(j)
                        changed = True
                        break
                if changed:
                    break

        # 3) 生成 TextBlock（气泡外的大块按段落拆开）
        default_dir = "v" if reading == "rtl" else "h"
        blocks: list[TextBlock] = []
        for g in groups:
            box = _to_int_box(g["box"], W, H)
            if box_area(box) <= 0:
                continue
            x1, y1, x2, y2 = box
            crop = img[y1:y2, x1:x2]
            direction = estimate_direction(crop, default_dir)
            bubble_box = bubbles[g["bubble"]] if g["bubble"] is not None else None
            parts = [(0, 0, x2 - x1, y2 - y1)]
            if g["kind"] == "free":
                try:
                    parts = split_paragraphs(crop, direction) or parts
                except Exception as e:  # noqa: BLE001 - 拆段只是锦上添花，失败就整块用
                    log.debug("拆段失败: %s", e)
            for px1, py1, px2, py2 in parts:
                sub = clamp_box((x1 + px1, y1 + py1, x1 + px2, y1 + py2), W, H)
                if box_area(sub) <= 0:
                    continue
                d = direction if len(parts) == 1 else estimate_direction(
                    img[sub[1]:sub[3], sub[0]:sub[2]], direction)
                blocks.append(TextBlock(id=0, box=sub, kind=g["kind"], score=round(float(g["score"]), 4),
                                        bubble_box=bubble_box, direction=d))
        return blocks

    @staticmethod
    def _should_merge(a: dict, b: dict) -> bool:
        ab, bb = a["box"], b["box"]
        if a["bubble"] is not None and a["bubble"] == b["bubble"]:
            # 同一气泡：相邻（间距不超过约一个字宽）才合并，防止把误检的“大气泡”里的远处文字并到一起
            char = min(ab[2] - ab[0], ab[3] - ab[1], bb[2] - bb[0], bb[3] - bb[1])
            return _box_gap(ab, bb) <= 1.2 * max(char, 1.0)
        if a["bubble"] is not None and b["bubble"] is not None:
            return False  # 不同气泡绝不合并
        # 其余情况（气泡外 / 没找到气泡）：只合并明显重叠的框
        return _fiou(ab, bb) >= 0.3 or _fcontain(ab, bb) >= 0.6 or _fcontain(bb, ab) >= 0.6


# --------------------------------------------------------------------------------------
# 阅读顺序
# --------------------------------------------------------------------------------------
def sort_reading_order(blocks: list[TextBlock], reading: str = "rtl") -> list[TextBlock]:
    """按纵向重叠聚成“行”，行从上到下；行内 rtl 从右到左、ltr 从左到右。

    入行条件：与行带的纵向重叠 ≥ 两者中较矮者高度的一半，且块的顶边落在行带的上半部分
    （防止一个很高的块把下面所有块都“串”进同一行）。
    """
    rows: list[list[TextBlock]] = []
    bands: list[list[int]] = []
    for b in sorted(blocks, key=lambda t: (t.box[1], t.box[3])):
        bh = max(1, b.box[3] - b.box[1])
        placed = False
        for row, band in zip(rows, bands):
            band_h = max(1, band[1] - band[0])
            ov = min(band[1], b.box[3]) - max(band[0], b.box[1])
            if ov > 0 and ov >= 0.5 * min(bh, band_h) and b.box[1] <= band[0] + 0.5 * band_h:
                row.append(b)
                band[0] = min(band[0], b.box[1])
                band[1] = max(band[1], b.box[3])
                placed = True
                break
        if not placed:
            rows.append([b])
            bands.append([b.box[1], b.box[3]])
    order = sorted(range(len(rows)), key=lambda k: bands[k][0])
    out: list[TextBlock] = []
    for k in order:
        row = rows[k]
        if reading == "rtl":
            row.sort(key=lambda t: -(t.box[0] + t.box[2]))
        else:
            row.sort(key=lambda t: (t.box[0] + t.box[2]))
        out.extend(row)
    return out


# --------------------------------------------------------------------------------------
# 调试图
# --------------------------------------------------------------------------------------
def _dashed_rect(img: np.ndarray, box: Box, color: tuple[int, int, int], t: int, dash: int) -> None:
    x1, y1, x2, y2 = box
    for x in range(x1, x2, dash * 2):
        cv2.line(img, (x, y1), (min(x + dash, x2), y1), color, t)
        cv2.line(img, (x, y2 - 1), (min(x + dash, x2), y2 - 1), color, t)
    for y in range(y1, y2, dash * 2):
        cv2.line(img, (x1, y), (x1, min(y + dash, y2)), color, t)
        cv2.line(img, (x2 - 1, y), (x2 - 1, min(y + dash, y2)), color, t)


def draw_debug(img: np.ndarray, blocks: list[TextBlock], bubbles: list[Box]) -> np.ndarray:
    """调试 / 界面预览图（BGR）：绿 = 气泡，红 = 气泡内文字，蓝 = 气泡外文字，
    橙 = 拟声词，灰色虚线 = skip（保留原文）。标签：序号 + 方向(v/h) + 检测分。"""
    base = ensure_bgr(img)
    if base is None:
        return np.zeros((64, 64, 3), np.uint8)
    vis = base.copy()
    H, W = vis.shape[:2]
    t = max(1, int(round(min(H, W) / 500)))
    fs = max(0.4, min(H, W) / 1600)
    for bb in bubbles or []:
        try:
            x1, y1, x2, y2 = clamp_box(tuple(int(v) for v in bb), W, H)  # type: ignore[arg-type]
        except Exception:  # noqa: BLE001
            continue
        cv2.rectangle(vis, (x1, y1), (max(x1, x2 - 1), max(y1, y2 - 1)), (60, 190, 60), t)
    overlay = vis.copy()
    for b in blocks or []:
        try:
            box = clamp_box(tuple(int(v) for v in b.box), W, H)  # type: ignore[arg-type]
        except Exception:  # noqa: BLE001
            continue
        if b.skip:
            color = (150, 150, 150)
        elif b.is_sfx:
            color = (0, 140, 255)
        elif b.kind == "bubble":
            color = (40, 40, 230)
        else:
            color = (230, 90, 30)
        if not b.skip:
            cv2.rectangle(overlay, (box[0], box[1]), (box[2] - 1, box[3] - 1), color, -1)
    vis = cv2.addWeighted(overlay, 0.18, vis, 0.82, 0)
    for b in blocks or []:
        try:
            box = clamp_box(tuple(int(v) for v in b.box), W, H)  # type: ignore[arg-type]
        except Exception:  # noqa: BLE001
            continue
        if b.skip:
            color = (150, 150, 150)
            _dashed_rect(vis, box, color, t, max(4, 4 * t))
        else:
            color = (0, 140, 255) if b.is_sfx else ((40, 40, 230) if b.kind == "bubble" else (230, 90, 30))
            cv2.rectangle(vis, (box[0], box[1]), (box[2] - 1, box[3] - 1), color, t + 1)
        label = f"{b.id} {b.direction} {b.score:.2f}" + (" skip" if b.skip else "") + (" sfx" if b.is_sfx else "")
        (tw, th), base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, max(1, t))
        lx, ly = box[0], max(th + 4, box[1] - 4)
        cv2.rectangle(vis, (lx, ly - th - 4), (lx + tw + 4, ly + base_line - 2), color, -1)
        cv2.putText(vis, label, (lx + 2, ly - 2), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), max(1, t), cv2.LINE_AA)
    return vis
