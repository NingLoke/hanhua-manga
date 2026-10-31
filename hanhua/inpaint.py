"""去字（inpaint）：只改文字笔画像素，绝不破坏背景。

3.0 的「背景被搞坏」有三个根源，这里逐条解决：
  1. 拟声词 mask 把 ROI 里所有白色 / 有饱和度的像素都抹掉
     → mask 只取「与局部背景色差大」的像素（强 / 弱双阈值，弱像素只能贴着强笔画生长），
       再按字号比例膨胀覆盖抗锯齿边缘、描边和外发光；气泡内文字与「气泡内部腐蚀若干像素」相交，
       绝不碰气泡描边；气泡外文字用 PP-OCR DB 概率图约束到真正的文字区域。
  2. LaMa 输入被硬拉成 512x512 变形
     → 按连通域聚类裁剪 + 上下文边距，只做等比缩小（INTER_AREA），reflect pad 到分桶尺寸。
  3. LaMa 结果整块贴回，帽子 / 衣服被抹糊
     → 只在 mask 内合成（羽化只在 mask 内侧），最后 np.where 强制 mask==0 处与输入逐位相同。

填充分三档（settings["inpaint"]["mode"] == "auto"）：
  - flat  ：mask 外围一圈 ≥95% 像素与中位色 ΔE < 6，且按网格分块的中位色处处一致（无渐变）
            → 直接填中位色（白气泡几乎都是这样，又快又完美）；
  - smooth：气泡内部是平滑渐变（彩漫 / CG 常见）→ push-pull 金字塔插值，边界连续、无接缝；
  - lama  ：其余（实景 / 网点 / 复杂背景）。
  "fill" 模式强制不用 LaMa（flat / smooth），"lama" 模式强制全部 LaMa。

性能实测要点（RTX 5060 Ti，onnxruntime-gpu 1.31）：CUDA 上 LaMa 每次「换输入尺寸」都要重建 cuFFT plan，
约多花 100~200ms（回到见过的尺寸也一样），约等于多跑一次 768x768，所以 CUDA 上用「只增不减的粘性画布」：
裁剪塞进已有画布（reflect pad），warmup 直接用最大画布 → 整个会话 0 次切换；ppocr_det 在 GPU 上也用固定画布。
DirectML / CPU 没有明显的切换代价，按最小分桶尺寸跑（CPU 的耗时与面积成正比）。
"""
from __future__ import annotations

import functools
import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Sequence

import cv2
import numpy as np

from .types import Box, HanhuaError, TextBlock, clamp_box, expand_box

if TYPE_CHECKING:  # 只用于类型注解，避免循环 / 无谓导入
    from .runtime import Model, Profile, Runtime

log = logging.getLogger("hanhua.inpaint")

# ---------------------------------------------------------------------------------------------
# 相对阈值（全部按字号 / 框尺寸的比例计算，不写死像素）
# ---------------------------------------------------------------------------------------------
_WIN_MARGIN = 0.35      # 分析窗口 = 文字框外扩 0.35 字高（覆盖描边 + 外发光）
_AA_DILATE = 0.10       # 抗锯齿 / 外发光膨胀半径 = 0.10 字高（至少 2px）
_GROW = 0.25            # 弱笔画像素只在强笔画 0.25 字高以内生长（hysteresis）
_RING = 0.12            # 判断「周围是否纯色」用的外环宽度
_HOLE = 0.6             # mask 内小于 (0.6 字高)^2 的洞直接填上（字内空白 / 描边包住的字芯）
_BG_SIGMA = 0.6         # 背景模型（归一化卷积）的高斯 sigma = 0.6 字高
_FLAT_DE = 6.0          # 纯色判定：外环像素与中位色 ΔE76 < 6 ...
_FLAT_FRAC = 0.95       # ... 的比例 ≥ 95%（排除网点 / 纹理）
_FLAT_CELL = 2.0        # ... 且外环按 2 字高的网格分块，各块中位色与整体中位色
_FLAT_CELL_DE = 2.0     #     ΔE ≤ 2（排除渐变：半透明气泡整列填一个色会留下文字形状的色差轮廓）
_SMOOTH_DE = 5.0        # 平滑判定：外环像素与其局部平滑估计的 ΔE p95 < 5
_MASK_MAX_RATIO = 0.85  # 框内 mask 占比异常 → 退回圆角矩形
_MASK_MIN_RATIO = 0.01
_FEATHER_BAND = 0.5     # 羽化：mask 最外 1px 只替换一半（mask 已比文字多膨胀 1px，文字像素本身始终全替换）
_DB_CANVAS = 640        # ppocr_det 在 GPU 上的固定输入边长（CUDA 换输入尺寸要重建计划，固定画布最快）

_LAMA_KEY = "lama"
_DB_KEY = "ppocr_det"
# models/ppocr/det.yml：BGR 图，scale 1/255，mean/std 按 hwc 顺序直接作用于 B,G,R（与 PaddleOCR 一致）
_DB_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_DB_STD = np.array([0.229, 0.224, 0.225], np.float32)
_DB_THRESH = 0.3        # det.yml PostProcess.thresh


# ---------------------------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------------------------
@functools.lru_cache(maxsize=64)
def _disk(r: int) -> np.ndarray:
    r = max(1, int(r))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def _ceil_to(v: int, b: int) -> int:
    return max(b, ((int(v) + b - 1) // b) * b)


def _dilate(m: np.ndarray, r: float) -> np.ndarray:
    """按圆盘半径 r 膨胀 → bool。小半径用形态学；大半径用距离变换（耗时与半径无关）。"""
    r = int(round(r))
    mu = m if m.dtype == np.uint8 else m.astype(np.uint8)
    if r <= 0 or not mu.any():
        return mu > 0
    if r <= 6:
        return cv2.dilate(mu, _disk(r)) > 0
    d = cv2.distanceTransform((mu == 0).astype(np.uint8), cv2.DIST_L2, 5)
    return d <= r


def _to_lab(bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 → Lab float32（L 0..100，a/b 约 ±127），ΔE76 直接用欧氏距离。"""
    return cv2.cvtColor(bgr.astype(np.float32) * np.float32(1.0 / 255.0), cv2.COLOR_BGR2Lab)


def _delta_e(lab: np.ndarray, ref: np.ndarray) -> np.ndarray:
    d = lab - ref
    return np.sqrt(np.sum(d * d, axis=-1))


def _union_box(a: Box, b: Box) -> Box:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _intersect_box(a: Box, b: Box) -> Optional[Box]:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def _rel(box: Box, origin: Box) -> Box:
    """页面坐标框 → origin 窗口内坐标（已裁剪到窗口范围）。"""
    w, h = origin[2] - origin[0], origin[3] - origin[1]
    return clamp_box((box[0] - origin[0], box[1] - origin[1], box[2] - origin[0], box[3] - origin[1]), w, h)


def _box_mask(h: int, w: int, box: Box) -> np.ndarray:
    m = np.zeros((h, w), bool)
    x1, y1, x2, y2 = clamp_box(box, w, h)
    m[y1:y2, x1:x2] = True
    return m


def _safe_box(box: Any, w: int, h: int) -> Optional[Box]:
    """任意输入 → 合法页面框；坏数据（None / NaN / 长度不对 / 反向）返回 None 或自动纠正。"""
    try:
        x1, y1, x2, y2 = (int(round(float(v))) for v in box)
    except Exception:  # noqa: BLE001
        return None
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    b = clamp_box((x1, y1, x2, y2), w, h)
    return b if (b[2] - b[0]) >= 2 and (b[3] - b[1]) >= 2 else None


def _coerce_block(b: Any) -> Optional[TextBlock]:
    if isinstance(b, TextBlock):
        return b
    if isinstance(b, dict):
        try:
            return TextBlock.from_dict(b)
        except Exception:  # noqa: BLE001
            return None
    return None


def _as_bgr(img: Any) -> np.ndarray:
    """输入规范化为 BGR uint8 (H, W, 3)。合法输入原样返回（不拷贝）。"""
    if not isinstance(img, np.ndarray) or img.size == 0 or img.ndim not in (2, 3):
        raise HanhuaError("去字失败：页面图片为空或格式不对")
    a = img
    if a.dtype != np.uint8:
        a = np.clip(np.nan_to_num(a.astype(np.float32)), 0, 255).astype(np.uint8)
    if a.ndim == 2:
        a = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
    elif a.shape[2] == 1:
        a = cv2.cvtColor(a[:, :, 0], cv2.COLOR_GRAY2BGR)
    elif a.shape[2] == 4:
        a = np.ascontiguousarray(a[:, :, :3])
    elif a.shape[2] != 3:
        raise HanhuaError("去字失败：不支持的图片通道数")
    return np.ascontiguousarray(a)


def _runs(profile: np.ndarray) -> np.ndarray:
    """一维投影里连续非零段的长度（忽略 <2px 的碎段）。"""
    if profile.size == 0 or profile.max() <= 0:
        return np.zeros(0, np.int64)
    on = (profile > max(1.0, 0.03 * float(profile.max()))).astype(np.int8)
    d = np.diff(np.concatenate(([0], on, [0])))
    lens = np.flatnonzero(d == -1) - np.flatnonzero(d == 1)
    return lens[lens >= 2]


def _font_from_mask(m: np.ndarray, direction: str, fallback: float) -> float:
    """字号估计：竖排看列宽、横排看行高（取各段中位数），再和另一方向取小（方向判错时兜底）。"""
    if m.size == 0 or not m.any():
        return fallback
    prim = _runs(m.sum(axis=0 if direction == "v" else 1).astype(np.float32))
    sec = _runs(m.sum(axis=1 if direction == "v" else 0).astype(np.float32))
    cands = [float(np.median(r)) for r in (prim, sec) if r.size]
    return min(cands) if cands else fallback


def _normconv(lab: np.ndarray, weight: np.ndarray, sigma: float, fallback: np.ndarray) -> np.ndarray:
    """归一化卷积：只用 weight 处的已知背景像素，平滑插值出整窗的背景色（能穿过文字区域、跟随渐变）。
    sigma 大时先缩小再模糊，耗时与 sigma 无关。"""
    h, w = weight.shape
    ds = max(1, int(sigma / 3.0))
    wf = weight.astype(np.float32)
    num = lab * wf[..., None]
    if ds > 1 and h // ds >= 2 and w // ds >= 2:
        sz = (max(1, w // ds), max(1, h // ds))
        num = cv2.resize(num, sz, interpolation=cv2.INTER_AREA)
        wf = cv2.resize(wf, sz, interpolation=cv2.INTER_AREA)
    else:
        ds = 1
    s = max(0.8, sigma / ds)
    num = cv2.GaussianBlur(num, (0, 0), s, borderType=cv2.BORDER_REPLICATE)
    den = cv2.GaussianBlur(wf, (0, 0), s, borderType=cv2.BORDER_REPLICATE)
    eps = np.float32(1e-3)
    out = (num + eps * fallback.astype(np.float32)) / (den[..., None] + eps)
    if ds > 1:
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
    return out


def _push_pull(img: np.ndarray, known: np.ndarray) -> Optional[np.ndarray]:
    """push-pull 金字塔插值：用 known 像素平滑填满整窗（近似调和插值，边界处与已知像素连续）。"""
    if not known.any():
        return None
    v = img.astype(np.float32) * known[..., None]
    w = known.astype(np.float32)
    pyr = [(v, w)]
    while min(w.shape) > 4 and len(pyr) < 14:
        v = cv2.pyrDown(v)
        w = cv2.pyrDown(w)
        pyr.append((v, w))
    v, w = pyr[-1]
    est = v / np.maximum(w, 1e-6)[..., None]
    low = w < 1e-4
    if low.any():
        est[low] = img[known].astype(np.float32).mean(axis=0)
    for k in range(len(pyr) - 2, -1, -1):
        v, w = pyr[k]
        up = cv2.pyrUp(est, dstsize=(w.shape[1], w.shape[0]))
        a = np.clip(w * 4.0, 0.0, 1.0)[..., None]        # 本层已知像素占比 ≥25% 就基本采用本层
        est = a * (v / np.maximum(w, 1e-6)[..., None]) + (1.0 - a) * up
    return est


def _fill_small_holes(m: np.ndarray, max_area: float) -> np.ndarray:
    """把 mask 内部（不碰窗口边）面积小于 max_area 的洞填上。"""
    inv = (m == 0).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    if n <= 1:
        return m
    h, w = m.shape
    x, y, ww, hh, area = st[:, 0], st[:, 1], st[:, 2], st[:, 3], st[:, 4]
    keep = (area <= max_area) & (x > 0) & (y > 0) & (x + ww < w) & (y + hh < h)
    keep[0] = False
    if not keep.any():
        return m
    out = m.copy()
    out[keep[lab]] = 255
    return out


def _rounded_rect(h: int, w: int, box: Box, radius: int) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    x1, y1, x2, y2 = clamp_box(box, w, h)
    if x2 - x1 < 1 or y2 - y1 < 1:
        return m
    r = int(max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2)))
    cv2.rectangle(m, (x1 + r, y1), (x2 - 1 - r, y2 - 1), 255, -1)
    cv2.rectangle(m, (x1, y1 + r), (x2 - 1, y2 - 1 - r), 255, -1)
    if r > 0:
        for cx, cy in ((x1 + r, y1 + r), (x2 - 1 - r, y1 + r), (x1 + r, y2 - 1 - r), (x2 - 1 - r, y2 - 1 - r)):
            cv2.circle(m, (cx, cy), r, 255, -1)
    return m


def _feather_alpha(mask: np.ndarray) -> np.ndarray:
    """羽化系数（float32，mask 外为 0）：只在 mask 内侧最外 1px 取 _FEATHER_BAND，其余为 1。
    mask 构造时已比「文字 + 抗锯齿」多膨胀 1px，所以文字像素本身始终是全替换，不会留下轮廓残影。"""
    alpha = np.zeros(mask.shape, np.float32)
    mu = (mask > 0).astype(np.uint8)
    x, y, w, h = cv2.boundingRect(mu)
    if w == 0 or h == 0:
        return alpha
    H, W = mask.shape
    x1, y1, x2, y2 = max(0, x - 1), max(0, y - 1), min(W, x + w + 1), min(H, y + h + 1)
    sub = mu[y1:y2, x1:x2]
    # 页面边缘按「延续」处理（BORDER_REPLICATE）：贴着页边的文字没有外侧可接缝，不必羽化
    core = cv2.erode(sub, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_REPLICATE)
    alpha[y1:y2, x1:x2] = np.where(core > 0, np.float32(1.0), np.float32(_FEATHER_BAND)) * sub
    return alpha


def _blend(work: np.ndarray, win: Box, m: np.ndarray, fill: np.ndarray, alpha: np.ndarray) -> None:
    """在窗口 win 内、m 为真处，把 fill 按羽化系数合成进 work（原地）。"""
    x1, y1, x2, y2 = win
    dst = work[y1:y2, x1:x2]
    a = alpha[y1:y2, x1:x2][m][:, None]
    src = dst[m].astype(np.float32)
    dst[m] = np.clip(a * fill[m].astype(np.float32) + (1.0 - a) * src + 0.5, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------------------------
# 中间结果
# ---------------------------------------------------------------------------------------------
@dataclass
class _Interior:
    """气泡内部（search 窗口坐标）。"""

    box: Box                 # 搜索窗口（页面坐标）
    filled: np.ndarray       # bool：气泡内部（含文字，已填洞）
    eroded: np.ndarray       # bool：内部再腐蚀若干 px，文字 mask 只能落在这里（远离气泡描边）
    bg: np.ndarray           # bool：洪水填充到的纯背景像素（不含文字），背景估计 / 取色用


@dataclass
class _Region:
    """一个文字块的分析结果（win 窗口坐标）。"""

    index: int               # blocks 里的下标
    win: Box                 # 分析窗口（页面坐标）
    mask: np.ndarray         # uint8 0/255：该块文字像素 mask
    valid: np.ndarray        # bool：允许改动 / 取样的区域（气泡内部腐蚀后，或整窗）
    font: float              # 估计字号（px）
    in_bubble: bool          # 成功求出了平滑的气泡内部（允许 smooth 填充）
    fallback: bool = False   # 退回了圆角矩形 mask


# ---------------------------------------------------------------------------------------------
# Inpainter
# ---------------------------------------------------------------------------------------------
class Inpainter:
    """去字器。构造一次、反复调用；可被多个线程同时调用（内部可变状态都有锁）。"""

    def __init__(self, rt: Optional["Runtime"], profile: "Profile", settings: dict) -> None:
        self.rt = rt
        self.profile = profile
        inp = settings.get("inpaint", {}) if isinstance(settings, dict) else {}
        inp = inp if isinstance(inp, dict) else {}
        mode = str(inp.get("mode", "auto"))
        if mode not in ("auto", "fill", "lama"):
            log.warning("未知去字模式 %r，改用 auto", mode)
            mode = "auto"
        self.mode = mode
        try:
            max_side = int(inp.get("max_side") or 0)
        except (TypeError, ValueError):
            max_side = 0
        if max_side <= 0:
            max_side = int(getattr(profile, "inpaint_max_side", 768) or 768)
        bucket = int(getattr(profile, "inpaint_bucket", 64) or 64)
        self.bucket = _ceil_to(max(8, bucket), 8)            # LaMa 下采样 8 倍：尺寸必须是 8 的倍数
        self.max_side = max(self.bucket * 2, max_side)
        self._lock = threading.Lock()
        self._sticky: Optional[tuple[int, int]] = None      # CUDA 粘性画布（只增不减的 LaMa 输入尺寸）
        self._last_shape: Optional[tuple[int, int]] = None
        self._seen_shapes: set[tuple[int, int]] = set()
        self._failed: set[str] = set()
        self._warned_lama_fallback = False
        self.shape_log: list[dict] = []                     # 每种新尺寸首跑耗时（诊断用，最多 64 条）
        self.last_stats: dict[str, Any] = {}

    # ------------------------------------------------------------------ 公共接口
    def text_mask(self, img: np.ndarray, blocks: Sequence[TextBlock]) -> np.ndarray:
        """uint8 HxW（0/255），所有 skip=False 的块的文字像素 mask 的并集。"""
        src = _as_bgr(img)
        union, _ = self._analyze(src, blocks)
        return union

    def clean(self, img: np.ndarray, blocks: Sequence[TextBlock]) -> tuple[np.ndarray, np.ndarray]:
        """返回 (去字后的图, 实际使用的 mask)。不变量：mask==0 的像素与输入逐位相同。"""
        t0 = time.perf_counter()
        src = _as_bgr(img)
        union, regions = self._analyze(src, blocks)
        t1 = time.perf_counter()
        stats: dict[str, Any] = {"regions": len(regions), "flat": 0, "smooth": 0, "lama": 0, "fallback_rect": 0,
                                 "lama_calls": 0, "lama_ms": 0.0, "shape_switches": 0, "new_shapes": 0,
                                 "methods": []}
        if not union.any():
            stats.update(analyze_ms=(t1 - t0) * 1000, fill_ms=0.0, total_ms=(t1 - t0) * 1000)
            self.last_stats = stats
            return src.copy(), union

        work = src.copy()
        alpha = _feather_alpha(union)
        lama_mask = np.zeros(union.shape, np.uint8)
        lama_font: list[float] = []
        for r in regions:
            stats["fallback_rect"] += int(r.fallback)
            method = "lama"
            try:
                method = self._fill_region(src, work, union, alpha, r)
            except Exception:  # noqa: BLE001 - 单块失败不影响整页
                log.exception("块 %d 填充失败，改用 LaMa", r.index)
                method = "lama"
            if method == "lama":
                x1, y1, x2, y2 = r.win
                np.maximum(lama_mask[y1:y2, x1:x2], r.mask, out=lama_mask[y1:y2, x1:x2])
                lama_font.append(r.font)
            stats[method] += 1
            stats["methods"].append((r.index, method))
        t2 = time.perf_counter()
        if lama_mask.any():
            try:
                self._lama_fill(work, lama_mask, alpha, float(np.median(lama_font)) if lama_font else 24.0, stats)
            except Exception:  # noqa: BLE001
                log.exception("LaMa 去字失败，改用传统修补")
                self._classic_fill(work, lama_mask, alpha)
        t3 = time.perf_counter()
        # 不变量：mask==0 的像素与输入逐位相同
        out = np.where(union[..., None] > 0, work, src)
        stats.update(analyze_ms=(t1 - t0) * 1000, fill_ms=(t2 - t1) * 1000, lama_total_ms=(t3 - t2) * 1000,
                     total_ms=(time.perf_counter() - t0) * 1000)
        self.last_stats = stats
        log.debug("去字完成：%s", {k: v for k, v in stats.items() if k != "methods"})
        return out, union

    def warmup(self) -> None:
        """可选：在后台线程预热 LaMa（加载模型 + 首次推理）。
        CUDA 直接用最大画布（max_side 分桶后的正方形）预热：之后所有裁剪都塞进这个画布，整个会话不再换尺寸。"""
        m = self._model(_LAMA_KEY)
        if m is None:
            return
        if m.provider == "CUDAExecutionProvider":
            s = _ceil_to(self.max_side, self.bucket)
        else:
            s = min(self.max_side, _ceil_to(512, self.bucket))
        img = np.full((s, s, 3), 255, np.uint8)
        mask = np.zeros((s, s), bool)
        mask[s // 3: s // 2, s // 3: s // 2] = True
        try:
            self._run_lama(m, img, mask)
        except Exception as e:  # noqa: BLE001
            log.warning("LaMa 预热失败：%s", e)
        # 气泡外文字要用的 ppocr_det（GPU 上是固定画布，跑一次就把这个尺寸预热好）
        self._db_region(np.full((64, 64, 3), 255, np.uint8), 20.0, np.ones((64, 64), bool))

    # ------------------------------------------------------------------ 模型
    def _model(self, key: str) -> Optional["Model"]:
        if self.rt is None or key in self._failed:
            return None
        try:
            return self.rt.model(key)
        except Exception as e:  # noqa: BLE001
            with self._lock:
                if key not in self._failed:
                    log.warning("模型 %s 不可用，改用备用方案：%s", key, e)
                self._failed.add(key)
            return None

    # ------------------------------------------------------------------ mask 分析
    def _analyze(self, img: np.ndarray, blocks: Sequence[Any]) -> tuple[np.ndarray, list[_Region]]:
        h, w = img.shape[:2]
        union = np.zeros((h, w), np.uint8)
        regions: list[_Region] = []
        active: list[Box] = []
        kept: list[Box] = []
        for i, raw in enumerate(blocks or []):
            tb = _coerce_block(raw)
            if tb is None:
                log.warning("第 %d 个文字块格式不对，已跳过", i)
                continue
            box = _safe_box(tb.box, w, h)
            if box is None:
                continue
            if tb.skip:
                kept.append(box)
                continue
            active.append(box)
            try:
                r = self._analyze_block(img, tb, box, i)
            except Exception:  # noqa: BLE001 - 单块分析失败 → 退回圆角矩形，不让整页崩
                log.exception("块 %d 的文字 mask 计算失败，退回圆角矩形", i)
                r = self._rect_region(img, tb, box, i)
            if r is not None and r.mask.any():
                regions.append(r)
        # 保留原文（skip）的块：它的框不被别的块擦到（与要擦的框重叠处以要擦的为准）
        protect = None
        if kept and regions:
            protect = np.zeros((h, w), bool)
            for b in kept:
                protect[b[1]:b[3], b[0]:b[2]] = True
            for b in active:
                protect[b[1]:b[3], b[0]:b[2]] = False
        for r in regions:
            x1, y1, x2, y2 = r.win
            if protect is not None:
                r.mask[protect[y1:y2, x1:x2]] = 0
            np.maximum(union[y1:y2, x1:x2], r.mask, out=union[y1:y2, x1:x2])
        return union, [r for r in regions if r.mask.any()]

    def _crude_font(self, img: np.ndarray, box: Box, direction: str) -> float:
        """粗估字号：框内灰度 Otsu，少数类当文字，按方向投影。只用来定窗口 / 模型缩放。"""
        x1, y1, x2, y2 = box
        short = float(max(4, min(x2 - x1, y2 - y1)))
        g = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        if g.size < 16 or int(g.max()) - int(g.min()) < 16:
            return short
        _, th = cv2.threshold(g, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        fg = th.astype(bool)
        if fg.mean() > 0.5:
            fg = ~fg
        return float(np.clip(_font_from_mask(fg, direction, short), 4.0, short))

    def _analyze_block(self, img: np.ndarray, tb: TextBlock, box: Box, idx: int) -> Optional[_Region]:
        h, w = img.shape[:2]
        bw, bh = box[2] - box[0], box[3] - box[1]
        short = float(min(bw, bh))
        direction = tb.direction if tb.direction in ("v", "h") else "v"
        fc = self._crude_font(img, box, direction)

        # ---- 气泡内部（气泡框已知时在气泡框里找；未知时在文字框附近找；找不到就当气泡外文字处理）
        interior: Optional[_Interior] = None
        bubble_rect: Optional[Box] = None
        if tb.kind != "free":
            bb = _safe_box(tb.bubble_box, w, h) if tb.bubble_box is not None else None
            if bb is not None:
                bubble_rect = bb
                search = expand_box(_union_box(bb, box), 3, w, h)
            else:
                search = expand_box(box, max(8.0, 0.35 * max(bw, bh)), w, h)
            interior = self._bubble_interior(img, search, box, fc)

        # ---- 分析窗口
        margin = int(round(max(4.0, _WIN_MARGIN * fc)))
        win = expand_box(box, margin, w, h)
        if interior is not None:
            ys, xs = np.nonzero(interior.filled)
            ib = (interior.box[0] + int(xs.min()), interior.box[1] + int(ys.min()),
                  interior.box[0] + int(xs.max()) + 1, interior.box[1] + int(ys.max()) + 1)
            cut = _intersect_box(win, ib)
            if cut is None:
                interior = None
            else:
                win = cut
        wx1, wy1, wx2, wy2 = win
        crop = img[wy1:wy2, wx1:wx2]
        hh, ww = crop.shape[:2]
        if hh < 2 or ww < 2:
            return None
        lab = _to_lab(crop)
        rb = _rel(box, win)
        inbox = _box_mask(hh, ww, rb)
        near = _box_mask(hh, ww, (rb[0] - margin // 2, rb[1] - margin // 2, rb[2] + margin // 2, rb[3] + margin // 2))

        db_region: Optional[np.ndarray] = None
        if interior is not None:
            ox, oy = win[0] - interior.box[0], win[1] - interior.box[1]
            valid = interior.eroded[oy:oy + hh, ox:ox + ww].copy()
            bgw = interior.bg[oy:oy + hh, ox:ox + ww] & valid
        else:
            valid = np.ones((hh, ww), bool)
            if bubble_rect is not None:      # 气泡内部没找到（描边有缺口 / 无描边）：至少不出气泡框
                bx = _rel(bubble_rect, win)
                s = max(2, int(round(0.03 * min(bubble_rect[2] - bubble_rect[0], bubble_rect[3] - bubble_rect[1]))))
                valid = _box_mask(hh, ww, (bx[0] + s, bx[1] + s, bx[2] - s, bx[3] - s))
                if not (valid & inbox).any():
                    valid = np.ones((hh, ww), bool)
            # 外圈背景 → 初始文字候选（排除出背景估计）
            ring0 = valid & ~near
            if ring0.sum() < 16:
                ring0 = valid & ~inbox
            if ring0.sum() < 16:
                ring0 = valid
            med0 = np.median(lab[ring0], axis=0)
            d0 = _delta_e(lab, med0)
            v0 = d0[inbox & valid]
            c0 = float(np.percentile(v0, 99)) if v0.size else 0.0
            cand = d0 > max(10.0, 0.3 * c0)
            if tb.kind == "free":
                db_region = self._db_region(crop, fc, near)
                if db_region is not None:
                    cand |= db_region
            cand = cv2.dilate(cand.astype(np.uint8), _disk(2)) > 0
            bgw = valid & ~cand

        # ---- 局部背景模型 → 与背景的色差
        if bgw.sum() >= 8:
            fb = np.median(lab[bgw], axis=0)
        else:
            fb = np.median(lab.reshape(-1, 3), axis=0)
            bgw = np.zeros_like(bgw)
        bgm = _normconv(lab, bgw, max(2.0, _BG_SIGMA * fc), fb)
        dist = _delta_e(lab, bgm)
        vals = dist[inbox & valid]
        contrast = float(np.percentile(vals, 99)) if vals.size else 0.0
        t_hi = float(np.clip(0.45 * contrast, 10.0, 40.0))
        t_lo = float(np.clip(0.12 * contrast, 4.0, 12.0))
        strong = (dist > t_hi) & valid & near
        if db_region is not None:
            strong &= db_region
        weak = (dist > t_lo) & valid

        # ---- 字号精估 → 生长 / 膨胀半径
        f = float(np.clip(_font_from_mask(strong & inbox, direction, fc), 4.0, max(4.0, short * 1.1)))
        r_grow = max(2, int(round(_GROW * f)))
        r_aa = max(2, int(round(_AA_DILATE * f)))
        m0 = strong | (weak & _dilate(strong, r_grow))
        if tb.kind == "free" or interior is None:
            m0 = self._drop_outside(m0, _box_mask(hh, ww, (rb[0] - r_aa, rb[1] - r_aa, rb[2] + r_aa, rb[3] + r_aa)))
            m0 &= near            # 气泡外：笔画不会远离文字框，顺着衣褶 / 阴影长出去的部分裁掉
        # 膨胀 = 抗锯齿 / 外发光半径 + 1px 羽化带（_feather_alpha 只在最外 1px 半替换）
        m1 = _dilate(m0, r_aa + 1).astype(np.uint8) * 255
        m1 = _fill_small_holes(m1, (_HOLE * f) ** 2)
        m1[~valid] = 0

        # ---- 面积异常 → 圆角矩形
        ratio = float(np.count_nonzero(m1[inbox])) / float(max(1, bw * bh))
        fallback = False
        if ratio > _MASK_MAX_RATIO or ratio < _MASK_MIN_RATIO:
            log.debug("块 %d mask 占比 %.3f 异常，退回圆角矩形", idx, ratio)
            rr = (rb[0] - r_aa, rb[1] - r_aa, rb[2] + r_aa, rb[3] + r_aa)
            m1 = _rounded_rect(hh, ww, rr, int(round(0.25 * min(bw, bh))))
            m1[~valid] = 0
            fallback = True
        return _Region(index=idx, win=win, mask=m1, valid=valid, font=f,
                       in_bubble=interior is not None, fallback=fallback)

    @staticmethod
    def _drop_outside(m: np.ndarray, keep_zone: np.ndarray) -> np.ndarray:
        """去掉「大半在文字框外」的连通块（肩膀 / 远景等被色差误选的东西）。"""
        n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
        if n <= 1:
            return m
        inside = np.bincount(lab[keep_zone].ravel(), minlength=n)
        keep = inside >= 0.5 * st[:, 4]
        keep[0] = False
        return keep[lab]

    def _rect_region(self, img: np.ndarray, tb: TextBlock, box: Box, idx: int) -> Optional[_Region]:
        """分析出错时的兜底：文字框略放大的圆角矩形（气泡内文字不超出气泡框）。"""
        h, w = img.shape[:2]
        bw, bh = box[2] - box[0], box[3] - box[1]
        pad = max(2, int(round(0.08 * min(bw, bh))))
        win = expand_box(box, pad, w, h)
        hh, ww = win[3] - win[1], win[2] - win[0]
        m = _rounded_rect(hh, ww, _rel(expand_box(box, pad, w, h), win), int(round(0.25 * min(bw, bh))))
        valid = np.ones((hh, ww), bool)
        if tb.kind != "free" and tb.bubble_box is not None:
            bb = _safe_box(tb.bubble_box, w, h)
            if bb is not None:
                inner = _box_mask(hh, ww, _rel(expand_box(bb, -max(2, pad), w, h), win))
                if (inner & (m > 0)).any():
                    valid = inner
                    m[~valid] = 0
        return _Region(index=idx, win=win, mask=m, valid=valid, font=float(max(4, min(bw, bh))),
                       in_bubble=False, fallback=True)

    def _bubble_interior(self, img: np.ndarray, search: Box, tbox: Box, font: float = 0.0) -> Optional[_Interior]:
        """在 search 窗口里用「浮动范围洪水填充」找气泡内部。
        相邻像素差 ≤3 才连通：能顺着渐变走，会在描边 / 文字边缘处停下；漏到窗口外（无封闭描边）→ None。
        font：粗估字高，用于补上「字列贴着描边」时形成的海湾（见下方闭运算）。"""
        H, W = img.shape[:2]
        x1, y1, x2, y2 = search
        crop = img[y1:y2, x1:x2]
        h, w = crop.shape[:2]
        if h < 8 or w < 8:
            return None
        sm = cv2.GaussianBlur(crop, (3, 3), 0)
        gray = cv2.cvtColor(sm, cv2.COLOR_BGR2GRAY)
        tx1, ty1, tx2, ty2 = _rel(tbox, search)
        if tx2 - tx1 < 2 or ty2 - ty1 < 2:
            return None
        e = max(3, int(round(0.08 * min(tx2 - tx1, ty2 - ty1))))
        outer = _box_mask(h, w, (tx1 - e, ty1 - e, tx2 + e, ty2 + e))
        inner = _box_mask(h, w, (tx1, ty1, tx2, ty2))
        band = outer & ~inner
        grad = np.abs(cv2.Sobel(gray, cv2.CV_16S, 1, 0)).astype(np.int32) + np.abs(cv2.Sobel(gray, cv2.CV_16S, 0, 1))
        calm = grad < 24
        ref = band & calm
        if ref.sum() < 4:
            ref = (band | inner) & calm
        if ref.sum() < 4:
            return None
        med = float(np.median(gray[ref]))
        close = np.abs(gray.astype(np.float32) - med) <= 6
        seeds: list[tuple[int, int]] = []
        for zone, k in ((band & calm & close, 24), (inner & calm & close, 8)):
            ys, xs = np.nonzero(zone)
            if ys.size:
                step = max(1, ys.size // k)
                seeds.extend(zip(ys[::step][:k].tolist(), xs[::step][:k].tolist()))
        if not seeds:
            return None
        ff = np.zeros((h + 2, w + 2), np.uint8)
        label = 0
        tol = (3, 3, 3)
        for sy, sx in seeds:
            if ff[sy + 1, sx + 1]:
                continue
            label += 1
            if label > 250:
                break
            cv2.floodFill(sm, ff, (int(sx), int(sy)), (0, 0, 0), tol, tol,
                          4 | cv2.FLOODFILL_MASK_ONLY | (label << 8))
        if label == 0:
            return None
        lab = ff[1:-1, 1:-1]
        areas = np.bincount(lab.ravel(), minlength=label + 1)
        sides = []
        perim = 0
        if y1 > 0:
            sides.append(lab[0, :]); perim += w
        if y2 < H:
            sides.append(lab[-1, :]); perim += w
        if x1 > 0:
            sides.append(lab[:, 0]); perim += h
        if x2 < W:
            sides.append(lab[:, -1]); perim += h
        touch = np.bincount(np.concatenate(sides), minlength=label + 1) if sides else np.zeros(label + 1, np.int64)
        leak = touch > max(3.0, 0.01 * perim)
        ok = (~leak) & (areas >= max(16.0, 0.005 * h * w))
        ok[0] = False
        if not ok.any():
            return None
        bg = ok[lab]
        filled = np.zeros((h, w), np.uint8)
        cnts, _ = cv2.findContours(bg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(filled, cnts, -1, 255, thickness=-1)
        # 字列紧贴气泡描边时，文字和描边连成一片：文字区域在白底上是「海湾」而不是洞，填洞补不上，
        # 结果文字被当成气泡外、mask 擦不到（p016 右侧气泡）。按字号闭运算把海湾补平再填一次洞；
        # 闭运算只填白底自身的凹陷，越不过描边（描边外不是白底），不会伤到气泡外的画面。
        if font > 0:
            r_close = max(3, int(round(0.65 * font)))
            closed = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, _disk(r_close))
            cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filled = np.zeros((h, w), np.uint8)
            cv2.drawContours(filled, cnts, -1, 255, thickness=-1)
        if float(filled[ty1:ty2, tx1:tx2].mean()) / 255.0 < 0.5:   # 内部没盖住文字框 → 找错了
            return None
        r_er = max(2, int(round(0.008 * min(h, w))))
        eroded = cv2.erode(filled, _disk(r_er)) > 0
        bg_core = cv2.erode(bg.astype(np.uint8), _disk(1)) > 0
        return _Interior(box=search, filled=filled > 0, eroded=eroded, bg=bg_core)

    def _db_region(self, crop: np.ndarray, font: float, near: np.ndarray) -> Optional[np.ndarray]:
        """PP-OCR DB 概率图 → 文字区域（膨胀到覆盖整个字形 + 描边）。模型不可用 / 没检出 → None。"""
        m = self._model(_DB_KEY)
        if m is None:
            return None
        h, w = crop.shape[:2]
        try:
            s = 1.0
            if font > 64:
                s = 48.0 / font                   # 特大字（拟声词）缩到 DB 擅长的字号
            elif font < 12:
                s = min(2.0, 16.0 / font)         # 特小字放大
            gpu = m.device != "cpu"
            lim = _DB_CANVAS if gpu else 1024     # GPU：固定画布（换尺寸代价大）；CPU：按 32 分桶（省计算）
            s = min(s, lim / float(max(h, w)))
            nh, nw = min(lim, max(16, int(round(h * s)))), min(lim, max(16, int(round(w * s))))
            x = crop if (nh, nw) == (h, w) else cv2.resize(
                crop, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
            ch, cw = (lim, lim) if gpu else (_ceil_to(nh, 32), _ceil_to(nw, 32))
            x = cv2.copyMakeBorder(x, 0, ch - nh, 0, cw - nw, cv2.BORDER_REPLICATE)
            t = (x.astype(np.float32) * np.float32(1.0 / 255.0) - _DB_MEAN) / _DB_STD
            t = np.ascontiguousarray(t.transpose(2, 0, 1)[None])
            out = m.run({m.input_names[0]: t})[0]
            p = np.asarray(out, np.float32).reshape(out.shape[-2], out.shape[-1])[:nh, :nw]
            if (nh, nw) != (h, w):
                p = cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR)
        except Exception as e:  # noqa: BLE001
            log.warning("ppocr_det 推理失败（不影响去字）：%s", e)
            return None
        reg = p > _DB_THRESH
        if not (reg & near).any():
            return None
        return _dilate(reg, 0.6 * font + 2)

    # ------------------------------------------------------------------ 填充
    def _fill_region(self, src: np.ndarray, work: np.ndarray, union: np.ndarray, alpha: np.ndarray,
                     r: _Region) -> str:
        """按外环判断填充方式；flat / smooth 直接写进 work，返回所用方式（lama 由调用方统一处理）。"""
        if self.mode == "lama":
            return "lama"
        x1, y1, x2, y2 = r.win
        crop = src[y1:y2, x1:x2]
        um = union[y1:y2, x1:x2] > 0
        m = r.mask > 0
        rr = max(2, int(round(_RING * r.font)))
        ring = (cv2.dilate(r.mask, _disk(rr)) > 0) & ~um & r.valid
        n = int(ring.sum())
        if n >= 12:
            px = crop[ring]
            lab = _to_lab(px.reshape(-1, 1, 3)).reshape(-1, 3)
            med = np.median(lab, axis=0)
            if (float((_delta_e(lab, med) < _FLAT_DE).mean()) >= _FLAT_FRAC
                    and self._uniform(ring, lab, med, r.font)):
                color = np.median(px, axis=0).round().astype(np.uint8)
                fill = np.empty_like(crop)
                fill[:] = color
                _blend(work, r.win, m, fill, alpha)
                return "flat"
            smooth_ok = self.mode == "fill" or (r.in_bubble and self._is_smooth(crop, ring, r.font, med))
        else:
            smooth_ok = self.mode == "fill"
        if not smooth_ok:
            return "lama"
        known = r.valid & ~um
        if known.sum() < 4:
            known = ~um
        est = _push_pull(work[y1:y2, x1:x2], known)
        if est is None:
            return "lama" if self.mode != "fill" else self._fill_median_any(work, r, um, alpha)
        _blend(work, r.win, m, np.clip(est + 0.5, 0, 255).astype(np.uint8), alpha)
        return "smooth"

    @staticmethod
    def _fill_median_any(work: np.ndarray, r: _Region, um: np.ndarray, alpha: np.ndarray) -> str:
        """fill 模式下连已知像素都没有时的最后兜底：窗口内非 mask 像素的中位色。"""
        x1, y1, x2, y2 = r.win
        crop = work[y1:y2, x1:x2]
        px = crop[~um] if (~um).any() else crop.reshape(-1, 3)
        fill = np.empty_like(crop)
        fill[:] = np.median(px, axis=0).round().astype(np.uint8)
        _blend(work, r.win, r.mask > 0, fill, alpha)
        return "flat"

    @staticmethod
    def _uniform(ring: np.ndarray, lab: np.ndarray, med: np.ndarray, font: float) -> bool:
        """外环按 _FLAT_CELL 字高的网格分块，每块（≥8 像素）的中位色都接近整体中位色 → 没有渐变。
        lab 与 ring 的 np.nonzero 顺序一致（行优先）。中位数对 JPEG 噪声不敏感，只反映空间上的色彩漂移。"""
        ys, xs = np.nonzero(ring)
        cs = max(4, int(round(_FLAT_CELL * font)))
        cell = (ys // cs).astype(np.int64) * 65536 + (xs // cs)
        _, inv, counts = np.unique(cell, return_inverse=True, return_counts=True)
        inv = inv.ravel()
        for k in np.flatnonzero(counts >= 8):
            if float(_delta_e(np.median(lab[inv == k], axis=0), med)) > _FLAT_CELL_DE:
                return False
        return True

    @staticmethod
    def _is_smooth(crop: np.ndarray, ring: np.ndarray, font: float, med: np.ndarray) -> bool:
        """外环像素与其局部平滑估计的偏差 p95 < 阈值 → 背景是平滑渐变（可插值），否则有纹理（交给 LaMa）。"""
        lab = _to_lab(crop)
        pred = _normconv(lab, ring, max(1.5, 0.3 * font), med)
        res = _delta_e(lab[ring], pred[ring])
        return bool(res.size) and float(np.percentile(res, 95)) < _SMOOTH_DE

    # ------------------------------------------------------------------ LaMa
    def _lama_fill(self, work: np.ndarray, lama_mask: np.ndarray, alpha: np.ndarray, font: float,
                   stats: dict[str, Any]) -> None:
        m = self._model(_LAMA_KEY)
        if m is None:
            if not self._warned_lama_fallback:
                log.warning("LaMa 不可用，复杂背景改用传统修补（效果会差一些）")
                self._warned_lama_fallback = True
            self._classic_fill(work, lama_mask, alpha)
            return
        h, w = lama_mask.shape
        n, labels, st, _ = cv2.connectedComponentsWithStats(lama_mask, connectivity=8)
        clusters = self._cluster(st[1:], n - 1)
        remaining = lama_mask > 0
        for ids, region in clusters:
            crop_box = self._context_box(region, w, h)
            x1, y1, x2, y2 = crop_box
            own = np.isin(labels[y1:y2, x1:x2], np.asarray(ids, np.int32) + 1)
            own &= remaining[y1:y2, x1:x2]
            if not own.any():
                continue
            inp_mask = remaining[y1:y2, x1:x2].copy()
            t = time.perf_counter()
            try:
                out = self._run_lama(m, work[y1:y2, x1:x2], inp_mask, stats)
            except Exception as e:  # noqa: BLE001 - 显存不足等：缩小一半重试，再不行用传统修补
                log.warning("LaMa 推理失败（%s），缩小重试", e)
                try:
                    out = self._run_lama(m, work[y1:y2, x1:x2], inp_mask, stats, extra_scale=0.5)
                except Exception as e2:  # noqa: BLE001
                    log.warning("LaMa 再次失败（%s），该区域改用传统修补", e2)
                    out = None
            stats["lama_ms"] += (time.perf_counter() - t) * 1000
            if out is None:
                sub = np.zeros_like(lama_mask)
                sub[y1:y2, x1:x2][own] = 255
                self._classic_fill(work, sub, alpha)
            else:
                _blend(work, crop_box, own, out, alpha)
            remaining[y1:y2, x1:x2][own] = False
        if remaining.any():   # 理论上不会发生：漏网的像素也补上
            self._classic_fill(work, remaining.astype(np.uint8) * 255, alpha)

    def _cluster(self, st: np.ndarray, n: int) -> list[tuple[list[int], Box]]:
        """连通域 → 裁剪组：上下文范围重叠的合并，合并后区域不超过 max_side（超了宁可分开跑，避免缩小变糊）。"""
        limit = self.max_side - 64
        items: list[tuple[list[int], Box]] = []
        order = np.argsort(-st[:, 4]) if n else []
        for i in order:
            x, y, ww, hh = (int(v) for v in st[i, :4])
            items.append(([int(i)], (x, y, x + ww, y + hh)))

        def ctx(b: Box) -> Box:
            mg = max(32.0, 0.25 * ((b[2] - b[0]) + (b[3] - b[1])))
            return (int(b[0] - mg), int(b[1] - mg), int(b[2] + mg), int(b[3] + mg))

        changed = True
        while changed and len(items) > 1:
            changed = False
            out: list[tuple[list[int], Box]] = []
            for ids, box in items:
                for k, (ids2, box2) in enumerate(out):
                    if _intersect_box(ctx(box), ctx(box2)) is None:
                        continue
                    u = _union_box(box, box2)
                    if max(u[2] - u[0], u[3] - u[1]) > limit:
                        continue
                    out[k] = (ids2 + ids, u)
                    changed = True
                    break
                else:
                    out.append((ids, box))
            items = out
        items.sort(key=lambda it: (it[1][1], it[1][0]))
        return items

    def _context_box(self, region: Box, w: int, h: int) -> Box:
        """上下文边距 ≈ 区域尺寸 0.5 倍（两边各 0.25×(宽+高)，至少 32px）；超过 max_side 先压缩边距。"""
        rw, rh = region[2] - region[0], region[3] - region[1]
        mg = max(32.0, 0.25 * (rw + rh))
        mx = min(mg, max(32.0, (self.max_side - rw) / 2.0))
        my = min(mg, max(32.0, (self.max_side - rh) / 2.0))
        return clamp_box((int(region[0] - mx), int(region[1] - my), int(math.ceil(region[2] + mx)),
                          int(math.ceil(region[3] + my))), w, h)

    def _canvas(self, hs: int, ws: int, sticky: bool) -> tuple[int, int]:
        """LaMa 输入尺寸：分桶到 bucket 的倍数。
        CUDA（sticky）：画布只增不减 —— 实测换一次尺寸（哪怕是见过的尺寸）多花 100~200ms，
        约等于多跑一次最大画布，所以小裁剪直接塞进已有的大画布（reflect pad），永不来回切换。"""
        b = self.bucket
        need = (_ceil_to(hs, b), _ceil_to(ws, b))
        if not sticky:
            return need
        with self._lock:
            s = self._sticky
            s = need if s is None else (max(s[0], need[0]), max(s[1], need[1]))
            self._sticky = s
            return s

    def _run_lama(self, m: "Model", crop: np.ndarray, mask: np.ndarray, stats: Optional[dict] = None,
                  extra_scale: float = 1.0) -> np.ndarray:
        """一次 LaMa：等比缩小（不变形）→ reflect pad 到分桶尺寸 → 推理 → 去 pad → 缩回原尺寸。"""
        h, w = crop.shape[:2]
        scale = min(1.0, self.max_side / float(max(h, w))) * extra_scale
        if scale < 0.999:
            sw, sh = max(8, int(round(w * scale))), max(8, int(round(h * scale)))
            img_s = cv2.resize(crop, (sw, sh), interpolation=cv2.INTER_AREA)
            m_s = cv2.resize(mask.astype(np.uint8) * 255, (sw, sh), interpolation=cv2.INTER_AREA) > 16
            m_s = cv2.dilate(m_s.astype(np.uint8), _disk(1)) > 0
        else:
            img_s, m_s = crop, mask
        hs, ws = img_s.shape[:2]
        sticky = m.provider == "CUDAExecutionProvider"
        ch, cw = self._canvas(hs, ws, sticky)
        top, left = (ch - hs) // 2, (cw - ws) // 2
        pads = (top, ch - hs - top, left, cw - ws - left)
        # 图和 mask 一起 reflect：镜像出来的文字也被标成待修补，LaMa 不会把它当上下文抄进去
        img_p = cv2.copyMakeBorder(img_s, *pads, borderType=cv2.BORDER_REFLECT_101)
        m_p = cv2.copyMakeBorder(m_s.astype(np.uint8), *pads, borderType=cv2.BORDER_REFLECT_101)
        mk = m_p.astype(np.float32)
        x = img_p[:, :, ::-1].astype(np.float32) * np.float32(1.0 / 255.0) * (1.0 - mk[..., None])
        names = m.input_names
        feeds = {names[0] if names else "image": np.ascontiguousarray(x.transpose(2, 0, 1)[None]),
                 names[1] if len(names) > 1 else "mask": np.ascontiguousarray(mk[None, None])}
        t = time.perf_counter()
        res = m.run(feeds)[0]
        dt = (time.perf_counter() - t) * 1000
        self._note_shape((ch, cw), dt, stats)
        o = np.asarray(res[0], np.float32).transpose(1, 2, 0)
        o = np.clip(o * 255.0 + 0.5, 0, 255).astype(np.uint8)[:, :, ::-1]
        o = np.ascontiguousarray(o[top:top + hs, left:left + ws])
        if (hs, ws) != (h, w):
            o = cv2.resize(o, (w, h), interpolation=cv2.INTER_CUBIC)
        return o

    def _note_shape(self, shape: tuple[int, int], ms: float, stats: Optional[dict]) -> None:
        """记录 LaMa 输入尺寸：first = 本实例第一次见到的分桶尺寸（GPU 要预热），switch = 与上一次不同。"""
        with self._lock:
            first = shape not in self._seen_shapes
            switched = self._last_shape is not None and self._last_shape != shape
            self._seen_shapes.add(shape)
            if first or switched:
                if len(self.shape_log) < 64:
                    self.shape_log.append({"shape": shape, "ms": round(ms, 1), "first": first, "switch": switched})
                log.debug("LaMa 输入尺寸 %s（%s）：%.0fms", shape, "首次" if first else "切换", ms)
            self._last_shape = shape
        if stats is not None:
            stats["lama_calls"] += 1
            stats["shape_switches"] += int(switched)
            stats["new_shapes"] += int(first)

    @property
    def bucket_warmups(self) -> int:
        """到目前为止 LaMa 用过的不同分桶尺寸数（= GPU 预热次数）。"""
        with self._lock:
            return len(self._seen_shapes)

    @staticmethod
    def _classic_fill(work: np.ndarray, mask: np.ndarray, alpha: np.ndarray) -> None:
        """LaMa 不可用时的传统修补（OpenCV Telea），按连通域裁剪处理，只写 mask 内。"""
        h, w = mask.shape
        n, labels, st, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
        for i in range(1, n):
            x, y, ww, hh, _a = (int(v) for v in st[i])
            pad = max(8, (ww + hh) // 4)
            box = clamp_box((x - pad, y - pad, x + ww + pad, y + hh + pad), w, h)
            x1, y1, x2, y2 = box
            sub = (labels[y1:y2, x1:x2] == i)
            if not sub.any():
                continue
            radius = max(3, min(15, (min(ww, hh) // 4) or 3))
            out = cv2.inpaint(work[y1:y2, x1:x2], sub.astype(np.uint8) * 255, radius, cv2.INPAINT_TELEA)
            _blend(work, box, sub, out, alpha)
