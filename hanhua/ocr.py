"""文字识别（OCR）：日文 manga-ocr + 韩 / 英 / 中 PP-OCRv5。

流程（OCREngine.recognize）：
  每个文字块按框比例加边距裁剪（气泡块不越出气泡框）→ 按语言分流：
    - ja：manga-ocr（ViT 编码 + BERT 解码，竖排 / 横排都能整块识别）。普通气泡整块识别（原版用法）；
      旁白之类的超大块按列 / 行（必要时在字间空隙）切成几片分别识别再拼接，避免 224×224 输入里字被压得太小。
    - ko / en / zh：PP-OCRv5（DB 在块内找出文字行 → 每行按高 48 等比缩放、同批右侧补零 → CTC greedy）。
    - auto：竖排 → manga-ocr；横排 → PP-OCR multi，结果含假名改走 manga-ocr，识别出谚文改用 korean 模型。
  → 文字颜色 / 描边颜色 → 拟声词判定 → 方向修正 → 非源语言文字的气泡外块（衣服 logo、招牌字母等）标 skip。

设计要点：
  - 只通过 Runtime.model(key).run(feeds) 推理；不依赖 torch / transformers。
  - 所有尺寸阈值按文字框 / 字号的比例计算。
  - 构造一次、多线程反复调用：内部只有懒加载的只读数据（加锁）。
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import cv2
import numpy as np
from PIL import Image

from .detect import _profile_runs, ensure_bgr, text_fg_mask
from .runtime import Profile, Runtime, models_dir
from .types import Box, HanhuaError, TextBlock, clamp_box

try:  # 白名单依赖；万一缺失用内置等价转换兜底，不让流程崩
    import jaconv as _jaconv
except Exception:  # noqa: BLE001
    _jaconv = None

log = logging.getLogger("hanhua.ocr")

LANGS = ("ja", "ko", "en", "zh", "auto")


# --------------------------------------------------------------------------------------
# 字符类别
# --------------------------------------------------------------------------------------
def _is_kana(c: str) -> bool:
    """平假名 / 片假名本体（不含 ー ・ 这类符号）。"""
    o = ord(c)
    return 0x3041 <= o <= 0x3096 or 0x30A1 <= o <= 0x30FA or 0x31F0 <= o <= 0x31FF or 0xFF66 <= o <= 0xFF9D


def _is_han(c: str) -> bool:
    o = ord(c)
    return (0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or 0xF900 <= o <= 0xFAFF
            or 0x20000 <= o <= 0x3134F or o == 0x3005)  # 々


def _is_hangul(c: str) -> bool:
    o = ord(c)
    return 0xAC00 <= o <= 0xD7A3 or 0x1100 <= o <= 0x11FF or 0x3131 <= o <= 0x318E


def _is_latin(c: str) -> bool:
    o = ord(c)
    return (0x41 <= o <= 0x5A or 0x61 <= o <= 0x7A or 0xFF21 <= o <= 0xFF3A or 0xFF41 <= o <= 0xFF5A
            or (0xC0 <= o <= 0x24F and c.isalpha()))


@dataclass
class ScriptCounts:
    kana: int = 0
    han: int = 0
    hangul: int = 0
    latin: int = 0
    digit: int = 0

    @property
    def letters(self) -> int:
        return self.kana + self.han + self.hangul + self.latin


def script_counts(text: str) -> ScriptCounts:
    sc = ScriptCounts()
    for c in text or "":
        if _is_kana(c):
            sc.kana += 1
        elif _is_han(c):
            sc.han += 1
        elif _is_hangul(c):
            sc.hangul += 1
        elif _is_latin(c):
            sc.latin += 1
        elif c.isdigit():
            sc.digit += 1
    return sc


def has_source_script(text: str, script: str) -> bool:
    """文本里是否含源语言的文字（ja：假名 / 汉字；zh：汉字；ko：谚文；en：拉丁字母）。"""
    sc = script_counts(text)
    if script == "ja":
        return sc.kana + sc.han > 0
    if script == "zh":
        return sc.han > 0
    if script == "ko":
        return sc.hangul > 0
    if script == "en":
        return sc.latin > 0
    return sc.letters > 0


def dominant_script(texts: Iterable[str]) -> str:
    """整页主语言（auto 模式用）：出现假名就是日文；否则按谚文 / 汉字 / 拉丁字母数量取多数。"""
    tot = ScriptCounts()
    for t in texts:
        c = script_counts(t)
        tot.kana += c.kana
        tot.han += c.han
        tot.hangul += c.hangul
        tot.latin += c.latin
    if tot.letters == 0:
        return ""
    if tot.kana >= 2 and tot.kana >= 0.08 * tot.letters:
        return "ja"
    best = max((("ko", tot.hangul), ("zh", tot.han), ("en", tot.latin)), key=lambda kv: kv[1])
    return best[0]


# --------------------------------------------------------------------------------------
# 日文后处理（与 manga_ocr 原版一致）
# --------------------------------------------------------------------------------------
_DOT_RUN = re.compile("[・.]{2,}")


def _h2z(text: str) -> str:
    """半角 → 全角（ASCII、数字、半角片假名）。优先 jaconv，缺失时用等价转换。"""
    if _jaconv is not None:
        try:
            return _jaconv.h2z(text, ascii=True, digit=True)
        except Exception:  # noqa: BLE001
            pass
    out = []
    for c in text:
        o = ord(c)
        if 0x21 <= o <= 0x7E:
            out.append(chr(o + 0xFEE0))
        elif o == 0x20:
            out.append("　")
        elif 0xFF61 <= o <= 0xFF9F:
            out.append(unicodedata.normalize("NFKC", c))
        else:
            out.append(c)
    return unicodedata.normalize("NFC", "".join(out))


def postprocess_ja(text: str) -> str:
    """去空白、`…`→`...`、连续 `・.` 归一成同样个数的 `.`、半角转全角。"""
    text = "".join((text or "").split())
    text = text.replace("…", "...")
    text = _DOT_RUN.sub(lambda m: "." * (m.end() - m.start()), text)
    return _h2z(text)


# --------------------------------------------------------------------------------------
# 小工具
# --------------------------------------------------------------------------------------
def _gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def _read_json(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception as e:  # noqa: BLE001
        log.warning("读取 %s 失败: %s", path, e)
        return {}


def _yaml_scalar(raw: str) -> str:
    """解析 YAML 列表项的标量（只需支持字典文件里出现的写法：单引号 / 双引号 / 普通）。"""
    s = raw.rstrip("\r\n")
    if len(s) >= 2 and s[0] == "'" and s[-1] == "'":
        return s[1:-1].replace("''", "'")
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        try:
            return json.loads(s)
        except Exception:  # noqa: BLE001
            return s[1:-1]
    # 普通标量：只去掉 YAML 空白（空格 / 制表符），全角空格 U+3000 是合法字符
    return s.strip(" \t")


def read_ppocr_dict(yml_path: str) -> list[str]:
    """从 PaddleOCR 导出的 inference.yml 里读 PostProcess.character_dict（不依赖 PyYAML）。"""
    with open(yml_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    chars: list[str] = []
    start = -1
    for i, ln in enumerate(lines):
        if ln.strip() == "character_dict:":
            start = i + 1
            break
    if start < 0:
        raise HanhuaError(f"OCR 字典缺失：{os.path.basename(yml_path)} 里没有 character_dict（请重新安装软件）")
    for ln in lines[start:]:
        body = ln.lstrip(" ")
        if body.startswith("- ") or body == "-":
            chars.append(_yaml_scalar(body[2:]) if body != "-" else "")
        elif body.strip() == "":
            continue
        else:
            break
    return chars


def _softmax_last(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    m = x.max(axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=-1, keepdims=True)


def _order_quad(pts: np.ndarray) -> np.ndarray:
    """四点排成 左上、右上、右下、左下。"""
    p = pts[np.argsort(pts[:, 0])]
    left, right = p[:2], p[2:]
    tl, bl = (left[0], left[1]) if left[0, 1] <= left[1, 1] else (left[1], left[0])
    tr, br = (right[0], right[1]) if right[0, 1] <= right[1, 1] else (right[1], right[0])
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _crop_quad(img: np.ndarray, quad: np.ndarray) -> Optional[np.ndarray]:
    """按四边形透视裁剪一行；竖长（高 ≥ 1.5 倍宽）的行逆时针转成横向（PaddleOCR 的做法）。"""
    q = quad.astype(np.float32)
    w = int(round(max(np.linalg.norm(q[0] - q[1]), np.linalg.norm(q[2] - q[3]))))
    h = int(round(max(np.linalg.norm(q[0] - q[3]), np.linalg.norm(q[1] - q[2]))))
    if w < 2 or h < 2:
        return None
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(q, dst)
    out = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    if h >= 1.5 * w:
        out = np.ascontiguousarray(np.rot90(out))
    return out


def text_mask(gray: np.ndarray) -> np.ndarray:
    """文字前景（bool）：Otsu 少数类，再去掉碰到裁剪边界的大连通域
    （椭圆气泡框角带进来的深色背景、实景背景的大色块），避免它们干扰字号 / 列分割 / 颜色估计。"""
    fg = text_fg_mask(gray)
    h, w = fg.shape[:2]
    if h < 4 or w < 4 or not fg.any():
        return fg
    n, lab, st, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), connectivity=8)
    drop = np.zeros(n, bool)
    for i in range(1, n):
        x, y, bw, bh, area = (int(v) for v in st[i])
        touch = x == 0 or y == 0 or x + bw >= w or y + bh >= h
        if touch and (bw >= 0.35 * w or bh >= 0.35 * h) and (bw >= 0.6 * w or bh >= 0.6 * h or area >= 0.04 * h * w):
            drop[i] = True
    if not drop.any():
        return fg
    cleaned = fg & ~drop[lab]
    return cleaned if cleaned.any() else fg


def char_size(gray: np.ndarray, direction: str) -> float:
    """估计字号（px）：竖排 = 列宽中位数，横排 = 行高中位数。"""
    if gray is None or gray.size == 0:
        return 0.0
    h, w = gray.shape[:2]
    s = min(1.0, 320.0 / max(h, w))
    g = cv2.resize(gray, (max(4, int(w * s)), max(4, int(h * s))), interpolation=cv2.INTER_AREA) if s < 1.0 else gray
    fg = text_mask(g)
    runs, med, _ = _profile_runs(fg, 0 if direction == "v" else 1)
    if not runs:
        return float(min(h, w))
    return float(med) / s


# --------------------------------------------------------------------------------------
# 颜色估计
# --------------------------------------------------------------------------------------
def _rgb(bgr: Sequence[float]) -> tuple[int, int, int]:
    b, g, r = (int(round(float(v))) for v in bgr[:3])
    return (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))


def estimate_colors(crop_bgr: np.ndarray, want_stroke: bool) -> tuple[Optional[tuple[int, int, int]],
                                                                       Optional[tuple[int, int, int]]]:
    """返回 (文字主色, 外描边色)，RGB。文字 = 框内 Otsu 后的少数类；描边 = 文字外一圈稳定且反差大的颜色。"""
    if crop_bgr is None or crop_bgr.size == 0 or min(crop_bgr.shape[:2]) < 3:
        return None, None
    img = crop_bgr
    h, w = img.shape[:2]
    s = min(1.0, 400.0 / max(h, w))
    if s < 1.0:
        img = cv2.resize(img, (max(3, int(w * s)), max(3, int(h * s))), interpolation=cv2.INTER_AREA)
    gray = _gray(img)
    fg = text_fg_mask(gray)
    n = int(fg.sum())
    if n < 4:
        return None, None
    fg_u8 = fg.astype(np.uint8)
    core = cv2.erode(fg_u8, np.ones((3, 3), np.uint8)) > 0
    sel = core if core.sum() >= max(4, 0.25 * n) else fg     # 去掉抗锯齿边缘，太细的笔画就不腐蚀
    pix = img[sel]
    text_bgr = np.median(pix, axis=0)
    text_rgb = _rgb(text_bgr)
    if not want_stroke:
        return text_rgb, None
    # 描边宽度约为字号的 8%~15%：按笔画粗细估一个圈宽
    dist = cv2.distanceTransform(fg_u8, cv2.DIST_L2, 3)
    stroke_w = float(np.median(dist[fg])) * 2.0 if n else 1.0
    r = int(max(1, min(12, round(0.6 * stroke_w + 1))))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    ring = (cv2.dilate(fg_u8, k) > 0) & ~fg
    if int(ring.sum()) < max(8, 0.3 * n):
        return text_rgb, None
    rp = img[ring].astype(np.float32)
    ring_bgr = np.median(rp, axis=0)
    spread = float(np.median(np.abs(rp - ring_bgr).sum(axis=1)))
    contrast = float(np.abs(ring_bgr - text_bgr.astype(np.float32)).sum())
    if spread <= 75.0 and contrast >= 180.0:
        return text_rgb, _rgb(ring_bgr)
    return text_rgb, None


# --------------------------------------------------------------------------------------
# 日文大块切片
# --------------------------------------------------------------------------------------
def _lines_of(fg: np.ndarray, axis: int) -> tuple[list[tuple[int, int, int, int]], float]:
    """前景按列（axis=0）/ 行（axis=1）切成“线”，返回 ([(x1, y1, x2, y2)], 字号估计 s)。"""
    runs, s, _ = _profile_runs(fg, axis)
    if not runs:
        return [], s
    # 垂直于阅读方向的投影在一列 / 一行内是连续的，只合并极小的缝（紧排旁白的列距可能只有 0.2 个字宽）
    merged: list[tuple[int, int]] = [runs[0]]
    for a, b in runs[1:]:
        if a - merged[-1][1] < max(1.0, 0.08 * s):
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    out: list[tuple[int, int, int, int]] = []
    for a, b in merged:
        band = fg[:, a:b] if axis == 0 else fg[a:b, :]
        idx = np.flatnonzero(band.any(axis=1 if axis == 0 else 0))
        if idx.size == 0:
            continue
        if axis == 0:
            out.append((a, int(idx[0]), b, int(idx[-1]) + 1))
        else:
            out.append((int(idx[0]), a, int(idx[-1]) + 1, b))
    return out, s


def _split_long(fg: np.ndarray, line: tuple[int, int, int, int], axis: int, s: float,
                piece_chars: float) -> list[tuple[int, int, int, int]]:
    """把过长的一列 / 一行在字间空隙（投影最小处）切成几段。"""
    x1, y1, x2, y2 = line
    length = (y2 - y1) if axis == 0 else (x2 - x1)
    n = int(math.ceil(length / max(1.0, s) / piece_chars))
    if n <= 1:
        return [line]
    band = fg[y1:y2, x1:x2]
    prof = band.sum(axis=1 if axis == 0 else 0).astype(np.float32)
    cuts: list[int] = []
    win = max(1, int(round(0.7 * s)))
    for k in range(1, n):
        t = int(round(k * length / n))
        lo, hi = max(1, t - win), min(length - 1, t + win)
        if hi <= lo:
            continue
        seg = prof[lo:hi]
        best = lo + int(np.argmin(seg + 1e-3 * np.abs(np.arange(lo, hi) - t)))  # 并列时取离目标最近的
        if not cuts or best - cuts[-1] > 0.5 * s:
            cuts.append(best)
    bounds = [0] + cuts + [length]
    out: list[tuple[int, int, int, int]] = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a < 2:
            continue
        out.append((x1, y1 + a, x2, y1 + b) if axis == 0 else (x1 + a, y1, x1 + b, y2))
    return out or [line]


# --------------------------------------------------------------------------------------
# manga-ocr
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class _MocrSpec:
    vocab: tuple[str, ...]
    start_id: int
    eos_id: int
    pad_id: int
    skip_ids: frozenset[int]
    size: tuple[int, int]          # (宽, 高)
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    rescale: float
    resample: int
    no_repeat_ngram: int


def load_mocr_spec(root: Optional[str] = None) -> _MocrSpec:
    """读 models/manga-ocr/ 下的 config / generation_config / preprocessor_config / vocab。"""
    root = root or os.path.join(models_dir(), "manga-ocr")
    vocab_path = os.path.join(root, "vocab.txt")
    try:
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = tuple(ln.rstrip("\r\n") for ln in f)
    except OSError as e:
        raise HanhuaError(f"日文识别模型的词表缺失（{vocab_path}），请重新安装软件") from e
    cfg = _read_json(os.path.join(root, "config.json"))
    gen = _read_json(os.path.join(root, "generation_config.json"))
    pre = _read_json(os.path.join(root, "preprocessor_config.json"))
    dec = cfg.get("decoder") if isinstance(cfg.get("decoder"), dict) else {}

    def tok(name: str, default: int) -> int:
        return vocab.index(name) if name in vocab else default

    def pick(key: str, default: int) -> int:
        for src in (gen, cfg, dec):
            v = src.get(key)
            if isinstance(v, int) and not isinstance(v, bool):
                return v
        return default

    start = pick("decoder_start_token_id", tok("[CLS]", 2))
    eos = pick("eos_token_id", tok("[SEP]", 3))
    pad = pick("pad_token_id", tok("[PAD]", 0))
    special = {vocab.index(t) for t in ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]") if t in vocab}
    special |= {start, eos, pad}
    special |= {i for i, t in enumerate(vocab) if t.startswith("<unused")}
    sz = pre.get("size") if isinstance(pre.get("size"), dict) else {}
    size = (int(sz.get("width", 224)), int(sz.get("height", 224)))
    do_norm = bool(pre.get("do_normalize", True))
    mean = tuple(float(v) for v in (pre.get("image_mean") or [0.5] * 3))[:3] if do_norm else (0.0, 0.0, 0.0)
    std = tuple(float(v) for v in (pre.get("image_std") or [0.5] * 3))[:3] if do_norm else (1.0, 1.0, 1.0)
    rescale = float(pre.get("rescale_factor", 1 / 255)) if pre.get("do_rescale", True) else 1.0
    resample = int(pre.get("resample", 2))
    nr = gen.get("no_repeat_ngram_size", cfg.get("no_repeat_ngram_size", 0))
    return _MocrSpec(vocab=vocab, start_id=int(start), eos_id=int(eos), pad_id=int(pad),
                     skip_ids=frozenset(special), size=size, mean=mean, std=std,  # type: ignore[arg-type]
                     rescale=rescale, resample=resample, no_repeat_ngram=int(nr or 0))


class MangaOcr:
    """manga-ocr 的 ONNX 实现：批量编码 + 批量 greedy（每条独立停止，已结束的行移出批次）。"""

    MAX_TOKENS = 160

    def __init__(self, rt: Runtime, enc_key: str, dec_key: str, batch: int) -> None:
        self.rt = rt
        self.enc_key = enc_key
        self.dec_key = dec_key
        self.batch = max(1, int(batch))
        self._spec: Optional[_MocrSpec] = None
        self._lock = threading.Lock()

    @property
    def spec(self) -> _MocrSpec:
        if self._spec is None:
            with self._lock:
                if self._spec is None:
                    self._spec = load_mocr_spec()
        return self._spec

    # -- 预处理：原版先转灰度 L 再转回 RGB → resize 224（PIL bilinear）→ rescale → normalize ------
    def preprocess(self, crop_bgr: np.ndarray) -> np.ndarray:
        sp = self.spec
        gray = _gray(crop_bgr)
        try:
            resample = Image.Resampling(sp.resample)
        except ValueError:
            resample = Image.Resampling.BILINEAR
        im = Image.fromarray(np.ascontiguousarray(gray)).resize(sp.size, resample)
        g = np.asarray(im, dtype=np.float32) * sp.rescale
        return np.stack([(g - sp.mean[c]) / sp.std[c] for c in range(3)]).astype(np.float32)

    def decode_ids(self, ids: Sequence[int]) -> str:
        sp = self.spec
        n = len(sp.vocab)
        return "".join(sp.vocab[i] for i in ids if 0 <= i < n and i not in sp.skip_ids)

    def __call__(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        """识别若干裁剪图，返回 [(后处理后的文本, 置信度)]，顺序与输入一致。"""
        out: list[tuple[str, float]] = [("", 0.0)] * len(crops)
        if not crops:
            return out
        enc = self.rt.model(self.enc_key)
        dec = self.rt.model(self.dec_key)
        for i in range(0, len(crops), self.batch):
            idx = list(range(i, min(len(crops), i + self.batch)))
            x = np.stack([self.preprocess(crops[k]) for k in idx])
            try:
                hidden = enc.run({"pixel_values": x})[0]
                res = self._greedy(dec, hidden)
            except Exception as e:  # noqa: BLE001
                raise HanhuaError(f"日文文字识别失败：{e}") from e
            for k, (ids, conf) in zip(idx, res):
                out[k] = (postprocess_ja(self.decode_ids(ids)), conf)
        return out

    def _greedy(self, dec, hidden: np.ndarray) -> list[tuple[list[int], float]]:
        sp = self.spec
        B = hidden.shape[0]
        n = sp.no_repeat_ngram
        ids = np.full((B, 1), sp.start_id, dtype=np.int64)
        active = np.arange(B)
        toks: list[list[int]] = [[] for _ in range(B)]
        probs: list[list[float]] = [[] for _ in range(B)]
        seen: list[dict[tuple[int, ...], set[int]]] = [{} for _ in range(B)]
        hidden = np.ascontiguousarray(hidden.astype(np.float32, copy=False))
        for _step in range(self.MAX_TOKENS + 1):
            feeds = {"input_ids": ids, "encoder_hidden_states": hidden[active] if len(active) < B else hidden}
            logits = dec.run(feeds)[0][:, -1, :]
            p = _softmax_last(logits)
            if n > 0 and ids.shape[1] >= n - 1:
                for r, b in enumerate(active):   # no_repeat_ngram_size（与原版生成配置一致，防止 greedy 复读）
                    ban = seen[b].get(tuple(int(t) for t in ids[r, ids.shape[1] - (n - 1):]))
                    if ban:
                        p[r, list(ban)] = 0.0
            nxt = p.argmax(axis=1)
            conf = p[np.arange(len(active)), nxt]
            if _step == self.MAX_TOKENS:              # 到上限：强制结束
                nxt = np.full_like(nxt, sp.eos_id)
            keep = nxt != sp.eos_id
            for r, b in enumerate(active):
                probs[b].append(float(conf[r]))
                if keep[r]:
                    toks[b].append(int(nxt[r]))
                    if n > 0 and ids.shape[1] >= n - 1:
                        key = tuple(int(t) for t in ids[r, ids.shape[1] - (n - 1):])
                        seen[b].setdefault(key, set()).add(int(nxt[r]))
            if not keep.any():
                break
            ids = np.concatenate([ids[keep], nxt[keep, None].astype(np.int64)], axis=1)
            active = active[keep]
        return [(toks[b], float(np.mean(probs[b])) if probs[b] else 0.0) for b in range(B)]


# --------------------------------------------------------------------------------------
# PP-OCRv5
# --------------------------------------------------------------------------------------
class PPOcr:
    """PP-OCRv5：DB 文字行检测 + CTC 识别（multi / korean / en）。"""

    DET_MAX_SIDE = 960
    DET_MIN_SIDE = 320
    REC_H = 48
    REC_MIN_W = 320
    REC_MAX_W = 3200
    REC_KEYS = {"multi": "ppocr_rec_multi", "korean": "ppocr_rec_korean", "en": "ppocr_rec_en"}

    def __init__(self, rt: Runtime, batch: int) -> None:
        self.rt = rt
        self.batch = max(1, int(batch))
        self._lock = threading.Lock()
        self._dicts: dict[str, list[str]] = {}
        self._det_cfg: Optional[dict] = None

    # -- 配置 ----------------------------------------------------------------------------
    def det_cfg(self) -> dict:
        if self._det_cfg is None:
            with self._lock:
                if self._det_cfg is None:
                    self._det_cfg = self._read_det_cfg(os.path.join(models_dir(), "ppocr", "det.yml"))
        return self._det_cfg

    @staticmethod
    def _read_det_cfg(path: str) -> dict:
        cfg = {"thresh": 0.3, "box_thresh": 0.6, "unclip_ratio": 1.5, "max_candidates": 1000,
               "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "scale": 1 / 255}
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        except OSError:
            log.warning("缺少 %s，PP-OCR 检测使用默认参数", path)
            return cfg
        post = text.split("PostProcess:", 1)[-1]
        for key in ("thresh", "box_thresh", "unclip_ratio", "max_candidates"):
            m = re.search(rf"^\s*{key}:\s*([0-9.]+)\s*$", post, re.M)
            if m:
                cfg[key] = float(m.group(1))
        norm = text.split("NormalizeImage:", 1)
        if len(norm) == 2:
            for key in ("mean", "std"):
                m = re.search(rf"{key}:\s*\n((?:\s*-\s*[0-9.]+\s*\n){{3}})", norm[1])
                if m:
                    cfg[key] = [float(v) for v in re.findall(r"[0-9.]+", m.group(1))][:3]
        return cfg

    def chars(self, which: str) -> list[str]:
        d = self._dicts.get(which)
        if d is None:
            with self._lock:
                d = self._dicts.get(which)
                if d is None:
                    name = self.REC_KEYS[which].replace("ppocr_", "")
                    d = read_ppocr_dict(os.path.join(models_dir(), "ppocr", f"{name}.yml"))
                    self._dicts[which] = d
        return d

    # -- DB 行检测 ------------------------------------------------------------------------
    def detect_lines(self, crop: np.ndarray) -> list[np.ndarray]:
        """返回块内文字行的四边形（crop 坐标，左上 / 右上 / 右下 / 左下）。"""
        h, w = crop.shape[:2]
        if h < 4 or w < 4:
            return []
        cfg = self.det_cfg()
        long_side = max(h, w)
        scale = 1.0
        if long_side > self.DET_MAX_SIDE:
            scale = self.DET_MAX_SIDE / long_side
        elif long_side < self.DET_MIN_SIDE:
            scale = min(3.0, self.DET_MIN_SIDE / long_side)
        nh = max(32, int(math.ceil(h * scale / 32.0)) * 32)
        nw = max(32, int(math.ceil(w * scale / 32.0)) * 32)
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        img = cv2.resize(crop, (nw, nh), interpolation=interp).astype(np.float32) * float(cfg["scale"])
        img = (img - np.asarray(cfg["mean"], np.float32)) / np.asarray(cfg["std"], np.float32)  # 与 PaddleOCR 一样直接作用在 BGR 上
        x = np.ascontiguousarray(img.transpose(2, 0, 1)[None])
        prob = np.asarray(self.rt.model("ppocr_det").run({"x": x})[0])
        prob = prob.reshape(prob.shape[-2], prob.shape[-1])
        return self._db_boxes(prob, w, h, cfg)

    @staticmethod
    def _db_boxes(prob: np.ndarray, dst_w: int, dst_h: int, cfg: dict) -> list[np.ndarray]:
        ph, pw = prob.shape
        bitmap = (prob > float(cfg["thresh"])).astype(np.uint8)
        contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        out: list[np.ndarray] = []
        for c in contours[: int(cfg["max_candidates"])]:
            if len(c) < 3:
                continue
            rect = cv2.minAreaRect(c)
            if min(rect[1]) < 3:
                continue
            pts = cv2.boxPoints(rect)
            x0 = int(np.clip(np.floor(pts[:, 0].min()), 0, pw - 1))
            x1 = int(np.clip(np.ceil(pts[:, 0].max()), 0, pw - 1))
            y0 = int(np.clip(np.floor(pts[:, 1].min()), 0, ph - 1))
            y1 = int(np.clip(np.ceil(pts[:, 1].max()), 0, ph - 1))
            m = np.zeros((y1 - y0 + 1, x1 - x0 + 1), np.uint8)
            cv2.fillPoly(m, [np.round(pts - [x0, y0]).astype(np.int32)], 1)
            score = float(cv2.mean(prob[y0:y1 + 1, x0:x1 + 1], m)[0])
            if score < float(cfg["box_thresh"]):
                continue
            (cx, cy), (rw, rh), ang = rect
            d = rw * rh * float(cfg["unclip_ratio"]) / max(1e-6, 2.0 * (rw + rh))   # 矩形的 unclip 等价于四边外扩 d
            big = ((cx, cy), (rw + 2 * d, rh + 2 * d), ang)
            if min(big[1]) < 5:
                continue
            q = cv2.boxPoints(big).astype(np.float32)
            q[:, 0] = np.clip(q[:, 0] * dst_w / pw, 0, dst_w)
            q[:, 1] = np.clip(q[:, 1] * dst_h / ph, 0, dst_h)
            out.append(_order_quad(q))
        return out

    # -- CTC 识别 -------------------------------------------------------------------------
    def recognize_lines(self, lines: list[np.ndarray], which: str) -> list[tuple[str, float]]:
        out: list[tuple[str, float]] = [("", 0.0)] * len(lines)
        valid = [i for i, im in enumerate(lines) if im is not None and im.size and min(im.shape[:2]) >= 2]
        if not valid:
            return out
        chars = self.chars(which)
        model = self.rt.model(self.REC_KEYS[which])
        valid.sort(key=lambda i: lines[i].shape[1] / max(1, lines[i].shape[0]))
        for s in range(0, len(valid), self.batch):
            idx = valid[s: s + self.batch]
            ratios = [lines[i].shape[1] / max(1, lines[i].shape[0]) for i in idx]
            W = int(min(self.REC_MAX_W, max(self.REC_MIN_W, math.ceil(self.REC_H * max(ratios)))))
            W = int(math.ceil(W / 32.0) * 32)
            x = np.zeros((len(idx), 3, self.REC_H, W), np.float32)
            for j, i in enumerate(idx):
                im = lines[i]
                if im.ndim == 2:
                    im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
                rw = int(min(W, max(1, math.ceil(self.REC_H * ratios[j]))))
                rs = cv2.resize(im, (rw, self.REC_H), interpolation=cv2.INTER_AREA if im.shape[0] > self.REC_H
                                else cv2.INTER_LINEAR).astype(np.float32)
                x[j, :, :, :rw] = ((rs / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)
            try:
                pred = np.asarray(model.run({"x": x})[0])
            except Exception as e:  # noqa: BLE001
                raise HanhuaError(f"文字识别失败：{e}") from e
            if pred.min() < -1e-3 or pred.max() > 1.0 + 1e-3:
                pred = _softmax_last(pred)
            for j, i in enumerate(idx):
                out[i] = self._ctc(pred[j], chars)
        return out

    @staticmethod
    def _ctc(p: np.ndarray, chars: list[str]) -> tuple[str, float]:
        """CTC greedy：取最大类、合并连续重复、去 blank（0）。索引 1..n = 字典，n+1 = 空格。"""
        idx = p.argmax(axis=1)
        mx = p.max(axis=1)
        keep = np.ones(len(idx), bool)
        keep[1:] = idx[1:] != idx[:-1]
        keep &= idx != 0
        n = len(chars)
        text = []
        confs = []
        for k in np.flatnonzero(keep):
            c = int(idx[k])
            if 1 <= c <= n:
                text.append(chars[c - 1])
            elif c == n + 1:
                text.append(" ")
            else:
                continue
            confs.append(float(mx[k]))
        return "".join(text), (float(np.mean(confs)) if confs else 0.0)


# --------------------------------------------------------------------------------------
# OCREngine
# --------------------------------------------------------------------------------------
@dataclass
class _Item:
    """一个文字块的识别中间结果。"""

    block: TextBlock
    crop_box: Box
    crop: np.ndarray
    route: str = "mocr"                     # "mocr" | "pp"
    text: str = ""
    conf: float = 0.0
    line_dir: Optional[str] = None          # 识别时观察到的行方向（PP-OCR 的行形状 / 单行字数）
    n_lines: int = 0
    lines: list[np.ndarray] = field(default_factory=list)


class OCREngine:
    """文字识别引擎：构造一次，多线程反复调用 recognize()。"""

    # 日文大块切片阈值（按估计字数，与分辨率无关）
    WHOLE_MAX_CHARS = 48      # 总字数不超过这个、列数不超过 WHOLE_MAX_LINES、单列不超过 LINE_MAX_CHARS → 整块识别
    WHOLE_MAX_LINES = 5
    LINE_MAX_CHARS = 26
    PIECE_CHARS = 16          # 超长列切成每段约这么多字
    PAD_RATIO = 0.06          # 裁剪边距（相对文字框短边）
    SFX_BIG = 1.7             # 字形是本页对白字号的这么多倍以上 → 像拟声词

    def __init__(self, rt: Runtime, profile: Profile, lang: str = "ja") -> None:
        self.rt = rt
        self.profile = profile
        if lang not in LANGS:
            log.warning("未知源语言 %s，按日文处理", lang)
            lang = "ja"
        self.lang = lang
        self.mocr = MangaOcr(rt, profile.ocr_encoder, profile.ocr_decoder, profile.ocr_batch)
        self.pp = PPOcr(rt, profile.ocr_batch)

    # -- 主流程 ----------------------------------------------------------------------------
    def recognize(self, img: np.ndarray, blocks: list[TextBlock]) -> None:
        """原地填写 src_text、ocr_conf、text_color、stroke_color、is_sfx（必要时修正 direction）。"""
        if not blocks:
            return
        page = ensure_bgr(img)
        if page is None:
            log.warning("OCR 输入为空或格式不支持，跳过")
            for b in blocks:
                b.src_text, b.ocr_conf = "", 0.0
            return
        H, W = page.shape[:2]
        items: list[_Item] = []
        for b in blocks:
            box = self._crop_box(b, W, H)
            x1, y1, x2, y2 = box
            items.append(_Item(block=b, crop_box=box, crop=page[y1:y2, x1:x2]))

        lang = self.lang
        for it in items:
            if lang == "ja":
                it.route = "mocr"
            elif lang == "auto":
                it.route = "mocr" if it.block.direction == "v" else "pp"
            else:
                it.route = "pp"

        pp_items = [it for it in items if it.route == "pp" and it.crop.size]
        if pp_items:
            self._run_pp(pp_items, lang)
        if lang == "auto":   # 横排但识别出假名 → 改走 manga-ocr
            for it in pp_items:
                if script_counts(it.text).kana > 0:
                    it.route = "mocr"
        mocr_items = [it for it in items if it.route == "mocr" and it.crop.size]
        if mocr_items:
            self._run_mocr(mocr_items)

        self._finish(items, lang)

    # -- 裁剪 ------------------------------------------------------------------------------
    def _crop_box(self, b: TextBlock, W: int, H: int) -> Box:
        try:
            x1, y1, x2, y2 = clamp_box(tuple(int(v) for v in b.box), W, H)  # type: ignore[arg-type]
        except Exception:  # noqa: BLE001
            return (0, 0, 0, 0)
        pad = max(2, int(round(self.PAD_RATIO * min(x2 - x1, y2 - y1))))
        bx = (x1 - pad, y1 - pad, x2 + pad, y2 + pad)
        if b.bubble_box is not None:   # 气泡块：边距不越出气泡框（避免带进气泡描边），但始终包含文字框
            try:
                q = tuple(int(v) for v in b.bubble_box)
                bx = (max(bx[0], min(x1, q[0])), max(bx[1], min(y1, q[1])),
                      min(bx[2], max(x2, q[2])), min(bx[3], max(y2, q[3])))
            except Exception:  # noqa: BLE001
                pass
        return clamp_box(bx, W, H)

    # -- manga-ocr ------------------------------------------------------------------------
    def _pieces(self, crop: np.ndarray, direction: str) -> list[Box]:
        """普通气泡整块；超大块（旁白等）按列 / 行切片，过长的列在字间切开。返回 crop 内坐标（阅读顺序）。"""
        h, w = crop.shape[:2]
        whole = [(0, 0, w, h)]
        if h < 8 or w < 8:
            return whole
        s0 = min(1.0, 640.0 / max(h, w))
        gray = _gray(crop)
        if s0 < 1.0:
            gray = cv2.resize(gray, (max(8, int(w * s0)), max(8, int(h * s0))), interpolation=cv2.INTER_AREA)
        fg = text_fg_mask(gray)
        axis = 0 if direction == "v" else 1
        lines, s = _lines_of(fg, axis)
        if not lines:
            return whole

        def thick(l: tuple[int, int, int, int]) -> int:
            return (l[2] - l[0]) if axis == 0 else (l[3] - l[1])

        def length(l: tuple[int, int, int, int]) -> int:
            return (l[3] - l[1]) if axis == 0 else (l[2] - l[0])

        main = [l for l in lines if thick(l) >= 0.55 * s] if len(lines) >= 2 else lines   # 细列多半是注音假名
        main = main or lines
        chars = [length(l) / max(1.0, s) for l in main]
        if len(main) <= self.WHOLE_MAX_LINES and sum(chars) <= self.WHOLE_MAX_CHARS and max(chars) <= self.LINE_MAX_CHARS:
            return whole
        main.sort(key=(lambda l: -(l[0] + l[2])) if axis == 0 else (lambda l: l[1] + l[3]))
        pieces: list[tuple[int, int, int, int]] = []
        for l, c in zip(main, [length(l) / max(1.0, s) for l in main]):
            pieces.extend(_split_long(fg, l, axis, s, self.PIECE_CHARS) if c > self.LINE_MAX_CHARS else [l])
        pad = 0.2 * s
        out: list[Box] = []
        for x1, y1, x2, y2 in pieces:
            out.append(clamp_box((int(math.floor((x1 - pad) / s0)), int(math.floor((y1 - pad) / s0)),
                                  int(math.ceil((x2 + pad) / s0)), int(math.ceil((y2 + pad) / s0))), w, h))
        out = [b for b in out if b[2] - b[0] >= 2 and b[3] - b[1] >= 2]
        return out or whole

    def _run_mocr(self, items: list[_Item]) -> None:
        jobs: list[tuple[int, np.ndarray]] = []   # (item 序号, 片段图)
        n_pieces: list[int] = []
        for k, it in enumerate(items):
            try:
                pieces = self._pieces(it.crop, it.block.direction)
            except Exception as e:  # noqa: BLE001 - 切片失败就整块识别
                log.debug("切片失败，整块识别: %s", e)
                pieces = [(0, 0, it.crop.shape[1], it.crop.shape[0])]
            n_pieces.append(len(pieces))
            for x1, y1, x2, y2 in pieces:
                jobs.append((k, it.crop[y1:y2, x1:x2]))
        # 按片段面积排序再分批：同一批的长度接近，greedy 解码浪费的步数少
        order = sorted(range(len(jobs)), key=lambda j: jobs[j][1].shape[0] * jobs[j][1].shape[1])
        res = self.mocr([jobs[j][1] for j in order])
        texts: list[list[tuple[int, str, float]]] = [[] for _ in items]
        for pos, j in enumerate(order):
            k = jobs[j][0]
            texts[k].append((j, res[pos][0], res[pos][1]))
        for k, it in enumerate(items):
            parts = sorted(texts[k])
            it.text = "".join(t for _, t, _ in parts)
            weights = [max(1, len(t)) for _, t, _ in parts]
            it.conf = float(np.average([c for _, _, c in parts], weights=weights)) if parts else 0.0
            it.n_lines = n_pieces[k]

    # -- PP-OCR ---------------------------------------------------------------------------
    def _run_pp(self, items: list[_Item], lang: str) -> None:
        all_lines: list[np.ndarray] = []
        owners: list[int] = []
        for k, it in enumerate(items):
            try:
                quads = self.pp.detect_lines(it.crop)
            except HanhuaError:
                raise
            except Exception as e:  # noqa: BLE001
                raise HanhuaError(f"文字行检测失败：{e}") from e
            quads = self._sort_quads(quads, it.block.direction)
            vert = sum(1 for q in quads if self._quad_hw(q)[0] >= 1.5 * self._quad_hw(q)[1])
            if quads:
                it.line_dir = "v" if vert * 2 > len(quads) else "h"
            imgs = [im for im in (_crop_quad(it.crop, q) for q in quads) if im is not None]
            if not imgs:   # DB 没找到行：整块当一行（竖长的转成横向）
                c = it.crop
                imgs = [np.ascontiguousarray(np.rot90(c)) if c.shape[0] >= 1.5 * c.shape[1] else c]
            it.lines = imgs
            it.n_lines = len(imgs)
            all_lines.extend(imgs)
            owners.extend([k] * len(imgs))
        which = {"ko": "korean", "en": "en", "zh": "multi", "auto": "multi"}.get(lang, "multi")
        res = self.pp.recognize_lines(all_lines, which)
        sep = " " if lang in ("ko", "en") else ""
        self._assign_pp(items, owners, res, sep)
        if lang != "auto":
            return
        # auto：没有假名的横排块再用 korean 模型读一遍，读出谚文且置信度不差就用它
        cand = [k for k, it in enumerate(items) if script_counts(it.text).kana == 0]
        if not cand:
            return
        sub_lines = [all_lines[i] for i, o in enumerate(owners) if o in cand]
        sub_owner = [o for o in owners if o in cand]
        res_ko = self.pp.recognize_lines(sub_lines, "korean")
        tmp = [_Item(block=items[k].block, crop_box=items[k].crop_box, crop=items[k].crop) for k in range(len(items))]
        self._assign_pp(tmp, sub_owner, res_ko, " ")
        for k in cand:
            ko = tmp[k]
            sc = script_counts(ko.text)
            if sc.hangul >= 1 and sc.hangul >= 0.3 * max(1, sc.letters) and ko.conf >= items[k].conf - 0.1:
                items[k].text, items[k].conf = ko.text, ko.conf

    @staticmethod
    def _quad_hw(q: np.ndarray) -> tuple[float, float]:
        h = float(max(np.linalg.norm(q[0] - q[3]), np.linalg.norm(q[1] - q[2])))
        w = float(max(np.linalg.norm(q[0] - q[1]), np.linalg.norm(q[2] - q[3])))
        return h, w

    @classmethod
    def _sort_quads(cls, quads: list[np.ndarray], direction: str) -> list[np.ndarray]:
        """行排序：多数是竖列 → 从右到左；否则按纵向重叠聚成行，行从上到下、行内从左到右。"""
        if len(quads) < 2:
            return quads
        boxes = [(float(q[:, 0].min()), float(q[:, 1].min()), float(q[:, 0].max()), float(q[:, 1].max())) for q in quads]
        vert = sum(1 for q in quads if cls._quad_hw(q)[0] >= 1.5 * cls._quad_hw(q)[1])
        idx = list(range(len(quads)))
        if vert * 2 > len(quads):
            idx.sort(key=lambda i: (-(boxes[i][0] + boxes[i][2]), boxes[i][1]))
            return [quads[i] for i in idx]
        idx.sort(key=lambda i: boxes[i][1])
        rows: list[list[int]] = []
        bands: list[list[float]] = []
        for i in idx:
            y0, y1 = boxes[i][1], boxes[i][3]
            for row, band in zip(rows, bands):
                ov = min(band[1], y1) - max(band[0], y0)
                if ov >= 0.5 * min(y1 - y0, band[1] - band[0]):
                    row.append(i)
                    band[0], band[1] = min(band[0], y0), max(band[1], y1)
                    break
            else:
                rows.append([i])
                bands.append([y0, y1])
        out: list[np.ndarray] = []
        for k in sorted(range(len(rows)), key=lambda r: bands[r][0]):
            out.extend(quads[i] for i in sorted(rows[k], key=lambda i: boxes[i][0]))
        return out

    @staticmethod
    def _assign_pp(items: list[_Item], owners: list[int], res: list[tuple[str, float]], sep: str) -> None:
        per: dict[int, list[tuple[str, float]]] = {}
        for o, r in zip(owners, res):
            per.setdefault(o, []).append(r)
        for k, it in enumerate(items):
            rs = [(t.strip(), c) for t, c in per.get(k, []) if t.strip()]
            it.text = _join_lines([t for t, _ in rs], sep)
            it.conf = float(np.average([c for _, c in rs], weights=[len(t) for t, _ in rs])) if rs else 0.0

    # -- 收尾：颜色 / 拟声词 / 方向 / skip ---------------------------------------------------------
    def _finish(self, items: list[_Item], lang: str) -> None:
        # 本页对白字号（气泡块的中位数）作为“大字”判断的参照
        sizes: dict[int, float] = {}
        for k, it in enumerate(items):
            try:
                if it.crop.size:
                    sizes[k] = char_size(_gray(it.crop), it.block.direction)
            except Exception:  # noqa: BLE001
                pass
        ref = [sizes[k] for k, it in enumerate(items) if k in sizes and it.block.kind == "bubble"]
        page_px = float(np.median(ref)) if ref else 0.0
        script = lang if lang != "auto" else dominant_script(it.text for it in items)

        for k, it in enumerate(items):
            b = it.block
            b.src_text = it.text
            b.ocr_conf = round(float(max(0.0, min(1.0, it.conf))), 4)
            try:
                b.text_color, b.stroke_color = estimate_colors(it.crop, want_stroke=(b.kind == "free"))
            except Exception as e:  # noqa: BLE001
                log.debug("颜色估计失败: %s", e)
                b.text_color, b.stroke_color = None, None
            self._fix_direction(it)
            b.is_sfx = self._is_sfx(b, it.text, b.ocr_conf, sizes.get(k, 0.0), page_px)
            # 不是源语言文字的气泡外块（衣服 / 帽子 logo、招牌字母等）、或者什么都没读出来 → 保留原样
            outside = b.kind == "free" or b.bubble_box is None
            if not it.text.strip():
                b.skip = True
            elif outside and script and not has_source_script(it.text, script):
                b.skip = True
            log.debug("OCR #%d %s conf=%.2f sfx=%s skip=%s: %s", b.id, b.direction, b.ocr_conf, b.is_sfx, b.skip, b.src_text)

    @staticmethod
    def _fix_direction(it: _Item) -> None:
        b = it.block
        if it.route == "pp" and it.line_dir:
            b.direction = it.line_dir
            return
        # manga-ocr：单行块用 “长宽比 ≈ 字数” 判断方向（比投影更可靠）
        n = sum(1 for c in it.text if not unicodedata.category(c).startswith(("P", "Z", "S")))
        if it.n_lines != 1 or n < 3:
            return
        x1, y1, x2, y2 = b.box
        bw, bh = max(1, x2 - x1), max(1, y2 - y1)
        if bh / bw >= 0.5 * n and bh > 1.5 * bw:
            b.direction = "v"
        elif bw / bh >= 0.5 * n and bw > 1.5 * bh:
            b.direction = "h"

    def _is_sfx(self, b: TextBlock, text: str, conf: float, px: float, page_px: float) -> bool:
        """拟声词：气泡外 + （很短 / 大量重复假名、长音、促音 / 置信度低 / 字形明显大于本页对白）中至少两条。"""
        if b.kind != "free":
            return False
        core = [c for c in text if _is_kana(c) or _is_han(c) or _is_hangul(c) or _is_latin(c) or c in "ー〜～"]
        if not core:
            return bool(text.strip())          # 只有 ！？ 之类的符号
        points = 0
        if len(core) <= 3:
            points += 1
        rep = sum(1 for c in core if c in "ッっーァィゥェォャュョぁぃぅぇぉゃゅょ〜～")
        dup = sum(1 for a, c in zip(core, core[1:]) if a == c)
        if len(set(core)) <= max(1, len(core) // 2) or (rep + dup) >= 0.4 * len(core):
            points += 1
        if conf < 0.55:
            points += 1
        if page_px > 0 and px >= self.SFX_BIG * page_px:
            points += 1
        return points >= 2


def _join_lines(texts: list[str], sep: str) -> str:
    """把多行拼成一段：韩 / 英用空格（行末连字符直接接上），中文直接相连。"""
    out = ""
    for t in texts:
        t = t.strip()
        if not t:
            continue
        if not out:
            out = t
        elif sep and out.endswith("-") and len(out) > 1 and out[-2].isalpha() and t[:1].islower():
            out = out[:-1] + t
        else:
            out += sep + t
    return re.sub(r"\s+", " ", out).strip() if sep else out


__all__ = ["OCREngine", "MangaOcr", "PPOcr", "postprocess_ja", "estimate_colors", "char_size",
           "script_counts", "has_source_script", "dominant_script", "read_ppocr_dict", "load_mocr_spec"]
