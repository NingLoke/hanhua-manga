"""合成漫画页面生成器（测试专用，带逐像素真值）。

    img, truth = make_page(seed=1, background="screentone", lang="ja")
    img, truth = make_webtoon(seed=2, lang="ko")
    path = make_book(tmp_dir, n=3, seed=3, fmt="zip")

约定（与 ARCHITECTURE.md 一致）：图片 BGR uint8 (H, W, 3)；颜色 tuple 为 RGB；Box = (x1, y1, x2, y2) 开区间。
truth = {
  "bubbles": [{"box", "mask", "inner_mask", "shape", "text", "lines", "direction", "lang",
               "text_box", "text_mask", "font_size", "fill", "text_color", "stroke_width"}],
  "free":    [{"box", "text", "lines", "text_mask", "fill_mask", "outline", "is_sfx", "direction", "lang",
               "font_size", "text_color", "stroke_color", "angle"}],
  "meta":    {"seed", "width", "height", "background", "lang", "kind", "panels"},
}
- 所有 mask 都是整页 bool (H, W)：
  * bubble["mask"]：整个气泡（黑描边 + 内部 + 尾巴），box = mask 的外接框；
  * bubble["inner_mask"]：描边以内的底色区域；保证 text_mask ⊆ inner_mask ⊆ mask；
  * text_mask：被字形影响到的全部像素（alpha > 0，描边字含描边）；text_box / free["box"] = 其外接框；
  * free["fill_mask"]：描边字的字芯部分（不含描边）。
- lines：竖排是“列”（从右到左），横排是“行”（从上到下）；text 是完整原文（ja/zh 无空格拼接，en/ko 空格拼接）。
- 气泡、气泡外文字两两不重叠（留间隙）；同参数同 seed → 逐位相同的图和真值。
- 字体：fonts/NotoSansSC-VF.ttf（含假名；可变字体默认 Thin，必须显式设字重）；韩文 malgun、英文 Comic Sans
  取自 Windows 字体目录，仅测试使用、不随产品分发，缺失时退回 Noto。
"""
from __future__ import annotations

import colorsys
import functools
import io
import logging
import math
import os
import sys
import threading
import zipfile
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_ROOT = Path(__file__).resolve().parent.parent
try:
    from hanhua import imageio as hio
    from hanhua.types import Box
except ImportError:  # 不经 conftest、直接 import 本文件时补上项目根
    sys.path.insert(0, str(_ROOT))
    from hanhua import imageio as hio
    from hanhua.types import Box

log = logging.getLogger("hanhua.tests.synth")

BACKGROUNDS: tuple[str, ...] = ("white", "screentone", "gradient", "texture", "photo")
SHAPES: tuple[str, ...] = ("ellipse", "roundrect", "cloud", "spiky", "tall", "wide", "tail")
LANGS: tuple[str, ...] = ("ja", "en", "ko", "zh")
FREE_KINDS: tuple[str, ...] = ("white", "black", "color")  # 白描边黑字 / 黑描边白字 / 彩色描边倾斜拟声词
_SHAPE_POOL = ("ellipse", "ellipse", "ellipse", "roundrect", "cloud", "spiky", "tall", "wide", "tail", "tail")

# ---- 语料：自拟的中性日常台词（覆盖 、。ー っ ？！ … 小假名 等竖排要点）----
JA_LINES: tuple[str, ...] = (
    "おはよう、今日もいい天気だね。", "ちょっと待って！", "えっ？本当に？", "そんなこと、聞いてないよ…",
    "よーし、やってみよう！", "ありがとう。助かったよ。", "どこへ行くんだ？", "もう時間がないっ！",
    "大丈夫、心配しないで。", "なんだって！？", "すごーい！", "早く行こうぜ！", "ここはどこ…？",
    "ずっと待ってたよ。", "ごめんね、遅くなっちゃった。", "いったい何が起きたんだ？", "お腹すいたー。",
    "気をつけて！", "明日また会おうね。", "それは秘密です。", "やったー！勝ったぞ！", "うそでしょ…",
    "静かにして。", "先生、質問があります！", "この本、すごく面白いよ。", "ふーん、そうなんだ。",
    "行ってきます！", "ただいまー。", "まさか…あいつが？", "絶対に負けない！",
    "もう一度だけ、チャンスをください。", "雨が降ってきたね。", "ちょっと、ずるいよ！", "はい、どうぞ。",
    "知らなかったの？", "準備はいい？", "いくぞっ！", "あれ？おかしいな…", "楽しみだなぁ。",
    "誰かいるのか？", "ほっとした…", "コーヒーでも飲む？", "しっかりしろ！", "電車に乗り遅れちゃう！",
    "ラーメン食べに行こうよ。", "そろそろ帰ろっか。",
)
JA_SFX = ("ドドド", "ゴゴゴ", "バン！", "ドン！！", "ザワザワ", "キラッ", "ガタッ", "ズドーン", "ピカッ",
          "ドキドキ", "シーン", "パチパチ", "ゴロゴロ", "ビュッ", "ガシャーン")
EN_LINES = ("WAIT FOR ME!", "WHAT ARE YOU DOING HERE?", "I CAN'T BELIEVE IT...", "LET'S GO!",
            "THANK YOU SO MUCH.", "WE NEED TO HURRY!", "IS ANYONE THERE?", "DON'T WORRY, I'VE GOT THIS.",
            "SEE YOU TOMORROW!", "THAT WAS CLOSE...", "HOW DID YOU KNOW?", "I'M SO HUNGRY.", "LOOK OUT!",
            "IT'S A SECRET.", "WHERE ARE WE GOING?", "NO WAY!", "THIS BOOK IS REALLY FUN.",
            "IT STARTED TO RAIN.", "ARE YOU READY?", "GOOD MORNING, EVERYONE!")
EN_SFX = ("BOOM!", "CRASH!", "WHAM!", "BANG!!", "THUD", "ZOOM", "KRAK!", "WHOOSH")
KO_LINES = ("잠깐만 기다려!", "정말이야?", "고마워, 덕분에 살았어.", "어디 가는 거야?", "괜찮아, 걱정하지 마.",
            "빨리 가자!", "여기가 어디지…?", "내일 또 만나자!", "배고프다…", "조심해!", "그건 비밀이야.",
            "말도 안 돼!", "준비됐어?", "누구 있어요?", "이제 시간이 없어!", "다녀왔습니다.",
            "비가 오기 시작했네.", "이 책 정말 재미있어.")
KO_SFX = ("쾅!", "두근두근", "휙", "쿵!", "와르르", "번쩍")
ZH_LINES = ("等一下！", "你是认真的吗？", "谢谢你，帮大忙了。", "我们快走吧！", "这里是哪里……？", "明天见！",
            "别担心，交给我。", "怎么可能！", "肚子好饿啊。", "小心！", "准备好了吗？", "这是秘密哦。")
ZH_SFX = ("轰！", "砰！", "咚咚", "唰", "哗啦")
LINES = {"ja": JA_LINES, "en": EN_LINES, "ko": KO_LINES, "zh": ZH_LINES}
SFX = {"ja": JA_SFX, "en": EN_SFX, "ko": KO_SFX, "zh": ZH_SFX}

# ---- 字体 ----
_FONT_CJK = _ROOT / "fonts" / "NotoSansSC-VF.ttf"
_WIN_FONTS = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
_SYS_FONTS = {"ko": ("malgun.ttf", "malgunbd.ttf"), "en": ("comic.ttf", "comicbd.ttf")}  # 仅测试用
_LOCK = threading.RLock()  # FreeType face 不保证线程安全：渲染整体加锁


@functools.lru_cache(maxsize=None)
def _font_file(lang: str, bold: bool) -> str:
    names = _SYS_FONTS.get(lang)
    if names:
        p = _WIN_FONTS / names[1 if bold else 0]
        if p.is_file():
            return str(p)
        log.warning("缺少系统字体 %s，%s 文字改用 Noto（可能显示为方框）", p, lang)
    return str(_FONT_CJK)


@functools.lru_cache(maxsize=4)
def _font_bytes(path: str) -> bytes:
    with open(path, "rb") as f:  # 先读进内存：多个字号共享一份 bytes，也避开 FreeType 的中文路径问题
        return f.read()


@functools.lru_cache(maxsize=192)
def _font(lang: str, size: int, weight: int) -> ImageFont.FreeTypeFont:
    path = _font_file(lang, weight >= 700)
    font = ImageFont.truetype(io.BytesIO(_font_bytes(path)), size, layout_engine=ImageFont.Layout.BASIC)
    if path == str(_FONT_CJK):
        try:
            font.set_variation_by_axes([float(weight)])
        except Exception as e:  # noqa: BLE001 - FreeType 不支持可变字体时用默认字重
            log.debug("设置字重失败：%s", e)
    return font


# ---- 文字排版 / 渲染 ----
_V_ROTATE = frozenset("ー－-—―～〜~…‥「」『』（）()【】〈〉《》＜＞<>［］[]＝=")  # 竖排时顺时针转 90°
_V_PUNCT = frozenset("、。，．,.")  # 竖排时挪到字格右上
_SMALL_KANA = frozenset("ぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮヵヶ")
_NO_HEAD = _V_PUNCT | _SMALL_KANA | frozenset("！？!?…‥ー～）」』】")  # 简单禁则：不放行首


def _pick(rng: np.random.Generator, seq: Sequence[Any]) -> Any:
    return seq[int(rng.integers(len(seq)))]


def _wrap(text: str, lang: str, max_chars: int) -> list[str]:
    """折行：ja / zh 逐字切（标点、长音、小假名不放行首）；en / ko 按空格贪心。"""
    max_chars, lines, cur = max(1, int(max_chars)), [], ""
    if lang in ("ja", "zh"):
        for ch in text:
            if len(cur) >= max_chars and ch not in _NO_HEAD:
                lines.append(cur)
                cur = ""
            cur += ch
    else:
        for word in text.split():
            if cur and len(cur) + 1 + len(word) > max_chars:
                lines.append(cur)
                cur = word
            else:
                cur = f"{cur} {word}" if cur else word
    if cur:
        lines.append(cur)
    return lines


def _gpad(size: int, stroke: int) -> int:
    return stroke + max(3, size // 6)


def _bbox(m: np.ndarray) -> Box | None:
    """bool 数组的外接框（开区间）；全空返回 None。"""
    ys = np.flatnonzero(m.any(axis=1))
    if ys.size == 0:
        return None
    xs = np.flatnonzero(m.any(axis=0))
    return (int(xs[0]), int(ys[0]), int(xs[-1]) + 1, int(ys[-1]) + 1)


@functools.lru_cache(maxsize=4096)
def _glyph(ch: str, font: ImageFont.FreeTypeFont, size: int, stroke: int) -> tuple[np.ndarray, np.ndarray]:
    """竖排单字 → (字芯 alpha, 含描边 alpha)，画布边长 size+2*gp，em 字格居中。"""
    gp = _gpad(size, stroke)
    n = size + 2 * gp
    x, y = float(gp), gp + size * 0.88  # 基线：CJK 表意字框 0.88em 处
    if ch in _V_PUNCT:
        x, y = x + size * 0.58, y - size * 0.58
    elif ch in _SMALL_KANA:
        x, y = x + size * 0.1, y - size * 0.1
    elif ch not in _V_ROTATE and (font.getlength(ch) < size * 0.9 or ch in "！？"):
        left, _, right, _ = font.getbbox(ch, anchor="ls")  # 半角 / 窄字符按墨迹水平居中
        x = gp + (size - (right - left)) / 2 - left
    out = []
    for sw in (0, stroke):
        im = Image.new("L", (n, n), 0)
        ImageDraw.Draw(im).text((x, y), ch, font=font, fill=255, anchor="ls", stroke_width=sw, stroke_fill=255)
        if ch in _V_ROTATE:
            im = im.transpose(Image.Transpose.ROTATE_270)
        out.append(np.asarray(im))
    return out[0], out[1]


def _render_text(lines: Sequence[str], font: ImageFont.FreeTypeFont, size: int, direction: str,
                 lang: str, stroke: int = 0) -> tuple[np.ndarray, np.ndarray] | None:
    """多行（横排，居中）/ 多列（竖排，从右到左、顶对齐）→ 紧贴裁剪的 (字芯 alpha, 含描边 alpha)。"""
    gp = _gpad(size, stroke)
    with _LOCK:
        if direction == "v":
            px, py = round(size * 1.35), max(size, round(size * 1.03))
            w = px * (len(lines) - 1) + size + 2 * gp
            h = py * max(len(c) for c in lines) + 2 * gp
            fill, alla = np.zeros((h, w), np.uint8), np.zeros((h, w), np.uint8)
            for ci, col in enumerate(lines):
                x0 = w - 2 * gp - size - ci * px
                for k, ch in enumerate(col):
                    if ch.isspace():
                        continue
                    gf, ga = _glyph(ch, font, size, stroke)
                    for dst, src in ((fill, gf), (alla, ga)):
                        sub = dst[k * py:k * py + src.shape[0], x0:x0 + src.shape[1]]
                        np.maximum(sub, src, out=sub)
        else:
            pitch = max(size + 1, round(size * {"en": 1.16, "ko": 1.3}.get(lang, 1.24)))
            widths = [font.getlength(t) for t in lines]
            w = int(math.ceil(max(widths))) + 2 * gp
            h = pitch * (len(lines) - 1) + round(size * 1.3) + 2 * gp
            ims = [Image.new("L", (w, h), 0) for _ in range(2 if stroke else 1)]
            for i, (t, tw) in enumerate(zip(lines, widths)):
                xy = (gp + (w - 2 * gp - tw) / 2, gp + size * 0.92 + i * pitch)
                for im, sw in zip(ims, (0, stroke)):
                    ImageDraw.Draw(im).text(xy, t, font=font, fill=255, anchor="ls", stroke_width=sw, stroke_fill=255)
            fill = np.array(ims[0])
            alla = np.array(ims[-1]) if stroke else fill
    bb = _bbox(alla > 0)
    if bb is None:
        return None
    x1, y1, x2, y2 = bb
    return np.ascontiguousarray(fill[y1:y2, x1:x2]), np.ascontiguousarray(alla[y1:y2, x1:x2])


def _rotate(fill: np.ndarray, alla: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray] | None:
    """整块文字旋转（扩展画布，双线性抗锯齿），再紧贴裁剪。"""
    h, w = alla.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    c, s = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * s + w * c) + 4, int(h * c + w * s) + 4
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    f2, a2 = (cv2.warpAffine(a, m, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=0) for a in (fill, alla))
    bb = _bbox(a2 > 0)
    if bb is None:
        return None
    x1, y1, x2, y2 = bb
    return np.ascontiguousarray(f2[y1:y2, x1:x2]), np.ascontiguousarray(a2[y1:y2, x1:x2])


# ---- 气泡形状 ----
def _draw_shape(rng: np.random.Generator, shape: str, hw: float, hh: float, pad: float,
                k: float) -> tuple[np.ndarray, float, float]:
    """在足够大的局部画布上画出气泡内部（uint8 0/255），返回 (画布, 中心 x, 中心 y)。
    hw / hh 是“文字框 + 内边距”的半宽半高；椭圆类半轴 = k 倍（k > √2 时文字框四角都在椭圆内）。"""
    a, b = hw * k, hh * k
    big_a, big_b = 0.0, 0.0
    if shape == "tall":
        b = max(b, a * 1.9)
    elif shape == "roundrect":
        big_a, big_b = hw + pad * 0.3, hh + pad * 0.3
    elif shape == "wide":  # 两端半圆的“药丸”
        big_b = hh * 1.15 + pad * 0.3
        big_a = max(hw + 0.75 * big_b, 2.3 * big_b)
    ex, ey = 1.85 * max(a, big_a) + 12, 1.85 * max(b, big_b) + 12
    m = np.zeros((int(2 * ey), int(2 * ex)), np.uint8)
    cx, cy = m.shape[1] / 2, m.shape[0] / 2
    if shape in ("roundrect", "wide"):
        r = big_b if shape == "wide" else min(rng.uniform(0.3, 1.0) * min(big_a, big_b), 3.0 * pad)
        x1, y1, x2, y2 = cx - big_a, cy - big_b, cx + big_a, cy + big_b
        cv2.rectangle(m, (round(x1 + r), round(y1)), (round(x2 - r), round(y2)), 255, -1)
        cv2.rectangle(m, (round(x1), round(y1 + r)), (round(x2), round(y2 - r)), 255, -1)
        for qx in (x1 + r, x2 - r):
            for qy in (y1 + r, y2 - r):
                cv2.circle(m, (round(qx), round(qy)), round(r), 255, -1)
        return m, cx, cy
    cv2.ellipse(m, (round(cx), round(cy)), (round(a), round(b)), 0, 0, 360, 255, -1)
    if shape == "cloud":  # 椭圆边上一圈小圆 → 云朵
        rb = rng.uniform(0.16, 0.24) * min(a, b) + 2
        n = max(8, int(2 * math.pi * math.sqrt((a * a + b * b) / 2) / (rb * 1.5)))
        for i in range(n):
            t = 2 * math.pi * (i + rng.uniform(-0.15, 0.15)) / n
            cv2.circle(m, (round(cx + a * math.cos(t)), round(cy + b * math.sin(t))),
                       round(rb * rng.uniform(0.85, 1.15)), 255, -1)
    elif shape == "spiky":  # 爆炸 / 喊叫气泡：谷点在椭圆上，尖角向外
        n = int(rng.integers(14, 24))
        pts = []
        for i in range(2 * n):
            t = math.pi * i / n + rng.uniform(-0.04, 0.04)
            rr = 1.0 if i % 2 == 0 else rng.uniform(1.28, 1.6)
            pts.append((cx + a * rr * math.cos(t), cy + b * rr * math.sin(t)))
        cv2.fillPoly(m, [np.round(np.array(pts)).astype(np.int32)], 255)
    elif shape == "tail":  # 椭圆 + 三角尾巴（多数朝下）
        ang = rng.uniform(0.15, 0.85) * math.pi + (0.0 if rng.random() < 0.75 else math.pi)
        d, length, e = rng.uniform(0.14, 0.26), rng.uniform(1.4, 1.8), rng.uniform(-0.3, 0.3)
        tri = [(cx + a * 0.92 * math.cos(ang - d), cy + b * 0.92 * math.sin(ang - d)),
               (cx + a * length * math.cos(ang + e), cy + b * length * math.sin(ang + e)),
               (cx + a * 0.92 * math.cos(ang + d), cy + b * 0.92 * math.sin(ang + d))]
        cv2.fillPoly(m, [np.round(np.array(tri)).astype(np.int32)], 255)
    return m, cx, cy


def _shape_masks(rng: np.random.Generator, shape: str, text_on: np.ndarray, size: int,
                 t: int) -> tuple[np.ndarray, np.ndarray, int, int] | None:
    """围绕文字生成气泡 → (outer, inner, 文字左上 x, y)，局部坐标，四周留 3px 给抗锯齿光晕。
    验证文字像素全部落在 inner 内，不满足就加大内边距重试。"""
    th, tw = text_on.shape
    pad = max(3.0, size * rng.uniform(0.3, 0.45))
    k = rng.uniform(1.42, 1.6)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * t + 1, 2 * t + 1))
    for _ in range(5):
        canvas, cx, cy = _draw_shape(rng, shape, tw / 2 + pad, th / 2 + pad, pad, k)
        bb = _bbox(canvas > 0)
        if bb is not None:
            mg = t + 3
            x1, y1 = max(0, bb[0] - mg), max(0, bb[1] - mg)
            crop = canvas[y1:bb[3] + mg, x1:bb[2] + mg]
            outer, inner = cv2.dilate(crop, kernel) > 0, crop > 0
            tx, ty = round(cx - x1 - tw / 2), round(cy - y1 - th / 2)
            sub = inner[max(0, ty):ty + th, max(0, tx):tx + tw]
            if tx >= 0 and ty >= 0 and sub.shape == text_on.shape and bool(sub[text_on].all()):
                return outer, inner, tx, ty
        pad, k = pad * 1.3, k + 0.12
    return None


# ---- 背景 ----
def _screentone(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    """45° 网点：周期 2q 的 tile（四角 + 中心各一个点），抗锯齿圆点。"""
    q = int(rng.integers(3, 6))
    n, cov = 2 * q, float(rng.uniform(0.12, 0.45))
    r = math.sqrt(cov * n * n / (2 * math.pi))
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32) + 0.5
    pts = np.array([(0, 0), (n, 0), (0, n), (n, n), (q, q)], np.float32)
    d = np.sqrt(((xx[..., None] - pts[:, 0]) ** 2 + (yy[..., None] - pts[:, 1]) ** 2).min(axis=-1))
    tile = (255.0 * (1.0 - np.clip(r + 0.5 - d, 0.0, 1.0)) + 0.5).astype(np.uint8)
    ox, oy = int(rng.integers(n)), int(rng.integers(n))
    g = np.tile(tile, (h // n + 2, w // n + 2))[oy:oy + h, ox:ox + w]
    return cv2.cvtColor(np.ascontiguousarray(g), cv2.COLOR_GRAY2BGR)


def _background(rng: np.random.Generator, kind: str, h: int, w: int) -> np.ndarray:
    if kind == "screentone":
        return _screentone(rng, h, w)
    if kind == "gradient":
        c1, c2 = rng.uniform(70, 255, 3)[::-1], rng.uniform(40, 255, 3)[::-1]  # RGB → BGR
        ang = rng.uniform(0, 2 * math.pi)
        t = np.add.outer(np.arange(h, dtype=np.float32) * math.sin(ang), np.arange(w, dtype=np.float32) * math.cos(ang))
        t = (t - t.min()) / max(float(np.ptp(t)), 1e-6)
        return (c1.astype(np.float32) + t[..., None] * (c2 - c1).astype(np.float32) + 0.5).astype(np.uint8)
    if kind == "texture":
        low = cv2.resize(rng.standard_normal((h // 24 + 2, w // 24 + 2), dtype=np.float32), (w, h),
                         interpolation=cv2.INTER_CUBIC)
        g = rng.uniform(150, 225) + rng.uniform(12, 30) * low
        g += rng.uniform(4, 10) * rng.standard_normal((h, w), dtype=np.float32)
        return np.clip(g[..., None] + rng.uniform(-12, 12, 3).astype(np.float32), 0, 255).astype(np.uint8)
    if kind == "photo":  # 低频色块 + 模糊物体 + 颗粒，模拟照片 / 彩色背景
        sh, sw = max(4, h // 8), max(4, w // 8)
        base = rng.uniform(30, 230, (int(rng.integers(2, 5)), int(rng.integers(2, 5)), 3)).astype(np.float32)
        small = cv2.resize(base, (sw, sh), interpolation=cv2.INTER_CUBIC)
        for _ in range(int(rng.integers(4, 10))):
            col = rng.uniform(0, 255, 3).tolist()
            c = (int(rng.integers(sw)), int(rng.integers(sh)))
            ax = (int(rng.integers(2, max(3, sw // 3))), int(rng.integers(2, max(3, sh // 3))))
            if rng.random() < 0.5:
                cv2.ellipse(small, c, ax, float(rng.uniform(0, 180)), 0, 360, col, -1, cv2.LINE_AA)
            else:
                cv2.rectangle(small, (c[0] - ax[0], c[1] - ax[1]), (c[0] + ax[0], c[1] + ax[1]), col, -1)
        img = cv2.resize(cv2.GaussianBlur(small, (0, 0), 1.0), (w, h), interpolation=cv2.INTER_LINEAR)
        img += rng.standard_normal((h, w, 1), dtype=np.float32) * rng.uniform(3, 8)
        return np.clip(img, 0, 255).astype(np.uint8)
    return np.full((h, w, 3), 255, np.uint8)


def _split(rng: np.random.Generator, total: int, n: int) -> list[int]:
    wts = rng.uniform(0.6, 1.4, n)
    parts = np.floor(wts / wts.sum() * total).astype(int)
    parts[-1] = total - parts[:-1].sum()
    return [int(p) for p in parts]


def _page_panels(rng: np.random.Generator, w: int, h: int) -> list[Box]:
    """日漫式分格：2~4 行、每行 1~3 格，页边 4.5%、格间 1.8%（相对宽度）。"""
    m, g = max(4, round(w * 0.045)), max(4, round(w * 0.018))
    rows = int(rng.integers(2, 5))
    out: list[Box] = []
    y = m
    for rh in _split(rng, h - 2 * m - g * (rows - 1), rows):
        cols = int(rng.integers(1, 4))
        x = m
        for cw in _split(rng, w - 2 * m - g * (cols - 1), cols):
            out.append((x, y, x + cw, y + rh))
            x += cw + g
        y += rh + g
    return out


# ---- 元素生成与合成 ----
def _rgb(v: Any) -> tuple[int, int, int]:
    return (int(v[0]), int(v[1]), int(v[2]))


def _vivid(rng: np.random.Generator) -> tuple[int, int, int]:
    r, g, b = colorsys.hsv_to_rgb(rng.uniform(0, 1), rng.uniform(0.75, 1), rng.uniform(0.7, 1))
    return (int(r * 255), int(g * 255), int(b * 255))


def _pick_text(rng: np.random.Generator, lang: str, sfx: bool = False) -> str:
    pool = (SFX if sfx else LINES)[lang]
    s = _pick(rng, pool)
    if not sfx and rng.random() < 0.2:  # 偶尔两句连成长对白
        s = s + (" " if lang in ("en", "ko") else "") + _pick(rng, pool)
    return s


def _direction(rng: np.random.Generator, lang: str, shape: str, forced: str | None, p_v: float) -> str:
    if lang != "ja":
        return "h"
    if forced in ("v", "h"):
        return forced
    if shape in ("wide", "tall"):
        return "h" if shape == "wide" else "v"
    return "v" if rng.random() < p_v else "h"


def _max_chars(rng: np.random.Generator, lang: str, d: str, shape: str) -> int:
    lo, hi = {"v": (4, 8), "en": (9, 16), "ko": (5, 10)}.get(d if d == "v" else lang, (6, 11))
    if shape == "tall":
        lo, hi = (7, 11) if d == "v" else (max(3, lo // 2), max(4, hi // 2))
    elif shape == "wide":
        lo, hi = (2, 4) if d == "v" else (hi, hi + 8)
    return int(rng.integers(lo, hi + 1))


def _make_bubble(rng: np.random.Generator, shape: str, lang: str, forced: str | None, s0: float,
                 unit: float, scale: float) -> dict[str, Any] | None:
    d = _direction(rng, lang, shape, forced, 0.8)
    size = max(10, round(s0 * rng.uniform(0.9, 1.1) * (1.15 if shape == "spiky" else 1.0) * scale))
    text = _pick_text(rng, lang)
    lines = _wrap(text, lang, _max_chars(rng, lang, d, shape))
    weight = 900 if shape == "spiky" else int(_pick(rng, (500, 600, 700)))
    res = _render_text(lines, _font(lang, size, weight), size, d, lang)
    if res is None:
        return None
    alpha = res[1]
    t = max(2, round(unit * rng.uniform(0.0015, 0.0045)))  # 描边粗细
    sm = _shape_masks(rng, shape, alpha > 0, size, t)
    if sm is None:
        return None
    fill = (255, 255, 255) if rng.random() < 0.7 else _rgb(rng.integers(222, 256, 3))
    return {"outer": sm[0], "inner": sm[1], "tx": sm[2], "ty": sm[3], "alpha": alpha, "shape": shape,
            "text": text if lang in ("ja", "zh") else " ".join(text.split()), "lines": lines, "direction": d,
            "lang": lang, "font_size": size, "fill": fill, "text_color": _rgb(rng.integers(0, 35, 3)),
            "stroke": (int(rng.integers(0, 30)),) * 3, "stroke_width": t}


def _make_free(rng: np.random.Generator, kind: str, lang: str, forced: str | None, s0: float,
               scale: float) -> dict[str, Any] | None:
    is_sfx = kind == "color"
    text = _pick_text(rng, lang, sfx=is_sfx)
    d = _direction(rng, lang, "free", forced, 0.5)
    if is_sfx:  # 彩色描边、大号、倾斜
        size = max(12, round(s0 * rng.uniform(2.0, 3.2) * scale))
        stroke = max(2, round(size * rng.uniform(0.07, 0.12)))
        lines, weight = [text], 900
        fill = _pick(rng, ((255, 255, 255), (20, 20, 20), _vivid(rng)))
        stroke_rgb, angle = _vivid(rng), float(rng.uniform(6, 25) * _pick(rng, (-1, 1)))
    else:
        size = max(10, round(s0 * rng.uniform(0.85, 1.05) * scale))
        stroke = max(2, round(size * rng.uniform(0.12, 0.2)))
        lines, weight, angle = _wrap(text, lang, _max_chars(rng, lang, d, "free")), 700, 0.0
        fill, stroke_rgb = ((15, 15, 15), (255, 255, 255)) if kind == "white" else ((255, 255, 255), (0, 0, 0))
    res = _render_text(lines, _font(lang, size, weight), size, d, lang, stroke)
    if res is not None and angle:
        res = _rotate(res[0], res[1], angle)
    if res is None:
        return None
    return {"fill_a": res[0], "alpha": res[1], "text": text if lang in ("ja", "zh") else " ".join(text.split()),
            "lines": lines, "outline": kind, "is_sfx": is_sfx, "direction": d, "lang": lang, "font_size": size,
            "text_color": fill, "stroke_color": stroke_rgb, "angle": angle}


def _blend(img: np.ndarray, a: np.ndarray, rgb: tuple[int, int, int], x: int, y: int) -> None:
    """img[y:, x:] = img*(1-a) + rgb*a（a 为 0..1 float 或 0..255 uint8），超出页面的部分裁掉。"""
    if a.dtype == np.uint8:
        a = a.astype(np.float32) * (1.0 / 255.0)
    hh, ww = img.shape[:2]
    x1, y1, x2, y2 = max(0, x), max(0, y), min(ww, x + a.shape[1]), min(hh, y + a.shape[0])
    if x1 >= x2 or y1 >= y2:
        return
    a = a[y1 - y:y2 - y, x1 - x:x2 - x, None]
    roi = img[y1:y2, x1:x2]
    roi[...] = (roi * (1.0 - a) + np.array(rgb[::-1], np.float32) * a + 0.5).astype(np.uint8)


def _full(hw: tuple[int, int], local: np.ndarray, x: int, y: int) -> np.ndarray:
    m = np.zeros(hw, bool)
    m[y:y + local.shape[0], x:x + local.shape[1]] = local
    return m


def _place(rng: np.random.Generator, hw: tuple[int, int], size: tuple[int, int], occupied: list[Box],
           margin: int, gap: int, tries: int = 80) -> tuple[int, int] | None:
    """拒绝采样找一个不和已有元素重叠（含间隙）的位置。"""
    (h, w), (H, W) = hw, size
    if w > W - 2 * margin or h > H - 2 * margin:
        return None
    for _ in range(tries):
        x, y = int(rng.integers(margin, W - margin - w + 1)), int(rng.integers(margin, H - margin - h + 1))
        if all(not (x < o[2] + gap and o[0] < x + w + gap and y < o[3] + gap and o[1] < y + h + gap)
               for o in occupied):
            return x, y
    return None


def _put_bubble(img: np.ndarray, truth: dict, b: dict[str, Any], x: int, y: int) -> None:
    outer, inner, tx, ty, alpha = b["outer"], b["inner"], b["tx"], b["ty"], b["alpha"]
    _blend(img, cv2.GaussianBlur(outer.astype(np.float32), (0, 0), 0.6), b["stroke"], x, y)
    _blend(img, cv2.GaussianBlur(inner.astype(np.float32), (0, 0), 0.6), b["fill"], x, y)
    _blend(img, alpha, b["text_color"], x + tx, y + ty)
    hw = img.shape[:2]
    ob = _bbox(outer)
    th, tw = alpha.shape
    truth["bubbles"].append({
        "box": (x + ob[0], y + ob[1], x + ob[2], y + ob[3]), "mask": _full(hw, outer, x, y),
        "inner_mask": _full(hw, inner, x, y), "text_box": (x + tx, y + ty, x + tx + tw, y + ty + th),
        "text_mask": _full(hw, alpha > 0, x + tx, y + ty),
        **{k: b[k] for k in ("shape", "text", "lines", "direction", "lang", "font_size", "fill", "text_color",
                             "stroke_width")}})


def _put_free(img: np.ndarray, truth: dict, f: dict[str, Any], x: int, y: int) -> None:
    alpha, fill_a = f["alpha"], f["fill_a"]
    _blend(img, alpha, f["stroke_color"], x, y)
    _blend(img, fill_a, f["text_color"], x, y)
    hw = img.shape[:2]
    truth["free"].append({
        "box": (x, y, x + alpha.shape[1], y + alpha.shape[0]), "text_mask": _full(hw, alpha > 0, x, y),
        "fill_mask": _full(hw, fill_a > 0, x, y),
        **{k: f[k] for k in ("text", "lines", "outline", "is_sfx", "direction", "lang", "font_size", "text_color",
                             "stroke_color", "angle")}})


def _populate(rng: np.random.Generator, img: np.ndarray, truth: dict, unit: float, nb: int, nf: int,
              lang: str, shapes: Sequence[str] | None, forced: str | None, size_range: tuple[float, float]) -> None:
    """放气泡、再放气泡外文字；放不下就缩小重试，仍不行则跳过（真值以实际放进去的为准）。"""
    hw = img.shape[:2]
    s0 = unit * rng.uniform(*size_range)  # 本页对白基准字号（相对页面）
    gap, margin = max(3, round(unit * 0.012)), max(2, round(unit * 0.015))
    occupied: list[Box] = []
    for i in range(nb):
        shape = shapes[i % len(shapes)] if shapes else _pick(rng, _SHAPE_POOL)
        for scale in (1.0, 0.8, 0.64):
            b = _make_bubble(rng, shape, lang, forced, s0, unit, scale)
            pos = None if b is None else _place(rng, b["outer"].shape, hw, occupied, margin, gap)
            if pos is not None:
                _put_bubble(img, truth, b, *pos)
                occupied.append((pos[0], pos[1], pos[0] + b["outer"].shape[1], pos[1] + b["outer"].shape[0]))
                break
        else:
            log.debug("气泡 %d（%s）放不下，跳过", i, shape)
    off = int(rng.integers(len(FREE_KINDS)))
    for j in range(nf):
        kind = FREE_KINDS[(off + j) % len(FREE_KINDS)]
        for scale in (1.0, 0.75, 0.55):
            f = _make_free(rng, kind, lang, forced, s0, scale)
            pos = None if f is None else _place(rng, f["alpha"].shape, hw, occupied, margin, gap)
            if pos is not None:
                _put_free(img, truth, f, *pos)
                occupied.append((pos[0], pos[1], pos[0] + f["alpha"].shape[1], pos[1] + f["alpha"].shape[0]))
                break
        else:
            log.debug("气泡外文字 %d（%s）放不下，跳过", j, kind)


def _check_args(width: int, height: int, background: str, lang: str, shapes: Sequence[str] | None) -> None:
    if background not in BACKGROUNDS:
        raise ValueError(f"未知背景 {background!r}，可选：{BACKGROUNDS}")
    if lang not in LANGS:
        raise ValueError(f"未知语言 {lang!r}，可选：{LANGS}")
    if width < 64 or height < 64:
        raise ValueError(f"页面太小：{width}x{height}（宽高至少 64px）")
    bad = [s for s in (shapes or ()) if s not in SHAPES]
    if bad:
        raise ValueError(f"未知气泡形状 {bad}，可选：{SHAPES}")


def _new_truth(seed: int, w: int, h: int, background: str, lang: str, kind: str, panels: list[Box]) -> dict:
    return {"bubbles": [], "free": [], "meta": {"seed": seed, "width": w, "height": h, "background": background,
                                                "lang": lang, "kind": kind, "panels": list(panels)}}


# ---- 公开接口 ----
def make_page(seed: int, width: int = 1140, height: int = 1600, background: str = "white",
              n_bubbles: int | None = None, n_free: int | None = None, lang: str = "ja", *,
              shapes: Sequence[str] | None = None, direction: str | None = None,
              panels: bool = True) -> tuple[np.ndarray, dict[str, Any]]:
    """生成一页合成漫画 → (BGR 图, 真值)。
    n_bubbles / n_free 为 None 时随机（3~6 / 1~3）；shapes 指定时气泡按顺序循环使用这些形状；
    direction 强制日文方向 "v" / "h"（其它语言恒为横排）；panels=False 时整页一种背景、不画分格。"""
    width, height = int(width), int(height)
    _check_args(width, height, background, lang, shapes)
    rng = np.random.default_rng(seed)
    img = np.full((height, width, 3), 255, np.uint8)
    unit = min(width, height / 1.4)
    boxes = _page_panels(rng, width, height) if panels and min(width, height) >= 240 else [(0, 0, width, height)]
    lw = max(2, round(unit * 0.003))
    for x1, y1, x2, y2 in boxes:
        img[y1:y2, x1:x2] = _background(rng, background, y2 - y1, x2 - x1)
        if len(boxes) > 1:
            cv2.rectangle(img, (x1, y1), (x2 - 1, y2 - 1), (0, 0, 0), lw)
    nb = int(rng.integers(3, 7)) if n_bubbles is None else max(0, int(n_bubbles))
    nf = int(rng.integers(1, 4)) if n_free is None else max(0, int(n_free))
    truth = _new_truth(seed, width, height, background, lang, "page", boxes)
    _populate(rng, img, truth, unit, nb, nf, lang, shapes, direction, (0.022, 0.030))
    return img, truth


def make_webtoon(seed: int, width: int = 800, height: int = 6000, lang: str = "ko", *,
                 n_bubbles: int | None = None, n_free: int | None = None,
                 shapes: Sequence[str] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """韩漫式长条：若干全宽 / 略内缩的场景块（随机背景）+ 白色间隔，横排气泡、旁白、拟声词。"""
    width, height = int(width), int(height)
    _check_args(width, height, "white", lang, shapes)
    rng = np.random.default_rng(seed)
    img = np.full((height, width, 3), 255, np.uint8)
    boxes: list[Box] = []
    y = int(rng.uniform(0.03, 0.08) * height)
    while y < height - 0.3 * width:
        ph = int(min(rng.uniform(0.7, 1.4) * width, height - y - 0.05 * width))
        if ph < 0.2 * width:
            break
        inset = 0 if rng.random() < 0.6 else int(width * rng.uniform(0.04, 0.08))
        img[y:y + ph, inset:width - inset] = _background(rng, _pick(rng, BACKGROUNDS), ph, width - 2 * inset)
        boxes.append((inset, y, width - inset, y + ph))
        y += ph + int(rng.uniform(0.25, 0.7) * width)
    nb = max(3, round(height / 700)) if n_bubbles is None else max(0, int(n_bubbles))
    nf = max(1, round(height / 2500)) if n_free is None else max(0, int(n_free))
    truth = _new_truth(seed, width, height, "mixed", lang, "webtoon", boxes)
    _populate(rng, img, truth, width, nb, nf, lang, shapes, None, (0.034, 0.044))
    return img, truth


def make_book(dir: str | os.PathLike[str], n: int, seed: int, fmt: str = "folder", *,  # noqa: A002 - 合同参数名
              name: str | None = None, ext: str = "png", return_truth: bool = False,
              **page_kw: Any) -> str | tuple[str, list[dict[str, Any]]]:
    """在 dir 下生成一本 n 页的书：fmt="folder" → dir/<name>/001.png…；fmt="zip" → dir/<name>.zip。
    name 默认 "合成漫画_<seed>"（故意带中文，顺便测中文路径）；未指定 background 时按页轮换五种背景。
    返回书的路径；return_truth=True 时返回 (路径, 每页真值)。"""
    if fmt not in ("folder", "zip"):
        raise ValueError(f"未知书格式 {fmt!r}（folder / zip）")
    ext = ext.lower().lstrip(".")
    if ext not in ("png", "jpg", "webp"):
        raise ValueError(f"不支持的图片格式 {ext!r}（png / jpg / webp）")
    base = os.path.join(os.fspath(dir), name or f"合成漫画_{seed}")
    out = base if fmt == "folder" else base + ".zip"
    os.makedirs(base if fmt == "folder" else os.path.dirname(out) or ".", exist_ok=True)
    truths: list[dict[str, Any]] = []
    part = out + ".part"
    zf = zipfile.ZipFile(part, "w", zipfile.ZIP_STORED) if fmt == "zip" else None
    try:
        for i in range(max(0, int(n))):
            kw = {"background": BACKGROUNDS[i % len(BACKGROUNDS)], **page_kw}
            img, truth = make_page(seed * 1009 + i, **kw)
            fname, data = f"{i + 1:03d}.{ext}", hio.encode(img, ext)
            if zf is not None:
                zf.writestr(fname, data)
            else:
                hio.write_bytes_atomic(os.path.join(out, fname), data)
            if return_truth:
                truths.append(truth)
    except BaseException:
        if zf is not None:
            zf.close()
            os.remove(part)
        raise
    if zf is not None:
        zf.close()
        os.replace(part, out)
    return (out, truths) if return_truth else out


def validate_truth(img: np.ndarray, truth: dict[str, Any]) -> list[str]:
    """检查真值自洽，返回问题列表（空 = 通过）：形状 / dtype、box 与 mask 外接框一致、
    text_mask ⊆ inner_mask ⊆ mask、元素两两不重叠、文字非空。"""
    errs: list[str] = []
    hw = img.shape[:2]
    union = np.zeros(hw, bool)
    items = [("bubble", i, b) for i, b in enumerate(truth.get("bubbles", []))]
    items += [("free", i, f) for i, f in enumerate(truth.get("free", []))]
    for kind, i, e in items:
        tag = f"{kind}[{i}]"
        keys = ("mask", "inner_mask", "text_mask") if kind == "bubble" else ("text_mask", "fill_mask")
        if any(e[k].shape != hw or e[k].dtype != bool for k in keys):
            errs.append(f"{tag} mask 尺寸 / dtype 不对")
            continue
        if not str(e["text"]).strip():
            errs.append(f"{tag} 文字为空")
        tbox = e["text_box"] if kind == "bubble" else e["box"]
        if _bbox(e["text_mask"]) != tuple(tbox):
            errs.append(f"{tag} 文字框 {tbox} ≠ text_mask 外接框 {_bbox(e['text_mask'])}")
        if kind == "bubble":
            if _bbox(e["mask"]) != tuple(e["box"]):
                errs.append(f"{tag} box ≠ mask 外接框")
            if (e["text_mask"] & ~e["inner_mask"]).any():
                errs.append(f"{tag}({e['shape']}) 文字超出气泡内部")
            if (e["inner_mask"] & ~e["mask"]).any():
                errs.append(f"{tag} inner_mask 超出 mask")
        elif (e["fill_mask"] & ~e["text_mask"]).any():
            errs.append(f"{tag} fill_mask 超出 text_mask")
        region = e["mask"] if kind == "bubble" else e["text_mask"]
        if (union & region).any():
            errs.append(f"{tag} 与其它元素重叠")
        union |= region
    return errs


def _outline(vis: np.ndarray, m: np.ndarray, rgb: tuple[int, int, int], th: int) -> None:
    bb = _bbox(m)
    if bb is None:
        return
    sub = np.ascontiguousarray(m[bb[1]:bb[3], bb[0]:bb[2]]).astype(np.uint8)
    cs, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, offset=(bb[0], bb[1]))
    cv2.drawContours(vis, cs, -1, rgb[::-1], th, cv2.LINE_AA)


def save_debug(img: np.ndarray, truth: dict[str, Any], path: str | os.PathLike[str] | None) -> np.ndarray:
    """画出真值叠加图（气泡轮廓绿、内部青、文字框红、文字像素染色；气泡外文字橙框、拟声词品红）并写盘。"""
    vis = img.copy()
    for b in truth.get("bubbles", []):
        m = b["text_mask"]
        vis[m] = (vis[m] * 0.45 + np.array((40, 40, 230), np.float32) * 0.55).astype(np.uint8)
        _outline(vis, b["mask"], (0, 200, 0), 3)
        _outline(vis, b["inner_mask"], (0, 200, 255), 1)
        x1, y1, x2, y2 = b["text_box"]
        cv2.rectangle(vis, (x1, y1), (x2 - 1, y2 - 1), (0, 0, 255), 2)
        cv2.putText(vis, f"{b['shape']}/{b['direction']}", (b["box"][0], max(14, b["box"][1] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 0), 2, cv2.LINE_AA)
    for f in truth.get("free", []):
        m, col = f["text_mask"], ((255, 0, 255) if f["is_sfx"] else (255, 140, 0))
        vis[m] = (vis[m] * 0.5 + np.array(col[::-1], np.float32) * 0.5).astype(np.uint8)
        x1, y1, x2, y2 = f["box"]
        cv2.rectangle(vis, (x1, y1), (x2 - 1, y2 - 1), col[::-1], 2)
        cv2.putText(vis, ("sfx/" if f["is_sfx"] else "") + f["outline"], (x1, max(14, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col[::-1], 2, cv2.LINE_AA)
    if path:
        hio.write(os.fspath(path), vis)
    return vis


def contact_sheet(images: Sequence[np.ndarray], cols: int = 4, cell_width: int = 300,
                  labels: Sequence[str] | None = None) -> np.ndarray:
    """把多张图缩成等宽格子拼成一张接触表（labels 需为 ASCII）。"""
    if not images:
        return np.full((cell_width, cell_width, 3), 128, np.uint8)
    cells = []
    for i, im in enumerate(images):
        h = max(1, round(im.shape[0] * cell_width / im.shape[1]))
        c = cv2.resize(im, (cell_width, h), interpolation=cv2.INTER_AREA)
        if labels and i < len(labels):
            cv2.putText(c, str(labels[i]), (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        cells.append(c)
    ch = max(c.shape[0] for c in cells)
    blank = np.full((ch, cell_width, 3), 128, np.uint8)
    rows = []
    for r in range(0, len(cells), cols):
        row = [np.pad(c, ((0, ch - c.shape[0]), (0, 0), (0, 0)), constant_values=128) for c in cells[r:r + cols]]
        row += [blank] * (cols - len(row))
        rows.append(np.hstack([np.pad(c, ((4, 4), (4, 4), (0, 0)), constant_values=128) for c in row]))
    return np.vstack(rows)


__all__ = ["make_page", "make_webtoon", "make_book", "save_debug", "validate_truth", "contact_sheet",
           "BACKGROUNDS", "SHAPES", "LANGS", "FREE_KINDS", "LINES", "SFX", "JA_LINES"]
