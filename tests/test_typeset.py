"""typeset 模块测试：合成气泡（带真值内部 mask）量化“字绝不出气泡”，外加禁则 / 竖排标点 / 字号范围 /
覆盖项 / 颜色 / 健壮性 / 性能 / 真实页面（p012）。调试图输出到 tests/out/typeset/。"""
from __future__ import annotations

import copy
import json
import logging
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from hanhua import imageio as hio
from hanhua.config import DEFAULTS
from hanhua.types import TextBlock
from hanhua.typeset import FontRegistry, Typesetter
from hanhua.typeset.layout import (
    NO_END,
    NO_START,
    VFORMS,
    _fix_orphan,
    build_tokens,
    estimate_src_font,
    font_range,
    page_ref,
)

log = logging.getLogger("hanhua.tests.typeset")

SHAPES = ("ellipse", "roundrect", "cloud", "spiky", "tall", "wide")
LENGTHS = (2, 8, 20, 50, 120)
SAMPLE = ("你以为现在都几点了啊！才没有那种事呢，经、经理！？你怎么会在这种地方……"
          "我们明天再一起去看看吧——好不好？「真的吗」那就这么说定了。") * 4


# ======================================================================================
# 公共工具
# ======================================================================================
@pytest.fixture(scope="module")
def fonts(project_root: Path) -> FontRegistry:
    return FontRegistry(str(project_root / "fonts"), [])


def _settings(**ts) -> dict:
    s = copy.deepcopy(DEFAULTS)
    s["typeset"].update(ts)
    return s


def _shape_mask(kind: str, page_hw: tuple[int, int], cx: int, cy: int, w: int, h: int) -> np.ndarray:
    m = np.zeros(page_hw, np.uint8)
    if kind in ("ellipse", "tall", "wide"):
        cv2.ellipse(m, (cx, cy), (w // 2, h // 2), 0, 0, 360, 1, -1)
    elif kind == "roundrect":
        r = int(0.22 * min(w, h))
        x1, y1, x2, y2 = cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2
        cv2.rectangle(m, (x1 + r, y1), (x2 - r, y2), 1, -1)
        cv2.rectangle(m, (x1, y1 + r), (x2, y2 - r), 1, -1)
        for px, py in ((x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)):
            cv2.circle(m, (px, py), r, 1, -1)
    elif kind == "cloud":
        cv2.ellipse(m, (cx, cy), (int(w * 0.40), int(h * 0.40)), 0, 0, 360, 1, -1)
        rr = int(0.17 * min(w, h))
        for t in np.linspace(0, 2 * np.pi, 13)[:-1]:
            px = int(cx + (w / 2 - rr) * np.cos(t))
            py = int(cy + (h / 2 - rr) * np.sin(t))
            cv2.circle(m, (px, py), rr, 1, -1)
    elif kind == "spiky":
        pts = []
        for i in range(32):
            t = i * np.pi / 16
            k = 1.0 if i % 2 == 0 else 0.74
            pts.append((cx + k * w / 2 * np.cos(t), cy + k * h / 2 * np.sin(t)))
        cv2.fillPoly(m, [np.array(pts, np.int32)], 1)
    else:
        raise ValueError(kind)
    return m.astype(bool)


SHAPE_SIZE = {  # (w, h)，在 1300 高的页面上
    "ellipse": (300, 420), "roundrect": (380, 260), "cloud": (400, 320),
    "spiky": (420, 420), "tall": (170, 540), "wide": (580, 180),
}


def _synth_page(lengths: dict[str, int], direction: str, seed: int = 0
                ) -> tuple[np.ndarray, list[TextBlock], dict[str, np.ndarray]]:
    """一页 6 个气泡（灰色噪声背景 + 白底黑边气泡）。返回 (页面, 块, {形状: 真值内部 mask})。"""
    H, W = 1300, 1400
    rng = np.random.default_rng(seed)
    page = np.clip(rng.normal(150, 18, (H, W, 1)), 0, 255).astype(np.uint8).repeat(3, axis=2)
    centers = {"ellipse": (220, 260), "roundrect": (700, 200), "cloud": (1170, 230),
               "spiky": (250, 830), "tall": (640, 860), "wide": (1080, 700)}
    t = max(3, int(round(0.003 * H)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * t + 1, 2 * t + 1))
    blocks: list[TextBlock] = []
    truth: dict[str, np.ndarray] = {}
    for i, kind in enumerate(SHAPES):
        cx, cy = centers[kind]
        w, h = SHAPE_SIZE[kind]
        s = _shape_mask(kind, (H, W), cx, cy, w, h)
        inner = cv2.erode(s.astype(np.uint8), k).astype(bool)
        page[s] = 0
        page[inner] = 255
        truth[kind] = inner
        ys, xs = np.nonzero(s)
        bb = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        fx, fy = (0.40, 0.40) if kind == "spiky" else (0.55, 0.62)
        tb = (int(cx - w * fx / 2), int(cy - h * fy / 2), int(cx + w * fx / 2), int(cy + h * fy / 2))
        n = lengths[kind]
        blocks.append(TextBlock(id=i, box=tb, kind="bubble", bubble_box=bb, direction=direction,
                                src_text="あ" * n, translation=SAMPLE[:n], out_direction=direction))
    return page, blocks, truth


def _changed(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.any(a != b, axis=2)


def _crop(img: np.ndarray, box: tuple[int, int, int, int], height: int = 200) -> np.ndarray:
    c = img[box[1]:box[3], box[0]:box[2]]
    s = height / c.shape[0]
    return cv2.resize(c, (max(1, int(c.shape[1] * s)), height), interpolation=cv2.INTER_AREA)


def _hstack(tiles: list[np.ndarray], pad: int = 6) -> np.ndarray:
    return np.hstack([np.pad(t, ((0, 0), (0, pad), (0, 0)), constant_values=255) for t in tiles])


def _vstack(rows: list[np.ndarray], pad: int = 6) -> np.ndarray:
    w = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, pad), (0, w - r.shape[1]), (0, 0)), constant_values=255) for r in rows]
    return np.vstack(rows)


# ======================================================================================
# 核心：合成气泡 × 长度 × 方向 —— 溢出像素必须为 0
# ======================================================================================
@pytest.mark.parametrize("direction", ["v", "h"])
def test_synthetic_bubbles_never_overflow(fonts: FontRegistry, out_dir: Path, direction: str) -> None:
    ts = Typesetter(_settings(), fonts)
    rows: dict[str, list[np.ndarray]] = {k: [] for k in SHAPES}
    stats: dict[str, dict] = {}
    for n in LENGTHS:
        page, blocks, truth = _synth_page({k: n for k in SHAPES}, direction, seed=n)
        out, details = ts.render_detailed(page, blocks)
        changed = _changed(out, page)
        inside = np.zeros_like(changed)
        for m in truth.values():
            inside |= m
        outside = int(np.count_nonzero(changed & ~inside))
        assert outside == 0, f"{direction} 长度 {n}：{outside} 个像素画出了气泡"
        sizes = []
        for b, d in zip(blocks, details):
            kind = SHAPES[b.id]
            assert d.layout is not None and b.render_font_size is not None
            if d.stage != "forced":
                assert d.clipped == 0, f"{kind} {n}字 {direction}：排版结果需要裁剪 {d.clipped}px"
            if n <= 50:
                assert not b.overflow, f"{kind} {n}字 {direction}：不应溢出（stage={d.stage}）"
            # 文字确实画出来了，而且在该气泡内部
            m = truth[kind]
            ink = changed & m
            assert ink.any(), f"{kind} {n}字：没有画出任何字"
            ys, xs = np.nonzero(m)
            area = float(m.sum())
            iy, ix = np.nonzero(ink)
            fill = (ix.max() - ix.min() + 1) * (iy.max() - iy.min() + 1) / area
            sizes.append((kind, b.render_font_size, round(fill, 3), d.stage))
            bb = b.bubble_box
            rows[kind].append(_crop(out, (bb[0] - 8, bb[1] - 8, bb[2] + 8, bb[3] + 8), 180))
        stats[str(n)] = {
            "mean_font_px": round(statistics.mean(s[1] for s in sizes), 1),
            "blocks": [{"shape": s[0], "font_px": s[1], "ink_bbox_fill": s[2], "stage": s[3]} for s in sizes],
        }
    sheet = _vstack([_hstack(rows[k]) for k in SHAPES])
    hio.write(str(out_dir / f"contact_{direction}.png"), sheet)
    with open(out_dir / f"stats_{direction}.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    log.info("合成气泡 %s：%s", direction, {k: v["mean_font_px"] for k, v in stats.items()})


def test_overflow_tiny_bubble_stays_inside(fonts: FontRegistry) -> None:
    """极小气泡 + 超长译文：overflow=True，仍然一个像素都不出气泡。"""
    H, W = 600, 600
    page = np.full((H, W, 3), 120, np.uint8)
    s = _shape_mask("ellipse", (H, W), 300, 300, 90, 70)
    inner = cv2.erode(s.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    page[s] = 0
    page[inner] = 255
    b = TextBlock(id=0, box=(270, 280, 330, 320), bubble_box=(255, 265, 345, 335), direction="v",
                  src_text="あいう", translation=SAMPLE[:120])
    out, det = Typesetter(_settings(), fonts).render_detailed(page, [b])
    assert b.overflow is True
    assert int(np.count_nonzero(_changed(out, page) & ~inner)) == 0
    assert det[0].stage in ("overflow", "forced")


def test_two_blocks_share_one_bubble(fonts: FontRegistry) -> None:
    """一个大气泡里两段话：按 Voronoi 分区，两块的渲染框不重叠、都在气泡里。"""
    H, W = 900, 1200
    page = np.full((H, W, 3), 140, np.uint8)
    s = _shape_mask("ellipse", (H, W), 600, 450, 760, 360)
    inner = cv2.erode(s.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    page[s] = 0
    page[inner] = 255
    bb = (220, 270, 980, 630)
    blocks = [TextBlock(id=0, box=(700, 340, 860, 560), bubble_box=bb, direction="v",
                        src_text="あ" * 10, translation="右边这一段话写在这里。"),
              TextBlock(id=1, box=(340, 340, 500, 560), bubble_box=bb, direction="v",
                        src_text="あ" * 10, translation="左边是另外一段话呢！")]
    out = Typesetter(_settings(), fonts).render(page, blocks)
    a, c = blocks[0].render_box, blocks[1].render_box
    assert a is not None and c is not None
    assert a[0] >= c[2] or c[0] >= a[2], f"两块文字重叠：{a} {c}"
    assert int(np.count_nonzero(_changed(out, page) & ~inner)) == 0


# ======================================================================================
# 排版规则：竖排标点 / 縦中横 / 禁则 / 孤字
# ======================================================================================
def test_vertical_punctuation_forms(fonts: FontRegistry) -> None:
    toks = build_tokens("你好，世界。「对」（嗯）、", "v", "noto-sans-sc-bold", fonts, "zh-Hans", 0.04)
    texts = [u.text for t in toks for u in t.units]
    for ch in "，。「」（）、":
        assert VFORMS[ch] in texts, f"{ch} 没有换成竖排形 {VFORMS[ch]}"


def test_tate_chu_yoko_and_ellipsis(fonts: FontRegistry) -> None:
    toks = build_tokens("真的！？等了12分钟……——", "v", "noto-sans-sc-bold", fonts, "zh-Hans", 0.04)
    units = [u for t in toks for u in t.units]
    kinds = {(u.kind, u.text) for u in units}
    assert ("tcy", "!?") in kinds
    assert ("tcy", "12") in kinds
    assert any(u.kind == "ell" and u.text == "……" for u in units), "…… 必须是一个不可拆的单元"
    assert any(u.kind == "dash" and u.text == "——" for u in units), "—— 必须是一个不可拆的单元"


def test_kinsoku_tokens(fonts: FontRegistry) -> None:
    text = "「你好」，他说：“我们（明天）去吧！”……真的吗？——嗯。"
    for d in ("h", "v"):
        toks = [t for t in build_tokens(text, d, "noto-sans-sc-bold", fonts, "zh-Hans", 0.04)
                if not t.space and not t.brk]
        rev = {v: k for k, v in VFORMS.items()}
        for i, t in enumerate(toks):
            first = rev.get(t.units[0].text, t.units[0].text)[0]
            last = rev.get(t.units[-1].text, t.units[-1].text)[-1]
            if i > 0:
                assert first not in NO_START, f"{d}: 行首禁则字 {first!r} 可能出现在行首"
            if i < len(toks) - 1:
                assert last not in NO_END, f"{d}: 行尾禁则字 {last!r} 可能出现在行尾"


def test_kinsoku_in_real_layout(fonts: FontRegistry) -> None:
    """窄气泡里排一段标点密集的话：每行 / 列的第一个字不是行首禁则字。"""
    H, W = 700, 500
    page = np.full((H, W, 3), 255, np.uint8)
    cv2.ellipse(page, (250, 350), (110, 300), 0, 0, 360, (0, 0, 0), 4)
    text = "唔，这样啊。好吧，那就这样！真的？嗯、嗯，明白了。「好」，走吧……"
    for d in ("h", "v"):
        b = TextBlock(id=0, box=(190, 150, 310, 550), bubble_box=(138, 48, 362, 652), direction="v",
                      src_text="あ" * 30, translation=text, out_direction=d)
        _, det = Typesetter(_settings(), fonts).render_detailed(page, [b])
        lay = det[0].layout
        assert lay is not None
        key = (lambda p: round(p.x)) if d == "v" else (lambda p: round(p.y))
        lines: dict[int, list] = {}
        for p in lay.placed:
            lines.setdefault(key(p), []).append(p)
        rev = {v: k for k, v in VFORMS.items()}
        for _, ps in lines.items():
            ps.sort(key=lambda p: p.y if d == "v" else p.x)
            head = rev.get(ps[0].unit.text, ps[0].unit.text)[0]
            assert head not in NO_START, f"{d}: 行首出现了禁则字 {head!r}"


def test_fix_orphan_moves_one_char(fonts: FontRegistry) -> None:
    toks = build_tokens("一二三四五六", "h", "noto-sans-sc-bold", fonts, "zh-Hans", 0.0)
    lines = _fix_orphan(toks, [5.2, 5.2], [(0, 5), (5, 6)])
    assert lines == [(0, 4), (4, 6)]


# ======================================================================================
# 字号范围：原文字号估计 + 页面相对化 + 设置覆盖
# ======================================================================================
def test_font_estimate_uses_main_axis() -> None:
    # 单列竖排 11 字，框横向很松（p012 右气泡的真实检测框）
    b = TextBlock(id=0, box=(913, 192, 1057, 630), direction="v", src_text="そんなことはありません")
    est = estimate_src_font(b, 1600)
    assert est is not None and 33 <= est <= 45
    # 两列
    b2 = TextBlock(id=1, box=(86, 517, 219, 945), direction="v", src_text="マ…マネージャー…！？なんでこんな所に…！？")
    est2 = estimate_src_font(b2, 1600)
    assert est2 is not None and 33 <= est2 <= 48


def test_font_range_scales_with_page() -> None:
    ts = dict(DEFAULTS["typeset"])
    b1 = TextBlock(id=0, box=(100, 100, 160, 400), direction="v", src_text="あ" * 8)
    b2 = TextBlock(id=0, box=(200, 200, 320, 800), direction="v", src_text="あ" * 8)
    r1 = font_range(estimate_src_font(b1, page_ref(1140, 1600)), page_ref(1140, 1600), ts)
    r2 = font_range(estimate_src_font(b2, page_ref(2280, 3200)), page_ref(2280, 3200), ts)
    assert abs(r2[0] / r1[0] - 2.0) < 0.15 and abs(r2[1] / r1[1] - 2.0) < 0.15
    # 韩漫长条：参考长度按宽度折算，不会因为页很长把字号放大
    assert page_ref(800, 20000) == pytest.approx(1280)
    ts2 = dict(ts, min_font_px=20, max_font_px=24)
    assert font_range(40.0, 1600, ts2) == (20, 24)


# ======================================================================================
# 样式与覆盖项
# ======================================================================================
def _one_bubble_page() -> tuple[np.ndarray, np.ndarray, TextBlock]:
    H, W = 800, 700
    page = np.full((H, W, 3), 130, np.uint8)
    s = _shape_mask("ellipse", (H, W), 350, 400, 360, 520)
    inner = cv2.erode(s.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    page[s] = 0
    page[inner] = 255
    b = TextBlock(id=0, box=(260, 240, 440, 560), bubble_box=(170, 140, 530, 660), direction="v",
                  src_text="あ" * 12, translation="这里是测试用的对白文字！")
    return page, inner, b


def test_block_overrides(fonts: FontRegistry) -> None:
    page, inner, base = _one_bubble_page()
    ts = Typesetter(_settings(), fonts)

    b = copy.deepcopy(base)
    b.font, b.font_size, b.out_direction = "lxgw-wenkai", 30, "h"
    _, det = ts.render_detailed(page, [b])
    assert det[0].font == "lxgw-wenkai" and det[0].direction == "h"
    assert b.render_font_size == 30 and not b.overflow

    b = copy.deepcopy(base)
    b.fill, b.stroke, b.stroke_width = (220, 20, 20), (255, 230, 0), 0.2
    out = ts.render(page, [b])
    px = out[_changed(out, page)].astype(int)            # BGR
    red = (px[:, 2] > 150) & (px[:, 1] < 80) & (px[:, 0] < 80)
    yellow = (px[:, 2] > 200) & (px[:, 1] > 180) & (px[:, 0] < 80)
    assert red.sum() > 50 and yellow.sum() > 50
    assert int(np.count_nonzero(_changed(out, page) & ~inner)) == 0

    for align in ("start", "end", "center"):
        b = copy.deepcopy(base)
        b.align = align
        out = ts.render(page, [b])
        assert b.render_box is not None and not b.overflow


def test_traditional_target_uses_tc_font(fonts: FontRegistry) -> None:
    page, _, b = _one_bubble_page()
    s = _settings()
    s["translate"]["target"] = "zh-Hant"
    _, det = Typesetter(s, fonts).render_detailed(page, [b])
    assert det[0].font.startswith("noto-sans-tc-")


def test_bubble_color_contrast(fonts: FontRegistry) -> None:
    """keep_original_color：原文颜色和气泡底色太接近时退回黑字；对比度够时用原文颜色。"""
    page, _, base = _one_bubble_page()
    ts = Typesetter(_settings(keep_original_color=True), fonts)
    b = copy.deepcopy(base)
    b.text_color = (245, 245, 245)
    out = ts.render(page, [b])
    assert out[_changed(out, page)].min() < 40
    b = copy.deepcopy(base)
    b.text_color = (20, 40, 160)                           # 深蓝（RGB）
    out = ts.render(page, [b])
    px = out[_changed(out, page)].astype(int)
    assert ((px[:, 0] > 120) & (px[:, 2] < 60)).sum() > 50


def test_free_text_has_contrast_stroke_and_sfx_font(fonts: FontRegistry) -> None:
    H, W = 600, 800
    page = np.full((H, W, 3), 40, np.uint8)                # 暗背景
    b = TextBlock(id=0, box=(200, 200, 600, 300), kind="free", direction="h", src_text="あ" * 6,
                  translation="突然出现的旁白", text_color=(255, 255, 255))
    sfx = TextBlock(id=1, box=(100, 400, 300, 520), kind="free", direction="h", src_text="ドン",
                    translation="咚！", is_sfx=True, text_color=(250, 40, 40))
    out, det = Typesetter(_settings(), fonts).render_detailed(page, [b, sfx])
    px = out[_changed(out, page)].astype(int)
    assert (px.min(axis=1) > 220).sum() > 50              # 白字
    assert det[1].font == "zcool-kuaile"
    assert b.render_box is not None and sfx.render_box is not None
    roi = det[0].region.roi
    rb = b.render_box
    assert roi[0] <= rb[0] and roi[1] <= rb[1] and rb[2] <= roi[2] and rb[3] <= roi[3]


# ======================================================================================
# 健壮性
# ======================================================================================
def test_robust_inputs(fonts: FontRegistry) -> None:
    H, W = 400, 500
    page = np.full((H, W, 3), 255, np.uint8)
    weird = [
        TextBlock(id=0, box=(-50, -50, -10, -10), translation="页外的框"),
        TextBlock(id=1, box=(100, 100, 100, 100), translation="零面积"),
        TextBlock(id=2, box=(450, 350, 900, 900), bubble_box=(0, 0, 10, 10), translation="气泡框不对"),
        TextBlock(id=3, box=(10, 10, 60, 60), translation="😀🎉\x00\x07控制字符​零宽"),
        TextBlock(id=4, box=(60, 60, 200, 200), direction="v", translation="Supercalifragilisticexpialidocious"),
        TextBlock(id=5, box=(200, 60, 300, 200), direction="h", translation="Supercalifragilisticexpialidocious"),
        TextBlock(id=6, box=(300, 200, 400, 300), translation="……！？"),
        TextBlock(id=7, box=(300, 300, 400, 380), translation="   \n  "),
        TextBlock(id=8, box=(10, 300, 90, 390), translation="长" * 2000, font="不存在的字体", font_size=-3),
        TextBlock(id=9, box=(100, 300, 200, 390), translation="跳过", skip=True),
        TextBlock(id=10, box=(200, 300, 280, 390), translation="颜色", fill=(999, -5, "x"), stroke_width=7.0),  # type: ignore[arg-type]
    ]
    ts = Typesetter(_settings(), fonts)
    for img in (page, page[:, :, 0], np.dstack([page, page[:, :, :1]]), page.astype(np.float32)):
        blocks = copy.deepcopy(weird)
        out = ts.render(img, blocks)
        assert out.shape == (H, W, 3) and out.dtype == np.uint8
        assert blocks[9].render_box is None and blocks[7].render_box is None
    assert Typesetter({}, fonts).render(page, []).shape == (H, W, 3)
    assert Typesetter(None, fonts).render(page, copy.deepcopy(weird[:3])).shape == (H, W, 3)  # type: ignore[arg-type]


# ======================================================================================
# 性能：单页 10 块 < 150ms
# ======================================================================================
def test_speed_ten_blocks(fonts: FontRegistry) -> None:
    H, W = 1600, 1140
    page = np.full((H, W, 3), 200, np.uint8)
    blocks = []
    texts = ["你以为现在都几点了啊！", "才没有那种事呢！", "经、经理！？你怎么会在这种地方……！？",
             "好的。", "我们明天再一起去看看吧——好不好？", "嗯……", "真的吗？", "那就这么说定了！",
             "等一下，你听我说完！", "喂！"]
    for i in range(10):
        cx, cy = 130 + (i % 5) * 220, 350 + (i // 5) * 700
        cv2.ellipse(page, (cx, cy), (95, 260), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(page, (cx, cy), (95, 260), 0, 0, 360, (0, 0, 0), 4)
        blocks.append(TextBlock(id=i, box=(cx - 50, cy - 200, cx + 50, cy + 200),
                                bubble_box=(cx - 97, cy - 262, cx + 97, cy + 262), direction="v",
                                src_text="あ" * 12, translation=texts[i]))
    ts = Typesetter(_settings(), fonts)
    ts.render(page, copy.deepcopy(blocks))                 # 预热（字体加载）
    times = []
    for _ in range(5):
        bs = copy.deepcopy(blocks)
        t = time.perf_counter()
        ts.render(page, bs)
        times.append((time.perf_counter() - t) * 1000)
    med = statistics.median(times)
    log.info("10 块渲染耗时 %.1fms（中位数），各次 %s", med, [round(x, 1) for x in times])
    assert med < 150, f"10 块渲染 {med:.0f}ms，超过 150ms"


# ======================================================================================
# 真实页面：p012（白色盖掉原文模拟去字），与 3.0 的痛点对照
# ======================================================================================
P012 = [  # (文字框, 气泡框, 原文, 译文) —— 框来自检测器在 p012 上的真实输出
    ((913, 192, 1057, 630), (869, 164, 1095, 661), "そんなことはありません", "才没有那种事呢！"),
    ((86, 517, 219, 945), (52, 482, 252, 969), "マ…マネージャー…！？なんでこんな所に…！？",
     "经、经理！？你怎么会在这种地方……！？"),
    ((168, 1147, 295, 1443), (139, 1111, 325, 1482), "今、何時だって思ってるの！", "你以为现在都几点了啊！"),
]


def _white_out(img: np.ndarray, box: tuple[int, int, int, int]) -> None:
    """模拟去字：框内和框边一圈底色差异大的像素（文字 + 描边）用底色盖掉。"""
    x1, y1, x2, y2 = box
    crop = img[y1:y2, x1:x2]
    ring = np.concatenate([crop[:3].reshape(-1, 3), crop[-3:].reshape(-1, 3),
                           crop[:, :3].reshape(-1, 3), crop[:, -3:].reshape(-1, 3)])
    bg = np.median(ring, axis=0)
    diff = np.abs(crop.astype(np.int16) - bg.astype(np.int16)).max(axis=2) > 28
    diff = cv2.dilate(diff.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    crop[diff] = bg.astype(np.uint8)


def test_real_page_p012(fonts: FontRegistry, real_images: dict, out_dir: Path) -> None:
    if "p012" not in real_images:
        pytest.skip("缺少 p012")
    src = real_images["p012"]
    clean = src.copy()
    blocks = []
    for i, (tb, bb, ja, zh) in enumerate(P012):
        _white_out(clean, tb)
        blocks.append(TextBlock(id=i, box=tb, bubble_box=bb, direction="v", src_text=ja, translation=zh))
    out, det = Typesetter(_settings(), fonts).render_detailed(clean, blocks)
    hio.write(str(out_dir / "real_p012.png"), out)
    changed = _changed(out, clean)
    for b, d in zip(blocks, det):
        bb = b.bubble_box
        short = min(bb[2] - bb[0], bb[3] - bb[1])
        assert not b.overflow and d.clipped == 0
        # 3.0 在这些气泡里只有约 14~22px；4.0 至少要 30px（原文约 40px）
        assert b.render_font_size is not None and b.render_font_size >= 30, b.render_font_size
        rb = b.render_box
        assert rb is not None
        gap = min(rb[0] - bb[0], rb[1] - bb[1], bb[2] - rb[2], bb[3] - rb[3])
        assert gap >= 0.08 * short, f"块 {b.id} 贴边：离气泡框只有 {gap}px"
        # 文字占气泡的合理比例（不是缩在中间一小撮）
        ink_h = (rb[3] - rb[1]) / (bb[3] - bb[1])
        assert ink_h >= 0.45, f"块 {b.id} 文字只占气泡高度的 {ink_h:.0%}"
    outside = changed.copy()
    for b in blocks:
        bb = b.bubble_box
        outside[bb[1]:bb[3], bb[0]:bb[2]] = False
    assert not outside.any()
