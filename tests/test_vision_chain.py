"""检测 → OCR → 去字 的真实模型回归测试（用 tests/fixtures/real 的 9 张非成人页面）。

锁住 3.0 的两个老问题：背景被搞坏（mask 外像素必须逐位不变、衣服 / 帽子 logo 必须跳过），
以及修过的 bug：字列贴着气泡描边时擦不干净（p016 右侧气泡）。有 GPU 用 GPU，没有用 CPU。
"""
from __future__ import annotations

import copy
import os

import numpy as np
import pytest

from hanhua import imageio
from hanhua.config import DEFAULTS
from hanhua.runtime import Runtime, resolve_profile

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "real")
PAGES = sorted(f[:-4] for f in os.listdir(FIX) if f.endswith(".jpg")) if os.path.isdir(FIX) else []
pytestmark = pytest.mark.skipif(not PAGES, reason="缺少 tests/fixtures/real 测试页")


@pytest.fixture(scope="module")
def chain():
    from hanhua.detect import TextDetector
    from hanhua.inpaint import Inpainter
    from hanhua.ocr import OCREngine

    rt = Runtime("auto")
    prof = resolve_profile("balanced", rt.device)
    return TextDetector(rt, prof), OCREngine(rt, prof, "ja"), Inpainter(rt, prof, copy.deepcopy(DEFAULTS))


def run(chain, name: str):
    det, ocr, inp = chain
    img = imageio.read(os.path.join(FIX, name + ".jpg"))
    blocks, bubbles = det.detect(img, "rtl")
    ocr.recognize(img, blocks)
    clean, mask = inp.clean(img, blocks)
    return img, blocks, bubbles, clean, mask


@pytest.mark.parametrize("name", PAGES)
def test_pixels_outside_mask_untouched(chain, name: str) -> None:
    img, blocks, _, clean, mask = run(chain, name)
    changed = (clean != img).any(axis=2)
    assert not (changed & (mask == 0)).any(), "文字区域以外的像素被改动了（背景被搞坏）"


@pytest.mark.parametrize("name", ["p012", "p014", "p016", "p019"])
def test_shirt_and_cap_logos_skipped(chain, name: str) -> None:
    _, blocks, _, _, _ = run(chain, name)
    for b in blocks:
        latin_only = b.src_text and not any("぀" <= c <= "ヿ" or "一" <= c <= "鿿" for c in b.src_text)
        if b.kind == "free" and latin_only:
            assert b.skip, f"衣服 / 帽子上的字母 logo 没被跳过：{b.src_text}"


def test_text_touching_outline_is_erased(chain) -> None:
    img, blocks, _, clean, _ = run(chain, "p016")
    b = next(b for b in blocks if b.src_text.startswith("何を言って"))
    x1, y1, x2, y2 = b.box
    dark_before = int((img[y1:y2, x1:x2].max(axis=2) < 110).sum())
    dark_after = int((clean[y1:y2, x1:x2].max(axis=2) < 110).sum())
    # 剩下的少量深色是文字框边缘压到的气泡描边（本来就该保留）；修复前这个气泡完全没擦（剩 ~100%）
    assert dark_before > 500 and dark_after < 0.10 * dark_before, f"字没擦干净：{dark_before} → {dark_after}"


def test_bubble_ocr(chain) -> None:
    _, blocks, _, _, _ = run(chain, "p016")
    texts = [b.src_text for b in blocks if b.kind == "bubble"]
    assert "何を言っているの！佐野君にも良いところはあるわ！" in texts
    assert "そう、例えばその鍛えられた肉体美とか！" in texts
    _, blocks, _, _, _ = run(chain, "p012")
    assert "そんなことはありません" in [b.src_text for b in blocks]
