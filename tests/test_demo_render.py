"""演示预览也必须原位替换对白，不能在人物上添加假白框。"""
from pathlib import Path

import cv2
import numpy as np
import pytest

from hanhua import imageio
from hanhua.types import TextBlock, box_iou
from hanhua.ui.demo import _Fonts, _SampleBank, find_bubbles, render_page


@pytest.mark.parametrize("name,count", [
    ("p005", 2), ("p008", 2), ("p011", 1), ("p012", 3),
    ("p014", 2), ("p016", 2), ("p019", 2),
])
def test_real_tinted_bubbles(real_pages, name, count):
    page = next(p for p in real_pages if p.stem == name)
    bubbles = find_bubbles(imageio.read(page))
    assert len(bubbles) == count
    assert all(box_iou(a["bubble_box"], b["bubble_box"]) < 0.5
               for i, a in enumerate(bubbles) for b in bubbles[i + 1:])


@pytest.mark.parametrize("direction", ["h", "v"])
def test_tinted_bubble_erased_and_rendered_inside(direction):
    src = np.full((1000, 900, 3), (130, 100, 150), np.uint8)
    shape = np.zeros(src.shape[:2], np.uint8)
    cv2.ellipse(shape, (250, 350), (100, 200), 0, 0, 360, 1, -1)
    inside = cv2.erode(shape, np.ones((7, 7), np.uint8)).astype(bool)
    src[shape > 0] = 0
    src[inside] = (192, 196, 210)
    # 两列假原文，位置已知，擦除后这些深色像素必须全部消失。
    for x in (225, 265):
        for y in range(240, 450, 35):
            cv2.rectangle(src, (x - 9, y), (x + 9, y + 22), (20, 20, 20), -1)
    found = find_bubbles(src)
    assert len(found) == 1
    block = TextBlock(id=0, **found[0], src_text="原文" * 6,
                      translation="等一下！你要去哪里？" * 6,
                      out_direction=direction, font_size=120)
    patch, out = render_page(src, [block], _Fonts())
    clean = np.where(patch[..., 3:] > 0, patch[..., :3], src)
    assert clean[230:470, 205:285].min() > 150
    assert not ((patch[..., 3] > 0) & ~inside).any()
    assert np.array_equal(src[~inside], out[~inside])
    assert (out != clean).any()
    assert block.render_box is not None


@pytest.mark.parametrize("name", ["p002", "p003"])
def test_no_bubble_keeps_page_without_fake_captions(real_pages, tmp_path, monkeypatch, name):
    page = next(p for p in real_pages if p.stem == name)
    monkeypatch.setenv("HANHUA_DEMO_PAGES", str(Path(page).parent))
    bank = _SampleBank(str(tmp_path))
    k = bank.sources.index(str(page))
    sample = bank.get(k)
    assert sample["blocks"] == []
    patch = cv2.imdecode(np.fromfile(sample["clean"], dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    assert not patch[..., 3].any()
