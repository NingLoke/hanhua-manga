"""列顺序必须贯穿正常排版、兜底排版和实际渲染。"""
import copy

import cv2
import numpy as np
import pytest

from hanhua.config import DEFAULTS, deep_merge, validate, SettingsStore
from hanhua.engine.job import layout_label
from hanhua.types import TextBlock
from hanhua.typeset.fonts import FontRegistry
from hanhua.typeset.layout import Fitter, LayoutParams, Region, _dist, build_tokens, forced_layout
from hanhua.typeset.render import Typesetter

TEXT = "天地玄黄宇宙洪荒日月盈昃辰宿列张寒来暑往秋收冬藏" * 2


@pytest.fixture(scope="module")
def fonts(project_root):
    return FontRegistry(str(project_root / "fonts"), [])


def assert_order(layout, order):
    columns = {}
    for glyph in layout.placed:
        columns.setdefault(glyph.x, []).append(glyph)
    assert len(columns) >= 2
    positions = list(columns)
    assert positions == sorted(positions, reverse=order == "rtl")
    for glyphs in columns.values():
        assert [g.y for g in glyphs] == sorted(g.y for g in glyphs)


@pytest.mark.parametrize("order", ["ltr", "rtl"])
@pytest.mark.parametrize("forced", [False, True])
def test_column_order_normal_and_fallback(fonts, order, forced):
    hard = np.ones((220, 180), bool)
    region = Region((0, 0, 180, 220), hard, _dist(hard), (90, 110), 3, "rect", (255, 255, 255))
    tokens = build_tokens(TEXT, "v", "noto-sans-sc-bold", fonts, "zh-Hans", 0)
    params = LayoutParams("v", column_order=order)
    layout = forced_layout(region, tokens, 40, params) if forced else Fitter(region, tokens, params).fit(20)
    assert layout is not None
    assert_order(layout, order)
    assert "".join(g.unit.text for g in layout.placed) == TEXT


@pytest.mark.parametrize("order", ["ltr", "rtl"])
def test_typesetter_passes_column_order(fonts, order):
    page = np.full((400, 360, 3), 255, np.uint8)
    cv2.rectangle(page, (25, 25), (335, 375), (0, 0, 0), 3)
    block = TextBlock(id=0, box=(70, 70, 290, 330), bubble_box=(25, 25, 336, 376),
                      direction="v", translation=TEXT, font_size=30)
    settings = copy.deepcopy(DEFAULTS)
    settings["typeset"].update(direction="v", column_order=order)
    settings["source"]["reading_order"] = order
    _, details = Typesetter(settings, fonts).render_detailed(page, [block])
    assert len(details) == 1
    assert_order(details[0].layout, order)
    assert not block.overflow


def test_horizontal_ignores_column_order(fonts):
    page = np.full((400, 360, 3), 255, np.uint8)
    settings = copy.deepcopy(DEFAULTS)
    settings["typeset"]["direction"] = "h"
    results = []
    for order in ("ltr", "rtl"):
        settings["typeset"]["column_order"] = order
        settings["source"]["reading_order"] = order
        block = TextBlock(id=0, box=(30, 30, 330, 370), direction="h", translation=TEXT, font_size=30)
        results.append(Typesetter(settings, fonts).render(page, [block]))
    assert np.array_equal(*results)


def test_settings_keeps_valid_column_order():
    assert DEFAULTS["source"]["reading_order"] == "ltr"
    assert DEFAULTS["typeset"]["column_order"] == "ltr"
    assert DEFAULTS["typeset"]["direction"] == "v"
    assert DEFAULTS["output"]["format"] == "folder"
    assert validate(deep_merge(DEFAULTS, {"typeset": {"column_order": "ltr"}}))["typeset"]["column_order"] == "ltr"
    assert validate(deep_merge(DEFAULTS, {"typeset": {"column_order": "bad"}}))["typeset"]["column_order"] == "ltr"


@pytest.mark.parametrize("order", ["ltr", "rtl"])
def test_visible_reading_order_setting_controls_real_columns(fonts, tmp_path, order):
    store = SettingsStore(str(tmp_path / 'settings.json'))
    saved = store.update({'source': {'reading_order': order}})
    assert saved['typeset']['column_order'] == order
    assert SettingsStore(store.path).get()['source']['reading_order'] == order
    # 复现旧版任务：阅读方向已保存，但另一个列设置仍是相反值。
    saved['typeset']['column_order'] = 'rtl' if order == 'ltr' else 'ltr'
    page = np.full((400, 360, 3), 255, np.uint8)
    block = TextBlock(id=0, box=(30, 30, 330, 370), direction='v', translation=TEXT, font_size=30)
    _, details = Typesetter(saved, fonts).render_detailed(page, [block])
    assert_order(details[0].layout, order)
    assert layout_label(saved) == ('竖排左到右' if order == 'ltr' else '竖排右到左')


def test_legacy_column_patch_updates_the_same_reading_direction(tmp_path):
    store = SettingsStore(str(tmp_path / 'settings.json'))
    result = store.update({'typeset': {'column_order': 'rtl'}})
    assert result['source']['reading_order'] == result['typeset']['column_order'] == 'rtl'
    result = store.update({'source': {'reading_order': 'ltr'}, 'typeset': {'column_order': 'rtl'}})
    assert result['source']['reading_order'] == result['typeset']['column_order'] == 'ltr'
