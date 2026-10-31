from types import SimpleNamespace

import numpy as np

from hanhua.detect import TextDetector, _Det, _Tile


def test_high_resolution_page_recovers_text_without_duplicate_or_cut_fragments():
    detector = TextDetector(None, SimpleNamespace(name='balanced', detector='detector', detector_size=640))
    calls = []

    def infer(img, tiles, axis):
        calls.append(tiles)
        first = _Det((250, 250, 350, 500), .95, 1)
        if len(tiles) == 1:
            return [first]
        return [first, _Det((600, 300, 730, 610), .92, 1),
                _Det((601, 301, 731, 611), .83, 1),
                _Det((1400, 200, 1535, 600), .99, 1)]

    detector._infer_tiles = infer
    blocks, _ = detector.detect(np.full((3615, 2560, 3), 255, np.uint8))
    assert len(calls) == 2 and len(calls[1]) > 1
    assert len(blocks) == 2
    assert {b.box for b in blocks} == {(250, 250, 350, 500), (600, 300, 730, 610)}


def test_detail_windows_cover_page_and_leave_existing_low_resolution_and_scroll_paths():
    assert TextDetector.make_detail_tiles(1140, 1600) == []
    assert TextDetector.make_detail_tiles(1000, 16000) == []
    tiles = TextDetector.make_detail_tiles(2560, 3615)
    mask = np.zeros((3615, 2560), bool)
    for tile in tiles:
        assert 0 <= tile.x0 < tile.x1 <= 2560 and 0 <= tile.y0 < tile.y1 <= 3615
        mask[tile.y0:tile.y1, tile.x0:tile.x1] = True
    assert mask.all()


def test_page_edge_text_is_kept_but_internal_cut_text_is_not():
    tile = _Tile(0, 0, 1536, 1536)
    assert TextDetector._inside_detail_tile(_Det((0, 0, 200, 300), .9, 1), tile, 2560, 3615)
    assert not TextDetector._inside_detail_tile(_Det((1200, 1300, 1536, 1500), .9, 1), tile, 2560, 3615)
