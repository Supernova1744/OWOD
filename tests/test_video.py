import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
from owod.video import sample_indices, hold_map


def test_sample_indices():
    assert sample_indices(300, 30, 1.0) == list(range(0, 300, 30))
    assert sample_indices(300, 30, 1.0, max_seconds=3) == [0, 30, 60]
    assert sample_indices(10, 30, 0.01) == list(range(10))           # step never below 1 frame
    with pytest.raises(ValueError):
        sample_indices(0, 30, 1.0)


def test_hold_map_holds_until_next_sample():
    m = hold_map(10, [0, 4, 8])
    assert m == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2]
    assert hold_map(5, [2]) == [None, None, 0, 0, 0]


def test_draw_results_skips_parts_and_marks_unknown():
    PIL = pytest.importorskip("PIL")
    from PIL import Image
    from owod.draw import draw_results
    from owod.crops import Box
    from owod.crop_classifier import Verdict
    from owod.pipeline import CropResult

    def res(box, label, unknown, part_of=None):
        return CropResult(0, box, [], None, False, Verdict(label, unknown, "not_in_known" if unknown else None, 0.5, {label: 0.9}), part_of)
    img = Image.new("RGB", (200, 100), (0, 0, 0))
    out = draw_results(img, [res(Box(10, 30, 60, 90), "cat", False), res(Box(100, 30, 150, 90), "unknown", True),
                             res(Box(20, 40, 30, 50), "cat", False, part_of=0)])
    assert out.getpixel((10, 60)) == (60, 220, 60)          # known: green outline
    assert out.getpixel((100, 60)) == (255, 60, 60)         # unknown: red outline
    assert out.getpixel((20, 45)) == (0, 0, 0)              # suppressed part: nothing drawn there
    assert img.getpixel((10, 60)) == (0, 0, 0)              # the input image is not changed
