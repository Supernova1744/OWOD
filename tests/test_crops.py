import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
from owod.crops import Box, iou, inside_fraction, group_boxes, link_inner, geometric_multi, crop_rect


def B(x1, y1, x2, y2, s=0.9, i=None):
    return Box(x1, y1, x2, y2, s, i)


def test_iou_and_inside_fraction():
    assert iou(B(0, 0, 10, 10), B(0, 0, 10, 10)) == pytest.approx(1.0)
    assert iou(B(0, 0, 10, 10), B(20, 20, 30, 30)) == 0.0
    assert inside_fraction(B(2, 2, 4, 4), B(0, 0, 10, 10)) == pytest.approx(1.0)
    assert inside_fraction(B(0, 0, 10, 10), B(0, 0, 5, 10)) == pytest.approx(0.5)


def test_group_merges_duplicates_keeps_best_score():  # G1 G3
    g = group_boxes([B(0, 0, 10, 10, 0.5, "a"), B(1, 1, 10, 10, 0.9, "b"), B(50, 50, 60, 60, 0.7, "c")], iou_thr=0.6)
    assert [x.rep.id for x in g] == ["b", "c"]
    assert {m.id for m in g[0].members} == {"a", "b"}


def test_group_keeps_inner_box_separate():  # G2
    g = group_boxes([B(0, 0, 100, 100, 0.9, "big"), B(10, 10, 30, 30, 0.8, "small")], iou_thr=0.6)
    assert len(g) == 2


def test_group_deterministic_ties():  # G4
    boxes = [B(0, 0, 10, 10, 0.5, "x"), B(0, 0, 10, 10, 0.5, "y")]
    assert [g.rep.id for g in group_boxes(boxes)] == [g.rep.id for g in group_boxes(list(reversed(boxes)))]


def test_zero_area_box_rejected():  # C4
    with pytest.raises(ValueError):
        group_boxes([B(5, 5, 5, 9)])


def test_crop_rect_margin_pad_and_clip():  # C1
    r = crop_rect(B(100, 100, 200, 200), 1000, 1000, margin=0.1, pad_px=5, square="none")
    assert r.clip == (85, 85, 215, 215) and r.canvas == (130, 130) and not r.padded
    r = crop_rect(B(0, 0, 50, 50), 100, 100, margin=0.2, pad_px=0, square="none")
    assert r.clip == (0, 0, 60, 60)


def test_crop_rect_square_pad_adds_no_image_pixels():  # C2 pad
    r = crop_rect(B(0, 0, 100, 40), 200, 200, margin=0.0, pad_px=0, square="pad")
    assert r.clip == (0, 0, 100, 40) and r.canvas == (100, 100) and r.offset == (0, 30) and r.padded


def test_crop_rect_square_context_pulls_in_image():  # C2 context
    r = crop_rect(B(0, 0, 100, 40), 200, 200, margin=0.0, pad_px=0, square="context")
    assert r.canvas == (100, 100) and r.clip == (0, 0, 100, 70) and r.offset == (0, 30)


def test_crop_rect_bad_mode():
    with pytest.raises(ValueError):
        crop_rect(B(0, 0, 10, 10), 100, 100, square="round")


def test_crop_rect_min_side():  # C3
    r = crop_rect(B(100, 100, 110, 110), 1000, 1000, margin=0.0, pad_px=0, min_side=32, square="none")
    assert r.canvas[0] >= 32 and r.canvas[1] >= 32


def _cluster():
    big = B(0, 0, 100, 100, 0.9, "scene")
    inner = [B(5, 5, 30, 30, 0.8, "a"), B(40, 5, 65, 30, 0.8, "b"), B(5, 50, 30, 75, 0.8, "c")]
    return [big] + inner


def test_geometric_multi_cluster():  # M1
    g = link_inner(group_boxes(_cluster()))
    top = [x for x in g if x.rep.id == "scene"][0]
    assert geometric_multi(top) is True


def test_part_inside_person_not_multi():  # M1 part box
    g = link_inner(group_boxes([B(0, 0, 100, 200, 0.9, "person"), B(30, 5, 70, 45, 0.8, "face")]))
    top = [x for x in g if x.rep.id == "person"][0]
    assert geometric_multi(top) is False


def test_dominant_inner_not_multi():  # M1 dominance
    g = link_inner(group_boxes([B(0, 0, 100, 100, 0.9, "outer"), B(5, 5, 95, 95, 0.8, "main"), B(1, 1, 8, 8, 0.7, "tiny")], iou_thr=0.95))
    top = [x for x in g if x.rep.id == "outer"][0]
    assert geometric_multi(top) is False
