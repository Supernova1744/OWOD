import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
PIL = pytest.importorskip("PIL")
from PIL import Image
from owod.crops import Box, crop_rect, cut_crop
from owod.crop_classifier import Verdict
from owod.pipeline import classify_detections, PipelineConfig, clamp_box


class FakeClassifier:
    def __init__(self):
        self.seen = []

    def classify(self, image, geometric_multi=False):
        self.seen.append((image.size, geometric_multi))
        return Verdict("thing", False, None, 0.1)


def test_cut_crop_pad_size_and_fill():  # C2
    img = Image.new("RGB", (100, 40), (255, 0, 0))
    cr = crop_rect(Box(0, 0, 100, 40), 100, 40, margin=0, pad_px=0, square="pad")
    out = cut_crop(img, cr, fill=(1, 2, 3))
    assert out.size == (100, 100)
    assert out.getpixel((50, 50)) == (255, 0, 0) and out.getpixel((50, 2)) == (1, 2, 3)


def test_tall_box_in_busy_image_gets_gray_bars_not_neighbors():  # advisor: neighbors must stay out
    img = Image.new("RGB", (400, 300), (255, 0, 0))                    # red = neighbors everywhere
    for y in range(50, 250):
        for x in range(180, 220):
            img.putpixel((x, y), (0, 255, 0))                            # green = the tall object
    cr = crop_rect(Box(180, 50, 220, 250), 400, 300, margin=0, pad_px=0, square="pad")
    out = cut_crop(img, cr, fill=(1, 2, 3))
    assert out.size == (200, 200)
    colors = {out.getpixel((x, y)) for x in (2, 197) for y in (2, 100, 197)}
    assert colors == {(1, 2, 3)}                                         # left and right bars are gray, no red
    assert out.getpixel((100, 100)) == (0, 255, 0)


def test_clamp_box():
    assert clamp_box(Box(-5, -5, 20, 20), 10, 10) == Box(0, 0, 10, 10)
    assert clamp_box(Box(50, 50, 60, 60), 10, 10) is None


def test_pipeline_groups_crops_and_ids():  # T17
    img = Image.new("RGB", (400, 300), (10, 10, 10))
    boxes = [Box(10, 10, 110, 110, 0.9, "a"), Box(12, 12, 110, 110, 0.5, "a2"), Box(200, 100, 300, 200, 0.8, "b")]
    clf = FakeClassifier()
    res = classify_detections(img, boxes, clf, PipelineConfig(square="none", margin=0.1, pad_px=0, min_side=0))
    assert [r.box.id for r in res] == ["a", "b"]
    assert res[0].member_ids == ["a", "a2"]
    assert clf.seen[0][0] == (120, 120)         # 100 + 2 * 10% margin
    assert all(g is False for _, g in clf.seen)


def test_pipeline_cluster_sets_geometry_hint():
    img = Image.new("RGB", (200, 200))
    boxes = [Box(0, 0, 100, 100, 0.9, "scene"), Box(5, 5, 30, 30, 0.8, "a"), Box(40, 5, 65, 30, 0.8, "b"),
             Box(5, 50, 30, 75, 0.8, "c")]
    clf = FakeClassifier()
    res = classify_detections(img, boxes, clf, PipelineConfig(max_crops=1))
    assert len(res) == 1 and res[0].geometric_multi is True and clf.seen[0][1] is True


def test_pipeline_min_score_and_max_crops():
    img = Image.new("RGB", (100, 100))
    boxes = [Box(0, 0, 20, 20, 0.9, 1), Box(30, 30, 50, 50, 0.2, 2), Box(60, 60, 90, 90, 0.8, 3)]
    res = classify_detections(img, boxes, FakeClassifier(), PipelineConfig(min_score=0.5))
    assert [r.box.id for r in res] == [1, 3]


class LabelClassifier:
    """Gives each crop a label by its crop size so the test can control which crops are 'person'."""
    def __init__(self, labels, part_answer=True):
        self.labels, self.part_answer, self.asked = labels, part_answer, 0
        self.n = 0

    def classify(self, image, geometric_multi=False):
        label = self.labels[self.n]; self.n += 1
        return Verdict(label, label == "unknown", None if label != "unknown" else "multi_object", 0.1)

    def is_part(self, image):
        self.asked += 1
        return self.part_answer, 0.9 if self.part_answer else 0.1


def test_same_label_inner_crop_is_suppressed_as_part():
    img = Image.new("RGB", (300, 300))
    boxes = [Box(0, 0, 200, 200, 0.9, "whole"), Box(60, 60, 100, 100, 0.8, "eye"), Box(220, 220, 280, 280, 0.7, "other")]
    clf = LabelClassifier(["person", "person", "person"])
    res = classify_detections(img, boxes, clf, PipelineConfig(square="none"))
    assert [r.part_of for r in res] == [None, 0, None]
    assert clf.asked == 1


def test_model_can_veto_part_suppression():
    img = Image.new("RGB", (300, 300))
    boxes = [Box(0, 0, 200, 200, 0.9, "adult"), Box(60, 60, 100, 100, 0.8, "child")]
    res = classify_detections(img, boxes, LabelClassifier(["person", "person"], part_answer=False), PipelineConfig(square="none"))
    assert [r.part_of for r in res] == [None, None]


def test_different_label_or_unknown_not_suppressed():
    img = Image.new("RGB", (300, 300))
    boxes = [Box(0, 0, 200, 200, 0.9, "a"), Box(60, 60, 100, 100, 0.8, "b"), Box(120, 120, 160, 160, 0.7, "c")]
    res = classify_detections(img, boxes, LabelClassifier(["person", "dog", "unknown"]), PipelineConfig(square="none"))
    assert [r.part_of for r in res] == [None, None, None]


def test_suppress_off():
    img = Image.new("RGB", (300, 300))
    boxes = [Box(0, 0, 200, 200, 0.9, "a"), Box(60, 60, 100, 100, 0.8, "b")]
    res = classify_detections(img, boxes, LabelClassifier(["person", "person"]), PipelineConfig(square="none", suppress_parts=False))
    assert [r.part_of for r in res] == [None, None]
