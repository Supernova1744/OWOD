import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
from owod.crop_classifier import (CropClassifier, Config, ONE, SEVERAL, NO_CLEAR, NONE_OPTION,
                                  scene_payload, class_payloads)


class FakeScorer:
    """Returns canned answers. scene: dict option->p. classes: list of dicts (one per chunk)."""
    def __init__(self, scene, classes, scene_abstain=False, class_abstain=False):
        self.scene, self.classes, self.calls = scene, list(classes), []
        self.scene_abstain, self.class_abstain = scene_abstain, class_abstain

    def ask(self, image, payload):
        self.calls.append(list(payload["questions"]))
        name = next(iter(payload["questions"]))
        if name == "scene":
            return {"answers": {"scene": {"type": "choice", "choice": max(self.scene, key=self.scene.get),
                                          "probabilities": self.scene, "unknown_probability": 0.0,
                                          "abstained": self.scene_abstain}}}
        probs = self.classes.pop(0)
        return {"answers": {name: {"type": "choice", "choice": max(probs, key=probs.get), "probabilities": probs,
                                   "unknown_probability": 0.0, "abstained": self.class_abstain}}}


S_ONE = {ONE: 0.9, SEVERAL: 0.05, NO_CLEAR: 0.05}
S_MANY = {ONE: 0.1, SEVERAL: 0.85, NO_CLEAR: 0.05}


def test_payloads_shape():
    p = scene_payload()
    assert list(p["questions"]) == ["scene"] and list(p["questions"]["scene"]["criteria"]) == [ONE, SEVERAL, NO_CLEAR]
    chunks = class_payloads([f"c{i}" for i in range(300)])
    assert len(chunks) == 2
    assert len(chunks[0]["questions"]["classes_0"]["criteria"]) == 255 and len(chunks[1]["questions"]["classes_1"]["criteria"]) == 47


def test_known_class():  # D2
    s = FakeScorer(S_ONE, [{"cat": 0.9, "dog": 0.05, NONE_OPTION: 0.05}])
    v = CropClassifier(s, ["cat", "dog"]).classify(object())
    assert (v.label, v.is_unknown, v.reason) == ("cat", False, None)
    assert v.unknown_score == pytest.approx(0.05)


def test_multi_object_is_unknown_and_skips_class_question():  # M3 M4
    s = FakeScorer(S_MANY, [])
    v = CropClassifier(s, ["cat"]).classify(object())
    assert v.is_unknown and v.reason == "multi_object" and v.label == "unknown" and v.unknown_score == 1.0
    assert s.calls == [["scene"]]


def test_geometry_hint_needs_some_model_support():  # M3
    mid = {ONE: 0.6, SEVERAL: 0.35, NO_CLEAR: 0.05}
    cls = [{"cat": 0.9, NONE_OPTION: 0.1}]
    assert CropClassifier(FakeScorer(mid, cls), ["cat"]).classify(object(), geometric_multi=True).reason == "multi_object"
    assert CropClassifier(FakeScorer(mid, cls), ["cat"]).classify(object(), geometric_multi=False).label == "cat"
    # geometry alone never decides:
    assert CropClassifier(FakeScorer(S_ONE, [{"cat": 0.9, NONE_OPTION: 0.1}]), ["cat"]).classify(object(), geometric_multi=True).label == "cat"


def test_no_clear_object():  # M4
    s = FakeScorer({ONE: 0.1, SEVERAL: 0.1, NO_CLEAR: 0.8}, [])
    v = CropClassifier(s, ["cat"]).classify(object())
    assert v.is_unknown and v.reason == "no_clear_object"


def test_not_in_known_and_low_conf_and_abstained():  # D2
    v = CropClassifier(FakeScorer(S_ONE, [{"cat": 0.1, NONE_OPTION: 0.9}]), ["cat"]).classify(object())
    assert v.is_unknown and v.reason == "not_in_known" and v.unknown_score == pytest.approx(0.9)
    v = CropClassifier(FakeScorer(S_ONE, [{"cat": 0.4, "dog": 0.35, NONE_OPTION: 0.25}]), ["cat", "dog"]).classify(object())
    assert v.is_unknown and v.reason == "low_confidence"
    v = CropClassifier(FakeScorer(S_ONE, [{"cat": 0.9, NONE_OPTION: 0.1}], class_abstain=True), ["cat"]).classify(object())
    assert v.is_unknown and v.reason == "abstained"


def test_scene_abstain_falls_back_to_geometry():
    s = FakeScorer(S_ONE, [{"cat": 0.9, NONE_OPTION: 0.1}], scene_abstain=True)
    assert CropClassifier(s, ["cat"]).classify(object(), geometric_multi=True).reason == "multi_object"
    s = FakeScorer(S_ONE, [{"cat": 0.9, NONE_OPTION: 0.1}], scene_abstain=True)
    assert CropClassifier(s, ["cat"]).classify(object(), geometric_multi=False).label == "cat"


def test_chunks_best_class_and_min_none():  # D1
    names = [f"c{i}" for i in range(300)]
    ch = class_payloads(names)
    first = {n: 0.0 for n in ch[0]["questions"]["classes_0"]["criteria"]}; first[NONE_OPTION] = 0.95; first["c0"] = 0.05
    second = {n: 0.0 for n in ch[1]["questions"]["classes_1"]["criteria"]}; second[NONE_OPTION] = 0.1; second["c299"] = 0.9
    v = CropClassifier(FakeScorer(S_ONE, [first, second]), names).classify(object())
    assert v.label == "c299" and not v.is_unknown and v.unknown_score == pytest.approx(0.1)


def test_empty_classes_rejected():
    with pytest.raises(ValueError):
        CropClassifier(FakeScorer(S_ONE, []), [])


def test_chunk_abstain_needs_every_chunk():
    names = [f"c{i}" for i in range(300)]
    ch = class_payloads(names)
    first = {n: 0.0 for n in ch[0]["questions"]["classes_0"]["criteria"]}; first[NONE_OPTION] = 0.9; first["c0"] = 0.1
    second = {n: 0.0 for n in ch[1]["questions"]["classes_1"]["criteria"]}; second[NONE_OPTION] = 0.1; second["c299"] = 0.9

    class OneChunkAbstains(FakeScorer):
        def ask(self, image, payload):
            r = super().ask(image, payload)
            name = next(iter(payload["questions"]))
            if name == "classes_0":
                r["answers"][name]["abstained"] = True
            return r
    v = CropClassifier(OneChunkAbstains(S_ONE, [first, second]), names).classify(object())
    assert v.label == "c299" and not v.is_unknown
