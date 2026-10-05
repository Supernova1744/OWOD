import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
pytest.importorskip("PIL")
from PIL import Image, ImageDraw
from owod.crops import Box
from owod.crop_classifier import ONE, SEVERAL, NO_CLEAR, WHOLE, PART, NONE_OPTION
from owod.detector_proc import SubprocessDetector
from owod.pipeline import PipelineConfig
from owod.state import OWState, UnknownStore
from owod.system import OpenWorldSystem

RED, BLUE, CYAN, WHITE, NEAR_RED = (255, 0, 0), (0, 0, 255), (0, 255, 255), (255, 255, 255), (250, 0, 0)


class ColorScorer:
    """A fake imajev that decides by the colour of the crop's top-left pixel."""
    def __init__(self):
        self.calls = []

    def ask(self, image, payload):
        name = next(iter(payload["questions"]))
        px = image.getpixel((0, 0))
        self.calls.append(name)
        if name == "scene":
            p = {ONE: 0.05, SEVERAL: 0.9, NO_CLEAR: 0.05} if px == WHITE else {ONE: 0.9, SEVERAL: 0.05, NO_CLEAR: 0.05}
            return self._choice(p)
        if name == "part":
            return self._choice({WHOLE: 0.1, PART: 0.9} if px == NEAR_RED else {WHOLE: 0.9, PART: 0.1}, "part")
        crit = payload["questions"][name]["criteria"]
        want = "cat" if px in (RED, NEAR_RED) else ("bird" if px in (BLUE, CYAN) else None)
        probs = {k: 0.0 for k in crit}
        probs[want if want in crit else NONE_OPTION] = 1.0
        return self._choice(probs, name)

    @staticmethod
    def _choice(p, name="scene"):
        return {"answers": {name: {"type": "choice", "choice": max(p, key=p.get), "probabilities": p,
                                   "unknown_probability": 0.0, "abstained": False}}}


def make_image(path):
    img = Image.new("RGB", (400, 300), (30, 30, 30))
    d = ImageDraw.Draw(img)
    d.rectangle([10, 10, 109, 109], fill=RED)
    d.rectangle([20, 20, 39, 39], fill=NEAR_RED)          # a part of the red object
    d.rectangle([150, 10, 249, 109], fill=BLUE)
    d.rectangle([280, 10, 379, 109], fill=CYAN)
    d.rectangle([10, 150, 209, 289], fill=WHITE)          # clutter: several objects, no main one
    img.save(path)
    return str(path)


BOXES = [Box(10, 10, 110, 110, 0.9, "red"), Box(20, 20, 40, 40, 0.8, "part"), Box(150, 10, 250, 110, 0.85, "blue"),
         Box(280, 10, 380, 110, 0.8, "cyan"), Box(10, 150, 210, 290, 0.7, "white")]
CFG = PipelineConfig(margin=0, pad_px=0, square="none", min_side=0)


def build(tmp_path, detect_calls):
    path = make_image(tmp_path / "a.png")

    def detect(paths):
        detect_calls.append(list(paths))
        return {str(os.path.abspath(p)): BOXES for p in paths}
    st = OWState(); st.add_class("cat")
    return OpenWorldSystem(detect, ColorScorer(), st, UnknownStore(tmp_path / "store"), pipeline_config=CFG), path


def test_process_returns_known_stores_unknown_not_parts(tmp_path):  # T7 T8 T9
    calls = []
    sysm, path = build(tmp_path, calls)
    res = sysm.process_image(path)
    by = {r.box.id: r for r in res}
    assert by["red"].verdict.label == "cat" and not by["red"].verdict.is_unknown
    assert by["part"].part_of == by["red"].index                       # suppressed part
    assert by["blue"].verdict.reason == "not_in_known" and by["white"].verdict.reason == "multi_object"
    reasons = sorted(r["reason"] for r in sysm.store.records.values())
    assert reasons == ["multi_object", "not_in_known", "not_in_known"]  # blue, cyan, white; the part is not stored


def test_one_detector_call_for_many_images(tmp_path):  # T10
    calls = []
    sysm, path = build(tmp_path, calls)
    p2 = make_image(tmp_path / "b.png")
    sysm.process_many([path, p2])
    assert len(calls) == 1 and len(calls[0]) == 2


def test_learn_labels_ids_and_recovers_similar(tmp_path):  # T11 T12 T13 T14
    sysm, path = build(tmp_path, [])
    sysm.process_image(path)
    blue = next(r["id"] for r in sysm.store.records.values() if r["box"][0] == 150)
    cyan = next(r["id"] for r in sysm.store.records.values() if r["box"][0] == 280)
    white = next(r["id"] for r in sysm.store.records.values() if r["reason"] == "multi_object")
    out = sysm.learn("bird", ids=[blue])
    assert sysm.state.known == ["cat", "bird"]
    assert sysm.store.get(blue)["status"] == "labelled" and sysm.store.get(blue)["label"] == "bird"
    assert out["recovered"] == 1 and sysm.store.get(cyan)["status"] == "recovered" and sysm.store.get(cyan)["label"] == "bird"
    assert sysm.store.get(white)["status"] == "unknown"                # multi_object is never re-labelled


def test_learn_rejects_bad_input(tmp_path):
    sysm, path = build(tmp_path, [])
    sysm.process_image(path)
    with pytest.raises(ValueError):
        sysm.learn("cat")                                              # duplicate
    rid = sysm.store.ids()[0]
    sysm.learn("bird", ids=[rid])
    with pytest.raises(ValueError):
        sysm.learn("fish", ids=[rid])                                  # already labelled
    assert "fish" not in sysm.state.known                              # failed learn changed nothing


def test_subprocess_detector_with_fake_script(tmp_path):  # T15
    script = tmp_path / "fake.py"
    script.write_text(
        "import argparse, json\n"
        "p = argparse.ArgumentParser()\n"
        "for n in ('--list', '--out', '--top'): p.add_argument(n)\n"
        "p.add_argument('--scale', nargs=2); p.add_argument('--fp16', action='store_true'); p.add_argument('--fast-predict', action='store_true')\n"
        "a = p.parse_args()\n"
        "paths = [l for l in open(a.list).read().split('\\n') if l]\n"
        "json.dump([{'image': q, 'width': 1, 'height': 1, 'boxes': [[0, 0, 5, 5, 0.9]]} for q in paths], open(a.out, 'w'))\n")
    img = make_image(tmp_path / "a.png")
    det = SubprocessDetector(sys.executable, str(tmp_path), str(tmp_path), script=str(script))
    out = det([img])
    assert det.calls == 1 and list(out.values())[0][0].x2 == 5
    bad = tmp_path / "bad.py"; bad.write_text("import sys; sys.exit(3)")
    with pytest.raises(RuntimeError):
        SubprocessDetector(sys.executable, str(tmp_path), str(tmp_path), script=str(bad))([img])


def test_state_is_saved_before_records_change_even_if_relabel_crashes(tmp_path):  # advisor 3
    calls = []
    sysm, path = build(tmp_path, calls)
    sysm.state_path = str(tmp_path / "state.json")
    sysm.process_image(path)
    blue = next(r["id"] for r in sysm.store.records.values() if r["box"][0] == 150)

    class Crash(Exception):
        pass
    orig = sysm.classifier.classify
    def boom(*a, **k):
        raise Crash()
    sysm._rebuild_orig = sysm._rebuild
    def rebuild_then_break():
        sysm._rebuild_orig()
        sysm.classifier.classify = boom
    sysm._rebuild = rebuild_then_break
    with pytest.raises(Crash):
        sysm.learn("bird", ids=[blue])
    assert OWState.load(tmp_path / "state.json").known == ["cat", "bird"]      # saved before the crash
    assert sysm.store.get(blue)["status"] == "labelled"                         # consistent with the state


def test_recovered_split_new_vs_old_and_no_duplicates(tmp_path):  # advisor 4 6
    sysm, path = build(tmp_path, [])
    sysm.process_image(path)
    n = len(sysm.store.records)
    sysm.process_image(path)                                                    # same image again
    assert len(sysm.store.records) == n
    blue = next(r["id"] for r in sysm.store.records.values() if r["box"][0] == 150)
    out = sysm.learn("bird", ids=[blue])
    assert out["recovered_new_class"] == 1 and out["recovered_old_class"] == 0


def test_same_pixels_from_another_path_is_not_duplicated(tmp_path):  # K bug found on the machine
    import shutil
    sysm, path = build(tmp_path, [])
    sysm.process_image(path)
    n = len(sysm.store.records)
    copy = str(tmp_path / "copy_of_a.png")
    shutil.copy(path, copy)
    sysm.process_image(copy)
    assert len(sysm.store.records) == n


class DriftScorer(ColorScorer):
    """After 'bird' exists, the blue crop is called 'cat' (an old class)."""
    def ask(self, image, payload):
        name = next(iter(payload["questions"]))
        if name.startswith("classes_") and image.getpixel((0, 0)) == BLUE and "bird" in payload["questions"][name]["criteria"]:
            crit = payload["questions"][name]["criteria"]
            return self._choice({k: (1.0 if k == "cat" else 0.0) for k in crit}, name)
        return super().ask(image, payload)


def test_learn_ignores_old_class_drift_unless_asked(tmp_path):
    for accept, expect_old in ((False, 0), (True, 1)):
        sub = tmp_path / str(accept); sub.mkdir()
        sysm, path = build(sub, [])
        sysm.scorer = DriftScorer(); sysm._rebuild()
        sysm.process_image(path)
        cyan = next(r["id"] for r in sysm.store.records.values() if r["box"][0] == 280)
        blue = next(r["id"] for r in sysm.store.records.values() if r["box"][0] == 150)
        out = sysm.learn("bird", ids=[cyan], accept_old_class=accept)
        assert out["recovered_old_class"] == expect_old
        assert sysm.store.get(blue)["status"] == ("recovered" if accept else "unknown")
