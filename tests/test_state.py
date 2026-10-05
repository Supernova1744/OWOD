import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
from owod.state import OWState, UnknownStore
from owod.metrics import u_recall
from owod.crops import Box


def test_add_class_validation_and_log():  # T1 T2
    s = OWState()
    s.add_class("cat"); s.add_class("dog", note="x")
    assert s.known == ["cat", "dog"] and [e["n_known"] for e in s.log] == [1, 2]
    for bad in ("", "  ", "Cat", "none of these", "unknown"):
        with pytest.raises(ValueError):
            s.add_class(bad)
    assert s.known == ["cat", "dog"]


def test_state_roundtrip(tmp_path):  # T3
    s = OWState(); s.add_class("cat")
    s.save(tmp_path / "s.json")
    t = OWState.load(tmp_path / "s.json")
    assert t.known == ["cat"] and t.log == s.log


def test_store_add_get_ids_persist(tmp_path):  # T4 T5 T6
    PIL = pytest.importorskip("PIL")
    from PIL import Image
    st = UnknownStore(tmp_path / "u")
    a = st.add(Image.new("RGB", (10, 10), (1, 2, 3)), "img.jpg", (0, 0, 10, 10, 0.9), "not_in_known", 0.8)
    b = st.add(Image.new("RGB", (10, 10)), "img.jpg", (5, 5, 9, 9, 0.5), "multi_object", 1.0)
    assert (a, b) == (1, 2) and st.ids() == [1, 2] and st.ids(reasons=["multi_object"]) == [2]
    st.mark(a, "labelled", "bird")
    assert st.ids() == [2] and st.ids("labelled") == [1] and st.get(1)["label"] == "bird"
    assert st.crop(1).getpixel((0, 0)) == (1, 2, 3)
    st2 = UnknownStore(tmp_path / "u")                      # restart
    assert st2.ids(None) == [1, 2]
    c = st2.add(Image.new("RGB", (4, 4)), "i", (0, 0, 4, 4, 1), "not_in_known", 0.7)
    assert c == 3                                           # ids never repeat


def test_u_recall():  # T16
    gt = [Box(0, 0, 10, 10), Box(50, 50, 60, 60), Box(100, 100, 110, 110)]
    pred = [Box(0, 0, 10, 10), Box(51, 51, 60, 60)]
    assert u_recall(gt, pred) == pytest.approx(2 / 3)
    assert u_recall(gt, []) == 0.0
    with pytest.raises(ValueError):
        u_recall([], pred)
