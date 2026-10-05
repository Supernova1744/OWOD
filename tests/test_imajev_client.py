import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
from owod.imajev_client import build_request, parse_response, chunk_classes, NONE_OPTION


def test_build_request_shape_small():  # T1
    r = json.loads(build_request(["cat", "dog"]))
    q = r["questions"]["crop_class"]
    assert q["type"] == "choice" and list(q["criteria"]) == ["cat", "dog", NONE_OPTION]
    assert r["state"] == {}


def test_build_request_300_classes_must_be_chunked():  # T1
    names = [f"c{i}" for i in range(300)]
    with pytest.raises(ValueError):
        build_request(names)
    chunks = chunk_classes(names)
    assert [len(c) for c in chunks] == [254, 46]
    for c in chunks:
        assert len(json.loads(build_request(c))["questions"]["crop_class"]["criteria"]) <= 255


def test_build_request_rejects_bad_names():
    with pytest.raises(ValueError):
        build_request([])
    with pytest.raises(ValueError):
        build_request(["a", "a"])
    with pytest.raises(ValueError):
        build_request(["a", NONE_OPTION])


def _resp(probs, choice, unk=0.01, abst=False):
    return {"answers": {"crop_class": {"type": "choice", "choice": choice, "probabilities": probs,
                                       "unknown_probability": unk, "abstained": abst}}}


def test_parse_known_and_unknown():  # T2
    v = parse_response(_resp({"cat": 0.95, "dog": 0.01, NONE_OPTION: 0.04}, "cat"), ["cat", "dog"])
    assert v.unknown_score == pytest.approx(0.04) and v.best == "cat" and not v.abstained
    u = parse_response(_resp({"cat": 0.05, "dog": 0.05, NONE_OPTION: 0.9}, NONE_OPTION, 0.3, True), ["cat", "dog"])
    assert u.unknown_score == pytest.approx(0.9) and u.abstained and u.unknown_probability == pytest.approx(0.3)


def test_parse_rejects_mismatch():  # T3
    with pytest.raises(ValueError):
        parse_response(_resp({"cat": 0.5, NONE_OPTION: 0.5}, "cat"), ["cat", "dog"])
    with pytest.raises(ValueError):
        parse_response({"answers": {"crop_class": {"type": "noul", "noul": 0.1}}}, ["cat"])
