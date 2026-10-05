"""Decide what a crop shows with imajev. The scorer is anything with ask(image, payload) -> dict (see imajev_engine)."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol

ONE, SEVERAL, NO_CLEAR = "one_main_object", "several_objects_none_dominant", "no_clear_object"
NONE_OPTION = "none of these"
UNKNOWN = "unknown"
CHUNK = 254   # imajev readout: 254 options + its own unknown code


class Scorer(Protocol):
    def ask(self, image, payload: dict) -> dict: ...


@dataclass
class Config:
    multi_hi: float = 0.6     # p(several) alone makes the crop multi
    multi_lo: float = 0.3     # p(several) needed when geometry also says multi
    none_hi: float = 0.6      # p(no clear object) that makes the crop unknown
    min_conf: float = 0.5     # p(top class) needed for a known verdict
    none_max: float = 0.5     # p("none of these") at or above this makes it unknown


@dataclass
class Verdict:
    label: str
    is_unknown: bool
    reason: Optional[str]
    unknown_score: float
    class_probs: Dict[str, float] = field(default_factory=dict)
    scene_probs: Dict[str, float] = field(default_factory=dict)


def scene_payload() -> dict:
    return {"state": {}, "questions": {"scene": {
        "type": "choice",
        "instructions": "Look at this image crop. How many distinct objects does it show, and does one of them clearly fill most of the crop?",
        "criteria": {
            ONE: "one object clearly fills most of the crop; small background items or parts of that object do not count",
            SEVERAL: "two or more separate objects are visible and none of them clearly dominates the crop",
            NO_CLEAR: "no recognizable object, only background, texture or fragments",
        }}}}


def chunk_names(known: List[str]) -> List[List[str]]:
    return [known[i:i + CHUNK] for i in range(0, len(known), CHUNK)]


def class_payloads(known: List[str]) -> List[dict]:
    out = []
    for i, names in enumerate(chunk_names(known)):
        criteria = {n: None for n in names}
        criteria[NONE_OPTION] = "the main object is none of the listed kinds"
        out.append({"state": {}, "questions": {f"classes_{i}": {
            "type": "choice",
            "instructions": "What is the main object in this image?",
            "criteria": criteria}}})
    return out


def _choice(resp: dict, name: str, expected: set):
    ans = resp["answers"][name]
    if ans.get("type") != "choice":
        raise ValueError(f"{name}: expected a choice answer")
    probs = ans["probabilities"]
    if set(probs) != expected:
        raise ValueError(f"{name}: options in the answer do not match the request")
    return {k: float(v) for k, v in probs.items()}, bool(ans.get("abstained", False))


class CropClassifier:
    def __init__(self, scorer: Scorer, known: List[str], config: Optional[Config] = None):
        if not known:
            raise ValueError("known classes must not be empty")
        if len(set(known)) != len(known) or NONE_OPTION in known:
            raise ValueError("class names must be unique and must not equal the none option")
        self.scorer, self.known, self.cfg = scorer, list(known), config or Config()
        self._class_payloads = class_payloads(self.known)

    def classify(self, image, geometric_multi: bool = False) -> Verdict:
        c = self.cfg
        scene, scene_abstained = _choice(self.scorer.ask(image, scene_payload()), "scene", {ONE, SEVERAL, NO_CLEAR})
        p_many = scene[SEVERAL]
        if scene_abstained:
            multi = geometric_multi            # the model cannot tell; the geometry hint is all we have
        else:
            multi = p_many >= c.multi_hi or (geometric_multi and p_many >= c.multi_lo)
        if multi:
            return Verdict(UNKNOWN, True, "multi_object", 1.0, scene_probs=scene)
        if not scene_abstained and scene[NO_CLEAR] >= c.none_hi:
            return Verdict(UNKNOWN, True, "no_clear_object", 1.0, scene_probs=scene)

        probs: Dict[str, float] = {}
        p_none, abstained = 1.0, True   # abstained only if EVERY chunk abstains (a chunk without the true class may abstain)
        for i, payload in enumerate(self._class_payloads):
            name = f"classes_{i}"
            names = set(payload["questions"][name]["criteria"])
            got, ab = _choice(self.scorer.ask(image, payload), name, names)
            p_none = min(p_none, got.pop(NONE_OPTION))
            probs.update(got)
            abstained = abstained and ab
        top = max(probs, key=probs.get)
        if abstained:
            return Verdict(UNKNOWN, True, "abstained", p_none, probs, scene)
        if p_none >= c.none_max:
            return Verdict(UNKNOWN, True, "not_in_known", p_none, probs, scene)
        if probs[top] < c.min_conf:
            return Verdict(UNKNOWN, True, "low_confidence", p_none, probs, scene)
        return Verdict(top, False, None, p_none, probs, scene)
