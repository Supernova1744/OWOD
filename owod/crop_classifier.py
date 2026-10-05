"""Decide what a crop shows with imajev. The scorer is anything with ask(image, payload) -> dict (see imajev_engine)."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol

ONE, SEVERAL, NO_CLEAR = "one_main_object", "several_objects_none_dominant", "no_clear_object"
NONE_OPTION = "none of these"
UNKNOWN = "unknown"
DEFAULT_MAX_OPTIONS = 254   # imajev adapters allow 254 (255-code readout) or 255 (256-code readout, imajev-4b)


class Scorer(Protocol):
    def ask(self, image, payload: dict) -> dict: ...


@dataclass
class Config:
    multi_hi: float = 0.6     # p(several) alone makes the crop multi
    multi_lo: float = 0.3     # p(several) needed when geometry also says multi
    none_hi: float = 0.6      # p(no clear object) that makes the crop unknown
    min_conf: float = 0.5     # p(top class) needed for a known verdict
    none_max: float = 0.5     # p("none of these") at or above this makes it unknown
    unknown_max: float = 0.4  # imajev's own "can't tell" mass at or above this makes the answer unreliable
    verify: bool = False      # ask "is this really a <label>?" for known verdicts (one more pass); below verify_min -> unknown
    verify_min: float = 0.5
    part_hi: float = 0.5      # p(part_of_larger_object) at or above this confirms a part crop


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


WHOLE, PART = "whole_object", "part_of_larger_object"


def part_payload() -> dict:
    return {"state": {}, "questions": {"part": {
        "type": "choice",
        "instructions": "Does this image show a complete object, or only one piece of a larger object?",
        "criteria": {
            WHOLE: "a complete object is visible, even if small or partly cut off at the edge",
            PART: "only a piece of a larger object, for example an eye, a mouth, a wheel, a window, a handle or a sign on a vehicle",
        }}}}


def verify_payload(label: str) -> dict:
    return {"state": {}, "questions": {"verify": {
        "type": "noul",
        "instructions": f"The main object in this image is a {label}.",
        "criteria": {"true": f"it really is a {label}", "false": f"it is something else, or only a detail of something else"}}}}


def chunk_names(known: List[str], max_options: int = DEFAULT_MAX_OPTIONS) -> List[List[str]]:
    size = max_options - 1                      # one slot is "none of these"
    return [known[i:i + size] for i in range(0, len(known), size)]


def class_payloads(known: List[str], max_options: int = DEFAULT_MAX_OPTIONS) -> List[dict]:
    out = []
    for i, names in enumerate(chunk_names(known, max_options)):
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
    return ({k: float(v) for k, v in probs.items()}, bool(ans.get("abstained", False)),
            float(ans.get("unknown_probability", 0.0)))


class CropClassifier:
    def __init__(self, scorer: Scorer, known: List[str], config: Optional[Config] = None,
                 max_options: int = DEFAULT_MAX_OPTIONS):
        if not known:
            raise ValueError("known classes must not be empty")
        if len(set(known)) != len(known) or NONE_OPTION in known:
            raise ValueError("class names must be unique and must not equal the none option")
        self.scorer, self.known, self.cfg = scorer, list(known), config or Config()
        self._class_payloads = class_payloads(self.known, max_options)

    def is_part(self, image):
        """-> (is_part, p_part). One extra pass; used only for crops that geometry flags as possible parts."""
        probs, ab, unk = _choice(self.scorer.ask(image, part_payload()), "part", {WHOLE, PART})
        if ab or unk >= self.cfg.unknown_max:
            return False, probs[PART]          # cannot tell: keep the crop
        return probs[PART] >= self.cfg.part_hi, probs[PART]

    def _verify(self, image, label):
        ans = self.scorer.ask(image, verify_payload(label))["answers"]["verify"]
        if ans.get("type") != "noul":
            raise ValueError("verify: expected a noul answer")
        return float(ans["noul"])

    def classify(self, image, geometric_multi: bool = False) -> Verdict:
        c = self.cfg
        scene, scene_abstained, scene_unk = _choice(self.scorer.ask(image, scene_payload()), "scene", {ONE, SEVERAL, NO_CLEAR})
        # Scene probabilities are used as returned (renormalized over the 3 options). If imajev's own "can't tell"
        # mass is high, the scene answer is treated like an abstention and only the geometry hint is used.
        scene_abstained = scene_abstained or scene_unk >= c.unknown_max
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
        unk_of: Dict[str, float] = {}
        p_none, abstained = 1.0, True   # abstained only if EVERY chunk abstains (a chunk without the true class may abstain)
        for i, payload in enumerate(self._class_payloads):
            name = f"classes_{i}"
            names = set(payload["questions"][name]["criteria"])
            got, ab, unk = _choice(self.scorer.ask(image, payload), name, names)
            p_none = min(p_none, got.pop(NONE_OPTION))
            probs.update(got)
            unk_of.update({k: unk for k in got})
            abstained = abstained and ab
        top = max(probs, key=probs.get)
        eff = probs[top] * (1.0 - unk_of[top])      # raw-scale confidence: the renormalized p hides imajev's own unknown mass
        if abstained:
            return Verdict(UNKNOWN, True, "abstained", p_none, probs, scene)
        if p_none >= c.none_max:
            return Verdict(UNKNOWN, True, "not_in_known", p_none, probs, scene)
        if eff < c.min_conf or unk_of[top] >= c.unknown_max:
            return Verdict(UNKNOWN, True, "low_confidence", p_none, probs, scene)
        if c.verify:
            p_yes = self._verify(image, top)
            if p_yes < c.verify_min:
                return Verdict(UNKNOWN, True, "failed_verification", p_none, probs, scene)
        return Verdict(top, False, None, p_none, probs, scene)
