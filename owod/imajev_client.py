"""Build imajev /v1/systemone requests for crops and parse the answers. No network, no torch."""
import json
from dataclasses import dataclass
from typing import Dict, List, Optional

NONE_OPTION = "none of these"
MAX_OPTIONS = 255


@dataclass
class Verdict:
    unknown_score: float
    unknown_probability: float
    abstained: bool
    best: str
    probabilities: Dict[str, float]


def chunk_classes(known: List[str], size: int = MAX_OPTIONS - 1) -> List[List[str]]:
    if not known:
        raise ValueError("known classes must not be empty")
    return [known[i:i + size] for i in range(0, len(known), size)]


def build_request(known: List[str], question_name: str = "crop_class") -> str:
    """JSON string for the `request` form field. The crop is sent as the `image` file."""
    if not known:
        raise ValueError("known classes must not be empty")
    if len(known) + 1 > MAX_OPTIONS:
        raise ValueError("too many options; use chunk_classes first")
    if len(set(known)) != len(known) or NONE_OPTION in known:
        raise ValueError("class names must be unique and must not equal the none option")
    criteria = {name: None for name in known}
    criteria[NONE_OPTION] = "the crop shows none of the listed kinds of object"
    q = {"type": "choice",
         "instructions": "Which kind of object does this image show?",
         "criteria": criteria}
    return json.dumps({"state": {}, "questions": {question_name: q}})


def parse_response(resp: dict, known: List[str], question_name: str = "crop_class") -> Verdict:
    ans = resp["answers"][question_name]
    if ans.get("type") != "choice":
        raise ValueError("expected a choice answer")
    probs = ans["probabilities"]
    expected = set(known) | {NONE_OPTION}
    if set(probs) != expected:
        raise ValueError("answer options do not match the request")
    return Verdict(unknown_score=float(probs[NONE_OPTION]),
                   unknown_probability=float(ans.get("unknown_probability", 0.0)),
                   abstained=bool(ans.get("abstained", False)),
                   best=ans["choice"], probabilities={k: float(v) for k, v in probs.items()})
