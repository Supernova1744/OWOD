"""Detector boxes -> groups -> padded crops -> imajev verdicts. See docs/SPEC-003."""
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

from .crops import Box, CropRect, Group, crop_rect, cut_crop, geometric_multi, group_boxes, inside_fraction, link_inner
from .crop_classifier import CropClassifier, Verdict


@dataclass
class PipelineConfig:
    iou_thr: float = 0.6
    inner_thr: float = 0.8
    dominance_thr: float = 0.5
    margin: float = 0.1
    pad_px: float = 8
    square: str = "pad"       # "pad" (gray bars), "context" (more image), "none"
    min_side: int = 48
    min_score: float = 0.0
    max_crops: Optional[int] = None
    suppress_parts: bool = True      # drop crops that are pieces of a larger crop with the same label
    part_inside_thr: float = 0.8     # part of the small box inside the large box
    part_area_ratio: float = 0.7     # small box area / large box area must be below this
    confirm_parts: bool = True       # ask the model "whole or part?" before dropping (needs classifier.is_part)
    noparent_part_thr: float = 0.75  # a part with NO detected parent needs p(part) >= this (stricter than part_hi)
    border_px: float = 2             # a box touching the image border may be a cut-off whole object, not a part
    check_all_parts: bool = False    # ask "whole or part?" for EVERY known crop, not only geometric candidates (+1 pass each).
                                     # Catches parts whose bigger object has another label (headlight labelled "car" on a bus).


@dataclass
class CropResult:
    index: int
    box: Box
    member_ids: List[object]
    rect: CropRect
    geometric_multi: bool
    verdict: Verdict
    part_of: Optional[int] = None     # index of the larger crop this one is a part of (then suppressed)
    p_part: Optional[float] = None


def clamp_box(b: Box, w: int, h: int) -> Optional[Box]:
    x1, y1, x2, y2 = max(0.0, b.x1), max(0.0, b.y1), min(float(w), b.x2), min(float(h), b.y2)
    if x2 - x1 <= 0 or y2 - y1 <= 0:
        return None
    return Box(x1, y1, x2, y2, b.score, b.id)


def classify_detections(image, boxes: Sequence[Box], classifier: CropClassifier,
                        cfg: Optional[PipelineConfig] = None,
                        on_crop: Optional[Callable[[CropResult, object], None]] = None) -> List[CropResult]:
    cfg = cfg or PipelineConfig()
    w, h = image.size
    clean = [c for c in (clamp_box(b, w, h) for b in boxes if b.score >= cfg.min_score) if c is not None]
    groups: List[Group] = link_inner(group_boxes(clean, cfg.iou_thr), cfg.inner_thr)
    if cfg.max_crops is not None:
        groups = groups[:cfg.max_crops]
    out: List[CropResult] = []
    crops: list = []
    for i, g in enumerate(groups):
        cr = crop_rect(g.rep, w, h, cfg.margin, cfg.pad_px, cfg.square, cfg.min_side)
        crop = cut_crop(image, cr)
        multi = geometric_multi(g, cfg.dominance_thr)
        res = CropResult(i, g.rep, [m.id for m in g.members], cr, multi, classifier.classify(crop, multi))
        out.append(res)
        crops.append(crop)
        if on_crop:
            on_crop(res, crop)
    if cfg.suppress_parts:
        _suppress_parts(out, crops, classifier, cfg, (w, h))
    return out


def _suppress_parts(results, crops, classifier, cfg, size):
    """A known crop that lies inside a larger crop with the SAME label is a part candidate (eye in a person,
    window in a bus). Geometry only proposes; the model confirms (is_part) when confirm_parts is on.
    With check_all_parts the model is asked for every known crop; a part with no detected parent gets part_of = -1."""
    ask = getattr(classifier, "is_part", None) if cfg.confirm_parts else None
    for r in results:
        if r.verdict.is_unknown:
            continue
        outers = [o for o in results if o is not r and o.part_of is None and not o.verdict.is_unknown
                  and o.verdict.label == r.verdict.label and o.box.area > 0
                  and r.box.area / o.box.area < cfg.part_area_ratio
                  and inside_fraction(r.box, o.box) >= cfg.part_inside_thr]
        if not outers and not (cfg.check_all_parts and ask is not None):
            continue
        if ask is not None:
            is_part, r.p_part = ask(crops[r.index])
            if not is_part:
                continue
        if outers:
            r.part_of = min(outers, key=lambda o: o.box.area).index
            continue
        # no detected parent: be stricter. A cut-off object at the image border looks like a "part" but is a whole.
        b = r.box
        at_border = b.x1 <= cfg.border_px or b.y1 <= cfg.border_px or b.x2 >= size[0] - cfg.border_px or b.y2 >= size[1] - cfg.border_px
        if r.p_part is not None and r.p_part >= cfg.noparent_part_thr and not at_border:
            r.part_of = -1
