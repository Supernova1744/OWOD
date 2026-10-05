"""Detector boxes -> groups -> padded crops -> imajev verdicts. See docs/SPEC-003."""
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

from .crops import Box, CropRect, Group, crop_rect, cut_crop, geometric_multi, group_boxes, link_inner
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


@dataclass
class CropResult:
    index: int
    box: Box
    member_ids: List[object]
    rect: CropRect
    geometric_multi: bool
    verdict: Verdict


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
    for i, g in enumerate(groups):
        cr = crop_rect(g.rep, w, h, cfg.margin, cfg.pad_px, cfg.square, cfg.min_side)
        crop = cut_crop(image, cr)
        multi = geometric_multi(g, cfg.dominance_thr)
        res = CropResult(i, g.rep, [m.id for m in g.members], cr, multi, classifier.classify(crop, multi))
        out.append(res)
        if on_crop:
            on_crop(res, crop)
    return out
