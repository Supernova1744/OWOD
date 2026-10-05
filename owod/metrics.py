"""Open-world metrics that need no known-class ground truth."""
from typing import Sequence

from .crops import Box, iou


def u_recall(gt_unknown: Sequence[Box], pred_unknown: Sequence[Box], iou_thr: float = 0.5) -> float:
    """Share of ground-truth unknown objects covered by at least one predicted unknown box (IoU >= iou_thr)."""
    if not gt_unknown:
        raise ValueError("no ground-truth unknown objects")
    hit = sum(any(iou(g, p) >= iou_thr for p in pred_unknown) for g in gt_unknown)
    return hit / len(gt_unknown)
