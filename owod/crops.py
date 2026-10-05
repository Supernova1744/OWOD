"""Box grouping and crop geometry. Pure Python (Pillow only in cut_crop). See docs/SPEC-003."""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import math


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float = 1.0
    id: Optional[object] = None

    @property
    def w(self):
        return self.x2 - self.x1

    @property
    def h(self):
        return self.y2 - self.y1

    @property
    def area(self):
        return self.w * self.h


def _check(b: Box):
    if not (b.w > 0 and b.h > 0):
        raise ValueError(f"box has no area: {b}")


def inter_area(a: Box, b: Box) -> float:
    w = min(a.x2, b.x2) - max(a.x1, b.x1)
    h = min(a.y2, b.y2) - max(a.y1, b.y1)
    return max(0.0, w) * max(0.0, h)


def iou(a: Box, b: Box) -> float:
    i = inter_area(a, b)
    u = a.area + b.area - i
    return i / u if u > 0 else 0.0


def inside_fraction(inner: Box, outer: Box) -> float:
    """Part of `inner`'s area that lies inside `outer`."""
    return inter_area(inner, outer) / inner.area


@dataclass
class Group:
    rep: Box
    members: List[Box] = field(default_factory=list)
    inner: List["Group"] = field(default_factory=list)


def group_boxes(boxes: List[Box], iou_thr: float = 0.6) -> List[Group]:
    """Greedy grouping: best score first; a box joins the first group whose representative has IoU >= iou_thr."""
    for b in boxes:
        _check(b)
    order = sorted(boxes, key=lambda b: (-b.score, b.x1, b.y1, b.x2, b.y2, str(b.id)))
    groups: List[Group] = []
    for b in order:
        for g in groups:
            if iou(g.rep, b) >= iou_thr:
                g.members.append(b)
                break
        else:
            groups.append(Group(rep=b, members=[b]))
    return groups


def link_inner(groups: List[Group], inner_thr: float = 0.8) -> List[Group]:
    """For each group, list the other groups whose representative is smaller and mostly inside it."""
    for g in groups:
        g.inner = [o for o in groups if o is not g and o.rep.area < g.rep.area
                   and inside_fraction(o.rep, g.rep) >= inner_thr]
    return groups


def geometric_multi(g: Group, dominance_thr: float = 0.5, min_inner: int = 2) -> bool:
    """Several inner objects and none of them fills dominance_thr of the group box. A hint, never a verdict."""
    if len(g.inner) < min_inner:
        return False
    biggest = max(o.rep.area for o in g.inner)
    return biggest / g.rep.area < dominance_thr


@dataclass(frozen=True)
class CropRect:
    rect: Tuple[int, int, int, int]   # crop area; may extend outside the image when square=True
    clip: Tuple[int, int, int, int]   # part of rect that lies inside the image


def _grow_to(lo, hi, side):
    c = (lo + hi) / 2.0
    return c - side / 2.0, c + side / 2.0


def crop_rect(box: Box, img_w: int, img_h: int, margin: float = 0.1, pad_px: float = 8,
              square: bool = False, min_side: int = 0) -> CropRect:
    _check(box)
    dx, dy = margin * box.w + pad_px, margin * box.h + pad_px
    x1, y1, x2, y2 = box.x1 - dx, box.y1 - dy, box.x2 + dx, box.y2 + dy
    if min_side:
        if x2 - x1 < min_side:
            x1, x2 = _grow_to(x1, x2, min_side)
        if y2 - y1 < min_side:
            y1, y2 = _grow_to(y1, y2, min_side)
    if square:
        s = max(x2 - x1, y2 - y1)
        x1, x2 = _grow_to(x1, x2, s)
        y1, y2 = _grow_to(y1, y2, s)
    rect = (math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2))
    clip = (max(0, rect[0]), max(0, rect[1]), min(img_w, rect[2]), min(img_h, rect[3]))
    if clip[2] <= clip[0] or clip[3] <= clip[1]:
        raise ValueError("crop lies outside the image")
    return CropRect(rect=clip if not square else rect, clip=clip)


def cut_crop(image, cr: CropRect, fill=(114, 114, 114)):
    """PIL crop. Square crops that leave the image are letterboxed with `fill`, not stretched."""
    from PIL import Image
    part = image.crop(cr.clip)
    if cr.rect == cr.clip:
        return part
    w, h = cr.rect[2] - cr.rect[0], cr.rect[3] - cr.rect[1]
    canvas = Image.new("RGB", (w, h), fill)
    canvas.paste(part, (cr.clip[0] - cr.rect[0], cr.clip[1] - cr.rect[1]))
    return canvas
