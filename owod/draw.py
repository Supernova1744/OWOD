"""Draw system results on a PIL image: green = known, red = unknown, suppressed parts are skipped."""
from PIL import ImageDraw


def draw_results(image, results, legend: str = ""):
    img = image.copy()
    d = ImageDraw.Draw(img)
    w = max(2, round(min(img.size) / 250))
    for r in results:
        if r.part_of is not None:
            continue
        b, v = r.box, r.verdict
        color = (255, 60, 60) if v.is_unknown else (60, 220, 60)
        d.rectangle([b.x1, b.y1, b.x2, b.y2], outline=color, width=w)
        text = (f"unknown: {v.reason}" if v.is_unknown else f"{v.label} {v.class_probs.get(v.label, 0):.2f}")
        tw = d.textlength(text)
        d.rectangle([b.x1, max(0, b.y1 - 14), b.x1 + tw + 6, max(14, b.y1)], fill=color)
        d.text((b.x1 + 3, max(0, b.y1 - 13)), text, fill=(0, 0, 0))
    if legend:
        d.rectangle([0, 0, d.textlength(legend) + 10, 16], fill=(0, 0, 0))
        d.text((5, 2), legend, fill=(255, 255, 255))
    return img
