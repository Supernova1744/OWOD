"""Step B (imajev venv): boxes JSON -> grouped, padded crops -> imajev verdicts + overlay images. UNTESTED on GPU.
  python scripts/classify_crops.py --boxes results/boxes.json --imajev-dir ~/imajev \
     --classes "person,car,dog,chair" --out results/crops [--save-crops] [--top 30]
imajev runs in this process as a Python package. No server."""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pathlib import Path
from PIL import Image, ImageDraw
from owod.crops import Box
from owod.crop_classifier import Config, CropClassifier
from owod.imajev_engine import ImajevEngine
from owod.pipeline import PipelineConfig, classify_detections


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--boxes", required=True); p.add_argument("--out", required=True)
    p.add_argument("--imajev-dir", default=str(Path.home() / "imajev"))
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--classes"); g.add_argument("--classes-file")
    p.add_argument("--rotations", type=int, default=1); p.add_argument("--fast", action="store_true")
    p.add_argument("--top", type=int, default=30, help="max crops per image")
    p.add_argument("--min-score", type=float, default=0.0)
    p.add_argument("--margin", type=float, default=0.1); p.add_argument("--pad", type=float, default=8)
    p.add_argument("--no-square", action="store_true"); p.add_argument("--save-crops", action="store_true")
    a = p.parse_args()
    known = [c.strip() for c in (a.classes.split(",") if a.classes else Path(a.classes_file).read_text().splitlines()) if c.strip()]
    engine = ImajevEngine(a.imajev_dir, rotations=a.rotations, fast=a.fast)
    print(f"imajev loaded in {engine.load_seconds:.1f}s; calibration={'yes' if engine.calibration else 'no'}")
    clf = CropClassifier(engine, known, Config())
    cfg = PipelineConfig(margin=a.margin, pad_px=a.pad, square=not a.no_square, min_score=a.min_score, max_crops=a.top)
    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    report = []
    for item in json.loads(Path(a.boxes).read_text()):
        img = Image.open(item["image"]).convert("RGB")
        boxes = [Box(*b[:4], b[4], i) for i, b in enumerate(item["boxes"])]
        stem = Path(item["image"]).stem
        def save(res, crop, stem=stem):
            if a.save_crops:
                crop.save(out_dir / f"{stem}_{res.index:03d}_{res.verdict.label}.png")
        results = classify_detections(img, boxes, clf, cfg, on_crop=save)
        draw = ImageDraw.Draw(img)
        for r in results:
            v, b = r.verdict, r.box
            color = (255, 60, 60) if v.is_unknown else (60, 220, 60)
            draw.rectangle([b.x1, b.y1, b.x2, b.y2], outline=color, width=3)
            draw.text((b.x1 + 3, b.y1 + 3), f"{v.label}" + (f" ({v.reason})" if v.reason else f" {v.class_probs.get(v.label, 0):.2f}"), fill=color)
        img.save(out_dir / f"{stem}_overlay.png")
        report.append({"image": item["image"], "crops": [{
            "index": r.index, "box": [r.box.x1, r.box.y1, r.box.x2, r.box.y2, r.box.score], "members": len(r.member_ids),
            "rect": list(r.rect.rect), "geometric_multi": r.geometric_multi, "label": r.verdict.label,
            "is_unknown": r.verdict.is_unknown, "reason": r.verdict.reason, "unknown_score": r.verdict.unknown_score,
            "scene": r.verdict.scene_probs, "top_classes": sorted(r.verdict.class_probs.items(), key=lambda kv: -kv[1])[:3]}
            for r in results]})
        print(stem, len(results), "crops;", sum(r.verdict.is_unknown for r in results), "unknown")
    (out_dir / "verdicts.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
