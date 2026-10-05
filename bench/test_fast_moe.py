"""GPU check: a fast patch == original, and how much time it saves (--patch predict|moe).
For predict the result must be IDENTICAL (labels, names, boxes, scores). For moe boxes are matched as a SET (fp16 noise
swaps near-tied boxes), so rank-by-rank comparison is not used.
Run from PF-RPN folder (pf-rpn venv): python ../bench/test_fast_moe.py --config ... --ckpt ... --images ../test_images [--fp16]
Pass criteria printed at the end. Also reports the selected router levels per image."""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pathlib import Path
import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True); p.add_argument("--ckpt", required=True)
    p.add_argument("--images", required=True); p.add_argument("--fp16", action="store_true")
    p.add_argument("--patch", choices=("predict", "moe"), default="predict")
    p.add_argument("--out", default="../results/test_fast_moe.json"); p.add_argument("--runs", type=int, default=20)
    a = p.parse_args()
    from mmengine.config import Config
    from mmengine.dataset import Compose, pseudo_collate
    from mmengine.registry import init_default_scope
    from mmdet.apis import init_detector
    from mmdet.apis.inference import get_test_pipeline_cfg
    from PIL import Image
    from owod.pfrpn_fast import apply_fast_moe, apply_fast_predict
    cfg = Config.fromfile(a.config); init_default_scope("mmdet")
    model = init_detector(cfg, a.ckpt, device="cuda").eval()
    pc = get_test_pipeline_cfg(cfg); pc[0].type = "mmdet.LoadImageFromNDArray"
    pipe = Compose([t for t in pc if t["type"] != "LoadAnnotations"])
    root = Path(a.images)
    files = sorted(f for f in ([root] if root.is_file() else root.iterdir()) if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
    datas = [pseudo_collate([pipe(dict(img=np.array(Image.open(f).convert("RGB"))[:, :, ::-1].copy(), img_id=0,
                                       text="object", custom_entities=False))]) for f in files]

    def run(d):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=a.fp16):
            return model.test_step(d)[0].pred_instances

    def timed(d):
        for _ in range(3):
            run(d)
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(a.runs):
            run(d)
        torch.cuda.synchronize()
        return (time.perf_counter() - t) * 1000 / a.runs

    ref, ref_levels = [], []
    for d in datas:
        ref.append(run(d)); ref_levels.append(model.router._last_idx_selected.flatten().tolist())
    t_orig = [timed(d) for d in datas]
    (apply_fast_predict if a.patch == "predict" else apply_fast_moe)(model)
    new = [run(d) for d in datas]
    t_fast = [timed(d) for d in datas]
    rows, ok = [], True
    for f, r, n, to, tf, lv in zip(files, ref, new, t_orig, t_fast, ref_levels):
        if a.patch == "predict":
            same = bool(torch.equal(r.bboxes, n.bboxes) and torch.equal(r.scores, n.scores) and torch.equal(r.labels, n.labels)
                        and list(getattr(r, "label_names", [])) == list(getattr(n, "label_names", [])))
            good, extra = same, {"identical": same}
        else:
            k = min(len(r.scores), len(n.scores), 100)
            rb, nb = r.bboxes[:k], n.bboxes[:k]
            lt = torch.max(rb[:, None, :2], nb[None, :, :2]); rbm = torch.min(rb[:, None, 2:], nb[None, :, 2:])
            inter = (rbm - lt).clamp(min=0).prod(-1)
            area = lambda b: (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
            iou = inter / (area(rb)[:, None] + area(nb)[None, :] - inter)
            best = iou.max(dim=1).values
            sd = float((r.scores[:k].sort().values - n.scores[:k].sort().values).abs().max())
            good = float(best.min()) > (0.9 if a.fp16 else 0.99) and sd < (5e-2 if a.fp16 else 1e-3)
            extra = {"min_best_iou_top100": float(best.min()), "max_sorted_score_diff": sd}
        ok &= good
        rows.append({"image": f.name, "levels": lv, **extra, "orig_ms": round(to, 1), "fast_ms": round(tf, 1),
                     "saved_ms": round(to - tf, 1), "ok": good})
    res = {"patch": a.patch, "fp16": a.fp16, "rows": rows, "all_ok": bool(ok)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1)); print("EQUIVALENT" if ok else "MISMATCH")


if __name__ == "__main__":
    main()
