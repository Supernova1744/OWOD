"""GPU check: fast pseudo-text builder == original, and how much time it saves. UNTESTED on GPU.
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
    p.add_argument("--out", default="../results/test_fast_moe.json"); p.add_argument("--runs", type=int, default=20)
    a = p.parse_args()
    from mmengine.config import Config
    from mmengine.dataset import Compose, pseudo_collate
    from mmengine.registry import init_default_scope
    from mmdet.apis import init_detector
    from mmdet.apis.inference import get_test_pipeline_cfg
    from PIL import Image
    from owod.pfrpn_fast import apply_fast_moe
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
    apply_fast_moe(model)
    new = [run(d) for d in datas]
    t_fast = [timed(d) for d in datas]
    rows, ok = [], True
    for f, r, n, to, tf, lv in zip(files, ref, new, t_orig, t_fast, ref_levels):
        k = min(len(r.scores), len(n.scores), 100)
        ds = float((r.scores[:k] - n.scores[:k]).abs().max()); db = float((r.bboxes[:k] - n.bboxes[:k]).abs().max())
        tol = 5e-2 if a.fp16 else 1e-3
        good = ds < tol and db < (5.0 if a.fp16 else 0.5)
        ok &= good
        rows.append({"image": f.name, "levels": lv, "max_score_diff_top100": ds, "max_box_diff_px_top100": db,
                     "orig_ms": round(to, 1), "fast_ms": round(tf, 1), "saved_ms": round(to - tf, 1), "ok": good})
    res = {"fp16": a.fp16, "rows": rows, "all_ok": bool(ok)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1)); print("EQUIVALENT" if ok else "MISMATCH")


if __name__ == "__main__":
    main()
