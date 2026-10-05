"""Find the ~1000 .item()/sync calls per forward, and which feature levels the router picks. UNTESTED on GPU.
Run from the PF-RPN folder (pf-rpn venv, GPU idle):
  python ../bench/find_syncs.py --config configs/pf-rpn/pf-rpn_coco-imagenet.py \
     --ckpt checkpoints/pf_rpn_swinb_5p_coco_imagenet.pth --images ../test_images --out ../results/syncs.json"""
import argparse, collections, json, os, sys, warnings
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pathlib import Path
import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True); p.add_argument("--ckpt", required=True)
    p.add_argument("--images", required=True); p.add_argument("--out", required=True)
    a = p.parse_args()
    from mmengine.config import Config
    from mmengine.dataset import Compose, pseudo_collate
    from mmengine.registry import init_default_scope
    from mmdet.apis import init_detector
    from mmdet.apis.inference import get_test_pipeline_cfg
    from PIL import Image
    cfg = Config.fromfile(a.config); init_default_scope("mmdet")
    model = init_detector(cfg, a.ckpt, device="cuda").eval()
    pc = get_test_pipeline_cfg(cfg); pc[0].type = "mmdet.LoadImageFromNDArray"
    pipe = Compose([t for t in pc if t["type"] != "LoadAnnotations"])
    root = Path(a.images)
    files = sorted(f for f in ([root] if root.is_file() else root.iterdir()) if f.suffix.lower() in {".jpg", ".jpeg", ".png"})

    def run(path):
        img = np.array(Image.open(path).convert("RGB"))[:, :, ::-1].copy()
        data = pseudo_collate([pipe(dict(img=img, img_id=0, text="object", custom_entities=False))])
        with torch.no_grad():
            model.test_step(data)

    run(files[0])                       # warm-up
    levels, counts = {}, collections.Counter()
    def show(message, category, filename, lineno, file=None, line=None):
        if "synchroniz" in str(message).lower():        # skip unrelated warnings
            counts[f"{filename}:{lineno}"] += 1
    old = warnings.showwarning; warnings.showwarning = show; warnings.simplefilter("always")
    for i, f in enumerate(files):
        if i == 0:
            torch.cuda.set_sync_debug_mode("warn")
        run(f)
        if i == 0:
            torch.cuda.set_sync_debug_mode("default")
        levels[f.name] = model.router._last_idx_selected.flatten().tolist()
    warnings.showwarning = old
    top = counts.most_common(25)
    res = {"syncs_first_image_top_locations": top, "total_syncs_first_image": sum(counts.values()),
           "router_selected_levels_per_image (0 = largest map, 3 = smallest)": levels}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
