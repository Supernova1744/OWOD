"""Step A (pf-rpn venv, run inside the PF-RPN folder): images -> boxes JSON. UNTESTED.
  python ../scripts/detect_boxes_pfrpn.py --images DIR_OR_FILE --out ../results/boxes.json [--top 100]
Output: [{"image": path, "width": W, "height": H, "boxes": [[x1,y1,x2,y2,score], ...]}, ...] in original pixels."""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pathlib import Path
import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/pf-rpn/pf-rpn_coco-imagenet.py")
    p.add_argument("--ckpt", default="checkpoints/pf_rpn_swinb_5p_coco_imagenet.pth")
    p.add_argument("--images", required=True); p.add_argument("--out", required=True)
    p.add_argument("--top", type=int, default=100); p.add_argument("--fp16", action="store_true")
    a = p.parse_args()
    from mmengine.config import Config
    from mmengine.dataset import Compose, pseudo_collate
    from mmengine.registry import init_default_scope
    from mmdet.apis import init_detector
    from mmdet.apis.inference import get_test_pipeline_cfg
    from PIL import Image
    cfg = Config.fromfile(a.config)
    init_default_scope("mmdet")
    model = init_detector(cfg, a.ckpt, device="cuda").eval()
    pc = get_test_pipeline_cfg(cfg); pc[0].type = "mmdet.LoadImageFromNDArray"
    pipeline = Compose([t for t in pc if t["type"] != "LoadAnnotations"])
    root = Path(a.images)
    files = sorted(f for f in ([root] if root.is_file() else root.iterdir()) if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    out = []
    for f in files:
        img = np.array(Image.open(f).convert("RGB"))[:, :, ::-1].copy()   # mmdet expects BGR
        data = pseudo_collate([pipeline(dict(img=img, img_id=0, text="object", custom_entities=False))])
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=a.fp16):
            inst = model.test_step(data)[0].pred_instances
        order = inst.scores.argsort(descending=True)[:a.top]
        boxes = torch.cat([inst.bboxes[order], inst.scores[order, None]], 1).float().cpu().tolist()
        out.append({"image": str(f.resolve()), "width": img.shape[1], "height": img.shape[0], "boxes": boxes})
        print(f.name, len(boxes), "boxes")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh)


if __name__ == "__main__":
    main()
