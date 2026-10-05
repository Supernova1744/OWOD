"""Class-agnostic average recall (AR@100/300/900) of PF-RPN on a COCO-format image set. UNTESTED on GPU.
Use it to check that a speed setting does not cost recall: run the full model once as the reference, then each
speed setting, and compare (SPEC-001 R5: candidate AR100 >= 0.98 x reference AR100 on the SAME images).
COCO val is in-domain for PF-RPN, so absolute AR is higher than on CD-FSOD/ODinW; use it for RELATIVE comparison.

Run from the PF-RPN folder (pf-rpn venv, GPU idle):
  python ../bench/eval_ar.py --config configs/pf-rpn/pf-rpn_coco-imagenet.py --ckpt checkpoints/pf_rpn_swinb_5p_coco_imagenet.pth \
     --ann /mnt/d/ditillation/data/coco_full/annotations/instances_val2017.json --images /mnt/d/ditillation/data/coco_full/val2017 \
     --n 300 --out ../results/ar_full.json  [--fp16] [--tf32] [--fast-predict] [--scale 640 1067] [--num-queries N] [--iters K]"""
import argparse, copy, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pathlib import Path
import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True); p.add_argument("--ckpt", required=True)
    p.add_argument("--ann", required=True); p.add_argument("--images", required=True)
    p.add_argument("--n", type=int, default=300); p.add_argument("--out", required=True)
    p.add_argument("--fp16", action="store_true"); p.add_argument("--tf32", action="store_true")
    p.add_argument("--fast-predict", action="store_true")
    p.add_argument("--scale", type=int, nargs=2, default=[800, 1333])
    p.add_argument("--num-queries", type=int); p.add_argument("--iters", type=int); p.add_argument("--topk", type=int)
    a = p.parse_args()
    if a.tf32:
        torch.set_float32_matmul_precision("high")
    from mmengine.config import Config
    from mmengine.dataset import Compose, pseudo_collate
    from mmengine.registry import init_default_scope
    from mmdet.apis import init_detector
    from mmdet.apis.inference import get_test_pipeline_cfg
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from PIL import Image
    from bench.run_pfrpn import shrink_queries

    cfg = Config.fromfile(a.config)
    if a.iters is not None:
        cfg.model.sp_iter_num = a.iters
    if a.topk is not None:
        cfg.model.topk = a.topk
    init_default_scope("mmdet")
    model = init_detector(cfg, a.ckpt, device="cuda").eval()
    if a.fast_predict:
        from owod.pfrpn_fast import apply_fast_predict
        apply_fast_predict(model)
    if a.num_queries:
        shrink_queries(model, a.num_queries)
    pc = get_test_pipeline_cfg(cfg); pc[0].type = "mmdet.LoadImageFromNDArray"
    pc = [t for t in pc if t["type"] != "LoadAnnotations"]
    for t in pc:
        if "Resize" in t["type"]:
            t["scale"] = tuple(a.scale)
    pipe = Compose(pc)

    gt = COCO(a.ann)
    ids = sorted(gt.getImgIds())[:a.n]
    # class-agnostic: every ground-truth box is category 1; crowd boxes stay ignored
    agn = copy.deepcopy(gt.dataset)
    agn["categories"] = [{"id": 1, "name": "object"}]
    for ann in agn["annotations"]:
        ann["category_id"] = 1
    agn["images"] = [im for im in agn["images"] if im["id"] in set(ids)]
    agn["annotations"] = [an for an in agn["annotations"] if an["image_id"] in set(ids)]
    gt_agn = COCO(); gt_agn.dataset = agn; gt_agn.createIndex()

    dets = []
    for i, img_id in enumerate(ids):
        info = gt.loadImgs(img_id)[0]
        img = np.array(Image.open(Path(a.images) / info["file_name"]).convert("RGB"))[:, :, ::-1].copy()
        data = pseudo_collate([pipe(dict(img=img, img_id=img_id, text="object", custom_entities=False))])
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=a.fp16):
            inst = model.test_step(data)[0].pred_instances
        b = inst.bboxes.float().cpu().numpy(); s = inst.scores.float().cpu().numpy()
        for box, sc in zip(b, s):
            dets.append({"image_id": img_id, "category_id": 1, "score": float(sc),
                         "bbox": [float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1])]})
        if (i + 1) % 50 == 0:
            print(i + 1, "images", flush=True)
    res = {"n_images": len(ids), "n_dets": len(dets), "knobs": vars(a)}
    if dets:
        ev = COCOeval(gt_agn, gt_agn.loadRes(dets), "bbox")
        ev.params.imgIds = ids
        ev.params.maxDets = [100, 300, 900]
        ev.evaluate(); ev.accumulate(); ev.summarize()
        st = ev.stats
        res.update({"ar100": float(st[6]) * 100, "ar300": float(st[7]) * 100, "ar900": float(st[8]) * 100,
                    "ar_small": float(st[9]) * 100, "ar_medium": float(st[10]) * 100, "ar_large": float(st[11]) * 100})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
