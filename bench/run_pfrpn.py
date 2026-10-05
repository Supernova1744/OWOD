"""Latency of PF-RPN with speed knobs, batch 1. UNTESTED on GPU (T4).
Run inside the PF-RPN checkout (needs mmdet from that repo) with PYTHONPATH including this repo.
Usage: python bench/run_pfrpn.py --config configs/pf-rpn/pf-rpn_coco-imagenet.py \
  --ckpt checkpoints/pf_rpn_swinb_5p_coco_imagenet.pth --out results/pfrpn_q300.json \
  --num-queries 300 --iters 1 --topk 2 --size 800 1333 [--fp16]"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from bench.stats import summarize
from bench.timing import time_fn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True); p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--num-queries", type=int); p.add_argument("--iters", type=int)
    p.add_argument("--topk", type=int)
    p.add_argument("--size", type=int, nargs=2, default=[800, 1333])
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--runs", type=int, default=50)
    a = p.parse_args()
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmdet.apis import init_detector
    cfg = Config.fromfile(a.config)
    if a.num_queries:
        cfg.model.num_queries = a.num_queries
        cfg.model.test_cfg = dict(max_per_img=a.num_queries)
    if a.iters is not None:
        cfg.model.sp_iter_num = a.iters
    if a.topk is not None:
        cfg.model.topk = a.topk
    init_default_scope("mmdet")
    # NOTE: if num_queries changes, decoder-query weights may not load strictly; see docs.
    model = init_detector(cfg, a.ckpt, device="cuda").eval()
    from mmdet.apis import inference_detector
    import numpy as np
    img = (np.random.rand(*a.size, 3) * 255).astype("uint8")

    @torch.no_grad()
    def run():
        with torch.autocast("cuda", dtype=torch.float16, enabled=a.fp16):
            inference_detector(model, img, text_prompt="object")

    lat = time_fn(run, runs=a.runs)
    res = {"model": "pfrpn", "knobs": {"num_queries": a.num_queries, "iters": a.iters,
           "topk": a.topk, "fp16": a.fp16}, "latency": summarize(lat), "size": a.size,
           "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
