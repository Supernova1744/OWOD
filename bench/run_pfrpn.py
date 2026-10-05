"""Latency of PF-RPN with speed knobs, batch 1. Needs the PF-RPN checkout (its mmdet) on the path.

Timed region = model.test_step(data): GPU normalize + backbone + transformer + head + postprocess.
Image loading and resize are done ONCE before timing (the old version rebuilt the pipeline every call).
The baseline (run_baseline.py) times its own transform + backbone + rpn the same way.

Usage (from the PF-RPN folder):
  python ../bench/run_pfrpn.py --config configs/pf-rpn/pf-rpn_coco-imagenet.py \
    --ckpt checkpoints/pf_rpn_swinb_5p_coco_imagenet.pth --out ../results/pfrpn_full.json \
    [--num-queries 300] [--iters 1] [--topk 1] [--scale 800 1333] [--fp16] [--stages]
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import torch
from bench.stats import summarize
from bench.timing import time_fn


class StageTimer:
    """CUDA-event timing of named submodules. Used only with --stages (adds small overhead)."""
    def __init__(self, model, names):
        self.pairs = {n: [] for n in names}
        self.pending = {}
        for n in names:
            mod = getattr(model, n, None)
            if mod is None:
                self.pairs.pop(n)
                continue
            mod.register_forward_pre_hook(self._pre(n))
            mod.register_forward_hook(self._post(n))

    def _pre(self, n):
        def h(_m, _i):
            e = torch.cuda.Event(enable_timing=True); e.record(); self.pending[n] = e
        return h

    def _post(self, n):
        def h(_m, _i, _o):
            e = torch.cuda.Event(enable_timing=True); e.record()
            self.pairs[n].append((self.pending.pop(n), e))
        return h

    def reset(self):
        for v in self.pairs.values():
            v.clear()

    def summary_ms(self):
        torch.cuda.synchronize()
        return {n: float(np.median([s.elapsed_time(e) for s, e in v])) if v else None
                for n, v in self.pairs.items()}


def shrink_queries(model, n):
    """Keep n of the trained content queries. The first half of the rows pairs with the confidence-ranked
    proposals and the second half with the class-ranked ones (see _select_score_guided_queries), so
    slice both blocks. Never rebuild the model with a smaller num_queries: the checkpoint would not load."""
    import torch.nn as nn
    old = model.num_queries
    if not 0 < n <= old:
        raise ValueError(f"num_queries must be in 1..{old}")
    n_conf, n_cls = n // 2, n - n // 2
    idx = torch.cat([torch.arange(0, n_conf), torch.arange(old // 2, old // 2 + n_cls)]).to(
        model.query_embedding.weight.device)
    w = model.query_embedding.weight.data[idx].clone()
    model.query_embedding = nn.Embedding(n, w.shape[1]).to(w.device)
    model.query_embedding.weight.data.copy_(w)
    model.num_queries = n
    if hasattr(model, "dn_query_generator"):
        model.dn_query_generator.num_matching_queries = n
    for holder in (model, model.bbox_head):
        tc = getattr(holder, "test_cfg", None)
        if tc is not None:
            tc["max_per_img"] = n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True); p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--num-queries", type=int); p.add_argument("--iters", type=int)
    p.add_argument("--topk", type=int)
    p.add_argument("--scale", type=int, nargs=2, default=[800, 1333], help="FixScaleResize scale")
    p.add_argument("--img-size", type=int, nargs=2, default=[800, 1333], help="synthetic image H W")
    p.add_argument("--fp16", action="store_true"); p.add_argument("--bf16", action="store_true")
    p.add_argument("--tf32", action="store_true", help="allow TF32 matmul")
    p.add_argument("--stages", action="store_true", help="also report per-stage median ms")
    p.add_argument("--fast-predict", action="store_true", help="owod.pfrpn_fast.apply_fast_predict (same results, no per-label syncs)")
    p.add_argument("--runs", type=int, default=50); p.add_argument("--warmup", type=int, default=10)
    a = p.parse_args()

    from mmengine.config import Config
    from mmengine.dataset import pseudo_collate
    from mmengine.registry import init_default_scope
    from mmengine.dataset import Compose
    from mmdet.apis import init_detector
    from mmdet.apis.inference import get_test_pipeline_cfg

    if a.tf32:
        torch.set_float32_matmul_precision("high")
    cfg = Config.fromfile(a.config)
    if a.iters is not None:
        cfg.model.sp_iter_num = a.iters
    if a.topk is not None:
        cfg.model.topk = a.topk
    init_default_scope("mmdet")
    model = init_detector(cfg, a.ckpt, device="cuda").eval()   # always load with the trained 900 queries
    if a.fast_predict:
        from owod.pfrpn_fast import apply_fast_predict
        apply_fast_predict(model)
    if a.num_queries:
        shrink_queries(model, a.num_queries)

    pipe_cfg = get_test_pipeline_cfg(cfg)
    pipe_cfg[0].type = "mmdet.LoadImageFromNDArray"
    pipe_cfg = [t for t in pipe_cfg if t["type"] != "LoadAnnotations"]
    for t in pipe_cfg:
        if "Resize" in t["type"]:
            t["scale"] = tuple(a.scale)
    pipeline = Compose(pipe_cfg)
    img = (np.random.rand(*a.img_size, 3) * 255).astype("uint8")
    data = pseudo_collate([pipeline(dict(img=img, img_id=0, text="object", custom_entities=False))])

    timer = StageTimer(model, ["backbone", "neck", "encoder", "decoder"]) if a.stages else None

    @torch.no_grad()
    def run():
        with torch.autocast("cuda", dtype=torch.bfloat16 if a.bf16 else torch.float16, enabled=a.fp16 or a.bf16):
            model.test_step(data)

    lat = time_fn(run, warmup=a.warmup, runs=a.runs)
    res = {"model": "pfrpn",
           "knobs": {"num_queries": int(model.num_queries),
                     "sp_iter_num": int(getattr(model, "sp_iter_num", -1)),
                     "topk": int(getattr(model, "topk", -1)),
                     "scale": list(a.scale), "fp16": a.fp16, "bf16": a.bf16, "tf32": a.tf32, "fast_predict": a.fast_predict},
           "latency": summarize(lat), "img_size": list(a.img_size),
           "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "mmdet_file": __import__("mmdet").__file__}
    if timer:
        timer.reset()
        for _ in range(20):
            run()
        res["stages_ms_median"] = timer.summary_ms()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
