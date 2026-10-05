"""Where do the ~650 ms go? Stage hooks showed backbone 157 + encoder 118 + decoder 31 ms; ~340 ms is elsewhere.
This script times PF-RPN's own methods (inclusive, CUDA events) and runs torch.profiler for the top ops and sync points.
Run from the PF-RPN folder, pf-rpn venv, GPU idle:
  python ../bench/profile_pfrpn.py --config configs/pf-rpn/pf-rpn_coco-imagenet.py \
     --ckpt checkpoints/pf_rpn_swinb_5p_coco_imagenet.pth --out ../results/profile_full.json [--tf32] [--fp16]
UNTESTED on GPU."""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import torch

METHODS = ["extract_feat", "pre_transformer", "_build_sparse_moe_pseudo_text", "cascade_self_prompt_enhancement",
           "forward_encoder", "pre_decoder", "gen_encoder_output_proposals", "_select_score_guided_queries",
           "forward_decoder", "forward_transformer"]
HEAD_METHODS = ["predict"]
SYNC_OPS = ("aten::item", "aten::_local_scalar_dense", "aten::nonzero", "aten::to", "cudaStreamSynchronize",
            "cudaDeviceSynchronize", "cudaMemcpyAsync", "aten::copy_")


def wrap(obj, name, store, label):
    fn = getattr(obj, name, None)
    if fn is None:
        return
    def w(*a, **k):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); out = fn(*a, **k); e.record()
        store.setdefault(label, []).append((s, e))
        return out
    setattr(obj, name, w)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True); p.add_argument("--ckpt", required=True); p.add_argument("--out", required=True)
    p.add_argument("--fp16", action="store_true"); p.add_argument("--tf32", action="store_true")
    p.add_argument("--img-size", type=int, nargs=2, default=[800, 1333]); p.add_argument("--runs", type=int, default=20)
    a = p.parse_args()
    if a.tf32:
        torch.set_float32_matmul_precision("high")
    from mmengine.config import Config
    from mmengine.dataset import Compose, pseudo_collate
    from mmengine.registry import init_default_scope
    from mmdet.apis import init_detector
    from mmdet.apis.inference import get_test_pipeline_cfg
    cfg = Config.fromfile(a.config); init_default_scope("mmdet")
    model = init_detector(cfg, a.ckpt, device="cuda").eval()
    pc = get_test_pipeline_cfg(cfg); pc[0].type = "mmdet.LoadImageFromNDArray"
    pipe = Compose([t for t in pc if t["type"] != "LoadAnnotations"])
    img = (np.random.rand(*a.img_size, 3) * 255).astype("uint8")
    data = pseudo_collate([pipe(dict(img=img, img_id=0, text="object", custom_entities=False))])

    store = {}
    for m in METHODS:
        wrap(model, m, store, m)
    for m in HEAD_METHODS:
        wrap(model.bbox_head, m, store, "bbox_head." + m)

    @torch.no_grad()
    def run():
        with torch.autocast("cuda", dtype=torch.float16, enabled=a.fp16):
            model.test_step(data)

    for _ in range(10):
        run()
    torch.cuda.synchronize(); store.clear()
    t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    totals = []
    for _ in range(a.runs):
        t0.record(); run(); t1.record(); torch.cuda.synchronize(); totals.append(t0.elapsed_time(t1))
    inclusive = {k: float(np.median([s.elapsed_time(e) for s, e in v])) for k, v in store.items()}

    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        run(); torch.cuda.synchronize()
    avg = prof.key_averages()
    rows = sorted(avg, key=lambda r: -getattr(r, "self_device_time_total", getattr(r, "self_cuda_time_total", 0)))[:25]
    def dev(r):
        return getattr(r, "self_device_time_total", getattr(r, "self_cuda_time_total", 0)) / 1000.0
    top_gpu = [{"op": r.key, "calls": r.count, "self_gpu_ms": round(dev(r), 2), "self_cpu_ms": round(r.self_cpu_time_total / 1000.0, 2)} for r in rows]
    syncs = [{"op": r.key, "calls": r.count, "cpu_ms": round(r.cpu_time_total / 1000.0, 2)} for r in avg if r.key in SYNC_OPS]
    gpu_busy_ms = sum(dev(r) for r in avg)
    res = {"total_ms_median": float(np.median(totals)), "knobs": {"fp16": a.fp16, "tf32": a.tf32, "img": list(a.img_size)},
           "inclusive_ms_median": inclusive, "gpu_busy_ms_one_run": round(gpu_busy_ms, 1),
           "top_ops_by_self_gpu": top_gpu, "sync_like_ops": syncs, "gpu": torch.cuda.get_device_name(0)}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))
    print("\nIf gpu_busy_ms is much lower than total_ms, the GPU is waiting on the CPU (Python loops, syncs).")


if __name__ == "__main__":
    main()
