"""Latency of the torchvision Faster R-CNN RPN (ResNet-50 FPN), batch 1. UNTESTED on GPU (T3).
Usage: python bench/run_baseline.py --out results/baseline.json [--size 800 1333]"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
import torchvision
from bench.stats import summarize
from bench.timing import time_fn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--size", type=int, nargs=2, default=[800, 1333])
    p.add_argument("--runs", type=int, default=50)
    a = p.parse_args()
    dev = "cuda"
    m = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None).to(dev).eval()
    img = [torch.rand(3, *a.size, device=dev)]

    @torch.no_grad()
    def run():
        images, _ = m.transform(img)
        feats = m.backbone(images.tensors)
        m.rpn(images, feats)   # proposals only; no ROI heads

    lat = time_fn(run, runs=a.runs)
    res = {"model": "torchvision_rpn_r50fpn", "latency": summarize(lat), "size": a.size,
           "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
