"""T18 smoke test (imajev venv, GPU): load imajev in-process and ask the scene question on one image. UNTESTED.
  python scripts/smoke_imajev.py --image some.jpg [--imajev-dir ~/imajev]
Prints the raw response, option keys, max_options, load time, peak VRAM and ms per question."""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pathlib import Path
from PIL import Image
from owod.crop_classifier import scene_payload, class_payloads
from owod.imajev_engine import ImajevEngine


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True); p.add_argument("--imajev-dir", default=str(Path.home() / "imajev"))
    p.add_argument("--rotations", type=int, default=1); p.add_argument("--runs", type=int, default=5)
    a = p.parse_args()
    import torch
    e = ImajevEngine(a.imajev_dir, rotations=a.rotations)
    print(f"load {e.load_seconds:.1f}s  max_options={e.max_options}  calibration={'yes' if e.calibration else 'no'}")
    img = Image.open(a.image).convert("RGB")
    r = e.ask(img, scene_payload())
    print(json.dumps(r, indent=1))
    ans = r["answers"]["scene"]
    assert ans["type"] == "choice" and set(ans["probabilities"]) == set(scene_payload()["questions"]["scene"]["criteria"])
    cls = class_payloads(["person", "car", "dog"], e.max_options)[0]
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(a.runs):
        e.ask(img, scene_payload()); e.ask(img, cls)
    torch.cuda.synchronize()
    print(f"{(time.perf_counter() - t) * 1000 / (2 * a.runs):.0f} ms per question; peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    print("SMOKE OK")


if __name__ == "__main__":
    main()
