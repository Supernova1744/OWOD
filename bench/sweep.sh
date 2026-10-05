#!/usr/bin/env bash
# Speed sweep with no retraining. Run from the PF-RPN folder, venv active.
# Rules: laptop on AC power; no imajev setup, training, or other GPU job running.
set -u
B="$(cd "$(dirname "$0")" && pwd)"; R="$B/../results"; mkdir -p "$R"
CFG=configs/pf-rpn/pf-rpn_coco-imagenet.py; CK=checkpoints/pf_rpn_swinb_5p_coco_imagenet.pth
while pgrep -f "setup_imajev_wsl.sh|pip install|hf download|download_model.py" >/dev/null; do echo "waiting for installs to finish..."; sleep 30; done
gpu() { nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu --format=csv,noheader | tee -a "$R/sweep_gpu.log"; }
run() { name=$1; shift; echo "== $name"; gpu >/dev/null
  python "$B/run_pfrpn.py" --config $CFG --ckpt $CK --out "$R/sweep_$name.json" "$@" > "$R/sweep_$name.log" 2>&1 \
    || echo "FAILED $name (see $R/sweep_$name.log)"
  grep -i "size mismatch" "$R/sweep_$name.log" && echo "WARNING: weights skipped in $name"; gpu >/dev/null; }
python "$B/run_baseline.py" --out "$R/baseline_start.json" > /dev/null 2>&1 || echo "baseline start failed"
run full
run stages --stages
run tf32 --tf32
run fp16 --fp16
run bf16 --bf16
run q300 --num-queries 300
run q100 --num-queries 100
run it1 --iters 1
run topk1 --topk 1
run best_a --tf32 --fp16 --num-queries 300 --iters 1
run best_b --tf32 --bf16 --num-queries 300 --iters 1
run best_a_s640 --tf32 --fp16 --num-queries 300 --iters 1 --scale 640 1067 --img-size 640 1067
run best_a_q100_s512 --tf32 --fp16 --num-queries 100 --iters 1 --scale 512 853 --img-size 512 853
python "$B/run_baseline.py" --out "$R/baseline_end.json" > /dev/null 2>&1 || echo "baseline end failed"
python - <<PY
import json,glob,os
r="$R"
b0=json.load(open(r+"/baseline_start.json"))["latency"]["median_ms"]; b1=json.load(open(r+"/baseline_end.json"))["latency"]["median_ms"]
print(f"baseline median start {b0:.1f} ms, end {b1:.1f} ms (drift {100*(b1-b0)/b0:+.1f}%)")
base=(b0+b1)/2
for f in sorted(glob.glob(r+"/sweep_*.json")):
    d=json.load(open(f)); m=d["latency"]["median_ms"]
    print(f"{os.path.basename(f)[6:-5]:16s} {m:8.1f} ms  ratio {m/base:5.2f}  {d['knobs']}")
    if "stages_ms_median" in d: print("   stages:", d["stages_ms_median"])
PY
