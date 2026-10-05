#!/usr/bin/env bash
# Speed sweep with no retraining. Run from the PF-RPN folder, venv active, GPU idle (no imajev, no training).
# Usage: bash ../bench/sweep.sh   -> results/sweep_*.json and a table
set -u
B="$(cd "$(dirname "$0")" && pwd)"; R="$B/../results"; mkdir -p "$R"
CFG=configs/pf-rpn/pf-rpn_coco-imagenet.py; CK=checkpoints/pf_rpn_swinb_5p_coco_imagenet.pth
run() { name=$1; shift; python "$B/run_pfrpn.py" --config $CFG --ckpt $CK --out "$R/sweep_$name.json" "$@" > "$R/sweep_$name.log" 2>&1 \
  || echo "FAILED $name (see $R/sweep_$name.log)"; }
run full --stages
run q300 --num-queries 300
run q100 --num-queries 100
run it1 --iters 1
run fp16 --fp16
run fp16_q300_it1 --fp16 --num-queries 300 --iters 1
run fp16_q300_it1_s640 --fp16 --num-queries 300 --iters 1 --scale 640 1067 --img-size 640 1067
run fp16_q100_it1_s512 --fp16 --num-queries 100 --iters 1 --scale 512 853 --img-size 512 853
python - <<PY
import json,glob,os
base=json.load(open("$R/baseline.json"))["latency"]["median_ms"]
print(f"baseline median {base:.1f} ms")
for f in sorted(glob.glob("$R/sweep_*.json")):
    d=json.load(open(f)); m=d["latency"]["median_ms"]
    print(f"{os.path.basename(f)[6:-5]:24s} {m:8.1f} ms  ratio {m/base:5.2f}  {d['knobs']}")
PY
