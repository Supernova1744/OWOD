# SPEC-001: PF-RPN speed parity with a standard RPN

## Goal
Make PF-RPN run at about the same speed as a standard RPN. Then use it as the
unknown-object proposer in an OWOD system (later specs).

## Terms
- Baseline RPN: torchvision Faster R-CNN RPN (ResNet-50 FPN). The paper compares to this RPN.
- Candidate: PF-RPN with one set of speed knobs.
- Speed ratio R = median latency (Candidate) / median latency (Baseline).

## Requirements
- R1: Measure Baseline and Candidate on the same machine, same input size, same batch size (1).
- R2: Use warm-up runs. Exclude them. Call `cuda.synchronize()` around each timed run.
- R3: Report the median, p90, and min of N runs. N is at least 50.
- R4: PASS speed when R <= 1.10.
- R5: PASS quality when Candidate AR100 on the fixed subset >= the floor.
  Default floor = 52.3 (paper, YOLO-World variant, CD-FSOD). Also report
  the full-model AR100 measured on the same machine. Quality FAIL overrides speed PASS.
- R6: Every result is written as JSON with the knob values, GPU name, torch version, and image size.

## Speed knobs (no retraining)
| Knob | Config key | Default | Try |
|---|---|---|---|
| Queries | `model.num_queries`, `test_cfg.max_per_img` | 900 | 300, 100 |
| Input size | test pipeline `Resize` | 800x1333 | 640, 512 |
| CSP iterations | `model.sp_iter_num` | 3 | 1 |
| Top-k levels | `model.topk` | 2 | 1 |
| Precision | autocast | fp32 | fp16 / bf16 |
| Export | TensorRT / ONNX | none | later |

## Speed knobs (retraining needed, GPU)
Lighter base (Swin-T, ResNet-50, YOLO-World), fewer encoder layers, distillation.

## Known facts
- Paper Supp. Table 9: CSP 1 -> 3 iterations costs about 4.6 ms of about 219 ms. The base model is the cost.
- Paper Supp. Table 10: PF-RPN 4.6 FPS, RPN 27.8 FPS, PF-RPN (YOLO-World) 25.1 FPS. RTX 4090.

## Acceptance tests
- T1 (unit, CPU): `bench/stats.py` computes median, p90, ratio, and the gate correctly.
- T2 (unit, CPU): gate FAILS when quality is under the floor, even if speed passes.
- T3 (GPU): `bench/run_baseline.py` writes a valid result JSON.
- T4 (GPU): `bench/run_pfrpn.py` writes a valid result JSON for each knob set.
- T5 (GPU): `bench/gate.py` prints PASS or FAIL from two result files.
