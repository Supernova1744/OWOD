# SPEC-002: imajev-4b as an optional unknown-object labeller

## Goal
For each crop from the PF-RPN proposer, decide: known class, or unknown object.
This is an OFFLINE or second-stage step. It does not replace PF-RPN and is not in the speed gate (SPEC-001).

## Terms
- Crop: an image region cut from a PF-RPN box.
- Known classes: the class names the OWOD system has learned so far.
- unknown_score: probability that the crop is NOT any known class.

## Requirements
- R1: Build one `POST /v1/systemone` request per crop. One `choice` question. Options = known classes + `none of these`.
- R2: unknown_score = P("none of these"). The model's own `unknown_probability` is kept as a separate field.
- R3: `abstained = true` is reported, never hidden. Abstained crops are not counted as known or unknown.
- R4: At most 255 options per question (model limit). More known classes are split in chunks.
- R5: Calibration file `calibration-rot4.json` is used when serving. It is a serving choice, not code.
- R6: Never run imajev on the GPU while a SPEC-001 latency benchmark runs. They share one GPU. Timings would be wrong.

## Acceptance tests
- T1 (CPU): request builder output matches the API shape for 1 and for 300 classes.
- T2 (CPU): parser gives the right unknown_score and abstained flag for the model-card example responses.
- T3 (CPU): parser rejects a response with a missing or mismatched option.
- T4 (GPU, later): on N labelled OWOD crops, report unknown AUROC and abstain rate. Floor is set after the first run.

## Known limits (model card)
Trained on business photos and records, not detection crops. English only. Two-image comparison is weak (41.8%).
Over-confident without calibration. Single pass, no reasoning. Max 400,000 pixels per image.
