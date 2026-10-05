# SPEC-003: classify detector crops with imajev (in-process package, no HTTP)

## Goal
Take boxes from the detector (PF-RPN), group them, cut padded crops, and ask imajev-4b what each crop shows.
A crop with several objects and no main object is `unknown`. imajev runs as a Python package in the same process. No server, no API.

## Terms
- Box: (x1, y1, x2, y2, score) in image pixels.
- Group: boxes that cover the same object (high IoU, or one inside another). The best-scoring box is the representative.
- Margin: extra context around a box, as a fraction of box width and height.
- Pad: extra context in pixels, and optional square letterbox fill.
- Main object: one object that fills most of the crop.
- unknown: the crop is not one of the known classes, or has no main object.

## Requirements
### Grouping (CPU, pure Python)
- G1: Two boxes with IoU >= iou_thr belong to one group.
- G2: A box with at least contain_thr of its area inside another box belongs to that group only if the IoU rule also holds. Smaller boxes inside a large box stay separate groups. They are "inner" boxes of the large one.
- G3: The representative is the highest-score box. Output order is by score, descending.
- G4: Grouping is deterministic and has no random choice.

### Crop geometry (CPU, pure Python)
- C1: expand(box, margin, pad_px) grows each side by margin*size + pad_px, then clips to the image.
- C2: With square=True the crop becomes square before clipping. Parts outside the image are filled with fill_color (letterbox), not stretched.
- C3: A crop smaller than min_side px is grown to min_side around its center (clipped to the image).
- C4: An image with zero or negative box area is rejected with ValueError.

### Multi-object signal
- M1 (geometry): count inner groups whose area is inside the group box by at least inner_thr. If there are >= 2 inner groups, and the largest inner group covers < dominance_thr of the group area, then `geometric_multi = True`.
- M2 (model): one choice question about the crop with options `one_main_object`, `several_objects_none_dominant`, `no_clear_object`.
- M3 (decision): the crop is multi when p(several) >= multi_hi, OR (geometric_multi AND p(several) >= multi_lo). A part box inside a person is NOT multi: geometry alone never decides.
- M4: If the crop is multi or has no clear object (p(no_clear) >= none_hi), the class question is skipped and the verdict is `unknown`.

### Class decision
- D1: Ask one choice question with the known classes plus `none of these`. More than 254 classes are split in chunks of 254. The best class is the highest chunk probability. The verdict is "none of these" only if every chunk says so: p_none = min over chunks.
- D2: Verdict is known when the model did not abstain, p(top) >= min_conf, and p(none of these) < none_max. Else unknown with a reason: `multi_object`, `no_clear_object`, `not_in_known`, `low_confidence`, `abstained`.
- D3: Every verdict carries unknown_score = p(none of these) (or 1.0 for multi / no_clear) and the raw probabilities.

### Engine
- E1: ImajevEngine loads the model once, in-process, from the imajev checkout (scripts/playground TorchBackend, vision_decision package, calibration file). It builds the typed request, scores it, applies calibration, and returns the same dict shape as the HTTP API. No port is opened.
- E2: The classifier talks to a `Scorer` protocol: `ask(image, payload) -> dict`. Tests use a fake scorer.
- E3: The engine never runs while a latency benchmark runs (SPEC-001 R6).

## Acceptance tests
- T1-T4 (CPU): grouping G1..G4. T5-T8: crop geometry C1..C4. T9-T11: geometric_multi M1 (cluster, single object with part, one dominant inner).
- T12-T16 (CPU, fake scorer): M3/M4 decisions, D1 chunking, D2 reasons, one-question vs two-question flow (class question skipped on multi).
- T17 (CPU): pipeline wires boxes -> groups -> crops -> verdicts and keeps box ids.
- T18 (GPU, user machine): ImajevEngine loads and answers one real crop; JSON saved with timing.
- T19 (GPU): on a few real images, overlay shows verdicts. A person with face box stays one person, a shelf of items is unknown.

## Limits
imajev is not trained on detector crops. Class names are free text, so the prompt wording matters. Thresholds are starting values, to be tuned on labelled crops (SPEC-002 T4).
