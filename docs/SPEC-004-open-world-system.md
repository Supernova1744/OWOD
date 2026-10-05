# SPEC-004: open-world detection system (PF-RPN proposals + imajev crops + incremental classes)

## Goal
One system that (1) finds objects with PF-RPN, (2) labels each as a known class or `unknown` with imajev (SPEC-003),
(3) keeps the unknown crops, and (4) learns new classes WITHOUT training: a new class is a new name in the known list.
Decision (user): PF-RPN runs about 3.7x slower than a standard RPN (fp16, fast predict, 640x1067, recall 99.7% of full). Accepted.

## Terms
- State: the ordered list of known classes plus a log of changes. Saved as JSON.
- Unknown store: a folder with crop images and an index. Each record has id, image, box, reason, unknown_score, status.
- Learn: add a class name, mark chosen unknown records with that name, then re-check the other stored unknowns.

## Requirements
- S1 State: add_class rejects empty, duplicate (case-insensitive) and reserved names. Every change is logged. Save/load is lossless.
- S2 Store: add() saves the crop and returns a new id; ids never repeat after removals; the index survives a restart.
- S3 Process: for each image, detect (one detector call for many images), group, crop, classify. Known crops are returned.
  Unknown crops that are not suppressed parts are stored. Suppressed parts are never stored.
- S4 Learn: learn(name, ids) adds the class, marks the given records `labelled` with that label, rebuilds the classifier, and
  re-classifies the remaining `unknown` records whose reason is not_in_known, low_confidence, abstained or failed_verification
  (never multi_object or no_clear_object). Records that become known are marked `recovered`.
- S5 Detector process: PF-RPN needs Python 3.10 and its own venv; imajev needs 3.11. The detector runs as a subprocess
  that returns boxes as JSON. It is called once per batch of images.
- S6 Metric U-Recall: share of ground-truth unknown objects covered (IoU >= 0.5) by at least one predicted unknown box.
- S3b Dedupe by image CONTENT (file hash) and box, not by path. The same pixels from another path add no records.
- S4b A stored unknown is recovered only if it now matches the NEW class. Switches to old classes (drift from the longer option list) stay unknown unless accept_old_class is set.
- S7 Nothing in the system opens a network port.

## Acceptance tests
T1-T3 state; T4-T6 store; T7-T10 process (known returned, unknown stored, parts not stored, one detector call);
T11-T14 learn (adds class, labels ids, recovers similar, skips multi_object); T15 subprocess detector with a fake script;
T16 U-Recall. T17 (GPU, user machine): end-to-end run on test_images, then `learn` on the pottery crops from the shelf image.

## Not in scope yet
WI and A-OSE need known-class ground truth on a standard benchmark (M-OWODB). Class-wise AP the same. Add with a dataset.
Clustering of unknown crops needs image embeddings; imajev gives none. A human picks the ids for now.
