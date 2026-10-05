"""PF-RPN proposals -> imajev crop verdicts -> unknown store -> learn new classes without training. See SPEC-004."""
import hashlib
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from .crop_classifier import Config, CropClassifier
from .pipeline import CropResult, PipelineConfig, classify_detections
from .state import OWState, UnknownStore

RELABEL_REASONS = ("not_in_known", "low_confidence", "abstained", "failed_verification")


class OpenWorldSystem:
    def __init__(self, detect_fn: Callable[[Sequence[str]], Dict[str, list]], scorer, state: OWState,
                 store: UnknownStore, classifier_config: Optional[Config] = None,
                 pipeline_config: Optional[PipelineConfig] = None, max_options: int = 254,
                 state_path: Optional[str] = None):
        self.detect_fn, self.scorer, self.state, self.store = detect_fn, scorer, state, store
        self.state_path = state_path
        self.cc, self.pc, self.max_options = classifier_config or Config(), pipeline_config or PipelineConfig(), max_options
        self._rebuild()

    def _rebuild(self):
        if not self.state.known:
            raise ValueError("add at least one known class first")
        self.classifier = CropClassifier(self.scorer, self.state.known, self.cc, self.max_options)

    def process_many(self, paths: Sequence[str]) -> Dict[str, List[CropResult]]:
        from PIL import Image
        boxes = self.detect_fn(list(paths))            # ONE detector call for the whole batch
        out = {}
        for p in paths:
            found = boxes.get(str(Path(p).resolve()), boxes.get(str(p)))
            if found is None:
                raise KeyError(f"detector returned no entry for {p}")
            key = hashlib.sha1(Path(p).read_bytes()).hexdigest()       # same pixels from another path = same image
            out[str(p)] = self._classify_image(Image.open(p).convert("RGB"), str(p), found, key)
        return out

    def process_image(self, path: str):
        return self.process_many([path])[str(path)]

    def _classify_image(self, image, name: str, boxes, key: Optional[str] = None) -> List[CropResult]:
        crops = {}
        results = classify_detections(image, boxes, self.classifier, self.pc, on_crop=lambda r, c: crops.__setitem__(r.index, c))
        for r in results:
            if r.verdict.is_unknown and r.part_of is None:       # suppressed parts are never stored
                b = r.box
                if self.store.find(key or name, (b.x1, b.y1, b.x2, b.y2)) is not None:
                    continue                                    # same image and box seen before: no duplicate record
                self.store.add(crops[r.index], name, (b.x1, b.y1, b.x2, b.y2, b.score), r.verdict.reason, r.verdict.unknown_score,
                               image_key=key)
        return results

    def learn(self, name: str, ids: Sequence[int] = (), relabel: bool = True, accept_old_class: bool = False) -> dict:
        """accept_old_class=False: a stored unknown is recovered only if it now matches the NEW class. Switches to an
        old class are side effects of the longer option list (drift), so they stay unknown unless asked for."""
        for rid in ids:
            if self.store.get(rid)["status"] != "unknown":
                raise ValueError(f"record {rid} is not unknown")
        self.state.add_class(name, note=f"{len(ids)} examples")
        if self.state_path:
            self.state.save(self.state_path)            # save BEFORE any record changes: state and store stay in step
        self._rebuild()
        for rid in ids:
            self.store.mark(rid, "labelled", name)
        recovered: Dict[int, str] = {}
        if relabel:
            for rid in self.store.ids("unknown", RELABEL_REASONS):
                v = self.classifier.classify(self.store.crop(rid), geometric_multi=False)
                if not v.is_unknown and (v.label == name or accept_old_class):
                    self.store.mark(rid, "recovered", v.label)
                    recovered[rid] = v.label
        new = {k: v for k, v in recovered.items() if v == name}
        return {"added": name, "n_known": len(self.state.known), "labelled": len(ids),
                "recovered": len(recovered), "recovered_new_class": len(new), "recovered_old_class": len(recovered) - len(new),
                "recovered_labels": recovered}
