"""Known-class state and the unknown-crop store. Pure Python (Pillow only to save crops)."""
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .crop_classifier import NONE_OPTION, UNKNOWN


@dataclass
class OWState:
    known: List[str] = field(default_factory=list)
    log: List[dict] = field(default_factory=list)

    def add_class(self, name: str, note: str = "") -> None:
        name = (name or "").strip()
        if not name:
            raise ValueError("class name must not be empty")
        if name.lower() in (NONE_OPTION, UNKNOWN):
            raise ValueError(f"{name!r} is reserved")
        if name.lower() in (k.lower() for k in self.known):
            raise ValueError(f"class {name!r} already exists")
        self.known.append(name)
        self.log.append({"event": "add_class", "name": name, "n_known": len(self.known), "note": note,
                         "time": time.strftime("%Y-%m-%dT%H:%M:%S")})

    def save(self, path) -> None:
        Path(path).write_text(json.dumps({"known": self.known, "log": self.log}, indent=1))

    @classmethod
    def load(cls, path) -> "OWState":
        d = json.loads(Path(path).read_text())
        return cls(known=list(d["known"]), log=list(d["log"]))


class UnknownStore:
    """Folder with crops/*.png and index.json. Records: unknown -> labelled | recovered."""

    def __init__(self, root):
        self.root = Path(root)
        (self.root / "crops").mkdir(parents=True, exist_ok=True)
        self._index = self.root / "index.json"
        d = json.loads(self._index.read_text()) if self._index.exists() else {"next_id": 1, "records": {}}
        self.next_id, self.records = d["next_id"], d["records"]

    def _save(self):
        self._index.write_text(json.dumps({"next_id": self.next_id, "records": self.records}, indent=1))

    def add(self, crop_image, image: str, box, reason: str, unknown_score: float, image_key: Optional[str] = None) -> int:
        rid = self.next_id
        self.next_id += 1
        rel = f"crops/{rid:06d}.png"
        crop_image.save(self.root / rel)
        self.records[str(rid)] = {"id": rid, "image": str(image), "box": [float(v) for v in box], "reason": reason,
                                  "unknown_score": float(unknown_score), "image_key": image_key or str(image), "crop": rel, "status": "unknown", "label": None,
                                  "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
        self._save()
        return rid

    def get(self, rid: int) -> dict:
        return self.records[str(rid)]

    def crop(self, rid: int):
        from PIL import Image
        return Image.open(self.root / self.get(rid)["crop"]).convert("RGB")

    def ids(self, status: Optional[str] = "unknown", reasons=None) -> List[int]:
        out = [r["id"] for r in self.records.values()
               if (status is None or r["status"] == status) and (reasons is None or r["reason"] in reasons)]
        return sorted(out)

    def find(self, image_key: str, box, tol: float = 1.0) -> Optional[int]:
        """Id of an existing record for the same image CONTENT (image_key = file hash) and (nearly) the same box."""
        for r in self.records.values():
            if r.get("image_key", r["image"]) == str(image_key) and all(abs(a - b) <= tol for a, b in zip(r["box"][:4], box[:4])):
                return r["id"]
        return None

    def mark(self, rid: int, status: str, label: str) -> None:
        r = self.get(rid)
        r["status"], r["label"] = status, label
        self._save()
