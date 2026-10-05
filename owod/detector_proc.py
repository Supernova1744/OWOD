"""Run PF-RPN in its own Python environment and read the boxes back (PF-RPN: Python 3.10; imajev: 3.11)."""
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Sequence

from .crops import Box


class SubprocessDetector:
    def __init__(self, python: str, pf_dir: str, repo_dir: str, top: int = 100, scale=(800, 1333),
                 fp16: bool = True, fast_predict: bool = True, script: str = None, extra_args: Sequence[str] = ()):
        self.python, self.pf_dir, self.repo_dir = python, pf_dir, repo_dir
        self.script = script or str(Path(repo_dir) / "scripts" / "detect_boxes_pfrpn.py")
        self.top, self.scale, self.fp16, self.fast_predict = top, tuple(scale), fp16, fast_predict
        self.extra = list(extra_args)
        self.calls = 0

    def __call__(self, paths: Sequence[str]) -> Dict[str, List[Box]]:
        """One subprocess for the whole batch. Returns {image path: boxes} in original pixels."""
        paths = [str(Path(p).resolve()) for p in paths]
        self.calls += 1
        with tempfile.TemporaryDirectory() as tmp:
            lst, out = Path(tmp) / "list.txt", Path(tmp) / "boxes.json"
            lst.write_text("\n".join(paths))
            cmd = [self.python, self.script, "--list", str(lst), "--out", str(out), "--top", str(self.top),
                   "--scale", str(self.scale[0]), str(self.scale[1])]
            if self.fp16:
                cmd.append("--fp16")
            if self.fast_predict:
                cmd.append("--fast-predict")
            env = dict(os.environ, PYTHONPATH=self.repo_dir + os.pathsep + os.environ.get("PYTHONPATH", ""))
            done = subprocess.run(cmd + self.extra, cwd=self.pf_dir, env=env, capture_output=True, text=True)
            if done.returncode != 0:
                raise RuntimeError(f"detector failed ({done.returncode}):\n{done.stderr[-2000:]}")
            data = json.loads(out.read_text())
        result = {}
        for item in data:
            result[item["image"]] = [Box(b[0], b[1], b[2], b[3], b[4], i) for i, b in enumerate(item["boxes"])]
        return result
