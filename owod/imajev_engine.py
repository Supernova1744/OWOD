"""Run imajev-4b inside this Python process (no HTTP server, no port).

It reuses imajev's own code from a checkout (scripts/playground TorchBackend + vision_decision) and follows the same
steps as its POST /v1/systemone handler: payload -> typed request -> backend.score -> calibration -> Jev-shaped answer.
Needs the imajev venv (torch, peft, transformers, fastapi for the module import) and the downloaded base model + adapter.
UNTESTED on GPU here: tested on the user's machine (SPEC-003 T18).
"""
import importlib.util
import sys
from pathlib import Path
from time import perf_counter
from typing import Optional


class ImajevEngine:
    def __init__(self, imajev_dir, adapter: Optional[str] = None, bundle: Optional[str] = None,
                 calibration: Optional[str] = "auto", rotations: int = 1, fast: bool = False,
                 max_input_tokens: int = 4096, model_name: str = "imajev-4b"):
        root = Path(imajev_dir).expanduser().resolve()
        for extra in (root / "src", root / "scripts", root / "scripts" / "playground"):
            if str(extra) not in sys.path:
                sys.path.insert(0, str(extra))
        spec = importlib.util.spec_from_file_location("imajev_playground_server", root / "scripts" / "playground" / "server.py")
        self._pg = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = self._pg
        spec.loader.exec_module(self._pg)
        from vision_decision.jev_api import to_response, to_request_with_plan
        self._to_response, self._to_request = to_response, to_request_with_plan

        adapter_dir = Path(adapter) if adapter else root / "adapters" / "imajev-4b"
        bundle_path = Path(bundle) if bundle else root / "artifacts" / "model-qwen4b.json"
        t0 = perf_counter()
        self.backend = self._pg.TorchBackend(bundle_path, adapter_dir, rotations=rotations,
                                             max_input_tokens=max_input_tokens, fast=fast)
        self.backend.model = model_name
        self.load_seconds = perf_counter() - t0
        self.max_options = int(getattr(self.backend, "max_options", 254))
        self.calibration = None
        if calibration:
            cal = (adapter_dir / ("calibration-rot4.json" if rotations == 4 else "calibration.json")
                   if calibration == "auto" else Path(calibration))
            if cal.is_file():
                from vision_decision.calibration import TemperatureCalibrator
                self.calibration = TemperatureCalibrator.load(cal)

    def ask(self, image, payload: dict) -> dict:
        """image: a PIL image. payload: Jev-style {"state":..., "questions":...}. Returns {"answers":..., "usage":...}."""
        request, plan = self._to_request(payload, request_id="owod", max_options=getattr(self.backend, "max_options", 254))
        image = image.convert("RGB")
        results, usage = self.backend.score([image], request)
        if self.calibration is not None:
            photo_only = not request.state
            results = [self.calibration.calibrate_result(r, f.type, len(r.scores) - 1, image=True, photo_only=photo_only)
                       for f, r in zip(request.fields, results)]
        body = self._to_response(request, results, model=self.backend.model, plan=plan)
        body["usage"] = usage
        return body
