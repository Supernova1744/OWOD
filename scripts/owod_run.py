"""Open-world detection CLI (imajev venv). PF-RPN runs in its own venv as a subprocess; imajev runs in this process. UNTESTED on GPU.

  run   : python scripts/owod_run.py run --images test_images --state results/owod/state.json --store results/owod/store \
              --classes-file results/voc20.txt --pf-dir /mnt/d/OWOD/OWOD/PF-RPN --detector-python ~/.venvs/pf-rpn/bin/python
  list  : python scripts/owod_run.py list --store results/owod/store
  learn : python scripts/owod_run.py learn --state results/owod/state.json --store results/owod/store --name pottery --ids 3,4,7
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pathlib import Path
from owod.crop_classifier import Config
from owod.detector_proc import SubprocessDetector
from owod.pipeline import PipelineConfig
from owod.state import OWState, UnknownStore
from owod.system import OpenWorldSystem

REPO = str(Path(__file__).resolve().parents[1])


def engine_args(p):
    p.add_argument("--imajev-dir", default=str(Path.home() / "imajev")); p.add_argument("--rotations", type=int, default=1)


def make_system(a, detect_fn):
    """Builds imajev AFTER the detector has finished, so both models are never on the GPU together."""
    from owod.imajev_engine import ImajevEngine
    eng = ImajevEngine(a.imajev_dir, rotations=a.rotations)
    state = OWState.load(a.state)
    store = UnknownStore(a.store)
    pc = PipelineConfig(max_crops=getattr(a, "top", 30), check_all_parts=True)
    return OpenWorldSystem(detect_fn, eng, state, store, Config(verify=True), pc, max_options=eng.max_options,
                           state_path=a.state), state


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--images", required=True); r.add_argument("--state", required=True)
    r.add_argument("--store", required=True); r.add_argument("--classes-file")
    r.add_argument("--pf-dir", required=True); r.add_argument("--detector-python", required=True)
    r.add_argument("--top", type=int, default=30); r.add_argument("--scale", type=int, nargs=2, default=[800, 1333]); r.add_argument("--report", default=None); engine_args(r)
    l = sub.add_parser("list"); l.add_argument("--store", required=True); l.add_argument("--status", default="unknown")
    n = sub.add_parser("learn"); n.add_argument("--state", required=True); n.add_argument("--store", required=True)
    n.add_argument("--name", required=True); n.add_argument("--ids", default=""); n.add_argument("--no-relabel", action="store_true")
    engine_args(n)
    a = ap.parse_args()

    if a.cmd == "list":
        st = UnknownStore(a.store)
        for rid in st.ids(None if a.status == "all" else a.status):
            x = st.get(rid)
            print(rid, x["status"], x["label"], x["reason"], f"{x['unknown_score']:.2f}", x["image"], [round(v) for v in x["box"][:4]], x["crop"])
        return
    if a.cmd == "run":
        if not Path(a.state).exists():
            s = OWState()
            for c in Path(a.classes_file).read_text().splitlines():
                if c.strip():
                    s.add_class(c.strip(), note="initial")
            Path(a.state).parent.mkdir(parents=True, exist_ok=True); s.save(a.state)
        root = Path(a.images)
        files = sorted(str(f) for f in ([root] if root.is_file() else root.iterdir()) if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
        det = SubprocessDetector(a.detector_python, a.pf_dir, REPO, scale=a.scale)
        boxes = det(files)                      # 1) detector subprocess runs and EXITS (frees the GPU)
        system, _ = make_system(a, lambda paths: boxes)    # 2) only then load imajev
        out = system.process_many(files)
        rep = {}
        for path, res in out.items():
            vis = [r for r in res if r.part_of is None]
            rep[path] = {"crops": len(res), "parts_suppressed": len(res) - len(vis),
                         "known": [(r.verdict.label, round(r.verdict.unknown_score, 2)) for r in vis if not r.verdict.is_unknown],
                         "unknown": [(r.verdict.reason, round(r.verdict.unknown_score, 2)) for r in vis if r.verdict.is_unknown]}
            print(Path(path).name, "known", len(rep[path]["known"]), "unknown", len(rep[path]["unknown"]), "parts", rep[path]["parts_suppressed"])
        Path(a.report or Path(a.store).parent / "report.json").write_text(json.dumps(rep, indent=1))
        return
    if a.cmd == "learn":
        system, state = make_system(a, lambda paths: {})
        ids = [int(x) for x in a.ids.split(",") if x.strip()]
        res = system.learn(a.name, ids, relabel=not a.no_relabel)
        state.save(a.state)
        print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
