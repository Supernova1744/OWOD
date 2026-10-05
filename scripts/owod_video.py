"""Run the open-world system on a video and save an annotated video (imajev venv). UNTESTED on GPU.
  python scripts/owod_video.py --video in.mp4 --out results/video/in_annotated.mp4 --classes-file results/voc20.txt \
      --pf-dir /mnt/d/OWOD/OWOD/PF-RPN --detector-python $HOME/.venvs/pf-rpn/bin/python --every-sec 1 --max-seconds 30 --top 15
Frames are processed every --every-sec seconds (the system needs seconds per frame); the boxes of the last processed frame
are held until the next one. No tracking. Needs opencv (pip install opencv-python-headless in the imajev venv)."""
import argparse, json, os, shutil, subprocess, sys, tempfile, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pathlib import Path
from PIL import Image
from owod.crop_classifier import Config
from owod.detector_proc import SubprocessDetector
from owod.draw import draw_results
from owod.pipeline import PipelineConfig
from owod.state import OWState, UnknownStore
from owod.system import OpenWorldSystem
from owod.video import hold_map, sample_indices

REPO = str(Path(__file__).resolve().parents[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--classes-file", required=True)
    ap.add_argument("--pf-dir", required=True); ap.add_argument("--detector-python", required=True)
    ap.add_argument("--imajev-dir", default=str(Path.home() / "imajev")); ap.add_argument("--rotations", type=int, default=1)
    ap.add_argument("--every-sec", type=float, default=1.0); ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--top", type=int, default=15); ap.add_argument("--scale", type=int, nargs=2, default=[800, 1333])
    ap.add_argument("--work", default=None, help="work folder (state, store, frames); default: next to --out")
    ap.add_argument("--dry-run", action="store_true", help="only print video info and the time estimate")
    a = ap.parse_args()
    import cv2

    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {a.video}")
    n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS) or 25.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    idx = sample_indices(n, fps, a.every_sec, a.max_seconds)
    end = n if a.max_seconds is None else min(n, int(a.max_seconds * fps))
    est = len(idx) * (a.top * 0.5 + 2)
    print(f"video {W}x{H} {fps:.2f} fps {n} frames ({n / fps:.1f} s); processing {len(idx)} frames up to frame {end}; "
          f"rough estimate {est / 60:.1f} min (about {a.top * 0.5 + 2:.0f} s per frame)", flush=True)
    if a.dry_run:
        return

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    work = Path(a.work) if a.work else out.parent / (out.stem + "_work")
    shutil.rmtree(work, ignore_errors=True); (work / "frames").mkdir(parents=True)
    wanted = set(idx); paths = {}
    i = 0
    while i < end:
        ok, frame = cap.read()
        if not ok:
            break
        if i in wanted:
            p = work / "frames" / f"f{i:07d}.png"
            Image.fromarray(frame[:, :, ::-1]).save(p); paths[i] = str(p)
        i += 1
    cap.release()
    samples = sorted(paths)

    t0 = time.time()
    det = SubprocessDetector(a.detector_python, a.pf_dir, REPO, scale=a.scale)
    boxes = det([paths[k] for k in samples])                         # detector first; it exits before imajev loads
    print(f"detection {time.time() - t0:.0f} s", flush=True)

    from owod.imajev_engine import ImajevEngine
    eng = ImajevEngine(a.imajev_dir, rotations=a.rotations)
    state = OWState()
    for c in Path(a.classes_file).read_text().splitlines():
        if c.strip():
            state.add_class(c.strip(), note="initial")
    system = OpenWorldSystem(lambda ps: boxes, eng, state, UnknownStore(work / "store"), Config(verify=True),
                             PipelineConfig(max_crops=a.top, check_all_parts=True), max_options=eng.max_options)
    results = {}
    for n_done, k in enumerate(samples, 1):
        t1 = time.time()
        results[k] = system.process_image(paths[k])
        r = results[k]
        print(f"frame {k} ({n_done}/{len(samples)}): {sum(not x.verdict.is_unknown and x.part_of is None for x in r)} known, "
              f"{sum(x.verdict.is_unknown and x.part_of is None for x in r)} unknown, {time.time() - t1:.1f} s", flush=True)

    # write the annotated video at the original fps; each frame shows the annotations of the last processed frame
    cap = cv2.VideoCapture(a.video)
    tmp = out.with_suffix(".tmp.mp4")
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    hm = hold_map(end, samples)
    for i in range(end):
        ok, frame = cap.read()
        if not ok:
            break
        j = hm[i]
        img = Image.fromarray(frame[:, :, ::-1])
        if j is not None:
            k = samples[j]
            img = draw_results(img, results[k], f"green=known red=unknown | processed frame {k} ({k / fps:.1f}s)")
        vw.write(cv2.cvtColor(__import__("numpy").array(img), cv2.COLOR_RGB2BGR))
    vw.release(); cap.release()
    if shutil.which("ffmpeg"):                                   # re-encode to H.264 so common players open it
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)], check=True)
        tmp.unlink()
    else:
        tmp.replace(out)
    summary = {"video": a.video, "frames_processed": len(samples), "every_sec": a.every_sec,
               "per_frame": {str(k): {"known": [(x.verdict.label) for x in results[k] if not x.verdict.is_unknown and x.part_of is None],
                                      "unknown": [(x.verdict.reason) for x in results[k] if x.verdict.is_unknown and x.part_of is None]}
                             for k in samples}, "total_seconds": round(time.time() - t0)}
    out.with_suffix(".json").write_text(json.dumps(summary, indent=1))
    print("saved", out, "and", out.with_suffix(".json"), flush=True)


if __name__ == "__main__":
    main()
