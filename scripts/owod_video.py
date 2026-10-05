"""Run the open-world system on a video and save an annotated video (imajev venv). UNTESTED on GPU.
  python scripts/owod_video.py --video in.mp4 --out results/video/in_annotated.mp4 --classes-file results/voc20.txt \
      --pf-dir /mnt/d/OWOD/OWOD/PF-RPN --detector-python $HOME/.venvs/pf-rpn/bin/python --every-sec 1 --max-seconds 30 --top 15
Frames are processed every --every-sec seconds (the system needs seconds per frame); the boxes of the last processed frame
are held until the next one. No tracking. Uses the ffmpeg command line for video input and output (no OpenCV)."""
import argparse, json, os, shutil, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pathlib import Path
from PIL import Image
from owod.crop_classifier import Config
from owod.detector_proc import SubprocessDetector
from owod.draw import draw_results
from owod.ffio import VideoWriter, probe, read_frames
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
    info = probe(a.video)
    n, fps, W, H = info["n_frames"], info["fps"], info["width"], info["height"]
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
    for i, frame in enumerate(read_frames(a.video, W, H, max_frames=end)):
        if i in wanted:
            p = work / "frames" / f"f{i:07d}.png"
            frame.save(p); paths[i] = str(p)
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
    hm = hold_map(end, samples)
    vw = VideoWriter(str(out), W, H, fps)
    for i, frame in enumerate(read_frames(a.video, W, H, max_frames=end)):
        j = hm[i]
        if j is not None:
            k = samples[j]
            frame = draw_results(frame, results[k], f"green=known red=unknown | processed frame {k} ({k / fps:.1f}s)")
        vw.write(frame)
    vw.close()
    summary = {"video": a.video, "frames_processed": len(samples), "every_sec": a.every_sec,
               "per_frame": {str(k): {"known": [(x.verdict.label) for x in results[k] if not x.verdict.is_unknown and x.part_of is None],
                                      "unknown": [(x.verdict.reason) for x in results[k] if x.verdict.is_unknown and x.part_of is None]}
                             for k in samples}, "total_seconds": round(time.time() - t0)}
    out.with_suffix(".json").write_text(json.dumps(summary, indent=1))
    print("saved", out, "and", out.with_suffix(".json"), flush=True)


if __name__ == "__main__":
    main()
