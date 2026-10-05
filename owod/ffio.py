"""Video input and output through the ffmpeg command line (no OpenCV, no NumPy). Frames are PIL images."""
import json
import subprocess
from fractions import Fraction
from typing import Iterator, Optional


def _rate(text) -> float:
    try:
        return float(Fraction(text or "0/1"))
    except (ZeroDivisionError, ValueError):
        return 0.0


def parse_probe(info: dict) -> dict:
    """ffprobe -show_streams -show_format JSON -> {width, height, fps, n_frames, duration}. Handles rotation and nb_frames=N/A."""
    v = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
    if v is None:
        raise ValueError("no video stream")
    w, h = int(v["width"]), int(v["height"])
    rot = int(float(v.get("tags", {}).get("rotate", 0) or 0))
    for sd in v.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(float(sd["rotation"]))
    if abs(rot) % 180 == 90:                                   # ffmpeg rotates on decode, so the frames are h x w
        w, h = h, w
    fps = _rate(v.get("avg_frame_rate")) or _rate(v.get("r_frame_rate"))
    dur = float(v.get("duration") or info.get("format", {}).get("duration") or 0)
    nb = v.get("nb_frames")
    n = int(nb) if nb not in (None, "N/A", "") else int(round(dur * fps))
    if fps <= 0 or n <= 0:
        raise ValueError("cannot determine fps or frame count")
    return {"width": w, "height": h, "fps": fps, "n_frames": n, "duration": dur or n / fps}


def probe(path: str) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams", "-show_format",
                          "-of", "json", path], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {out.stderr[-500:]}")
    return parse_probe(json.loads(out.stdout))


def read_frames(path: str, width: int, height: int, max_frames: Optional[int] = None) -> Iterator:
    from PIL import Image
    cmd = ["ffmpeg", "-loglevel", "error", "-i", path] + (["-frames:v", str(max_frames)] if max_frames else []) + \
          ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    size = width * height * 3
    try:
        while True:
            buf = p.stdout.read(size)
            if len(buf) < size:
                break
            yield Image.frombytes("RGB", (width, height), buf)
    finally:
        p.stdout.close()
        p.terminate()
        p.wait()


class VideoWriter:
    """H.264 mp4 through an ffmpeg pipe."""
    def __init__(self, path: str, width: int, height: int, fps: float):
        self.size = (width, height)
        self.p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                   "-s", f"{width}x{height}", "-r", f"{fps:.5f}", "-i", "-", "-c:v", "libx264",
                                   "-pix_fmt", "yuv420p", "-movflags", "+faststart", path], stdin=subprocess.PIPE)

    def write(self, img) -> None:
        if img.size != self.size:
            raise ValueError(f"frame size {img.size} != {self.size}")
        self.p.stdin.write(img.convert("RGB").tobytes())

    def close(self) -> None:
        self.p.stdin.close()
        if self.p.wait() != 0:
            raise RuntimeError("ffmpeg writer failed")
