"""Frame sampling and annotation hold for the video script. Pure Python."""
from typing import List, Optional


def sample_indices(n_frames: int, fps: float, every_sec: float, max_seconds: Optional[float] = None) -> List[int]:
    """Frame indices to process: every `every_sec` seconds, always starting at frame 0, optionally only the first `max_seconds`."""
    if n_frames <= 0 or fps <= 0 or every_sec <= 0:
        raise ValueError("n_frames, fps and every_sec must be positive")
    step = max(1, round(fps * every_sec))
    end = n_frames if max_seconds is None else min(n_frames, int(max_seconds * fps))
    return list(range(0, end, step))


def hold_map(n_frames: int, samples: List[int]) -> List[Optional[int]]:
    """For each output frame, the position in `samples` of the last processed frame at or before it
    (the annotations of that frame are drawn until the next processed frame). None before the first sample."""
    out, j = [], -1
    for i in range(n_frames):
        while j + 1 < len(samples) and samples[j + 1] <= i:
            j += 1
        out.append(j if j >= 0 else None)
    return out
