"""GPU timing helper (needs torch). Warm-up, sync, N timed runs."""
import time


def time_fn(fn, warmup: int = 10, runs: int = 50):
    import torch
    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    for _ in range(warmup):
        fn()
    sync()
    out = []
    for _ in range(runs):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        out.append((time.perf_counter() - t0) * 1000.0)
    return out
