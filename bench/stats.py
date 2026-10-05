"""Pure-Python latency statistics and the SPEC-001 gate. No torch needed."""
import math
from typing import Dict, List


def percentile(values: List[float], q: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    s = sorted(values)
    k = (len(s) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def summarize(latencies_ms: List[float]) -> Dict[str, float]:
    return {
        "median_ms": percentile(latencies_ms, 0.5),
        "p90_ms": percentile(latencies_ms, 0.9),
        "min_ms": min(latencies_ms),
        "n": len(latencies_ms),
    }


def speed_ratio(candidate_ms: float, baseline_ms: float) -> float:
    if baseline_ms <= 0:
        raise ValueError("baseline latency must be positive")
    return candidate_ms / baseline_ms


def gate(candidate_ms: float, baseline_ms: float, ar100: float,
         max_ratio: float = 1.10, ar100_floor: float = 52.3) -> Dict[str, object]:
    ratio = speed_ratio(candidate_ms, baseline_ms)
    speed_ok = ratio <= max_ratio
    quality_ok = ar100 >= ar100_floor
    return {"ratio": ratio, "speed_ok": speed_ok, "quality_ok": quality_ok,
            "pass": speed_ok and quality_ok}
