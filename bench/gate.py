"""Usage: python bench/gate.py baseline.json candidate.json [--floor 52.3] [--max-ratio 1.10]"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bench.stats import gate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("baseline"); p.add_argument("candidate")
    p.add_argument("--floor", type=float, default=52.3)
    p.add_argument("--max-ratio", type=float, default=1.10)
    p.add_argument("--quality-ref", help="JSON with ar100 of the full model on the same images; floor = ratio x its ar100")
    p.add_argument("--candidate-ar", help="JSON from eval_ar.py for the candidate (default: ar100 inside the candidate speed JSON)")
    p.add_argument("--quality-ratio", type=float, default=0.98)
    p.add_argument("--speed-only", action="store_true", help="exploration only: ignore quality")
    a = p.parse_args()
    b, c = json.load(open(a.baseline)), json.load(open(a.candidate))
    floor = a.floor
    if a.quality_ref:
        floor = json.load(open(a.quality_ref))["ar100"] * a.quality_ratio
    g = gate(c["latency"]["median_ms"], b["latency"]["median_ms"],
             float("inf") if a.speed_only else (json.load(open(a.candidate_ar))["ar100"] if a.candidate_ar else c.get("ar100", float("-inf"))), a.max_ratio, floor)
    print(json.dumps(g, indent=2))
    print("PASS" if g["pass"] else "FAIL")
    sys.exit(0 if g["pass"] else 1)


if __name__ == "__main__":
    main()
