"""Usage: python bench/gate.py baseline.json candidate.json [--floor 52.3] [--max-ratio 1.10]"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bench.stats import gate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("baseline"); p.add_argument("candidate")
    p.add_argument("--floor", type=float, default=52.3)
    p.add_argument("--max-ratio", type=float, default=1.10)
    a = p.parse_args()
    b, c = json.load(open(a.baseline)), json.load(open(a.candidate))
    g = gate(c["latency"]["median_ms"], b["latency"]["median_ms"],
             c.get("ar100", float("-inf")), a.max_ratio, a.floor)
    print(json.dumps(g, indent=2))
    print("PASS" if g["pass"] else "FAIL")
    sys.exit(0 if g["pass"] else 1)


if __name__ == "__main__":
    main()
