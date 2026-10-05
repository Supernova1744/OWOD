import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
from bench.stats import percentile, summarize, speed_ratio, gate


def test_percentile_median_odd_and_even():
    assert percentile([3, 1, 2], 0.5) == 2
    assert percentile([1, 2, 3, 4], 0.5) == 2.5


def test_percentile_empty_raises():
    with pytest.raises(ValueError):
        percentile([], 0.5)


def test_summarize_fields():
    s = summarize([10.0, 20.0, 30.0])
    assert s["median_ms"] == 20.0 and s["min_ms"] == 10.0 and s["n"] == 3


def test_speed_ratio_and_zero_baseline():
    assert speed_ratio(11.0, 10.0) == pytest.approx(1.1)
    with pytest.raises(ValueError):
        speed_ratio(1.0, 0.0)


def test_gate_pass():
    assert gate(10.5, 10.0, 55.0)["pass"] is True


def test_gate_slow_fails():
    g = gate(20.0, 10.0, 60.0)
    assert g["speed_ok"] is False and g["pass"] is False


def test_gate_low_quality_fails_even_if_fast():  # T2
    g = gate(5.0, 10.0, 40.0)
    assert g["speed_ok"] is True and g["quality_ok"] is False and g["pass"] is False
