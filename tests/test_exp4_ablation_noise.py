"""Lightweight unit tests for exp/exp4_ablation_noise.py helpers.

These tests use small synthetic inputs and do not call the network/data
loader; they must not call ``run()`` (which needs live market data).
"""

from __future__ import annotations

import numpy as np
import pytest

from exp.exp4_ablation_noise import boxplot_stats, calibrate_noise_sigma


def test_calibrate_noise_sigma_decreases_with_higher_snr() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5, 50))
    sigma_low_snr = calibrate_noise_sigma(X, 10.0)
    sigma_high_snr = calibrate_noise_sigma(X, 40.0)
    assert sigma_high_snr < sigma_low_snr


def test_calibrate_noise_sigma_matches_target_snr() -> None:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(7, 100)) * 0.02
    for snr_db in (40.0, 20.0, 10.0):
        sigma = calibrate_noise_sigma(X, snr_db)
        recovered_snr = 10 * np.log10(np.mean(X ** 2) / sigma ** 2)
        assert recovered_snr == pytest.approx(snr_db)


def test_boxplot_stats_known_array() -> None:
    stats = boxplot_stats([1, 2, 3, 4, 5])
    assert stats["median"] == pytest.approx(3.0)
    assert stats["mean"] == pytest.approx(3.0)
    assert stats["q1"] == pytest.approx(2.0)
    assert stats["q3"] == pytest.approx(4.0)
    assert stats["lower_whisker"] <= stats["q1"]
    assert stats["upper_whisker"] >= stats["q3"]


def test_boxplot_stats_whiskers_bounded_by_data() -> None:
    values = [1.0, 2.0, 2.0, 3.0, 100.0]
    stats = boxplot_stats(values)
    assert stats["lower_whisker"] >= min(values)
    assert stats["upper_whisker"] <= max(values)
