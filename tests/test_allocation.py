"""Unit tests for long-only allocation strategies (src/allocation.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.allocation import equal_weight, hrp_weights, mvo_min_variance


def test_equal_weight_sums_to_one_and_uniform():
    w = equal_weight(5)
    assert w.shape == (5,)
    assert w.sum() == pytest.approx(1.0)
    assert np.allclose(w, 0.2)


def test_mvo_min_variance_long_only_budget_constraint():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((5, 5))
    Sigma = A @ A.T + np.eye(5)
    w = mvo_min_variance(Sigma)
    assert w.shape == (5,)
    assert np.all(w >= -1e-6)
    assert w.sum() == pytest.approx(1.0, abs=1e-4)


def test_mvo_min_variance_favours_low_variance_assets():
    Sigma = np.diag([1.0, 1.0, 100.0])
    w = mvo_min_variance(Sigma)
    assert w[2] < w[0]
    assert w[2] < w[1]


def test_hrp_weights_sum_to_one_and_nonnegative():
    rng = np.random.default_rng(1)
    returns = pd.DataFrame(rng.standard_normal((200, 6)) * 0.01, columns=list("ABCDEF"))
    w = hrp_weights(returns)
    assert w.shape == (6,)
    assert np.all(w >= -1e-8)
    assert w.sum() == pytest.approx(1.0, abs=1e-4)
