"""Unit tests for the exp3 portfolio-backtest building blocks, using small
synthetic covariance matrices and return panels (no network access)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from exp.exp3_portfolio_backtest import STRATEGIES, _strategy_weights


def _synthetic_block_covariance(p: int = 8, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((p, p))
    Sigma = A @ A.T + p * np.eye(p)
    return Sigma


def test_strategy_weights_all_long_only_and_sum_to_one():
    p = 8
    Sigma_S = _synthetic_block_covariance(p, seed=0)
    Sigma_LW = _synthetic_block_covariance(p, seed=1)
    Sigma_B = _synthetic_block_covariance(p, seed=2)
    Sigma_R = _synthetic_block_covariance(p, seed=3)
    W = np.abs(np.random.default_rng(4).standard_normal((p, p)))
    W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0.0)
    rng = np.random.default_rng(5)
    simple_window = pd.DataFrame(rng.standard_normal((100, p)) * 0.01,
                                  columns=[f"T{i}" for i in range(p)])

    weights = _strategy_weights(p, Sigma_S, Sigma_LW, Sigma_B, Sigma_R, W, simple_window)

    assert set(weights.keys()) == set(STRATEGIES)
    for name, w in weights.items():
        assert w.shape == (p,)
        assert np.all(w >= -1e-6), f"{name} has negative weight"
        assert w.sum() == pytest.approx(1.0, abs=1e-3), f"{name} weights do not sum to 1"


def test_equal_weight_strategy_is_uniform():
    p = 6
    Sigma = _synthetic_block_covariance(p, seed=10)
    W = np.zeros((p, p))
    rng = np.random.default_rng(11)
    simple_window = pd.DataFrame(rng.standard_normal((60, p)) * 0.01,
                                  columns=[f"T{i}" for i in range(p)])
    weights = _strategy_weights(p, Sigma, Sigma, Sigma, Sigma, W, simple_window)
    assert np.allclose(weights["Equal Weight"], 1.0 / p)
