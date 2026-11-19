"""Unit tests for the No-LR ablation variant (src/mingle_variants.py)."""

from __future__ import annotations

import numpy as np

from src import config
from src.mingle import build_omega
from src.mingle_variants import fit_no_lr


def _synthetic_returns(p=12, T=80, seed=0):
    rng = np.random.default_rng(seed)
    k = 3
    B = rng.normal(size=(p, k)) * 0.1
    F = rng.normal(size=(k, T)) * 0.05
    noise = rng.normal(size=(p, T)) * 0.01
    return B @ F + noise


def test_no_lr_runs_and_shapes():
    p, T = 12, 80
    X = _synthetic_returns(p, T)
    Omega = np.diag(build_omega(T, config.OMEGA_DECAY))
    result = fit_no_lr(X, Omega, max_iter=15)
    assert result.Y.shape == (p, T)
    assert result.W.shape == (p, p)
    assert result.Sigma.shape == (p, p)
    assert result.n_iter > 0


def test_no_lr_graph_properties():
    p, T = 12, 80
    X = _synthetic_returns(p, T)
    Omega = np.diag(build_omega(T, config.OMEGA_DECAY))
    result = fit_no_lr(X, Omega, max_iter=15)
    assert np.allclose(result.W, result.W.T)
    assert np.allclose(np.diag(result.W), 0.0)
    assert np.all(result.W >= -1e-10)
    laplacian_eigs = np.linalg.eigvalsh(0.5 * (result.L + result.L.T))
    assert laplacian_eigs.min() > -1e-8


def test_no_lr_objective_decreases_overall():
    p, T = 12, 80
    X = _synthetic_returns(p, T)
    Omega = np.diag(build_omega(T, config.OMEGA_DECAY))
    result = fit_no_lr(X, Omega, max_iter=25)
    assert result.objective_history[-1] < result.objective_history[0]


def test_no_lr_sigma_is_symmetric_and_finite():
    p, T = 10, 60
    X = _synthetic_returns(p, T)
    Omega = np.diag(build_omega(T, config.OMEGA_DECAY))
    result = fit_no_lr(X, Omega, max_iter=10)
    assert np.allclose(result.Sigma, result.Sigma.T)
    assert np.all(np.isfinite(result.Sigma))
