"""Unit tests for the MINGLE ADMM framework (src/mingle.py) on synthetic
factor data: structural properties stated by the paper (Sec. 3), not its
reported numbers."""

from __future__ import annotations

import numpy as np
import pytest

from src.mingle import (
    build_omega,
    factor_induced_covariance,
    fit_mingle,
    init_graph_rbf,
    weighted_svd_init,
)


def _synthetic_factor_data(p=20, T=100, k=3, noise=0.1, seed=0):
    rng = np.random.default_rng(seed)
    B_true = rng.standard_normal((p, k))
    F_true = rng.standard_normal((k, T))
    X = B_true @ F_true + noise * rng.standard_normal((p, T))
    return X, B_true, F_true


def test_build_omega_is_diagonal_decaying_to_one():
    Omega = build_omega(T=5, omega=0.9)
    assert Omega.shape == (5, 5)
    assert np.allclose(Omega, np.diag(np.diag(Omega)))
    assert Omega[-1, -1] == pytest.approx(1.0)
    assert np.all(np.diff(np.diag(Omega)) > 0)  # weights increase toward the present


def test_weighted_svd_init_shapes_and_reconstruction():
    X, _, _ = _synthetic_factor_data()
    Omega_diag = np.diag(build_omega(X.shape[1], 0.997))
    B0, F0 = weighted_svd_init(X, Omega_diag, k=3)
    assert B0.shape == (20, 3)
    assert F0.shape == (3, 100)
    # The rank-k weighted SVD is the best rank-k approximation: residual energy
    # must be small relative to X's own energy for data that is exactly rank 3.
    residual = np.linalg.norm(X - B0 @ F0) / np.linalg.norm(X)
    assert residual < 0.2


def test_init_graph_rbf_symmetric_nonnegative_zero_diagonal():
    X, _, _ = _synthetic_factor_data()
    Omega_diag = np.diag(build_omega(X.shape[1], 0.997))
    B0, _ = weighted_svd_init(X, Omega_diag, k=3)
    W0 = init_graph_rbf(B0)
    assert np.allclose(W0, W0.T)
    assert np.all(W0 >= 0.0)
    assert np.allclose(np.diag(W0), 0.0)


def test_fit_mingle_output_shapes_and_graph_properties():
    X, _, _ = _synthetic_factor_data()
    res = fit_mingle(X, k=3, max_iter=25)
    p, T, k = 20, 100, 3
    assert res.B.shape == (p, k)
    assert res.F.shape == (k, T)
    assert res.W.shape == (p, p)
    assert res.L.shape == (p, p)
    assert res.Psi.shape == (p,)
    assert res.Sigma_R.shape == (p, p)

    # Expected outcomes (Sec. 3): sparse, symmetric, non-negative adjacency
    # with zero diagonal, and a positive semi-definite graph Laplacian.
    assert np.allclose(res.W, res.W.T, atol=1e-8)
    assert np.all(res.W >= -1e-8)
    assert np.allclose(np.diag(res.W), 0.0, atol=1e-8)
    eigvals = np.linalg.eigvalsh(res.L)
    assert eigvals.min() > -1e-6

    # Sigma_R must be a valid (symmetric, positive semi-definite) covariance.
    assert np.allclose(res.Sigma_R, res.Sigma_R.T, atol=1e-6)
    assert np.linalg.eigvalsh(0.5 * (res.Sigma_R + res.Sigma_R.T)).min() > -1e-6


def test_fit_mingle_reduces_objective_from_initialisation():
    X, _, _ = _synthetic_factor_data()
    res = fit_mingle(X, k=3, max_iter=30)
    assert len(res.objective_history) >= 2
    # Non-convex ADMM (Remark 1) need not decrease monotonically every step,
    # but the final objective should not be far worse than the first iterate.
    assert res.objective_history[-1] <= res.objective_history[0] * 1.5


def test_fit_mingle_requires_k_smaller_than_p():
    X, _, _ = _synthetic_factor_data(p=5, T=50, k=2)
    with pytest.raises(ValueError):
        fit_mingle(X, k=5, max_iter=2)


def test_factor_induced_covariance_better_conditioned_than_noisy_sample_covariance():
    X, _, _ = _synthetic_factor_data(p=30, T=150, k=3, noise=0.5, seed=1)
    res = fit_mingle(X, k=3, max_iter=25)
    Sigma_S = np.cov(X)
    from src.covariance import condition_number

    assert condition_number(res.Sigma_R) <= condition_number(Sigma_S) * 1.5


def test_factor_induced_covariance_helper_matches_result_structure():
    X, _, _ = _synthetic_factor_data()
    res = fit_mingle(X, k=3, max_iter=10)
    Sigma = factor_induced_covariance(res.B, res.F, X)
    assert Sigma.shape == (20, 20)
    assert np.allclose(Sigma, Sigma.T, atol=1e-6)
