"""Ablation variants of the MINGLE objective (Fig. 7; Sec. "Ablation Study").

Only the "No low-rank prior" (No-LR) variant needs new optimisation code: it
removes the ``Y = BF`` low-rank constraint (Eq. 2's unconstrained vertex
representation) while keeping the graph-smoothness, sparsity and degree terms
jointly learnt with the same W/V/d Sherman-Morrison, soft-threshold and
quadratic-root updates as the full model (``src.mingle``). The "No
decorrelation" variant is simply ``fit_mingle`` with ``delta=0`` and the
"Sequential" variant is standard PCA plus a fixed RBF graph
(``src.covariance.pca_factor_covariance`` and ``src.mingle.init_graph_rbf``);
neither needs new producer code, so only this module is added.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import solve_sylvester

from src import config
from src.mingle import _d_update, _laplacian, _v_update, _w_update, init_graph_rbf


@dataclass
class NoLRResult:
    """Output of the No-LR ablation fit."""

    Y: np.ndarray          # p x T unconstrained (full-rank) denoised representation
    W: np.ndarray          # p x p graph topology
    L: np.ndarray          # p x p graph Laplacian
    Sigma: np.ndarray      # p x p induced covariance, Omega-weighted Cov(Y)
    objective_history: list = field(default_factory=list)
    primal_residual_history: list = field(default_factory=list)
    n_iter: int = 0
    converged: bool = False


def _no_lr_objective(X, Y, W, Omega_diag, lam, alpha, beta, eps) -> float:
    recon = float(np.sum(((X - Y) ** 2) @ Omega_diag))
    sq_norms = np.sum(Y ** 2, axis=1)
    D = sq_norms[:, None] + sq_norms[None, :] - 2 * Y @ Y.T
    smooth = lam * float(np.sum(W * np.maximum(D, 0.0)))
    sparsity = alpha * float(np.sum(np.abs(W)))
    degree_arg = np.maximum(W.sum(axis=1) + eps, 1e-12)
    degree = -beta * float(np.sum(np.log(degree_arg)))
    return recon + smooth + sparsity + degree


def fit_no_lr(
    X: np.ndarray,
    Omega_diag: np.ndarray,
    lam: float = config.LAMBDA_SMOOTH,
    alpha: float = config.ALPHA_SPARSITY,
    beta: float = config.BETA_DEGREE,
    rho_W: float = config.RHO_W,
    rho_d: float = config.RHO_D,
    eps: float = config.EPS_DEGREE,
    max_iter: int = config.ADMM_MAX_ITER,
    tol: float = config.ADMM_TOL,
) -> NoLRResult:
    """Fit ``min_{Y,W} ||X-Y||_Omega^2 + lambda*tr(Y^T L Y) - beta*1^Tlog(d+eps*1)
    + alpha*||W||_1`` subject to ``W=V>=0, W1=d, V^T=V, diag(V)=0`` (removes
    the ``Y=BF`` low-rank prior of the full MINGLE objective, Eq. 5).

    The Y-subproblem (holding W fixed) is quadratic: setting its gradient to
    zero, ``-2(X-Y)Omega + 2*lambda*L*Y = 0``, i.e. the Sylvester equation
    ``lambda*L*Y + Y*Omega = X*Omega``, solved exactly each iteration. The
    W/V/d updates reuse the full model's Sherman-Morrison, soft-threshold and
    quadratic-root subproblems (``src.mingle``), with the pairwise distances
    ``||y_i-y_j||^2`` taken over Y's rows in place of the exposure rows of B,
    since Y now plays the role of the per-asset representation that defines
    graph locality.
    """
    p, T = X.shape
    Y = X.copy()
    W = init_graph_rbf(Y)
    V = W.copy()
    d = W.sum(axis=1)
    Lambda_W = np.zeros_like(W)
    Lambda_d = np.zeros_like(d)

    objective_history, primal_history = [], []
    converged = False
    n_iter = 0
    for it in range(max_iter):
        L = _laplacian(W)
        Y = solve_sylvester(lam * L, np.diag(Omega_diag), X * Omega_diag[None, :])

        W = _w_update(Y, V, d, Lambda_W, Lambda_d, lam, rho_W, rho_d)
        d = _d_update(W, Lambda_d, beta, rho_d, eps)
        V = _v_update(W, Lambda_W, alpha, rho_W)

        Lambda_W = Lambda_W + rho_W * (W - V)
        Lambda_d = Lambda_d + rho_d * (W.sum(axis=1) - d)

        primal_res = np.linalg.norm(W - V, "fro") + np.linalg.norm(W.sum(axis=1) - d)
        obj = _no_lr_objective(X, Y, W, Omega_diag, lam, alpha, beta, eps)
        objective_history.append(obj)
        primal_history.append(primal_res)
        n_iter = it + 1
        if primal_res < tol:
            converged = True
            break

    W_final = V
    L = _laplacian(W_final)
    sqrt_omega = np.sqrt(Omega_diag)
    Sigma = np.cov(Y * sqrt_omega[None, :])
    return NoLRResult(
        Y=Y, W=W_final, L=L, Sigma=Sigma,
        objective_history=objective_history, primal_residual_history=primal_history,
        n_iter=n_iter, converged=converged,
    )
