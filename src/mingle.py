"""MINGLE: joint factor-graph learning via ADMM.

Implements the unified objective of Eq. 5, its ADMM split (Eq. 6) and the six
subproblem updates of Algorithm 1 (Chehab, Iacovides, Yazdanparast and
Mandic, "Beyond Co-Movement: Locality by Exposures Enables a Joint
Factor-Graph Framework for Portfolio Diversification", ICAIF'26, Sec. 3).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import solve_sylvester, svd

from src import config


@dataclass
class MingleResult:
    """Output of one MINGLE ADMM fit."""

    B: np.ndarray          # p x k exposure profiles
    F: np.ndarray          # k x T factor dynamics
    W: np.ndarray          # p x p graph topology (sparse, symmetric, >=0, zero diag)
    L: np.ndarray          # p x p graph Laplacian, diag(W1) - W
    Psi: np.ndarray        # p-vector of idiosyncratic variances
    Sigma_R: np.ndarray    # p x p representation-induced covariance
    objective_history: list = field(default_factory=list)
    primal_residual_history: list = field(default_factory=list)
    n_iter: int = 0
    converged: bool = False


def build_omega(T: int, omega: float = config.OMEGA_DECAY) -> np.ndarray:
    """Diagonal exponential time-decay matrix, Omega_tt = omega^(T-t) (Eq.,
    Sec. 4): the most recent observation (t=T) gets weight 1."""
    exponents = np.arange(T - 1, -1, -1)
    return np.diag(omega ** exponents.astype(float))


def _weighted_frobenius_sq(A: np.ndarray, Omega_diag: np.ndarray) -> float:
    """||A||_Omega^2 = tr(A Omega A^T), Omega diagonal given as a vector."""
    return float(np.sum((A ** 2) @ Omega_diag))


def weighted_svd_init(X: np.ndarray, Omega_diag: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Initialise B^0, F^0 via the weighted SVD of Eq. 7: X Omega^(1/2) = U
    Sigma V^T yields B^0 = U[:, :k], F^0 = Sigma[:k,:k] V[:,:k]^T Omega^(-1/2)."""
    sqrt_omega = np.sqrt(Omega_diag)
    inv_sqrt_omega = 1.0 / sqrt_omega
    weighted = X * sqrt_omega[None, :]
    U, S, Vt = svd(weighted, full_matrices=False)
    B0 = U[:, :k]
    F0 = (S[:k, None] * Vt[:k, :]) * inv_sqrt_omega[None, :]
    return B0, F0


def init_graph_rbf(B0: np.ndarray) -> np.ndarray:
    """RBF-kernel graph initialisation (Eq. 8): w_ij = exp(-||b_i-b_j||^2 /
    (2 sigma_hat^2)), sigma_hat^2 = median pairwise squared distance, w_ii=0."""
    sq_norms = np.sum(B0 ** 2, axis=1)
    sq_dist = sq_norms[:, None] + sq_norms[None, :] - 2 * B0 @ B0.T
    sq_dist = np.maximum(sq_dist, 0.0)
    p = B0.shape[0]
    off_diag = sq_dist[~np.eye(p, dtype=bool)]
    sigma2 = np.median(off_diag)
    sigma2 = sigma2 if sigma2 > 0 else 1e-8
    W0 = np.exp(-sq_dist / (2 * sigma2))
    np.fill_diagonal(W0, 0.0)
    return W0


def _laplacian(W: np.ndarray) -> np.ndarray:
    return np.diag(W.sum(axis=1)) - W


def _f_subobjective(
    X: np.ndarray, B: np.ndarray, F: np.ndarray, Omega_diag: np.ndarray, delta: float
) -> float:
    """F-subproblem value ||X-BF||_Omega^2 + delta*||offdiag(F Omega F^T)||_F^2,
    used by the backtracking line search below."""
    recon = _weighted_frobenius_sq(X - B @ F, Omega_diag)
    M = (F * Omega_diag[None, :]) @ F.T
    M_off = M - np.diag(np.diag(M))
    return recon + delta * float(np.sum(M_off ** 2))


def _f_gradient(
    X: np.ndarray, B: np.ndarray, F: np.ndarray, Omega_diag: np.ndarray, delta: float
) -> np.ndarray:
    """Gradient of ||X-BF||_Omega^2 + delta*||offdiag(F Omega F^T)||_F^2 w.r.t. F.

    The reconstruction term's gradient is -2 B^T (X-BF) Omega.
    The decorrelation penalty is delta*||offdiag(F Omega F^T)||_F^2; writing
    M = F Omega F^T and M_off = M - diag(M), its gradient w.r.t. F is
    4 * M_off @ F @ Omega (standard matrix-calculus result for
    d/dF tr(M_off M_off^T) with M = F Omega F^T symmetric).
    """
    residual = X - B @ F
    grad_recon = -2.0 * (B.T @ (residual * Omega_diag[None, :]))
    M = (F * Omega_diag[None, :]) @ F.T
    M_off = M - np.diag(np.diag(M))
    grad_decorr = 4.0 * delta * (M_off @ (F * Omega_diag[None, :]))
    return grad_recon + grad_decorr


def _f_step_backtracking(
    X: np.ndarray, B: np.ndarray, F: np.ndarray, Omega_diag: np.ndarray, delta: float,
    grad: np.ndarray, lr0: float, shrink: float = 0.5, max_backtracks: int = 30,
) -> tuple[np.ndarray, float]:
    """Backtracking (Armijo) line search for the F gradient-descent step.

    The paper specifies a "gradient descent step" for the F update (Sec. 3.2)
    without fixing a step size. A fixed step can diverge because the
    decorrelation term's curvature (scaled by delta) differs by orders of
    magnitude from the reconstruction term's, so the step is instead chosen by
    standard backtracking: shrink the candidate step until the sub-objective
    decreases, which keeps the update a valid (if adaptive) gradient step.
    """
    f0 = _f_subobjective(X, B, F, Omega_diag, delta)
    grad_sq = float(np.sum(grad ** 2))
    lr = lr0
    for _ in range(max_backtracks):
        F_new = F - lr * grad
        if np.all(np.isfinite(F_new)):
            f_new = _f_subobjective(X, B, F_new, Omega_diag, delta)
            if np.isfinite(f_new) and f_new <= f0 - 1e-4 * lr * grad_sq:
                return F_new, lr
        lr *= shrink
    return F, 0.0


def _b_update(
    X: np.ndarray, F: np.ndarray, Omega_diag: np.ndarray, L: np.ndarray,
    Q: np.ndarray, Lambda_B: np.ndarray, lam: float, rho_B: float,
) -> np.ndarray:
    """Sylvester solve for exposure profiles (Algorithm 1, B update):
    (2 lambda L + rho_B I) B + 2 B (F Omega F^T) = 2 X Omega F^T + rho_B Q - Lambda_B."""
    p = X.shape[0]
    FOFt = (F * Omega_diag[None, :]) @ F.T
    A = 2.0 * lam * L + rho_B * np.eye(p)
    C = 2.0 * (X * Omega_diag[None, :]) @ F.T + rho_B * Q - Lambda_B
    return solve_sylvester(A, 2.0 * FOFt, C)


def _q_update(B: np.ndarray, Lambda_B: np.ndarray, rho_B: float) -> np.ndarray:
    """Stiefel projection via polar decomposition: SVD(B+Lambda_B/rho_B)=U S
    V^T, Q = U V^T."""
    M = B + Lambda_B / rho_B
    U, _, Vt = svd(M, full_matrices=False)
    return U @ Vt


def _w_update(
    B: np.ndarray, V: np.ndarray, d: np.ndarray, Lambda_W: np.ndarray,
    Lambda_d: np.ndarray, lam: float, rho_W: float, rho_d: float,
) -> np.ndarray:
    """Sherman-Morrison solve for the graph topology (Algorithm 1, W update,
    ref. [15]).

    Holding B, V, d, Lambda_W, Lambda_d fixed, the W-subproblem of the
    augmented Lagrangian of Eq. 6 is
        lambda * <D, W> + (rho_W/2)||W-V+Lambda_W/rho_W||_F^2
                        + (rho_d/2)||W1-d+Lambda_d/rho_d||_2^2,
    with D_ij = ||b_i-b_j||_2^2. Its stationarity condition is
        rho_W W + rho_d (W1) 1^T = rho_W(V-Lambda_W/rho_W) - lambda*D
                                    + rho_d(d-Lambda_d/rho_d) 1^T = C,
    i.e. W (rho_W I + rho_d * 1 1^T) = C. Applying the Sherman-Morrison
    formula to invert the rank-one update rho_W I + rho_d 1 1^T gives the
    closed form below; W is symmetrised afterwards since C need not be
    symmetric.
    """
    p = B.shape[0]
    sq_norms = np.sum(B ** 2, axis=1)
    D = sq_norms[:, None] + sq_norms[None, :] - 2 * B @ B.T
    D = np.maximum(D, 0.0)
    C = rho_W * (V - Lambda_W / rho_W) - lam * D + rho_d * np.outer(d - Lambda_d / rho_d, np.ones(p))
    # (rho_W I + rho_d * 1 1^T)^{-1} = I/rho_W - rho_d/(rho_W*(rho_W+rho_d*p)) * 1 1^T
    inv_scale = rho_d / (rho_W * (rho_W + rho_d * p))
    row_sums = C.sum(axis=1)
    W = C / rho_W - inv_scale * np.outer(row_sums, np.ones(p))
    W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0.0)
    return W


def _d_update(W: np.ndarray, Lambda_d: np.ndarray, beta: float, rho_d: float, eps: float) -> np.ndarray:
    """Quadratic-root solve of the degree optimality condition:
    -beta/(d_i+eps) + rho_d(d_i - (W1)_i - Lambda_d,i/rho_d) = 0, for each i.

    Rearranged to rho_d d_i^2 + rho_d(eps-c_i) d_i - (rho_d*eps*c_i + beta) = 0
    with c_i = (W1)_i + Lambda_d,i/rho_d; the positive root (d_i > -eps) is kept.
    """
    c = W.sum(axis=1) + Lambda_d / rho_d
    a_coef = rho_d
    b_coef = rho_d * (eps - c)
    c_coef = -(rho_d * eps * c + beta)
    disc = np.maximum(b_coef ** 2 - 4 * a_coef * c_coef, 0.0)
    d = (-b_coef + np.sqrt(disc)) / (2 * a_coef)
    return np.maximum(d, 1e-8)


def _v_update(W: np.ndarray, Lambda_W: np.ndarray, alpha: float, rho_W: float) -> np.ndarray:
    """Proximal soft-thresholding for L1 sparsity with V>=0, V_ii=0, V=V^T."""
    M = W + Lambda_W / rho_W
    V = np.maximum(M - alpha / rho_W, 0.0)
    V = 0.5 * (V + V.T)
    np.fill_diagonal(V, 0.0)
    return V


def _objective(X, B, F, W, Omega_diag, lam, alpha, beta, delta, eps) -> float:
    recon = _weighted_frobenius_sq(X - B @ F, Omega_diag)
    sq_norms = np.sum(B ** 2, axis=1)
    D = sq_norms[:, None] + sq_norms[None, :] - 2 * B @ B.T
    smooth = lam * float(np.sum(W * np.maximum(D, 0.0)))
    sparsity = alpha * float(np.sum(np.abs(W)))
    degree_arg = np.maximum(W.sum(axis=1) + eps, 1e-12)  # guard the monitoring
                                                          # objective only; W's
                                                          # row sums can be
                                                          # transiently negative
                                                          # before the W=V
                                                          # consensus converges
    degree = -beta * float(np.sum(np.log(degree_arg)))
    M = (F * Omega_diag[None, :]) @ F.T
    M_off = M - np.diag(np.diag(M))
    decorr = delta * float(np.sum(M_off ** 2))
    return recon + smooth + sparsity + degree + decorr


def fit_mingle(
    X: np.ndarray,
    k: int = config.K_FACTORS,
    lam: float = config.LAMBDA_SMOOTH,
    delta: float = config.DELTA_DECORR,
    alpha: float = config.ALPHA_SPARSITY,
    beta: float = config.BETA_DEGREE,
    omega: float = config.OMEGA_DECAY,
    rho_B: float = config.RHO_B,
    rho_W: float = config.RHO_W,
    rho_d: float = config.RHO_D,
    eps: float = config.EPS_DEGREE,
    max_iter: int = config.ADMM_MAX_ITER,
    tol: float = config.ADMM_TOL,
    lr_F: float = config.LR_F,
    f_inner_steps: int = config.F_INNER_STEPS,
) -> MingleResult:
    """Fit the MINGLE factor-graph model to log-returns X (p assets x T obs)
    via Algorithm 1 ADMM iterations.
    """
    p, T = X.shape
    if k >= p:
        raise ValueError(f"k={k} must be smaller than the number of assets p={p}")
    Omega_diag = np.diag(build_omega(T, omega))

    B, F = weighted_svd_init(X, Omega_diag, k)
    W = init_graph_rbf(B)
    Q = B.copy()
    V = W.copy()
    d = W.sum(axis=1)
    Lambda_B = np.zeros_like(B)
    Lambda_W = np.zeros_like(W)
    Lambda_d = np.zeros_like(d)

    objective_history, primal_history = [], []
    converged = False
    n_iter = 0
    for it in range(max_iter):
        for _ in range(f_inner_steps):
            grad = _f_gradient(X, B, F, Omega_diag, delta)
            F, _ = _f_step_backtracking(X, B, F, Omega_diag, delta, grad, lr_F)

        L = _laplacian(W)
        B = _b_update(X, F, Omega_diag, L, Q, Lambda_B, lam, rho_B)
        Q = _q_update(B, Lambda_B, rho_B)

        W = _w_update(B, V, d, Lambda_W, Lambda_d, lam, rho_W, rho_d)
        d = _d_update(W, Lambda_d, beta, rho_d, eps)
        V = _v_update(W, Lambda_W, alpha, rho_W)

        Lambda_B = Lambda_B + rho_B * (B - Q)
        Lambda_W = Lambda_W + rho_W * (W - V)
        Lambda_d = Lambda_d + rho_d * (W.sum(axis=1) - d)

        primal_res = (
            np.linalg.norm(B - Q, "fro")
            + np.linalg.norm(W - V, "fro")
            + np.linalg.norm(W.sum(axis=1) - d)
        )
        obj = _objective(X, B, F, W, Omega_diag, lam, alpha, beta, delta, eps)
        objective_history.append(obj)
        primal_history.append(primal_res)
        n_iter = it + 1
        if primal_res < tol:
            converged = True
            break

    # V, not the raw Sherman-Morrison W, is reported as the final graph
    # topology: V carries the hard constraints (symmetry, non-negativity,
    # zero diagonal, L1 sparsity) exactly at every iteration, while W only
    # satisfies them approximately until the W=V consensus residual vanishes
    # (Sec. 3.2, expected outcome: "sparse, symmetric, non-negative adjacency
    # matrix with zero diagonal").
    W_final = V
    L = _laplacian(W_final)
    Psi = np.var(X - B @ F, axis=1, ddof=1)
    F_cov = np.cov(F * np.sqrt(Omega_diag)[None, :]) if k > 1 else np.array([[np.var(F[0])]])
    Sigma_R = B @ F_cov @ B.T + np.diag(Psi)
    return MingleResult(
        B=B, F=F, W=W_final, L=L, Psi=Psi, Sigma_R=Sigma_R,
        objective_history=objective_history, primal_residual_history=primal_history,
        n_iter=n_iter, converged=converged,
    )


def factor_induced_covariance(
    B: np.ndarray, F: np.ndarray, X: np.ndarray, Omega_diag: np.ndarray | None = None
) -> np.ndarray:
    """Sigma = B Cov(F) B^T + Psi, Psi = Var(X_i - (BF)_i) (Sec. 4, "Output")."""
    if Omega_diag is None:
        F_cov = np.cov(F)
    else:
        F_cov = np.cov(F * np.sqrt(Omega_diag)[None, :])
    if F_cov.ndim == 0:
        F_cov = np.array([[float(F_cov)]])
    Psi = np.var(X - B @ F, axis=1, ddof=1)
    return B @ F_cov @ B.T + np.diag(Psi)
