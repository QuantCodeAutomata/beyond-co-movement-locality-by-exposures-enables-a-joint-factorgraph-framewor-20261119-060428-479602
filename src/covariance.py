"""Covariance estimators compared in the backtest (Sec. 5, Table 2):
sample covariance, Ledoit-Wolf shrinkage, standard-PCA factor-induced
covariance, and the MINGLE representation-induced covariance (the latter is
produced directly by ``src.mingle.fit_mingle``)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from src.mingle import build_omega


def sample_covariance(simple_returns: pd.DataFrame) -> np.ndarray:
    """Sigma_S: sample covariance of simple returns (Sec. 5, methodology step 2.b)."""
    return simple_returns.cov().to_numpy()


def ledoit_wolf_covariance(simple_returns: pd.DataFrame) -> np.ndarray:
    """LW-Sigma_S: Ledoit-Wolf shrunk sample covariance (sklearn.covariance.LedoitWolf)."""
    lw = LedoitWolf().fit(simple_returns.to_numpy())
    return lw.covariance_


def pca_factor_covariance(
    X_log: np.ndarray, k: int, omega: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Sigma_B: factor-induced covariance using standard (unweighted) PCA
    loadings as the first k principal components (Sec. 5, methodology step
    2.b: "using standard PCA loadings B_PCA as first k principal
    components"), contrasted with the Omega-weighted MINGLE exposures.
    """
    p, T = X_log.shape
    X_centered = X_log - X_log.mean(axis=1, keepdims=True)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    B_pca = U[:, :k]
    F_pca = (S[:k, None] * Vt[:k, :])
    Omega_diag = np.diag(build_omega(T, omega)) if omega else None
    F_cov = np.cov(F_pca * np.sqrt(Omega_diag)[None, :]) if Omega_diag is not None else np.cov(F_pca)
    if F_cov.ndim == 0:
        F_cov = np.array([[float(F_cov)]])
    Psi = np.var(X_log - B_pca @ F_pca, axis=1, ddof=1)
    Sigma_B = B_pca @ F_cov @ B_pca.T + np.diag(Psi)
    return B_pca, Sigma_B


def condition_number(Sigma: np.ndarray) -> float:
    eigenvalues = np.linalg.eigvalsh(0.5 * (Sigma + Sigma.T))
    eigenvalues = eigenvalues[eigenvalues > 0]
    if len(eigenvalues) == 0:
        return float("inf")
    return float(eigenvalues.max() / eigenvalues.min())
