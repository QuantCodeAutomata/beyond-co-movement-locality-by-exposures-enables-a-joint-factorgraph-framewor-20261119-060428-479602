"""Long-only allocation strategies compared in the backtest (Sec. 5,
methodology step 3): Equal Weight, MVO, MVO with Ledoit-Wolf shrinkage, HRP,
Adaptive CutV (equal- and Contagion-weighted)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import cvxpy as cp
from pypfopt.hierarchical_portfolio import HRPOpt


def equal_weight(p: int) -> np.ndarray:
    return np.full(p, 1.0 / p)


def mvo_min_variance(Sigma: np.ndarray) -> np.ndarray:
    """Long-only minimum-variance MVO (Sec. 5: "minimise w^T Sigma_S w
    subject to [long-only, budget] constraints"), solved as a QP with cvxpy.
    """
    p = Sigma.shape[0]
    Sigma_psd = 0.5 * (Sigma + Sigma.T) + 1e-8 * np.eye(p)
    w = cp.Variable(p)
    objective = cp.Minimize(cp.quad_form(w, cp.psd_wrap(Sigma_psd)))
    constraints = [cp.sum(w) == 1, w >= 0]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.OSQP)
    if w.value is None:
        problem.solve(solver=cp.ECOS)
    weights = np.maximum(np.asarray(w.value).flatten(), 0.0)
    return weights / weights.sum()


def hrp_weights(simple_returns: pd.DataFrame) -> np.ndarray:
    """Hierarchical Risk Parity (Lopez de Prado, ref. [20]) via PyPortfolioOpt."""
    hrp = HRPOpt(simple_returns)
    hrp.optimize()
    weights_dict = hrp.clean_weights()
    return np.array([weights_dict[c] for c in simple_returns.columns])
