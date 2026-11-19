"""Experiment 1 - MINGLE ADMM framework: joint factor-graph learning.

Fits the MINGLE model (``src.mingle.fit_mingle``) on one rolling two-year
lookback window of log-returns for the paper's equity universe (Sec. 4), and
reports the structural properties the paper claims for the learnt
representation (Sec. 3, Remark 1, Sec. 4 "Output"): ADMM convergence to a
stationary point, a sparse symmetric non-negative zero-diagonal graph W, a
positive semi-definite graph Laplacian, and a representation-induced
covariance Sigma_R better conditioned than the sample covariance Sigma_S.

Run: python -m exp.exp1_mingle_admm [--full]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data.data_loader import select_universe, load_close_panel
from src import config
from src.covariance import condition_number
from src.mingle import build_omega, fit_mingle_best_of
from src.pipeline import log_returns, lookback_window, monthly_rebalance_dates

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def run(full: bool) -> dict:
    universe_size = config.UNIVERSE_SIZE_PAPER if full else config.UNIVERSE_SIZE_RUN
    n_restarts = 3 if full else config.N_RESTARTS

    print(f"[exp1] universe_size_per_group={universe_size} full={full}")
    # Market data: QuantX Dataloader -- https://quantx.limex.pro/
    #
    # Price history in this project is loaded through the QuantX Dataloader, a
    # research-ready market-data service available through a REST API and as an
    # MCP server for AI agents. It is built on market data from Massive
    # (https://massive.com/) and prepares every stock history for backtesting:
    #   - prices adjusted for dividends, ready for total-return research;
    #   - one continuous history when a company changes its ticker;
    #   - clean separation when a ticker has belonged to different companies;
    #   - careful corporate-action handling and fine-grained data polishing.
    #
    # Get your DATALOADER_API_KEY at https://quantx.limex.pro/ and your
    # MASSIVE_API_KEY at https://massive.com/.
    universe = select_universe(
        f"{config.LIQUIDITY_RANKING_YEAR}-01-01",
        f"{config.LIQUIDITY_RANKING_YEAR}-12-31",
        n_per_group=universe_size,
    )
    tickers = universe["ticker"].tolist()
    print(f"[exp1] realised universe size={len(tickers)} "
          f"(groups={universe['group'].value_counts().to_dict()})")

    close_panel = load_close_panel(tickers, config.SAMPLE_DOWNLOAD_START, config.BACKTEST_END_RUN)
    returns_log = log_returns(close_panel)
    print(f"[exp1] close panel rows={len(close_panel)} "
          f"date_range=({close_panel.index.min().date()},{close_panel.index.max().date()}) "
          f"assets={close_panel.shape[1]}")

    reb_dates = monthly_rebalance_dates(
        returns_log.index, config.BACKTEST_START, config.BACKTEST_END_RUN, config.LOOKBACK_TRADING_DAYS
    )
    if not reb_dates:
        raise ValueError("No rebalance date has a full lookback window available")
    rebalance_date = reb_dates[0]
    window = lookback_window(returns_log, rebalance_date, config.LOOKBACK_TRADING_DAYS)
    X = window.to_numpy().T  # p x T: assets x time
    p, T = X.shape
    print(f"[exp1] fit window ending {rebalance_date.date()}: p={p} assets, T={T} days")

    start = time.time()
    result = fit_mingle_best_of(
        X,
        n_restarts=n_restarts,
        k=config.K_FACTORS,
        omega=config.OMEGA_DECAY,
        lam=config.LAMBDA_SMOOTH,
        delta=config.DELTA_DECORR,
        alpha=config.ALPHA_SPARSITY,
        beta=config.BETA_DEGREE,
        rho_B=config.RHO_B,
        rho_W=config.RHO_W,
        rho_d=config.RHO_D,
        lr_F=config.LR_F,
        max_iter=config.ADMM_MAX_ITER,
        tol=config.ADMM_TOL,
    )
    elapsed = time.time() - start
    print(f"[exp1] ADMM finished in {elapsed:.1f}s, n_iter={result.n_iter}, "
          f"converged={result.converged}")

    # Structural diagnostics (Sec. 3 Remark 1, Sec. 4 "Output").
    Omega_diag = np.diag(build_omega(T, config.OMEGA_DECAY))
    Sigma_S = np.cov(X)
    kappa_R = condition_number(result.Sigma_R)
    kappa_S = condition_number(Sigma_S)
    laplacian_eigs = np.linalg.eigvalsh(0.5 * (result.L + result.L.T))
    graph_density = float((result.W > 1e-10).sum()) / (p * (p - 1))
    symmetry_error = float(np.max(np.abs(result.W - result.W.T)))

    diagnostics = {
        "universe_size_per_group": universe_size,
        "n_assets": p,
        "n_days_window": T,
        "rebalance_date": str(rebalance_date.date()),
        "k_factors": config.K_FACTORS,
        "n_restarts": n_restarts,
        "n_iter": result.n_iter,
        "converged": bool(result.converged),
        "fit_seconds": elapsed,
        "objective_initial": float(result.objective_history[0]) if result.objective_history else None,
        "objective_final": float(result.objective_history[-1]) if result.objective_history else None,
        "primal_residual_final": float(result.primal_residual_history[-1]) if result.primal_residual_history else None,
        "W_symmetry_max_abs_error": symmetry_error,
        "W_min_entry": float(result.W.min()),
        "W_max_diag_abs": float(np.max(np.abs(np.diag(result.W)))),
        "W_density_offdiag": graph_density,
        "laplacian_min_eigenvalue": float(laplacian_eigs.min()),
        "condition_number_Sigma_R": kappa_R,
        "condition_number_Sigma_S": kappa_S,
        "condition_number_ratio_S_over_R": kappa_S / kappa_R if kappa_R > 0 else float("inf"),
    }
    print(f"[exp1] diagnostics: {json.dumps(diagnostics, indent=2)}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(RESULTS_DIR / "exp1_B.npy", result.B)
    np.save(RESULTS_DIR / "exp1_F.npy", result.F)
    np.save(RESULTS_DIR / "exp1_W.npy", result.W)
    np.save(RESULTS_DIR / "exp1_Sigma_R.npy", result.Sigma_R)
    with open(RESULTS_DIR / "exp1_mingle_admm.json", "w") as handle:
        json.dump(diagnostics, handle, indent=2)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(result.objective_history, marker="o", markersize=3)
    axes[0].set_title("MINGLE objective (Eq. 5) across ADMM iterations")
    axes[0].set_xlabel("ADMM iteration")
    axes[0].set_ylabel("Objective value")

    axes[1].plot(result.primal_residual_history, marker="o", markersize=3, color="darkorange")
    axes[1].set_yscale("log")
    axes[1].set_title("Primal residual norm (consensus gap)")
    axes[1].set_xlabel("ADMM iteration")
    axes[1].set_ylabel("||B-Q||+||W-V||+||W1-d|| (log scale)")

    im = axes[2].imshow(result.W, cmap="viridis")
    axes[2].set_title("Learnt graph topology W")
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp1_convergence_and_graph.png", dpi=150)
    plt.close(fig)

    return diagnostics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Run the paper's full universe size and restart count")
    args = parser.parse_args()
    run(full=args.full)
