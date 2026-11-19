"""Experiment 4 - Ablation study under synthetic noise corruption (Fig. 7).

Measures how much each regularisation term of the MINGLE objective
contributes to downstream allocation performance when the equity universe's
log-returns are corrupted with calibrated Gaussian noise at three SNR levels
(Sec. 5, "Ablation Study"). Four variants are compared on the same noisy
window:

  - Proposed:    full MINGLE objective (``src.mingle.fit_mingle_best_of``).
  - Sequential:  standard PCA loadings then a fixed RBF graph, no joint
                 re-optimisation (``src.covariance.pca_factor_covariance``).
  - No-LR:       MINGLE without the ``Y = BF`` low-rank prior
                 (``src.mingle_variants.fit_no_lr``).
  - No-Decorr:   MINGLE with the factor-decorrelation weight ``delta=0``.

For each SNR level, 100 (paper) / 20 (default) independent noise draws are
repeated and the resulting Sharpe ratios are compared across variants with a
paired two-sided t-test against Proposed (Sec. 5, "p < 0.001").

Methodological assumptions not specified by the paper, made explicit here:
  - A single representative 2-year lookback window (the first rebalance date
    with a full lookback available) is used for the whole ablation, rather
    than averaging over all backtest windows, since Fig. 7 is presented as
    one controlled ablation rather than a time series.
  - Sharpe ratio is computed in-sample, on the same noisy log-return window
    used for clustering (there is no separate out-of-sample holding period
    for this synthetic ablation).

Run: python -m exp.exp4_ablation_noise [--full]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

from data.data_loader import select_universe, load_close_panel
from src import config
from src.cutv import adaptive_cutv, cutv_equal_weights
from src.metrics import sharpe_ratio
from src.mingle import build_omega, fit_mingle_best_of
from src.mingle_variants import fit_no_lr
from src.covariance import pca_factor_covariance
from src.pipeline import log_returns, lookback_window, monthly_rebalance_dates

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

VARIANT_NAMES = ("Proposed", "Sequential", "No-LR", "No-Decorr")


def calibrate_noise_sigma(X: np.ndarray, snr_db: float) -> float:
    """Gaussian noise standard deviation achieving the target SNR (dB) for a
    p x T signal matrix ``X`` (Sec. 5, methodology step 1.a):
    ``P_signal = mean(X**2)``, ``sigma**2 = P_signal / 10**(snr_db/10)``.
    """
    p, T = X.shape
    p_signal = np.sum(X ** 2) / (p * T)
    sigma2 = p_signal / (10 ** (snr_db / 10))
    return float(np.sqrt(sigma2))


def boxplot_stats(values: list[float]) -> dict:
    """Median, quartiles, Tukey whiskers and mean of a 1-D sample."""
    data = np.asarray(values, dtype=float)
    q1, median, q3 = np.percentile(data, [25, 50, 75])
    iqr = q3 - q1
    lower_whisker = max(q1 - 1.5 * iqr, float(data.min()))
    upper_whisker = min(q3 + 1.5 * iqr, float(data.max()))
    return {
        "median": float(median),
        "q1": float(q1),
        "q3": float(q3),
        "lower_whisker": float(lower_whisker),
        "upper_whisker": float(upper_whisker),
        "mean": float(data.mean()),
    }


def _variant_sharpe(variant_name: str, X_noisy: np.ndarray, Omega_diag: np.ndarray, n_restarts: int, seed: int) -> float:
    if variant_name == "Proposed":
        Sigma_variant = fit_mingle_best_of(
            X_noisy,
            n_restarts=n_restarts,
            k=config.K_FACTORS,
            lam=config.LAMBDA_SMOOTH,
            delta=config.DELTA_DECORR,
            alpha=config.ALPHA_SPARSITY,
            beta=config.BETA_DEGREE,
            omega=config.OMEGA_DECAY,
            rho_B=config.RHO_B,
            rho_W=config.RHO_W,
            rho_d=config.RHO_D,
            lr_F=config.LR_F,
            max_iter=config.ADMM_MAX_ITER,
            tol=config.ADMM_TOL,
            seed=seed,
        ).Sigma_R
    elif variant_name == "Sequential":
        Sigma_variant = pca_factor_covariance(X_noisy, k=config.K_FACTORS, omega=config.OMEGA_DECAY)[1]
    elif variant_name == "No-LR":
        Sigma_variant = fit_no_lr(
            X_noisy,
            Omega_diag,
            lam=config.LAMBDA_SMOOTH,
            alpha=config.ALPHA_SPARSITY,
            beta=config.BETA_DEGREE,
            rho_W=config.RHO_W,
            rho_d=config.RHO_D,
            eps=config.EPS_DEGREE,
            max_iter=config.ADMM_MAX_ITER,
            tol=config.ADMM_TOL,
        ).Sigma
    elif variant_name == "No-Decorr":
        Sigma_variant = fit_mingle_best_of(
            X_noisy,
            n_restarts=n_restarts,
            k=config.K_FACTORS,
            lam=config.LAMBDA_SMOOTH,
            delta=0.0,
            alpha=config.ALPHA_SPARSITY,
            beta=config.BETA_DEGREE,
            omega=config.OMEGA_DECAY,
            rho_B=config.RHO_B,
            rho_W=config.RHO_W,
            rho_d=config.RHO_D,
            lr_F=config.LR_F,
            max_iter=config.ADMM_MAX_ITER,
            tol=config.ADMM_TOL,
            seed=seed,
        ).Sigma_R
    else:
        raise ValueError(f"Unknown variant: {variant_name}")

    labels, _ = adaptive_cutv(Sigma_variant, n_clusters=config.N_CLUSTERS, is_graph=False)
    weights = cutv_equal_weights(labels)
    portfolio_returns = pd.Series(X_noisy.T @ weights)
    return sharpe_ratio(portfolio_returns)


def run(full: bool) -> dict:
    universe_size = config.UNIVERSE_SIZE_PAPER if full else config.UNIVERSE_SIZE_RUN
    n_seeds = config.ABLATION_N_SEEDS_PAPER if full else config.ABLATION_N_SEEDS_RUN
    n_restarts = 3 if full else config.N_RESTARTS

    print(f"[exp4] universe_size_per_group={universe_size} n_seeds={n_seeds} full={full}")
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
    print(f"[exp4] realised universe size={len(tickers)} "
          f"(groups={universe['group'].value_counts().to_dict()})")

    close_panel = load_close_panel(tickers, config.SAMPLE_DOWNLOAD_START, config.BACKTEST_END_RUN)
    returns_log = log_returns(close_panel)
    print(f"[exp4] close panel rows={len(close_panel)} "
          f"date_range=({close_panel.index.min().date()},{close_panel.index.max().date()}) "
          f"assets={close_panel.shape[1]}")

    reb_dates = monthly_rebalance_dates(
        returns_log.index, config.BACKTEST_START, config.BACKTEST_END_RUN, config.LOOKBACK_TRADING_DAYS
    )
    if not reb_dates:
        raise ValueError("No rebalance date has a full lookback window available")
    rebalance_date = reb_dates[0]
    print(f"[exp4] using a single representative lookback window ending {rebalance_date.date()} "
          f"(the paper does not specify whether Fig. 7 averages over all backtest windows)")
    X_clean = lookback_window(returns_log, rebalance_date, config.LOOKBACK_TRADING_DAYS).to_numpy().T
    p, T = X_clean.shape
    print(f"[exp4] fit window: p={p} assets, T={T} days")

    Omega_diag = np.diag(build_omega(T, config.OMEGA_DECAY))

    sharpes: dict[float, dict[str, list[float]]] = {}
    for snr_db in config.ABLATION_SNR_DB:
        sigma = calibrate_noise_sigma(X_clean, snr_db)
        print(f"[exp4] SNR={snr_db}dB sigma={sigma:.6g}")
        variant_sharpes: dict[str, list[float]] = {name: [] for name in VARIANT_NAMES}
        for seed in range(n_seeds):
            rng = np.random.default_rng(1000 * int(snr_db) + seed)
            noise = sigma * rng.standard_normal(X_clean.shape)
            X_noisy = X_clean + noise

            for variant_name in VARIANT_NAMES:
                sharpe = _variant_sharpe(variant_name, X_noisy, Omega_diag, n_restarts, seed)
                variant_sharpes[variant_name].append(sharpe)
            if (seed + 1) % max(1, n_seeds // 5) == 0 or seed == n_seeds - 1:
                print(f"[exp4] SNR={snr_db}dB seed {seed + 1}/{n_seeds} done")
        sharpes[snr_db] = variant_sharpes
        for name in VARIANT_NAMES:
            print(f"[exp4] SNR={snr_db}dB {name}: mean Sharpe="
                  f"{np.mean(variant_sharpes[name]):.4f}")

    summary: dict = {
        "universe_size_per_group": universe_size,
        "n_assets": p,
        "n_seeds": n_seeds,
        "window_rebalance_date": str(rebalance_date.date()),
        "full": full,
    }
    for snr_db in config.ABLATION_SNR_DB:
        variant_sharpes = sharpes[snr_db]
        proposed = variant_sharpes["Proposed"]
        snr_summary: dict = {}
        for name in VARIANT_NAMES:
            snr_summary[name] = boxplot_stats(variant_sharpes[name])
        ttest_summary: dict = {}
        for name in ("Sequential", "No-LR", "No-Decorr"):
            statistic, pvalue = ttest_rel(proposed, variant_sharpes[name])
            ttest_summary[name] = {
                "statistic": float(statistic),
                "pvalue": float(pvalue),
                "significant_at_0.001": bool(pvalue < config.ABLATION_TTEST_ALPHA),
            }
        snr_summary["ttest_vs_proposed"] = ttest_summary
        summary[str(snr_db)] = snr_summary
        print(f"[exp4] SNR={snr_db}dB summary: {json.dumps(snr_summary, indent=2)}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "exp4_ablation_sharpes.json", "w") as handle:
        json.dump({str(snr): sharpes[snr] for snr in config.ABLATION_SNR_DB}, handle, indent=2)
    with open(RESULTS_DIR / "exp4_ablation_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)

    fig, axes = plt.subplots(1, len(config.ABLATION_SNR_DB), figsize=(15, 5), sharey=True)
    for ax, snr_db in zip(axes, config.ABLATION_SNR_DB):
        variant_sharpes = sharpes[snr_db]
        ax.boxplot([variant_sharpes[name] for name in VARIANT_NAMES], labels=VARIANT_NAMES)
        ax.set_title(f"SNR = {snr_db} dB")
        ax.tick_params(axis="x", rotation=30)
    axes[0].set_ylabel("Sharpe ratio")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp4_ablation_boxplots.png", dpi=150)
    plt.close(fig)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Run the paper's full universe size and seed count")
    args = parser.parse_args()
    run(full=args.full)
