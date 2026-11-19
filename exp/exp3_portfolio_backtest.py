"""Experiment 3 - Portfolio performance backtest (Sec. 5, Table 2, Figures
5-6): nine long-only allocation strategies compared across five transaction
cost levels and three VIX-defined volatility regimes, over a monthly
rebalanced 2019-01 .. 2026-03 backtest with a rolling 2-year lookback.

Strategies: Equal Weight, MVO (sample covariance), MVO with Ledoit-Wolf
shrinkage, HRP, Adaptive CutV on four covariance estimates (Sigma_S, LW-Sigma_S,
Sigma_B [standard-PCA factor-induced], Sigma_R [MINGLE representation-induced]),
and Contagion Cut (CutV on Sigma_R with within-cluster weights inversely
proportional to MINGLE graph degree, Eq. 9).

Run: python -m exp.exp3_portfolio_backtest [--full]
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
import pandas as pd

from data.data_loader import select_universe, load_close_panel, load_vix
from src import config
from src.allocation import equal_weight, hrp_weights, mvo_min_variance
from src.backtest import run_backtest
from src.covariance import ledoit_wolf_covariance, pca_factor_covariance, sample_covariance
from src.cutv import adaptive_cutv, contagion_cut_weights, cutv_equal_weights
from src.metrics import cagr, calmar_ratio, classify_regime, sharpe_ratio
from src.mingle import fit_mingle_best_of
from src.pipeline import log_returns, lookback_window, monthly_rebalance_dates, simple_returns

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

STRATEGIES = (
    "Equal Weight",
    "MVO",
    "MVO, LW-Sigma_S",
    "HRP",
    "CutV, Sigma_S",
    "CutV, LW-Sigma_S",
    "CutV, Sigma_B",
    "CutV, Sigma_R",
    "Contagion Cut",
)


def _strategy_weights(
    p: int,
    Sigma_S: np.ndarray,
    Sigma_LW: np.ndarray,
    Sigma_B: np.ndarray,
    Sigma_R: np.ndarray,
    W_mingle: np.ndarray,
    simple_window: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Compute the nine strategies' weight vectors for one rebalance date
    (Sec. 5, methodology step 3). All strategies are long-only, w>=0, sum=1."""
    labels_S, _ = adaptive_cutv(Sigma_S, n_clusters=config.N_CLUSTERS, is_graph=False)
    labels_LW, _ = adaptive_cutv(Sigma_LW, n_clusters=config.N_CLUSTERS, is_graph=False)
    labels_B, _ = adaptive_cutv(Sigma_B, n_clusters=config.N_CLUSTERS, is_graph=False)
    labels_R, _ = adaptive_cutv(Sigma_R, n_clusters=config.N_CLUSTERS, is_graph=False)
    return {
        "Equal Weight": equal_weight(p),
        "MVO": mvo_min_variance(Sigma_S),
        "MVO, LW-Sigma_S": mvo_min_variance(Sigma_LW),
        "HRP": hrp_weights(simple_window),
        "CutV, Sigma_S": cutv_equal_weights(labels_S),
        "CutV, LW-Sigma_S": cutv_equal_weights(labels_LW),
        "CutV, Sigma_B": cutv_equal_weights(labels_B),
        "CutV, Sigma_R": cutv_equal_weights(labels_R),
        # Contagion Cut (Eq. 9): same Sigma_R clusters as "CutV, Sigma_R", but
        # weighted within each cluster by inverse MINGLE graph degree so that
        # peripheral (low-degree) assets receive more weight.
        "Contagion Cut": contagion_cut_weights(labels_R, W_mingle),
    }


def run(full: bool) -> dict:
    universe_size = config.UNIVERSE_SIZE_PAPER if full else config.UNIVERSE_SIZE_RUN
    n_restarts = 3 if full else config.N_RESTARTS

    print(f"[exp3] universe_size_per_group={universe_size} full={full}")
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

    close_panel = load_close_panel(tickers, config.SAMPLE_DOWNLOAD_START, config.BACKTEST_END_RUN)
    returns_log = log_returns(close_panel)
    returns_simple = simple_returns(close_panel)
    p = close_panel.shape[1]
    print(f"[exp3] close panel rows={len(close_panel)} "
          f"date_range=({close_panel.index.min().date()},{close_panel.index.max().date()}) assets={p}")

    vix = load_vix(config.BACKTEST_START, config.BACKTEST_END_RUN)

    reb_dates = monthly_rebalance_dates(
        returns_log.index, config.BACKTEST_START, config.BACKTEST_END_RUN, config.LOOKBACK_TRADING_DAYS
    )
    if not reb_dates:
        raise ValueError("No rebalance date has a full lookback window available")
    print(f"[exp3] n_rebalances={len(reb_dates)} "
          f"({reb_dates[0].date()} .. {reb_dates[-1].date()}), n_restarts={n_restarts}")

    weights_schedules: dict[str, dict[pd.Timestamp, np.ndarray]] = {name: {} for name in STRATEGIES}

    start = time.time()
    for i, reb_date in enumerate(reb_dates):
        X_log = lookback_window(returns_log, reb_date, config.LOOKBACK_TRADING_DAYS).to_numpy().T
        simple_window = lookback_window(returns_simple, reb_date, config.LOOKBACK_TRADING_DAYS)

        result = fit_mingle_best_of(
            X_log, n_restarts=n_restarts,
            k=config.K_FACTORS, lam=config.LAMBDA_SMOOTH, delta=config.DELTA_DECORR,
            alpha=config.ALPHA_SPARSITY, beta=config.BETA_DEGREE, omega=config.OMEGA_DECAY,
            rho_B=config.RHO_B, rho_W=config.RHO_W, rho_d=config.RHO_D, lr_F=config.LR_F,
            max_iter=config.ADMM_MAX_ITER, tol=config.ADMM_TOL,
        )
        Sigma_S = sample_covariance(simple_window)
        Sigma_LW = ledoit_wolf_covariance(simple_window)
        _, Sigma_B = pca_factor_covariance(X_log, k=config.K_FACTORS, omega=config.OMEGA_DECAY)
        Sigma_R = result.Sigma_R

        weights = _strategy_weights(p, Sigma_S, Sigma_LW, Sigma_B, Sigma_R, result.W, simple_window)
        for name, w in weights.items():
            weights_schedules[name][reb_date] = w

        if (i + 1) % 10 == 0 or i == len(reb_dates) - 1:
            print(f"[exp3] rebalanced {i + 1}/{len(reb_dates)} windows "
                  f"(elapsed {time.time() - start:.1f}s)")

    print(f"[exp3] all rebalances fitted in {time.time() - start:.1f}s")

    # Backtest every strategy at every transaction-cost level (Sec. 5,
    # methodology steps 4-6).
    performance_rows = []
    cumulative_returns: dict[str, pd.Series] = {}
    regime_sharpe: dict[str, dict[str, float]] = {}
    vix_asof = vix.reindex(returns_simple.index.union(vix.index)).sort_index().ffill()

    for name in STRATEGIES:
        tc_results = run_backtest(returns_simple, weights_schedules[name], tc_bps_list=config.TRANSACTION_COST_BPS)
        for tc_bps, daily_returns in tc_results.items():
            performance_rows.append({
                "strategy": name,
                "tc_bps": tc_bps,
                "cagr": cagr(daily_returns),
                "sharpe": sharpe_ratio(daily_returns),
                "calmar": calmar_ratio(daily_returns),
                "n_days": len(daily_returns),
            })
            if tc_bps == 0:
                cumulative_returns[name] = daily_returns
                regimes = vix_asof.reindex(daily_returns.index).apply(
                    lambda v: classify_regime(v) if pd.notna(v) else "unknown"
                )
                regime_sharpe[name] = {
                    regime: sharpe_ratio(daily_returns[regimes == regime])
                    for regime in ("calm", "elevated", "crisis")
                    if (regimes == regime).sum() > 5
                }

    performance_table = pd.DataFrame(performance_rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    performance_table.to_csv(RESULTS_DIR / "exp3_performance_table.csv", index=False)
    print(f"[exp3] performance table (tc_bps=0):\n"
          f"{performance_table[performance_table.tc_bps == 0].to_string(index=False)}")

    with open(RESULTS_DIR / "exp3_regime_sharpe.json", "w") as handle:
        json.dump(regime_sharpe, handle, indent=2, default=float)
    print(f"[exp3] regime-conditional Sharpe (tc=0bps): {json.dumps(regime_sharpe, indent=2, default=float)}")

    # Figure 5: cumulative returns normalised to 1.0 at January 2020.
    fig, ax = plt.subplots(figsize=(10, 6))
    rebase_date = pd.Timestamp("2020-01-01")
    for name, daily_returns in cumulative_returns.items():
        wealth = (1.0 + daily_returns).cumprod()
        rebase_idx = wealth.index.searchsorted(rebase_date)
        rebase_idx = min(rebase_idx, len(wealth) - 1)
        rebased = wealth / wealth.iloc[rebase_idx]
        ax.plot(rebased.index, rebased.values, label=name)
    ax.axhline(1.0, color="black", linewidth=0.5, linestyle="--")
    ax.set_title("Cumulative portfolio returns (normalised to 1.0 at January 2020), tc=0bps")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative wealth (rebased)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp3_cumulative_returns.png", dpi=150)
    plt.close(fig)

    # Figure 6: regime-conditional annualised Sharpe ratio by strategy.
    regimes_order = ["calm", "elevated", "crisis"]
    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.09
    x = np.arange(len(regimes_order))
    for j, name in enumerate(STRATEGIES):
        values = [regime_sharpe.get(name, {}).get(r, np.nan) for r in regimes_order]
        ax.bar(x + j * width, values, width=width, label=name)
    ax.set_xticks(x + width * (len(STRATEGIES) - 1) / 2)
    ax.set_xticklabels([r.capitalize() for r in regimes_order])
    ax.set_ylabel("Annualised Sharpe ratio")
    ax.set_title("Regime-conditional Sharpe ratio by strategy (VIX regimes, Remark 2), tc=0bps")
    ax.legend(fontsize=6, ncol=3)
    ax.axhline(0.0, color="black", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp3_regime_sharpe.png", dpi=150)
    plt.close(fig)

    summary = {
        "universe_size_per_group": universe_size,
        "n_assets": p,
        "n_rebalances": len(reb_dates),
        "n_restarts": n_restarts,
        "full": full,
        "backtest_start": str(reb_dates[0].date()),
        "backtest_end": str(returns_simple.index.max().date()),
        "performance_table_tc0": performance_table[performance_table.tc_bps == 0]
        .set_index("strategy")[["cagr", "sharpe", "calmar"]].to_dict(orient="index"),
        "regime_sharpe_tc0": regime_sharpe,
    }
    with open(RESULTS_DIR / "exp3_portfolio_backtest.json", "w") as handle:
        json.dump(summary, handle, indent=2, default=float)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Run the paper's full universe size and restart count")
    args = parser.parse_args()
    run(full=args.full)
