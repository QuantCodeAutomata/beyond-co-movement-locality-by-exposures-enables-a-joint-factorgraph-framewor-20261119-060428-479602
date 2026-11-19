"""Monthly-rebalanced backtest engine with transaction costs (Sec. 5,
methodology steps 4-5)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


def one_way_turnover(w_prev: np.ndarray, w_new: np.ndarray) -> float:
    """One-way portfolio turnover = (1/2) sum|w_new - w_prev| (Sec. 5,
    "Transaction costs")."""
    return 0.5 * float(np.abs(w_new - w_prev).sum())


def run_backtest(
    simple_returns: pd.DataFrame,
    weights_schedule: dict[pd.Timestamp, np.ndarray],
    tc_bps_list: tuple[float, ...] = config.TRANSACTION_COST_BPS,
) -> dict[float, pd.Series]:
    """Simulate daily portfolio returns under a monthly rebalancing schedule.

    ``simple_returns`` is a date x ticker panel of simple daily returns, used
    for portfolio performance only (log-returns are used solely for MINGLE
    training, Sec. 4). ``weights_schedule`` maps each rebalance date to a
    weight vector aligned with ``simple_returns.columns``; weights are held
    constant between rebalances. Transaction costs (Sec. 5, methodology step
    5) are applied as a one-off drag on the rebalance day: cost = tc_bps *
    1e-4 * one_way_turnover.

    Returns one daily net-return series per transaction-cost level in
    ``tc_bps_list``.
    """
    rebalance_dates = sorted(weights_schedule.keys())
    if not rebalance_dates:
        raise ValueError("weights_schedule must contain at least one rebalance date")
    tickers = simple_returns.columns
    all_dates = simple_returns.index

    results: dict[float, pd.Series] = {}
    for tc_bps in tc_bps_list:
        daily_returns = pd.Series(0.0, index=all_dates)
        w_prev = np.zeros(len(tickers))
        for i, reb_date in enumerate(rebalance_dates):
            w_new = weights_schedule[reb_date]
            turnover = one_way_turnover(w_prev, w_new)
            cost = tc_bps * 1e-4 * turnover
            period_end = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else all_dates[-1]
            period_mask = (all_dates > reb_date) & (all_dates <= period_end) if i > 0 else (
                (all_dates >= reb_date) & (all_dates <= period_end)
            )
            period_dates = all_dates[period_mask]
            if len(period_dates) == 0:
                w_prev = w_new
                continue
            period_returns = simple_returns.loc[period_dates].to_numpy() @ w_new
            daily_returns.loc[period_dates] = period_returns
            daily_returns.loc[period_dates[0]] -= cost
            w_prev = w_new
        results[tc_bps] = daily_returns.loc[daily_returns.index >= rebalance_dates[0]]
    return results
