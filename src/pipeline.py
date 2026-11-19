"""Shared data-preparation and rebalancing utilities (Sec. 4-5): log-returns
for MINGLE training, simple returns for portfolio performance, and the
monthly rebalancing schedule built on a rolling 2-year lookback window."""

from __future__ import annotations

import numpy as np
import pandas as pd


def log_returns(close_panel: pd.DataFrame) -> pd.DataFrame:
    """Daily log-returns from a date x ticker close-price panel (Sec. 4:
    "Log-returns are used for factor-graph training")."""
    return np.log(close_panel / close_panel.shift(1)).iloc[1:]


def simple_returns(close_panel: pd.DataFrame) -> pd.DataFrame:
    """Daily simple returns, used only for portfolio performance (Sec. 5:
    "Simple returns for portfolio performance computation")."""
    return close_panel.pct_change().iloc[1:]


def monthly_rebalance_dates(
    trading_days: pd.DatetimeIndex, start: str, end: str, lookback_days: int
) -> list[pd.Timestamp]:
    """Last trading day of each calendar month in ``[start, end]`` that has
    at least ``lookback_days`` of history available before it (Sec. 4-5:
    "monthly rebalancing", "2-year lookback window").
    """
    trading_days = pd.DatetimeIndex(trading_days)
    position = {day: i for i, day in enumerate(trading_days)}
    in_range = trading_days[(trading_days >= pd.Timestamp(start)) & (trading_days <= pd.Timestamp(end))]
    if len(in_range) == 0:
        return []
    frame = pd.DataFrame({"date": in_range})
    frame["period"] = frame["date"].dt.to_period("M")
    last_of_month = frame.groupby("period")["date"].max().tolist()
    return sorted(d for d in last_of_month if position[d] >= lookback_days - 1)


def lookback_window(returns: pd.DataFrame, rebalance_date: pd.Timestamp, lookback_days: int) -> pd.DataFrame:
    """The ``lookback_days`` rows of ``returns`` ending at ``rebalance_date``
    inclusive, preventing look-ahead: only information up to and including
    the rebalance date is used to fit the model for the following period."""
    position = returns.index.get_loc(rebalance_date)
    start_position = position - lookback_days + 1
    if start_position < 0:
        raise ValueError("Not enough history before the rebalance date for the requested lookback window")
    return returns.iloc[start_position : position + 1]


def next_period_mask(all_dates: pd.DatetimeIndex, rebalance_date: pd.Timestamp, next_rebalance: pd.Timestamp | None) -> np.ndarray:
    """Boolean mask over ``all_dates`` for the holding period that starts the
    day after ``rebalance_date`` and ends at ``next_rebalance`` inclusive (or
    at the last available date when ``next_rebalance`` is ``None``)."""
    end = next_rebalance if next_rebalance is not None else all_dates[-1]
    return (all_dates > rebalance_date) & (all_dates <= end)
