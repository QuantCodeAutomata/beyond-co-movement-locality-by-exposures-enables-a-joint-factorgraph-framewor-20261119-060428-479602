"""Performance metrics for the backtest (Sec. 5, methodology step 6)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


def cagr(returns: pd.Series, periods_per_year: int = config.ANNUALISATION_DAYS) -> float:
    """CAGR = (final_value/initial_value)^(periods_per_year/n_days) - 1."""
    cumulative = (1.0 + returns).cumprod()
    if len(cumulative) == 0 or cumulative.iloc[-1] <= 0:
        return float("nan")
    n_days = len(returns)
    return float(cumulative.iloc[-1] ** (periods_per_year / n_days) - 1.0)


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = config.RISK_FREE_RATE,
    periods_per_year: int = config.ANNUALISATION_DAYS,
) -> float:
    """Annualised Sharpe ratio: mean(excess)/std(excess) * sqrt(periods_per_year)."""
    daily_rf = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    excess = returns - daily_rf
    std = excess.std(ddof=1)
    if std == 0 or np.isnan(std):
        return float("nan")
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> float:
    cumulative = (1.0 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1.0
    return float(drawdown.min())


def calmar_ratio(returns: pd.Series, periods_per_year: int = config.ANNUALISATION_DAYS) -> float:
    """Calmar ratio = CAGR / |max drawdown|."""
    mdd = max_drawdown(returns)
    if mdd == 0 or np.isnan(mdd):
        return float("nan")
    return float(cagr(returns, periods_per_year) / abs(mdd))


def classify_regime(vix_value: float) -> str:
    """Calm (VIX<20), Elevated (20<=VIX<=30), Crisis (VIX>30) (Remark 2)."""
    if vix_value < config.VIX_CALM_MAX:
        return "calm"
    if vix_value <= config.VIX_ELEVATED_MAX:
        return "elevated"
    return "crisis"
