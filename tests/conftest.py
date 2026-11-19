"""Shared pytest fixtures: marker registration and a cached, reduced
market-data slice for tests marked ``market_data``."""

from __future__ import annotations

import os

import pandas as pd
import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "market_data: test requires a live QuantX Dataloader/Massive API call"
    )


@pytest.fixture(scope="session")
def market_data_slice(request: pytest.FixtureRequest):
    """A few tickers, about one year of daily bars, loaded through
    ``data.data_loader`` and cached between test runs via pytest's cache so
    repeated runs do not re-download it."""
    if not os.environ.get("DATALOADER_API_KEY") or not os.environ.get("MASSIVE_API_KEY"):
        pytest.skip("DATALOADER_API_KEY/MASSIVE_API_KEY not set: skipping market-data test")

    cache_dir = request.config.cache.mkdir("mingle_market_data_slice")
    cache_file = cache_dir / "close_panel.csv"
    from data.data_loader import load_close_panel

    tickers = ["AAPL", "MSFT", "JPM", "XOM", "KO"]
    start, end = "2022-01-01", "2023-06-30"
    if cache_file.exists():
        panel = pd.read_csv(cache_file, index_col=0, parse_dates=True)
    else:
        panel = load_close_panel(tickers, start, end)
        panel.to_csv(cache_file)
    return panel
