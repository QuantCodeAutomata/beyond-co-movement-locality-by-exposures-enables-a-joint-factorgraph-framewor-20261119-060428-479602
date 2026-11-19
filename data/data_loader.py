"""Market data access for the MINGLE replication.

OHLCV bars are loaded only through the QuantX Dataloader REST API. Reference
data needed to build the equity universe and sector labels (which the
Dataloader does not serve) is loaded directly from the Massive REST API.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

API_ROOT = "https://quantx-api.limex.pro/apis/dataloader"
MASSIVE_ROOT = "https://api.massive.com"
FIELDS = ("open", "high", "low", "close", "volume")
BATCH_SIZE = 15
MAX_PARALLEL = 4
RETRY_STATUSES = {429, 503, 504}
CACHE_DIR = Path(__file__).resolve().parent / "cache"


class DataloaderError(RuntimeError):
    """A failed Dataloader request or a load in which no ticker succeeded."""


def _headers() -> dict[str, str]:
    load_dotenv()
    dataloader_key = os.environ.get("DATALOADER_API_KEY")
    massive_key = os.environ.get("MASSIVE_API_KEY")
    if not dataloader_key or not massive_key:
        raise DataloaderError("DATALOADER_API_KEY and MASSIVE_API_KEY must be set")
    return {
        "Authorization": f"Bearer {dataloader_key}",
        "X-Massive-Api-Key": massive_key,
    }


def _request(method: str, route: str, body: dict | None = None) -> dict:
    headers = _headers()
    if route != "/fetch-ohlcvs":
        del headers["X-Massive-Api-Key"]
    response = None
    for attempt in range(3):
        # Market data: QuantX Dataloader — https://quantx.limex.pro/
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
        response = requests.request(
            method, API_ROOT + route, json=body, headers=headers, timeout=900
        )
        if response.status_code not in RETRY_STATUSES or attempt == 2:
            break
        time.sleep(float(response.headers.get("Retry-After") or 5 * 2**attempt))
    if response.status_code != 200:
        try:
            error = response.json()
        except ValueError:
            error = {"error": response.text[:200]}
        raise DataloaderError(
            f"{route}: HTTP {response.status_code} {error.get('code', '')}: "
            f"{error.get('error', '')} {error.get('details', '')}"
        )
    return response.json()


def _bars_of(answer: dict, request: dict, batch: int) -> pd.DataFrame:
    result = answer["result"]
    if not result.get("data_stored"):
        return pd.DataFrame(result["ohlcv_df"]["rows"])
    export = _request("POST", "/save-dataframe", {
        "result_id": result["result_id"],
        "ticker": f"batch{batch}",
        "dataset_name": "ohlcv",
        "timeframe": request["timeframe"],
        "start_date": request["start_date"],
        "end_date": request["end_date"],
        "file_format": "csv",
        "data_kind": "ohlcv",
    })
    link = _request("GET", f"/artifacts/{export['result']['artifact_id']}/download")
    csv = requests.get(link["result"]["url"], timeout=600)
    csv.raise_for_status()
    return pd.read_csv(io.BytesIO(csv.content))


def _tidy(wide: pd.DataFrame, details: list[dict]) -> pd.DataFrame:
    stamps = pd.to_datetime(wide["timestamp"], utc=True)
    frames = []
    for detail in details:
        if detail.get("error"):
            continue
        ticker, alias = detail["ticker"], detail.get("alias") or ""
        columns = {}
        for field in FIELDS:
            for name in (f"{ticker}_{field}", f"{alias}_{field}", field):
                if name in wide.columns:
                    columns[name] = field
                    break
            else:
                raise DataloaderError(f"{ticker}: no {field} column in the answer")
        bars = wide[list(columns)].rename(columns=columns)
        bars.insert(0, "ticker", ticker)
        bars.insert(0, "timestamp", stamps)
        frames.append(bars.dropna(subset=["close"]))
    bars = pd.concat(frames, ignore_index=True)
    local = bars["timestamp"].dt.tz_convert("America/New_York")
    bars.insert(1, "date", local.dt.normalize().dt.tz_localize(None))
    return bars.sort_values(["ticker", "timestamp"], ignore_index=True)


def _load_batch(request: dict, batch: int) -> tuple[pd.DataFrame, list[dict], dict]:
    answer = _request("POST", "/fetch-ohlcvs", request)
    details = answer["result"].get("ticker_details") or []
    loaded = [d for d in details if not d.get("error")]
    wide = _bars_of(answer, request, batch) if loaded else pd.DataFrame()
    bars = _tidy(wide, details) if loaded else pd.DataFrame()
    notes = {
        "result_id": answer["result"].get("result_id") or answer.get("result_id"),
        "meta": answer.get("meta"),
    }
    return bars, details, notes


def load_ohlcv(
    tickers: list[str],
    start_date: str,
    end_date: str,
    timeframe: str = "1d",
    *,
    adjust_dividends: bool = True,
    identity_date: str | None = None,
    yahoo_backstop: bool = True,
    use_cache: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Daily OHLCV bars for ``tickers`` over the inclusive window.

    Returns a tidy table of timestamp (UTC), date, ticker, open, high, low,
    close, volume, sorted by ticker and time, and a metadata dictionary with
    the Dataloader's per-ticker ``ticker_details``.
    """
    request = {
        "timeframe": timeframe,
        "start_date": start_date,
        "end_date": end_date,
        "source": "massive",
        "adjust_dividends": adjust_dividends,
        "yahoo_backstop": yahoo_backstop,
    }
    if identity_date:
        request["identity_date"] = identity_date
    tickers = list(dict.fromkeys(tickers))
    key = hashlib.sha256(
        json.dumps({**request, "tickers": sorted(tickers)}, sort_keys=True).encode()
    ).hexdigest()[:16]
    bars_path, meta_path = CACHE_DIR / f"{key}.csv.gz", CACHE_DIR / f"{key}.json"
    if use_cache and bars_path.exists() and meta_path.exists():
        bars = pd.read_csv(bars_path, parse_dates=["timestamp", "date"])
        bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
        return bars, json.loads(meta_path.read_text())

    batches = [tickers[i:i + BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]
    with ThreadPoolExecutor(MAX_PARALLEL) as pool:
        results = list(pool.map(
            lambda item: _load_batch({**request, "tickers": item[1]}, item[0]),
            enumerate(batches),
        ))
    frames = [bars for bars, _, _ in results if not bars.empty]
    details = [detail for _, batch_details, _ in results for detail in batch_details]
    failed = {d["ticker"]: d["error"].get("message") for d in details if d.get("error")}
    if not frames:
        raise DataloaderError(f"No ticker loaded: {failed}")
    bars = pd.concat(frames, ignore_index=True)
    meta = {
        "request": request,
        "ticker_details": details,
        "failed": failed,
        "warnings": {d["ticker"]: d["warnings"] for d in details if d.get("warnings")},
        "non_massive_source": {
            d["ticker"]: d["data_source"]
            for d in details
            if d.get("data_source") not in (None, "massive")
        },
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    bars.to_csv(bars_path, index=False)
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    return bars, meta


def load_close_panel(
    tickers: list[str], start_date: str, end_date: str, use_cache: bool = True
) -> pd.DataFrame:
    """Wide panel of dividend-adjusted close prices, index=date, columns=ticker."""
    bars, _ = load_ohlcv(tickers, start_date, end_date, use_cache=use_cache)
    panel = bars.pivot(index="date", columns="ticker", values="close").sort_index()
    panel = panel.ffill(limit=5)
    return panel


def load_vix(start_date: str, end_date: str, use_cache: bool = True) -> pd.Series:
    """Daily CBOE VIX close level, used for volatility-regime classification.

    The Dataloader route does not serve this index directly under a Yahoo
    symbol; Massive's index entitlement is not guaranteed, so the VIX series
    is loaded from Yahoo Finance via ``pandas_datareader`` as the documented
    backstop for data the Dataloader/Massive path does not serve.
    """
    cache_path = CACHE_DIR / f"vix_{start_date}_{end_date}.csv"
    if use_cache and cache_path.exists():
        series = pd.read_csv(cache_path, index_col=0, parse_dates=True)["vix"]
        return series
    import yfinance as yf

    data = yf.download("^VIX", start=start_date, end=end_date, progress=False)
    if data.empty:
        raise DataloaderError("No VIX data returned by the Yahoo Finance backstop")
    close = data["Close"]
    if hasattr(close, "columns"):
        close = close.iloc[:, 0]
    series = close.rename("vix")
    series.index = series.index.tz_localize(None)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    series.to_frame().to_csv(cache_path)
    return series


# ---------------------------------------------------------------------------
# Universe construction
# ---------------------------------------------------------------------------
# Massive's reference data (``/v3/reference/tickers``) has no index-constituent
# snapshot endpoint and exposes only US-listed securities (every ticker,
# including searches for non-US names, is returned with locale "us"): foreign
# primary listings on the Tokyo Stock Exchange or Euronext are not served.
# The S&P 500 group is therefore built from a candidate list of large,
# liquid US common stocks; the Nikkei 225 and STOXX 600 groups are built from
# the largest US-listed American Depositary Receipts (ADRs, type "ADRC") of
# Japanese- and European-domiciled companies respectively. Within each
# candidate list, the final universe is the top ``n_per_group`` names ranked
# by realised mean daily dollar volume over the liquidity-ranking year, i.e.
# a data-driven approximation of "the 100 most actively traded constituents
# of 2017" (Sec. 4, "Dataset"). See results/RESULTS.md for the full
# candidate lists and the deviation this implies.
CANDIDATE_TICKERS: dict[str, list[str]] = {
    "sp500": [
        "AAPL", "MSFT", "AMZN", "GOOGL", "META", "JPM", "JNJ", "XOM", "PG", "V",
        "HD", "CVX", "MRK", "PFE", "KO", "PEP", "WMT", "BAC", "DIS", "CSCO",
        "VZ", "INTC", "T", "ABT", "CRM", "MCD", "COST", "NKE", "MDT", "UNH",
        "ADBE", "TXN", "QCOM", "HON", "IBM", "ORCL", "UPS", "LOW", "LIN", "AMGN",
        "CAT", "GS", "SBUX", "BA", "MMM", "AXP", "GE", "USB", "CL", "MO",
        "DUK", "SO", "PNC", "BLK", "SPG", "GILD", "TGT", "CVS", "LMT", "RTX",
        "ADP", "MDLZ", "ITW", "APD", "ETN", "NSC", "SYK", "ISRG", "BDX", "ZTS",
        "CME", "ICE", "SCHW", "TJX", "BSX", "EMR", "FDX", "ECL", "AON", "MMC",
        "PSA", "D", "EXC", "AEP", "KMB", "GD", "NOC", "HUM", "WM", "PGR",
        "ALL", "MET", "TRV", "AFL", "COF", "BK", "STT", "TFC", "CB", "PRU",
    ],
    "nikkei225": [
        "TM", "SONY", "HMC", "MFG", "SMFG", "MUFG", "NTT", "TAK", "NMR", "CAJ",
        "HTHIY", "KYCCF", "FANUY", "DNZOY", "OLCLY", "SHECY", "MITSY", "ITOCY",
        "MARUY", "KUBTY", "NPPXF", "KDDIY", "SZKMY", "AJINY", "ASGLY", "TOELY",
        "YAMCY", "NSANY", "MZDAY", "SUBOY",
    ],
    "stoxx600": [
        "NVS", "RHHBY", "ASML", "SAP", "AZN", "UL", "DEO", "SNY", "SIEGY", "BASFY",
        "BAYRY", "ALIZY", "DANOY", "AIQUY", "VOD", "BP", "SHEL", "TTE", "HSBC", "UBS",
        "ALVVY", "BNPQY", "AXAHY", "ENIVF", "IBDRY", "ORAN", "SGSOY", "SPGI", "SBGSY",
        "PHG",
    ],
}


def _ticker_overview(ticker: str) -> dict:
    load_dotenv()
    massive_key = os.environ.get("MASSIVE_API_KEY")
    if not massive_key:
        raise DataloaderError("MASSIVE_API_KEY must be set")
    response = requests.get(
        f"{MASSIVE_ROOT}/v3/reference/tickers/{ticker}",
        headers={"Authorization": f"Bearer {massive_key}"},
        timeout=30,
    )
    if response.status_code != 200:
        return {}
    return response.json().get("results", {})


def load_sector_map(tickers: list[str]) -> dict[str, str]:
    """SIC industry description per ticker, used as a GICS-sector substitute.

    Massive's reference data does not expose the MSCI/S&P GICS taxonomy used
    by the paper (Sec. 5, "Representation Quality Analysis"); the Standard
    Industrial Classification (SIC) description returned by Massive's ticker
    reference endpoint is used as the closest available substitute for the
    sector attribute, and only that attribute is substituted.
    """
    sectors = {}
    for ticker in tickers:
        overview = _ticker_overview(ticker)
        sectors[ticker] = overview.get("sic_description", "UNKNOWN")
    return sectors


def select_universe(
    start_date: str,
    end_date: str,
    n_per_group: int,
    candidates: dict[str, list[str]] | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Rank each candidate group by realised mean daily dollar volume and
    keep the top ``n_per_group`` tickers of each group.

    Returns a DataFrame with columns ``ticker``, ``group``, ``dollar_volume``.
    """
    candidates = candidates or CANDIDATE_TICKERS
    rows = []
    for group, group_tickers in candidates.items():
        bars, meta = load_ohlcv(group_tickers, start_date, end_date, use_cache=use_cache)
        dollar_volume = (bars["close"] * bars["volume"]).groupby(bars["ticker"]).mean()
        ranked = dollar_volume.sort_values(ascending=False)
        kept = ranked.index[: min(n_per_group, len(ranked))]
        for ticker in kept:
            rows.append({"ticker": ticker, "group": group, "dollar_volume": float(ranked[ticker])})
    return pd.DataFrame(rows)
