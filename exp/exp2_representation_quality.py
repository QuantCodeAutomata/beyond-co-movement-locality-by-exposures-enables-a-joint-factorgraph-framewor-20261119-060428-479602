"""Experiment 2 - Representation Quality Analysis (Sec. 5, Fig. 2-4).

For every monthly rebalance of the paper's backtest (Sec. 4-5), fits MINGLE
on a rolling 2-year lookback window of log-returns and compares the
representation-induced covariance Sigma_R against the sample covariance
Sigma_S on four axes:

  * temporal stability of the Adaptive CutV cluster assignment across
    consecutive rebalances, measured by normalised mutual information (NMI);
  * alignment of the learnt graph (and of |Sigma_S|) with the GICS-sector
    substitute (SIC industry description, see ``data.data_loader.
    load_sector_map``), via the within/between-sector mean-edge-weight ratio,
    split by VIX regime (Remark 2);
  * covariance conditioning (condition number) of Sigma_R vs Sigma_S over
    time;
  * Fiedler-vector (first non-trivial Laplacian eigenvector) selection
    frequency under Adaptive CutV recursive bipartitioning (Sec. 5.1).

Run: python -m exp.exp2_representation_quality [--full]
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
from sklearn.metrics import normalized_mutual_info_score

from data.data_loader import load_close_panel, load_sector_map, load_vix, select_universe
from src import config
from src.covariance import condition_number, sample_covariance
from src.cutv import adaptive_cutv
from src.metrics import classify_regime
from src.mingle import fit_mingle_best_of
from src.pipeline import log_returns, lookback_window, monthly_rebalance_dates, simple_returns

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def sector_alignment_ratio(matrix: np.ndarray, sector_labels: list[str]) -> float:
    """Mean within-sector / mean between-sector ratio of off-diagonal entries
    of ``matrix`` (Sec. 5, GICS/sector alignment analysis), given one sector
    label per asset in the same row/column order as ``matrix``.

    Returns ``inf`` if the between-sector mean is zero or undefined (e.g. a
    single sector present), so the ratio stays well-defined and large rather
    than a division error.
    """
    p = matrix.shape[0]
    labels = np.asarray(sector_labels)
    same = labels[:, None] == labels[None, :]
    offdiag = ~np.eye(p, dtype=bool)
    same_offdiag = same & offdiag
    diff_offdiag = (~same) & offdiag
    within_mean = float(matrix[same_offdiag].mean()) if same_offdiag.any() else float("nan")
    between_mean = float(matrix[diff_offdiag].mean()) if diff_offdiag.any() else float("nan")
    if not np.isfinite(between_mean) or between_mean == 0.0:
        return float("inf")
    return within_mean / between_mean


def run(full: bool) -> dict:
    universe_size = config.UNIVERSE_SIZE_PAPER if full else config.UNIVERSE_SIZE_RUN
    n_restarts = 3 if full else config.N_RESTARTS

    print(f"[exp2] universe_size_per_group={universe_size} full={full}")
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
    print(f"[exp2] realised universe size={len(tickers)} "
          f"(groups={universe['group'].value_counts().to_dict()})")

    close_panel = load_close_panel(tickers, config.SAMPLE_DOWNLOAD_START, config.BACKTEST_END_RUN)
    returns_log = log_returns(close_panel)
    returns_simple = simple_returns(close_panel)
    print(f"[exp2] close panel rows={len(close_panel)} "
          f"date_range=({close_panel.index.min().date()},{close_panel.index.max().date()}) "
          f"assets={close_panel.shape[1]}")

    vix = load_vix(config.BACKTEST_START, config.BACKTEST_END_RUN)
    print(f"[exp2] vix rows={len(vix)}")

    sector_map = load_sector_map(list(close_panel.columns))
    sector_labels = [sector_map.get(ticker, "UNKNOWN") for ticker in close_panel.columns]
    print(f"[exp2] sector map loaded for {len(sector_map)} tickers, "
          f"{len(set(sector_labels))} unique sectors")

    reb_dates = monthly_rebalance_dates(
        returns_log.index, config.BACKTEST_START, config.BACKTEST_END_RUN, config.LOOKBACK_TRADING_DAYS
    )
    if not reb_dates:
        raise ValueError("No rebalance date has a full lookback window available")
    print(f"[exp2] n_rebalances={len(reb_dates)} n_restarts={n_restarts}")

    # Keep only six consecutive rebalances roughly in the middle of the
    # backtest for the Figure-2-style weight-matrix snapshot, to avoid
    # accumulating 80+ dense p x p matrices in memory.
    mid_start = max(0, len(reb_dates) // 2 - 3)
    snapshot_indices = set(range(mid_start, min(mid_start + 6, len(reb_dates))))

    per_rebalance_conditioning: list[dict] = []
    nmi_series: list[float] = []
    gics_records: list[dict] = []
    all_selected_R: list[int] = []
    all_selected_S: list[int] = []
    first_cut_fiedler_R: list[int] = []
    weight_snapshots: list[tuple[pd.Timestamp, np.ndarray]] = []
    prev_labels_R: np.ndarray | None = None

    for i, reb_date in enumerate(reb_dates):
        X_log = lookback_window(returns_log, reb_date, config.LOOKBACK_TRADING_DAYS).to_numpy().T
        result = fit_mingle_best_of(
            X_log,
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
        )
        simple_window = lookback_window(returns_simple, reb_date, config.LOOKBACK_TRADING_DAYS)
        Sigma_S = sample_covariance(simple_window)

        labels_R, selected_R = adaptive_cutv(result.Sigma_R, n_clusters=config.N_CLUSTERS, is_graph=False)
        labels_S, selected_S = adaptive_cutv(Sigma_S, n_clusters=config.N_CLUSTERS, is_graph=False)

        kappa_R = condition_number(result.Sigma_R)
        kappa_S = condition_number(Sigma_S)
        per_rebalance_conditioning.append({
            "date": str(reb_date.date()), "kappa_R": kappa_R, "kappa_S": kappa_S,
        })

        all_selected_R.extend(selected_R)
        all_selected_S.extend(selected_S)
        first_cut_fiedler_R.append(int(selected_R[0] == 1) if selected_R else 0)

        if i in snapshot_indices:
            weight_snapshots.append((reb_date, result.W.copy()))

        ratio_W_B = sector_alignment_ratio(result.W, sector_labels)
        ratio_Sigma_S = sector_alignment_ratio(np.abs(Sigma_S), sector_labels)
        vix_value = vix.asof(reb_date)
        regime = classify_regime(float(vix_value)) if pd.notna(vix_value) else "unknown"
        gics_records.append({
            "date": str(reb_date.date()), "regime": regime,
            "ratio_W_B": ratio_W_B, "ratio_Sigma_S": ratio_Sigma_S,
        })

        if prev_labels_R is not None:
            nmi = normalized_mutual_info_score(prev_labels_R, labels_R)
            nmi_series.append(float(nmi))
        prev_labels_R = labels_R

        print(f"[exp2] rebalance {i + 1}/{len(reb_dates)} date={reb_date.date()} "
              f"kappa_R={kappa_R:.2f} kappa_S={kappa_S:.2f} regime={regime}")

    p = close_panel.shape[1]

    mean_temporal_nmi = float(np.mean(nmi_series)) if nmi_series else float("nan")
    print(f"[exp2] mean_temporal_nmi={mean_temporal_nmi:.4f} (paper: ~0.768) "
          f"n_transitions={len(nmi_series)}")

    gics_frame = pd.DataFrame(gics_records)
    gics_alignment: dict = {}
    for regime in ["calm", "elevated", "crisis"]:
        subset = gics_frame[gics_frame["regime"] == regime]
        if len(subset) == 0:
            continue
        gics_alignment[regime] = {
            "n_windows": int(len(subset)),
            "mean_ratio_W_B": float(subset["ratio_W_B"].replace([np.inf, -np.inf], np.nan).mean()),
            "mean_ratio_Sigma_S": float(subset["ratio_Sigma_S"].replace([np.inf, -np.inf], np.nan).mean()),
        }
        print(f"[exp2] regime={regime} n_windows={gics_alignment[regime]['n_windows']} "
              f"mean_ratio_W_B={gics_alignment[regime]['mean_ratio_W_B']:.3f} "
              f"(paper Calm~5.30, Elevated~4.49) "
              f"mean_ratio_Sigma_S={gics_alignment[regime]['mean_ratio_Sigma_S']:.3f} "
              f"(paper Calm~2.42, Elevated~1.64)")

    kappa_R_values = [rec["kappa_R"] for rec in per_rebalance_conditioning]
    kappa_S_values = [rec["kappa_S"] for rec in per_rebalance_conditioning]
    mean_kappa_R = float(np.mean(kappa_R_values))
    mean_kappa_S = float(np.mean(kappa_S_values))
    mean_ratio_S_over_R = float(mean_kappa_S / mean_kappa_R) if mean_kappa_R > 0 else float("inf")

    all_selected_R_arr = np.asarray(all_selected_R)
    all_selected_S_arr = np.asarray(all_selected_S)
    fiedler_frequency_R = float((all_selected_R_arr == 1).mean()) if len(all_selected_R_arr) else float("nan")
    fiedler_frequency_S = float((all_selected_S_arr == 1).mean()) if len(all_selected_S_arr) else float("nan")
    fiedler_frequency_ratio = (
        fiedler_frequency_R / fiedler_frequency_S if fiedler_frequency_S else float("inf")
    )
    first_cut_fiedler_frequency_R = float(np.mean(first_cut_fiedler_R)) if first_cut_fiedler_R else float("nan")
    print(f"[exp2] mean_kappa_R={mean_kappa_R:.2f} mean_kappa_S={mean_kappa_S:.2f} "
          f"mean_ratio_S_over_R={mean_ratio_S_over_R:.2f}")
    print(f"[exp2] fiedler_frequency_R={fiedler_frequency_R:.3f} "
          f"fiedler_frequency_S={fiedler_frequency_S:.3f} "
          f"fiedler_frequency_ratio={fiedler_frequency_ratio:.2f} "
          f"(paper: ~85.9% selected, ~5.4x) "
          f"first_cut_fiedler_frequency_R={first_cut_fiedler_frequency_R:.3f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    temporal_nmi_payload = {
        "mean_temporal_nmi": mean_temporal_nmi,
        "n_transitions": len(nmi_series),
        "nmi_series": nmi_series,
    }
    with open(RESULTS_DIR / "exp2_temporal_nmi.json", "w") as handle:
        json.dump(temporal_nmi_payload, handle, indent=2)

    with open(RESULTS_DIR / "exp2_gics_alignment.json", "w") as handle:
        json.dump(gics_alignment, handle, indent=2)

    conditioning_payload = {
        "per_rebalance": per_rebalance_conditioning,
        "mean_kappa_R": mean_kappa_R,
        "mean_kappa_S": mean_kappa_S,
        "mean_ratio_S_over_R": mean_ratio_S_over_R,
        "fiedler_frequency_R": fiedler_frequency_R,
        "fiedler_frequency_S": fiedler_frequency_S,
        "fiedler_frequency_ratio": fiedler_frequency_ratio,
        "first_cut_fiedler_frequency_R": first_cut_fiedler_frequency_R,
    }
    with open(RESULTS_DIR / "exp2_covariance_conditioning.json", "w") as handle:
        json.dump(conditioning_payload, handle, indent=2)

    # Figure 2 style: persistence of the learnt graph's block structure
    # across six consecutive monthly rebalances.
    fig, axes = plt.subplots(1, len(weight_snapshots), figsize=(4 * max(len(weight_snapshots), 1), 4.5))
    if len(weight_snapshots) == 1:
        axes = [axes]
    for ax, (date, W) in zip(axes, weight_snapshots):
        im = ax.imshow(W, cmap="viridis")
        ax.set_title(str(date.date()))
    fig.suptitle("MINGLE graph W across consecutive rebalances")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp2_weight_matrices.png", dpi=150)
    plt.close(fig)

    # Figure 4 style: condition-number time series.
    dates = [pd.Timestamp(rec["date"]) for rec in per_rebalance_conditioning]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(dates, kappa_R_values, label="Sigma_R (MINGLE)", marker="o", markersize=3)
    ax.plot(dates, kappa_S_values, label="Sigma_S (sample)", marker="o", markersize=3)
    ax.set_yscale("log")
    ax.set_title("Covariance condition number: Sigma_R vs Sigma_S")
    ax.set_xlabel("Rebalance date")
    ax.set_ylabel("Condition number (log scale)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp2_condition_number_timeseries.png", dpi=150)
    plt.close(fig)

    # Fiedler-vector selection-frequency histogram, Sigma_R vs Sigma_S.
    max_index = int(max(all_selected_R_arr.max() if len(all_selected_R_arr) else 1,
                         all_selected_S_arr.max() if len(all_selected_S_arr) else 1))
    bins = np.arange(1, max_index + 2)
    counts_R, _ = np.histogram(all_selected_R_arr, bins=bins)
    counts_S, _ = np.histogram(all_selected_S_arr, bins=bins)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    width = 0.4
    ax.bar(bins[:-1] - width / 2, counts_R, width=width, label="Sigma_R (MINGLE)")
    ax.bar(bins[:-1] + width / 2, counts_S, width=width, label="Sigma_S (sample)")
    ax.set_xlabel("Selected eigenvalue index (1 = Fiedler vector)")
    ax.set_ylabel("Number of recursive cuts")
    ax.set_title("Adaptive CutV eigenvector-selection frequency")
    ax.set_xticks(bins[:-1])
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp2_fiedler_histogram.png", dpi=150)
    plt.close(fig)

    summary = {
        "mean_temporal_nmi": mean_temporal_nmi,
        "n_transitions": len(nmi_series),
        "gics_alignment": gics_alignment,
        "mean_kappa_R": mean_kappa_R,
        "mean_kappa_S": mean_kappa_S,
        "mean_ratio_S_over_R": mean_ratio_S_over_R,
        "fiedler_frequency_R": fiedler_frequency_R,
        "fiedler_frequency_S": fiedler_frequency_S,
        "fiedler_frequency_ratio": fiedler_frequency_ratio,
        "first_cut_fiedler_frequency_R": first_cut_fiedler_frequency_R,
        "universe_size_per_group": universe_size,
        "n_assets": p,
        "n_rebalances": len(reb_dates),
        "n_restarts": n_restarts,
        "full": full,
    }
    print(f"[exp2] summary: {json.dumps(summary, indent=2)}")
    with open(RESULTS_DIR / "exp2_representation_quality.json", "w") as handle:
        json.dump(summary, handle, indent=2)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Run the paper's full universe size and restart count")
    args = parser.parse_args()
    run(full=args.full)
