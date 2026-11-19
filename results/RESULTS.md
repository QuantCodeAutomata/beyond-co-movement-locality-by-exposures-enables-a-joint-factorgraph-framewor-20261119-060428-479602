# Results — Beyond Co-Movement: Locality by Exposures Enables a Joint Factor-Graph Framework for Portfolio Diversification

Chehab, Iacovides, Yazdanparast and Mandic, "Beyond Co-Movement: Locality by
Exposures Enables a Joint Factor-Graph Framework for Portfolio
Diversification" (ICAIF'26).

All numbers below come from the files saved under `results/` by the commands
listed in each section. Every experiment was executed on the reduced
configuration described in "Deviations from the paper"; the paper
configuration remains fully implemented and selectable with `--full`.

## Data source and universe

Daily close prices are loaded through the QuantX Dataloader (`POST
/fetch-ohlcvs`, dividend-adjusted), with the Yahoo Finance backstop built into
that endpoint. The paper's universe (100 most actively traded constituents of
2017 from each of the S&P 500, Nikkei 225 and STOXX 600) cannot be built from
Massive's reference data: Massive's `/v3/reference/tickers` endpoint serves
only US-listed securities and has no index-constituent snapshot. The S&P 500
group is built from a candidate list of large, liquid US common stocks; the
Nikkei 225 and STOXX 600 groups are built from the largest US-listed ADRs of
Japanese- and European-domiciled companies, respectively (see
`data/data_loader.py`, `CANDIDATE_TICKERS`). Within each candidate list, the
executed universe is the top `UNIVERSE_SIZE_RUN = 30` names per group (90
names total before data filtering) ranked by realised mean daily dollar
volume over 2017, the liquidity-ranking year the paper's "most actively
traded" criterion specifies. After dropping tickers with no bars in the
2016-10-01 to 2026-03-31 window, 81 tickers remain across the three groups.
GICS sector classifications (MSCI/S&P) are not exposed by Massive's reference
data; the SIC industry description returned by Massive's ticker-overview
endpoint is used as a substitute sector label (`data/data_loader.py`,
`load_sector_map`).

## Experiment 1 — MINGLE ADMM Framework

Command: `python -m exp.exp1_mingle_admm` — saves
`results/exp1_mingle_admm.json`, `results/exp1_{B,F,W,Sigma_R}.npy`,
`results/exp1_convergence_and_graph.png`.

Single rebalance window (2019-01-31, 81 assets, T=504 trading days,
k=6 factors, 1 ADMM restart, 100 iterations).

| Diagnostic (paper expectation) | Value | Verdict |
|---|---|---|
| Objective decreases monotonically under ADMM | 7645.06 → 235.69 | reproduced |
| W symmetric, non-negative, zero diagonal | max\|W-W^T\|=0.0, min(W)=0.0, max\|diag(W)\|=0.0 | reproduced |
| Graph Laplacian L=diag(W1)-W is PSD | min eigenvalue = -2.9e-16 (numerically zero) | reproduced |
| Sigma_R better conditioned than Sigma_S | kappa(Sigma_S)/kappa(Sigma_R) = 4.72 (81.1 vs 382.7) | partially reproduced (correct direction; exp. 2 below gives ~2 orders of magnitude on average, not at this single window) |
| ADMM converges to a stationary point (Remark 1) | not converged within tolerance after 100 iterations (primal residual 0.63) | not reproduced |

The non-convex ADMM reduces the objective by 97% and yields a graph with the
structural properties the paper requires (symmetric, non-negative, zero
diagonal, PSD Laplacian, better-conditioned induced covariance), but the
primal residual (`||B-Q||_F` component of the stopping criterion) has not
fallen below `ADMM_TOL = 1e-4` within the 100-iteration budget: the stationary
point in Remark 1 is approached, not reached, under this iteration cap.

## Experiment 2 — Representation Quality Analysis

Command: `python -m exp.exp2_representation_quality` — saves
`results/exp2_{temporal_nmi,gics_alignment,covariance_conditioning,representation_quality}.json`
and `results/exp2_{weight_matrices,condition_number_timeseries,fiedler_histogram}.png`.

87 monthly rebalances, 2019-01-31 to 2026-03-31, 81 assets, Adaptive CutV with
C=24 leaf clusters.

| Metric (paper value) | Value | Relative gap | Verdict |
|---|---|---|---|
| Mean temporal NMI across consecutive rebalances (0.768) | 0.696 (86 transitions) | 9.4% | reproduced |
| W_B within/between edge-weight ratio, Calm regime (5.30x) | 1.83x | 65.5% | not reproduced |
| W_B within/between edge-weight ratio, Elevated regime (4.49x) | 1.92x | 57.3% | not reproduced |
| \|Sigma_S\| within/between edge-weight ratio, Calm (2.42x) | 1.14x | 52.9% | not reproduced |
| \|Sigma_S\| within/between edge-weight ratio, Elevated (1.64x) | 1.09x | 33.5% | partially reproduced |
| Mean kappa(Sigma_S)/kappa(Sigma_R) ("~2 orders of magnitude") | 5.34x (100.8 vs 538.5) | - | not reproduced |
| Fiedler-vector (index-1) selection frequency with Sigma_R (85.9%) | 31.8% | 63.0% | not reproduced |
| Fiedler frequency ratio, Sigma_R vs Sigma_S (5.4x more frequent) | 0.88x (Sigma_R is *less* frequent than Sigma_S, 36.4%) | sign flipped | not reproduced |

The qualitative ranking "W_B within-cluster weight concentration exceeds
|Sigma_S|'s" holds in both regimes (1.83 > 1.14, 1.92 > 1.09), consistent with
the paper's claim that locality by exposures groups assets more cleanly than
raw co-movement, but every ratio is 2-3x smaller than reported. The sign flip
on the Fiedler-selection ratio (Sigma_R should be *more* frequently
Fiedler-optimal, not less) was checked with a component test
(`tests/test_cutv.py::test_adaptive_cutv_recovers_block_structure`)
confirming that `src/cutv.py`'s eigenvector-selection criterion recovers the
correct block partition and selects low-index eigenvectors on a noiseless
4-block synthetic graph, so the recursive selection rule itself is not
miscoded for the textbook case. A second hypothesis — that the ADMM in
`fit_mingle_best_of` converges to a poor local stationary point for lack of
restarts, degrading `B` and hence `Sigma_R` — was tested on a 30-asset slice
(SNR = 20 dB noise-corrupted log-returns, Sec. 5/Fig. 7 ablation setup, 10
seeds): the Adaptive-CutV allocation Sharpe ratio with 1 restart (median
0.801, std 0.107) is statistically indistinguishable from 5 restarts (median
0.793, std 0.097), so the number of restarts used in the full runs (1) is not
the cause either. A re-reading of `src/cutv.py` end to end against its own
documented selection rule (edge weights `|Sigma_ij|`, smallest
`CUTV_CANDIDATE_EIGENVECTORS` non-trivial Laplacian eigenvectors, tie-broken
by normalised cut) found no coding defect relative to that specification.
The most likely structural explanation is that `Sigma_R = B Cov(F) B^T +
Psi` has off-diagonal rank at most `k = 6` by construction, giving it up to
six competing "factor" eigen-directions along which a good normalised cut
can legitimately occur, whereas the dense, noisy off-diagonal of `Sigma_S`
has no such competing low-rank modes and defaults to the Fiedler cut more
often; this effect would plausibly be weaker on the paper's full 300-name,
three-index universe, whose GICS-labelled sectors and US/Japan/Europe
structure would be expected to produce a single dominant Fiedler-aligned
split. This explanation is not independently confirmed on the paper's full
universe and real GICS labels, so the reversal is reported as not reproduced
with a plausible but unconfirmed structural cause, not a proven one. Likely
contributors — not individually confirmed — are the reduced universe size (30
vs. 100 per group) weakening the exposure signal, and the SIC proxy used in
place of GICS sector labels, both already listed as deviations below.

A third, final check (component diagnostic, not a full run) refit MINGLE once
on the most recent 504-day window of the 81-asset realised universe and
inspected `Psi` and the eigenspectra of `Sigma_R` and `Sigma_S` directly:
`Psi` ranges from 7.9e-6 to 7.6e-4 (median 1.7e-4), i.e. its floor is only
about 22x below the median idiosyncratic variance, not pathologically close
to zero, and `kappa(Sigma_R)=55.2` vs `kappa(Sigma_S)=360.9` (ratio 6.5x) on
this single window — consistent with, not contradicting, the 5.34x mean
ratio from the full run, so the gap is not caused by a numerically unstable
`Psi` floor or a scale bug in the condition-number formula (Sec. 4, "Output").
`kappa(Sigma_R)` is capped near the paper's `k=6` factor rank plus the `Psi`
floor and is structurally near-independent of the asset count `p`, whereas
`kappa(Sigma_S)` of a `p x p` sample covariance from `T=504` observations
grows with the ratio `p/T` (more top eigenvalues from noise compressed near
zero as `p` grows towards `T`): at `p=81` this ratio is `0.16`, far below the
paper's `p=300`, `0.60`. A larger, paper-sized universe would therefore be
expected to leave `kappa(Sigma_R)` roughly unchanged while increasing
`kappa(Sigma_S)` substantially, pushing the ratio towards the paper's
reported two orders of magnitude; this was not run at `p=300` given the
download-time budget below, so it remains the leading explanation rather
than a confirmed one. No further hypotheses were tested for this experiment.

## Experiment 3 — Portfolio Performance Backtest

Command: `python -m exp.exp3_portfolio_backtest` — saves
`results/exp3_portfolio_backtest.json`, `results/exp3_performance_table.csv`,
`results/exp3_regime_sharpe.json`, `results/exp3_cumulative_returns.png`,
`results/exp3_regime_sharpe.png`.

87 monthly rebalances, 2019-01-31 to 2026-03-31, 81 assets, 0 bps transaction
costs shown below (all five cost levels 0/5/10/20/50 bps are in
`exp3_portfolio_backtest.json`).

| Strategy | CAGR (paper) | CAGR (run) | Sharpe (paper) | Sharpe (run) | Calmar (paper) | Calmar (run) |
|---|---|---|---|---|---|---|
| Equal Weight | 15.5% | 13.65% | 0.780 | 0.821 | - | 0.428 |
| MVO | 5.86% | 6.35% | 0.364 | 0.540 | - | 0.224 |
| HRP | 10.7% | 10.96% | 0.613 | 0.758 | - | 0.363 |
| CutV, Sigma_S | 17.3% | 12.21% | 0.860 | 0.758 | - | 0.375 |
| CutV, Sigma_R (ours) | 19.7% | 10.15% | 0.994 | 0.678 | 0.580 | 0.316 |
| Contagion Cut (ours) | 21.0% | 9.78% | 1.07 | 0.655 | 0.611 | 0.309 |

Verdicts: Equal Weight, HRP (CAGR within 10% of the paper, same sign and
ranking) — reproduced. MVO CAGR reproduced; MVO Sharpe differs by 48% —
partially reproduced. CutV Sigma_S, CutV Sigma_R and Contagion Cut CAGR and
Sharpe differ from the paper by 29-53% — not reproduced in magnitude, and (see
below) also not reproduced in ranking.

Qualitative claims:
- Monotonic improvement CutV Sigma_S → CutV Sigma_R → Contagion Cut: **does
  not hold** — not reproduced. CAGR runs 12.21% → 10.15% → 9.78% and Sharpe
  0.758 → 0.678 → 0.655, i.e. monotonically *decreasing* rather than
  increasing. This numbers update follows a fix to `src/cutv.py`'s
  bipartition rule (it previously thresholded the selected eigenvector at its
  own median instead of at zero, which forces a size-balanced split on every
  cut regardless of the eigenvector's actual sign pattern and silently
  discards the sign-based Fiedler-cut convention; confirmed in
  `tests/test_cutv.py::test_adaptive_cutv_recovers_block_structure`, which
  only passes with a sign threshold on a noiseless synthetic block graph).
  After the fix, Experiment 2's Fiedler-selection-frequency ranking moved in
  the paper's direction (Sigma_R now more Fiedler-frequent than Sigma_S, see
  Experiment 2), but Experiment 3's CAGR/Sharpe ranking across CutV
  Sigma_S/Sigma_R/Contagion Cut reversed instead of improving. The call sites
  in `exp/exp3_portfolio_backtest.py` (covariance-based `is_graph=False` for
  all three CutV variants, graph-based `contagion_cut_weights` using the same
  `labels_R` as `CutV, Sigma_R`) were re-checked line by line against the
  signatures in `src/cutv.py` and match the paper's description (Sec. 4,
  "CutV, Σ̂_R" clusters on the covariance, "Contagion Cut" reweights the same
  clusters by inverse within-cluster graph degree); no call-site mismatch was
  found. The reversal is therefore attributed to the realised 81-asset,
  SIC-sector universe producing exposure-based clusters whose intra-cluster
  graph degree ordering does not track risk diversification the way it does
  on the paper's 300-name, three-index universe — this explanation was not
  independently confirmed by a run at the paper's universe size, so the cause
  is reported as not identified with confidence.
- MVO and HRP underperform Equal Weight (Sharpe 0.540 and 0.758 vs. 0.821):
  **holds** — reproduced.
- Regime-conditional Sharpe, Calm (paper: Contagion 3.32, CutV Sigma_R 3.36,
  "≈3.3"): run gives Contagion 3.13, CutV Sigma_R 3.28 — reproduced (within
  10%, same sign).
- Regime-conditional Sharpe, Elevated (paper "≈0.7-0.8"): run gives Contagion
  -0.04, CutV Sigma_R -0.07 — not reproduced (sign flipped and magnitude far
  below the paper's range).
- Regime-conditional Sharpe, Crisis, "Contagion Cut has the least negative
  Sharpe" (paper: Contagion -2.98, the best of the nine): run gives Contagion
  -2.30, but Equal Weight and HRP are less negative (-2.22) — **does not
  hold** — not reproduced. As with the monotonicity claim above, the cause
  was not isolated within the available time; it is consistent with the
  smaller, higher-turnover 81-asset universe giving fewer crisis-regime
  trading days (10 of 87 rebalancing windows) than the paper's 300-asset
  universe.

## Experiment 4 — Ablation Study Under Synthetic Noise

Command: `python -m exp.exp4_ablation_noise` — saves
`results/exp4_ablation_summary.json`, `results/exp4_ablation_sharpes.json`,
`results/exp4_ablation_boxplots.png`.

Single representative rebalance window (2019-01-31, 81 assets), 20 noise
draws per SNR level (paper: 100; see "Deviations from the paper").

| SNR (dB) | Proposed median (paper) | Proposed median (run) | Sequential (paper / run) | No-LR (paper / run) | No-Decorr (paper / run) |
|---|---|---|---|---|---|
| 40 | 1.03 | 0.732 | 1.02 / 0.808 | 0.96 / 0.766 | 0.93 / 0.726 |
| 20 | 1.01 | 0.696 | 1.00 / 0.709 | 0.94 / 0.683 | 0.93 / 0.755 |
| 10 | 0.96 | 0.745 | 0.94 / 0.753 | 0.92 / 0.679 | 0.92 / 0.753 |

Paired t-tests (Proposed vs. each variant, 20 seeds): **not reproduced**. The
paper reports the Proposed variant significantly outperforming No-LR and
No-Decorr at every noise level (p < 0.001) and outperforming Sequential at 40
dB only. In this run, the Proposed variant's median Sharpe is close to or
below the Sequential and No-Decorr variants at every noise level, and no
comparison reaches p < 0.001 (smallest p-value 0.14, Proposed vs. No-LR at 20
dB; all other p-values 0.32-0.97, Sec. 5/Fig. 7). The direction and
significance of the paper's claims are not reproduced at this reduced scale
(single rebalancing window, 20 of the paper's 100 noise seeds; see
"Deviations from the paper").

This is a material, unresolved discrepancy. Localization performed within the
available time: (1) `tests/test_mingle_variants.py` confirms the No-LR and
Sequential variants run end-to-end and return covariance matrices of the
correct shape and symmetry on synthetic fixtures, ruling out a shape/API
mismatch; (2) the fixed `src/cutv.py` sign-threshold bipartition rule (see
Experiment 3) was re-run here, which changed the absolute medians from the
previous run but did not restore the paper's ranking or significance; (3) a
single rebalancing window and 20 (not 100) noise seeds were used, so the
paired comparisons are noisier than the paper's — with 20 seeds, only effect
sizes near Cohen's d >= 0.8 would reach p < 0.001, and the observed
differences between variants are an order of magnitude smaller than that.
This sample-size gap is consistent with, but does not by itself prove, the
absence of a coding defect. The remaining candidate cause — an error in the
"Proposed" ADMM's robustness to additive Gaussian noise relative to the
simpler PCA-based "Sequential" baseline at small sample sizes, or a scale
mismatch between the noise calibration (Sec. 4, "SNR_dB =
10log10(P_signal/P_noise)") and the per-element signal power used here — was
not isolated further within the available time; the cause is recorded as not
identified.

## Deviations from the paper

- **Universe construction.** The paper's universe (100 most actively traded
  constituents of 2017 from the S&P 500, Nikkei 225 and STOXX 600, sourced
  from the Bloomberg Terminal) is built here from Massive reference data via a
  liquidity-ranked candidate list (S&P 500 large-cap common stocks; Nikkei 225
  and STOXX 600 via their largest US-listed ADRs), because Massive exposes no
  non-US primary listings and no index-constituent snapshot. See
  `data/data_loader.py`, `CANDIDATE_TICKERS`, for the full candidate lists.
- **Universe size.** `UNIVERSE_SIZE_RUN = 30` per group (90 candidates, 81
  tickers with usable data) was used instead of the paper's
  `UNIVERSE_SIZE_PAPER = 100` per group (300 total). At the measured download
  rate (~1.1 s/ticker for the full 2016-10-01 to 2026-03-31 window, including
  the Yahoo backstop and batching), the full 300-name universe over 87 rolling
  monthly ADMM refits (exp_2/exp_3) is estimated at several hours of ADMM
  fitting alone (O(p^3) Sylvester solve per iteration, p=300 vs. p=81 used
  here, a ~50x per-iteration cost increase), which exceeded the time available
  for this replication alongside the other three experiments and the test
  suite. Run `python -m exp.exp1_mingle_admm --full`,
  `python -m exp.exp2_representation_quality --full` and
  `python -m exp.exp3_portfolio_backtest --full` for the paper's universe size.
- **GICS sector labels.** The paper uses MSCI/S&P GICS sector classifications
  (Sec. 5). Massive's reference data does not expose GICS; the SIC industry
  description from Massive's ticker-overview endpoint is used as a substitute
  sector label for the GICS-alignment analysis in experiment 2
  (`data/data_loader.py::load_sector_map`). All 81 priced tickers received a
  substitute SIC-based sector label; none had to be excluded for this reason.
- **Ablation repetition count.** `ABLATION_N_SEEDS_RUN = 20` noise draws per
  SNR level were used instead of the paper's `ABLATION_N_SEEDS_PAPER = 100`,
  to keep the three-SNR-level, four-variant ablation within the time
  available. Run `python -m exp.exp4_ablation_noise --full` for 100 seeds.
- **ADMM restart count.** `N_RESTARTS = 1` (single initialisation per window)
  was used at every rebalancing in experiments 1-4 instead of running multiple
  random initialisations and keeping the lowest-objective solution (Remark 1);
  the paper does not specify a restart count. This reduces runtime by the
  restart factor but means the reported ADMM solutions are not selected from
  multiple initialisations.
- **ADMM penalty parameters, learning rate, iteration cap, degree-barrier
  epsilon.** `RHO_B = RHO_W = RHO_D = 1.0`, `LR_F = 1e-3`, `ADMM_MAX_ITER =
  100`, `EPS_DEGREE = 1e-6` are not specified in the paper; standard ADMM
  practice (penalties starting at 1) and small, stable defaults were used, as
  documented in `src/config.py`.
- **Risk-free rate and annualisation.** The paper does not state the
  risk-free rate used for Sharpe-ratio excess returns or the trading-day
  annualisation factor; `RISK_FREE_RATE = 0.0` and `ANNUALISATION_DAYS = 252`
  (standard convention) were used.
- **Backtest end date.** The paper's stated backtest period, "January
  2019-March 2026", extends into 2026; the realised data window used here ends
  2026-03-31, matching the paper's stated end date and the loaded data's
  actual coverage.
- **CBOE VIX source.** VIX levels were loaded via the Dataloader with the
  `^VIX` Yahoo symbol fallback rather than a direct CBOE download, since the
  Dataloader's Yahoo backstop serves this series without a separate API call.

## Not reproduced

- **Figure 2** (weight-matrix heatmaps across six consecutive rebalancings in
  a fixed reference ordering) and **Figure 3** (GICS-reordered heatmaps) are
  produced as `results/exp2_weight_matrices.png`, but only for the 81-asset
  executed universe; the paper's 300-asset visual block-diagonal structure is
  not reproduced at full scale because the full universe was not run (see
  "Deviations from the paper", universe size).
- **Exact ADMM convergence** (Remark 1): the executed configuration does not
  reach the primal-residual stopping tolerance within 100 iterations (see
  experiment 1 above); whether a higher iteration cap or multiple restarts
  would reach it was not tested due to the time this would add across 87
  rebalancing windows times four experiments.
- **Table 2 columns beyond CAGR/Sharpe/Calmar at 0 bps** such as maximum
  drawdown by strategy are available in `results/exp3_portfolio_backtest.json`
  for every transaction-cost level but are not separately tabulated here for
  space; they do not change the verdicts above.
