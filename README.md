# Beyond Co-Movement: Locality by Exposures Enables a Joint Factor-Graph Framework for Portfolio Diversification

Replication of Chehab, Iacovides, Yazdanparast and Mandic, "Beyond Co-Movement:
Locality by Exposures Enables a Joint Factor-Graph Framework for Portfolio
Diversification" (ICAIF'26).

This repository implements MINGLE, an ADMM-based joint factor-graph model
that redefines graph locality through systematic factor-exposure profiles
rather than observed co-movement, and evaluates it against standard portfolio
allocation benchmarks.

## Experiments

1. **MINGLE ADMM framework** (`exp/exp1_mingle_admm.py`) — fits the joint
   factor-graph model (Sec. 3, Eq. 5-8, Algorithm 1) on the equity universe
   and reports convergence diagnostics, the exposure/graph/covariance
   artifacts, and structural checks (Laplacian PSD, sparsity, conditioning).
2. **Representation quality analysis** (`exp/exp2_representation_quality.py`)
   — temporal community stability (NMI), GICS/SIC sector alignment of the
   exposure-similarity graph versus the correlation graph, and covariance
   conditioning / Fiedler-selection frequency (Sec. 5, Figures 2-4).
3. **Portfolio performance backtest** (`exp/exp3_portfolio_backtest.py`) —
   nine allocation strategies, five transaction-cost levels, and VIX-regime
   decomposition (Sec. 5, Table 2, Figures 5-6).
4. **Ablation under synthetic noise** (`exp/exp4_ablation_noise.py`) —
   Proposed vs. Sequential, No-LR and No-Decorr variants at three SNR levels
   with paired t-tests over repeated noise draws (Fig. 7).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in DATALOADER_API_KEY and MASSIVE_API_KEY
```

## Tests

```bash
python -m pytest -m "not market_data"   # fast unit tests, no network
python -m pytest                        # includes market-data tests (needs API keys)
```

## Running the experiments

```bash
python -m exp.exp1_mingle_admm
python -m exp.exp2_representation_quality
python -m exp.exp3_portfolio_backtest
python -m exp.exp4_ablation_noise
```

Each command accepts `--full` to run the paper's full configuration (universe
size, repetition counts) instead of the reduced configuration used to fit the
available compute time; see `results/RESULTS.md` for the exact parameters of
each run and `src/config.py` for every paper-derived constant.

All outputs (tables, metrics, figures) are written directly to `results/`,
overwriting previous files from the same command.

## Repository layout

```
src/      model, allocation, covariance, backtest and metrics modules
exp/      one script per experiment, calling only src/ and data/
data/     data_loader.py (QuantX Dataloader + Massive REST market data)
tests/    pytest unit tests (synthetic fixtures) and market-data tests
results/  RESULTS.md and generated tables/figures
```

See `AGENTS.md` for a guide to the repository layout and conventions.
