# Repository guide for coding agents and contributors

This repository replicates Chehab, Iacovides, Yazdanparast and Mandic,
"Beyond Co-Movement: Locality by Exposures Enables a Joint Factor-Graph
Framework for Portfolio Diversification" (ICAIF'26): the MINGLE ADMM
factor-graph model, its representation-quality diagnostics, a multi-strategy
portfolio backtest built on it, and an ablation study under synthetic noise.

## Layout

- `src/` — flat library modules: `config.py` (every paper-derived parameter,
  with a source comment), `mingle.py` (ADMM factor-graph estimation, Algorithm
  1), `mingle_variants.py` (Sequential, No-LR, No-Decorr ablation variants),
  `cutv.py` (Adaptive CutV spectral partitioning and Contagion Cut weighting),
  `allocation.py` (Equal Weight, MVO, Ledoit-Wolf MVO, HRP), `backtest.py`
  (rolling rebalancing, transaction costs, performance metrics, VIX regimes).
- `exp/` — one script per experiment (`exp1_mingle_admm.py` ...
  `exp4_ablation_noise.py`), each calling only the `src/` functions above and
  writing its artifacts to `results/`.
- `data/` — `data_loader.py` loads OHLCV bars through the QuantX Dataloader
  REST API and builds the equity universe and SIC sector map from Massive
  reference data; no other data-loading code exists outside this file.
- `tests/` — unit tests for every `src/` module on synthetic fixtures, plus
  market-data tests (marked `market_data`) that load a small real slice
  through `data/data_loader.py` and are skipped without API keys.
- `results/` — every table, figure and `RESULTS.md` produced by the
  experiments; rerunning an experiment command overwrites its files here.

## Commands

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in DATALOADER_API_KEY and MASSIVE_API_KEY

python -m pytest -m "not market_data"   # unit tests, no network
python -m pytest                        # all tests, needs API keys

python -m exp.exp1_mingle_admm
python -m exp.exp2_representation_quality
python -m exp.exp3_portfolio_backtest
python -m exp.exp4_ablation_noise
# add --full to any of the above for the paper's full universe size /
# repetition count instead of the reduced configuration described in
# results/RESULTS.md

./reproduce.sh   # tests + all four experiments, same commands as above
```

## Conventions

- Every experiment parameter lives in `src/config.py` as a module-level
  `UPPER_CASE` variable with a comment naming its meaning, unit and source
  (a paper section/equation, or "not specified in the paper"). No parameter
  is hardcoded inside `src/` or `exp/` modules.
- Market data is loaded only through `data/data_loader.py`
  (`load_ohlcv`, `build_universe`, `load_sector_map`, `load_vix`). No other
  module calls the Dataloader or Massive APIs directly, and downloaded bars
  are never committed (see `.gitignore`, `data/cache/`).
- API keys (`DATALOADER_API_KEY`, `MASSIVE_API_KEY`) are read from an
  uncommitted `.env` via `python-dotenv`; `.env.example` holds only empty
  placeholders.
- Unit tests use small synthetic inputs; market-data tests load a short,
  reduced real slice through `data/data_loader.py`, cached across runs via
  pytest's cache directory, and are skipped when `DATALOADER_API_KEY` is
  unset.
- After rerunning an experiment, the files under `results/` are overwritten
  by the new run; the figures, numbers and verdicts written by hand in
  `results/RESULTS.md` must then be checked against the new files and updated
  to match.
