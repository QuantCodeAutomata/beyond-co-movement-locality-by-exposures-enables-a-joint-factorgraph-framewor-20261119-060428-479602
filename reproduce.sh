#!/usr/bin/env bash
# Runs the test suite and every experiment of the MINGLE replication,
# using the same commands listed in README.md.
set -uo pipefail

cd "$(dirname "$0")"

echo "== Unit tests (no network) =="
python -m pytest -m "not market_data" -q

echo "== Experiment 1: MINGLE ADMM framework =="
python -m exp.exp1_mingle_admm

echo "== Experiment 2: Representation quality analysis =="
python -m exp.exp2_representation_quality

echo "== Experiment 3: Portfolio performance backtest =="
python -m exp.exp3_portfolio_backtest

echo "== Experiment 4: Ablation under synthetic noise =="
python -m exp.exp4_ablation_noise

echo "Done. See results/RESULTS.md for the comparison against the paper."
