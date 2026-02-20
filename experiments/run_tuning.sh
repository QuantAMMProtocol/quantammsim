#!/usr/bin/env bash
set -euo pipefail

# Run CMA-ES and BFGS hyperparameter tuning sequentially.
#
# Usage:
#   ./experiments/run_tuning.sh                                    # defaults
#   ./experiments/run_tuning.sh --n-trials 100                     # override trials
#   ./experiments/run_tuning.sh --objective mean_oos_daily_log_sharpe  # override objective
#
# All flags are passed through to both scripts.

N_TRIALS=400
OBJECTIVE="mean_oos_returns_over_hodl"
N_WFA=4
MEM_FRAC=0.95
EXTRA_ARGS=()

# Parse known args, collect the rest
while [[ $# -gt 0 ]]; do
    case "$1" in
        --n-trials|-n)  N_TRIALS="$2"; shift 2 ;;
        --objective|-o) OBJECTIVE="$2"; shift 2 ;;
        --n-wfa-cycles|-c) N_WFA="$2"; shift 2 ;;
        --mem-frac)     MEM_FRAC="$2"; shift 2 ;;
        *)              EXTRA_ARGS+=("$1"); shift ;;
    esac
done

export XLA_PYTHON_CLIENT_MEM_FRACTION="$MEM_FRAC"

echo "================================================"
echo "  Hyperparameter Tuning"
echo "  Trials:    ${N_TRIALS} per optimizer"
echo "  Objective: ${OBJECTIVE}"
echo "  WFA:       ${N_WFA} cycles, 2019-01-01 → 2025-01-01"
echo "  Holdout:   2025-01-01 → 2026-01-01"
echo "  GPU mem:   ${MEM_FRAC}"
echo "================================================"

echo ""
echo "=== CMA-ES ==="
python tune_training_hyperparams_innercmaes.py \
    --n-trials "$N_TRIALS" \
    --n-wfa-cycles "$N_WFA" \
    --objective "$OBJECTIVE" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"

echo ""
echo "=== BFGS ==="
python tune_training_hyperparams_innerbfgs.py \
    --n-trials "$N_TRIALS" \
    --n-wfa-cycles "$N_WFA" \
    --objective "$OBJECTIVE" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"

echo ""
echo "Done. Results in experiments/hyperparam_studies/"
