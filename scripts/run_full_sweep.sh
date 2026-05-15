#!/bin/bash
# Full reClAMM parameter sweep: AAVE/ETH + COW/ETH
# Train: 2025-01-01 → 2025-10-05 (pre flash crash)
# Test:  2025-10-25 → 2026-05-01 (post flash crash, separate run)
#
# Usage: bash scripts/run_full_sweep.sh [--method optuna|cma_es]
# Monitor: tail -5 /tmp/tune_full_*.log
# Results: results/full_sweep/

source ~/miniconda3/etc/profile.d/conda.sh && conda activate qsim_reclamm_public

# Parse arguments
METHOD="optuna"
PAIR_FILTER=""  # empty = all pairs
while [ $# -gt 0 ]; do
    case $1 in
        --method=*) METHOD="${1#*=}" ;;
        --method)   shift; METHOD="$1" ;;
        --pair=*)   PAIR_FILTER="${1#*=}" ;;
        --pair)     shift; PAIR_FILTER="$1" ;;
    esac
    shift
done

TRIALS=300
MAX_PARALLEL="${MAX_WORKERS:-8}"
if [ "$METHOD" = "cma_es" ] && [ "$MAX_PARALLEL" = "8" ]; then
    MAX_PARALLEL=4
fi

OBJECTIVES=(
    returns_over_hodl
    fee_revenue_over_value
    calmar
    daily_log_sharpe_excess
)

OVERFITTING_PENALTIES=("" "1.0" "5.0")

# Train period: pre flash crash
TRAIN_START="2025-01-01 00:00:00"
TRAIN_END="2025-10-05 00:00:00"

# token_a  token_b  pool_id            gas_cost  fees    tvl_label  initial_tvl
CONFIGS=(
    "AAVE  ETH  0x9d1fcf346ea1b0  1.0   0.0025  aave_1m    1000000"
    "AAVE  ETH  0x9d1fcf346ea1b0  1.0   0.0025  aave_5m    5000000"
    "AAVE  ETH  0x9d1fcf346ea1b0  1.0   0.0025  aave_20m   20000000"
    "COW   ETH  0xd321300ef77067  3.0   0.003   cow_500k   500000"
    "COW   ETH  0xd321300ef77067  3.0   0.003   cow_2m     2000000"
    "COW   ETH  0xd321300ef77067  3.0   0.003   cow_20m    20000000"
)

OUTDIR="results/full_sweep"
mkdir -p "$OUTDIR"

# Build COMMON command based on method
if [ "$METHOD" = "cma_es" ]; then
    CMA_GENS="${CMA_GENERATIONS:-500}"
    COMMON="python scripts/tune_reclamm_calibrated_noise.py --noise-model mm_observed --artifact-dir results/mm_noise --method cma_es --cma-generations $CMA_GENS"
    METHOD_TAG="_cmaes"
    echo "=== CMA-ES mode ($CMA_GENS generations) ==="
else
    PR_MAX_FLAG=""
    if [ -n "${PR_MAX:-}" ]; then
        PR_MAX_FLAG="--pr-max $PR_MAX"
        METHOD_TAG="_prmax${PR_MAX}"
    else
        METHOD_TAG=""
    fi
    COMMON="python scripts/tune_reclamm_calibrated_noise.py --noise-model mm_observed --artifact-dir results/mm_noise --n-trials $TRIALS $PR_MAX_FLAG"
    echo "=== Optuna mode ($TRIALS trials${PR_MAX:+, PR max=$PR_MAX}) ==="
fi

wait_for_slot() {
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do
        sleep 10
    done
}

N=0
SKIPPED=0
for config_line in "${CONFIGS[@]}"; do
    read -r tok_a tok_b pool_id gas_cost fees tvl_label initial_tvl <<< "$config_line"

    # Filter by pair if specified (e.g. --pair cow matches cow_500k, cow_2m, cow_20m)
    if [ -n "$PAIR_FILTER" ] && [[ "$tvl_label" != ${PAIR_FILTER}* ]]; then
        continue
    fi

    for obj in "${OBJECTIVES[@]}"; do
        for penalty in "${OVERFITTING_PENALTIES[@]}"; do
            tag="${obj}"
            penalty_flag=""
            if [ -n "$penalty" ]; then
                tag="${tag}_penalty${penalty}"
                penalty_flag="--overfitting-penalty $penalty"
            fi
            tag="${tag}${METHOD_TAG}_${tvl_label}"

            logfile="/tmp/tune_full_${tag}.log"
            outfile="${OUTDIR}/${tag}.json"

            # Skip if result already exists
            if [ -f "$outfile" ]; then
                echo "[$N] Skipping (exists): ${tag}"
                N=$((N + 1))
                SKIPPED=$((SKIPPED + 1))
                continue
            fi

            wait_for_slot

            echo "[$N] Launching: ${tag}"
            $COMMON --tokens "$tok_a" "$tok_b" \
                --pool-id "$pool_id" \
                --gas-cost "$gas_cost" \
                --fees "$fees" \
                --initial-pool-value "$initial_tvl" \
                --objective "$obj" \
                --start-date "$TRAIN_START" \
                --end-date "$TRAIN_END" \
                $penalty_flag \
                --output "$outfile" \
                > "$logfile" 2>&1 &

            N=$((N + 1))
        done
    done
done

echo ""
echo "$N jobs total ($SKIPPED skipped, $((N - SKIPPED)) launched)"
echo "Max parallel: $MAX_PARALLEL"
echo ""
echo "Monitor:  tail -5 /tmp/tune_full_*.log"
echo "Results:  ls $OUTDIR/"
echo "Summary:  grep -A3 'Best trial' /tmp/tune_full_*.log"
echo ""
echo "Waiting for all jobs to finish..."
wait
echo "Done."
