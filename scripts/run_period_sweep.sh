#!/bin/bash
# Full sweep: objectives × periods × robust temps × overfitting penalties × 400 trials
# Usage: bash scripts/run_period_sweep.sh
# Monitor: tail -5 /tmp/tune_*.log
# Results: results/sweep/

set -e
source ~/miniconda3/etc/profile.d/conda.sh && conda activate qsim_reclamm_public

TRIALS=400
MAX_PARALLEL=8
COMMON="python scripts/tune_reclamm_calibrated_noise.py --noise-model mm_observed --artifact-dir results/mm_noise --n-trials $TRIALS"

OBJECTIVES=(
    daily_log_sharpe
    daily_log_sharpe_excess
    fee_revenue_over_value
    returns_over_hodl
    calmar
    sterling
    weekly_rovar
)

# period_name  start_date  end_date(train+val)  end_test_date  [val_fraction]
PERIODS=(
    "bull_2023     2023-06-01  2024-06-01  2025-06-01"
    "default_2024  2024-06-01  2025-06-01  2026-03-01"
    "recent_2025   2025-01-01  2025-09-01  2026-03-01"
    "long_2021     2021-06-01  2025-01-01  2026-03-01  0.4"
)

# Robust temperatures: "" = standard mean, otherwise --robust-temperature X
# Lower = more pessimistic (0.1 = very robust, 0.01 = near worst-case).
ROBUST_TEMPS=("" "0.5" "1.0" "0.1" "0.01")

ROBUST_TEMPS=("")


# Overfitting penalty: "" = default (0.2), otherwise --overfitting-penalty X
# Higher = stronger train-vs-val regularization. penalty=1 means objective
# becomes val_score when train > val; penalty>1 rewards val > train.
OVERFITTING_PENALTIES=("" "1.0" "5.0")

OVERFITTING_PENALTIES=("" "5.0")


OUTDIR="results/sweep"
mkdir -p "$OUTDIR"

wait_for_slot() {
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do
        sleep 10
    done
}

N=0
for period_line in "${PERIODS[@]}"; do
    read -r period_name start_date end_date end_test_date val_fraction <<< "$period_line"
    val_flag=""
    if [ -n "$val_fraction" ]; then
        val_flag="--val-fraction $val_fraction"
    fi
    for obj in "${OBJECTIVES[@]}"; do
        for temp in "${ROBUST_TEMPS[@]}"; do
            for penalty in "${OVERFITTING_PENALTIES[@]}"; do
                # Build tag and flags
                tag="${obj}"
                robust_flag=""
                penalty_flag=""
                if [ -n "$temp" ]; then
                    tag="${tag}_robust${temp}"
                    robust_flag="--robust-temperature $temp"
                fi
                if [ -n "$penalty" ]; then
                    tag="${tag}_penalty${penalty}"
                    penalty_flag="--overfitting-penalty $penalty"
                fi
                tag="${tag}_${period_name}"

                logfile="/tmp/tune_${tag}.log"
                outfile="${OUTDIR}/${tag}.json"

                # Skip if result already exists
                if [ -f "$outfile" ]; then
                    echo "[$N] Skipping (exists): ${tag}"
                    N=$((N + 1))
                    continue
                fi

                wait_for_slot

                echo "[$N] Launching: ${tag}"
                $COMMON --objective "$obj" \
                    --start-date "${start_date} 00:00:00" \
                    --end-date "${end_date} 00:00:00" \
                    --end-test-date "${end_test_date} 00:00:00" \
                    $robust_flag $penalty_flag $val_flag \
                    --output "$outfile" \
                    > "$logfile" 2>&1 &

                N=$((N + 1))
            done
        done
    done
done

echo ""
echo "$N jobs total (existing results skipped)"
echo "Max parallel: $MAX_PARALLEL"
echo ""
echo "Monitor:  tail -5 /tmp/tune_*.log"
echo "Results:  ls $OUTDIR/"
echo "Summary:  grep -A3 'Best trial' /tmp/tune_*.log"
echo ""
echo "Waiting for all jobs to finish..."
wait
echo "Done."
