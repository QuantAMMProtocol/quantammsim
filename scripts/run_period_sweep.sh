#!/bin/bash
# Full sweep: all objectives × all periods × 400 trials
# Usage: bash scripts/run_period_sweep.sh
# Monitor: tail -5 /tmp/tune_*.log
# Results: results/sweep/

set -e
source ~/miniconda3/etc/profile.d/conda.sh && conda activate qsim_reclamm_public

TRIALS=400
MAX_PARALLEL=8
COMMON="python experiments/tune_reclamm_calibrated_noise.py --noise-model mm_observed --artifact-dir results/mm_noise --n-trials $TRIALS"

OBJECTIVES=(
    daily_log_sharpe
    daily_log_sharpe_excess
    fee_revenue_over_value
    returns_over_hodl
    calmar
    sterling
    weekly_rovar
)

# period_name  start_date  end_date(train)  end_test_date
PERIODS=(
    "bull_2023     2023-06-01  2024-06-01  2025-06-01"
    "default_2024  2024-06-01  2025-06-01  2026-03-01"
    "recent_2025   2025-01-01  2025-09-01  2026-03-01"
)

OUTDIR="results/sweep"
mkdir -p "$OUTDIR"

wait_for_slot() {
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do
        sleep 10
    done
}

N=0
for period_line in "${PERIODS[@]}"; do
    read -r period_name start_date end_date end_test_date <<< "$period_line"
    for obj in "${OBJECTIVES[@]}"; do
        tag="${obj}_${period_name}"
        logfile="/tmp/tune_${tag}.log"
        outfile="${OUTDIR}/${tag}.json"

        wait_for_slot

        echo "[$N] Launching: ${tag}"
        $COMMON --objective "$obj" \
            --start-date "${start_date} 00:00:00" \
            --end-date "${end_date} 00:00:00" \
            --end-test-date "${end_test_date} 00:00:00" \
            --output "$outfile" \
            > "$logfile" 2>&1 &

        N=$((N + 1))
    done
done

echo ""
echo "$N jobs queued (${#OBJECTIVES[@]} objectives × ${#PERIODS[@]} periods × $TRIALS trials)"
echo "Max parallel: $MAX_PARALLEL"
echo ""
echo "Monitor:  tail -5 /tmp/tune_*.log"
echo "Results:  ls $OUTDIR/"
echo "Summary:  grep -A3 'Best trial' /tmp/tune_*.log"
echo ""
echo "Waiting for all jobs to finish..."
wait
echo "Done."
