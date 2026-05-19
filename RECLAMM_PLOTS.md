# Reproducing the AAVE/ETH and COW/ETH plots

This branch (`noise-modelling-fixes-data`) carries the minimum data needed to
produce the final train/test panels and PR-sweep heatmaps for both pairs
without re-running the training sweep.

## What's on the branch

- `results/mm_noise/model.npz` + `meta.json` — the frozen MM noise model.
- `results/competitor_tvl/competitor_tvl.npz` — DeFi Llama competitor TVL.
- `results/full_sweep/<6 files>` — sweep summaries for the 6 winning configs.
- `results/run_<6 hashes>.json` — trial trajectories for the same 6 winners
  (read by `scripts/run_final_sims.py`).

What's **not** included and must be pulled locally:

- Binance minute parquets for `BTC`, `ETH`, `AAVE`, `COW`.

## Prerequisites

Conda env per the README:

```
conda activate qsim
```

## 1. Pull Binance price data

```
python scripts/download_data.py BTC ETH AAVE COW
```

Writes `quantammsim/data/<TOKEN>_USD.parquet` per token. `BTC` is required by
the MM noise model's market features; `AAVE`/`COW`/`ETH` are the pair tokens
the simulator reads prices from.

## 2. Final-sims: train/test forward passes and selection

```
python scripts/run_final_sims.py --all
```

Loads the 6 committed `run_<hash>.json` files, picks the single candidate per
`(pair, TVL)` tier, runs train and test forward passes at the picked params,
and writes:

- `results/final_sims/{aave,cow}_{train,test}.png` — share price, fee revenue,
  cumulative volume per TVL tier.
- `results/final_sims/{aave,cow}_{train,test}_weights.png` — effective weight
  trajectories.
- `results/final_sims/{aave,cow}_sim_results.pkl` — per-tier results cache.

Per-tier picks for this branch's data (printed to stdout as `Params:` lines):

| Tier      | PR      | margin | shift  |
|-----------|---------|--------|--------|
| AAVE 1m   | 1.068   | 0.0240 | 0.0225 |
| AAVE 5m   | 1.296   | 0.0101 | 0.0675 |
| AAVE 20m  | 1.348   | 0.0103 | 0.5411 |
| COW 500k  | 99.665  | 0.8235 | 0.1418 |
| COW 2m    | 54.549  | 0.8351 | 0.1422 |
| COW 20m   | 172.463 | 0.9265 | 0.0521 |

## 3. PR-sweep heatmaps

One `run_pr_sweep.py` invocation per (pair, TVL), with that tier's
`margin`, `shift`, `initial-pool-value`, and the selected PR as a marker.

### AAVE/ETH

Override the pair-specific flags (`run_pr_sweep.py` defaults are COW).

```
mkdir -p results/final_sims/for_fabio/aave/price_ratio_sweep

python scripts/run_pr_sweep.py --tokens AAVE ETH --pool-id 0x9d1fcf346ea1b0 \
    --gas-cost 1.0 --fees 0.0025 \
    --multi-period --period-months 3 --onchain-pr 2.02 \
    --output-dir results/final_sims/for_fabio/aave/price_ratio_sweep \
    --prs 1.01 1.1 1.2 1.4 1.6 1.8 2.0 2.5 3.0 4.0 5.0 7.5 10.0 \
    --margin 0.0240 --shift 0.0225 --initial-pool-value 1000000  --selected-pr 1.068

python scripts/run_pr_sweep.py --tokens AAVE ETH --pool-id 0x9d1fcf346ea1b0 \
    --gas-cost 1.0 --fees 0.0025 \
    --multi-period --period-months 3 --onchain-pr 2.02 \
    --output-dir results/final_sims/for_fabio/aave/price_ratio_sweep \
    --prs 1.01 1.1 1.2 1.4 1.6 1.8 2.0 2.5 3.0 4.0 5.0 7.5 10.0 \
    --margin 0.0101 --shift 0.0675 --initial-pool-value 5000000  --selected-pr 1.296

python scripts/run_pr_sweep.py --tokens AAVE ETH --pool-id 0x9d1fcf346ea1b0 \
    --gas-cost 1.0 --fees 0.0025 \
    --multi-period --period-months 3 --onchain-pr 2.02 \
    --output-dir results/final_sims/for_fabio/aave/price_ratio_sweep \
    --prs 1.01 1.1 1.2 1.4 1.6 1.8 2.0 2.5 3.0 4.0 5.0 7.5 10.0 \
    --margin 0.0103 --shift 0.5411 --initial-pool-value 20000000 --selected-pr 1.348
```

### COW/ETH

The COW selections are at PR ≈ 50–200, so the `--prs` list extends past 10.
The script's defaults already match the COW pair, so the pair-specific flags
can be omitted.

```
mkdir -p results/final_sims/for_fabio/cow/price_ratio_sweep

python scripts/run_pr_sweep.py --multi-period --period-months 3 --onchain-pr 2.02 \
    --output-dir results/final_sims/for_fabio/cow/price_ratio_sweep \
    --prs 1.01 1.1 1.2 1.4 1.6 1.8 2.0 2.5 3.0 5.0 10.0 30.0 50.0 80.0 100.0 120.0 150.0 200.0 \
    --margin 0.8235 --shift 0.1418 --initial-pool-value 500000   --selected-pr 99.665

python scripts/run_pr_sweep.py --multi-period --period-months 3 --onchain-pr 2.02 \
    --output-dir results/final_sims/for_fabio/cow/price_ratio_sweep \
    --prs 1.01 1.1 1.2 1.4 1.6 1.8 2.0 2.5 3.0 5.0 10.0 30.0 50.0 80.0 100.0 120.0 150.0 200.0 \
    --margin 0.8351 --shift 0.1422 --initial-pool-value 2000000  --selected-pr 54.549

python scripts/run_pr_sweep.py --multi-period --period-months 3 --onchain-pr 2.02 \
    --output-dir results/final_sims/for_fabio/cow/price_ratio_sweep \
    --prs 1.01 1.1 1.2 1.4 1.6 1.8 2.0 2.5 3.0 5.0 10.0 30.0 50.0 80.0 100.0 120.0 150.0 200.0 \
    --margin 0.9265 --shift 0.0521 --initial-pool-value 20000000 --selected-pr 172.463
```

### Outputs

Per-period single-config PNGs and one heatmap per invocation, named
`pr_heatmap_<TOKENS>_3mo_m<margin>_s<shift>.png`, under
`results/final_sims/for_fabio/{aave,cow}/price_ratio_sweep/`.

## Gotchas

- **zsh and shell variables**. `python ... $PRS_ARGS` with
  `PRS_ARGS="--prs 1.01 1.1 ..."` does not word-split in zsh by default;
  argparse receives the whole string as one token and emits
  `unrecognized arguments: --prs ...`. Either inline the list (as above)
  or use `${=PRS_ARGS}`.
- **COW early-period failures**. The COW pool's data starts around 2024-12.
  `run_pr_sweep.py --multi-period` tries each period from `2024-01-01`
  onwards; the early periods raise inside `run_single_period`, the script
  catches the exception and the final heatmap only contains the surviving
  rows. Expected.
- **Re-running with the same `(start, end)` reuses the cached noise array**
  in `results/mm_noise/_sim_arrays/<pool_id>_<start>_<end>_mm.npz` —
  rebuilding only the first time each period is touched.
