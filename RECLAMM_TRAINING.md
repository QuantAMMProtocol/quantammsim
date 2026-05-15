# reCLAMM training: end-to-end

Soup-to-nuts guide for going from no data on disk to a set of trained reCLAMM
params and the heatmap / weight / fee-revenue plots used in reports.

All commands assume the working directory is the repo root and the conda env
`qsim_reclamm_public` is active:

```
cd /Users/matthew/Projects/quantammsim-reclamm-public/quantammsim
source ~/miniconda3/etc/profile.d/conda.sh && conda activate qsim_reclamm_public
```

The pipeline has five stages. Outputs of each stage feed the next, so order
matters.

```
[1] Data pull           → quantammsim/data/*.parquet, local_data/...,
                          results/competitor_tvl/, results/pool_grids_v2/
[2] Grid build          → results/pool_grids_v2/*_daily.parquet
[3] MM noise model      → results/mm_noise/model.npz + meta.json
[4] Training sweep      → results/full_sweep/*.json + results/run_*.json
[5] Selection + plots   → results/final_sims/{aave,cow}*.png + .pkl
                          results/final_sims/for_fabio/<pair>/price_ratio_sweep/*
```

### What depends on what

The pipeline has two distinct dependency surfaces, separated by the MM model
artifact.

Training the MM noise model (stages 1+2+3) needs:

- Balancer V3 API (`api-v3.balancer.fi`) — panel snapshots
- Binance minute parquets — token price series
- DeFi Llama — competitor TVL
- Pool grids (`results/pool_grids_v2/`) — built locally from the panel

Training a reCLAMM (stages 4+5) only needs:

- Binance minute parquets (`quantammsim/data/<TOKEN>_USD.parquet`)
- The MM artifact (`results/mm_noise/model.npz` + `meta.json`)
- The competitor-TVL bundle (`results/competitor_tvl/competitor_tvl.npz`)

`quantammsim/runners/jax_runners.py` does not import from
`quantammsim.noise_calibration.*`, and
`quantammsim/calibration/noise_model_arrays.build_mm_simulator_arrays` reads
only Binance data plus the two artifacts (standardisation stats are stored
inside `model.npz` as `x_mean` / `x_std`). Once the MM model exists you can
train reCLAMMs on a different machine by carrying `results/mm_noise/`,
`results/competitor_tvl/`, and the Binance parquets for the pair.

---

## 1. Pull the data

Three data sources, three scripts.

### 1a. Token price history (Binance minute bars)

```
python scripts/download_data.py BTC ETH AAVE COW USDC USDT WBTC
```

Reads from `scripts/ticker_list.txt` if no tickers are passed. Output:
`quantammsim/data/<TOKEN>_USD.parquet` (minute resolution) and
`<TOKEN>_USD_daily.csv`. Used by all simulator runs and noise calibration.

### 1b. Balancer pool snapshots (volume + TVL)

```
python -m quantammsim.noise_calibration --fetch --chain ethereum
python -m quantammsim.noise_calibration --fetch --chain base
python -m quantammsim.noise_calibration --fetch --chain gnosis
```

Fetches the V3 API (`api-v3.balancer.fi`) for WEIGHTED and RECLAMM pools, then
pulls daily snapshots and assembles a panel. Outputs the panel parquet under
`local_data/noise_calibration/panel.parquet`.

#### How the pool list is determined

`--fetch` does not take a pool or pair argument. It calls
`enumerate_balancer_pools(min_tvl=args.min_tvl)`, which asks the Balancer V3
API for every WEIGHTED + RECLAMM pool currently above `--min-tvl` (default
$10k) on the specified chain. The threshold is checked against the pool's
*current* TVL at fetch time, so a pool that once held large TVL but is now
below the floor will be excluded — even if its earlier history would have
been useful.

The calibration set the MM model is trained on is the survivor set after
downstream filters: pools with both tokens matched to Binance data and with
enough clean daily snapshots. The survivors are cached in
`local_data/noise_calibration/_cache/stage1.pkl` (the `matched_clean` dict).
`scripts/fetch_competitor_tvl.py` and `scripts/run_mm_noise.py` both read
that file for their pool list and do not expose a per-pool selector.

To get a specific pool included:

1. Confirm its current TVL is above `--min-tvl`, or lower the threshold.
2. Make sure both tokens have Binance parquets in `quantammsim/data/`
   (`scripts/download_data.py <TOKEN>`).
3. Re-run `--fetch` → `fetch_competitor_tvl.py` → `run_mm_noise.py`. Any
   pool that passes the threshold and Binance-match check ends up in the
   trained MM artifact.

#### How the pool address propagates downstream

`scripts/tune_reclamm_calibrated_noise.py` takes `--pool-id`. It is used
only as a lookup key into the frozen MM artifact:
`_find_pool_index(pool_id, meta["pool_ids"])` returns the per-pool
`log_alpha`, `gamma`, `log_K`, and `log_cadence`. The address is not used
to fetch price data — prices come from Binance via `--tokens <A> <B>`. So
`--tokens AAVE ETH --pool-id 0xnotreal` will run reCLAMM training with
AAVE/ETH price feeds but with the median MM parameters and `K = $10M`
fallback documented in §6.

`scripts/run_full_sweep.sh` hard-codes the 6 (token_a, token_b, pool_id,
gas, fees, tvl_label, initial_tvl) tuples it sweeps, which is where the
pool address is pinned per-job.

### 1c. Competitor TVL (per-pair, per-chain) — DeFi Llama

```
python scripts/fetch_competitor_tvl.py
# optional: python scripts/fetch_competitor_tvl.py --cache-dir results/competitor_tvl
```

For each of the calibration pools, finds all other DEX pools trading the same
token pair and sums their daily TVL. Output:

- `results/competitor_tvl/competitor_tvl.npz` — `(n_dates, n_pools)` array of
  daily competitor TVL in USD, used as `K_i(t) = sum_{j ≠ i} TVL_j(t)` in the
  MM noise model.
- `results/competitor_tvl/<TOKA>_<TOKB>_<chain>_history.pkl` — per-pair cache
  for re-runs.

Optional token-mcaps cache (used by some plotting / classification code):

```
python scripts/fetch_token_mcaps.py
```

writes `local_data/noise_calibration/token_mcaps.json`.

---

## 2. Build the per-pool arb-volume grids

```
python scripts/build_pool_grids.py --workers 6 --train-days 90
```

Sweeps `(cadence, gas)` per real Balancer pool to produce PCHIP grids of daily
arb volume. Output: `results/pool_grids_v2/<pool_id_prefix>_daily.parquet` per
pool, plus a summary CSV. The grids are what the joint calibration model
interpolates over to attribute total observed volume to arb vs noise.

This step is slow; it forward-simulates each pool across `cadence × gas` for
the chosen training window. Use a non-default `--workers` to parallelise.

---

## 3. Train the Michaelis-Menten noise model

Fits the per-pool MM model that the simulator uses to predict noise volume at
run-time:

```
log(V_noise) = log_alpha_i + x_market @ gamma + log(TVL) − log(K_i + TVL)
V_total      = V_arb(cadence_i) + exp(log_V_noise)
Loss         = Huber(log(V_total) − log(V_obs))
```

```
python scripts/run_mm_noise.py \
    --per-pool-gamma --epochs 5000 --lr 1e-4 \
    --huber-delta 0.5 --observed-K --no-split
```

The hparams the existing artifact was fit with are stored in
`results/mm_noise/meta.json` under `hparams`; reproduce by matching them on
the command line.

Loads the panel from `local_data/noise_calibration/panel.parquet`, matches
each row to the right per-day grid in `results/pool_grids_v2/`, and fits the
MM model jointly across all pools. Output:

- `results/mm_noise/model.npz` — fitted parameters (per-pool `log_alpha`,
  `log_K`, `log_cadence`, shared `gamma`).
- `results/mm_noise/meta.json` — `pool_ids`, `n_market_feat`,
  `per_pool_gamma`, `hparams`.
- `results/mm_noise/trials/trial_NNNN/` — checkpoint per trial during a
  hyperparameter sweep.

`meta.json` has the canonical schema:

```json
{
  "model": "michaelis_menten",
  "pool_ids": ["0x9d1fcf346ea1b0", "0xd321300ef77067", ...],
  "n_market_feat": 18,
  "per_pool_gamma": true,
  "hparams": {"epochs": 5000, "lr": 1e-4, "l2_alpha": 1e-3,
              "huber_delta": 0.5, "init_log_K": 17.0}
}
```

Diagnostic plot:

```
python scripts/plot_mm_noise_fit.py
```

writes per-pool fit overlays to `results/mm_noise/plots/`.

---

## 4. Training sweep

`scripts/run_full_sweep.sh` is the orchestrator; it shells out to
`scripts/tune_reclamm_calibrated_noise.py` per job. Each (pair, TVL tier,
objective, penalty) combination is one job. With the default config there are
72 jobs per method:

- 2 pairs (AAVE/ETH, COW/ETH) × 3 TVL tiers = 6 configs
- 4 objectives: `returns_over_hodl`, `fee_revenue_over_value`, `calmar`,
  `daily_log_sharpe_excess`
- 3 penalty variants: no penalty, `--overfitting-penalty 1.0`, `5.0`

```
# Optuna (default, 300 trials per job)
MAX_WORKERS=6 bash scripts/run_full_sweep.sh

# CMA-ES (500 generations per job)
MAX_WORKERS=6 bash scripts/run_full_sweep.sh --method cma_es

# Single pair only
bash scripts/run_full_sweep.sh --pair aave
bash scripts/run_full_sweep.sh --pair cow
```

Each job writes two files:

- `results/full_sweep/<tag>.json` — per-job summary with chosen best params
- `results/run_<sha256>.json` — full trial trajectory used by the
  selection step. Hash is computed from the run_fingerprint, so changes to
  any fingerprint field (method, penalty, TVL, objective, token pair, …)
  produce a fresh file.

Per-job stdout / stderr go to `/tmp/tune_full_<tag>.log` — useful for spot
checks while a sweep runs.

`MAX_WORKERS` defaults to 8 for optuna and 4 for CMA-ES (CMA-ES uses more RAM).
Override either with the env var.

### What each method optimises

| Method | Optimiser | Per-job time (300 trials / 500 gens) | Notes |
|---|---|---|---|
| optuna | TPE sampler, median pruner | ~20–40 min | Single-objective; multi-objective optional via `--multi-objective`. |
| cma_es | CMA-ES with box constraints from `parameter_config` | ~30–60 min | One restart per `n_parameter_sets`. |
| bfgs | `jax.scipy.optimize.minimize(method="BFGS")` | depends on `bfgs_maxiter` | Unconstrained; needs sp_/logit_ reparametrisation to stay in bounds. |

---

## 5. Selection, final sims, and plots

### 5a. Pick the best params per (pair, TVL tier) and run train+test forward sims

```
python scripts/run_final_sims.py --all
# or
python scripts/run_final_sims.py --pair aave
python scripts/run_final_sims.py --pair cow
```

`run_final_sims.py` loads every `results/run_*.json`, filters to its
hardcoded train period (see `TRAIN_START` / `TRAIN_END` at the top of the
script), ranks candidates by val-RoH pulled from
`continuous_test_metrics["returns_over_hodl"]`, and picks the best for each
`(token, TVL)` combination. Then runs two independent forward passes (one
each for the train and test windows) at the picked params, and writes:

- `results/final_sims/<pair>_sim_results.pkl` — the full per-tier results (large; ~200 MB per pair).
- `results/final_sims/<pair>_{train,test}.png` — share-price / fee revenue / cumulative volume per TVL tier.
- `results/final_sims/<pair>_{train,test}_weights.png` — effective weight trajectories.

The printed `Best:` / `Params:` lines are the per-tier selections — that's
where `(PR, margin, shift)` for the next step come from.

### 5b. PR-sweep heatmaps (PR × period grid)

Once a tier's `(margin, shift, initial_pool_value)` is chosen, sweep PR across
a multi-period window:

```
python scripts/run_pr_sweep.py --tokens COW ETH \
    --pool-id 0xd321300ef77067 --gas-cost 3.0 --fees 0.003 \
    --noise-model mm_observed \
    --multi-period --period-months 3 \
    --onchain-pr 2.02 \
    --prs 1.01 1.1 1.2 1.4 1.6 1.8 2.0 2.5 3.0 5.0 10.0 \
          30.0 50.0 80.0 100.0 120.0 150.0 200.0 \
    --output-dir results/final_sims/for_fabio/cow/price_ratio_sweep \
    --margin 0.8235 --shift 0.1418 --initial-pool-value 500000 \
    --selected-pr 99.665
```

The `--prs` list should bracket the selected PR for the tier. The
`run_final_sims.py` output tells you the selected PR; extend the upper bound
of `--prs` past it so the heatmap shows the surrounding landscape. Output:
`pr_heatmap_<TOKENS>_<period>_m<margin>_s<shift>.png` in the output dir.

The script tries each period from `2024-01-01` onwards in 1-month steps;
periods that start before the pool's data is available raise inside
`run_single_period`, the script catches the exception, and the final heatmap
only contains the surviving rows.

### 5c. Other diagnostic plots

| Script | What it produces |
|---|---|
| `scripts/plot_reclamm_optuna_result.py` | per-config train/test panels from a single sweep result |
| `scripts/plot_mm_noise_fit.py` | per-pool MM-model fit overlay |
| `scripts/plot_calibrated_vs_real.py` | total volume: modelled vs observed |
| `scripts/compare_modelled_vs_real.py` | noise volume of model pool A vs real volume of pool B |
| `scripts/select_best_params.py` | ranked summary of sweep results, optional `--export` of best params |

---

## 6. Extending to a new pair

The frozen artifacts in `results/mm_noise/` and `results/competitor_tvl/`
cover the set of pools that survived calibration filtering at training time
(see §1b). The list lives in `results/mm_noise/meta.json` under `pool_ids`,
and the same set is the column index of `results/competitor_tvl/competitor_tvl.npz`.

If the pair you want to train on is in that set, training picks up the
per-pool `log_alpha` / `gamma` / `log_K` automatically.

If the pair is not in the set, `build_mm_simulator_arrays` falls back:

- MM noise level → cross-pool median `log_alpha` and `gamma`. Logs
  `MM model: pool not found, using median alpha=...`.
- Competitor TVL → constant `K = $10M`. Logs
  `WARNING: pool not in competitor TVL data, using K=$10M`.

The simulator still runs, with cross-pool averages replacing the pool-specific
saturation curve and intercept. For pairs close to the calibration set this is
a reasonable approximation; for very small / illiquid / very large pairs the
median is likely well off and the constant `K` flattens the TVL→noise response.

To get proper per-pool parameters for a new pair:

```
# 1. Make sure the pool is in the panel
python -m quantammsim.noise_calibration --fetch --chain ethereum

# 2. Refresh the competitor-TVL bundle (DeFi Llama)
python scripts/fetch_competitor_tvl.py

# 3. Re-train the MM model so the new pool gets its own entry
python scripts/run_mm_noise.py \
    --per-pool-gamma --epochs 5000 --lr 1e-4 \
    --huber-delta 0.5 --observed-K --no-split
```

Step 1 requires the pool to have TVL above `--min-tvl` on the Balancer API.
Step 2 caches a fresh `<TOKA>_<TOKB>_<chain>_history.pkl` and rebuilds
`competitor_tvl.npz`. Step 3 produces fresh `model.npz` + `meta.json` whose
`pool_ids` list now includes the new pool. Existing reCLAMM training results
for other pairs are unaffected (they're keyed off the run_fingerprint hash;
re-running step 3 doesn't change those hashes).

After this, any `run_full_sweep.sh` / `run_final_sims.py` / `run_pr_sweep.py`
call against the new pair reads its per-pool MM parameters and per-day
competitor TVL series instead of falling back to medians.

---

## Caveats / things to watch

- **The `results/run_*.json` cache is fingerprint-keyed**. Changing any field
  in `run_fingerprint` (method, penalty, TVL, ste_temperature, …) produces a
  fresh hash. Identical fingerprints will reload from cache instead of
  re-training. Delete the matching `run_<hash>.json` to force a fresh run.

- **CMA-ES penalty** factors into the run_fingerprint. If you have older
  `run_*.json` files from a version where the penalty was not in the
  fingerprint, they share a hash across penalty variants and should be
  deleted before a fresh sweep, otherwise the candidate pool mixes
  configurations.

- **Stale data ranges**. `run_pr_sweep.py --multi-period` starts at
  `2024-01-01`; pools that didn't exist that early will fail the early
  periods. The script catches the exception and produces a heatmap of only
  the surviving rows.

- **`MAX_WORKERS=N`** in `run_full_sweep.sh` controls shell parallelism only —
  each job is a separate Python process. Memory is the binding constraint;
  4–6 workers is typical for a laptop.

- **PR axis on heatmaps** is what you pass via `--prs`. The default upper
  bound is 10; pass higher values when the selected PR for the pair exceeds
  that.

---

## Quick re-run shortcut

If the data + MM model + grids are already on disk and you just want fresh
trained params and plots:

```
# (1) sweep
MAX_WORKERS=6 bash scripts/run_full_sweep.sh
MAX_WORKERS=6 bash scripts/run_full_sweep.sh --method cma_es

# (2) select + final sims
python scripts/run_final_sims.py --all

# (3) PR heatmaps per pair × tier — adjust margin/shift/TVL/selected-PR
#     from the run_final_sims.py output
python scripts/run_pr_sweep.py --tokens AAVE ETH ...
python scripts/run_pr_sweep.py --tokens COW ETH ...
```

The whole chain from a clean cache is ~half a day on 6 workers for AAVE+COW
at the canonical 6-tier sweep.
