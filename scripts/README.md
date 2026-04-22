# quantammsim training → deployment pipeline

Three scripts, one flow:

```
train_strategy.py    →   evaluate_trials.py    →    readout.py
     ↓                          ↓                        ↓
 run_*.json            filled_analysis_*.csv    sc_deployment_fields.json
 (trained trials)      (pick a winner)          (deploy to chain)
```

1. **`train_strategy.py`** sweeps hyperparameters to produce trained trials.
2. **`evaluate_trials.py`** runs each trial on your chosen evaluation windows and emits a CSV so you can pick a winner.
3. **`readout.py`** takes one chosen trial and produces deployment-ready JSON (smart-contract-format params + current state variables, permuted into the on-chain token ordering).

All three scripts are CLI-driven. `--help` is reliable for any of them.

---

## Quick start

Assumes you've done stages 1–3 of the install (`conda activate qsim-noise-modelling`, data downloaded via `scripts/download_data.py ETH USDC`).

```bash
# 1. Train a sweep of momentum strategies on ETH/USDC.
python scripts/train_strategy.py \
    --optimiser adam \
    --rule momentum \
    --tokens ETH USDC

# 2. Compare all trials and pick a winner.
python scripts/evaluate_trials.py \
    --base-dir ./results \
    --tokens ETH USDC \
    --force-reload
# → open results/filled_analysis_*.csv, eyeball the metrics, note study_id + trial_number

# 3. Read out the winner for deployment on mainnet.
python scripts/readout.py \
    --base-dir ./results \
    --study-id run_<hash> \
    --trial-number 0 \
    --token-addresses 0xC02aAA39b223FE8D0A0e5C4F27eAD9083C756Cc2 \
                      0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48
# → results/readouts/<study_id>_trial0_runUntil<date>.json — hand to deployment tool
```

---

## `train_strategy.py` — sweep-train strategies

Produces `run_*.json` training artifacts by running a cartesian product over an optimiser's relevant hyperparameters. Which optimiser, which pre-canned strategy, which assets, and which date window are CLI-controlled; the sweep grid itself is declared at the top of the script (edit in place to tighten/widen axes).

### Supported optimisers

| `--optimiser` | method | what gets swept by default |
|---|---|---|
| `adam` | gradient descent (Adam / AdamW) | `optimiser_recipe` (adam vs adamw+wd), `base_lr`, `lr_schedule_type`, `return_val` |
| `optuna` | TPE search | `n_trials`, `overfitting_penalty`, `return_val` |
| `cma-es` | evolution strategy | `sigma0`, `n_generations`, `return_val` |
| `l-bfgs` | quasi-Newton | `maxiter`, `noise_scale`, `return_val` |

All four grids additionally cross over the same **cross-cutting axes** (`ste`, `maximum_change`, `turnover_penalty`, `price_noise_sigma`) — the knobs that target generalisation.

### Useful flags

| flag | default | what it does |
|---|---|---|
| `--optimiser` | `adam` | picks the sweep grid |
| `--rule` | `momentum` | pre-canned strategy name (see the quantamm pools: `momentum`, `anti_momentum`, `power_channel`, `mean_reversion_channel`) |
| `--tokens ETH USDC` | ETH USDC | pool assets, must match downloaded data |
| `--start` / `--end` / `--test-end` | 2023-06-01 / 2025-06-01 / 2026-01-01 | train window + held-out test window |
| `--n-parameter-sets` | 4 | parallel multi-start param sets per run |
| `--seeds 0 1 2` | `[0]` | repeat each combo per seed for robustness |
| `--max-runs N` | — | cap total runs (useful for smoke tests) |
| `--process-id 0 --process-total 4` | 0 / 1 | parallel-worker chunking (shard the sweep across processes) |
| `--dry-run` | — | print planned combos, run nothing |
| `--smoke` | — | force every inner budget (iterations / trials / generations / maxiter) to 1 — pipeline validation in seconds |
| `--force-init` | — | ignore cached `run_*.json` and retrain |

### Outputs

Written to `./results/`:

- `run_<sha256>.json` — per-call training cache. Hash = sha256 of the run_fingerprint, so re-running with the same config is a cache hit unless `--force-init`.
- `sweep_<optimiser>_<timestamp>.jsonl` — one summary row per sweep combo (combo axes, seed, duration, status, best metrics).
- `optuna_studies/` — only when `--optimiser optuna`. Contains `optimization.log` (shared across all optuna calls ever — check this when trials silently fail) and per-study sub-dirs with trial data.

### Common invocations

```bash
# Smoke-test the whole pipeline for every optimiser (fast, ~seconds per run).
for o in adam optuna cma-es l-bfgs; do
    python scripts/train_strategy.py --optimiser "$o" --smoke --max-runs 12
done

# Preview the planned adam sweep without running anything.
python scripts/train_strategy.py --optimiser adam --dry-run

# Shard one large adam sweep across 4 terminals.
# Terminal 1:  python scripts/train_strategy.py --optimiser adam --process-id 0 --process-total 4
# Terminal 2:  python scripts/train_strategy.py --optimiser adam --process-id 1 --process-total 4
# ...
```

### Gotcha: optuna + `expand_around`

There's an upstream bug in `create_trial_params` where `expand_around=True` produces invalid `low > high` bounds for `logit_lamb` (because the optuna path overrides `parameter_config["logit_lamb"]` with absolute bounds but the expand_around branch treats them as deltas). The script pins `expand_around=False` in `base_fingerprint` as a workaround. If optuna silently returns 0 completed trials, check `./optuna_studies/optimization.log`.

---

## `evaluate_trials.py` — pick a winner

Reads a directory of `run_*.json` files, runs each trial's trained params forward on one or more **evaluation windows**, and emits a comparison CSV you use to pick the best trial.

An "evaluation window" is a `(start_date, end_date, end_test_date)` triple. By default, each trial is evaluated on its own trained window. You can add more windows via `--run-period` (e.g. evaluate all trials on a longer OOS window to see which generalise best).

### Useful flags

| flag | default | what it does |
|---|---|---|
| `--base-dir ./results` | **required** | directory containing `run_*.json` |
| `--tokens ETH USDC` | ETH USDC | filter trials to this exact token set |
| `--load-method` | `best_train_min_test_objective` | how to pick which iteration of each trial to evaluate (see choices in `--help`) |
| `--force-reload` | — | ignore cached CSV and rescan all run files |
| `--no-plots` | — | skip weight/value PNGs (analysis CSVs still written) |
| `--run-period NAME START END TEST_END` | — | extra evaluation window, repeatable. NAME is the short label that appears in CSV column headers. |
| `--daily-to-hourly --scale-k` | — | convert a daily-chunk trial to hourly for re-analysis |

### Outputs

Written to `./results/` (and `./results/analysis_results/`):

- `sgd_analysis_result_*.csv` — full per-trial row for every (trial × period × pool-config) combination
- `simplified_analysis_*.csv` — a tighter per-trial-per-period view with the headline metrics
- `filled_analysis_*.csv` — **this is the one you read**. Sorted, column-ordered, with blank lines between token groups. Grouped by metric type (Returns over HODL train/test across all periods, Sharpe train/test across all periods, detailed train, detailed test, config tail).
- `analysis_results/analysis_unified_results_*.json` — all trial results serialised (useful for programmatic inspection)
- `plots/*.png` — weight-over-time plots for train + continuous-test per trial

### Common invocations

```bash
# Default: evaluate all trials on their own training window ("trained" period).
python scripts/evaluate_trials.py --base-dir ./results --tokens ETH USDC --force-reload

# Add a stricter OOS window to see which trials generalise.
python scripts/evaluate_trials.py --base-dir ./results --tokens ETH USDC --force-reload \
    --run-period oos_2025h2 "2025-07-01 00:00:00" "2025-12-31 00:00:00" "2026-06-30 00:00:00"

# Skip plots for speed when you just want the CSV.
python scripts/evaluate_trials.py --base-dir ./results --tokens ETH USDC --force-reload --no-plots
```

### Interpreting column names

Columns named `(name)` where `name` is the period name you provided. Defaults:

- `Returns over HODL train (trained)` — in-sample return vs holding a portfolio of `initial_weights`
- `Returns over HODL test (trained)` — out-of-sample (the period after `--end`)
- `Sharpe train (…) / Sharpe test (…)` — daily-return Sharpe ratio in each window
- Detailed per-period metrics: `Returns train`, `Annualized Ulcer Index [M]`, `Annualized Calmer Ratio [M]`

Every column tracks the trial identity via `study_id` + `trial_number` — note these down for the next step.

---

## `readout.py` — deploy-ready output

Takes **one trial**, optionally refreshes historic data up to now, runs the trained strategy forward, and emits a JSON with the SC-format parameter values + current readout state (EWMAs, intermediate values). The JSON is directly consumable by the v3 QuantAMM deployment task (field names match `input.ts`).

### Intended cadence

QuantAMM strategies refresh at midnight UTC. The intended use is: after ~06:00 UTC each day (once Binance has posted the previous day's 1-minute bars), run `readout.py` for each pool you're deploying or maintaining. `--run-until` defaults to the most recent midnight UTC.

### Required: declare your deployment intent

Exactly **one** of these is required:

- `--token-addresses 0xAAA 0xBBB …` — one address per `--tokens`, **same order**. Produces a deploy-ready JSON with `sc_deployment_fields` populated and permuted into address-ascending order (the protocol's convention).
- `--no-addresses` — explicit opt-out. Produces a sim-view JSON (no `sc_deployment_fields`, ticker order). Useful for debugging/inspection, never for deployment.

Omitting both is a hard error. This is deliberate: without addresses, per-token arrays are in ticker order — deploying those silently wires trained state to the wrong tokens. The file's shape now matches its readiness.

### Useful flags

| flag | default | what it does |
|---|---|---|
| `--base-dir ./results` | **required** | directory containing `run_*.json` |
| `--study-id run_<hash>` | pick first | which trial to read out |
| `--trial-number N` | first match | specific iteration within that study |
| `--load-method` | `best_train_min_test_objective` | picking criterion if `--trial-number` omitted |
| `--tokens ETH USDC` | from trial | usually inferred from the trial's fingerprint |
| `--token-addresses 0xAAA 0xBBB` | — | see above |
| `--no-addresses` | — | see above |
| `--run-until "YYYY-MM-DD HH:MM:SS"` | last midnight UTC | where to end the forward pass |
| `--no-refresh` | — | skip the auto-refresh check (don't call the download script even if cache is stale) |
| `--out PATH` | auto | override output file path |

### Auto data-refresh

By default, the script peeks at each token's cached parquet and **runs the download script only for tokens whose cache doesn't cover `--run-until`**. This is idempotent (appends new data, never destroys old). If all tokens are fresh, no network hit. Pass `--no-refresh` to skip the check entirely — useful for offline work or tight dev loops.

### Outputs

Written to `./results/readouts/` by default:

- `{study_id}_trial{N}_runUntil{YYYY-MM-DD}.json` — the one file that matters.

Two shapes depending on mode (check `meta.mode`):

#### Deployment mode (`--token-addresses` was given)

```json
{
  "meta": {
    "mode": "deployment",
    "token_order": "address-ascending",
    "tokens": ["USDC", "ETH"],
    "token_addresses": ["0xa0b86991...", "0xc02aaa39..."],
    "scale": "1e18 except updateInterval (raw seconds)",
    "encoding": "decimal-string (uint256/int256 safe) for 1e18-scaled values",
    …
  },
  "sc_deployment_fields": {
    "lambda":                     ["...", "..."],   // per-asset, 1e18-scaled
    "_initialWeights":            ["...", "..."],
    "_initialMovingAverages":     ["...", "..."],
    "_initialIntermediateValues": ["...", "..."],
    "absoluteWeightGuardRail":    "10000000000000000",  // scalar (minimum_weight × 1e18)
    "updateInterval":             86400                  // scalar, seconds, NOT 1e18-scaled
  },
  "contract_params": {
    // sim-native names for the rule-specific trained values. These go INTO
    // ruleParameters in the SC, but the list-of-lists shape varies per rule
    // and is left for the deployment tool to assemble.
    "memory_days": ["...", "..."],
    "k_per_day":   ["...", "..."],
    "k":           ["...", "..."],
    // plus exponents / pre_exp_scaling / amplitude / width depending on rule
  }
}
```

All values in `sc_deployment_fields` and `contract_params` are **1e18-fixed-point decimal strings** (uint256/int256-safe for JavaScript — large values won't be mangled by `JSON.parse` on the deployment side), **except `updateInterval`** which is a plain integer in seconds. Consumers should BigNumber-parse all the string values (`ethers.BigNumber.from("...")` or equivalent) before submitting.

#### Sim-view mode (`--no-addresses`)

```json
{
  "meta": {
    "mode": "sim-view",
    "token_order": "ticker",
    "tokens": ["ETH", "USDC"],
    "note": "Invoked with --no-addresses; sc_deployment_fields omitted. …"
  },
  "contract_params": { /* sim-native */ },
  "readouts": { /* ewma, running_a, gradients — sim-native names */ }
}
```

No `sc_deployment_fields`. A deployment tool reading this will `KeyError` immediately — the fail-fast design is intentional.

### Common invocations

```bash
# Daily deployment refresh on mainnet (run after 06:00 UTC).
python scripts/readout.py \
    --base-dir ./results \
    --study-id run_<hash> \
    --token-addresses 0xC02aAA39b223FE8D0A0e5C4F27eAD9083C756Cc2 \
                      0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48

# Same trial, different chain (base). Reorder is automatic — just pass base's addresses.
python scripts/readout.py \
    --base-dir ./results \
    --study-id run_<hash> \
    --token-addresses 0x4200000000000000000000000000000000000006 \
                      0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913

# Quick inspection of the trained values — no deployment intent.
python scripts/readout.py --base-dir ./results --study-id run_<hash> --no-addresses

# Reproduce yesterday's readout exactly (no auto-refresh, pinned end date).
python scripts/readout.py \
    --base-dir ./results \
    --study-id run_<hash> \
    --run-until "2026-04-21 00:00:00" \
    --no-refresh \
    --token-addresses 0xC02a... 0xA0b8...
```

---

## Typical daily / deployment workflow

### First time: train + select a strategy

```bash
# Train a sweep (choose optimiser; adam for fast iteration, optuna for thorough).
python scripts/train_strategy.py --optimiser adam --rule momentum --tokens ETH USDC

# Compare all trials across any evaluation windows of interest.
python scripts/evaluate_trials.py --base-dir ./results --tokens ETH USDC --force-reload \
    --run-period baseline "2023-06-01 00:00:00" "2025-06-01 00:00:00" "2026-01-01 00:00:00" \
    --run-period stress   "2022-01-01 00:00:00" "2023-06-01 00:00:00" "2024-01-01 00:00:00"

# Open results/filled_analysis_*.csv, pick the study_id + trial_number you like.
```

### Every day at 06:00 UTC: refresh deployment

```bash
# Auto-refresh Binance data, run the chosen trial forward to last midnight UTC,
# produce deployment JSON permuted for mainnet token addresses.
python scripts/readout.py \
    --base-dir ./results \
    --study-id run_<your_chosen_hash> \
    --trial-number <n> \
    --token-addresses <mainnet_token_0> <mainnet_token_1>
# → results/readouts/<study_id>_trial<n>_runUntil<today>.json
# → hand to deployment tool to push on-chain
```

### Multi-chain deployment

Same trained trial, once per chain. Each invocation produces a chain-specific JSON because the address-ascending permutation is chain-dependent:

```bash
for chain in mainnet base arbitrum; do
    python scripts/readout.py \
        --base-dir ./results \
        --study-id run_<hash> \
        --token-addresses $(addresses_for "$chain") \
        --out ./results/readouts/${chain}_<hash>.json
done
```

---

## Gotchas (quick reference)

| gotcha | where | fix |
|---|---|---|
| Optuna silently returns 0 completed trials | `train_strategy.py` with `expand_around=True` | pin `expand_around=False` (already handled in the script); check `optuna_studies/optimization.log` for the underlying `ValueError` |
| L-BFGS + STE-both-on → NaN metrics | `train_strategy.py` | these get flagged `status=nan_metrics` in the sweep summary and skipped by the BEST picker |
| `run_*.json` files are double-JSON-encoded (known upstream quirk) | all three scripts | `json.loads(json.load(f))`. Scripts handle this internally |
| `readout.py` errors "exactly one of --token-addresses or --no-addresses is required" | `readout.py` | pass one. See "Required: declare your deployment intent" above |
| Data-fetch `IndexError: index 0 is out of bounds` | `readout.py` with `--no-refresh` and stale cache | remove `--no-refresh` to enable auto-refresh, or pick an earlier `--run-until` |
| Multi-run sweep — I want to cancel | `train_strategy.py` | `Ctrl-C`. Already-completed runs are cached; resuming is free. |
