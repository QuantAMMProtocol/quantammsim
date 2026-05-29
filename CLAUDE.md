# quantammsim — running simulations

JAX-accelerated simulator for backtesting and tuning AMM pools (Balancer, CowAMM, Gyroscope, QuantAMM, reCLAMM) against historic minute-resolution token data. This file is for agents driving simulations; read it before running anything.

## Environment

- Use the existing venv: `.venv/` (Python 3.12). Prefix commands with `.venv/bin/python`, or `source .venv/bin/activate` first. Do not assume conda.
- Package is installed editable (`pip install -e .`). If imports fail, re-run that from the repo root.
- reCLAMM lives on the `training-pipeline` branch.
- Quick sanity check: `.venv/bin/python -c "from quantammsim.runners.jax_runners import do_run_on_historic_data; print('ok')"`

## Data

- Historic data is parquet at `quantammsim/data/<TICKER>_USD.parquet` (minute resolution). Available now: `AAVE`, `ETH`, `USDC`. Add more with `.venv/bin/python scripts/download_data.py <TICKERS...>`.
- `tokens` in any run config must match these filenames exactly (`ETH`, not `WETH`; `BTC`, not `WBTC`).
- The downloader already forces `mpire` `start_method="spawn"` (JAX is multithreaded; `fork` deadlocks on macOS) and normalizes Binance's 2025+ microsecond timestamps. Don't strip those patches in `scripts/download_data.py` / `historic_data_utils.py`.
- First download per ticker is heavy (~1 min, ETH parquet ≈ 180 MB).

## Running one backtest

Entrypoint: `do_run_on_historic_data(run_fingerprint, params)` in `quantammsim.runners.jax_runners`.

```python
import numpy as np, jax.numpy as jnp
from quantammsim.runners.jax_runners import do_run_on_historic_data

def to_daily_price_shift_base(exp):  # daily price-shift % -> Solidity base
    return 1.0 - exp / 124649.0

run_fingerprint = {
    "tokens": ["AAVE", "ETH"], "rule": "reclamm",
    "startDateString": "2025-05-20 00:00:00", "endDateString": "2026-05-20 00:00:00",
    "initial_pool_value": 10_000.0, "do_arb": True,
    "fees": 0.003, "gas_cost": 0.0, "arb_fees": 0.0,
    "chunk_period": 60, "weight_interpolation_period": 60,
}
params = {  # reCLAMM knobs
    "price_ratio": jnp.array(4.0),
    "centeredness_margin": jnp.array(0.1),                      # 10%
    "daily_price_shift_base": jnp.array(to_daily_price_shift_base(0.02)),  # 2%
}
r = do_run_on_historic_data(run_fingerprint=run_fingerprint, params=params)
```

Parameter units: percentages are fractions — `fees=0.003` is 0.3%, `centeredness_margin=0.1` is 10%. `price_ratio` is a raw multiplier. Daily price-shift % must be converted via `to_daily_price_shift_base`.

`result` keys: `value`, `prices`, `reserves`, `weights`, `fee_revenue` (all over time), plus `final_value`, `final_reserves`. `result["value"]` may need `np.asarray(...).reshape(-1)`. `result["prices"]` is `(n_minutes, n_tokens)` in USD, column-aligned with `tokens`.

## Comparing against HODL (do this — pools are judged relative to holding)

A pool's USD value alone is meaningless without a hold baseline. Compute from the same run:

```python
val = np.asarray(r["value"]).reshape(-1)
prices = np.asarray(r["prices"]); res0 = np.asarray(r["reserves"])[0]
deposit_hodl = (res0 * prices[-1]).sum()                       # hold the deposited basket
init_usd = (res0 * prices[0]).sum()
uniform_hodl = ((init_usd / 2.0) / prices[0] * prices[-1]).sum()  # equal-$ 50/50 hold
# pool minus HODL = fees earned minus impermanent loss
```

## Tuning (finding good params)

Entrypoint: `train_on_historic_data(fp, return_training_metadata=True)` → `(best_params, metadata)`. Set `fp["optimisation_settings"]["method"]="optuna"`. The objective is `fp["return_val"]`; all objectives are normalized so **higher = better** (the optimizer maximizes). HODL-relative objectives: `returns_over_hodl`, `annualised_returns_over_hodl`, `returns_over_uniform_hodl`, `daily_log_sharpe_excess`. Full list is in `quantammsim/core_simulator/forward_pass.py` (`_calculate_return_value`). The CLI wrapper is `experiments/tune_reclamm_params.py` (defaults to AAVE/ETH, search over `price_ratio`/`centeredness_margin`/`shift_exponent`).

Always split train vs out-of-sample: `startDateString` (train start), `endDateString` (train end / test start), `endTestDateString` (test end). `metadata["best_continuous_test_metrics"][0]` holds OOS metrics (incl. both HODL variants) for any objective — use it as a common yardstick across objectives.

## Critical caveats (learned the hard way)

- **Continuous metrics ≠ fresh-deposit reality.** Every in-framework metric (train, validation, and the OOS `test_objective`) comes from a *continuous* simulation where the pool enters each window carrying reserves evolved from the prior period. A real LP deposits *fresh* (rebalanced to the pool's initial split). These diverge a lot. **For any LP-facing claim, re-run the chosen config fresh with `do_run_on_historic_data` on the target window** and compare to HODL there. Do not quote the tuner's OOS number as the LP outcome.
- **Optuna selects on the (penalized) train objective**, which overfits: the train-best config routinely loses out-of-sample, and configs that look best OOS in the continuous run can be the worst on a fresh deposit. Rank candidates by fresh-deposit OOS before recommending.
- reCLAMM is a price-following AMM: it tends to *track* the held basket and earns its edge from fees in **ranging** markets, not one-directional trends (where IL dominates). "No config beats HODL for this pair/period" is a valid, common result — report it honestly rather than tuning until something looks good.
- Optuna runs in-memory (no sqlite file); per-trial records persist to `results/run_<hash>.json` (double-JSON-encoded: `json.loads` twice; element 0 is the fingerprint, elements 1..N are trials). Live progress is logged to `optuna_studies/optimization.log`.

## Conventions

- Generated artifacts land in `results/` and `optuna_studies/` (untracked); clean or ignore as needed.
- Long runs: launch in the background and poll the log, don't block.
- `train_on_historic_data` sets JAX x64 mode internally; for standalone numeric work prefer float64 to match.
