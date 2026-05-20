# Using the QuantAMM Simulator

**Repo:** https://github.com/balancer/quantammsim

`quantammsim` is a JAX-accelerated Python library for simulating and tuning AMM pools (Balancer, CowAMM, Gyroscope, QuantAMM, reCLAMM) against historic token-pair data.

---

## 1. Setup

Requires Python 3.10+.

> **Branch:** to simulate reCLAMM, use `origin/training-pipeline` (check it out after cloning with `git checkout training-pipeline`).

```bash
# Clone
git clone git@github.com:balancer/quantammsim.git
cd quantammsim
git checkout training-pipeline

# Create env (venv, Python 3.12 — any 3.10+ works)
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Install in editable mode (pulls JAX, Optuna, pandas, matplotlib, …)
pip install -e .
```

> If you prefer conda: `conda create -n qsim python=3.10 -y && conda activate qsim` instead of the venv lines.

Verify the install:

```bash
python -c "from quantammsim.runners.jax_runners import do_run_on_historic_data; print('ok')"
```

Subsequent shells need `source .venv/bin/activate` (or prefix commands with `.venv/bin/python`).

---

## 2. Load token data

Historic price data is downloaded into `quantammsim/data/` as `<TICKER>_USD.parquet` files. Pass the tickers you need as CLI args:

```bash
python scripts/download_data.py BTC ETH USDC AAVE
```

- Each ticker fetches minute-resolution data and writes both a parquet (full) and a CSV (daily) to `quantammsim/data/`.
- Re-running updates the existing files incrementally.
- The `tokens` field in any simulation fingerprint must match these ticker names exactly (e.g. `["AAVE", "ETH"]`, not `WETH`).

---

## 3. Run a simple reCLAMM simulation

`do_run_on_historic_data` is the single entrypoint for a one-shot backtest. You pass a **`run_fingerprint`** (config) and a **`params`** dict (pool parameters).

```python
import jax.numpy as jnp
from quantammsim.runners.jax_runners import do_run_on_historic_data

# Convert daily price-shift exponent → Solidity `daily_price_shift_base`
def to_daily_price_shift_base(exp):
    return 1.0 - exp / 124649.0

run_fingerprint = {
    "tokens": ["AAVE", "ETH"],
    "rule": "reclamm",
    "startDateString": "2024-06-01 00:00:00",
    "endDateString":   "2025-06-01 00:00:00",
    "initial_pool_value": 1_000_000.0,
    "do_arb": True,
    "fees": 0.0025,
    "gas_cost": 0.0,
    "arb_fees": 0.0,
    "chunk_period": 60,
    "weight_interpolation_period": 60,
}

params = {
    "price_ratio":             jnp.array(1.5),
    "centeredness_margin":     jnp.array(0.5),
    "daily_price_shift_base":  jnp.array(to_daily_price_shift_base(0.1)),
}

result = do_run_on_historic_data(run_fingerprint=run_fingerprint, params=params)

print("Initial:", float(result["value"][0]))
print("Final:  ", float(result["final_value"]))
```

`result` contains `reserves`, `prices`, `value` over time, and `final_value`.

For a ready-made multi-scenario example (incl. a 50/50 Balancer baseline for comparison) see [`scripts/demo_run_reclamm.py`](scripts/demo_run_reclamm.py):

```bash
python scripts/demo_run_reclamm.py
```

---

## 4. Tune the best reCLAMM parameters for a token pair

Use `train_on_historic_data` with `optimisation_settings.method = "optuna"`. The wrapper script [`experiments/tune_reclamm_params.py`](experiments/tune_reclamm_params.py) handles the standard search space (`price_ratio`, `centeredness_margin`, `shift_exponent`).

**Default — fee-revenue objective, 50 trials on AAVE/ETH:**

```bash
python experiments/tune_reclamm_params.py
```

**Custom token pair / window / objective:**

```bash
python experiments/tune_reclamm_params.py \
  --n-trials 200 \
  --fees 0.003 \
  --start-date "2024-01-01 00:00:00" \
  --end-date   "2025-01-01 00:00:00" \
  --end-test-date "2025-06-01 00:00:00" \
  --objective fee_revenue_over_value
```

To target a different pair, edit the `pool_tokens` default at the top of `tune_reclamm_params.py` (e.g. `["BTC", "ETH"]`) — the tokens must match files in `quantammsim/data/`.

Other useful flags:

| Flag | Purpose |
|---|---|
| `--objective daily_log_sharpe` | Optimise risk-adjusted return instead of fees |
| `--interpolation constant_arc_length` | Adds `arc_length_speed` to the search |
| `--noise-trader-ratio 0.5` | Add noise traders alongside arb |
| `--noise-model calibrated --noise-params-json ... --noise-pool-id ...` | Use a pre-calibrated 4/8-covariate noise model |

The script prints the best parameters and train/test metrics on completion. Visualise the Optuna study with [`scripts/plot_reclamm_optuna_result.py`](scripts/plot_reclamm_optuna_result.py).
