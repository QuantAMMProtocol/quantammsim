#!/usr/bin/env python3
"""Run a chosen trained strategy forward to a given date and print:
  - smart-contract-form parameters (via ``pool.to_contract_params``)
  - final readout state variables (EWMAs, running_a, gradients — whatever the
    pool's ``calculate_readouts`` returns)
  - final weights, prices, and pool value

Pipeline: optionally refresh historic data → locate a trial in a base_dir of
run_*.json files → load its trained params via ``retrieve_best`` → run a
forward pass from the trial's training start to ``--run-until`` → print.

QuantAMM strategies refresh at midnight UTC, so the intended cadence is to
run this after ~06:00 UTC each day (when the previous day's Binance data is
available on binance.vision). ``--run-until`` defaults to the most recent
midnight UTC.

Examples
--------
Default (picks first trial in ./results, runs to last midnight UTC, auto-refreshes
data if the local parquet cache doesn't cover --run-until):

    python scripts/readout.py --base-dir ./results

Read out a specific trial, skipping the auto-refresh check:

    python scripts/readout.py --base-dir ./results \\
        --study-id run_21004... --trial-number 0 --no-refresh
"""

import argparse
import json
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quantammsim.core_simulator.param_utils import retrieve_best, calc_lamb
from quantammsim.pools.creator import create_pool
from quantammsim.runners.jax_runners import do_run_on_historic_data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def default_run_until() -> str:
    """Last midnight UTC, formatted to match run_fingerprint date convention."""
    now = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    return now.strftime("%Y-%m-%d %H:%M:%S")


def build_cli_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--base-dir", required=True,
                   help="Directory containing run_*.json training result files.")
    p.add_argument("--study-id", default=None,
                   help="Specific study_id (filename without .json). If omitted, "
                        "picks the first/best trial across all run_*.json in base-dir.")
    p.add_argument("--trial-number", type=int, default=None,
                   help="Specific trial_number (== retrieve_best's step). If omitted, "
                        "takes the first result from retrieve_best for the chosen study.")
    p.add_argument("--load-method", default="best_train_min_test_objective",
                   choices=["last", "best_objective", "best_train_objective",
                            "best_test_objective", "best_train_min_test_objective"])
    p.add_argument("--tokens", nargs="+", default=None,
                   help="Override tokens from the trial's fingerprint (rarely needed).")
    p.add_argument("--token-addresses", nargs="+", default=None,
                   metavar="0xADDR",
                   help="Token contract addresses, one per --tokens, SAME ORDER. "
                        "The saved JSON includes sc_deployment_fields permuted into "
                        "address-ascending order (the protocol's convention). Required "
                        "unless --no-addresses is set.")
    p.add_argument("--no-addresses", action="store_true",
                   help="Opt out of producing SC-deploy-ready output. Saved JSON will "
                        "contain only the sim-view (ticker order) and the SC block will "
                        "be omitted entirely. For inspection/debugging, not deployment. "
                        "Exactly one of --token-addresses or --no-addresses is required.")
    p.add_argument("--run-until", default=default_run_until(),
                   help="End of the forward-pass window. Defaults to last midnight UTC.")
    p.add_argument("--no-refresh", action="store_true",
                   help="Skip the auto-refresh check. Default behaviour peeks at each "
                        "token's cached parquet and only runs scripts/download_data.py "
                        "for tokens whose cache ends before --run-until.")
    p.add_argument("--out", default=None,
                   help="Path to write the Solidity-ready JSON output. Default: "
                        "{base_dir}/readouts/{study_id}_trial{N}_runUntil{YYYY-MM-DD}.json")
    return p


# ---------------------------------------------------------------------------
# Solidity-compatible 1e18-scaled encoding
# ---------------------------------------------------------------------------
# Solidity has no floats. The convention used by most ERC20 / TFMM contracts
# is 18-decimal fixed-point: the on-chain integer value = round(x * 1e18).
# We emit the scaled values as STRINGS in JSON because:
#   - many scaled values exceed 2^53-1 (JavaScript's safe-integer cap), so
#     parsing as Number loses precision
#   - ethers.js / viem / web3.py all accept decimal strings for BigNumber /
#     uint256 fields
SCALE_1E18 = 10 ** 18


def to_wei_strings(value):
    """Scale a float (or 1-D float array) by 1e18, round to int, return as string(s)."""
    arr = np.asarray(value).flatten()
    return [str(int(round(float(x) * SCALE_1E18))) for x in arr]


# ---------------------------------------------------------------------------
# Chain-order permutation: sim runs in ticker order; contracts want
# address-ascending order. Permute per-token arrays on the way out.
# ---------------------------------------------------------------------------

def _validate_address(a: str) -> str:
    """Accept any 20-byte hex address; return lowercased form for stable sorting."""
    if not isinstance(a, str) or not a.startswith("0x") or len(a) != 42:
        raise ValueError(f"Invalid token address {a!r}: expected '0x' + 40 hex chars")
    try:
        int(a, 16)
    except ValueError:
        raise ValueError(f"Invalid token address {a!r}: not valid hex")
    return a.lower()


def compute_address_permutation(tokens, addresses):
    """Return the indices that reorder per-token arrays from ticker-order to
    address-ascending order. Also returns the reordered (tokens, addresses) pair.
    """
    if len(tokens) != len(addresses):
        raise ValueError(
            f"Length mismatch: {len(tokens)} tokens vs {len(addresses)} addresses. "
            f"Pass --token-addresses in the SAME order as --tokens."
        )
    lowered = [_validate_address(a) for a in addresses]
    if len(set(lowered)) != len(lowered):
        raise ValueError(f"Duplicate token addresses: {lowered}")
    perm = sorted(range(len(lowered)), key=lambda i: lowered[i])
    return perm, [tokens[i] for i in perm], [lowered[i] for i in perm]


def _reorder(value, perm):
    """Apply a permutation of indices to a 1-D float array; leave scalars alone."""
    arr = np.asarray(value)
    if arr.ndim == 0:
        return value
    return arr[list(perm)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_run_fingerprint(run_file: Path) -> dict:
    """run_*.json files are double-JSON-encoded lists. Entry [0] is the fingerprint."""
    with run_file.open() as f:
        data = json.loads(json.load(f))
    return data[0]


def find_trial(base_dir: Path, study_id: str | None):
    """Locate a training result file in ``base_dir`` and return its path."""
    if study_id:
        run_file = base_dir / f"{study_id}.json"
        if not run_file.exists():
            raise FileNotFoundError(f"Study file not found: {run_file}")
        return run_file

    candidates = sorted(base_dir.glob("run_*.json"))
    if not candidates:
        raise FileNotFoundError(f"No run_*.json files in {base_dir}")
    if len(candidates) > 1:
        print(f"[info] {len(candidates)} studies in {base_dir}; picking first: "
              f"{candidates[0].name}. Use --study-id to disambiguate.")
    return candidates[0]


def load_trial_params(run_file: Path, trial_number: int | None, load_method: str):
    """Load trained params for a specific trial (or the first match) from a run file."""
    params_list, steps = retrieve_best(
        str(run_file), load_method, re_calc_hess=False, min_alt_obj=0.0,
        return_as_iterables=True,
    )
    if not params_list:
        raise ValueError(f"No trials in {run_file.name} matched load_method={load_method}")

    if trial_number is not None:
        try:
            idx = steps.index(trial_number)
        except ValueError:
            raise ValueError(
                f"trial_number={trial_number} not found in {run_file.name}. "
                f"Available: {steps[:10]}{'…' if len(steps) > 10 else ''}"
            )
    else:
        idx = 0

    trial_params = params_list[idx]
    trial_step = steps[idx]

    # Strip scalar metadata that retrieve_best doesn't pop but isn't a real param.
    for k in ("train_objective", "test_objective", "objective",
              "optuna_trial_number", "train_return", "train_returns_over_hodl",
              "train_sharpe", "validation_return", "validation_returns_over_hodl",
              "validation_returns_over_uniform_hodl", "validation_sharpe"):
        trial_params.pop(k, None)

    # Params loaded from JSON come back as Python lists; coerce to jax arrays.
    trial_params = {
        k: (jnp.asarray(v) if isinstance(v, (list, tuple)) else v)
        for k, v in trial_params.items()
    }

    # optuna doesn't save initial_weights_logits; synthesise zero template.
    fingerprint = _load_run_fingerprint(run_file)
    n_assets = len(fingerprint["tokens"])
    if "initial_weights_logits" not in trial_params:
        trial_params["initial_weights_logits"] = jnp.zeros(n_assets)

    return trial_params, trial_step, fingerprint


def refresh_data(tokens: list[str]) -> None:
    """Call scripts/download_data.py for the given tickers to fetch any new data."""
    script = Path(__file__).resolve().parent / "download_data.py"
    print(f"[refresh] Running {script.name} {' '.join(tokens)}")
    subprocess.run([sys.executable, str(script), *tokens], check=True)


def _data_dir() -> Path:
    """Location the download script writes parquet files to."""
    return Path(__file__).resolve().parent.parent / "quantammsim" / "data"


def _cache_end(token: str) -> datetime | None:
    """Last timestamp present in ``{token}_USD.parquet`` (UTC), or None if absent."""
    parquet = _data_dir() / f"{token}_USD.parquet"
    if not parquet.exists():
        return None
    df = pd.read_parquet(parquet, columns=["unix"])
    if df.empty:
        return None
    # download_data.py stores unix as milliseconds.
    return datetime.fromtimestamp(int(df["unix"].max()) / 1000, tz=timezone.utc)


def auto_refresh_if_needed(tokens: list[str], run_until: str) -> None:
    """Refresh any token whose cache doesn't cover run_until. No-op if all are fresh."""
    run_until_dt = datetime.strptime(run_until, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    needy = []
    for t in tokens:
        end = _cache_end(t)
        if end is None:
            print(f"[data] {t:6s} no cache found → refreshing")
            needy.append(t)
        elif end < run_until_dt:
            print(f"[data] {t:6s} cache ends {end:%Y-%m-%d %H:%M} < {run_until_dt:%Y-%m-%d %H:%M} → refreshing")
            needy.append(t)
        else:
            print(f"[data] {t:6s} cache ends {end:%Y-%m-%d %H:%M} ≥ {run_until_dt:%Y-%m-%d %H:%M}, using cached")
    if needy:
        refresh_data(needy)


def build_readout_fingerprint(trial_fingerprint: dict, run_until: str,
                              tokens_override: list[str] | None) -> dict:
    """Build a fingerprint for the readout forward-pass.

    Keeps the trial's own start (so readout state warms up over the full
    training history), extends end to ``run_until``, drops any test window.
    """
    fp = deepcopy(trial_fingerprint)
    fp["endDateString"] = run_until
    fp["endTestDateString"] = None   # no OOS slice — we just want up-to-now state
    if tokens_override:
        fp["tokens"] = tokens_override
    return fp


# ---------------------------------------------------------------------------
# Printers
# ---------------------------------------------------------------------------

def _fmt_array(v, precision=8):
    arr = np.asarray(v)
    if arr.ndim == 0:
        return f"{float(arr):.{precision}f}"
    return "[" + " ".join(f"{float(x):.{precision}f}" for x in arr) + "]"


def print_sc_params(pool, params, fingerprint):
    contract = pool.to_contract_params(params, fingerprint)
    print("Smart-contract form parameters")
    print("-" * 70)
    for k, v in contract.items():
        print(f"  {k:20s} {_fmt_array(v)}")


def print_readouts(result):
    print("Final readout state (most recent time step)")
    print("-" * 70)
    readouts = result.get("readouts") or {}
    if not readouts:
        print("  (pool produced no readouts — strategy has no internal state?)")
        return
    for name, series in readouts.items():
        arr = np.asarray(series)
        last = arr[-1] if arr.ndim >= 1 else arr
        print(f"  {name:20s} {_fmt_array(last)}")


def print_final_state(result, tokens):
    print("Final simulation state")
    print("-" * 70)
    print(f"  {'tokens':20s} {tokens}")
    print(f"  {'final weights':20s} {_fmt_array(result['weights'][-1])}")
    print(f"  {'final prices':20s} {_fmt_array(result['prices'][-1])}")
    print(f"  {'initial value':20s} ${float(result['value'][0]):,.2f}")
    print(f"  {'final value':20s} ${float(result['final_value']):,.2f}")
    ret = float(result["final_value"]) / float(result["value"][0]) - 1.0
    print(f"  {'return':20s} {ret*100:+.2f}%")


def build_solidity_payload(pool, params, fingerprint, result, trial_step,
                           run_file, run_until, tokens, token_addresses):
    """Assemble the disk-bound JSON payload.

    Two output shapes, selected by ``token_addresses``:

    * **Deployment mode** (``token_addresses`` is a list): per-token arrays
      permuted into protocol (address-ascending) order, and the payload
      includes a ``sc_deployment_fields`` block with SC-named fields
      (``lambda``, ``_initialWeights``, ``_initialMovingAverages``,
      ``_initialIntermediateValues``, ``absoluteWeightGuardRail``,
      ``updateInterval``). Drop-in for the v3 deployment input.ts.

    * **Sim-view mode** (``token_addresses`` is None, callee already validated
      that ``--no-addresses`` was set): per-token arrays in ticker order, no
      ``sc_deployment_fields``. The absence of that block IS the signal that
      the file isn't deploy-ready — a deployment tool reading it will fail
      fast rather than silently mis-permute.

    Scaling: values are 1e18 fixed-point decimal strings UNLESS stated
    otherwise. ``updateInterval`` is a plain integer (seconds).
    """
    contract = pool.to_contract_params(params, fingerprint)
    readouts = result.get("readouts") or {}
    final_readouts = {
        name: np.asarray(series)[-1] if np.asarray(series).ndim >= 1 else np.asarray(series)
        for name, series in readouts.items()
    }

    meta = {
        "study_id": run_file.stem,
        "trial_number": int(trial_step),
        "rule": fingerprint["rule"],
        "train_start": fingerprint["startDateString"],
        "run_until": run_until,
        "scale": "1e18 except updateInterval (raw seconds)",
        "encoding": "decimal-string (uint256/int256 safe) for 1e18-scaled values",
    }

    # Sim-view mode: no addresses → emit contract_params + readouts only.
    if token_addresses is None:
        meta["mode"] = "sim-view"
        meta["token_order"] = "ticker"
        meta["tokens"] = list(tokens)
        meta["note"] = (
            "Invoked with --no-addresses; sc_deployment_fields omitted. "
            "For deployment, re-run with --token-addresses <addr_per_token>."
        )
        return {
            "meta": meta,
            "contract_params": {k: to_wei_strings(v) for k, v in contract.items()},
            "readouts": {k: to_wei_strings(v) for k, v in final_readouts.items()},
        }

    # Deployment mode: permute + emit sc_deployment_fields.
    perm, reordered_tokens, reordered_addresses = compute_address_permutation(
        tokens, token_addresses,
    )
    meta["mode"] = "deployment"
    meta["token_order"] = "address-ascending"
    meta["tokens"] = reordered_tokens
    meta["token_addresses"] = reordered_addresses

    # initial_weights goes to _initialWeights; peel out so contract_params
    # holds only rule-specific values.
    initial_weights = contract.pop("initial_weights", None)
    lamb = np.asarray(calc_lamb(params))

    contract = {k: _reorder(v, perm) for k, v in contract.items()}
    final_readouts = {k: _reorder(v, perm) for k, v in final_readouts.items()}
    if initial_weights is not None:
        initial_weights = _reorder(initial_weights, perm)
    lamb = _reorder(lamb, perm)

    sc_fields: dict = {"lambda": to_wei_strings(lamb)}
    if initial_weights is not None:
        sc_fields["_initialWeights"] = to_wei_strings(initial_weights)
    # EWMA → _initialMovingAverages; running_a → _initialIntermediateValues.
    # Other readouts (e.g. gradients) are derived on-chain and not deployed.
    if "ewma" in final_readouts:
        sc_fields["_initialMovingAverages"] = to_wei_strings(final_readouts["ewma"])
    if "running_a" in final_readouts:
        sc_fields["_initialIntermediateValues"] = to_wei_strings(final_readouts["running_a"])
    if fingerprint.get("minimum_weight") is not None:
        sc_fields["absoluteWeightGuardRail"] = to_wei_strings(fingerprint["minimum_weight"])[0]
    if "chunk_period" in fingerprint:
        # chunk_period is minutes in sim convention; SC expects seconds.
        # Plain integer — not fp-scaled.
        sc_fields["updateInterval"] = int(fingerprint["chunk_period"]) * 60

    return {
        "meta": meta,
        "sc_deployment_fields": sc_fields,
        "contract_params": {k: to_wei_strings(v) for k, v in contract.items()},
    }


def default_out_path(base_dir: Path, run_file: Path, trial_step: int, run_until: str) -> Path:
    date_slug = run_until.split(" ")[0]  # YYYY-MM-DD
    return base_dir / "readouts" / f"{run_file.stem}_trial{trial_step}_runUntil{date_slug}.json"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = build_cli_parser().parse_args()

    # Force the user to declare intent: either provide deploy-ready addresses,
    # or explicitly opt into sim-view mode. Stops the silent-wrong-order
    # failure mode where a deployment tool picks up ticker-ordered data.
    if args.token_addresses is None and not args.no_addresses:
        raise SystemExit(
            "Error: exactly one of --token-addresses or --no-addresses is required.\n"
            "  • --token-addresses 0xA... 0xB... (one per --tokens, same order)\n"
            "    → produces a deploy-ready file with sc_deployment_fields in\n"
            "      address-ascending order.\n"
            "  • --no-addresses\n"
            "    → sim-view only (no sc_deployment_fields); for inspection."
        )
    if args.token_addresses is not None and args.no_addresses:
        raise SystemExit(
            "Error: --token-addresses and --no-addresses are mutually exclusive."
        )

    base_dir = Path(args.base_dir)

    run_file = find_trial(base_dir, args.study_id)
    params, trial_step, trial_fingerprint = load_trial_params(
        run_file, args.trial_number, args.load_method,
    )

    tokens = args.tokens or trial_fingerprint["tokens"]
    if not args.no_refresh:
        auto_refresh_if_needed(tokens, args.run_until)

    fingerprint = build_readout_fingerprint(trial_fingerprint, args.run_until, args.tokens)
    pool = create_pool(fingerprint["rule"])

    print("=" * 70)
    print(f"Readout: {fingerprint['rule']} on {'/'.join(tokens)}")
    print("=" * 70)
    print(f"Study:       {run_file.name}")
    print(f"Trial step:  {trial_step}  (load_method={args.load_method})")
    print(f"Train start: {fingerprint['startDateString']}")
    print(f"Run until:   {fingerprint['endDateString']}")
    print("=" * 70)

    print_sc_params(pool, params, fingerprint)
    print()

    result = do_run_on_historic_data(
        run_fingerprint=fingerprint,
        params=params,
        verbose=False,
    )

    print_readouts(result)
    print()
    print_final_state(result, tokens)
    print("=" * 70)

    # Save Solidity-ready payload to disk.
    payload = build_solidity_payload(
        pool, params, fingerprint, result, trial_step, run_file, args.run_until,
        tokens, args.token_addresses,
    )
    out_path = Path(args.out) if args.out else default_out_path(
        base_dir, run_file, trial_step, args.run_until,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[saved] Solidity-ready (1e18 scaled, decimal-string) JSON → {out_path}")


if __name__ == "__main__":
    main()
