"""PCHIP interpolation layer for precomputed arb-volume grids.

Two grid formats:
  - v1 (scalar): cadence x gas_cost -> median daily V_arb (single scalar)
  - v2 (daily): cadence x gas_cost x day -> per-day V_arb (vector output)

Interpolation in (log(cadence), gas_cost) space using PCHIP (monotone
piecewise cubic Hermite), which avoids Runge oscillation on non-uniform grids.

Two interfaces:
  1. scipy-based: RegularGridInterpolator(method='pchip') for validation/plotting
  2. JAX-compatible: precomputed slopes + Hermite cubic eval, fully differentiable

The JAX path uses tensor-product evaluation:
  - Along cadence: Hermite cubic with scipy-precomputed PCHIP slopes
  - Along gas: PCHIP slopes computed on the fly from intermediate values
"""

import os
from typing import Dict, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator, RegularGridInterpolator

GRID_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "results",
    "pool_grids",
)

GRID_DIR_V2 = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "results",
    "pool_grids_v2",
)


# ── Grid loading ─────────────────────────────────────────────────────────


def load_pool_grid(pool_id_prefix: str, grid_dir: str = GRID_DIR) -> pd.DataFrame:
    """Load a single pool's grid CSV."""
    path = os.path.join(grid_dir, f"{pool_id_prefix}_grid.csv")
    return pd.read_csv(path)


def load_valid_pool_grids(grid_dir: str = GRID_DIR) -> Dict[str, pd.DataFrame]:
    """Load all pool grid CSVs that have valid (non-NaN) data."""
    grids = {}
    for f in sorted(os.listdir(grid_dir)):
        if f.endswith("_grid.csv") and f != "grid_summary.csv":
            prefix = f.replace("_grid.csv", "")
            df = pd.read_csv(os.path.join(grid_dir, f))
            if df["median_daily_arb_volume"].notna().any():
                grids[prefix] = df
    return grids


def pivot_grid(
    df: pd.DataFrame, value_col: str = "median_daily_arb_volume"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pivot grid DataFrame to (log_cadences, gas_costs, values) arrays."""
    pivot = df.pivot(index="cadence", columns="gas_cost", values=value_col)
    cadences = pivot.index.values.astype(float)
    gas_costs = pivot.columns.values.astype(float)
    values = pivot.values.astype(float)
    return np.log(cadences), gas_costs, values


# ── Scipy interpolation ─────────────────────────────────────────────────


def build_scipy_interpolator(
    df: pd.DataFrame, value_col: str = "median_daily_arb_volume"
) -> RegularGridInterpolator:
    """Build a scipy RegularGridInterpolator with PCHIP method."""
    log_cadences, gas_costs, values = pivot_grid(df, value_col)
    return RegularGridInterpolator(
        (log_cadences, gas_costs),
        values,
        method="pchip",
        bounds_error=False,
        fill_value=None,
    )


def query_scipy(
    interp: RegularGridInterpolator, cadence: float, gas_cost: float
) -> float:
    """Query scipy interpolator at (cadence, gas_cost). Cadence in minutes."""
    log_cad = np.log(np.clip(cadence, 1.0, 60.0))
    return float(interp(np.array([[log_cad, gas_cost]]))[0])


# ── JAX-compatible PCHIP ────────────────────────────────────────────────


class PoolCoeffs(NamedTuple):
    """Precomputed coefficients for one pool's 2D PCHIP interpolation."""

    log_cadences: jnp.ndarray  # (n_cad,)
    gas_costs: jnp.ndarray  # (n_gas,)
    values: jnp.ndarray  # (n_cad, n_gas)
    slopes_cad: jnp.ndarray  # (n_cad, n_gas) PCHIP slopes along cadence axis


def precompute_pool_coeffs(
    df: pd.DataFrame, value_col: str = "median_daily_arb_volume"
) -> PoolCoeffs:
    """Precompute PCHIP slopes along cadence axis using scipy.

    These slopes are used by the JAX evaluation function for the first
    interpolation axis (cadence). The second axis (gas) is computed on
    the fly in JAX to maintain full differentiability.
    """
    log_cadences, gas_costs, values = pivot_grid(df, value_col)

    n_gas = values.shape[1]
    slopes = np.zeros_like(values)
    for j in range(n_gas):
        pchip = PchipInterpolator(log_cadences, values[:, j])
        slopes[:, j] = pchip.derivative()(log_cadences)

    return PoolCoeffs(
        log_cadences=jnp.array(log_cadences),
        gas_costs=jnp.array(gas_costs),
        values=jnp.array(values),
        slopes_cad=jnp.array(slopes),
    )


@jax.jit
def _pchip_slopes(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Compute PCHIP slopes via Fritsch-Carlson method. JAX-compatible.

    x: (n,) sorted knot positions
    y: (n,) values at knots
    Returns: (n,) slopes at knots
    """
    h = x[1:] - x[:-1]
    delta = (y[1:] - y[:-1]) / h

    # Interior points: weighted harmonic mean of neighboring secants
    w1 = 2 * h[1:] + h[:-1]
    w2 = h[1:] + 2 * h[:-1]

    sign_agree = (delta[:-1] * delta[1:]) > 0

    # When sign_agree is False, d_mid is masked to 0. But the harmonic mean
    # can produce Inf when deltas have opposite signs and w1==w2 (denominator
    # cancels). JAX's where can't mask the NaN gradient of Inf (0*NaN=NaN).
    # Fix: replace deltas with 1.0 when sign_agree is False, ensuring hm
    # is always finite. The value is irrelevant since it gets masked.
    d0 = jnp.where(sign_agree, delta[:-1], 1.0)
    d1 = jnp.where(sign_agree, delta[1:], 1.0)
    d0 = jnp.where(d0 == 0, 1e-30, d0)
    d1 = jnp.where(d1 == 0, 1e-30, d1)

    hm = (w1 + w2) / (w1 / d0 + w2 / d1)
    hm = jnp.where(jnp.isfinite(hm), hm, 0.0)
    d_mid = jnp.where(sign_agree, hm, 0.0)

    # Endpoints: one-sided shape-preserving
    d0 = ((2 * h[0] + h[1]) * delta[0] - h[0] * delta[1]) / (h[0] + h[1])
    d0 = jnp.where(d0 * delta[0] <= 0, 0.0, d0)
    d0 = jnp.where(
        (delta[0] * delta[1] < 0) & (jnp.abs(d0) > 3 * jnp.abs(delta[0])),
        3 * delta[0],
        d0,
    )

    dn = ((2 * h[-1] + h[-2]) * delta[-1] - h[-1] * delta[-2]) / (h[-1] + h[-2])
    dn = jnp.where(dn * delta[-1] <= 0, 0.0, dn)
    dn = jnp.where(
        (delta[-1] * delta[-2] < 0) & (jnp.abs(dn) > 3 * jnp.abs(delta[-1])),
        3 * delta[-1],
        dn,
    )

    return jnp.concatenate([d0[None], d_mid, dn[None]])


@jax.jit
def interpolate_pool(
    coeffs: PoolCoeffs, log_cadence: jnp.ndarray, gas_cost: jnp.ndarray
) -> jnp.ndarray:
    """Evaluate 2D PCHIP at (log_cadence, gas_cost). JAX-differentiable.

    Tensor-product approach:
      1. Hermite cubic along cadence for all gas columns (precomputed slopes)
      2. PCHIP slopes along gas through intermediate values (computed on the fly)
      3. Hermite cubic along gas to final value

    Args:
        coeffs: PoolCoeffs from precompute_pool_coeffs
        log_cadence: scalar, log of cadence in minutes
        gas_cost: scalar, effective profit threshold in USD
    Returns:
        V_arb: scalar, interpolated median daily arb volume
    """
    log_cads = coeffs.log_cadences
    gas = coeffs.gas_costs
    vals = coeffs.values
    sl_cad = coeffs.slopes_cad

    # Clamp to grid bounds
    log_cadence = jnp.clip(log_cadence, log_cads[0], log_cads[-1])
    gas_cost = jnp.clip(gas_cost, gas[0], gas[-1])

    # ── Step 1: Hermite along cadence for all gas columns ──
    idx = jnp.searchsorted(log_cads, log_cadence) - 1
    idx = jnp.clip(idx, 0, log_cads.shape[0] - 2)

    h = log_cads[idx + 1] - log_cads[idx]
    t = (log_cadence - log_cads[idx]) / h
    t2 = t * t
    t3 = t2 * t

    h00 = 2 * t3 - 3 * t2 + 1
    h10 = t3 - 2 * t2 + t
    h01 = -2 * t3 + 3 * t2
    h11 = t3 - t2

    v_at_gas = (
        h00 * vals[idx, :]
        + h01 * vals[idx + 1, :]
        + h * (h10 * sl_cad[idx, :] + h11 * sl_cad[idx + 1, :])
    )

    # ── Step 2: PCHIP slopes along gas ──
    gas_slopes = _pchip_slopes(gas, v_at_gas)

    # ── Step 3: Hermite along gas ──
    jdx = jnp.searchsorted(gas, gas_cost) - 1
    jdx = jnp.clip(jdx, 0, gas.shape[0] - 2)

    hg = gas[jdx + 1] - gas[jdx]
    s = (gas_cost - gas[jdx]) / hg
    s2 = s * s
    s3 = s2 * s

    g00 = 2 * s3 - 3 * s2 + 1
    g10 = s3 - 2 * s2 + s
    g01 = -2 * s3 + 3 * s2
    g11 = s3 - s2

    return (
        g00 * v_at_gas[jdx]
        + g01 * v_at_gas[jdx + 1]
        + hg * (g10 * gas_slopes[jdx] + g11 * gas_slopes[jdx + 1])
    )


# ── Per-day (v2) grid support ──────────────────────────────────────────────


class PoolCoeffsDaily(NamedTuple):
    """Precomputed coefficients for per-day 2D PCHIP interpolation.

    Like PoolCoeffs but values/slopes have a day dimension:
      values: (n_cad, n_gas, n_days)
      slopes_cad: (n_cad, n_gas, n_days)
      dates: (n_days,) ordinal dates for alignment with panel
    """

    log_cadences: jnp.ndarray  # (n_cad,)
    gas_costs: jnp.ndarray  # (n_gas,)
    values: jnp.ndarray  # (n_cad, n_gas, n_days)
    slopes_cad: jnp.ndarray  # (n_cad, n_gas, n_days)
    dates: jnp.ndarray  # (n_days,) ordinal dates


def load_daily_grid(
    pool_id_prefix: str, grid_dir: str = GRID_DIR_V2
) -> pd.DataFrame:
    """Load a pool's per-day grid parquet."""
    path = os.path.join(grid_dir, f"{pool_id_prefix}_daily.parquet")
    return pd.read_parquet(path)


def precompute_pool_coeffs_daily(df: pd.DataFrame) -> PoolCoeffsDaily:
    """Build PoolCoeffsDaily from per-day grid DataFrame.

    Args:
        df: DataFrame with columns [cadence, gas_cost, date, daily_arb_volume]

    Returns:
        PoolCoeffsDaily with 3D values (n_cad, n_gas, n_days)
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    cadences = np.array(sorted(df["cadence"].unique()), dtype=float)
    gas_costs = np.array(sorted(df["gas_cost"].unique()), dtype=float)
    dates = np.array(sorted(df["date"].unique()))
    log_cadences = np.log(cadences)

    n_cad = len(cadences)
    n_gas = len(gas_costs)
    n_days = len(dates)

    # Build 3D array: cadence x gas x day
    values = np.zeros((n_cad, n_gas, n_days))
    cad_idx = {c: i for i, c in enumerate(cadences)}
    gas_idx = {g: i for i, g in enumerate(gas_costs)}
    date_idx = {d: i for i, d in enumerate(dates)}

    for _, row in df.iterrows():
        ci = cad_idx.get(float(row["cadence"]))
        gi = gas_idx.get(float(row["gas_cost"]))
        di = date_idx.get(row["date"])
        if ci is not None and gi is not None and di is not None:
            values[ci, gi, di] = row["daily_arb_volume"]

    # Compute PCHIP slopes along cadence axis for each (gas, day)
    slopes = np.zeros_like(values)
    for j in range(n_gas):
        for k in range(n_days):
            col = values[:, j, k]
            if np.all(np.isfinite(col)):
                pchip = PchipInterpolator(log_cadences, col)
                slopes[:, j, k] = pchip.derivative()(log_cadences)

    # Convert dates to ordinals for JAX
    date_ordinals = np.array([
        pd.Timestamp(d).toordinal() for d in dates
    ], dtype=np.int32)

    return PoolCoeffsDaily(
        log_cadences=jnp.array(log_cadences),
        gas_costs=jnp.array(gas_costs),
        values=jnp.array(values),
        slopes_cad=jnp.array(slopes),
        dates=jnp.array(date_ordinals),
    )


@jax.jit
def interpolate_pool_daily(
    coeffs: PoolCoeffsDaily,
    log_cadence: jnp.ndarray,
    gas_cost: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate 2D PCHIP at (log_cadence, gas_cost) for all days.

    Same tensor-product approach as interpolate_pool, but values are 3D
    (n_cad, n_gas, n_days) so the output is (n_days,).

    The Hermite basis coefficients are scalars that broadcast over the
    day dimension of values.

    Args:
        coeffs: PoolCoeffsDaily from precompute_pool_coeffs_daily
        log_cadence: scalar, log of cadence in minutes
        gas_cost: scalar, effective profit threshold in USD
    Returns:
        V_arb: (n_days,) interpolated daily arb volume
    """
    log_cads = coeffs.log_cadences
    gas = coeffs.gas_costs
    vals = coeffs.values  # (n_cad, n_gas, n_days)
    sl_cad = coeffs.slopes_cad  # (n_cad, n_gas, n_days)

    # Clamp to grid bounds
    log_cadence = jnp.clip(log_cadence, log_cads[0], log_cads[-1])
    gas_cost = jnp.clip(gas_cost, gas[0], gas[-1])

    # ── Step 1: Hermite along cadence for all gas columns, all days ──
    idx = jnp.searchsorted(log_cads, log_cadence) - 1
    idx = jnp.clip(idx, 0, log_cads.shape[0] - 2)

    h = log_cads[idx + 1] - log_cads[idx]
    t = (log_cadence - log_cads[idx]) / h
    t2 = t * t
    t3 = t2 * t

    h00 = 2 * t3 - 3 * t2 + 1
    h10 = t3 - 2 * t2 + t
    h01 = -2 * t3 + 3 * t2
    h11 = t3 - t2

    # vals[idx, :, :] is (n_gas, n_days) — scalars broadcast
    v_at_gas = (
        h00 * vals[idx, :, :]
        + h01 * vals[idx + 1, :, :]
        + h * (h10 * sl_cad[idx, :, :] + h11 * sl_cad[idx + 1, :, :])
    )  # (n_gas, n_days)

    # ── Step 2: PCHIP slopes along gas, vmapped over days ──
    gas_slopes = jax.vmap(
        lambda y_col: _pchip_slopes(gas, y_col),
        in_axes=1, out_axes=1,
    )(v_at_gas)  # (n_gas, n_days)

    # ── Step 3: Hermite along gas ──
    jdx = jnp.searchsorted(gas, gas_cost) - 1
    jdx = jnp.clip(jdx, 0, gas.shape[0] - 2)

    hg = gas[jdx + 1] - gas[jdx]
    s = (gas_cost - gas[jdx]) / hg
    s2 = s * s
    s3 = s2 * s

    g00 = 2 * s3 - 3 * s2 + 1
    g10 = s3 - 2 * s2 + s
    g01 = -2 * s3 + 3 * s2
    g11 = s3 - s2

    # v_at_gas[jdx] is (n_days,) — scalars broadcast
    return (
        g00 * v_at_gas[jdx]
        + g01 * v_at_gas[jdx + 1]
        + hg * (g10 * gas_slopes[jdx] + g11 * gas_slopes[jdx + 1])
    )  # (n_days,)


# ── Convenience class ────────────────────────────────────────────────────


class PoolGridInterpolator:
    """Collection of PCHIP interpolators for all valid pool grids.

    Provides both scipy (for validation) and JAX (for optimization) access.
    """

    def __init__(self, grid_dir: str = GRID_DIR):
        self.grid_dir = grid_dir
        grids = load_valid_pool_grids(grid_dir)

        self._scipy_interps = {}
        self._jax_coeffs = {}
        self._pool_ids = sorted(grids.keys())

        for pid, df in grids.items():
            self._scipy_interps[pid] = build_scipy_interpolator(df)
            self._jax_coeffs[pid] = precompute_pool_coeffs(df)

    @property
    def pool_ids(self):
        return list(self._pool_ids)

    @property
    def n_pools(self):
        return len(self._pool_ids)

    def query_scipy(self, pool_id: str, cadence: float, gas_cost: float) -> float:
        """Query scipy PCHIP at (cadence_minutes, gas_cost_usd)."""
        return query_scipy(self._scipy_interps[pool_id], cadence, gas_cost)

    def query_jax(self, pool_id: str, log_cadence, gas_cost):
        """Query JAX PCHIP at (log_cadence, gas_cost). Differentiable."""
        return interpolate_pool(self._jax_coeffs[pool_id], log_cadence, gas_cost)

    def get_coeffs(self, pool_id: str) -> PoolCoeffs:
        """Get precomputed JAX coefficients for a single pool."""
        return self._jax_coeffs[pool_id]

    def get_scipy(self, pool_id: str) -> RegularGridInterpolator:
        """Get scipy interpolator for a single pool."""
        return self._scipy_interps[pool_id]
