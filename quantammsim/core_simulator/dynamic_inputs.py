from dataclasses import dataclass
from typing import Any, NamedTuple, Optional

import jax.numpy as jnp


@dataclass(frozen=True)
class DynamicInputFrames:
    """Outer-layer container for optional pandas-backed dynamic inputs."""

    trades: Optional[Any] = None
    fees: Optional[Any] = None
    gas_cost: Optional[Any] = None
    arb_fees: Optional[Any] = None
    lp_supply: Optional[Any] = None


class DynamicInputArrays(NamedTuple):
    """JAX pytree for dynamic simulation inputs with optional trade data."""

    trades: Optional[jnp.ndarray]
    fees: jnp.ndarray
    gas_cost: jnp.ndarray
    arb_fees: jnp.ndarray
    lp_supply: jnp.ndarray


def default_dynamic_input_flags() -> dict:
    """Static dispatch flags for forward-pass path selection."""
    return {
        "use_dynamic_inputs": False,
        "has_trades": False,
        "has_dynamic_fees": False,
        "has_dynamic_gas_cost": False,
        "has_dynamic_arb_fees": False,
        "has_lp_supply": False,
    }


def dynamic_input_flags_from_frames(dynamic_input_frames: Optional[DynamicInputFrames]) -> dict:
    """Build stable dispatch flags from the outer-layer frame container."""
    if dynamic_input_frames is None:
        return default_dynamic_input_flags()

    flags = {
        "use_dynamic_inputs": False,
        "has_trades": dynamic_input_frames.trades is not None,
        "has_dynamic_fees": dynamic_input_frames.fees is not None,
        "has_dynamic_gas_cost": dynamic_input_frames.gas_cost is not None,
        "has_dynamic_arb_fees": dynamic_input_frames.arb_fees is not None,
        "has_lp_supply": dynamic_input_frames.lp_supply is not None,
    }
    flags["use_dynamic_inputs"] = any(flags.values())
    return flags


def resolve_dynamic_input_flags(
    dynamic_inputs: Optional[DynamicInputArrays],
    dynamic_input_flags: Optional[dict] = None,
) -> dict:
    """Return a safe dispatch flag set for the provided hot-path bundle."""
    flags = (
        default_dynamic_input_flags()
        if dynamic_input_flags is None
        else dict(dynamic_input_flags)
    )
    if dynamic_inputs is not None:
        flags["use_dynamic_inputs"] = True
    return flags


def empty_dynamic_input_arrays() -> DynamicInputArrays:
    """Create a canonical empty bundle."""
    return DynamicInputArrays(
        trades=None,
        fees=jnp.zeros((1,), dtype=jnp.float64),
        gas_cost=jnp.zeros((1,), dtype=jnp.float64),
        arb_fees=jnp.zeros((1,), dtype=jnp.float64),
        lp_supply=jnp.ones((1,), dtype=jnp.float64),
    )


def resolve_dynamic_input_components(
    dynamic_inputs: Optional[DynamicInputArrays],
    dynamic_input_flags: dict,
    static_dict: dict,
) -> dict:
    """Resolve dynamic-input leaves against static scalar defaults."""
    arrays = empty_dynamic_input_arrays() if dynamic_inputs is None else dynamic_inputs
    return {
        "trades": arrays.trades if dynamic_input_flags["has_trades"] else None,
        "fees": (
            arrays.fees
            if dynamic_input_flags["has_dynamic_fees"]
            else jnp.asarray([static_dict["fees"]], dtype=jnp.float64)
        ),
        "gas_cost": (
            arrays.gas_cost
            if dynamic_input_flags["has_dynamic_gas_cost"]
            else jnp.asarray([static_dict["gas_cost"]], dtype=jnp.float64)
        ),
        "arb_fees": (
            arrays.arb_fees
            if dynamic_input_flags["has_dynamic_arb_fees"]
            else jnp.asarray([static_dict["arb_fees"]], dtype=jnp.float64)
        ),
        "lp_supply": (
            arrays.lp_supply
            if dynamic_input_flags["has_lp_supply"]
            else jnp.ones((1,), dtype=jnp.float64)
        ),
    }


def _broadcast_dynamic_input_leaf(
    input_name: str,
    values: jnp.ndarray,
    scan_len: int,
    dtype,
) -> jnp.ndarray:
    """Broadcast a singleton dynamic-input leaf to the scan length."""
    values = jnp.asarray(values, dtype=dtype)
    if values.ndim == 0:
        values = values.reshape((1,))
    if values.shape[0] == scan_len:
        return values
    if values.shape[0] == 1:
        return jnp.broadcast_to(values, (scan_len,) + values.shape[1:])
    raise ValueError(
        f"{input_name} has leading axis {values.shape[0]}, expected 1 or {scan_len}"
    )


def materialize_dynamic_inputs(
    dynamic_inputs: Optional[DynamicInputArrays],
    dynamic_input_flags: Optional[dict],
    static_dict: dict,
    scan_len: int,
    do_trades: bool,
    dtype=jnp.float64,
) -> DynamicInputArrays:
    """Resolve and broadcast dynamic inputs for a specific scan length."""
    if dynamic_input_flags is None and dynamic_inputs is not None:
        flags = {
            "use_dynamic_inputs": True,
            "has_trades": do_trades,
            "has_dynamic_fees": True,
            "has_dynamic_gas_cost": True,
            "has_dynamic_arb_fees": True,
            "has_lp_supply": True,
        }
    else:
        flags = resolve_dynamic_input_flags(dynamic_inputs, dynamic_input_flags)

    resolved = resolve_dynamic_input_components(dynamic_inputs, flags, static_dict)

    trades = None
    if do_trades:
        if resolved["trades"] is None:
            raise ValueError("Trades must be provided when do_trades=True.")
        trades = _broadcast_dynamic_input_leaf(
            "trades", resolved["trades"], scan_len, dtype
        )

    return DynamicInputArrays(
        trades=trades,
        fees=_broadcast_dynamic_input_leaf("fees", resolved["fees"], scan_len, dtype),
        gas_cost=_broadcast_dynamic_input_leaf(
            "gas_cost", resolved["gas_cost"], scan_len, dtype
        ),
        arb_fees=_broadcast_dynamic_input_leaf(
            "arb_fees", resolved["arb_fees"], scan_len, dtype
        ),
        lp_supply=_broadcast_dynamic_input_leaf(
            "lp_supply", resolved["lp_supply"], scan_len, dtype
        ),
    )
