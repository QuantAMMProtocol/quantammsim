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
    """Fixed-structure JAX pytree for dynamic simulation inputs."""

    trades: jnp.ndarray
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
    """Create a canonical empty bundle with stable pytree structure."""
    return DynamicInputArrays(
        trades=jnp.zeros((1, 3), dtype=jnp.float64),
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
