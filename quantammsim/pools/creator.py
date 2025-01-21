from typing import Type, TypeVar
from abc import ABC

from jax import tree_util

from quantammsim.pools.G3M.balancer.balancer import BalancerPool
from quantammsim.pools.G3M.quantamm.momentum_pool import MomentumPool
from quantammsim.pools.G3M.quantamm.power_channel_pool import PowerChannelPool
from quantammsim.pools.G3M.quantamm.mean_reversion_channel_pool import (
    MeanReversionChannelPool,
)
from quantammsim.pools.G3M.quantamm.difference_momentum_pool import DifferenceMomentumPool
from quantammsim.pools.G3M.quantamm.min_variance_pool import MinVariancePool
from quantammsim.pools.hodl_pool import HODLPool
from quantammsim.pools.FM_AMM.cow_pool import CowPool
from quantammsim.pools.ECLP.gyroscope import GyroscopePool
from quantammsim.pools.base_pool import AbstractPool
from quantammsim.hooks.versus_rebalancing import (
    CalculateLossVersusRebalancing,
    CalculateRebalancingVersusRebalancing,
)

# Create a type variable bound to AbstractPool
P = TypeVar("P", bound=AbstractPool)
H = TypeVar("H", bound=ABC)  # For hooks


def create_hooked_pool_instance(base_pool_class: Type[P], *hooks: Type) -> P:
    """Create a pool instance with the specified hooks mixed in."""

    # Hooks should be applied right-to-left to maintain correct MRO
    hooks = hooks[::-1]

    # Create the mixed class with hooks before base_pool_class
    mixed_class = type("_HookedPool", (*hooks, base_pool_class), {})

    # Verify MRO is correct by checking method resolution
    mro = mixed_class.__mro__
    hook_index = mro.index(hooks[0])
    base_index = mro.index(base_pool_class)

    assert hook_index < base_index, (
        f"Hook {hooks[0].__name__} should come before {base_pool_class.__name__} in MRO. "
        f"Current MRO: {[cls.__name__ for cls in mro]}"
    )

    def _tree_flatten(self):
        all_static_params = {}
        for cls in self.__class__.__mro__:
            if hasattr(cls, "_static_params"):
                all_static_params.update(getattr(cls, "_static_params", {}))
        dynamics = ()
        return dynamics, all_static_params

    @classmethod
    def _tree_unflatten(cls, aux_data, dynamics):
        self = cls()
        self._static_params = aux_data
        return self

    mixed_class._tree_flatten = _tree_flatten
    mixed_class._tree_unflatten = _tree_unflatten

    tree_util.register_pytree_node(
        mixed_class, mixed_class._tree_flatten, mixed_class._tree_unflatten
    )

    return mixed_class()


def create_pool(rule):
    """
    Create a pool instance based on the specified rule type.

    This function acts as a central registry for all available pool types in the system.
    New pool implementations must be:
    1. Imported at the top of this file
    2. Added to the if/elif chain below with a unique string identifier
    to be accessible through the simulation runners.

    Parameters
    ----------
    rule : str
        The identifier string for the desired pool type. Valid options are:
        - "balancer": Standard Balancer pool implementation
        - "momentum": Momentum-based G3M pool variant
        - "power_channel": Power law G3M pool variant
        - "mean_reversion_channel": Mean reversion G3M pool variant
        - "hodl": Basic HODL strategy pool
        - "cow": CoW AMM pool implementation

    Returns
    -------
    Pool
        An instance of the specified pool class ready for use in simulations

    Raises
    ------
    NotImplementedError
        If the provided rule string does not match any registered pool types

    Notes
    -----
    This factory function centralizes pool creation and provides a simple string-based
    interface for specifying pool types in configuration. To add a new pool type:

    1. Create the new pool class implementing required interfaces
    2. Import the class at the top of this file
    3. Add a new elif clause matching the desired identifier string
    4. Return an instance of the new pool class

    The returned pool instance will be automatically compatible with the JAX-based
    simulation runners as long as it implements the required interfaces.

    Examples
    --------
    >>> pool = create_pool("balancer")  # Creates a BalancerPool pool instance
    >>> pool = create_pool("momentum")  # Creates a MomentumPool pool instance
    """
    # Split rule into hook_type and base_rule if double hyphen exists
    hook_type = None
    base_rule = rule
    if "__" in rule:
        hook_type, base_rule = rule.split("__")

    # Create base pool instance
    if base_rule == "balancer":
        base_pool = BalancerPool()
    elif base_rule == "momentum":
        base_pool = MomentumPool()
    elif base_rule == "power_channel":
        base_pool = PowerChannelPool()
    elif base_rule == "mean_reversion_channel":
        base_pool = MeanReversionChannelPool()
    elif base_rule == "difference_momentum":
        base_pool = DifferenceMomentumPool()
    elif base_rule == "min_variance":
        base_pool = MinVariancePool()
    elif base_rule == "hodl":
        base_pool = HODLPool()
    elif base_rule == "cow":
        base_pool = CowPool()
    elif base_rule == "gyroscope":
        base_pool = GyroscopePool()
    else:
        raise NotImplementedError(f"Unknown base pool type: {base_rule}")

    # Apply hook if specified
    if hook_type == "lvr":
        return create_hooked_pool_instance(
            base_pool.__class__, CalculateLossVersusRebalancing
        )
    elif hook_type == "rvr":
        return create_hooked_pool_instance(
            base_pool.__class__, CalculateRebalancingVersusRebalancing
        )
    elif hook_type is not None:
        raise NotImplementedError(f"Unknown hook type: {hook_type}")

    return base_pool
