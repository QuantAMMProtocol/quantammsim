from typing import Type, TypeVar
from abc import ABC

from jax import tree_util

from quantammsim.pools.G3M.balancer.balancer import BalancerPool
from quantammsim.pools.G3M.quantamm.momentum_pool import MomentumPool
from quantammsim.pools.G3M.quantamm.antimomentum_pool import AntiMomentumPool
from quantammsim.pools.G3M.quantamm.power_channel_pool import PowerChannelPool
from quantammsim.pools.G3M.quantamm.mean_reversion_channel_pool import (
    MeanReversionChannelPool,
)
from quantammsim.pools.G3M.quantamm.triple_threat_mean_reversion_channel_pool import TripleThreatMeanReversionChannelPool
from quantammsim.pools.G3M.quantamm.difference_momentum_pool import DifferenceMomentumPool
from quantammsim.pools.G3M.quantamm.index_market_cap_pool import IndexMarketCapPool
from quantammsim.pools.G3M.quantamm.hodling_index_pool import HodlingIndexPool
from quantammsim.pools.G3M.quantamm.trad_hodling_index_pool import TradHodlingIndexPool
from quantammsim.pools.G3M.quantamm.min_variance_pool import MinVariancePool
from quantammsim.pools.hodl_pool import HODLPool
from quantammsim.pools.FM_AMM.cow_pool import CowPool
from quantammsim.pools.ECLP.gyroscope import GyroscopePool
from quantammsim.pools.base_pool import AbstractPool
from quantammsim.hooks.versus_rebalancing import (
    CalculateLossVersusRebalancing,
    CalculateRebalancingVersusRebalancing,
)
from quantammsim.hooks.bounded_weights_hook import BoundedWeightsHook
from quantammsim.hooks.ensemble_averaging_hook import EnsembleAveragingHook

# Create a type variable bound to AbstractPool
P = TypeVar("P", bound=AbstractPool)
H = TypeVar("H", bound=ABC)  # For hooks


def create_hooked_pool_instance(base_pool_class: Type[P], *hooks: Type) -> P:
    """
    Create a pool instance with hook classes mixed in via Python MRO.

    Constructs a dynamic ``_HookedPool`` class whose MRO places the hook
    classes before the base pool class, then instantiates it. Hooks are
    applied right-to-left so that the **first** hook listed has the highest
    priority in method dispatch (i.e. its methods shadow those of later hooks
    and the base pool).

    The resulting mixed class is registered as a JAX pytree, which is
    required for the instance to be passed through ``jit``-compiled
    functions.

    Parameters
    ----------
    base_pool_class : Type[P]
        The pool class to use as the base, e.g. ``MomentumPool``,
        ``MeanReversionChannelPool``.
    *hooks : Type
        One or more hook classes to mix in, e.g.
        ``BoundedWeightsHook``, ``EnsembleAveragingHook``. The first
        hook listed takes highest priority in the MRO.

    Returns
    -------
    P
        An instance of the dynamically created ``_HookedPool`` class,
        with all hook and base-pool methods available via standard
        Python method resolution.

    Examples
    --------
    >>> pool = create_hooked_pool_instance(
    ...     MomentumPool, EnsembleAveragingHook, BoundedWeightsHook
    ... )
    """

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

    This function acts as a central registry for all available pool types in
    the system. New pool implementations must be:

    1. Imported at the top of this file
    2. Added to the if/elif chain below with a unique string identifier

    to be accessible through the simulation runners.

    Parameters
    ----------
    rule : str
        The identifier string for the desired pool type. May optionally
        include hook prefixes using double-underscore syntax (see Notes).

        Valid base pool types:

        - ``"balancer"`` : Standard Balancer constant-weight pool.
        - ``"momentum"`` : Momentum (trend-following) QuantAMM pool.
        - ``"anti_momentum"`` : Anti-momentum (contrarian) QuantAMM pool.
        - ``"power_channel"`` : Power-law channel QuantAMM pool.
        - ``"mean_reversion_channel"`` : Mean-reversion channel QuantAMM pool.
        - ``"triple_threat_mean_reversion_channel"`` : Combined mean-reversion
          channel + trend-following QuantAMM pool.
        - ``"difference_momentum"`` : Difference-of-momentum QuantAMM pool.
        - ``"index_market_cap"`` : Market-cap-weighted index pool.
        - ``"hodling_index_market_cap"`` : HODLing variant of the market-cap
          index pool (on-chain reserve mechanics).
        - ``"trad_hodling_index_market_cap"`` : Traditional (off-chain) HODLing
          variant with realistic CEX trading costs.
        - ``"min_variance"`` : Minimum-variance QuantAMM pool.
        - ``"hodl"`` : Pure buy-and-hold (no rebalancing) pool.
        - ``"cow"`` : CoW (Coincidence of Wants) AMM pool.
        - ``"gyroscope"`` : Gyroscope E-CLP pool.

        Available hook prefixes (prepended with ``__`` separator):

        - ``"lvr"`` : Loss-versus-rebalancing accounting hook.
        - ``"rvr"`` : Rebalancing-versus-rebalancing accounting hook.
        - ``"bounded"`` : Bounded-weights guardrail hook.
        - ``"ensemble"`` : Ensemble-averaging hook.

    Returns
    -------
    AbstractPool
        An instance of the specified pool class (or a hooked composite
        class) ready for use in simulations.

    Raises
    ------
    NotImplementedError
        If the base pool type or any hook prefix is unrecognised.

    Notes
    -----
    **Hook prefix system**

    Hooks are chained onto a base pool using double-underscore (``__``)
    delimiters. The base pool type is always the **last** segment. For
    example::

        "ensemble__bounded__momentum"

    is parsed as hooks ``[EnsembleAveragingHook, BoundedWeightsHook]``
    applied to ``MomentumPool``, with ``EnsembleAveragingHook`` having
    highest priority in the MRO.

    To add a new pool type:

    1. Create the new pool class implementing the required interfaces.
    2. Import the class at the top of this file.
    3. Add a new ``elif`` clause matching the desired identifier string.
    4. Return an instance of the new pool class.

    Examples
    --------
    >>> pool = create_pool("balancer")
    >>> pool = create_pool("momentum")
    >>> pool = create_pool("ensemble__bounded__momentum")
    """
    # Split rule into hook_types and base_rule
    # Supports multiple hooks: "ensemble__bounded__momentum" -> hooks=[ensemble, bounded], base=momentum
    parts = rule.split("__")
    base_rule = parts[-1]
    hook_types = parts[:-1] if len(parts) > 1 else []

    # Create base pool instance
    if base_rule == "balancer":
        base_pool = BalancerPool()
    elif base_rule == "momentum":
        base_pool = MomentumPool()
    elif base_rule == "anti_momentum":
        base_pool = AntiMomentumPool()
    elif base_rule == "power_channel":
        base_pool = PowerChannelPool()
    elif base_rule == "mean_reversion_channel":
        base_pool = MeanReversionChannelPool()
    elif base_rule == "triple_threat_mean_reversion_channel":
        base_pool = TripleThreatMeanReversionChannelPool()
    elif base_rule == "difference_momentum":
        base_pool = DifferenceMomentumPool()
    elif base_rule == "index_market_cap":
        base_pool = IndexMarketCapPool()
    elif base_rule == "hodling_index_market_cap":
        base_pool = HodlingIndexPool()
    elif base_rule == "trad_hodling_index_market_cap":
        base_pool = TradHodlingIndexPool()
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

    # Map hook names to classes
    hook_map = {
        "lvr": CalculateLossVersusRebalancing,
        "rvr": CalculateRebalancingVersusRebalancing,
        "bounded": BoundedWeightsHook,
        "ensemble": EnsembleAveragingHook,
    }

    # Apply hooks if specified
    if hook_types:
        hooks = []
        for hook_type in hook_types:
            if hook_type not in hook_map:
                raise NotImplementedError(f"Unknown hook type: {hook_type}")
            hooks.append(hook_map[hook_type])
        return create_hooked_pool_instance(base_pool.__class__, *hooks)

    return base_pool
