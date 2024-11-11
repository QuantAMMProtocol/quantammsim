from quantammsim.pools.G3M.balancer.balancer import BalancerPool
from quantammsim.pools.G3M.quantamm.momentum_pool import MomentumPool
from quantammsim.pools.G3M.quantamm.power_channel_pool import PowerChannelPool
from quantammsim.pools.G3M.quantamm.mean_reversion_channel_pool import MeanReversionChannelPool
from quantammsim.pools.hodl_pool import HODLPool
from quantammsim.pools.FM_AMM.cow_pool import CowPool
from quantammsim.pools.FM_AMM.cow_pool_one_arb import CowPoolOneArb

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
    # Create pool
    if rule == "balancer":
        pool = BalancerPool()
    elif rule == "momentum":
        pool = MomentumPool()
    elif rule == "power_channel":
        pool = PowerChannelPool()
    elif rule == "mean_reversion_channel":
        pool = MeanReversionChannelPool()
    elif rule == "hodl":
        pool = HODLPool()
    elif rule == "cow_5050":
        pool = CowPool()
    elif rule == "cow_5050_one_arb":
        pool = CowPoolOneArb()
    else:
        raise NotImplementedError
    return pool