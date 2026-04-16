import numpy.testing as npt

import jax.numpy as jnp

from quantammsim.pools.G3M.balancer.hypersurge_balancer_reserves import (
    _hypersurge_fee_for_trade,
)
from quantammsim.pools.creator import create_pool
from quantammsim.pools.hypersurge_utils import HYPERSURGE_PARAM_KEYS
from quantammsim.pools.reCLAMM.reclamm_hypersurge import ReClammHyperSurgePool
from quantammsim.runners.jax_runner_utils import NestedHashabledict


ALL_SIG_VARIATIONS_2 = tuple(map(tuple, [[1, -1], [-1, 1]]))


def _run_fingerprint(n_steps=4):
    return NestedHashabledict(
        {
            "n_assets": 2,
            "bout_length": n_steps + 1,
            "initial_pool_value": 1_000_000.0,
            "arb_frequency": 1,
            "do_arb": True,
            "fees": 0.003,
            "gas_cost": 0.0,
            "arb_fees": 0.0,
            "all_sig_variations": ALL_SIG_VARIATIONS_2,
            "noise_model": "arb_only",
            "noise_trader_ratio": 0.0,
            "hypersurge_arb_max_fee": 0.02,
            "hypersurge_arb_threshold": 0.10,
            "hypersurge_arb_cap_deviation": 0.50,
            "hypersurge_noise_max_fee": 0.10,
            "hypersurge_noise_threshold": 0.10,
            "hypersurge_noise_cap_deviation": 0.50,
        }
    )


def test_creator_registers_hypersurge_reclamm_aliases():
    assert isinstance(create_pool("reclamm_hypersurge"), ReClammHyperSurgePool)
    assert isinstance(create_pool("hypersurge_reclamm"), ReClammHyperSurgePool)


def test_hypersurge_reclamm_params_are_trainable_by_default():
    pool = create_pool("reclamm_hypersurge")
    run_fingerprint = _run_fingerprint()
    initial_values = pool.get_initial_values(run_fingerprint)

    params = pool.init_parameters(
        initial_values,
        run_fingerprint,
        n_assets=2,
        n_parameter_sets=3,
        noise="gaussian",
    )

    assert pool.is_trainable()
    assert "price_ratio" in params
    assert "centeredness_margin" in params
    for key in HYPERSURGE_PARAM_KEYS:
        assert key in params
        assert params[key].shape == (3, 1)


def test_hypersurge_fee_falls_back_to_base_fee_when_oracle_invalid():
    reserves = jnp.array([5000.0, 2500.0])
    weights = jnp.array([0.5, 0.5])
    hypersurge_params = jnp.array([0.02, 0.10, 0.50, 0.10, 0.10, 0.50])

    zero_oracle_fee = _hypersurge_fee_for_trade(
        reserves,
        candidate_trade=jnp.array([1000.0, -500.0]),
        weights=weights,
        oracle_prices=jnp.array([0.0, 200.0]),
        token_in=0,
        token_out=1,
        base_fee=0.003,
        hypersurge_params=hypersurge_params,
    )
    nan_oracle_fee = _hypersurge_fee_for_trade(
        reserves,
        candidate_trade=jnp.array([1000.0, -500.0]),
        weights=weights,
        oracle_prices=jnp.array([jnp.nan, 200.0]),
        token_in=0,
        token_out=1,
        base_fee=0.003,
        hypersurge_params=hypersurge_params,
    )

    npt.assert_allclose(zero_oracle_fee, 0.003, rtol=1e-12)
    npt.assert_allclose(nan_oracle_fee, 0.003, rtol=1e-12)
