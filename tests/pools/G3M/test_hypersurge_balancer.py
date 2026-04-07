import numpy as np
import numpy.testing as npt

import jax.numpy as jnp

from quantammsim.core_simulator.dynamic_inputs import (
    DynamicInputArrays,
    empty_dynamic_input_arrays,
)
from quantammsim.pools.G3M.balancer.hypersurge_balancer import (
    HYPERSURGE_PARAM_KEYS,
    HyperSurgeBalancerPool,
)
from quantammsim.pools.G3M.balancer.hypersurge_balancer_reserves import (
    _hypersurge_fee_for_trade,
    _pair_deviation,
)
from quantammsim.pools.creator import create_pool
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
            "do_trades": False,
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


def _unbatch_params(params):
    return {
        key: value if key == "subsidary_params" else value[0]
        for key, value in params.items()
    }


def test_creator_registers_hypersurge_balancer_aliases():
    assert isinstance(create_pool("balancer_hypersurge"), HyperSurgeBalancerPool)
    assert isinstance(create_pool("hypersurge_balancer"), HyperSurgeBalancerPool)


def test_hypersurge_params_are_trainable_by_default():
    pool = create_pool("balancer_hypersurge")
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
    for key in HYPERSURGE_PARAM_KEYS:
        assert key in params
        assert params[key].shape == (3, 1)
    assert "initial_weights_logits" in params


def test_pair_deviation_is_zero_when_pool_matches_oracle():
    reserves = jnp.array([5000.0, 2500.0])
    weights = jnp.array([0.5, 0.5])
    oracle_prices = jnp.array([100.0, 200.0])

    deviation = _pair_deviation(
        reserves,
        weights,
        oracle_prices,
        token_in=0,
        token_out=1,
    )

    npt.assert_allclose(np.asarray(deviation), 0.0, atol=1e-12)


def test_fee_uses_noise_params_when_trade_worsens_deviation():
    reserves = jnp.array([5000.0, 2500.0])
    weights = jnp.array([0.5, 0.5])
    oracle_prices = jnp.array([100.0, 200.0])
    hypersurge_params = jnp.array([0.02, 0.10, 0.50, 0.10, 0.10, 0.50])

    fee = _hypersurge_fee_for_trade(
        reserves,
        candidate_trade=jnp.array([1000.0, -500.0]),
        weights=weights,
        oracle_prices=oracle_prices,
        token_in=0,
        token_out=1,
        base_fee=0.003,
        hypersurge_params=hypersurge_params,
    )

    assert float(fee) > 0.02


def test_fee_uses_arb_params_when_trade_improves_deviation():
    reserves = jnp.array([6000.0, 2000.0])
    weights = jnp.array([0.5, 0.5])
    oracle_prices = jnp.array([100.0, 200.0])
    hypersurge_params = jnp.array([0.02, 0.10, 0.50, 0.10, 0.10, 0.50])

    fee = _hypersurge_fee_for_trade(
        reserves,
        candidate_trade=jnp.array([-1000.0, 1000.0]),
        weights=weights,
        oracle_prices=oracle_prices,
        token_in=1,
        token_out=0,
        base_fee=0.003,
        hypersurge_params=hypersurge_params,
    )

    npt.assert_allclose(np.asarray(fee), 0.02, rtol=1e-12)


def test_hypersurge_balancer_reserve_scan_returns_positive_reserves():
    pool = create_pool("balancer_hypersurge")
    prices = jnp.array(
        [
            [100.0, 200.0],
            [105.0, 200.0],
            [110.0, 200.0],
            [115.0, 200.0],
        ]
    )
    run_fingerprint = _run_fingerprint(n_steps=prices.shape[0])
    params = _unbatch_params(
        pool.init_parameters(
            pool.get_initial_values(run_fingerprint),
            run_fingerprint,
            n_assets=2,
            n_parameter_sets=1,
            noise="gaussian",
        )
    )

    reserves = pool.calculate_reserves_with_fees(
        params,
        run_fingerprint,
        prices,
        jnp.array([0, 0]),
        additional_oracle_input=prices,
    )

    assert reserves.shape == prices.shape
    assert bool(jnp.all(jnp.isfinite(reserves)))
    assert bool(jnp.all(reserves > 0.0))


def test_hypersurge_balancer_dynamic_inputs_accept_oracle_prices():
    pool = create_pool("balancer_hypersurge")
    prices = jnp.array(
        [
            [100.0, 200.0],
            [105.0, 200.0],
            [110.0, 200.0],
            [115.0, 200.0],
        ]
    )
    empty_inputs = empty_dynamic_input_arrays()
    dynamic_inputs = DynamicInputArrays(
        trades=None,
        fees=jnp.full((prices.shape[0],), 0.003),
        gas_cost=jnp.zeros((prices.shape[0],)),
        arb_fees=jnp.zeros((prices.shape[0],)),
        lp_supply=jnp.ones((prices.shape[0],)),
        reclamm_price_ratio_updates=empty_inputs.reclamm_price_ratio_updates,
        oracle_prices=prices,
    )
    run_fingerprint = _run_fingerprint(n_steps=prices.shape[0])
    run_fingerprint = NestedHashabledict(
        {
            **run_fingerprint,
            "dynamic_input_flags": {
                "use_dynamic_inputs": True,
                "has_trades": False,
                "has_dynamic_fees": True,
                "has_dynamic_gas_cost": True,
                "has_dynamic_arb_fees": True,
                "has_lp_supply": True,
                "has_reclamm_price_ratio_updates": False,
                "has_oracle_prices": True,
            },
        }
    )

    reserves = pool.calculate_reserves_with_dynamic_inputs(
        {"initial_weights": jnp.array([0.5, 0.5])},
        run_fingerprint,
        prices,
        jnp.array([0, 0]),
        dynamic_inputs,
    )

    assert reserves.shape == prices.shape
    assert bool(jnp.all(jnp.isfinite(reserves)))
    assert bool(jnp.all(reserves > 0.0))
