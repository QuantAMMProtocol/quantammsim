from quantammsim.runners.jax_runners import do_run_on_historic_data
import jax.numpy as jnp
import pandas as pd

# Print the results
print("=" * 100)
print("Simulation Results:")
print("=" * 100)


run_fingerprint = {
    "startDateString": "2021-02-03 00:00:00",
    "endDateString": "2022-07-22 23:59:00",
    "endTestDateString": "2022-07-24 00:00:00",
    "tokens": ["ETH", "DAI"],
    "rule": "balancer",
    "bout_offset": 14400,
    "initial_weights_logits": jnp.array([-0.69314718, -0.69314718], dtype=jnp.float64),
    "initial_pool_value": 60000000,
    "use_alt_lamb": False,
    "return_val": "final_reserves_value_and_weights",
}

run_fingerprint["fees"] = 0.0
params = {"initial_weights_logits": jnp.array([-0.69314718, -0.69314718])}

gas_df = pd.read_csv("Gas.csv")
gas_df = gas_df.rename(columns={"USD": "trade_gas_cost_usd"})
fees_df = pd.read_csv("fees.csv")
fees_df = fees_df.rename(columns={"bps": "fees"})
fees_df["fees"] = fees_df["fees"] / 10000

run_fingerprint["do_trades"] = False

result_w_gas_and_fees = do_run_on_historic_data(
    run_fingerprint, params, gas_cost_df=gas_df, fees_df=fees_df
)
result_w_gas_only = do_run_on_historic_data(run_fingerprint, params, gas_cost_df=gas_df)

result_w_fees_only = do_run_on_historic_data(run_fingerprint, params, fees_df=fees_df)

print(result_w_gas_and_fees["value"][-1440+1])
print(result_w_gas_only["value"][-1440+1])
print(result_w_fees_only["value"][-1440+1])
print("-"*10)
print(result_w_gas_and_fees["final_value"])
print(result_w_gas_only["final_value"])
print(result_w_fees_only["final_value"])
