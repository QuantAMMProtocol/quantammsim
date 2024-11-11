import cvxpy as cp
import numpy as np
from jax.config import config
config.update("jax_enable_x64", True)
from scipy.special import softmax
from itertools import product
from jax import vmap, jit
import jax.numpy as jnp
from jax.tree_util import Partial
from collections import Counter
from time import time
from functools import partial

import pandas as pd

import multiprocessing
from multiprocessing import set_start_method

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator

sns.set(rc={"text.usetex": True})

# set_start_method("spawn",force=True)

try:
   set_start_method('spawn', force=True)
   # print("spawned")
except RuntimeError:
   pass

def get_signature(trade):
    sig = np.zeros_like(trade)
    sig[trade != 0] = trade[trade != 0] / np.abs(trade[trade != 0])
    return sig


def compare_signatures(sig1, sig2):
    if sum(sig1[sig1 != 0] == sig2[sig1 != 0]) == len(sig1[sig1 != 0]):
        return True
    elif sum(sig1[sig2 != 0] == sig2[sig2 != 0]) == len(sig2[sig2 != 0]):
        return True
    elif sum(sig1 == sig2) != len(sig1):
        # print('SIG NOT MATCH')
        return False
    else:
        return True


def direction_to_sig(trade_direction):
    return get_signature(trade_direction - 0.5)


def sig_to_tokens_to_keep(sig):
    return sig != 0


def sig_to_direction(sig):
    trade_direction = np.zeros_like(sig)
    trade_direction[sig == 1] = 1
    return trade_direction

def sig_to_direction_jnp(sig):
    return jnp.where(sig == 1, 1, 0)


def is_sig_a_swap(sig):
    count = Counter(sig)
    return Counter(sig)[1]==1 and Counter(sig)[-1]==1

def sample_weights(n_samples, n_tokens):
    # need to make initial weights,
    # do this using the Rubin method of differences
    # https://cs.stackexchange.com/questions/3227/uniform-sampling-from-a-simplex
    vals = np.random.uniform(
        size=(
            n_samples,
            n_tokens + 1,
        ))
    vals[:, 0:2] = [0.0, 1.0]
    # [:,0:2] = np.array([0.0, 1.0])
    weights = np.diff(np.sort(vals))
    return weights


def construct_optimal_trade_np(
    local_prices, initial_weights, initial_reserves, fee_gamma, sig=None, numerator=2
):
    n = len(initial_weights)
    current_value = (initial_reserves * local_prices).sum()
    # central_reserves = current_value * dex_weights_local/market_prices

    # get current quoted prices
    quoted_prices = current_value * initial_weights / initial_reserves
    upper_range = quoted_prices / fee_gamma
    lower_range = quoted_prices * fee_gamma
    lqp = local_prices / quoted_prices
    gamma_matrix = lqp[:, np.newaxis] / lqp

    # tokens_to_drop = np.array([False,False,True,True])
    if type(sig) == type(None):
        # tokens_to_drop = ((gamma_matrix.prod(0) < 1/fee_gamma) * (gamma_matrix.prod(0) > fee_gamma) * (gamma_matrix.prod(1) < 1/fee_gamma) * (gamma_matrix.prod(1) > fee_gamma))
        # tokens_to_keep = np.invert(tokens_to_drop)
        # tokens_to_keep = (local_prices < lower_range)|(local_prices > upper_range)
        tokens_to_keep = (gamma_matrix > (numerator/n)* 1.0/fee_gamma).sum(0) + (gamma_matrix <  (numerator/n)* fee_gamma).sum(0) > 0
        # tokens_to_drop = (
        #     (gamma_matrix.prod(0) ** (numerator / n) < 1 / fee_gamma)
        #     * (gamma_matrix.prod(0) ** (numerator / n) > fee_gamma)
        #     * (gamma_matrix.prod(1) ** (numerator / n) < 1 / fee_gamma)
        #     * (gamma_matrix.prod(1) ** (numerator / n) > fee_gamma)
        # )
        # tokens_to_keep = np.invert(tokens_to_drop)
        # tokens_to_keep = np.ones_like(tokens_to_keep, dtype='bool')
        tokens_to_drop = np.invert(tokens_to_keep)
    else:
        tokens_to_keep = sig_to_tokens_to_keep(sig)
        tokens_to_drop = np.invert(tokens_to_keep)
    active_local_prices = local_prices[tokens_to_keep]
    active_initial_reserves = initial_reserves[tokens_to_keep]
    active_initial_weights = initial_weights[tokens_to_keep]
    active_initial_weights = active_initial_weights / sum(active_initial_weights)
    active_current_value = (active_initial_reserves * active_local_prices).sum()
    active_quoted_prices = (
        active_current_value * active_initial_weights / active_initial_reserves
    )
    active_n = n - sum(tokens_to_drop)

    if type(sig) == type(None):
        # trade_direction = (quoted_prices > local_prices).astype('int')
        active_trade_direction = (active_quoted_prices > active_local_prices).astype(
            "int"
        )
    # assert np.array_equal(trade_direction[tokens_to_keep], active_trade_direction)
    else:
        active_trade_direction = sig_to_direction(sig[tokens_to_keep])
    # per_asset_ratio = (
    #     (active_initial_weights * (fee_gamma ** (active_trade_direction)))
    #     / (active_local_prices * active_initial_reserves)
    # ) ** (1.0 - active_initial_weights)
    # # log_per_asset_ratio = (1.0-initial_weights) * (np.log(initial_weights) + trade_direction*np.log(fee_gamma)-np.log(local_prices)-np.log(initial_reserves))
    # all_other_assets_quantities = (
    #     (active_local_prices * active_initial_reserves)
    #     / ((fee_gamma ** (active_trade_direction)) * active_initial_weights)
    # ) ** (active_initial_weights)
    # # log_all_other_assets_quantities = (initial_weights) * (np.log(local_prices)+ np.log(initial_reserves) - trade_direction*np.log(fee_gamma)- np.log(initial_weights))
    # leave_one_out_idx = np.arange(1, active_n) - np.tri(
    #     active_n, active_n - 1, k=-1, dtype=bool
    # )
    # all_other_assets_ratio = np.prod(
    #     all_other_assets_quantities[leave_one_out_idx], axis=-1
    # )
    # # log_all_other_assets_ratio = np.sum(log_all_other_assets_quantities[leave_one_out_idx], axis=-1)

    # # overall_trade = (initial_reserves/((2*trade_direction-1.0)*(fee_gamma**(trade_direction))))*((per_asset_ratio*all_other_assets_ratio)-1.0)
    # active_overall_trade = (
    #     active_initial_reserves / ((fee_gamma ** (active_trade_direction)))
    # ) * ((per_asset_ratio * all_other_assets_ratio) - 1.0)

    per_asset_ratio = ((active_initial_weights * (fee_gamma ** (active_trade_direction)))/ (active_local_prices)) ** (1.0 - active_initial_weights)
    # log_per_asset_ratio = (1.0-initial_weights) * (np.log(initial_weights) + trade_direction*np.log(fee_gamma)-np.log(local_prices)-np.log(initial_reserves))
    all_other_assets_quantities = (
        (active_local_prices)
        / ((fee_gamma ** (active_trade_direction)) * active_initial_weights)
    ) ** (active_initial_weights)
    # log_all_other_assets_quantities = (initial_weights) * (np.log(local_prices)+ np.log(initial_reserves) - trade_direction*np.log(fee_gamma)- np.log(initial_weights))
    leave_one_out_idx = np.arange(1, active_n) - np.tri(
        active_n, active_n - 1, k=-1, dtype=bool
    )
    all_other_assets_ratio = np.prod(
        all_other_assets_quantities[leave_one_out_idx], axis=-1
    )
    active_initial_constant = np.prod(active_initial_reserves ** active_initial_weights)
    active_overall_trade = (1.0 / ((fee_gamma ** (active_trade_direction)))) * (((active_initial_constant)*per_asset_ratio * all_other_assets_ratio) - active_initial_reserves)

    overall_trade = np.zeros(n)
    overall_trade[tokens_to_keep] = active_overall_trade
    overall_trade_direction = np.zeros(n)
    overall_trade_direction[tokens_to_keep] = active_trade_direction
    if type(sig) == type(None):
        sig = overall_trade_direction * 2 - 1
        sig[tokens_to_drop] = 0
    return overall_trade, sig, get_signature(overall_trade)

@partial(jit, static_argnums=(5,))
def construct_optimal_trade_jnp(
    initial_weights, local_prices, initial_reserves, fee_gamma, sig, n):
    current_value = (initial_reserves * local_prices).sum()
    # central_reserves = current_value * dex_weights_local/market_prices

    tokens_to_keep = sig_to_tokens_to_keep(sig)
    tokens_to_drop = jnp.invert(tokens_to_keep)
    active_local_prices = local_prices
    active_initial_reserves = initial_reserves
    active_initial_reserves = jnp.where(tokens_to_drop, 1.0, active_initial_reserves)
    partial_initial_weigts = jnp.where(tokens_to_drop, 0.0, initial_weights)
    active_initial_weights = initial_weights / partial_initial_weigts.sum()
    # active_initial_weights = active_initial_weights / jnp.sum(active_initial_weights)
    active_current_value = (active_initial_reserves * active_local_prices).sum()
    active_quoted_prices = (
        active_current_value * active_initial_weights / active_initial_reserves
    )
    active_n = n

    active_trade_direction = sig_to_direction_jnp(sig)

    per_asset_ratio = ((active_initial_weights * (fee_gamma ** (active_trade_direction)))/ (active_local_prices)) ** (1.0 - active_initial_weights)
    # log_per_asset_ratio = (1.0-initial_weights) * (np.log(initial_weights) + trade_direction*np.log(fee_gamma)-np.log(local_prices)-np.log(initial_reserves))
    all_other_assets_quantities = (
        (active_local_prices)
        / ((fee_gamma ** (active_trade_direction)) * active_initial_weights)
    ) ** (active_initial_weights)

    all_other_assets_quantities = jnp.where(tokens_to_drop, 1.0, all_other_assets_quantities)
    # log_all_other_assets_quantities = (initial_weights) * (np.log(local_prices)+ np.log(initial_reserves) - trade_direction*np.log(fee_gamma)- np.log(initial_weights))
    leave_one_out_idx = jnp.arange(1, active_n) - jnp.tri(
        active_n, active_n - 1, k=-1, dtype=bool
    )
    all_other_assets_ratio = jnp.prod(
        all_other_assets_quantities[leave_one_out_idx], axis=-1
    )
    active_initial_constant = jnp.prod(active_initial_reserves ** active_initial_weights)
    active_overall_trade = (1.0 / ((fee_gamma ** (active_trade_direction)))) * (((active_initial_constant)*per_asset_ratio * all_other_assets_ratio) - active_initial_reserves)

    active_overall_trade = jnp.where(tokens_to_drop, 0.0, active_overall_trade)

    initial_constant = jnp.prod((initial_reserves)**initial_weights)
    return active_overall_trade, jnp.prod((initial_reserves+active_overall_trade* (fee_gamma ** (active_trade_direction)))**initial_weights) - initial_constant



if __name__ == '__main__':
    n_top = 7
    n_range = list(range(3,n_top))
    # Problem data.
    # n = 3
    n_bouts = 1000
    initial_value = 10
    np.random.seed(1)
    fees = 0.05
    fee_gamma = 1.0 - fees
    quad_weight = 0.001
    gamma_range = 0.1
    # initial_weights = softmax(np.random.randn(n))

    initial_weights_list = list()
    prices_list = list()

    cxv_timing = list()
    heuristic_timing = list()
    np_brute_force_loop_timing = list()
    np_brute_force_map_timing = list()
    jnp_brute_force_timing = list()

    for n in n_range:
        initial_weights_list.append(sample_weights(1,n)[0])
        prices_list.append(np.random.rand(n))



    for j in range(len(n_range)):
        initial_weights = initial_weights_list[j]
        prices = prices_list[j]
        n = n_range[j]
        np.random.seed(n)

        all_sig_variations = np.array(list(product([1, 0, -1], repeat=n)))
        all_sig_variations = all_sig_variations[(all_sig_variations != 0).sum(-1) > 1]
        all_sig_variations = all_sig_variations[np.any(all_sig_variations == 1, -1)]
        all_sig_variations = all_sig_variations[np.any(all_sig_variations == -1, -1)]


        initial_reserves = initial_value * initial_weights / prices
        gamma_list = gamma_range * np.random.rand(n_bouts, n) + (1 - 0.5 * gamma_range)

        local_prices_list = [prices * g for g in gamma_list]

        initial_constant = np.prod(initial_reserves**initial_weights)

        all_tokens_to_keep_variations = np.array(list(product([True, False], repeat=n)))
        all_tokens_to_keep_variations = all_tokens_to_keep_variations[
            all_tokens_to_keep_variations.sum(-1) > 1
        ]
        pre_trade_constant = np.prod((initial_reserves) ** initial_weights)

        # # for n in range(20):
        # all_sig_variations = np.array(list(product([1, 0, -1], repeat=n)))
        # all_sig_variations = all_sig_variations[(all_sig_variations != 0).sum(-1) > 1]
        # all_sig_variations = all_sig_variations[np.any(all_sig_variations == 1, -1)]
        # all_sig_variations = all_sig_variations[np.any(all_sig_variations == -1, -1)]
        # swap_sigs_idx = [is_sig_a_swap(sig) for sig in all_sig_variations]
        # swap_sigs = all_sig_variations[swap_sigs_idx]
        # print(n, len(all_sig_variations), len(swap_sigs))

        # DO CONVEX ITERATIONS AND TIME
        start_of_cvx_time = time()
        for i in range(len(gamma_list)):
            gamma = gamma_list[i]
            # [0.999, 0.997, 0.97, 0.95, 0.94, 0.93,0.9,0.8]:
            # for gamma in [0.945]:
            # final_weights = softmax(np.random.randn(n))
            # Construct the problem.
            local_prices = prices * gamma
            # local_prices[0]=local_prices[0]*gamma
            # local_prices[1]=local_prices[1]*gamma
            reserves = cp.Variable(n)
            to_give = cp.Variable((n), pos=True)
            to_receive = cp.Variable((n), pos=True)
            # objective = cp.Maximize(reserves @ prices)
            objective = cp.Maximize(
                cp.sum(cp.multiply((to_receive - to_give), local_prices))
                - quad_weight * (cp.sum_squares(to_give) + cp.sum_squares(to_receive))
            )

            R_plus_coins_to_minus_coins_from = (
                initial_reserves + fee_gamma * to_give - to_receive
            )

            constant = cp.geo_mean(initial_reserves, initial_weights.tolist())

            constraints = [
                (cp.geo_mean(R_plus_coins_to_minus_coins_from, initial_weights.tolist()))
                / initial_constant
                >= (1 + 1e-8),
                to_give >= 0,
                to_receive >= 0,
                to_receive <= initial_reserves,
            ]
            # initial_pool_value == initial_value]

            # constraints = [reserves >= 0, reserves == (reserves @ prices) * final_weights / prices, reserves @ prices == initial_reserves @ prices - fees * (cp.abs((reserves-initial_reserves) @ prices))]
            prob = cp.Problem(objective, constraints)

            # The optimal objective value is returned by `prob.solve()`.
            # try:
            result = prob.solve(verbose=False, solver="ECOS")
        end_of_cvx_time = time()
        print(n, ' token, CXV elapsed time:', - start_of_cvx_time + end_of_cvx_time)
        print(n, ' token, CXV time per bout:', (- start_of_cvx_time + end_of_cvx_time)/n_bouts)
        cxv_timing.append(- start_of_cvx_time + end_of_cvx_time)
        # DO HEURISTIC ITERATIONS AND TIME
        start_of_heuristic_time = time()
        for i in range(len(gamma_list)):
            gamma = gamma_list[i]
            # [0.999, 0.997, 0.97, 0.95, 0.94, 0.93,0.9,0.8]:
            # for gamma in [0.945]:
            # final_weights = softmax(np.random.randn(n))
            # Construct the problem.
            local_prices = prices * gamma
            overall_trade, sig, empirical_sig = construct_optimal_trade_np(
            local_prices, initial_weights, initial_reserves, fee_gamma, numerator=n
            )

        end_of_heuristic_time = time()
        print(n, ' token, HEURISTIC elapsed time:', - start_of_heuristic_time + end_of_heuristic_time)
        print(n, ' token, HEURISTIC time per bout:', (- start_of_heuristic_time + end_of_heuristic_time)/n_bouts)
        heuristic_timing.append(- start_of_heuristic_time + end_of_heuristic_time)

        # DO POOL MAP NP AND TIME
        start_of_pool_time = time()

        local_run_wrapper = partial(construct_optimal_trade_np, initial_weights=initial_weights, initial_reserves=initial_reserves, fee_gamma=fee_gamma, numerator=n)

        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        results = pool.map(local_run_wrapper, local_prices_list)
        end_of_pool_time = time()

        print(n, ' token, BRUTE POOL MAP elapsed time:', - start_of_pool_time + end_of_pool_time)
        print(n, ' token, BRUTE POOL MAP time per bout:', (- start_of_pool_time + end_of_pool_time)/n_bouts)
        np_brute_force_map_timing.append(- start_of_pool_time + end_of_pool_time)
        # DO NP BRUTE FORCE ITERATIONS AND TIME
        start_of_np_time = time()
        for i in range(len(gamma_list)):
            gamma = gamma_list[i]
            # [0.999, 0.997, 0.97, 0.95, 0.94, 0.93,0.9,0.8]:
            # for gamma in [0.945]:
            # final_weights = softmax(np.random.randn(n))
            # Construct the problem.
            local_prices = prices * gamma
            best_profit = 0
            best_trade = None
            best_sig = None
            sig_rez = list()
            for s_sig in all_sig_variations:
                # s_sig = np.array([1,  1, 0, -1])
                (
                    s_sig_overall_trade,
                    s_sig,
                    empirical_s_sig_anal_sig,
                ) = construct_optimal_trade_np(
                    local_prices, initial_weights, initial_reserves, fee_gamma, s_sig
                )
                to_give_anal = np.zeros_like(to_give.value)
                to_receive_anal = np.zeros_like(to_give.value)

                to_give_anal[s_sig_overall_trade > 0] = s_sig_overall_trade[
                    s_sig_overall_trade > 0
                ]
                to_receive_anal[s_sig_overall_trade < 0] = -s_sig_overall_trade[
                    s_sig_overall_trade < 0
                ]
                post_trade_constant_anal = np.prod(
                    (initial_reserves + fee_gamma * to_give_anal - to_receive_anal)
                    ** initial_weights
                )
                s_sig_profit = -np.sum(s_sig_overall_trade * local_prices)
                sig_rez.append(s_sig_overall_trade)
                if s_sig_profit > best_profit and post_trade_constant_anal - initial_constant > -1e-8:
                    best_profit = s_sig_profit
                    best_trade = s_sig_overall_trade
                    best_sig = s_sig

        end_of_np_time = time()
        print(n, ' token, NP BRUTE FORCE elapsed time:', - start_of_np_time + end_of_np_time)
        print(n, ' token, NP BRUTE FORCE time per bout:', (- start_of_np_time + end_of_np_time)/n_bouts)
        np_brute_force_loop_timing.append(- start_of_np_time + end_of_np_time)
        # DO JNP BRUTE FORCE ITERATIONS AND TIME
        construct_optimal_trade_jnp_vmapped = jit(vmap(Partial(construct_optimal_trade_jnp, n=n), in_axes=[None,None,None,None,0]))
        test = construct_optimal_trade_jnp_vmapped(initial_weights, local_prices, initial_reserves, fee_gamma, all_sig_variations)
        start_of_jax_time = time()
        for i in range(len(gamma_list)):
            gamma = gamma_list[i]
            # [0.999, 0.997, 0.97, 0.95, 0.94, 0.93,0.9,0.8]:
            # for gamma in [0.945]:
            # final_weights = softmax(np.random.randn(n))
            # Construct the problem.
            local_prices = prices * gamma
            overall_trades, constant_differences = construct_optimal_trade_jnp_vmapped(
            initial_weights, local_prices, initial_reserves, fee_gamma, all_sig_variations
            )

        end_of_jax_time = time()
        print(n, ' token, JAX BRUTE FORCE elapsed time:', - start_of_jax_time + end_of_jax_time)
        print(n, ' token, JAX BRUTE FORCE time per bout:', (- start_of_jax_time + end_of_jax_time)/n_bouts)
        jnp_brute_force_timing.append(- start_of_jax_time + end_of_jax_time)
    
    # rez=np.array([cxv_timing, heuristic_timing, np_brute_force_loop_timing, np_brute_force_map_timing, jnp_brute_force_timing]).T

    # names = ["CVX", "Heuristic", "NumPy Brute Force", "NumPy Map", "Jax jit+vmap"]
    # cols = ["$$\mathrm{" + name + "}$$" for name in names]
    # df = pd.DataFrame(rez,columns=cols, index=n_range)
    # df_wide = df.pivot(*cols)
    # # df = df.set_index([pd.Index(np.arange(len(df)) / 30)])
    # fig, ax = plt.subplots()
    # sns.lineplot(data=df, ax=ax).set_title(
    #     "$$\mathrm{Time}$$"
    # )
    # plt.legend(prop={"size": 10})
    # # plt.xlabel("$$\mathrm{Time}\,\mathrm{(/months)}$$")
    # # plt.ylabel("$$\mathrm{Value}$$")
    # ax.set_facecolor("#0F0614")
    # rez = list()

    #     # to_give.value = np.zeros(n)
    #     # to_receive.value = np.zeros(n)
    #     # except:
    #     #     to_receive.value=np.zeros(n)
    #     #     to_give.value=np.zeros(n)

    #     # The optimal value for x is stored in `x.value`.
    #     # print(reserves.value)
    #     # The optimal Lagrange multiplier for a constraint is stored in
    #     # `constraint.dual_value`.
    #     # print(constraints[0].dual_value)
    #     # initial_reserves=initial_reserves[:-1]
    #     # local_prices=local_prices[:-1]
    #     # initial_weights=initial_weights[:-1]
    #     # initial_weights = initial_weights/sum(initial_weights)

    #     current_value = (initial_reserves * local_prices).sum()
    #     # central_reserves = current_value * dex_weights_local/market_prices

    #     # get current quoted prices
    #     quoted_prices = current_value * initial_weights / initial_reserves
    #     upper_range = quoted_prices / fee_gamma
    #     lower_range = quoted_prices * fee_gamma
    #     lqp = local_prices / quoted_prices
    #     gamma_matrix = lqp[:, np.newaxis] / lqp

    #     current_value = (initial_reserves * local_prices).sum()
    #     # central_reserves = current_value * dex_weights_local/market_prices

    #     # get current quoted prices
    #     quoted_prices = current_value * initial_weights / initial_reserves

    #     price_change_ratio = local_prices / quoted_prices
    #     price_product_change_ratio = np.prod(price_change_ratio**initial_weights)
    #     reserves_ratios_from_price_change = (
    #         price_product_change_ratio / price_change_ratio
    #     )

    #     post_price_reserves = initial_reserves * reserves_ratios_from_price_change

    #     # check if this is worth the cost to arbs
    #     delta = post_price_reserves - initial_reserves

    #     dd = delta.copy()
    #     dd[dd > 0] = dd[dd > 0] / fee_gamma
    #     pp_r = initial_reserves + dd

    #     cvx_sig = get_signature(to_give.value - to_receive.value)
    #     cvx_trade = -to_receive.value + to_give.value

    #     trade_direction = (quoted_prices > local_prices).astype("int")

    #     outside_no_arb_region = np.any((local_prices < lower_range) | (local_prices > upper_range))
    #     alt_outside_no_arb_region = np.any(
    #         (gamma_matrix < fee_gamma) | (gamma_matrix > 1.0 / fee_gamma)
    #     )
    #     # trade_direction = ((cvx_sig+1)/2).astype('int')
    #     # trade_direction = trade_direction[1:-1]
    #     # local_prices=local_prices[1:-1]
    #     # initial_weights=initial_weights[1:-1]
    #     # initial_weights=initial_weights/sum(initial_weights)
    #     # initial_reserves=initial_reserves[1:-1]
    #     # n=n-2

    #     # if alt_outside_no_arb_region:
    #     overall_trade, sig, empirical_sig = construct_optimal_trade(
    #         initial_weights, local_prices, initial_reserves, fee_gamma
    #     )
    #     # else:
    #     # overall_trade, sig, empirical_sig = np.zeros(n), np.zeros(n), np.zeros(n)

    #     s_sig_results = []
    #     for s_sig in all_sig_variations:
    #         # s_sig = np.array([1,  1, 0, -1])
    #         (
    #             s_sig_overall_trade,
    #             s_sig,
    #             empirical_s_sig_anal_sig,
    #         ) = construct_optimal_trade(
    #             initial_weights, local_prices, initial_reserves, fee_gamma, s_sig
    #         )
    #         to_give_anal = np.zeros_like(to_give.value)
    #         to_receive_anal = np.zeros_like(to_give.value)

    #         to_give_anal[s_sig_overall_trade > 0] = s_sig_overall_trade[
    #             s_sig_overall_trade > 0
    #         ]
    #         to_receive_anal[s_sig_overall_trade < 0] = -s_sig_overall_trade[
    #             s_sig_overall_trade < 0
    #         ]
    #         post_trade_constant_anal = np.prod(
    #             (initial_reserves + fee_gamma * to_give_anal - to_receive_anal)
    #             ** initial_weights
    #         )
    #         constant_difference = post_trade_constant_anal - initial_constant
    #         profit = -sum(s_sig_overall_trade * local_prices)
    #         # print(s_sig, profit, constant_difference)
    #         if profit > 0:
    #         # if profit > 0 and constant_difference > -1e-8:
    #             s_sig_results.append(
    #                 {
    #                     "s_sig": s_sig,
    #                     "overall_trade": s_sig_overall_trade,
    #                     "empirical_sig": empirical_s_sig_anal_sig,
    #                     "constant_difference": constant_difference,
    #                     "sig_agree": compare_signatures(
    #                         empirical_s_sig_anal_sig, s_sig
    #                     ),
    #                     "profit": profit,
    #                 }
    #             )
    #     sig_agree_empirical_anal = compare_signatures(sig, empirical_sig)
    #     # sig_agree_cxv_trade = compare_signatures(cvx_sig, trade_sig)

    #     to_give_anal = np.zeros_like(to_give.value)
    #     to_receive_anal = np.zeros_like(to_give.value)

    #     to_give_anal[overall_trade > 0] = overall_trade[overall_trade > 0]
    #     to_receive_anal[overall_trade < 0] = -overall_trade[overall_trade < 0]
    #     post_trade_constant_cxv = np.prod(
    #         (initial_reserves + fee_gamma * to_give.value - to_receive.value)
    #         ** initial_weights
    #     )
    #     post_trade_constant_anal = np.prod(
    #         (initial_reserves + fee_gamma * to_give_anal - to_receive_anal)
    #         ** initial_weights
    #     )

    #     # overall_trade[-1]=0
    #     to_give_anal = np.zeros_like(to_give.value)
    #     to_receive_anal = np.zeros_like(to_give.value)
    #     to_give_anal[overall_trade > 0] = overall_trade[overall_trade > 0]
    #     to_receive_anal[overall_trade < 0] = -overall_trade[overall_trade < 0]
    #     post_trade_constant_anal_alt = np.prod(
    #         (initial_reserves + fee_gamma * to_give_anal - to_receive_anal)
    #         ** initial_weights
    #     )
    #     if (post_trade_constant_anal- initial_constant) < -1e-10:
    #         overall_trade = np.zeros_like(overall_trade)
    #         post_trade_constant_anal = initial_constant
    #     # if sum((to_receive.value - to_give.value) * local_prices) > 0:
    #     #     raise Exception
    #     if i>10 and sum((to_receive.value - to_give.value) * local_prices)>0:
    #         raise Exception
    #     # measured_gamma = (local_prices[0]/local_prices[-1])/(quoted_prices[0]/quoted_prices[-1])
    #     rez.append(
    #         {
    #             "profit": sum((to_receive.value - to_give.value) * local_prices),
    #             "anal_profit": -sum(overall_trade * local_prices),
    #             "delta": delta,
    #             "post_price_reserves": post_price_reserves,
    #             "to_give": to_give.value,
    #             "to_receive": to_receive.value,
    #             "upper_range": upper_range,
    #             "lower_range": lower_range,
    #             "fee_gamma": fee_gamma,
    #             "gamma": gamma,
    #             "sig": sig,
    #             "empirical_sig": empirical_sig,
    #             "local_prices": local_prices,
    #             "quoted_prices": quoted_prices,
    #             # 'measured_gamma': measured_gamma,
    #             "outside_no_arb_region": outside_no_arb_region,
    #             "alt_outside_no_arb_region": alt_outside_no_arb_region,
    #             "sig_agree_empirical_anal": sig_agree_empirical_anal,
    #             # 'sig_agree_anal_trade': sig_agree_anal_trade,
    #             # 'sig_agree_cxv_trade': sig_agree_cxv_trade,
    #             "post_trade_constant_anal_minus_init": post_trade_constant_anal
    #             - initial_constant,
    #             "post_trade_constant_cxv_minus_init": post_trade_constant_cxv
    #             - initial_constant,
    #             "constant_difference": post_trade_constant_anal
    #             - post_trade_constant_cxv,
    #             "s_sig_results": s_sig_results,
    #             "overall_trade": overall_trade,
    #             "cvx_sig": cvx_sig,
    #             "cvx_trade": cvx_trade,
    #         }
    #     )
    #     # if sig_agree_anal_cxv and sig_agree_anal_trade and sig_agree_cxv_trade:

    # post_trade_constant_cxv = np.array(
    #     [r["post_trade_constant_cxv_minus_init"] for r in rez]
    # )

    # post_trade_constant_anal = np.array(
    #     [r["post_trade_constant_anal_minus_init"] for r in rez]
    # )

    # profit_anal = np.array([r["anal_profit"] for r in rez])
    # profit = np.array([r["profit"] for r in rez])

    # # sig_agree_anal_cxv=np.array([r['sig_agree_anal_cxv'] for r in rez])
    # # sig_agree_cxv_trade=np.array([r['sig_agree_cxv_trade'] for r in rez])
    # sig_agree_empirical_anal = np.array([r["sig_agree_empirical_anal"] for r in rez])

    # alt_outside_no_arb_region = np.array([r["alt_outside_no_arb_region"] for r in rez])
    # outside_no_arb_region = np.array([r["outside_no_arb_region"] for r in rez])

    # # tokens_to_keep = np.array([r['tokens_to_keep'] for r in rez])

    # empirical_sig = np.array([r["empirical_sig"] for r in rez])

    # s_sig_rez = [r["s_sig_results"] for r in rez]
    # s_sig_profit = np.array([np.array([l["profit"] for l in t]) for t in s_sig_rez])
    # s_sig_agree = np.array([np.array([l["sig_agree"] for l in t]) for t in s_sig_rez])
    # s_sig = np.array([np.array([l["s_sig"] for l in t]) for t in s_sig_rez])

    # # s_sig_constant_difference = np.array([np.array([l['constant_difference'] for l in t]) for t in s_sig_rez])[:,:,0]

    # s_best_profit = []
    # s_best_sig = []
    # s_best_profit_swap = []
    # s_best_sig_swap = []
    # for i in range(len(s_sig_profit)):
    #     if len(s_sig_profit[i]) > 0:
    #         s_best_profit.append(s_sig_profit[i].max())
    #         idx = s_sig_profit[i].argmax()
    #         s_best_sig.append(s_sig[i][idx])
    #         swap_sigs_local_idx = [is_sig_a_swap(sig) for sig in s_sig[i]]
    #         if sum(swap_sigs_local_idx)>0:
    #             s_best_profit_swap.append(s_sig_profit[i][swap_sigs_local_idx].max())
    #             idx = s_sig_profit[i][swap_sigs_local_idx].argmax()
    #             s_best_sig_swap.append(s_sig[i][swap_sigs_local_idx][idx])
    #         else:
    #             s_best_profit_swap.append(0)
    #             s_best_sig_swap.append([])
    #     else:
    #         s_best_profit.append(0)
    #         s_best_sig.append([])
    #         s_best_profit_swap.append(0)
    #         s_best_sig_swap.append([])



    # s_best_profit = np.array(s_best_profit)
    # s_best_sig = np.array(s_best_sig)
    # s_best_profit_swap = np.array(s_best_profit_swap)
    # s_best_sig_swap = np.array(s_best_sig_swap)


    # arb_when_alt_says_arbable = sum(
    #     [s_best_profit[i] > 0 and rez[i]["alt_outside_no_arb_region"] for i in range(len(rez))]
    # )
    # arb_when_alt_says_not_arbable = sum(
    #     [s_best_profit[i] > 0 and rez[i]["alt_outside_no_arb_region"] == False for i in range(len(rez))]
    # )
    # no_arb_when_alt_says_arbable = sum(
    #     [s_best_profit[i] < 0 and rez[i]["alt_outside_no_arb_region"] for i in range(len(rez))]
    # )
    # no_arb_when_alt_says_not_arbable = sum(
    #     [s_best_profit[i] < 0 and rez[i]["alt_outside_no_arb_region"] == False for i in range(len(rez))]
    # )

    # arb_when_OG_says_arbable = sum(
    #     [s_best_profit[i] > 0 and rez[i]["outside_no_arb_region"] for i in range(len(rez))]
    # )
    # arb_when_OG_says_not_arbable = sum(
    #     [s_best_profit[i] > 0 and rez[i]["outside_no_arb_region"] == False for i in range(len(rez))]
    # )
    # no_arb_when_OG_says_arbable = sum(
    #     [s_best_profit[i] < 0 and rez[i]["outside_no_arb_region"] for i in range(len(rez))]
    # )
    # no_arb_when_OG_says_not_arbable = sum(
    #     [s_best_profit[i] < 0 and rez[i]["outside_no_arb_region"] == False for i in range(len(rez))]
    # )

    # combined_bool = [
    #     (r["alt_outside_no_arb_region"] and r["outside_no_arb_region"]) for r in rez
    # ]

    # arb_when_combined_says_arbable = sum(
    #     [s_best_profit[i] > 0 and combined_bool[i] for i in range(len(rez))]
    # )
    # arb_when_combined_says_not_arbable = sum(
    #     [s_best_profit[i] > 0 and combined_bool[i] == False for i in range(len(rez))]
    # )
    # no_arb_when_combined_says_arbable = sum(
    #     [s_best_profit[i] < 0 and combined_bool[i] for i in range(len(rez))]
    # )
    # no_arb_when_combined_says_not_arbable = sum(
    #     [s_best_profit[i] < 0 and combined_bool[i] == False for i in range(len(rez))]
    # )

    # print('arb_when_alt_says_arbable: ', arb_when_alt_says_arbable)
    # print('arb_when_alt_says_not_arbable: ', arb_when_alt_says_not_arbable)
    # print('no_arb_when_alt_says_arbable: ', no_arb_when_alt_says_arbable)
    # print('no_arb_when_alt_says_not_arbable: ', no_arb_when_alt_says_not_arbable)

    # print('arb_when_OG_says_arbable: ', arb_when_OG_says_arbable)
    # print('arb_when_OG_says_not_arbable: ', arb_when_OG_says_not_arbable)
    # print('no_arb_when_OG_says_arbable: ', no_arb_when_OG_says_arbable)
    # print('no_arb_when_OG_says_not_arbable: ', no_arb_when_OG_says_not_arbable)

    # print('arb_when_combined_says_arbable: ', arb_when_combined_says_arbable)
    # print('arb_when_combined_says_not_arbable: ', arb_when_combined_says_not_arbable)
    # print('no_arb_when_combined_says_arbable: ', no_arb_when_combined_says_arbable)
    # print('no_arb_when_combined_says_not_arbable: ', no_arb_when_combined_says_not_arbable)


    # num_rez = []
    # for numerator in np.linspace(n-1,n,11):
    #     rez = list()
    #     for i in range(len(gamma_list)):
    #         gamma = gamma_list[i]
    #         # [0.999, 0.997, 0.97, 0.95, 0.94, 0.93,0.9,0.8]:
    #         # for gamma in [0.945]:
    #         # final_weights = softmax(np.random.randn(n))
    #         # Construct the problem.
    #         local_prices = prices.copy()
    #         local_prices = local_prices * gamma
    #         # local_prices[0]=local_prices[0]*gamma
    #         # local_prices[1]=local_prices[1]*gamma
    #         # except:
    #         #     to_receive.value=np.zeros(n)
    #         #     to_give.value=np.zeros(n)
    #         # The optimal value for x is stored in `x.value`.
    #         # print(reserves.value)
    #         # The optimal Lagrange multiplier for a constraint is stored in
    #         # `constraint.dual_value`.
    #         # print(constraints[0].dual_value)
    #         # initial_reserves=initial_reserves[:-1]
    #         # local_prices=local_prices[:-1]
    #         # initial_weights=initial_weights[:-1]
    #         # initial_weights = initial_weights/sum(initial_weights)
    #         current_value = (initial_reserves * local_prices).sum()
    #         # central_reserves = current_value * dex_weights_local/market_prices
    #         # get current quoted prices
    #         quoted_prices = current_value * initial_weights / initial_reserves
    #         upper_range = quoted_prices / fee_gamma
    #         lower_range = quoted_prices * fee_gamma
    #         lqp = local_prices / quoted_prices
    #         gamma_matrix = lqp[:, np.newaxis] / lqp
    #         current_value = (initial_reserves * local_prices).sum()
    #         # central_reserves = current_value * dex_weights_local/market_prices
    #         # get current quoted prices
    #         quoted_prices = current_value * initial_weights / initial_reserves
    #         price_change_ratio = local_prices / quoted_prices
    #         price_product_change_ratio = np.prod(price_change_ratio**initial_weights)
    #         reserves_ratios_from_price_change = (price_product_change_ratio / price_change_ratio)
    #         post_price_reserves = initial_reserves * reserves_ratios_from_price_change
    #         # check if this is worth the cost to arbs
    #         delta = post_price_reserves - initial_reserves
    #         dd = delta.copy()
    #         dd[dd > 0] = dd[dd > 0] / fee_gamma
    #         pp_r = initial_reserves + dd
    #         cvx_sig = get_signature(to_give.value - to_receive.value)
    #         cvx_trade = -to_receive.value + to_give.value
    #         trade_direction = (quoted_prices > local_prices).astype("int")
    #         outside_no_arb_region = np.any((local_prices < lower_range) | (local_prices > upper_range))
    #         alt_outside_no_arb_region = np.any((gamma_matrix < fee_gamma) | (gamma_matrix > 1.0 / fee_gamma))
    #         # trade_direction = ((cvx_sig+1)/2).astype('int')
    #         # trade_direction = trade_direction[1:-1]
    #         # local_prices=local_prices[1:-1]
    #         # initial_weights=initial_weights[1:-1]
    #         # initial_weights=initial_weights/sum(initial_weights)
    #         # initial_reserves=initial_reserves[1:-1]
    #         # n=n-2
    #         # if alt_outside_no_arb_region:
    #         overall_trade, sig, empirical_sig = construct_optimal_trade(initial_weights, local_prices, initial_reserves, fee_gamma, numerator=numerator)
    #         to_give_anal = np.zeros_like(to_give.value)
    #         to_receive_anal = np.zeros_like(to_give.value)
    #         to_give_anal[overall_trade > 0] = overall_trade[overall_trade > 0]
    #         to_receive_anal[overall_trade < 0] = -overall_trade[overall_trade < 0]
    #         post_trade_constant_cxv = np.prod((initial_reserves + fee_gamma * to_give.value - to_receive.value)** initial_weights)
    #         post_trade_constant_anal = np.prod((initial_reserves + fee_gamma * to_give_anal - to_receive_anal)** initial_weights)
    #         if (post_trade_constant_anal- initial_constant) < -1e-10:
    #             overall_trade = np.zeros_like(overall_trade)
    #             post_trade_constant_anal = initial_constant
    #         rez.append(
    #             {
    #                 "profit": sum((to_receive.value - to_give.value) * local_prices),
    #                 "anal_profit": -sum(overall_trade * local_prices),
    #                 "delta": delta,
    #                 "post_price_reserves": post_price_reserves,
    #                 "to_give": to_give.value,
    #                 "to_receive": to_receive.value,
    #                 "upper_range": upper_range,
    #                 "lower_range": lower_range,
    #                 "fee_gamma": fee_gamma,
    #                 "gamma": gamma,
    #                 "sig": sig,
    #                 "empirical_sig": empirical_sig,
    #                 "local_prices": local_prices,
    #                 "quoted_prices": quoted_prices,
    #                 # 'measured_gamma': measured_gamma,
    #                 "outside_no_arb_region": outside_no_arb_region,
    #                 "alt_outside_no_arb_region": alt_outside_no_arb_region,
    #                 "sig_agree_empirical_anal": sig_agree_empirical_anal,
    #                 # 'sig_agree_anal_trade': sig_agree_anal_trade,
    #                 # 'sig_agree_cxv_trade': sig_agree_cxv_trade,
    #                 "post_trade_constant_anal_minus_init": post_trade_constant_anal
    #                 - initial_constant,
    #                 "post_trade_constant_cxv_minus_init": post_trade_constant_cxv
    #                 - initial_constant,
    #                 "constant_difference": post_trade_constant_anal
    #                 - post_trade_constant_cxv,
    #                 "s_sig_results": s_sig_results,
    #                 "overall_trade": overall_trade,
    #                 "cvx_sig": cvx_sig,
    #                 "cvx_trade": cvx_trade,
    #             })
    #     profit_anal = np.array([r["anal_profit"] for r in rez])
    #     num_rez.append({"rez": rez,"numerator": numerator,"profit_anal": profit_anal,"sum_profit_diff_cvx": sum((profit-profit_anal)**2.0),"sum_profit_diff_s_sig": sum((s_best_profit-profit_anal)**2.0),})
    #     print(num_rez[-1]['numerator'],num_rez[-1]['sum_profit_diff_cvx'],num_rez[-1]['sum_profit_diff_s_sig'])


    # # current_value = (initial_reserves*local_prices).sum()
    # # # central_reserves = current_value * dex_weights_local/market_prices

    # # # get current quoted prices
    # # quoted_prices = current_value * initial_weights/initial_reserves


    # # trade_direction = (quoted_prices > local_prices).astype('int')

    # # per_asset_ratio = ((initial_weights*(fee_gamma**(trade_direction)))/(local_prices*initial_reserves))**(1.0-initial_weights)
    # # all_other_assets_quantities = ((local_prices*initial_reserves)/((fee_gamma**(trade_direction))*initial_weights))**(initial_weights)
    # # leave_one_out_idx = np.arange(1, n) - np.tri(n, n-1, k=-1, dtype=bool)
    # # all_other_assets_ratio = np.prod(all_other_assets_quantities[leave_one_out_idx], axis=-1)
    # # overall_trade = (initial_reserves/(fee_gamma**(trade_direction)))*((per_asset_ratio*all_other_assets_ratio)-1.0)


    # # initial_weights=initial_weights[0:2]
    # # initial_weights=initial_weights/sum(initial_weights)

    # # initial_reserves = initial_reserves[0:2]

    # # local_prices=local_prices[0:2]

    # # local_prices=np.array([13.0,2.0])
    # # n=2

    # # reserves = cp.Variable(n)
    # # to_give = cp.Variable((n), pos=True)
    # # to_receive = cp.Variable((n), pos=True)
    # # # objective = cp.Maximize(reserves @ prices)
    # # objective = cp.Maximize(cp.sum(cp.multiply((to_receive - to_give), local_prices)) - quad_weight * (cp.sum_squares(to_give) + cp.sum_squares(to_receive)))

    # # R_plus_coins_to_minus_coins_from = initial_reserves + fee_gamma * to_give - to_receive

    # # constant = cp.geo_mean(initial_reserves, initial_weights.tolist())

    # # constraints = [cp.geo_mean(R_plus_coins_to_minus_coins_from, initial_weights.tolist())>= constant,
    # #                to_give >= 0,
    # #                to_receive >= 0,
    # #                to_give <= initial_reserves]
    # #                # initial_pool_value == initial_value]

    # # # constraints = [reserves >= 0, reserves == (reserves @ prices) * final_weights / prices, reserves @ prices == initial_reserves @ prices - fees * (cp.abs((reserves-initial_reserves) @ prices))]
    # # prob = cp.Problem(objective, constraints)

    # # # The optimal objective value is returned by `prob.solve()`.
    # # result = prob.solve()

    # # two_token_out = initial_reserves[1]*(1.0-((local_prices[0]*initial_reserves[0]*initial_weights[1])/(fee_gamma*local_prices[1]*initial_reserves[1]*initial_weights[0]))**(1.0/(1.0+(initial_weights[1]/initial_weights[0]))))
    # # two_token_in = (initial_reserves[0]/fee_gamma)*(((local_prices[0]*initial_reserves[0]*initial_weights[1])/(fee_gamma*local_prices[1]*initial_reserves[1]*initial_weights[0]))**(-(initial_weights[1]/initial_weights[0])/(1.0+(initial_weights[1]/initial_weights[0])))-1.0)

    # # two_token_out_nofees = initial_reserves[1]*(1.0-((local_prices[0]*initial_reserves[0]*initial_weights[1])/(local_prices[1]*initial_reserves[1]*initial_weights[0]))**(1.0/(1.0+(initial_weights[1]/initial_weights[0]))))
    # # two_token_in_nofees = (initial_reserves[0])*(((local_prices[0]*initial_reserves[0]*initial_weights[1])/(local_prices[1]*initial_reserves[1]*initial_weights[0]))**(-(initial_weights[1]/initial_weights[0])/(1.0+(initial_weights[1]/initial_weights[0])))-1.0)
    # # post_trade_constant_cxv=np.array([r['post_trade_constant_cxv_minus_init'] for r in rez])

    # # post_trade_constant_anal=np.array([r['post_trade_constant_anal_minus_init'] for r in rez])

    # # profit_anal=np.array([r['anal_profit'] for r in rez])
    # # profit=np.array([r['profit'] for r in rez])

    # # # sig_agree_anal_cxv=np.array([r['sig_agree_anal_cxv'] for r in rez])
    # # # sig_agree_cxv_trade=np.array([r['sig_agree_cxv_trade'] for r in rez])
    # # sig_agree_empirical_anal=np.array([r['sig_agree_empirical_anal'] for r in rez])

    # # alt_outside_no_arb_region=np.array([r['alt_outside_no_arb_region'] for r in rez])
    # # outside_no_arb_region=np.array([r['outside_no_arb_region'] for r in rez])

    # # # tokens_to_keep = np.array([r['tokens_to_keep'] for r in rez])

    # # empirical_sig = np.array([r['empirical_sig'] for r in rez])

    # # s_sig_rez = [r['s_sig_results'] for r in rez]
    # # s_sig_profit = np.array([np.array([l['profit'] for l in t]) for t in s_sig_rez])
    # # s_sig_agree = np.array([np.array([l['sig_agree'] for l in t]) for t in s_sig_rez])
    # # s_sig = np.array([np.array([l['s_sig'] for l in t]) for t in s_sig_rez])

    # # # s_sig_constant_difference = np.array([np.array([l['constant_difference'] for l in t]) for t in s_sig_rez])[:,:,0]

    # # s_best_profit = []
    # # s_best_sig = []
    # # for i in range(len(s_sig_profit)):
    # #     if len(s_sig_profit[i]) > 0:
    # #         s_best_profit.append(s_sig_profit[i].max())
    # #         idx = s_sig_profit[i].argmax()
    # #         s_best_sig.append(s_sig[i][idx])
    # #     else:
    # #         s_best_profit.append(0)
    # #         s_best_sig.append([])

    # # s_best_profit = np.array(s_best_profit)
    # # s_best_sig = np.array(s_best_sig)

    # # s_sig_constant_difference[(np.arange(len(s_sig_constant_difference)),best_trade_per_bout)]


    # (local_prices*initial_reserves*(1+s_sig_results[0]['overall_trade']*(fee_gamma**sig_to_direction(s_sig_results[0]['s_sig']))**initial_weights))/(initial_weights * (fee_gamma**sig_to_direction(s_sig_results[0]['s_sig'])))
    # # filter_for_s_runs_with_constant
















