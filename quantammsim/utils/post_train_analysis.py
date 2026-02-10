"""Performance metric calculations for SGD analysis.

Includes:
- Period metric calculations (Sharpe, Calmar, etc.)
- Deflated Sharpe Ratio (Bailey & López de Prado, 2014)
- Block bootstrap confidence intervals for Sharpe
- Impermanent loss decomposition
"""

import numpy as np
import jax.numpy as jnp
from typing import Dict, List, Tuple, Optional
from scipy import stats
from quantammsim.core_simulator.forward_pass import _calculate_return_value


def calculate_period_metrics(results_dict, prices=None):
    """Calculate comprehensive performance metrics for a simulation period.

    Computes Sharpe ratios (minute-resolution, daily arithmetic, daily log),
    return metrics (absolute, vs HODL, vs uniform HODL, annualised variants),
    drawdown metrics (Calmar, Sterling), and the Ulcer Index.

    Parameters
    ----------
    results_dict : dict
        Simulation output containing:

        - ``"reserves"`` : array of shape ``(T, n_assets)``
        - ``"value"`` : array of shape ``(T,)``
        - ``"prices"`` : array of shape ``(T, n_assets)``, optional if
          ``prices`` kwarg is provided

    prices : array-like, optional
        Price data of shape ``(T, n_assets)``.  Overrides
        ``results_dict["prices"]`` when provided.

    Returns
    -------
    dict
        Metric dictionary with keys:

        - ``"sharpe"`` : daily arithmetic-return Sharpe (annualised)
        - ``"jax_sharpe"`` : minute-resolution Sharpe from forward pass
        - ``"daily_log_sharpe"`` : daily log-return Sharpe (annualised)
        - ``"return"`` : total cumulative return
        - ``"returns_over_hodl"`` : return relative to initial-reserve HODL
        - ``"returns_over_uniform_hodl"`` : return relative to equal-value HODL
        - ``"annualised_returns"`` : annualised total return
        - ``"annualised_returns_over_hodl"`` : annualised return vs HODL
        - ``"annualised_returns_over_uniform_hodl"`` : annualised return vs uniform HODL
        - ``"ulcer"`` : negated Ulcer Index (higher = less pain)
        - ``"calmar"`` : Calmar ratio (return / max drawdown)
        - ``"sterling"`` : Sterling ratio (return / avg drawdown)
        - ``"daily_returns"`` : ``numpy.ndarray`` of daily arithmetic returns
          (used downstream for bootstrap CIs and DSR)
    """
    # Use provided prices if available, otherwise get from results_dict
    price_data = prices if prices is not None else results_dict["prices"]

    sharpe = _calculate_return_value(
        "sharpe",
        results_dict["reserves"],
        price_data,
        results_dict["value"],
    )
    returns = _calculate_return_value(
        "returns",
        results_dict["reserves"],
        price_data,
        results_dict["value"],
    )
    returns_over_hodl = _calculate_return_value(
        "returns_over_hodl",
        results_dict["reserves"],
        price_data,
        results_dict["value"],
        initial_reserves=results_dict["reserves"][0],
    )
    returns_over_uniform_hodl = _calculate_return_value(
        "returns_over_uniform_hodl",
        results_dict["reserves"],
        price_data,
        results_dict["value"],
        initial_reserves=results_dict["reserves"][0],
    )
    annualised_returns = _calculate_return_value(
        "annualised_returns",
        results_dict["reserves"],
        price_data,
        results_dict["value"],
    )
    annualised_returns_over_hodl = _calculate_return_value(
        "annualised_returns_over_hodl",
        results_dict["reserves"],
        price_data,
        results_dict["value"],
        initial_reserves=results_dict["reserves"][0],
    )
    annualised_returns_over_uniform_hodl = _calculate_return_value(
        "annualised_returns_over_uniform_hodl",
        results_dict["reserves"],
        price_data,
        results_dict["value"],
        initial_reserves=results_dict["reserves"][0],
    )
    daily_returns = (
        jnp.diff(results_dict["value"][::24 * 60])
        / results_dict["value"][::24 * 60][:-1]
    )
    daily_std = daily_returns.std()
    daily_sharpe = jnp.where(
        daily_std > 1e-12,
        jnp.sqrt(365) * daily_returns.mean() / daily_std,
        0.0,
    )

    daily_log_sharpe = _calculate_return_value(
        "daily_log_sharpe",
        results_dict["reserves"],
        price_data,
        results_dict["value"],
    )

    ulcer_index = _calculate_return_value(
        "ulcer",
        results_dict["reserves"],
        price_data,
        results_dict["value"],
        initial_reserves=results_dict["reserves"][0],
    )
    calmar_ratio = _calculate_return_value(
        "calmar",
        results_dict["reserves"],
        price_data,
        results_dict["value"],
        initial_reserves=results_dict["reserves"][0],
    )
    sterling_ratio = _calculate_return_value(
        "sterling",
        results_dict["reserves"],
        price_data,
        results_dict["value"],
        initial_reserves=results_dict["reserves"][0],
    )
    return {
        "sharpe": float(daily_sharpe),
        "jax_sharpe": float(sharpe),
        "daily_log_sharpe": float(daily_log_sharpe),
        "return": float(returns),
        "returns_over_hodl": float(returns_over_hodl),
        "returns_over_uniform_hodl": float(returns_over_uniform_hodl),
        "annualised_returns": float(annualised_returns),
        "annualised_returns_over_hodl": float(annualised_returns_over_hodl),
        "annualised_returns_over_uniform_hodl": float(annualised_returns_over_uniform_hodl),
        "ulcer": float(ulcer_index),
        "calmar": float(calmar_ratio),
        "sterling": float(sterling_ratio),
        "daily_returns": np.asarray(daily_returns, dtype=np.float64),
    }

def calculate_continuous_test_metrics(continuous_results, train_len, test_len, prices):
    """Calculate metrics for the test portion of a continuous simulation.

    Slices the test period from a train+test forward pass and delegates
    to :func:`calculate_period_metrics`.  The continuous forward pass
    avoids pool re-initialisation at the train/test boundary.

    Parameters
    ----------
    continuous_results : dict
        Output from a forward pass spanning train + test, with keys
        ``"value"`` and ``"reserves"``.
    train_len : int
        Number of timesteps in the training period (used as slice offset).
    test_len : int
        Number of timesteps in the test period.
    prices : array-like
        Price data covering the full train + test window.

    Returns
    -------
    dict
        Same keys as :func:`calculate_period_metrics`, computed on the
        test slice only.
    """
    # Extract test period portion

    price_data = prices if prices is not None else continuous_results["prices"]
    continuous_test_results = {
        "value": continuous_results["value"][train_len : train_len + test_len],
        "reserves": continuous_results["reserves"][train_len : train_len + test_len],
        "prices": price_data[train_len : train_len + test_len],
    }

    metrics = calculate_period_metrics(continuous_test_results)
    return metrics


def process_continuous_outputs(
    continuous_outputs,
    data_dict,
    n_parameter_sets,
    use_ensemble_mode=False,
):
    """
    Process continuous forward pass outputs into per-parameter-set metrics.

    This function handles both standard mode (outputs batched over param sets)
    and ensemble mode (single output from averaged rule outputs).

    Parameters
    ----------
    continuous_outputs : dict
        Output from continuous forward pass containing:
        - "value": Pool value over time
        - "reserves": Pool reserves over time
        Shape depends on mode:
        - Standard: (n_parameter_sets, time_steps) / (n_parameter_sets, time_steps, n_assets)
        - Ensemble: (time_steps,) / (time_steps, n_assets)
    data_dict : dict
        Data dictionary containing:
        - "start_idx": Start index for the simulation
        - "bout_length": Length of training period
        - "bout_length_test": Length of test period
        - "prices": Price data array
    n_parameter_sets : int
        Number of parameter sets (used for standard mode iteration)
    use_ensemble_mode : bool, optional
        If True, outputs are unbatched (ensemble averaging was used).
        Default is False.

    Returns
    -------
    tuple of (list, list, list)
        (train_metrics_list, test_metrics_list, continuous_test_metrics_list)
        Each list contains one dict per parameter set (or one dict total for ensemble mode).
    """
    train_metrics_list = []
    test_metrics_list = []
    continuous_test_metrics_list = []

    start_idx = data_dict["start_idx"]
    bout_length = data_dict["bout_length"]
    bout_length_test = data_dict["bout_length_test"]
    prices = data_dict["prices"]

    if use_ensemble_mode:
        # Ensemble mode: single output (not batched over param sets)
        param_value = continuous_outputs["value"]
        param_reserves = continuous_outputs["reserves"]

        # Slice train period
        train_dict = {
            "value": param_value[:bout_length],
            "reserves": param_reserves[:bout_length],
        }
        train_prices = prices[start_idx : start_idx + bout_length]

        # Slice test period
        test_dict = {
            "value": param_value[bout_length:],
            "reserves": param_reserves[bout_length:],
        }
        test_prices = prices[
            start_idx + bout_length : start_idx + bout_length + bout_length_test
        ]

        # Continuous dict for test metrics
        param_continuous_dict = {
            "value": param_value,
            "reserves": param_reserves,
        }
        continuous_prices = prices[
            start_idx : start_idx + bout_length + bout_length_test
        ]

        # Calculate metrics
        train_metrics = calculate_period_metrics(train_dict, train_prices)
        test_metrics = calculate_period_metrics(test_dict, test_prices)
        continuous_test_metrics = calculate_continuous_test_metrics(
            param_continuous_dict, bout_length, bout_length_test, continuous_prices
        )

        train_metrics_list.append(train_metrics)
        test_metrics_list.append(test_metrics)
        continuous_test_metrics_list.append(continuous_test_metrics)

    else:
        # Standard mode: outputs batched over param sets
        for param_idx in range(n_parameter_sets):
            # Extract outputs for this parameter set
            param_value = continuous_outputs["value"][param_idx]
            param_reserves = continuous_outputs["reserves"][param_idx]

            # Slice train period
            train_dict = {
                "value": param_value[:bout_length],
                "reserves": param_reserves[:bout_length],
            }
            train_prices = prices[start_idx : start_idx + bout_length]

            # Slice test period
            test_dict = {
                "value": param_value[bout_length:],
                "reserves": param_reserves[bout_length:],
            }
            test_prices = prices[
                start_idx + bout_length : start_idx + bout_length + bout_length_test
            ]

            # Continuous dict for test metrics
            param_continuous_dict = {
                "value": param_value,
                "reserves": param_reserves,
            }
            continuous_prices = prices[
                start_idx : start_idx + bout_length + bout_length_test
            ]

            # Calculate metrics
            train_metrics = calculate_period_metrics(train_dict, train_prices)
            test_metrics = calculate_period_metrics(test_dict, test_prices)
            continuous_test_metrics = calculate_continuous_test_metrics(
                param_continuous_dict, bout_length, bout_length_test, continuous_prices
            )

            train_metrics_list.append(train_metrics)
            test_metrics_list.append(test_metrics)
            continuous_test_metrics_list.append(continuous_test_metrics)

    return train_metrics_list, test_metrics_list, continuous_test_metrics_list


# =============================================================================
# Deflated Sharpe Ratio (Bailey & López de Prado, 2014)
# =============================================================================

def _expected_max_sr(n_trials: int, T: int, skew: float = 0.0, kurt: float = 3.0) -> float:
    """Expected maximum Sharpe ratio under the null (all strategies are noise).

    Eq. (6) from Bailey & López de Prado (2014):
      E[max(SR)] ≈ sqrt(V(SR)) * ((1 - γ)*Φ^{-1}(1-1/N) + γ*Φ^{-1}(1-1/(N*e)))
    where γ ≈ 0.5772 (Euler-Mascheroni), V(SR) is the variance of the SR estimator.

    Parameters
    ----------
    n_trials : int
        Number of independent strategies tested (Optuna trials).
    T : int
        Number of return observations per strategy.
    skew : float
        Skewness of returns (0 for normal).
    kurt : float
        Kurtosis of returns (3 for normal).
    """
    if n_trials <= 0 or T <= 1:
        return 0.0

    # Variance of SR estimator (Lo, 2002, corrected for non-normality)
    # V(SR) ≈ (1 + 0.25*skew*SR - (kurt-3)/4*SR^2) / T
    # Under null (SR=0): V(SR) ≈ 1/T
    var_sr = (1.0 + 0.25 * skew * 0 - (kurt - 3.0) / 4.0 * 0) / T
    std_sr = np.sqrt(max(var_sr, 1e-12))

    euler_mascheroni = 0.5772156649
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials) if n_trials > 1 else 0.0
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e)) if n_trials > 1 else 0.0

    return std_sr * ((1 - euler_mascheroni) * z1 + euler_mascheroni * z2)


def deflated_sharpe_ratio(
    observed_sr: float,
    n_trials: int,
    T: int,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> Dict[str, float]:
    """Compute the Deflated Sharpe Ratio (DSR).

    Tests whether the observed Sharpe ratio exceeds the expected maximum SR
    under the null hypothesis that all tested strategies are noise.

    Parameters
    ----------
    observed_sr : float
        Best observed OOS Sharpe ratio (annualised).
    n_trials : int
        Number of independent strategies/hyperparameter sets tested.
    T : int
        Number of OOS return observations.
    skew : float
        Skewness of OOS returns.
    kurt : float
        Excess kurtosis of OOS returns (0 for normal, not 3).

    Returns
    -------
    dict
        sr0: expected max SR under null
        dsr: probability that observed SR exceeds sr0
        significant: True if DSR >= 0.95
    """
    sr0 = _expected_max_sr(n_trials, T, skew, kurt + 3.0)  # convert excess to raw

    # Variance of SR estimator
    var_sr = (1.0 + 0.25 * skew * observed_sr - (kurt) / 4.0 * observed_sr ** 2) / T
    std_sr = np.sqrt(max(var_sr, 1e-12))

    # DSR = P(SR > SR0) = Φ((SR - SR0) / std(SR))
    if std_sr > 0:
        dsr = float(stats.norm.cdf((observed_sr - sr0) / std_sr))
    else:
        dsr = 1.0 if observed_sr > sr0 else 0.0

    return {
        "sr0": float(sr0),
        "dsr": dsr,
        "significant": dsr >= 0.95,
        "observed_sr": float(observed_sr),
        "n_trials": n_trials,
        "T": T,
    }


# =============================================================================
# Block Bootstrap Confidence Intervals for Sharpe Ratio
# =============================================================================

def block_bootstrap_sharpe_ci(
    daily_returns: np.ndarray,
    n_bootstrap: int = 10000,
    block_length: int = 10,
    confidence: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Block bootstrap confidence intervals for annualised Sharpe ratio.

    Uses circular block bootstrap to preserve autocorrelation structure
    in the return series.

    Parameters
    ----------
    daily_returns : array-like
        Daily (or per-period) return series.
    n_bootstrap : int
        Number of bootstrap samples.
    block_length : int
        Block length for block bootstrap. Should be long enough to capture
        autocorrelation (10-20 for daily crypto returns).
    confidence : float
        Confidence level (e.g., 0.95 for 95% CI).
    seed : int
        Random seed.

    Returns
    -------
    dict
        point_estimate, lower, upper, std, confidence_level
    """
    returns = np.asarray(daily_returns, dtype=np.float64)
    T = len(returns)

    if T < 2 * block_length:
        return {
            "point_estimate": float("nan"),
            "lower": float("nan"),
            "upper": float("nan"),
            "std": float("nan"),
            "confidence_level": confidence,
            "warning": f"Too few observations ({T}) for block length {block_length}",
        }

    rng = np.random.RandomState(seed)
    n_blocks = int(np.ceil(T / block_length))

    bootstrap_sharpes = np.empty(n_bootstrap)
    annualisation = np.sqrt(365.0)

    for i in range(n_bootstrap):
        # Circular block bootstrap: sample random start indices
        starts = rng.randint(0, T, size=n_blocks)
        indices = np.concatenate([
            np.arange(s, s + block_length) % T for s in starts
        ])[:T]

        sample = returns[indices]
        std = sample.std()
        if std > 1e-12:
            bootstrap_sharpes[i] = annualisation * sample.mean() / std
        else:
            bootstrap_sharpes[i] = 0.0

    alpha = 1 - confidence
    lower = float(np.percentile(bootstrap_sharpes, 100 * alpha / 2))
    upper = float(np.percentile(bootstrap_sharpes, 100 * (1 - alpha / 2)))
    full_std = returns.std()
    point = annualisation * returns.mean() / full_std if full_std > 1e-12 else 0.0

    return {
        "point_estimate": float(point),
        "lower": lower,
        "upper": upper,
        "std": float(np.std(bootstrap_sharpes)),
        "confidence_level": confidence,
    }


# =============================================================================
# Impermanent Loss Decomposition
# =============================================================================

def decompose_pool_returns(
    values: np.ndarray,
    reserves: np.ndarray,
    prices: np.ndarray,
    weights: Optional[np.ndarray] = None,
    fees_earned: float = 0.0,
) -> Dict[str, float]:
    """Decompose pool returns into components.

    Components:
    - hodl_return: What the initial portfolio would be worth if weights never changed
    - pool_return: Actual pool return (value[-1] / value[0] - 1)
    - divergence_loss: Value lost due to arb-driven rebalancing (always <= 0 for G3M)
    - fee_income: Value gained from swap fees (external input, not computed from sim)
    - strategy_alpha: Residual — return from dynamic weight changes beyond passive AMM

    The decomposition: pool_return = hodl_return + divergence_loss + fee_income + strategy_alpha

    Parameters
    ----------
    values : array-like
        Pool value over time, shape (T,).
    reserves : array-like
        Pool reserves over time, shape (T, n_assets).
    prices : array-like
        Asset prices over time, shape (T, n_assets).
    weights : array-like, optional
        Pool weights over time, shape (T, n_assets). If None, inferred from reserves * prices / value.
    fees_earned : float
        Total fees earned by pool (as fraction of initial value). Default 0.0.

    Returns
    -------
    dict
        hodl_return, pool_return, divergence_loss, fee_income, strategy_alpha,
        and diagnostic: hodl_value_final, pool_value_final.
    """
    values = np.asarray(values, dtype=np.float64)
    reserves = np.asarray(reserves, dtype=np.float64)
    prices = np.asarray(prices, dtype=np.float64)

    T = len(values)
    n_assets = reserves.shape[1] if reserves.ndim > 1 else 1

    # Pool return
    pool_return = values[-1] / values[0] - 1.0

    # HODL return: hold initial reserves, revalue at final prices
    initial_reserves = reserves[0]
    hodl_final_value = float(np.sum(initial_reserves * prices[-1]))
    hodl_return = hodl_final_value / values[0] - 1.0

    # Constant-weight AMM return (passive rebalancing to initial weights)
    # For a constant-weight AMM, value = C * prod(price_i ^ w_i)
    # This is the "passive AMM" benchmark — what you'd get with fixed weights
    if weights is not None:
        initial_weights = np.asarray(weights[0], dtype=np.float64)
    else:
        initial_weights = (initial_reserves * prices[0]) / values[0]

    # Avoid log(0) for zero-price assets
    safe_prices_0 = np.maximum(prices[0], 1e-18)
    safe_prices_T = np.maximum(prices[-1], 1e-18)
    price_ratios = safe_prices_T / safe_prices_0

    # Constant-weight AMM value: V_T = V_0 * prod(price_ratio_i ^ w_i)
    cw_amm_return = float(np.prod(price_ratios ** initial_weights)) - 1.0

    # Divergence loss = constant-weight AMM return - HODL return
    # This is always <= 0 (the "cost" of continuous rebalancing)
    divergence_loss = cw_amm_return - hodl_return

    # Strategy alpha = pool return - constant-weight AMM return - fees
    # This is the residual from dynamic weight changes
    strategy_alpha = pool_return - cw_amm_return - fees_earned

    return {
        "pool_return": float(pool_return),
        "hodl_return": float(hodl_return),
        "cw_amm_return": float(cw_amm_return),
        "divergence_loss": float(divergence_loss),
        "fee_income": float(fees_earned),
        "strategy_alpha": float(strategy_alpha),
        # Diagnostics
        "hodl_value_final": hodl_final_value,
        "pool_value_final": float(values[-1]),
        "initial_weights": initial_weights.tolist() if hasattr(initial_weights, 'tolist') else list(initial_weights),
    }
