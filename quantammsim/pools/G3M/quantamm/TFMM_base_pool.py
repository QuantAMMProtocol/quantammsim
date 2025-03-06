# again, this only works on startup!
from jax import config

config.update("jax_enable_x64", True)
from jax import default_backend
from jax import local_device_count, devices

DEFAULT_BACKEND = default_backend()
CPU_DEVICE = devices("cpu")[0]
if DEFAULT_BACKEND != "cpu":
    GPU_DEVICE = devices("gpu")[0]
    config.update("jax_platform_name", "gpu")
else:
    GPU_DEVICE = devices("cpu")[0]
    config.update("jax_platform_name", "cpu")

import jax.numpy as jnp
from jax import jit, vmap
from jax import devices, device_put
from jax.lax import stop_gradient, dynamic_slice

from quantammsim.pools.base_pool import AbstractPool
from quantammsim.pools.G3M.quantamm.quantamm_reserves import (
    _jax_calc_quantAMM_reserve_ratios,
    _jax_calc_quantAMM_reserves_with_fees_using_precalcs,
    _jax_calc_quantAMM_reserves_with_dynamic_inputs,
)
from quantammsim.core_simulator.param_utils import make_vmap_in_axes_dict
from quantammsim.core_simulator.param_utils import memory_days_to_lamb
import numpy as np

from typing import Dict, Any, Optional
from functools import partial
from abc import abstractmethod


class TFMMBasePool(AbstractPool):
    """
    TFMMBasePool is an abstract base class for implementing TFMM (Temporal Function Market Making) liquidity pools.

    This class extends the AbstractPool class and provides a foundation for specific TFMM pool implementations.
    It defines additional abstract methods that are specific to TFMM pools, such as weight calculation.

    Abstract Methods:
        calculate_raw_weights_outputs: Calculate the raw weight outputs of assets in the pool based on oracle values and parameters.
        fine_weight_output: Function to handle how raw weights get mapped to per-block/per-minute weights. Two standard methods
        are provided, for when 1) rules output raw weight _changes_ and 2) when rule output raw _weights_ themselves. See MomentumPool
        and MinVariancePool as prototypical examples of each respectively.

    In addition to the methods from AbstractPool, subclasses of TFMMBasePool must implement these
    TFMM-specific methods to define the behavior of the pool.

    Note:
        This class is designed to be subclassed, not instantiated directly. Concrete implementations
        should provide specific logic for weight calculation and slippage estimation. It is reccomended
        to implement the functions used within implementations of these methods as external JAX functions
        that are jitted and then used within pool methods. This separation of concerns comes from that JAX
        is a functional programming language and we want to keep the pool methods pure. Finally, note that due
        to this separation of concerns this class does not hold any state, for example pool parameters.
    """

    def __init__(self):
        """
        Initialize a new TFMMBasePool instance.
        """
        super().__init__()

    @partial(jit, static_argnums=(2))
    def calculate_reserves_with_fees(
        self,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        prices: jnp.ndarray,
        start_index: jnp.ndarray,
        additional_oracle_input: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Calculate reserves with fees and dynamic weights.

        TFMM pools calculate weights dynamically based on price history.
        This method handles the full complexity of weight adjustments, fees, and arbitrage.

        Implementation Steps:
        ---------------------
        1. Extract local price window
        2. Calculate dynamic weights based on price history
        3. Apply arbitrage frequency adjustments
        4. Initialize reserves based on pool value
        5. Calculate reserve changes using quantAMM precalcs

        Parameters
        ----------
        params : Dict[str, Any]
            Pool parameters including weight calculation parameters
        run_fingerprint : Dict[str, Any]
            Simulation settings including:
            - bout_length: Simulation window length
            - n_assets: Number of tokens
            - arb_frequency: Arbitrage check frequency
            - initial_pool_value: Starting pool value
            - fees: Trading fees
            - gas_cost: Arbitrage threshold
            - arb_fees: Arbitrage fees
            - do_arb: Enable arbitrage
            - all_sig_variations: Valid trade combinations
        prices : jnp.ndarray
            Historical price data
        start_index : jnp.ndarray
            Window start position
        additional_oracle_input : Optional[jnp.ndarray]
            Extra data for weight calculation

        Returns
        -------
        jnp.ndarray
            Time series of pool reserves
        """
        bout_length = run_fingerprint["bout_length"]
        n_assets = run_fingerprint["n_assets"]

        local_prices = dynamic_slice(prices, start_index, (bout_length - 1, n_assets))

        weights = self.calculate_weights(
            params, run_fingerprint, prices, start_index, additional_oracle_input
        )
        if run_fingerprint["arb_frequency"] != 1:
            arb_acted_upon_weights = weights[:: run_fingerprint["arb_frequency"]]
            arb_acted_upon_local_prices = local_prices[
                :: run_fingerprint["arb_frequency"]
            ]
        else:
            arb_acted_upon_weights = weights
            arb_acted_upon_local_prices = local_prices

        initial_pool_value = run_fingerprint["initial_pool_value"]
        initial_value_per_token = arb_acted_upon_weights[0] * initial_pool_value
        initial_reserves = initial_value_per_token / arb_acted_upon_local_prices[0]
        if run_fingerprint["do_arb"]:
            reserves = _jax_calc_quantAMM_reserves_with_fees_using_precalcs(
                initial_reserves,
                arb_acted_upon_weights,
                arb_acted_upon_local_prices,
                fees=run_fingerprint["fees"],
                arb_thresh=run_fingerprint["gas_cost"],
                arb_fees=run_fingerprint["arb_fees"],
                all_sig_variations=jnp.array(run_fingerprint["all_sig_variations"]),
            )
        else:
            reserves = jnp.broadcast_to(
                initial_reserves, arb_acted_upon_local_prices.shape
            )

        return reserves

    @partial(jit, static_argnums=(2))
    def calculate_reserves_zero_fees(
        self,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        prices: jnp.ndarray,
        start_index: jnp.ndarray,
        additional_oracle_input: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:

        bout_length = run_fingerprint["bout_length"]
        n_assets = run_fingerprint["n_assets"]

        local_prices = dynamic_slice(prices, start_index, (bout_length - 1, n_assets))

        weights = self.calculate_weights(
            params, run_fingerprint, prices, start_index, additional_oracle_input
        )

        # calculate initial reserves
        initial_pool_value = run_fingerprint["initial_pool_value"]
        initial_value_per_token = weights[0] * initial_pool_value
        initial_reserves = initial_value_per_token / local_prices[0]

        if run_fingerprint["do_arb"]:
            if run_fingerprint["arb_frequency"] != 1:
                arb_acted_upon_weights = weights[:: run_fingerprint["arb_frequency"]]
                arb_acted_upon_local_prices = local_prices[:: run_fingerprint["arb_frequency"]]
            else:
                arb_acted_upon_weights = weights
                arb_acted_upon_local_prices = local_prices

            reserve_ratios = _jax_calc_quantAMM_reserve_ratios(
                arb_acted_upon_weights[:-1],
                arb_acted_upon_local_prices[:-1],
                arb_acted_upon_weights[1:],
                arb_acted_upon_local_prices[1:],
            )

            # calculate the reserves by cumprod of reserve ratios
            reserves = jnp.vstack(
                [
                    initial_reserves,
                    initial_reserves * jnp.cumprod(reserve_ratios, axis=0),
                ]
            )
        else:
            reserves = jnp.broadcast_to(
                initial_reserves, arb_acted_upon_local_prices.shape
            )

        return reserves

    @partial(jit, static_argnums=(2))
    def calculate_reserves_with_dynamic_inputs(
        self,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        prices: jnp.ndarray,
        start_index: jnp.ndarray,
        fees_array: jnp.ndarray,
        arb_thresh_array: jnp.ndarray,
        arb_fees_array: jnp.ndarray,
        trade_array: jnp.ndarray,
        additional_oracle_input: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        bout_length = run_fingerprint["bout_length"]
        n_assets = run_fingerprint["n_assets"]

        local_prices = dynamic_slice(prices, start_index, (bout_length - 1, n_assets))

        weights = self.calculate_weights(
            params, run_fingerprint, prices, start_index, additional_oracle_input
        )
        if run_fingerprint["arb_frequency"] != 1:
            arb_acted_upon_weights = weights[:: run_fingerprint["arb_frequency"]]
            arb_acted_upon_local_prices = local_prices[
                :: run_fingerprint["arb_frequency"]
            ]
        else:
            arb_acted_upon_weights = weights
            arb_acted_upon_local_prices = local_prices

        initial_pool_value = run_fingerprint["initial_pool_value"]
        initial_value_per_token = arb_acted_upon_weights[0] * initial_pool_value
        initial_reserves = initial_value_per_token / arb_acted_upon_local_prices[0]

        # any of fees_array, arb_thresh_array, arb_fees_array, trade_array
        # can be singletons, in which case we repeat them for the length of the bout

        # Determine the maximum leading dimension
        max_len = bout_length - 1
        if run_fingerprint["arb_frequency"] != 1:
            max_len = max_len // run_fingerprint["arb_frequency"]
        # Broadcast input arrays to match the maximum leading dimension.
        # If they are singletons, this will just repeat them for the length of the bout.
        # If they are arrays of length bout_length, this will cause no change.
        fees_array_broadcast = jnp.broadcast_to(
            fees_array, (max_len,) + fees_array.shape[1:]
        )
        arb_thresh_array_broadcast = jnp.broadcast_to(
            arb_thresh_array, (max_len,) + arb_thresh_array.shape[1:]
        )
        arb_fees_array_broadcast = jnp.broadcast_to(
            arb_fees_array, (max_len,) + arb_fees_array.shape[1:]
        )
        # if we are doing trades, the trades array must be of the same length as the other arrays
        if run_fingerprint["do_trades"]:
            assert trade_array.shape[0] == max_len
        reserves = _jax_calc_quantAMM_reserves_with_dynamic_inputs(
            initial_reserves,
            arb_acted_upon_weights,
            arb_acted_upon_local_prices,
            fees_array_broadcast,
            arb_thresh_array_broadcast,
            arb_fees_array_broadcast,
            jnp.array(run_fingerprint["all_sig_variations"]),
            trade_array,
            run_fingerprint["do_trades"],
            run_fingerprint["do_arb"],
        )
        return reserves

    @abstractmethod
    def calculate_raw_weights_outputs(
        self,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        prices: jnp.ndarray,
        additional_oracle_input: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Calculate raw weight adjustments based on price history.

        This is the first step in TFMM's two-step weight calculation process.
        Subclasses must implement their specific weight adjustment logic.

        Parameters
        ----------
        params : Dict[str, Any]
            Pool parameters for weight calculation
        run_fingerprint : Dict[str, Any]
            Simulation settings
        prices : jnp.ndarray
            Historical price data
        additional_oracle_input : Optional[jnp.ndarray]
            Extra data for weight calculation

        Returns
        -------
        jnp.ndarray
            Raw weight adjustment values
        """
        pass

    @abstractmethod
    def fine_weight_output(
        self,
        raw_weight_output: jnp.ndarray,
        initial_weights: jnp.ndarray,
        run_fingerprint: Dict[str, Any],
        params: Dict[str, Any],
    ) -> jnp.ndarray:
        """
        Refine raw weight outputs into final weights.

        Second step of TFMM's weight calculation process. Converts raw weight
        adjustments into valid pool weights.

        Parameters
        ----------
        raw_weight_output : jnp.ndarray
            Output from calculate_raw_weights_outputs
        initial_weights : jnp.ndarray
            Starting weights
        run_fingerprint : Dict[str, Any]
            Simulation settings
        params : Dict[str, Any]
            Pool parameters

        Returns
        -------
        jnp.ndarray
            Final refined weights
        """
        pass

    @partial(jit, static_argnums=(2, 5))
    def calculate_weights(
        self,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        prices: jnp.ndarray,
        start_index: jnp.ndarray,
        additional_oracle_input: Optional[jnp.ndarray] = None,
        *args,
        **kwargs,
    ) -> jnp.ndarray:
        """
        Calculate the weights of assets in the pool.

        This method should be implemented by subclasses to define how weights are calculated
        based on current prices, pool parameters, and optional additional oracle input.

        Args:
            prices (jnp.ndarray): Current prices of the assets.
            params (Dict[str, Any]): Pool parameters.
            additional_oracle_input (Optional[jnp.ndarray], optional): Additional input from an oracle. Defaults to None.

        Returns:
            jnp.ndarray: Calculated weights for each asset in the pool.
        """
        chunk_period = run_fingerprint["chunk_period"]
        bout_length = run_fingerprint["bout_length"]
        n_assets = run_fingerprint["n_assets"]
        raw_weight_outputs = self.calculate_raw_weights_outputs(
            params, run_fingerprint, prices, additional_oracle_input
        )
        # we dont't want to change the initial weights during any training
        # so wrap them in a stop_grad
        initial_weights = self.calculate_initial_weights(params)

        # we have a sequence now of weight changes, but if we are doing
        # a burnin operation, we need to cut off the changes associated
        # with the burnin period, ie everything before the start of the sequence

        start_index_coarse = ((start_index[0] / chunk_period).astype("int64"), 0)

        raw_weight_outputs = dynamic_slice(
            raw_weight_outputs,
            start_index_coarse,
            (int((bout_length) / chunk_period), n_assets),
        )
        raw_weight_outputs_cpu = device_put(raw_weight_outputs, CPU_DEVICE)
        initial_weights_cpu = device_put(initial_weights, CPU_DEVICE)

        weights = self.fine_weight_output(
            raw_weight_outputs_cpu,
            initial_weights_cpu,
            run_fingerprint,
            params,
        )

        weights = dynamic_slice(weights, (0, 0), (bout_length - 1, n_assets))

        return weights

    def calculate_all_signature_variations(self, params: Dict[str, Any]) -> jnp.ndarray:
        """
        Calculate all valid trading signature combinations.

        Abstract method that subclasses may implement to define valid trading patterns.
        Can be used by reserve calculation methods to determine possible arbitrage opportunities.

        Parameters
        ----------
        params : Dict[str, Any]
            Pool parameters that may influence valid trade combinations

        Returns
        -------
        jnp.ndarray
            Array of valid trading signature combinations

        Raises
        ------
        NotImplementedError
            Base class does not implement this method
        """
        raise NotImplementedError

    def make_vmap_in_axes(self, params: Dict[str, Any], n_repeats_of_recurred: int = 0):
        """
        Configure JAX vectorization axes for pool parameters.

        FMM pools handle subsidiary parameters differently
        for vectorization due to their potentially more complex parameter structure.

        Parameters
        ----------
        params : Dict[str, Any]
            Pool parameters to vectorize
        n_repeats_of_recurred : int, optional
            Number of times to repeat recurrent parameters, by default 0

        Returns
        -------
        Dict[str, Any]
            vmap axes configuration with subsidiary_params handled separately
        """
        return make_vmap_in_axes_dict(
            params, 0, [], ["subsidary_params"], n_repeats_of_recurred
        )

    def is_trainable(self):
        """
        Indicate if pool weights can be trained.

        TFMM pools are trainable by default, as their weights
        change based on market conditions.

        Returns
        -------
        bool
            Always True for TFMM pools as weights are trainable
        """
        return True

    @classmethod
    def process_parameters(cls, update_rule_parameters, n_assets):
        """
        Process TFMM pool parameters from web interface input.

        Handles common TFMM parameters and delegates pool-specific processing
        to subclasses. Supports both per-token and global parameters.

        Parameters
        ----------
        update_rule_parameters : List[Parameter]
            Raw parameters from web interface, each containing:
            - name: Parameter identifier
            - value: Parameter values per token
        n_assets : int
            Number of tokens in pool

        Returns
        -------
        Dict[str, np.ndarray]
            Processed parameters including:
            - logit_lamb: Memory parameter
            - k: Update rate parameter
            - Additional pool-specific parameters

        Notes
        -----
        - Handles parameter broadcasting for single values
        - Validates parameter dimensions
        - Processes memory_days and k_per_day specially
        - Allows subclasses to add specific parameters
        """
        result = {}
        processed_params = set()

        # Process TFMM common parameters
        memory_days_values = cls._process_memory_days(update_rule_parameters, n_assets)
        if memory_days_values is not None:
            result.update(memory_days_values)
            processed_params.add("memory_days")

        k_values = cls._process_k_per_day(update_rule_parameters, n_assets)
        if k_values is not None:
            result.update(k_values)
            processed_params.add("k_per_day")

        # Let specific pools process their parameters
        specific_params = cls._process_specific_parameters(update_rule_parameters, n_assets)
        if specific_params is not None:
            result.update(specific_params)
            # Assume any parameters returned by specific processing are handled
            processed_params.update(specific_params.keys())

        # Process any remaining parameters in a default way
        for urp in update_rule_parameters:
            if urp.name not in processed_params:
                value = []
                for tokenValue in urp.value:
                    value.append(tokenValue)
                if len(value) != n_assets:
                    value = [value[0]] * n_assets
                result[urp.name] = np.array(value)

        return result

    @classmethod
    def _process_memory_days(cls, update_rule_parameters, n_assets):
        """
        Process memory_days parameter into logit_lamb values.

        Converts memory_days into a logit-transformed lambda parameter
        that determines how quickly the pool forgets past price information.

        Parameters
        ----------
        update_rule_parameters : List[Parameter]
            Raw parameters containing memory_days values
        n_assets : int
            Number of tokens in pool

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary with 'logit_lamb' key containing transformed values,
            or None if memory_days not found

        Notes
        -----
        - Converts memory days to lambda using memory_days_to_lamb
        - Applies logit transform for numerical stability
        - Broadcasts single values to match n_assets
        """
        for urp in update_rule_parameters:
            if urp.name == "memory_days":
                logit_lamb_vals = []
                memory_days_values = urp.value
                for tokenValue in urp.value:
                    initial_lamb = memory_days_to_lamb(tokenValue)
                    logit_lamb = np.log(initial_lamb / (1.0 - initial_lamb))
                    logit_lamb_vals.append(logit_lamb)
                if len(logit_lamb_vals) != n_assets:
                    logit_lamb_vals = [logit_lamb_vals[0]] * n_assets
                return {"logit_lamb": np.array(logit_lamb_vals)}
        return None

    @classmethod
    def _process_k_per_day(cls, update_rule_parameters, n_assets):
        """
        Process k_per_day parameter into update rate values.

        The k parameter determines how quickly weights adjust to new prices.
        Higher values mean faster adjustments.

        Parameters
        ----------
        update_rule_parameters : List[Parameter]
            Raw parameters containing k_per_day values
        n_assets : int
            Number of tokens in pool

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary with 'k' key containing update rates,
            or None if k_per_day not found

        Notes
        -----
        - Uses raw k values without transformation
        - Broadcasts single values to match n_assets
        """
        for urp in update_rule_parameters:
            if urp.name == "k_per_day":
                k_vals = []
                for tokenValue in urp.value:
                    k_vals.append(tokenValue)
                if len(k_vals) != n_assets:
                    k_vals = [k_vals[0]] * n_assets
                return {"k": np.array(k_vals)}
        return None

    @classmethod
    def _process_specific_parameters(cls, update_rule_parameters, n_assets):
        """
        Process pool-specific parameters.

        Abstract method that subclasses should override to handle any
        parameters specific to their implementation.

        Parameters
        ----------
        update_rule_parameters : List[Parameter]
            Raw parameters to process
        n_assets : int
            Number of tokens in pool

        Returns
        -------
        Dict[str, np.ndarray] or None
            Processed pool-specific parameters if any,
            None if no specific parameters needed
        """
        return None
