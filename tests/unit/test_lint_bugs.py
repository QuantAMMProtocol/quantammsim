"""Tests documenting bugs found by ruff linting on dev branch (f52f125).

Every test in this file is EXPECTED TO FAIL until the corresponding bug
is fixed.  Once a bug is fixed, the test should pass and can be kept as
a regression test.

Organised by ruff rule:
  F821 — Undefined name (NameError at runtime)
  B006 — Mutable default argument
  B023 — Loop variable captured in closure
  PLW0127 — Self-assigning variable
  E711 — Comparison to None using ==
  W605 — Invalid escape sequence
"""

import ast
import importlib
import inspect
import os
import warnings

import jax.numpy as jnp
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_package_root():
    """Return the directory containing the quantammsim package source."""
    import quantammsim
    return os.path.dirname(os.path.dirname(quantammsim.__file__))


def _get_default(module_path, func_name, param_name):
    """Return the default value of *param_name* on *func_name* in *module_path*.

    Handles JIT-wrapped functions by inspecting ``__wrapped__`` when available.
    """
    module = importlib.import_module(module_path)
    func = getattr(module, func_name)
    if hasattr(func, "__wrapped__"):
        func = func.__wrapped__
    sig = inspect.signature(func)
    return sig.parameters[param_name].default


def _find_undefined_returns(source_path, func_name):
    """Check whether *func_name* returns a name never assigned in its body."""
    with open(source_path) as f:
        tree = ast.parse(f.read(), source_path)

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            # Collect all names assigned (including tuple unpacking)
            assigned = set()
            for child in ast.walk(node):
                if isinstance(child, ast.Assign):
                    for target in child.targets:
                        if isinstance(target, ast.Name):
                            assigned.add(target.id)
                        elif isinstance(target, ast.Tuple):
                            for elt in target.elts:
                                if isinstance(elt, ast.Name):
                                    assigned.add(elt.id)
            # Also count function arguments as assigned
            for arg in node.args.args:
                assigned.add(arg.arg)

            # Check return statements
            bad = []
            for child in ast.walk(node):
                if isinstance(child, ast.Return) and isinstance(child.value, ast.Name):
                    if child.value.id not in assigned:
                        bad.append((child.lineno, child.value.id))
            return bad
    return []


# ---------------------------------------------------------------------------
# F821 — Undefined names  (each should produce NameError when exercised)
# ---------------------------------------------------------------------------


class TestF821UndefinedNames:
    """Each test calls a function containing an undefined name.

    The test asserts *correct* behaviour; it will fail with NameError (or
    similar) because the code references a name that was never defined.
    """

    def test_forward_pass_local_prices_undefined(self):
        """F821: forward_pass.py:902 — `local_prices` used before assignment.

        When arb_frequency != 1, the code tries local_prices.shape[0]
        but local_prices is not defined until line 909.
        Uses __wrapped__ to bypass JIT (forward_pass is a PjitFunction).
        """
        from quantammsim.core_simulator.forward_pass import forward_pass

        class _MockPool:
            def calculate_reserves_zero_fees(self, params, static_dict, prices, start_index):
                return jnp.ones((5, 2))

        static_dict = {
            "training_data_kind": "historic",
            "n_assets": 2,
            "return_val": "daily_log_sharpe",
            "bout_length": 11,
            "fees": 0.0,
            "gas_cost": 0.0,
            "arb_fees": 0.0,
            "arb_frequency": 2,  # triggers the buggy branch
        }
        prices = jnp.ones((20, 2))
        start_index = jnp.array([0, 0])

        # __wrapped__ arg order: params, start_index, prices, trades, fees,
        # gas_cost, arb_fees, pool, static_dict
        result = forward_pass.__wrapped__(
            {},            # params
            start_index,   # start_index
            prices,        # prices
            None,          # trades_array
            None,          # fees_array
            None,          # gas_cost_array
            None,          # arb_fees_array
            _MockPool(),   # pool
            static_dict,   # static_dict
        )
        assert "reserves" in result or isinstance(result, jnp.ndarray)

    def test_cow_reserves_bid_ask_undefined(self):
        """F821: cow_reserves_.py:644,646 — `bid_price`/`ask_price` never defined.

        Function takes `price` (singular) but uses `bid_price` and `ask_price`.
        """
        from quantammsim.pools.FM_AMM.cow_reserves_ import align_cowamm_position_jax

        reserves = jnp.array([100.0, 200.0])
        price = 2.0
        result = align_cowamm_position_jax(reserves, price)
        assert result.shape == (2,)

    def test_cow_reserves_prev_weights_undefined(self):
        """F821: cow_reserves_.py:697 — `prev_weights` should be `weights`.

        Parameter is named `weights` but body uses `prev_weights`.
        """
        from quantammsim.pools.FM_AMM.cow_reserves_ import (
            _jax_calc_cowamm_reserve_ratio_n_assets,
        )

        prev_prices = jnp.array([1.0, 2.0])
        weights = jnp.array([0.5, 0.5])
        prices = jnp.array([1.1, 2.1])
        result = _jax_calc_cowamm_reserve_ratio_n_assets(prev_prices, weights, prices)
        assert result.shape[0] == 2

    def test_cow_reserves_trade_undefined(self):
        """F821: cow_reserves_.py:951 — `trade` never extracted from input_list.

        When do_trades=True, the code uses `trade` which is never unpacked.
        """
        from quantammsim.pools.FM_AMM.cow_reserves_ import (
            _jax_calc_cowamm_reserves_with_dynamic_fees_and_trades_scan_function,
        )

        carry = [jnp.array([1.0, 2.0]), jnp.array([100.0, 200.0])]
        inputs = [
            jnp.array([1.1, 2.1]),  # prices
            jnp.array(0.997),       # gamma
            jnp.array(0.01),        # arb_thresh
            jnp.array(0.001),       # arb_fees
        ]
        result = _jax_calc_cowamm_reserves_with_dynamic_fees_and_trades_scan_function(
            carry, inputs, True  # do_trades=True triggers the bug
        )
        assert len(result) == 2

    def test_tfmm_calculate_weights_direct_returns_undefined(self):
        """F821: TFMM_base_pool.py:1460 — `return weights` but `weights` never assigned.

        _jax_calc_coarse_weights returns (actual_starts_cpu, scaled_diffs_cpu,
        target_weights_cpu) but the method returns `weights` which is undefined.
        Tested via AST inspection because the method's JIT decorator has
        unhashable static_argnums (prices marked static — itself a bug).
        """
        import quantammsim.pools.G3M.quantamm.TFMM_base_pool as mod
        source_path = inspect.getfile(mod)
        bad = _find_undefined_returns(source_path, "calculate_weights_direct")
        assert len(bad) == 0, (
            "calculate_weights_direct returns undefined name(s): "
            + ", ".join(f"line {ln}: `{name}`" for ln, name in bad)
        )

    def test_min_variance_n_assets_undefined(self):
        """F821: min_variance_pool.py:330 — `n_assets` not defined.

        The memory_days loop uses len(run_fingerprint["tokens"]) correctly,
        but the second loop uses the undefined `n_assets`.
        """
        from quantammsim.pools.G3M.quantamm.min_variance_pool import MinVariancePool

        class _MockParam:
            def __init__(self, name, value):
                self.name = name
                self.value = value

        params = [
            _MockParam("memory_days", [30.0, 30.0]),
            _MockParam("other_param", [1.0]),  # triggers the n_assets branch
        ]
        run_fingerprint = {"tokens": ["ETH", "BTC"]}

        result = MinVariancePool.process_parameters(params, run_fingerprint)
        assert "other_param" in result
        assert len(result["other_param"]) == 2

    def test_estimator_primitives_np_undefined(self):
        """F821: estimator_primitives.py:334 — `np` used but only `jnp` imported.

        np.zeros should be jnp.zeros.
        """
        from quantammsim.pools.G3M.quantamm.update_rule_estimators.estimator_primitives import (
            _jax_covariance_at_infinity_via_conv,
        )

        n_assets = 2
        T = 20
        K = 5
        arr_in = jnp.cumsum(jnp.ones((T, n_assets)) * 0.01, axis=0)
        ewma = arr_in[:-1]  # (T-1, n_assets)
        # conv_vmap = vmap(vmap(convolve, [-1,-1], -1), [1,None], 1)
        # kernel shape: (K, n_assets) to match the double-vmap convention
        kernel = jnp.ones((K, n_assets)) / K
        lamb = 0.94

        result = _jax_covariance_at_infinity_via_conv(arr_in, ewma, kernel, lamb)
        assert result.shape == (T, n_assets, n_assets)

    def test_param_financial_calculator_multiple_undefined(self):
        """F821: param_financial_calculator.py:962 — multiple undefined names.

        `generate_interarray_permutations`, `get_data_dict` are undefined.
        Also `price_data_cache = {}` but `.append()` is called on it (dict has
        no append).
        """
        from quantammsim.simulator_analysis_tools.finance.param_financial_calculator import (
            retrieve_mc_param_financial_results,
        )

        run_fingerprint = {
            "tokens": ["ETH"],
            "startDateString": "2023-01-01",
            "endDateString": "2023-06-01",
        }
        params = {}
        result = retrieve_mc_param_financial_results(
            run_fingerprint, params, "2024-01-01"
        )
        assert isinstance(result, list)

    def test_coinbase_prefix_undefined(self):
        """F821: coinbase_data_utils.py:283,288 — `prefix` not in scope.

        `prefix` was copy-pasted from fill_in_missing_rows_with_exchange_data.
        Tested via AST: checks that `prefix` is used inside
        fill_missing_rows_with_coinbase_data but is not a parameter or local.
        """
        import quantammsim.utils.data_processing.coinbase_data_utils as mod
        source_path = inspect.getfile(mod)
        with open(source_path) as f:
            tree = ast.parse(f.read(), source_path)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "fill_missing_rows_with_coinbase_data":
                # Collect parameter names
                param_names = {arg.arg for arg in node.args.args}
                # Collect assigned names in function body
                assigned = set()
                for child in ast.walk(node):
                    if isinstance(child, ast.Assign):
                        for target in child.targets:
                            if isinstance(target, ast.Name):
                                assigned.add(target.id)
                            elif isinstance(target, ast.Tuple):
                                for elt in target.elts:
                                    if isinstance(elt, ast.Name):
                                        assigned.add(elt.id)
                defined = param_names | assigned
                # Check for use of `prefix` which is not defined
                uses_of_prefix = []
                for child in ast.walk(node):
                    if isinstance(child, ast.Name) and child.id == "prefix":
                        uses_of_prefix.append(child.lineno)
                assert len(uses_of_prefix) == 0 or "prefix" in defined, (
                    f"`prefix` used on lines {uses_of_prefix} but not defined "
                    f"in fill_missing_rows_with_coinbase_data "
                    f"(params: {param_names}, locals: {assigned})"
                )
                return
        pytest.fail("fill_missing_rows_with_coinbase_data not found in source")

    def test_price_data_mc_path_undefined(self):
        """F821: price_data_fingerprint_utils.py:125,128,136 —
        `list_of_tickers`, `mc_data_available_for`, `cols` all undefined.

        The 'mc' branch uses names that are never defined in this function.
        """
        from quantammsim.utils.data_processing.price_data_fingerprint_utils import (
            load_price_data_if_fingerprints_match,
        )

        run_fingerprint = {
            "tokens": ["ETH", "BTC"],
            "subsidary_pools": [],
            "optimisation_settings": {
                "training_data_kind": "mc",
                "max_mc_version": 9,
            },
        }
        result = load_price_data_if_fingerprints_match(
            [run_fingerprint, run_fingerprint],
        )
        assert result is not None


# ---------------------------------------------------------------------------
# B006 — Mutable default arguments
# ---------------------------------------------------------------------------


_B006_CASES = [
    ("quantammsim.core_simulator.forward_pass", "forward_pass", "static_dict"),
    ("quantammsim.core_simulator.forward_pass", "forward_pass_nograd", "static_dict"),
    ("quantammsim.core_simulator.param_utils", "make_vmap_in_axes_dict", "keys_with_no_vamp"),
    ("quantammsim.runners.jax_runners", "do_run_on_historic_data", "params"),
    ("quantammsim.runners.jax_runners", "do_run_on_historic_data_with_provided_coarse_weights", "params"),
    ("quantammsim.utils.data_processing.historic_data_utils", "get_historic_parquet_data", "cols"),
    ("quantammsim.utils.data_processing.historic_data_utils", "get_historic_csv_data", "cols"),
    ("quantammsim.utils.data_processing.historic_data_utils", "get_historic_csv_data_w_versions", "cols"),
    ("quantammsim.utils.data_processing.price_data_fingerprint_utils", "load_price_data_if_fingerprints_match", "base_keys_to_check"),
    ("quantammsim.utils.data_processing.price_data_fingerprint_utils", "load_price_data_if_fingerprints_match", "optimisation_keys_to_check"),
    ("quantammsim.utils.data_processing.price_data_fingerprint_utils", "load_price_data_if_fingerprints_in_dir_match", "base_keys_to_check"),
    ("quantammsim.utils.data_processing.price_data_fingerprint_utils", "load_price_data_if_fingerprints_in_dir_match", "optimisation_keys_to_check"),
    ("quantammsim.utils.plot_utils", "calc_values_from_results", "keepcols"),
]


class TestB006MutableDefaults:
    """Default argument values should not be mutable (list, dict, set).

    Each test checks a function's default value via inspect and asserts
    it is either None or an immutable type.
    """

    @pytest.mark.parametrize("module_path,func_name,param_name", _B006_CASES,
                             ids=[f"{m.split('.')[-1]}.{f}.{p}" for m, f, p in _B006_CASES])
    def test_no_mutable_default(self, module_path, func_name, param_name):
        default = _get_default(module_path, func_name, param_name)
        assert not isinstance(default, (dict, list, set)), (
            f"{func_name}({param_name}=...) uses mutable {type(default).__name__} "
            f"as default: {default!r}"
        )


# ---------------------------------------------------------------------------
# B023 — Loop variable captured in closure
# ---------------------------------------------------------------------------


class TestB023LoopVariableCapture:
    """Lambda inside loop should bind the loop variable by value, not by reference."""

    def test_nan_rollback_binds_bool_idx_correctly(self):
        """B023: jax_runner_utils.py:884 — bool_idx captured in lambda.

        When both 'log_k' and 'logit_lamb' have NaN gradients, each key
        should use its OWN bool_idx mask, not the last one computed.
        """
        from quantammsim.runners.jax_runner_utils import nan_rollback

        # log_k has NaN in position [0], logit_lamb has NaN in position [1]
        grads = {
            "log_k": jnp.array([[float("nan")], [1.0]]),
            "logit_lamb": jnp.array([[1.0], [float("nan")]]),
        }
        params = {
            "log_k": jnp.array([[10.0], [20.0]]),
            "logit_lamb": jnp.array([[30.0], [40.0]]),
        }
        old_params = {
            "log_k": jnp.array([[1.0], [2.0]]),
            "logit_lamb": jnp.array([[3.0], [4.0]]),
        }

        result = nan_rollback(grads, params, old_params)

        # Position 0: log_k had NaN → should rollback log_k[0] to old value
        assert float(result["log_k"][0, 0]) == pytest.approx(1.0), (
            "log_k[0] should be rolled back to old_params value (NaN grad)"
        )
        # Position 1: log_k was fine → should keep params value
        assert float(result["log_k"][1, 0]) == pytest.approx(20.0), (
            "log_k[1] should keep params value (valid grad)"
        )
        # Position 0: logit_lamb was fine → should keep params value
        assert float(result["logit_lamb"][0, 0]) == pytest.approx(30.0), (
            "logit_lamb[0] should keep params value (valid grad)"
        )
        # Position 1: logit_lamb had NaN → should rollback to old value
        assert float(result["logit_lamb"][1, 0]) == pytest.approx(4.0), (
            "logit_lamb[1] should be rolled back to old_params value (NaN grad)"
        )


# ---------------------------------------------------------------------------
# PLW0127 — Self-assigning variable
# ---------------------------------------------------------------------------


class TestPLW0127SelfAssignment:
    """Self-assignments are dead code and should be removed."""

    def test_no_self_assignment_in_plot_utils(self):
        """PLW0127: plot_utils.py:195 — `plot_data = plot_data`."""
        import quantammsim.utils.plot_utils as mod
        source_path = inspect.getfile(mod)
        with open(source_path) as f:
            tree = ast.parse(f.read(), source_path)

        self_assignments = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                value = node.value
                if (
                    isinstance(target, ast.Name)
                    and isinstance(value, ast.Name)
                    and target.id == value.id
                ):
                    self_assignments.append(
                        f"line {node.lineno}: {target.id} = {value.id}"
                    )

        assert len(self_assignments) == 0, (
            f"Found {len(self_assignments)} self-assignment(s):\n"
            + "\n".join(f"  {s}" for s in self_assignments)
        )


# ---------------------------------------------------------------------------
# E711 — Comparison to None should use `is` / `is not`
# ---------------------------------------------------------------------------


class TestE711NoneComparison:
    """None comparisons should use `is None`, not `== None`."""

    def test_fine_weights_uses_is_none(self):
        """E711: fine_weights.py:448 — `if minimum_weight == None:`."""
        import quantammsim.pools.G3M.quantamm.weight_calculations.fine_weights as mod
        source_path = inspect.getfile(mod)
        with open(source_path) as f:
            tree = ast.parse(f.read(), source_path)

        bad_comparisons = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare):
                for op, comparator in zip(node.ops, node.comparators):
                    if (
                        isinstance(op, (ast.Eq, ast.NotEq))
                        and isinstance(comparator, ast.Constant)
                        and comparator.value is None
                    ):
                        bad_comparisons.append(
                            f"line {node.lineno}: uses {'==' if isinstance(op, ast.Eq) else '!='} None"
                        )

        assert len(bad_comparisons) == 0, (
            f"Found {len(bad_comparisons)} `== None` / `!= None` comparison(s):\n"
            + "\n".join(f"  {s}" for s in bad_comparisons)
        )


# ---------------------------------------------------------------------------
# W605 — Invalid escape sequences
# ---------------------------------------------------------------------------


_W605_FILES = [
    "quantammsim/core_simulator/forward_pass.py",
    "quantammsim/core_simulator/result_exporter.py",
    "quantammsim/utils/plot_utils.py",
]


class TestW605InvalidEscapeSequences:
    """Source files should not contain invalid escape sequences.

    These are DeprecationWarning in Python 3.9, SyntaxWarning in 3.12,
    and will become SyntaxError in a future Python version.
    """

    @pytest.mark.parametrize("rel_path", _W605_FILES,
                             ids=[p.split("/")[-1] for p in _W605_FILES])
    def test_no_invalid_escape_sequences(self, rel_path):
        source_path = os.path.join(_get_package_root(), rel_path)
        with open(source_path) as f:
            source = f.read()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            compile(source, source_path, "exec")

        escape_warnings = [
            w for w in caught
            if issubclass(w.category, (DeprecationWarning, SyntaxWarning))
            and "invalid escape sequence" in str(w.message)
        ]
        assert len(escape_warnings) == 0, (
            f"Found {len(escape_warnings)} invalid escape sequence(s) in {rel_path}:\n"
            + "\n".join(
                f"  line {w.lineno}: {w.message}" for w in escape_warnings
            )
        )
