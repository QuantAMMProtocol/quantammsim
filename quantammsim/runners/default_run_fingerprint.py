run_fingerprint_defaults = {
    "freq": "minute",
    "startDateString": "2021-02-03 00:00:00",
    "endDateString": "2022-06-03 00:00:00",
    "endTestDateString": "2022-07-03 00:00:00",
    "tokens": ["BTC", "USDC", "ETH"],
    "rule": "mean_reversion_channel",
    "optimisation_settings": {
        "base_lr": 0.01,
        "optimiser": "sgd",
        "decay_lr_ratio": 0.8,
        "decay_lr_plateau": 100,
        "batch_size": 8,
        "train_on_hessian_trace": False,
        "min_lr": 1e-6,
        "n_iterations": 1000,
        "n_cycles": 5,
        "sample_method": "uniform",
        "n_parameter_sets": 3,
        "training_data_kind": "historic",
        "max_mc_version": 9,
        "include_flipped_training_data": False,
        "initial_random_key": 0,
        "method": "gradient_descent",
    },
    "initial_memory_length": 10.0,
    "initial_memory_length_delta": 0.0,
    "initial_k_per_day": 20,
    "bout_offset": 24 * 60 * 7,
    "initial_weights_logits": 1.0,
    "initial_log_amplitude": -10.0,
    "initial_raw_width": -8.0,
    "initial_raw_exponents": 0.0,
    "subsidary_pools": [],
    "maximum_change": 3e-4,
    "chunk_period": 60,
    "weight_interpolation_period": 60,
    "return_val": "sharpe",
    "initial_pool_value": 1000000.0,
    "fees": 0.0,
    "arb_fees": 0.0,
    "gas_cost": 0.0,
    "use_alt_lamb": True,
    "use_pre_exp_scaling": True,
    "weight_interpolation_method": "linear",
    "arb_frequency": 1,
    "arb_quality": 1.0,
    "do_trades": False,
    "numeraire": None,
    "do_arb": True,
    "max_memory_days": 365,
}


optuna_settings = {
    "study_name": None,  # Will be auto-generated if None
    "storage": {
        "type": "sqlite",  # or "mysql", "postgresql"
        "url": None,  # e.g., "sqlite:///studies.db"
    },
    "n_trials": 20,
    "n_jobs": 4,  # Number of parallel workers
    "timeout": 7200,  # Maximum optimization time in seconds
    "n_startup_trials": 10,
    "early_stopping": {
        "enabled": False,
        "patience": 100,  # Trials without improvement
        "min_improvement": 0.001,  # Minimum relative improvement
    },
    "multi_objective": False,
    "make_scalar": False,
    "parameter_config": {
        "memory_length": {
            "low": 1,
            "high": 200,
            "log_scale": True,
            "scalar": False,
        },
        "memory_length_delta": {
            "low": 0.1,
            "high": 100,
            "log_scale": True,
            "scalar": False,
        },
        "log_k": {
            "low": -10.0,
            "high": 10.0,
            "log_scale": False,
            "scalar": False,
        },
        "k_per_day": {
            "low": 0.1,
            "high": 1000,
            "log_scale": True,
            "scalar": False,
        },
        "weights_logits": {
            "low": -10,
            "high": 10,
            "log_scale": False,
            "scalar": False,
        },
        "log_amplitude": {
            "low": -10,
            "high": 10,
            "log_scale": False,
            "scalar": False,
        },
        "raw_width": {
            "low": -10,
            "high": 10,
            "log_scale": False,
            "scalar": False,
        },
        "raw_exponents": {
            "low": 0,
            "high": 10,
            "log_scale": False,
            "scalar": False,
        },
        "raw_pre_exp_scaling": {
            "low": -10,
            "high": 10,
            "log_scale": False,
            "scalar": False,
        },
        "memory_days_1": {
            "low": 0.5,
            "high": 200,
            "log_scale": True,
            "scalar": False,
        },
        "memory_days_2": {
            "low": 0.5,
            "high": 200,
            "log_scale": True,
            "scalar": False,
        },
        "logit_lamb": {
            "low": -10,
            "high": 10,
            "log_scale": False,
            "scalar": False,
        },
    },
}

run_fingerprint_defaults["optimisation_settings"]["optuna_settings"] = optuna_settings
