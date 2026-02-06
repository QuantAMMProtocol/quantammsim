run_fingerprint_defaults = {
    "freq": "minute",
    "startDateString": "2023-01-01 00:00:00",
    "endDateString": "2023-06-01 00:00:00",
    "endTestDateString": None,
    "tokens": ["BTC", "ETH", "USDC"],
    "rule": "mean_reversion_channel",
    "optimisation_settings": {
        "base_lr": 0.1,
        "optimiser": "adamw",  # adamw supports weight_decay; use "adam" or "sgd" if no decay needed
        "weight_decay": 0.01,  # L2 regularization coefficient (only effective with adamw)
        "decay_lr_ratio": 0.8,
        "decay_lr_plateau": 100,
        "batch_size": 8,
        "train_on_hessian_trace": False,
        "min_lr": 1e-6,  # Fallback if lr_decay_ratio not set; prefer lr_decay_ratio
        "n_iterations": 1000,
        "n_cycles": 5,
        "sample_method": "uniform",
        "n_parameter_sets": 4,
        # noise_scale: Gaussian noise added to param sets 1+ (set 0 is canonical).
        # Larger values = more diverse initialization = better exploration but more variance.
        # Only effective when n_parameter_sets > 1.
        "noise_scale": 0.1,
        "parameter_init_method": "gaussian",  # "gaussian", "sobol", "lhs", "centered_lhs"
        "training_data_kind": "historic",
        "max_mc_version": 9,
        "include_flipped_training_data": False,
        "initial_random_key": 0,
        "method": "gradient_descent",
        "force_scalar": False,
        "use_plateau_decay": False,
        "use_gradient_clipping": True,  # Prevents gradient explosion
        "clip_norm": 10.0,
        "lr_schedule_type": "constant",  # Options: "constant", "cosine", "warmup_cosine", "exponential"
        "lr_decay_ratio": 1000,  # For decay schedules: min_lr = base_lr / lr_decay_ratio
        "warmup_steps": 100,  # Only used by "warmup_cosine" schedule
        # Early stopping settings
        "early_stopping": True,  # Stop training when validation metric stops improving
        "early_stopping_patience": 200,  # Iterations without validation improvement before stopping
        "early_stopping_metric": "daily_log_sharpe",  # Metric to monitor: "sharpe", "daily_log_sharpe", "returns", etc.
        # Validation holdout - fraction of training data held out for early stopping
        # If 0.0, early stopping uses test data (not recommended - data leakage)
        # If > 0.0, carves out this fraction from end of training for validation
        # Constraint: (1 - val_fraction) * bout_length > bout_offset
        "val_fraction": 0.2,
        # Stochastic Weight Averaging (SWA) settings
        "use_swa": False,  # Average parameters from last N checkpoints
        "swa_start_frac": 0.75,  # Start SWA after this fraction of training
        "swa_freq": 10,  # Collect parameters every N iterations for averaging
        # Checkpoint tracking for Rademacher complexity estimation
        # Tracks parameter performance at intervals to measure overfitting risk
        "track_checkpoints": False,  # Enable checkpoint tracking
        "checkpoint_interval": 10,  # Save checkpoint every N iterations
    },
    # Ensemble training settings
    # Use "ensemble__<rule>" in run_fingerprint["rule"] to enable ensemble averaging
    # e.g., "ensemble__momentum" or "ensemble__bounded__mean_reversion_channel"
    # n_ensemble_members: number of param sets averaged together per "parameter set"
    # When > 1, params shape becomes (n_parameter_sets, n_ensemble_members, ...)
    # and rule outputs are averaged across ensemble members before fine weights
    "n_ensemble_members": 1,  # Default 1 = no ensembling (backwards compatible)
    # Ensemble initialization method - controls how members are distributed in param space
    # Options: "gaussian" (random noise), "lhs" (Latin Hypercube), "centered_lhs",
    #          "sobol" (quasi-random), "grid" (regular grid)
    "ensemble_init_method": "gaussian",  # Default to Gaussian for backwards compatibility
    "ensemble_init_scale": 0.5,  # Spread of ensemble members (multiplier for offsets)
    "ensemble_init_seed": 42,  # Random seed for reproducible initialization
    "initial_memory_length": 10.0,
    "initial_memory_length_delta": 0.0,
    "initial_k_per_day": 20,
    "bout_offset": 24 * 60 * 7,
    "initial_weights_logits": 1.0,
    "initial_log_amplitude": 0.0,
    "initial_raw_width": 0.0,
    "initial_raw_exponents": 0.0,
    "initial_pre_exp_scaling": 0.5,
    "subsidary_pools": [],
    "maximum_change": 3e-4,
    "chunk_period": 1440,
    "weight_interpolation_period": 1440,
    "turnover_penalty": 0.0,
    "price_noise_sigma": 0.0,  # Log-normal price noise during training (0 = disabled)
    "return_val": "daily_log_sharpe",
    "initial_pool_value": 1000000.0,
    "fees": 0.0,
    "arb_fees": 0.0,
    "gas_cost": 0.0,
    "use_alt_lamb": False,
    "use_pre_exp_scaling": True,
    "weight_interpolation_method": "linear",
    "arb_frequency": 1,
    "arb_quality": 1.0,
    "do_trades": False,
    "numeraire": None,
    "do_arb": True,
    "max_memory_days": 365,
    "noise_trader_ratio": 0.0,
    "minimum_weight": None,  # will be set to 0.1 / n_assets
    "ste_max_change": False,
    "ste_min_max_weight": False,
    "weight_calculation_method": "auto",  # "auto", "vectorized", or "scan"
    # Learnable bounds settings - for per-asset min/max weight constraints
    # Control is via rule string prefix (e.g., "bounded__momentum")
    "learnable_bounds_settings": {
        "freeze_bounds": False,  # If True, treat bounds as hyperparameters (no gradients)
        "min_weights_per_asset": None,  # Must be set if using bounded pool, e.g., [0.05, 0.05]
        "max_weights_per_asset": None,  # Must be set if using bounded pool, e.g., [0.60, 0.60]
    },
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
    # expand_around: If True, search within a window around initial param values.
    # If False, search the full range specified in parameter_config.
    # For financial strategies, False often gives better exploration.
    "expand_around": True,
    # overfitting_penalty: Penalize solutions where train performance >> validation.
    # Value of 0.5 means: if train=1.0 and val=0.5, penalty = 0.5 * (1.0 - 0.5) = 0.25
    # Set to 0.0 to disable. Range [0.0, 1.0] recommended.
    "overfitting_penalty": 0.2,
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
