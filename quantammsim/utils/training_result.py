
class TrainingResult:
    def __init__(self, alphabetic, bout_offset, chunk_period, endDateString, endDateUnix, endTestDateString, fees,
                 filename_override, freq, initial_k, initial_k_per_day, initial_log_amplitude, initial_memory_length,
                 initial_memory_length_delta, initial_pool_value, initial_raw_exponents, initial_raw_width,
                 initial_weights_logits, maximum_change, optimisation_settings, return_val, rule, startDateString,
                 startDateUnix, subsidary_pools, tokens, use_alt_lamb, use_pre_exp_scaling, weight_interpolation_period):
        self.alphabetic = alphabetic
        self.bout_offset = bout_offset
        self.chunk_period = chunk_period
        self.endDateString = endDateString
        self.endDateUnix = endDateUnix
        self.endTestDateString = endTestDateString
        self.fees = fees
        self.filename_override = filename_override
        self.freq = freq
        self.initial_k = initial_k
        self.initial_k_per_day = initial_k_per_day
        self.initial_log_amplitude = initial_log_amplitude
        self.initial_memory_length = initial_memory_length
        self.initial_memory_length_delta = initial_memory_length_delta
        self.initial_pool_value = initial_pool_value
        self.initial_raw_exponents = initial_raw_exponents
        self.initial_raw_width = initial_raw_width
        self.initial_weights_logits = initial_weights_logits
        self.maximum_change = maximum_change
        self.optimisation_settings = optimisation_settings
        self.return_val = return_val
        self.rule = rule
        self.startDateString = startDateString
        self.startDateUnix = startDateUnix
        self.subsidary_pools = subsidary_pools
        self.tokens = tokens
        self.use_alt_lamb = use_alt_lamb
        self.use_pre_exp_scaling = use_pre_exp_scaling
        self.weight_interpolation_period = weight_interpolation_period
