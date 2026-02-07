class TrainingResult:
    """
    Data class storing the configuration and results of a training run.

    Captures the full run-fingerprint settings used during a training run,
    enabling exact reproducibility of the simulation and optimisation
    pipeline. An instance of this class is typically serialised alongside
    the learned parameters so that any downstream consumer (analysis
    notebooks, walk-forward evaluators, deployment scripts) can reconstruct
    the training context without ambiguity.

    Attributes are grouped into several logical categories:

    **Date range**
        ``startDateString``, ``startDateUnix``, ``endDateString``,
        ``endDateUnix``, ``endTestDateString`` -- define the in-sample
        training window and the out-of-sample test boundary.

    **Pool configuration**
        ``rule``, ``tokens``, ``fees``, ``initial_pool_value``,
        ``subsidary_pools``, ``weight_interpolation_period`` -- specify
        the pool type, asset universe, fee tier, and related settings.

    **Strategy parameters (initial values)**
        ``initial_k_per_day``, ``initial_k``, ``initial_memory_length``,
        ``initial_memory_length_delta``, ``initial_log_amplitude``,
        ``initial_raw_width``, ``initial_raw_exponents``,
        ``initial_weights_logits``, ``use_alt_lamb``,
        ``use_pre_exp_scaling`` -- seed values and flags for the
        learnable strategy parameters.

    **Simulation settings**
        ``chunk_period``, ``freq``, ``bout_offset``, ``alphabetic``,
        ``maximum_change`` -- discretisation and simulation-level knobs.

    **Optimisation settings**
        ``optimisation_settings``, ``return_val``, ``filename_override``
        -- optimiser configuration and objective choice.
    """

    def __init__(
        self,
        alphabetic,
        bout_offset,
        chunk_period,
        endDateString,
        endDateUnix,
        endTestDateString,
        fees,
        filename_override,
        freq,
        initial_k,
        initial_k_per_day,
        initial_log_amplitude,
        initial_memory_length,
        initial_memory_length_delta,
        initial_pool_value,
        initial_raw_exponents,
        initial_raw_width,
        initial_weights_logits,
        maximum_change,
        optimisation_settings,
        return_val,
        rule,
        startDateString,
        startDateUnix,
        subsidary_pools,
        tokens,
        use_alt_lamb,
        use_pre_exp_scaling,
        weight_interpolation_period,
    ):
        """
        Initialise a TrainingResult from run-fingerprint fields.

        Parameters
        ----------
        alphabetic : bool
            Whether tokens are sorted alphabetically (affects weight
            ordering).
        bout_offset : int
            Offset (in timesteps) into the price series at which the
            training bout begins.
        chunk_period : int
            Number of minutes per discretisation chunk (e.g. 60 for
            hourly, 1440 for daily).
        endDateString : str
            End date of the in-sample training window, ISO-8601 format.
        endDateUnix : int
            Unix timestamp corresponding to ``endDateString``.
        endTestDateString : str
            End date of the out-of-sample test window, ISO-8601 format.
        fees : float
            Swap fee fraction charged per trade (e.g. 0.003 for 30 bps).
        filename_override : str or None
            Optional override for the output filename stem.
        freq : str
            Price data frequency identifier (e.g. ``"1min"``, ``"1h"``).
        initial_k : float or array_like
            Initial value of the raw ``k`` (responsiveness) parameter.
        initial_k_per_day : float or array_like
            Initial ``k`` expressed in per-day units (converted to
            per-chunk internally).
        initial_log_amplitude : float
            Initial log2-amplitude for channel-based rules.
        initial_memory_length : float
            Initial EWMA memory length in days.
        initial_memory_length_delta : float
            Initial delta added to memory length for secondary EWMAs.
        initial_pool_value : float
            Notional USD value of the pool at initialisation.
        initial_raw_exponents : float
            Initial raw (pre-squareplus) exponent for power-law rules.
        initial_raw_width : float
            Initial raw (pre-transform) channel width.
        initial_weights_logits : array_like
            Initial weight logits (softmax inputs) for each asset.
        maximum_change : float
            Per-step cap on absolute weight change (guardrail).
        optimisation_settings : dict
            Nested dictionary of optimiser hyperparameters (learning
            rate, schedule, gradient clipping, etc.).
        return_val : str
            Name of the financial objective used as the training loss
            (e.g. ``"sharpe"``, ``"calmar"``).
        rule : str
            Pool rule identifier string, as accepted by
            ``create_pool``.
        startDateString : str
            Start date of the in-sample training window, ISO-8601
            format.
        startDateUnix : int
            Unix timestamp corresponding to ``startDateString``.
        subsidary_pools : list of str or None
            Optional list of subsidiary pool rule identifiers used for
            comparison hooks (LVR / RVR).
        tokens : list of str
            Ordered list of token ticker symbols in the pool.
        use_alt_lamb : bool
            Whether to use the alternative lambda parameterisation.
        use_pre_exp_scaling : bool
            Whether the ``pre_exp_scaling`` parameter is active
            (relevant for triple-threat and channel rules).
        weight_interpolation_period : int
            Number of timesteps over which weight changes are
            interpolated within each chunk.
        """
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
