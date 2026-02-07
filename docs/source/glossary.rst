Glossary
========

Key terms and concepts used throughout the quantammsim documentation. Terms are
cross-referenced where relevant; click any :term: link to jump to its definition.

.. glossary::
   :sorted:

   AMM (Automated Market Maker)
      A smart-contract protocol that holds reserves of two or more tokens and
      algorithmically prices trades via a trading function, removing the need for
      a traditional order book. See also :term:`CFMM (Constant Function Market Maker)`.

   G3M (Generalised Mean Market Maker)
      A family of :term:`AMMs <AMM (Automated Market Maker)>` whose trading function
      is defined by a weighted power mean of token reserves. Balancer-style pools are
      the canonical example. Token weights control the pool's price exposure to each asset.

   CFMM (Constant Function Market Maker)
      An :term:`AMM <AMM (Automated Market Maker)>` defined by a trading invariant
      :math:`f(R_1, \ldots, R_n) = k` that is conserved across all trades. :term:`G3M
      <G3M (Generalised Mean Market Maker)>` pools are a special case.

   Arbitrage
   Arb
      The process by which an external agent exploits price discrepancies between
      the pool's implied price and the external market price. In quantammsim the
      arbitrageur is modelled as an optimal rational actor who trades to align pool
      prices with the external oracle, subject to gas costs and fees.

   Reserves
      The quantities of each token held by the pool. Reserves change when trades
      (including :term:`arbitrage <Arbitrage>`) occur and determine the pool's
      implied marginal price.

   Pool Value
      The total value of the pool's :term:`reserves <Reserves>`, denominated in a
      chosen numeraire (typically the last token). This is the primary quantity
      tracked for financial performance.

   LP (Liquidity Provider)
      An agent who deposits tokens into a pool in exchange for a proportional claim
      on the pool's :term:`reserves <Reserves>`. LP returns are the core performance
      metric that training seeks to maximise.

   CoW AMM (Coincidence of Wants AMM)
      A pool design in which post-trade quoted prices must match the trade execution
      price, preventing sandwich attacks by construction. Implemented in the CowAMM
      DeFi protocol.

   ECLP (Elliptic Concentrated Liquidity Pool)
      A pool type (as used by Gyroscope) whose trading function is an elliptic curve,
      providing concentrated liquidity within user-specified price bounds while
      retaining a smooth, differentiable trading function and fungible LP positions.

   Coarse Weights
      The target portfolio weights computed by the :term:`update rule <Update Rule>` at
      each :term:`chunk period <Chunk Period>` boundary. These are the direct output of
      the strategy before any interpolation to finer time resolution.

   Fine Weights
      Weights interpolated from :term:`coarse weights <Coarse Weights>` to
      minute-level resolution. The interpolation method (linear or optimal) determines
      how the pool transitions between coarse weight updates. See
      :term:`Weight Interpolation`.

   Weight Interpolation
      The process of smoothly transitioning between consecutive :term:`coarse weight
      <Coarse Weights>` updates to produce :term:`fine weights <Fine Weights>`.
      Linear interpolation spreads the change uniformly; optimal interpolation
      minimises tracking error. The interpolation granularity is controlled by
      :term:`Weight Interpolation Period`.

   STE (Straight-Through Estimator)
      A gradient estimation technique that passes gradients through non-differentiable
      clipping operations unchanged. Used in quantammsim to allow backpropagation
      through :term:`maximum change <Maximum Change>` enforcement and min/max weight
      bound clipping, which would otherwise have zero gradients almost everywhere.

   Maximum Change
      The maximum permitted weight change per :term:`update rule <Update Rule>` step.
      Prevents extreme rebalancing that could be exploited on-chain or indicate
      numerical instability. Enforced via clipping with :term:`STE (Straight-Through
      Estimator)` to preserve gradient flow.

   Update Rule
      The parameterised function that maps price data (via :term:`EWMA
      <EWMA (Exponentially Weighted Moving Average)>` estimators and other features)
      to target portfolio weights. Examples include momentum, anti-momentum, mean
      reversion channel, and power channel rules. Each rule exposes a set of
      learnable parameters (e.g. :term:`lambda <Lambda>`, :term:`k <k (k_per_day)>`,
      :term:`amplitude <Amplitude>`).

   EWMA (Exponentially Weighted Moving Average)
      The core estimator underlying most :term:`update rules <Update Rule>`. Computes a
      recursive moving average of price data with exponential decay controlled by
      :term:`lambda <Lambda>`. Implemented as a JAX scan for the CPU path and as a
      causal convolution for the GPU path.

   Lambda
      The decay parameter of the :term:`EWMA <EWMA (Exponentially Weighted Moving
      Average)>` estimator. Defined as ``lambda = sigmoid(logit_lambda)`` so that
      optimisation operates in unconstrained logit space. Higher values produce longer
      effective :term:`memory <Memory Length>`.

   Memory Length
      The effective lookback period (in days) of an :term:`EWMA <EWMA (Exponentially
      Weighted Moving Average)>` estimator, determined by :term:`lambda <Lambda>`. The
      conversion is handled by ``memory_days_to_logit_lamb()`` and its inverse.
      Capped at ``max_memory_days`` (default 365) in the :term:`run fingerprint
      <Run Fingerprint>`.

   k (k_per_day)
      Strategy aggressiveness parameter controlling how strongly the pool reacts to
      signals from the :term:`update rule <Update Rule>`. Larger values produce
      more concentrated weight tilts and higher turnover.

   Amplitude
      A log-space scaling factor applied to strategy signals before they are converted
      to weights. Provides an additional degree of freedom for controlling the
      magnitude of weight deviations from the base allocation.

   Channel Width
      The width parameter for mean reversion channel strategies. Defines the
      threshold between the mean-reverting regime (small price moves) and the
      trend-following regime (large price moves).

   Run Fingerprint
      The master configuration dictionary specifying all settings for a simulation or
      training run: tokens, :term:`update rule <Update Rule>`, date range,
      :term:`chunk period <Chunk Period>`, fees, gas costs, :term:`return metric
      <Return Metric>`, and all robustness settings. Serialisable to JSON for
      reproducibility.

   Forward Pass
      A single end-to-end evaluation of the simulation pipeline: price data is fed
      through :term:`EWMA <EWMA (Exponentially Weighted Moving Average)>` estimators,
      the :term:`update rule <Update Rule>` produces :term:`coarse weights
      <Coarse Weights>`, these are interpolated to :term:`fine weights
      <Fine Weights>`, :term:`arbitrage <Arbitrage>` is simulated, and financial
      metrics are computed. In training, the :term:`return metric <Return Metric>`
      from the forward pass is differentiated to update strategy parameters.

   Bout Offset
      A random offset (in minutes) applied to shorten the start of each training
      window within a batch. Provides data diversity during SGD by ensuring that
      different gradient steps see slightly different sub-windows of the training
      period.

   Batch Size
      The number of randomly-sampled time windows evaluated per gradient computation.
      Each window in the batch uses an independent :term:`bout offset <Bout Offset>`.
      Larger batches reduce gradient variance at the cost of memory and compute.

   Parameter Set
      One complete set of strategy parameters (e.g. :term:`lambda <Lambda>`,
      :term:`k <k (k_per_day)>`, :term:`amplitude <Amplitude>`). Multiple parameter
      sets are trained in parallel via ``n_parameter_sets`` for
      :term:`ensemble training <Ensemble Training>`.

   Return Metric
   return_val
      The scalar objective function that training maximises (or whose negation is
      minimised). Common choices include :term:`daily_log_sharpe`, ``sharpe``,
      ``calmar``, and ``returns_over_hodl``. Specified in the :term:`run fingerprint
      <Run Fingerprint>` via the ``return_val`` key.

   daily_log_sharpe
      The default training objective: the Sharpe ratio computed on daily log-returns
      of :term:`pool value <Pool Value>`. Preferred over raw Sharpe because
      log-returns are additive, better-behaved under differentiation, and less
      susceptible to outlier-driven gradients.

   Walk-Forward Analysis (WFA)
      An out-of-sample validation methodology in which the historical data is divided
      into sequential :term:`in-sample <In-Sample (IS)>` training windows and
      subsequent :term:`out-of-sample <Out-of-Sample (OOS)>` test windows. The
      strategy is retrained on each IS window and evaluated on the corresponding OOS
      window. Provides a realistic estimate of live performance (Pardo, 2008).

   Walk-Forward Efficiency (WFE)
      The ratio of :term:`out-of-sample <Out-of-Sample (OOS)>` to
      :term:`in-sample <In-Sample (IS)>` performance across walk-forward folds.
      WFE > 0.5 is generally taken as evidence that the strategy captures genuine
      structure rather than overfitting to the training period.

   In-Sample (IS)
      The training period within a single :term:`walk-forward <Walk-Forward Analysis
      (WFA)>` cycle, on which strategy parameters are optimised.

   Out-of-Sample (OOS)
      The held-out test period immediately following the :term:`in-sample
      <In-Sample (IS)>` window in a :term:`walk-forward <Walk-Forward Analysis (WFA)>`
      cycle. Performance on this period is never used for parameter updates.

   IS-OOS Gap
      The difference between :term:`in-sample <In-Sample (IS)>` and
      :term:`out-of-sample <Out-of-Sample (OOS)>` performance. A large gap is the
      primary diagnostic for overfitting. Robustness features such as
      :term:`ensemble training <Ensemble Training>`, :term:`price noise
      <Price Noise>`, and :term:`early stopping <Early Stopping>` all aim to
      reduce this gap.

   Rademacher Complexity
      An empirical measure of the effective complexity of the strategy search space.
      Estimated by evaluating the best training objective achievable on data with
      randomly flipped return signs; a high value indicates the search can fit
      noise, signalling overfitting risk from the optimisation process itself
      (Paleologo, 2024).

   Ensemble Training
      Training multiple :term:`parameter sets <Parameter Set>` simultaneously and
      averaging their weight outputs at inference time. Provides implicit
      regularisation by reducing variance across the ensemble, analogous to
      bagging. Initialisation strategies (LHS, Sobol, grid) control the diversity
      of the initial parameter sets.

   Early Stopping
      Terminating training when a monitored validation metric (typically
      :term:`out-of-sample <Out-of-Sample (OOS)>` performance or a moving average
      of the training loss) stops improving. Prevents the optimiser from continuing
      into the overfitting regime.

   SWA (Stochastic Weight Averaging)
      A technique that averages parameter snapshots from later training epochs,
      producing a solution that lies in a flatter region of the loss landscape.
      Flat minima tend to generalise better, reducing the :term:`IS-OOS gap
      <IS-OOS Gap>`.

   Price Noise
      Multiplicative log-normal noise applied to training prices as a data
      augmentation strategy. Forces the strategy to be robust to small price
      perturbations, acting as a regulariser that discourages reliance on
      exact historical price paths.

   Turnover Penalty
      A regularisation term added to the training objective that penalises excessive
      weight changes between consecutive :term:`coarse weight <Coarse Weights>`
      updates. Encourages smoother strategies that are cheaper to execute on-chain
      and less prone to overfitting via high-frequency signal chasing.

   Chunk Period
      The time interval (in minutes) between consecutive :term:`coarse weight
      <Coarse Weights>` updates. The default of 1440 corresponds to daily updates.
      Shorter chunk periods allow faster strategy response but increase turnover
      and computational cost.

   Weight Interpolation Period
      The time interval (in minutes) between :term:`fine weight <Fine Weights>`
      evaluations. Usually set equal to :term:`chunk period <Chunk Period>` but can
      be made finer for higher-resolution interpolation.

   Arb Frequency
      How often (in minutes) the simulator checks for and executes :term:`arbitrage
      <Arbitrage>` opportunities. Finer arb frequency produces more realistic
      simulations but increases computation. Typically set to match the data
      resolution (e.g. 1 for minute-level data).

   Hook
      A mixin class that modifies pool behaviour by composing with the base pool via
      Python's MRO. Hooks are specified using double-underscore syntax in the
      :term:`run fingerprint <Run Fingerprint>` (e.g. ``bounded_weight__momentum``)
      and can implement custom fee logic, weight bounds, ensemble averaging, or
      performance tracking. Multiple hooks compose left-to-right.

   Burn-in
      The initial period of :term:`memory length <Memory Length>` (up to
      ``max_memory_days``) before the training window during which :term:`EWMA
      <EWMA (Exponentially Weighted Moving Average)>` estimators accumulate
      history but no gradients are computed and no performance metrics are recorded.
      Ensures that estimator state is warm before evaluation begins.
