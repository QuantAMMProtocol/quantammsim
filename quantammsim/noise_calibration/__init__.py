"""Noise calibration package for Balancer pool volume models.

Public API re-exports from submodules.
"""

# scipy.signal patch (must run before arviz import)
try:
    from scipy.signal import gaussian as _  # noqa: F401
except ImportError:
    from scipy.signal.windows import gaussian as _gauss
    import scipy.signal
    scipy.signal.gaussian = _gauss

from .constants import (
    K_COEFF, COEFF_NAMES, BALANCER_API_URL, BALANCER_API_CHAINS, CACHE_DIR,
    K_CLUSTERS_DEFAULT, K_FEATURES_DEFAULT,
)
from .token_classification import classify_token_tier, _normalise_symbol
from .data_pipeline import (
    _graphql_request, enumerate_balancer_pools, fetch_pool_snapshots,
    fetch_all_snapshots, fetch_token_prices, compute_pair_volatility,
    assemble_panel,
)
from .data_validation import validate_panel
from .covariate_encoding import encode_covariates, encode_covariates_structural
from .model import noise_model, noise_model_dp_sigma, noise_model_ibp, noise_model_ibp_dp, stick_breaking_weights, structural_noise_model
from .formula_arb import formula_arb_volume_daily_jax
from .inference import (
    _get_theta_samples, _build_model_kwargs, run_svi, run_nuts,
    run_svi_then_nuts,
)
from .postprocessing import (
    extract_noise_params, predict_new_pool, check_convergence,
    run_prior_predictive, assign_dp_clusters, assign_ibp_dp_joint,
    extract_structural_params, predict_new_pool_structural,
)
from .plotting import plot_diagnostics
from .output import generate_output_json, _save_sample_cache
from .cli import main
