"""Compare modelled reClAMM noise volume against a real pool's observed volume.

Plots the modelled noise volume for one pool (at a specified counterfactual TVL)
against the actual observed volume of another pool (or the same pool), to
sanity-check the noise model's predictions.

Usage:
  # reClAMM AAVE/ETH modelled at $7M vs weighted wstETH/AAVE real
  python scripts/compare_modelled_vs_real.py \
      --model-pool 0x9d1fcf346ea1b0 --model-tvl 7e6 \
      --real-pool 0x3de27efa2f1aa6

  # Same but at $20M
  python scripts/compare_modelled_vs_real.py \
      --model-pool 0x9d1fcf346ea1b0 --model-tvl 20e6 \
      --real-pool 0x3de27efa2f1aa6

  # Multiple TVL levels
  python scripts/compare_modelled_vs_real.py \
      --model-pool 0x9d1fcf346ea1b0 --model-tvl 1e6 7e6 20e6 50e6 \
      --real-pool 0x3de27efa2f1aa6
"""

import argparse
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quantammsim.calibration.noise_model_arrays import (
    build_simulator_arrays, load_artifact, _find_pool_index,
)


CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "results", "token_factored_calibration", "_cache",
)
ARTIFACT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "results", "linear_market_noise",
)
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "results", "noise_comparison",
)

# Token mapping for pools
POOL_TOKENS = {
    "0x9d1fcf346ea1b0": ("AAVE", "ETH"),
    "0x3de27efa2f1aa6": ("AAVE", "ETH"),  # wstETH/AAVE ≈ same pair
    "0x0b09dea16768f0": ("DAI", "ETH"),
    "0xa6f548df93de92": ("BTC", "ETH"),
    "0x96646936b91d6b": ("USDC", "ETH"),
}


def load_real_pool(pid, mc):
    """Load real observed volume + TVL for a pool."""
    entry = mc[pid]
    panel = entry["panel"]
    dates = pd.to_datetime(panel["date"])
    vol = np.exp(panel["log_volume"].values.astype(float))
    tvl = np.exp(panel["log_tvl_lag1"].values.astype(float))
    tokens = entry["tokens"]
    chain = entry["chain"]
    return dates, vol, tvl, tokens, chain


def compute_modelled_noise(pid, tvl_value, start_date, end_date,
                           artifact_dir, token_a, token_b):
    """Compute modelled daily noise for a pool at a given TVL."""
    arrays = build_simulator_arrays(
        token_a=token_a, token_b=token_b,
        start_date=start_date, end_date=end_date,
        artifact_dir=artifact_dir, pool_id=pid,
    )

    n_days = arrays["n_days"]
    std_lt = (np.log(tvl_value) - arrays["tvl_mean"]) / arrays["tvl_std"]

    noise_base = arrays["noise_base"][::1440][:n_days]
    tvl_coeff = arrays["noise_tvl_coeff"][::1440][:n_days]
    noise_daily = np.exp(noise_base + tvl_coeff * std_lt)

    dates = pd.to_datetime(arrays["dates"][:n_days])
    return dates, noise_daily


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-pool", default="0x9d1fcf346ea1b0",
                        help="Pool ID for modelled noise")
    parser.add_argument("--model-tvl", type=float, nargs="+",
                        default=[7_000_000],
                        help="Counterfactual TVL(s) for the modelled pool")
    parser.add_argument("--model-tokens", nargs=2, default=None,
                        help="Token A and B for the modelled pool (auto-detected)")
    parser.add_argument("--real-pool", default="0x3de27efa2f1aa6",
                        help="Pool ID for real observed data")
    parser.add_argument("--artifact-dir", default=ARTIFACT_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load calibration data
    with open(os.path.join(CACHE_DIR, "stage1.pkl"), "rb") as f:
        data = pickle.load(f)
    mc = data["matched_clean"]

    # Real pool data
    real_dates, real_vol, real_tvl, real_tokens, real_chain = load_real_pool(
        args.real_pool, mc)
    print(f"Real pool: {args.real_pool} ({real_tokens}, {real_chain})")
    print(f"  {len(real_dates)} days: {real_dates.min().date()} → {real_dates.max().date()}")
    print(f"  TVL: ${real_tvl.min():,.0f} – ${real_tvl.max():,.0f}")
    print(f"  Volume: ${real_vol.min():,.0f} – ${real_vol.max():,.0f}")

    # Model tokens
    if args.model_tokens:
        tok_a, tok_b = args.model_tokens
    elif args.model_pool[:16] in POOL_TOKENS:
        tok_a, tok_b = POOL_TOKENS[args.model_pool[:16]]
    else:
        tok_a, tok_b = "ETH", "USDC"
        print(f"  Warning: unknown pool, using {tok_a}/{tok_b}")

    # Date range from real pool
    start = str(real_dates.min().date())
    end = str(real_dates.max().date())

    # Compute modelled noise at each TVL
    model_results = []
    for tvl_val in args.model_tvl:
        print(f"\nModelled: {args.model_pool} at ${tvl_val:,.0f} TVL")
        m_dates, m_noise = compute_modelled_noise(
            args.model_pool, tvl_val, start, end,
            args.artifact_dir, tok_a, tok_b)
        model_results.append((tvl_val, m_dates, m_noise))
        print(f"  Median noise: ${np.median(m_noise):,.0f}/day"
              f"  ({np.median(m_noise)/tvl_val*100:.2f}% of TVL)")

    # Align dates
    common_start = real_dates.min()
    common_end = real_dates.max()
    for _, md, _ in model_results:
        common_start = max(common_start, md.min())
        common_end = min(common_end, md.max())

    real_mask = (real_dates >= common_start) & (real_dates <= common_end)

    # Colors for different TVL levels
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))

    # 1. Volume comparison
    ax = axes[0]
    ax.plot(real_dates[real_mask], real_vol[real_mask] / 1e6,
            "k-", linewidth=0.8, alpha=0.7,
            label=f"{real_tokens} weighted (real,"
                  f" TVL ${np.median(real_tvl[real_mask])/1e6:.0f}M)")

    for i, (tvl_val, m_dates, m_noise) in enumerate(model_results):
        m_mask = (m_dates >= common_start) & (m_dates <= common_end)
        c = colors[i % len(colors)]
        ax.plot(m_dates[m_mask], m_noise[m_mask] / 1e6,
                "-", color=c, linewidth=0.8, alpha=0.7,
                label=f"reClAMM noise (modelled, TVL ${tvl_val/1e6:.0f}M)")

    ax.set_ylabel("Volume ($M/day)")
    ax.set_yscale("log")
    ax.set_title(f"Real weighted pool vs Modelled reClAMM noise")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. Vol/TVL comparison
    ax = axes[1]
    real_vol_tvl = real_vol[real_mask] / real_tvl[real_mask] * 100
    ax.plot(real_dates[real_mask], real_vol_tvl,
            "k-", linewidth=0.8, alpha=0.7,
            label=f"{real_tokens} weighted real vol/TVL")
    ax.axhline(np.median(real_vol_tvl), color="black", linestyle="--",
               alpha=0.3, label=f"weighted median: {np.median(real_vol_tvl):.2f}%")

    for i, (tvl_val, m_dates, m_noise) in enumerate(model_results):
        m_mask = (m_dates >= common_start) & (m_dates <= common_end)
        noise_tvl = m_noise[m_mask] / tvl_val * 100
        c = colors[i % len(colors)]
        ax.plot(m_dates[m_mask], noise_tvl,
                "-", color=c, linewidth=0.8, alpha=0.7,
                label=f"reClAMM noise/TVL (${tvl_val/1e6:.0f}M)")
        ax.axhline(np.median(noise_tvl), color=c, linestyle="--", alpha=0.3,
                   label=f"median: {np.median(noise_tvl):.2f}%")

    ax.set_ylabel("Volume / TVL (%)")
    ax.set_xlabel("Date")
    ax.set_title("Volume as Fraction of TVL")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)
    ymax = min(
        max(np.percentile(real_vol_tvl, 95),
            max(np.percentile(m_noise[m_mask] / tvl_val * 100, 95)
                for tvl_val, m_dates, m_noise in model_results
                for m_mask in [(m_dates >= common_start) & (m_dates <= common_end)])) * 1.5,
        50)
    ax.set_ylim(0, ymax)

    fig.tight_layout()
    tvl_str = "_".join(f"{t/1e6:.0f}M" for t in args.model_tvl)
    out = os.path.join(args.output_dir,
                       f"{args.model_pool[:8]}_vs_{args.real_pool[:8]}_{tvl_str}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")

    # Summary table
    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"  Real {real_tokens} weighted:")
    print(f"    Median TVL: ${np.median(real_tvl[real_mask]):,.0f}")
    print(f"    Median vol: ${np.median(real_vol[real_mask]):,.0f}/day")
    print(f"    Median vol/TVL: {np.median(real_vol_tvl):.2f}%")
    for tvl_val, m_dates, m_noise in model_results:
        m_mask = (m_dates >= common_start) & (m_dates <= common_end)
        med_noise = np.median(m_noise[m_mask])
        print(f"  Modelled reClAMM at ${tvl_val/1e6:.0f}M:")
        print(f"    Median noise: ${med_noise:,.0f}/day")
        print(f"    Median noise/TVL: {med_noise/tvl_val*100:.2f}%")


if __name__ == "__main__":
    main()
