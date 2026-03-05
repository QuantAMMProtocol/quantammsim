"""Compare geometric vs constant-arc-length thermostats on historic data.

Runs AAVE/ETH reClAMM pool simulations with both interpolation methods.
Plots: pool value, cumulative LVR, price path, empirical weights,
value difference, LVR ratio, and per-step LVR distribution (∝ Δs²).

Usage:
    cd <repo-root>
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate qsim-reclamm
    python scripts/compare_reclamm_thermostats.py
"""

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from quantammsim.runners.jax_runners import do_run_on_historic_data


def to_daily_price_shift_base(daily_price_shift_exponent):
    """Convert shift rate to daily price shift base (matches Solidity)."""
    return 1.0 - daily_price_shift_exponent / 124649.0


# Pool configurations to compare
CONFIGS = [
    {
        "name": "AAVE/ETH on-chain (25bps, narrow range)",
        "tokens": ["AAVE", "ETH"],
        "start": "2024-06-01 00:00:00",
        "end": "2025-06-01 00:00:00",
        "fees": 0.0025,
        "price_ratio": 1.5,
        "centeredness_margin": 0.5,
        "daily_price_shift_exponent": 0.1,
    },
    {
        "name": "AAVE/ETH wide range (25bps)",
        "tokens": ["AAVE", "ETH"],
        "start": "2024-06-01 00:00:00",
        "end": "2025-06-01 00:00:00",
        "fees": 0.0025,
        "price_ratio": 4.0,
        "centeredness_margin": 0.2,
        "daily_price_shift_exponent": 1.0,
    },
    {
        "name": "AAVE/ETH zero fees (narrow)",
        "tokens": ["AAVE", "ETH"],
        "start": "2024-06-01 00:00:00",
        "end": "2025-06-01 00:00:00",
        "fees": 0.0,
        "price_ratio": 1.5,
        "centeredness_margin": 0.5,
        "daily_price_shift_exponent": 0.1,
    },
]


def make_fingerprint(cfg, interpolation_method, centeredness_scaling=False):
    """Build run fingerprint for a given config and interpolation method."""
    return {
        "tokens": cfg["tokens"],
        "rule": "reclamm",
        "startDateString": cfg["start"],
        "endDateString": cfg["end"],
        "initial_pool_value": 1000000.0,
        "do_arb": True,
        "fees": cfg["fees"],
        "gas_cost": 0.0,
        "arb_fees": 0.0,
        "reclamm_interpolation_method": interpolation_method,
        "reclamm_arc_length_speed": None,  # auto-calibrate
        "reclamm_centeredness_scaling": centeredness_scaling,
    }


def make_params(cfg):
    """Build pool params from config."""
    return {
        "price_ratio": jnp.array(cfg["price_ratio"]),
        "centeredness_margin": jnp.array(cfg["centeredness_margin"]),
        "daily_price_shift_base": jnp.array(
            to_daily_price_shift_base(cfg["daily_price_shift_exponent"])
        ),
    }


def run_comparison(cfg):
    """Run all thermostat variants, return results dict."""
    params = make_params(cfg)

    results = {}
    for method in ["geometric", "constant_arc_length"]:
        fp = make_fingerprint(cfg, method)
        results[method] = do_run_on_historic_data(
            run_fingerprint=fp, params=params
        )

    # Geometric + centeredness-proportional scaling (scales decay duration)
    fp_geo_scaled = make_fingerprint(cfg, "geometric", centeredness_scaling=True)
    results["geometric_scaled"] = do_run_on_historic_data(
        run_fingerprint=fp_geo_scaled, params=params
    )

    # Arc-length + centeredness-proportional scaling (scales speed)
    fp_cal_scaled = make_fingerprint(cfg, "constant_arc_length", centeredness_scaling=True)
    results["cal_scaled"] = do_run_on_historic_data(
        run_fingerprint=fp_cal_scaled, params=params
    )

    return results


def print_comparison(cfg, results):
    """Print text summary table."""
    methods = [
        ("Geometric", results["geometric"]),
        ("Geo+Scaled", results["geometric_scaled"]),
        ("Const Arc", results["constant_arc_length"]),
        ("Arc+Scaled", results["cal_scaled"]),
    ]

    hodl_value = float((methods[0][1]["reserves"][0] * methods[0][1]["prices"][-1]).sum())

    print("=" * 105)
    print(f"  {cfg['name']}")
    print(f"  price_ratio={cfg['price_ratio']}, "
          f"margin={cfg['centeredness_margin']}, "
          f"shift_exp={cfg['daily_price_shift_exponent']}, "
          f"fees={cfg['fees']}")
    print("-" * 105)
    header = "  {:20s}".format("")
    for name, _ in methods:
        header += f" {name:>14s}"
    print(header)

    row = "  {:20s}".format("Final value")
    for _, r in methods:
        row += f" ${float(r['final_value']):>13,.0f}"
    print(row)

    print(f"  {'HODL value':20s} ${hodl_value:>13,.0f}")

    row = "  {:20s}".format("LVR (HODL - final)")
    for _, r in methods:
        lvr = hodl_value - float(r["final_value"])
        row += f" ${lvr:>13,.0f}"
    print(row)

    row = "  {:20s}".format("Return")
    for _, r in methods:
        ret = (float(r["final_value"]) / float(r["value"][0]) - 1) * 100
        row += f" {ret:>13.2f}%"
    print(row)

    row = "  {:20s}".format("vs HODL")
    for _, r in methods:
        vs = (float(r["final_value"]) / hodl_value - 1) * 100
        row += f" {vs:>13.2f}%"
    print(row)
    print("=" * 105)


def plot_comparison(cfg, results, fig_idx):
    """Plot 4-panel comparison for one config."""
    # Method name → (result dict, color, linestyle)
    variants = {
        "Geometric": (results["geometric"], "C0", "-"),
        "Geo+Scaled": (results["geometric_scaled"], "C1", "-"),
        "Const arc-len": (results["constant_arc_length"], "C2", "--"),
        "Arc+Scaled": (results["cal_scaled"], "C3", "--"),
    }

    geo = results["geometric"]
    geo_prices = np.array(geo["prices"])
    geo_reserves = np.array(geo["reserves"])
    n_steps = len(np.array(geo["value"]))
    t_days = np.arange(n_steps) / (60 * 24)

    hodl_traj = (geo_reserves[0] * geo_prices[:n_steps]).sum(axis=-1)
    price_ratio_traj = geo_prices[:n_steps, 0] / geo_prices[:n_steps, 1]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(cfg["name"], fontsize=13, fontweight="bold")

    # (0,0) Pool value over time
    ax = axes[0, 0]
    for name, (r, color, ls) in variants.items():
        vals = np.array(r["value"])
        ax.plot(t_days, vals / 1e6, color=color, ls=ls, label=name, alpha=0.9)
    ax.plot(t_days, np.array(hodl_traj) / 1e6, color="gray", ls=":",
            alpha=0.5, label="HODL")
    ax.set_xlabel("Days")
    ax.set_ylabel("Pool value ($M)")
    ax.set_title("Pool value")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (0,1) Cumulative LVR
    ax = axes[0, 1]
    for name, (r, color, ls) in variants.items():
        vals = np.array(r["value"])
        lvr = np.array(hodl_traj) - vals
        ax.plot(t_days, lvr / 1e3, color=color, ls=ls, label=name, alpha=0.9)
    ax.set_xlabel("Days")
    ax.set_ylabel("Cumulative LVR ($K)")
    ax.set_title("Cumulative LVR (HODL - pool value)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,0) Price ratio
    ax = axes[1, 0]
    ax.plot(t_days, price_ratio_traj, color="C4", alpha=0.7)
    ax.set_xlabel("Days")
    ax.set_ylabel(f"{cfg['tokens'][0]}/{cfg['tokens'][1]} price ratio")
    ax.set_title("Price path")
    ax.grid(True, alpha=0.3)

    # (1,1) Empirical weights
    ax = axes[1, 1]
    for name, (r, color, ls) in variants.items():
        w = np.array(r["weights"])
        n_w = min(len(w), n_steps)
        t_w = np.arange(n_w) / (60 * 24)
        ax.plot(t_w, w[:n_w, 0], color=color, ls=ls, label=name, alpha=0.9)
    ax.set_xlabel("Days")
    ax.set_ylabel(f"Weight ({cfg['tokens'][0]})")
    ax.set_title("Empirical weight (token 0)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f"reclamm_thermostat_comparison_{fig_idx}.png"
    plt.savefig(fname, dpi=150)
    print(f"Saved {fname}")
    plt.close(fig)

    # Second figure: diagnostics
    geo_values = np.array(geo["value"])
    geo_lvr = np.array(hodl_traj) - geo_values

    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
    fig2.suptitle(f"{cfg['name']} — diagnostics", fontsize=13, fontweight="bold")

    # (left) Value difference vs geometric
    ax = axes2[0]
    for name, (r, color, ls) in variants.items():
        if name == "Geometric":
            continue
        vals = np.array(r["value"])
        ax.plot(t_days, (vals - geo_values) / 1e3, color=color, ls=ls,
                label=name, alpha=0.9)
    ax.axhline(0, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("Days")
    ax.set_ylabel("Value difference ($K)")
    ax.set_title("Minus Geometric")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (middle) LVR ratio over time
    ax = axes2[1]
    mask = np.abs(geo_lvr) > 100
    if mask.any():
        for name, (r, color, ls) in variants.items():
            if name == "Geometric":
                continue
            vals = np.array(r["value"])
            method_lvr = np.array(hodl_traj) - vals
            ratio = np.full_like(geo_lvr, np.nan)
            ratio[mask] = method_lvr[mask] / geo_lvr[mask]
            ax.plot(t_days, ratio, color=color, ls=ls, alpha=0.7, label=name)
        ax.axhline(1.0, color="gray", ls="--", alpha=0.5)
        ax.set_ylabel("LVR ratio (method / geometric)")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "LVR too small to compare",
                transform=ax.transAxes, ha="center", va="center")
    ax.set_xlabel("Days")
    ax.set_title("Relative LVR")
    ax.grid(True, alpha=0.3)

    # (right) Per-step LVR histogram
    ax = axes2[2]
    all_pos = []
    for name, (r, color, ls) in variants.items():
        vals = np.array(r["value"])
        method_lvr = np.array(hodl_traj) - vals
        step_lvr = np.diff(method_lvr)
        pos = step_lvr[step_lvr > 0]
        all_pos.append((name, pos, color))
    has_data = [len(p) > 10 for _, p, _ in all_pos]
    if any(has_data):
        max_val = max(np.percentile(p, 99) for _, p, _ in all_pos if len(p) > 10)
        bins = np.linspace(0, max_val, 50)
        for name, pos, color in all_pos:
            if len(pos) > 10:
                ax.hist(pos, bins=bins, color=color, alpha=0.3, label=name,
                        density=True)
        ax.set_xlabel("Per-step LVR ($)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "Too few thermostat steps",
                transform=ax.transAxes, ha="center", va="center")
    ax.set_title("Per-step LVR distribution")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname2 = f"reclamm_thermostat_diff_{fig_idx}.png"
    plt.savefig(fname2, dpi=150)
    print(f"Saved {fname2}")
    plt.close(fig2)


if __name__ == "__main__":
    all_results = []
    for i, cfg in enumerate(CONFIGS):
        print(f"\n>>> Running {cfg['name']}...")
        try:
            results = run_comparison(cfg)
            print_comparison(cfg, results)
            plot_comparison(cfg, results, i)
            all_results.append((cfg, results))
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    # Summary overlay: all configs on one figure (pool value normalised)
    if len(all_results) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        fig.suptitle("Cross-config comparison (normalised)", fontsize=13,
                     fontweight="bold")

        method_keys = [
            ("geometric", "geo", "-"),
            ("geometric_scaled", "geo+s", "-."),
            ("constant_arc_length", "arc", "--"),
            ("cal_scaled", "arc+s", ":"),
        ]

        for i, (cfg, results) in enumerate(all_results):
            geo_v = np.array(results["geometric"]["value"])
            t = np.arange(len(geo_v)) / (60 * 24)
            short_name = cfg["name"].split("(")[0].strip()

            for j, (key, suffix, ls) in enumerate(method_keys):
                v = np.array(results[key]["value"])
                color_idx = i * len(method_keys) + j

                # (left) Normalised pool value
                axes[0].plot(t, v / v[0], ls=ls, alpha=0.8,
                             label=f"{short_name} {suffix}",
                             color=f"C{color_idx % 10}")

                # (right) Value difference vs geometric (skip geo itself)
                if key != "geometric":
                    pct_diff = (v - geo_v) / geo_v * 100
                    axes[1].plot(t, pct_diff, ls=ls, alpha=0.8,
                                 label=f"{short_name} {suffix}",
                                 color=f"C{color_idx % 10}")

        axes[0].set_xlabel("Days")
        axes[0].set_ylabel("Normalised pool value")
        axes[0].set_title("Pool value (V/V0)")
        axes[0].legend(fontsize=6, ncol=2)
        axes[0].grid(True, alpha=0.3)

        axes[1].set_xlabel("Days")
        axes[1].set_ylabel("(Method - Geo) / Geo (%)")
        axes[1].set_title("Relative value difference vs Geometric")
        axes[1].axhline(0, color="gray", ls="--", alpha=0.5)
        axes[1].legend(fontsize=6, ncol=2)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("reclamm_thermostat_summary.png", dpi=150)
        print("\nSaved reclamm_thermostat_summary.png")
        plt.close(fig)
