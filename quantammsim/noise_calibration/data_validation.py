"""Data validation for noise calibration panels."""

import numpy as np
import pandas as pd


def validate_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Run data validation checks. Prints warnings but does NOT drop rows."""
    print("\n  Data validation:")

    # Pools with constant volume
    vol_std = panel.groupby("pool_id")["log_volume"].std()
    constant_vol = vol_std[vol_std < 0.01]
    if len(constant_vol) > 0:
        print(f"    WARNING: {len(constant_vol)} pools have near-constant "
              f"log(volume) (std < 0.01)")
        for pid in constant_vol.index[:5]:
            print(f"      {pid[:16]}... std={constant_vol[pid]:.4f}")
        if len(constant_vol) > 5:
            print(f"      ... and {len(constant_vol) - 5} more")

    # TVL jumps > 10x between consecutive days
    panel_sorted = panel.sort_values(["pool_id", "date"])
    tvl_ratio = panel_sorted.groupby("pool_id")["log_tvl"].diff().abs()
    big_jumps = tvl_ratio[tvl_ratio > np.log(10)]
    if len(big_jumps) > 0:
        affected_pools = panel_sorted.loc[big_jumps.index, "pool_id"].nunique()
        print(f"    WARNING: {len(big_jumps)} TVL jumps > 10x across "
              f"{affected_pools} pools")

    # Days where volume > TVL
    high_vol = panel[panel["log_volume"] > panel["log_tvl"]]
    if len(high_vol) > 0:
        affected_pools = high_vol["pool_id"].nunique()
        print(f"    WARNING: {len(high_vol)} days where volume > TVL across "
              f"{affected_pools} pools (potential wash trading)")

    if len(constant_vol) == 0 and len(big_jumps) == 0 and len(high_vol) == 0:
        print("    All checks passed.")

    return panel
