#!/usr/bin/env python3
"""
audit_log_report.py

Reads all parquet files under ./audit_log (recursive) and produces:
- CSV reports under ./audit_reports
- PNG plots under ./audit_reports/plots

Rules implemented:
- Use flask_user as the user identifier everywhere.
- Metric column name is always 'count'.
- Per-day only reports: day,count
- Single-selector reports (day + X): pivot wide (rows=day, columns=X values, cells=count)
- Per-page reports: write separate CSVs (and plots) per page.
- "Per timezone per user" reports: pivot wide with MultiIndex columns (2-row header):
  top header row: timezone, second header row: flask_user; rows=day; values=count.
- For wide plots with many series: plot Top-N series by total sum (configurable).

Country/region inference:
- Best-effort timezone -> country using pytz if installed.
  (Accurate geolocation is not possible with hashed IP.)

Usage:
  pip install pandas pyarrow matplotlib pytz
  python audit_log_report.py --input ./audit_log --output ./audit_reports
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt

try:
    import pytz
except Exception:  # pragma: no cover
    pytz = None


# --------------------------
# Configuration
# --------------------------
USER_ID_COL = "flask_user"
INPUT_REQUIRED_COLUMNS = [
    "timestamp",
    "user",
    "page",
    "tosAgreement",
    "isMobile",
    "timezone",
    USER_ID_COL,
    "count",
]

TOP_N_SERIES = 12
FIGSIZE = (14, 6)
DPI = 140


# --------------------------
# Utility
# --------------------------
def sanitize_filename_component(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return "EMPTY"
    s = s.replace("/", "_").replace("\\", "_")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s[:180]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _safe_int_filename_day(path: str) -> Optional[str]:
    """
    If filename is like '1734489600.parquet', interpret it as Unix midnight.
    Treat as UTC for reporting consistency. Returns YYYY-MM-DD or None.
    """
    base = os.path.basename(path)
    name, _ext = os.path.splitext(base)
    if not name.isdigit():
        return None
    try:
        ts = int(name)
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        return dt.date().isoformat()
    except Exception:
        return None


def _day_from_timestamp_col(ts_series: pd.Series) -> pd.Series:
    """Fallback day derivation from 'timestamp' column (Unix seconds), interpreted as UTC."""
    s = pd.to_numeric(ts_series, errors="coerce")
    dt = pd.to_datetime(s, unit="s", utc=True, errors="coerce")
    return dt.dt.date.astype("string")


def load_all_parquets(input_dir: str) -> pd.DataFrame:
    pattern = os.path.join(input_dir, "**", "*.parquet")
    paths = sorted(glob.glob(pattern, recursive=True))
    if not paths:
        raise FileNotFoundError(f"No .parquet files found under: {input_dir}")

    frames: List[pd.DataFrame] = []
    for p in paths:
        df = pd.read_parquet(p, engine="pyarrow")

        # Ensure required columns exist
        for col in INPUT_REQUIRED_COLUMNS:
            if col not in df.columns:
                df[col] = pd.NA

        # Normalize count
        df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(1).astype("int64")

        # Derive day: prefer filename unix midnight else timestamp column
        day_from_name = _safe_int_filename_day(p)
        if day_from_name is not None:
            df["day"] = day_from_name
        else:
            df["day"] = _day_from_timestamp_col(df["timestamp"])

        df["source_file"] = os.path.basename(p)
        frames.append(df[INPUT_REQUIRED_COLUMNS + ["day", "source_file"]])

    out = pd.concat(frames, ignore_index=True)

    # Normalize types
    out["day"] = out["day"].astype("string")
    out["page"] = out["page"].astype("string")
    out["timezone"] = out["timezone"].astype("string")
    out[USER_ID_COL] = out[USER_ID_COL].astype("string")

    # Drop rows missing essentials
    out = out.dropna(subset=["day", USER_ID_COL])

    return out


def build_timezone_to_country_maps() -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    if pytz is None:
        return {}, {}
    tz_to_countries: Dict[str, List[str]] = {}
    for cc, tz_list in getattr(pytz, "country_timezones", {}).items():
        for tzname in tz_list:
            tz_to_countries.setdefault(tzname, []).append(cc)
    country_names = getattr(pytz, "country_names", {})
    return tz_to_countries, country_names


def infer_country_from_timezone(df: pd.DataFrame) -> pd.DataFrame:
    tz_to_countries, country_names = build_timezone_to_country_maps()
    if not tz_to_countries:
        df["inferred_country_name"] = pd.NA
        return df

    def _infer_name(tzname: str) -> Optional[str]:
        if not tzname or tzname in ("Unknown", "nan", "<NA>"):
            return None
        ccs = tz_to_countries.get(tzname, [])
        if not ccs:
            return None
        chosen = ccs[0]
        return country_names.get(chosen, chosen)

    df["inferred_country_name"] = df["timezone"].fillna("").astype(str).map(_infer_name)
    return df


def write_csv(df: pd.DataFrame, out_dir: str, filename: str) -> str:
    ensure_dir(out_dir)
    path = os.path.join(out_dir, filename)
    df.to_csv(path, index=False)
    return path


def day_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[["day", "count"]].sort_values("day")


def pivot_single_selector(df: pd.DataFrame, selector: str) -> pd.DataFrame:
    wide = (
        df.pivot_table(
            index="day",
            columns=selector,
            values="count",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
    )
    cols = ["day"] + sorted([c for c in wide.columns if c != "day"], key=lambda x: str(x))
    return wide[cols]


def pivot_timezone_user_multiheader(df: pd.DataFrame, tz_col: str = "timezone", user_col: str = USER_ID_COL) -> pd.DataFrame:
    """
    MultiIndex columns => CSV with 2-row header:
      row 1: timezone
      row 2: user
    """
    wide = df.pivot_table(
        index="day",
        columns=[tz_col, user_col],
        values="count",
        aggfunc="sum",
        fill_value=0,
    )
    wide = wide.sort_index(axis=1, level=[0, 1])
    return wide.reset_index()


# --------------------------
# Plotting (corrected / robust)
# --------------------------
def _parse_day_to_datetime(day_series: pd.Series) -> pd.Series:
    return pd.to_datetime(day_series.astype(str), errors="coerce")


def plot_day_only(df: pd.DataFrame, title: str, out_png: str) -> None:
    if df is None or df.empty:
        return
    if "day" not in df.columns or "count" not in df.columns:
        return

    x = _parse_day_to_datetime(df["day"])
    y = pd.to_numeric(df["count"], errors="coerce").fillna(0)

    mask = x.notna()
    if not mask.any():
        return

    plt.figure(figsize=FIGSIZE, dpi=DPI)
    plt.plot(x[mask], y[mask])
    plt.title(title)
    plt.xlabel("day")
    plt.ylabel("count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def plot_wide(df: pd.DataFrame, title: str, out_png: str, top_n: int = TOP_N_SERIES) -> None:
    """
    Works for:
      - normal wide: columns are strings
      - multiheader wide: columns are tuples (timezone, user)
    """
    if df is None or df.empty:
        return
    if "day" not in df.columns:
        return

    plot_df = df.copy()

    helper_col = "__day_dt__"
    plot_df[helper_col] = _parse_day_to_datetime(plot_df["day"])
    mask = plot_df[helper_col].notna()
    if not mask.any():
        return

    plot_df = plot_df.loc[mask].sort_values(helper_col)

    # series columns = everything except day/helper
    series_cols = [c for c in plot_df.columns if c not in ("day", helper_col)]
    if not series_cols:
        return

    # numeric
    for c in series_cols:
        plot_df[c] = pd.to_numeric(plot_df[c], errors="coerce").fillna(0)

    totals = plot_df[series_cols].sum(axis=0).sort_values(ascending=False)
    top_cols = list(totals.head(top_n).index)

    plt.figure(figsize=FIGSIZE, dpi=DPI)
    for c in top_cols:
        plt.plot(plot_df[helper_col], plot_df[c], label=str(c))

    plt.title(f"{title} (Top {min(top_n, len(top_cols))} series)")
    plt.xlabel("day")
    plt.ylabel("count")
    plt.xticks(rotation=45, ha="right")
    plt.legend(loc="best", fontsize=("small" if len(top_cols) <= 12 else "x-small"))
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def plot_timezone_user_multiheader(df: pd.DataFrame, title: str, out_png: str, top_n: int = TOP_N_SERIES) -> None:
    plot_wide(df, title, out_png, top_n=top_n)


def save_plot_for_df(df: pd.DataFrame, plots_dir: str, output_stem: str, title: str, kind: str) -> None:
    ensure_dir(plots_dir)
    out_png = os.path.join(plots_dir, f"{output_stem}.png")

    try:
        if kind == "day_only":
            plot_day_only(df, title, out_png)
        elif kind == "wide":
            plot_wide(df, title, out_png)
        elif kind == "tz_user_multi":
            plot_timezone_user_multiheader(df, title, out_png)
        else:
            plot_wide(df, title, out_png)
    except Exception as e:
        print(f"[WARN] plot failed for {output_stem}: {e}")


def write_per_page(
    df: pd.DataFrame,
    out_dir: str,
    plots_dir: str,
    base_filename: str,
    builder_fn,
    plot_kind: str,
    plot_title_prefix: str,
) -> None:
    pages = sorted(df["page"].dropna().unique().tolist(), key=lambda x: str(x))
    for page in pages:
        page_df = df[df["page"] == page].copy()
        out_df = builder_fn(page_df)

        safe_page = sanitize_filename_component(str(page))
        csv_name = f"{base_filename}__page_{safe_page}.csv"
        stem = os.path.splitext(csv_name)[0]

        write_csv(out_df, out_dir, csv_name)
        save_plot_for_df(out_df, plots_dir, stem, f"{plot_title_prefix} | page={page}", plot_kind)


# --------------------------
# Main
# --------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CSV reports and plots from audit parquet logs.")
    parser.add_argument("--input", default="./audit_log", help="Folder containing parquet audit logs.")
    parser.add_argument("--output", default="./audit_reports", help="Folder to write CSV outputs.")
    args = parser.parse_args()

    out_dir = args.output
    plots_dir = os.path.join(out_dir, "plots")
    ensure_dir(out_dir)
    ensure_dir(plots_dir)

    raw = infer_country_from_timezone(load_all_parquets(args.input))
    uid = USER_ID_COL

    # 1) distinct users per day (day-only)
    r1 = raw.groupby("day", as_index=False).agg(count=(uid, "nunique"))
    r1 = day_only(r1)
    csv1 = "distinct_users_per_day.csv"
    write_csv(r1, out_dir, csv1)
    save_plot_for_df(r1, plots_dir, os.path.splitext(csv1)[0], "Distinct users per day", "day_only")

    # 2) total reports per day (day-only; sum count)
    r2 = raw.groupby("day", as_index=False).agg(count=("count", "sum"))
    r2 = day_only(r2)
    csv2 = "logs_total_per_day.csv"
    write_csv(r2, out_dir, csv2)
    save_plot_for_df(r2, plots_dir, os.path.splitext(csv2)[0], "Total audit reports per day", "day_only")

    # 3) reports per user per day (wide)
    r3 = raw.groupby(["day", uid], as_index=False).agg(count=("count", "sum"))
    r3w = pivot_single_selector(r3, uid)
    csv3 = "reports_per_user_per_day.csv"
    write_csv(r3w, out_dir, csv3)
    save_plot_for_df(r3w, plots_dir, os.path.splitext(csv3)[0], "Reports per user per day", "wide")

    # 4) inferred country distinct users per day (wide)
    r4 = (
        raw.dropna(subset=["inferred_country_name"])
        .groupby(["day", "inferred_country_name"], as_index=False)
        .agg(count=(uid, "nunique"))
    )
    r4w = pivot_single_selector(r4, "inferred_country_name")
    csv4 = "inferred_country_distinct_users_per_day.csv"
    write_csv(r4w, out_dir, csv4)
    save_plot_for_df(r4w, plots_dir, os.path.splitext(csv4)[0], "Distinct users per day by inferred country", "wide")

    # 5a) timezone distinct users per day (wide)
    r5a = raw.groupby(["day", "timezone"], as_index=False).agg(count=(uid, "nunique"))
    r5aw = pivot_single_selector(r5a, "timezone")
    csv5a = "timezone_distinct_users_per_day.csv"
    write_csv(r5aw, out_dir, csv5a)
    save_plot_for_df(r5aw, plots_dir, os.path.splitext(csv5a)[0], "Distinct users per day by timezone", "wide")

    # 5b) distinct timezones per day (day-only)
    r5b = raw.groupby("day", as_index=False).agg(count=("timezone", "nunique"))
    r5b = day_only(r5b)
    csv5b = "distinct_timezones_per_day.csv"
    write_csv(r5b, out_dir, csv5b)
    save_plot_for_df(r5b, plots_dir, os.path.splitext(csv5b)[0], "Distinct timezones per day", "day_only")

    # 6) distinct logs per day per user (wide) => distinct logs = stored rows (not sum(count))
    r6 = raw.groupby(["day", uid], as_index=False).agg(count=(uid, "size"))
    r6w = pivot_single_selector(r6, uid)
    csv6 = "distinct_logs_per_day_per_user.csv"
    write_csv(r6w, out_dir, csv6)
    save_plot_for_df(r6w, plots_dir, os.path.splitext(csv6)[0], "Distinct log rows per user per day", "wide")

    # 7) reports per timezone per user per day (multiheader) => sum(count)
    r7 = raw.groupby(["day", "timezone", uid], as_index=False).agg(count=("count", "sum"))
    r7w = pivot_timezone_user_multiheader(r7, "timezone", uid)
    csv7 = "reports_per_timezone_per_user_per_day.csv"
    write_csv(r7w, out_dir, csv7)
    save_plot_for_df(r7w, plots_dir, os.path.splitext(csv7)[0], "Reports per timezone per user per day", "tz_user_multi")

    # -------------------------
    # PER-PAGE: separate CSVs + plots per page
    # -------------------------

    # (2) per page: total reports per day (day-only per page)
    r2p = raw.groupby(["page", "day"], as_index=False).agg(count=("count", "sum"))
    write_per_page(
        r2p, out_dir, plots_dir,
        base_filename="logs_total_per_day",
        builder_fn=lambda d: day_only(d[["day", "count"]]),
        plot_kind="day_only",
        plot_title_prefix="Total audit reports per day",
    )

    # (3) per page: reports per user per day (wide per page)
    r3p = raw.groupby(["page", "day", uid], as_index=False).agg(count=("count", "sum"))
    write_per_page(
        r3p, out_dir, plots_dir,
        base_filename="reports_per_user_per_day",
        builder_fn=lambda d: pivot_single_selector(d[["day", uid, "count"]], uid),
        plot_kind="wide",
        plot_title_prefix="Reports per user per day",
    )

    # (4) per page: inferred country distinct users per day (wide per page)
    r4p = (
        raw.dropna(subset=["inferred_country_name"])
        .groupby(["page", "day", "inferred_country_name"], as_index=False)
        .agg(count=(uid, "nunique"))
    )
    write_per_page(
        r4p, out_dir, plots_dir,
        base_filename="inferred_country_distinct_users_per_day",
        builder_fn=lambda d: pivot_single_selector(d[["day", "inferred_country_name", "count"]], "inferred_country_name"),
        plot_kind="wide",
        plot_title_prefix="Distinct users per day by inferred country",
    )

    # (5) per page: timezone distinct users per day (wide per page)
    r5p = raw.groupby(["page", "day", "timezone"], as_index=False).agg(count=(uid, "nunique"))
    write_per_page(
        r5p, out_dir, plots_dir,
        base_filename="timezone_distinct_users_per_day",
        builder_fn=lambda d: pivot_single_selector(d[["day", "timezone", "count"]], "timezone"),
        plot_kind="wide",
        plot_title_prefix="Distinct users per day by timezone",
    )

    # (6) per page: distinct logs per day per user (wide per page)
    r6p = raw.groupby(["page", "day", uid], as_index=False).agg(count=(uid, "size"))
    write_per_page(
        r6p, out_dir, plots_dir,
        base_filename="distinct_logs_per_day_per_user",
        builder_fn=lambda d: pivot_single_selector(d[["day", uid, "count"]], uid),
        plot_kind="wide",
        plot_title_prefix="Distinct log rows per user per day",
    )

    # (7) per page: reports per timezone per user per day (multiheader per page)
    r7p = raw.groupby(["page", "day", "timezone", uid], as_index=False).agg(count=("count", "sum"))
    write_per_page(
        r7p, out_dir, plots_dir,
        base_filename="reports_per_timezone_per_user_per_day",
        builder_fn=lambda d: pivot_timezone_user_multiheader(d[["day", "timezone", uid, "count"]], "timezone", uid),
        plot_kind="tz_user_multi",
        plot_title_prefix="Reports per timezone per user per day",
    )

    print(f"Done. CSVs in:   {os.path.abspath(out_dir)}")
    print(f"Plots in:        {os.path.abspath(plots_dir)}")
    if pytz is None:
        print("Note: 'pytz' not installed; inferred-country reports will be empty/NA.")


if __name__ == "__main__":
    main()
