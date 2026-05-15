import argparse
from pathlib import Path

import pandas as pd


FILE_PAIRS = [
    ("USDC_USD.parquet", "USDT_USD.parquet"),
    ("USDC_USD_daily.csv", "USDT_USD_daily.csv"),
    ("combined_data/USDC_USD_hourly.csv", "combined_data/USDT_USD_hourly.csv"),
]


def rewrite_usdc_frame_as_usdt(df: pd.DataFrame) -> pd.DataFrame:
    rewritten = df.copy()

    if "Volume USDC" in rewritten.columns:
        rewritten = rewritten.rename(columns={"Volume USDC": "Volume USDT"})

    if "symbol" in rewritten.columns:
        rewritten["symbol"] = "USDT/USD"

    return rewritten


def process_file(data_dir: Path, source_rel: str, target_rel: str, dry_run: bool) -> None:
    source_path = data_dir / source_rel
    target_path = data_dir / target_rel

    if not source_path.exists():
        raise FileNotFoundError(f"Missing source file: {source_path}")

    if source_path.suffix == ".parquet":
        df = pd.read_parquet(source_path)
    else:
        df = pd.read_csv(source_path)

    rewritten = rewrite_usdc_frame_as_usdt(df)

    print(f"{source_path} -> {target_path} ({len(rewritten)} rows)")

    if dry_run:
        return

    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.suffix == ".parquet":
        rewritten.to_parquet(target_path, index=False)
    else:
        rewritten.to_csv(target_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Overwrite USDT data files with USDC data rewritten as USDT/USD."
    )
    parser.add_argument(
        "--data-dir",
        default=Path(__file__).resolve().parent.parent / "quantammsim" / "data",
        type=Path,
        help="Base data directory containing the USDC/USDT files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the files that would be rewritten without modifying them.",
    )
    args = parser.parse_args()

    for source_rel, target_rel in FILE_PAIRS:
        process_file(args.data_dir, source_rel, target_rel, args.dry_run)


if __name__ == "__main__":
    main()
