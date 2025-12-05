# path: tests/test_model_input.py
#
# Sanity checks for model_input files produced by:
#   ml/build_model_input_from_features.py
#
# Usage (from project root):
#   python tests/test_model_input.py --coin btc_usd --horizon-min 15
#

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Sanity check a model_input_{horizon}m file under "
            "ml/models/XGBoost/<coin_dir>/."
        )
    )
    p.add_argument(
        "--coin",
        required=True,
        help='Coin folder name like "btc_usd", "eth_usd", "sol_usd".',
    )
    p.add_argument(
        "--horizon-min",
        type=int,
        default=15,
        help="Forward horizon in minutes/bars (default: 15).",
    )
    p.add_argument(
        "--models-root",
        default="ml/models/XGBoost",
        help="Root directory containing per-coin subfolders (default: ml/models/XGBoost).",
    )
    p.add_argument(
        "--sample-checks",
        type=int,
        default=1000,
        help="Max number of rows to sample for label correctness checks (default: 1000).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    coin_dir_name = args.coin.lower()
    horizon = args.horizon_min
    models_root = Path(args.models_root).resolve()

    coin_dir = models_root / coin_dir_name

    print(f"[test_model_input] coin_dir={coin_dir_name} horizon={horizon}m")
    print(f"[test_model_input] models_root={models_root}")
    print(f"[test_model_input] looking in: {coin_dir}")

    if not coin_dir.exists() or not coin_dir.is_dir():
        print(f"[ERROR] Coin directory not found or not a directory: {coin_dir}")
        return 1

    pattern = f"*_model_input_{horizon}m.parquet"
    candidates = list(coin_dir.glob(pattern))

    if not candidates:
        print(f"[ERROR] No Parquet files matching {pattern} found in {coin_dir}")
        return 1

    if len(candidates) > 1:
        print(f"[WARN] Multiple files found for pattern {pattern}. Using the first one:")
        for c in candidates:
            print(f"       - {c.name}")

    pq_path = candidates[0]
    print(f"[test_model_input] using file: {pq_path}")

    df = pd.read_parquet(pq_path)
    print(f"[OK] Loaded {len(df)} rows from {pq_path}")

    y_ret_col = f"y_ret_fwd_{horizon}m"
    y_dir_col = f"y_dir_fwd_{horizon}m"

    required_cols = ["close", y_ret_col, y_dir_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"[ERROR] Missing required columns: {missing}")
        return 1

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.set_index("timestamp").sort_index()

    # --- Check numeric forward returns: close[t+h]/close[t] - 1 ---
    close = pd.to_numeric(df["close"], errors="coerce")

    expected_fwd_ret = close.shift(-horizon) / close - 1.0
    actual_ret = pd.to_numeric(df[y_ret_col], errors="coerce")

    mask = expected_fwd_ret.notna() & actual_ret.notna()
    expected = expected_fwd_ret[mask].reset_index(drop=True)
    actual = actual_ret[mask].reset_index(drop=True)

    if len(actual) == 0:
        print("[ERROR] No valid rows to compare for forward returns (NaNs everywhere?).")
        return 1

    if len(actual) > args.sample_checks:
        idx = np.random.choice(len(actual), size=args.sample_checks, replace=False)
        expected = expected.iloc[idx]
        actual = actual.iloc[idx]

    diff = (actual - expected).abs()
    max_diff = float(diff.max())

    print(f"[CHECK] Max abs diff between stored and recomputed forward returns: {max_diff:.3e}")
    if max_diff > 1e-8:
        print(
            "[ERROR] Forward return labels appear inconsistent with close prices "
            f"(max diff {max_diff:.3e} > 1e-8). Possible lookahead bug or misalignment."
        )
        return 1

    print("[OK] Forward return labels match close[t+h]/close[t] - 1 within tolerance.")

    # --- Check ternary direction labels: -1, 0, 1 with neutral band ±0.001 ---
    neutral_threshold = 0.001  # must match build_model_input_from_features

    ret_all = actual_ret.copy()
    dir_expected_vals = np.where(
        ret_all > neutral_threshold,
        1,
        np.where(ret_all < -neutral_threshold, -1, 0),
    ).astype("int8")
    dir_expected = pd.Series(dir_expected_vals, index=ret_all.index)

    dir_actual = df[y_dir_col].astype("int8")

    dir_mask = dir_expected.notna() & dir_actual.notna()
    mismatches = int((dir_expected[dir_mask] != dir_actual[dir_mask]).sum())

    print(f"[CHECK] Direction label mismatches (ternary): {mismatches} rows")
    if mismatches > 0:
        print("[ERROR] Direction labels are not consistent with ternary rule on forward returns.")
        return 1

    print("[OK] Direction labels match ternary rule on forward returns (±0.001 neutral band).")

    # --- NaN checks ---
    cols_for_nan_check = [c for c in df.columns if c not in ("symbol",)]
    n_nans = int(df[cols_for_nan_check].isna().any(axis=1).sum())
    print(f"[CHECK] Rows with any NaN in features+labels (excluding symbol): {n_nans}")
    if n_nans > 0:
        print("[WARN] There are rows with NaNs. You may be dropping them before training, which is fine.")
    else:
        print("[OK] No NaNs detected in features+labels (excluding symbol).")

    # --- Label distribution (ternary) ---
    neg = int((df[y_dir_col] == -1).sum())
    zero = int((df[y_dir_col] == 0).sum())
    pos = int((df[y_dir_col] == 1).sum())
    total = int(len(df))

    frac_neg = neg / total if total > 0 else 0.0
    frac_zero = zero / total if total > 0 else 0.0
    frac_pos = pos / total if total > 0 else 0.0

    print(f"[STATS] Label distribution for {y_dir_col}:")
    print(f"        -1: {neg} ({frac_neg:.3%})")
    print(f"         0: {zero} ({frac_zero:.3%})")
    print(f"        +1: {pos} ({frac_pos:.3%})")

    if total == 0:
        print("[ERROR] model_input file has 0 rows after loading.")
        return 1

    print("[test_model_input] All core checks passed ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
