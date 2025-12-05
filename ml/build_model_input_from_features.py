# path: ml/build_model_input_from_features.py

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def add_forward_labels(
    df: pd.DataFrame,
    *,
    horizon_min: int = 15,
    close_col: str = "close",
    neutral_threshold: float = 0.001,  # 0.1% neutral band
) -> pd.DataFrame:
    """
    Add forward return + ternary direction labels based on SIMPLE return over `horizon_min` bars.
    Assumes rows are in time order, 1 row per bar (1m in your pipeline).

    y_ret_fwd_{horizon}m  = close_{t+h} / close_t - 1
    y_dir_fwd_{horizon}m  =  1 if y_ret >  neutral_threshold
                             -1 if y_ret < -neutral_threshold
                              0 otherwise
    """
    out = df.copy()
    close = pd.to_numeric(out[close_col], errors="coerce")

    shift_steps = horizon_min  # 1 row = 1 minute in your pipeline
    fwd_price = close.shift(-shift_steps)
    fwd_ret = fwd_price / close - 1.0

    y_ret_col = f"y_ret_fwd_{horizon_min}m"
    y_dir_col = f"y_dir_fwd_{horizon_min}m"

    out[y_ret_col] = fwd_ret

    # Ternary label: -1, 0, 1 with neutral band
    y_dir = np.where(
        fwd_ret > neutral_threshold,
        1,
        np.where(fwd_ret < -neutral_threshold, -1, 0),
    ).astype("int8")
    out[y_dir_col] = y_dir

    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert features.{parquet,csv} into model_input with forward-return labels."
    )
    p.add_argument(
        "--features-path",
        required=True,
        help="Path to features.parquet or features.csv (output of build_multitimeframe_features).",
    )
    p.add_argument(
        "--horizon-min",
        type=int,
        default=15,
        help="Forward horizon in minutes/bars for the label (default: 15).",
    )
    p.add_argument(
        "--out-dir",
        default="ml/models/XGBoost",
        help=(
            "Root output directory for model_input files. "
            "Per-coin subfolders will be created here. "
            "Default: ml/models/XGBoost (relative to current working directory)."
        ),
    )
    p.add_argument(
        "--no-dropna",
        action="store_true",
        help="Do NOT drop NA rows after computing labels (only used to keep tail rows).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    features_path = Path(args.features_path).resolve()
    horizon_min = args.horizon_min
    dropna = not args.no_dropna

    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[model_input] features={features_path} horizon_min={horizon_min} out_root={out_root}")

    # Load features
    if features_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(features_path)
    elif features_path.suffix.lower() == ".csv":
        df = pd.read_csv(features_path)
    else:
        raise ValueError(f"Unsupported features file type: {features_path.suffix}")

    # Ensure timestamp is parsed + sorted so label logic is correct
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.sort_values("timestamp")

    # 1) CLEAN FEATURE ROWS FIRST (no labels yet)
    #    Treat everything except timestamp/symbol AND existing label cols as features.
    label_prefixes = ("y_ret_fwd_", "y_dir_fwd_")
    feature_cols = [
        c
        for c in df.columns
        if c not in ("timestamp", "symbol")
        and not any(c.startswith(pfx) for pfx in label_prefixes)
    ]

    df = df.dropna(subset=feature_cols)

    # 2) ADD LABELS on the cleaned, time-ordered data
    df_labeled = add_forward_labels(df, horizon_min=horizon_min, close_col="close")

    # 3) DROP ONLY ROWS WITH NaN LABELS (tail, or any weirdness)
    if dropna:
        y_ret_col = f"y_ret_fwd_{horizon_min}m"
        df_labeled = df_labeled.dropna(subset=[y_ret_col])

    # ---- Derive names ----
    # Example features path:
    #   data/processed/features/btc_usd/btc_usd_1m_features.parquet
    features_path = Path(args.features_path).resolve()

    # Coin dir from parent folder name: "btc_usd", "eth_usd", etc.
    symbol_dir_raw = features_path.parent.name   # e.g. "btc_usd"
    symbol_dir = symbol_dir_raw.lower()

    # Per-coin subfolder inside out_root, e.g. ml/models/XGBoost/btc_usd/
    coin_dir = out_root / symbol_dir
    coin_dir.mkdir(parents=True, exist_ok=True)

   # Basename does NOT include timeframe, just coin + horizon:
   #   btc_usd_model_input_15m
    model_basename = f"{symbol_dir}_model_input_{horizon_min}m"

    pq_path = coin_dir / f"{model_basename}.parquet"
    csv_path = coin_dir / f"{model_basename}.csv"

    df_labeled.to_parquet(pq_path)
    df_labeled.to_csv(csv_path, index=False)

    print(f"[model_input] → {pq_path} / {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
