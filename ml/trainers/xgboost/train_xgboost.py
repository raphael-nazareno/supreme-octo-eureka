# path: ml/train_xgboost.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Train an XGBoost classifier on per-coin model_input data "
            "(e.g. btc_1m_model_input_15m.parquet) to predict y_dir_fwd_{horizon}m."
        )
    )
    p.add_argument(
        "--coin",
        required=True,
        help='Coin folder name like "btc_usd", "eth_usd", "sol_usd".',
    )
    p.add_argument(
        "--config",
        default=None,
        help="Optional path to a JSON config file to override defaults (per-coin + global).",
    )
    p.add_argument(
        "--horizon-min",
        type=int,
        default=15,
        help="Forward horizon in minutes for the label (default: 15 → y_dir_fwd_15m).",
    )
    p.add_argument(
        "--model-input-path",
        default=None,
        help=(
            "Optional explicit path to model_input parquet. "
            "If omitted, script will look under ml/models/XGBoost/<coin> "
            "for '*_model_input_{horizon}m.parquet'."
        ),
    )
    p.add_argument(
        "--models-root",
        default="ml/models/XGBoost",
        help="Root directory for models and default model_input lookup (default: ml/models/XGBoost).",
    )
    p.add_argument(
        "--feature-version",
        type=int,
        default=1,
        help="Feature version integer used in model_id (fv<feature_version>, default: 1).",
    )
    p.add_argument(
        "--model-version",
        type=int,
        default=1,
        help="Model version integer used in model_id (v<model_version>, default: 1).",
    )
    p.add_argument(
        "--train-frac",
        type=float,
        default=0.70,
        help="Fraction of data for training (default: 0.70). Validation fraction is 0.15 by default.",
    )
    p.add_argument(
        "--val-frac",
        type=float,
        default=0.15,
        help="Fraction of data for validation (default: 0.15). Remainder is used for test.",
    )
    p.add_argument(
        "--label-mode",
        type=str,
        default="gross",
        choices=["gross", "net"],
        help=(
            "Which label to train on: "
            "'gross' -> y_dir_fwd_{horizon}m, "
            "'net'   -> y_dir_net_fwd_{horizon}m."
        ),
    )




    # Basic XGBoost hyperparameters
    p.add_argument("--n-estimators", type=int, default=500)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--subsample", type=float, default=0.9)
    p.add_argument("--colsample-bytree", type=float, default=0.8)
    p.add_argument("--min-child-weight", type=float, default=1.0)
    p.add_argument("--reg-alpha", type=float, default=0.0)
    p.add_argument("--reg-lambda", type=float, default=1.0)
    p.add_argument("--early-stopping-rounds", type=int, default=50)
    p.add_argument("--n-jobs", type=int, default=-1)

    return p.parse_args()


def apply_config(args: argparse.Namespace) -> argparse.Namespace:
    """
    Load a JSON config and apply settings.
    Priority: config overrides CLI for all keys except 'coin' and 'config'.
    Expected structure:

    {
      "default": { ... global defaults ... },
      "coins": {
        "btc_usd": { ... overrides ... },
        "eth_usd": { ... }
      }
    }
    """
    if not args.config:
        return args

    cfg_path = Path(args.config).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    coin_key = args.coin.lower()
    default_cfg = cfg.get("default", {})
    coins_cfg = cfg.get("coins", {})
    coin_cfg = coins_cfg.get(coin_key, {})

    merged: Dict[str, Any] = {}
    merged.update(default_cfg)
    merged.update(coin_cfg)

    # Apply to args (config wins), except for coin/config themselves
    for key, value in merged.items():
        if key in ("coin", "config"):
            continue
        if hasattr(args, key):
            setattr(args, key, value)

    print(f"[train_xgboost] Loaded config from {cfg_path}")
    print(f"[train_xgboost] Effective settings (after config):")
    for k in sorted(merged.keys()):
        if hasattr(args, k):
            print(f"  {k} = {getattr(args, k)}")

    return args


def _find_model_input_path(
    coin: str,
    horizon_min: int,
    models_root: Path,
    explicit: str | None,
) -> Path:
    # If user provided an explicit path, honour it.
    if explicit is not None:
        path = Path(explicit).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Explicit model_input_path not found: {path}")
        return path

    coin_dir = models_root / coin
    if not coin_dir.exists():
        raise FileNotFoundError(f"Coin dir not found: {coin_dir}")

    filename = f"{coin}_model_input_{horizon_min}m.parquet"
    path = coin_dir / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Expected model_input file not found: {path}. "
            f"Check that you created {filename} in {coin_dir}."
        )

    return path.resolve()



def _select_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return list of numeric feature columns, excluding timestamp/symbol/labels & cost columns."""
    drop_cols = {"timestamp", "symbol"}

    # Drop any label or cost columns
    drop_cols.update(
        [
            c
            for c in df.columns
            if c.startswith("y_ret_fwd_")
            or c.startswith("y_dir_fwd_")
            or c.startswith("y_ret_net_fwd_")
            or c.startswith("y_dir_net_fwd_")
            or c.startswith("y_fee_")
            or c.startswith("y_micro_cost_")
            or c.startswith("y_cost_")
        ]
    )

    candidate_cols = [c for c in df.columns if c not in drop_cols]
    numeric_cols = df[candidate_cols].select_dtypes(include=[np.number]).columns.tolist()
    return numeric_cols


def _compute_split_indices(n: int, train_frac: float, val_frac: float) -> tuple[int, int]:
    if train_frac <= 0 or val_frac < 0 or train_frac + val_frac >= 1.0:
        raise ValueError(f"Invalid train/val fractions: train_frac={train_frac}, val_frac={val_frac}")
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    train_end = max(1, min(train_end, n - 2))
    val_end = max(train_end + 1, min(val_end, n - 1))
    return train_end, val_end


def _eval_split(
    name: str,
    model: XGBClassifier,
    X: np.ndarray,
    y: np.ndarray,
) -> Dict[str, Any]:
    """Compute metrics for a given split."""
    if X.shape[0] == 0:
        return {
            "n_samples": 0,
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "roc_auc": None,
            "confusion_matrix": None,
        }

    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    acc = float(accuracy_score(y, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, y_pred, average="binary", zero_division=0
    )
    try:
        auc = float(roc_auc_score(y, y_proba))
    except ValueError:
        auc = None

    cm = confusion_matrix(y, y_pred).tolist()

    return {
        "n_samples": int(X.shape[0]),
        "accuracy": acc,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": auc,
        "confusion_matrix": cm,
    }


def main() -> int:
    args = parse_args()
    args.coin = args.coin.lower()
    args = apply_config(args)

    coin = args.coin
    horizon = args.horizon_min
    models_root = Path(args.models_root).resolve()

    print(f"[train_xgboost] coin={coin} horizon={horizon}m models_root={models_root}")

    model_input_path = _find_model_input_path(
        coin=coin,
        horizon_min=horizon,
        models_root=models_root,
        explicit=args.model_input_path,
    )

    print(f"[train_xgboost] Using model_input: {model_input_path}")

    if model_input_path.suffix == ".csv":
        df = pd.read_csv(model_input_path)
    elif model_input_path.suffix == ".parquet":
        df = pd.read_parquet(model_input_path)
    else:
        raise ValueError(f"Unsupported model_input extension: {model_input_path.suffix}")

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.sort_values("timestamp")

    label_mode = getattr(args, "label_mode", "gross").lower()
    if label_mode == "gross":
        label_col = f"y_dir_fwd_{horizon}m"
    else:
        label_col = f"y_dir_net_fwd_{horizon}m"

    if label_col not in df.columns:
        raise KeyError(f"Label column {label_col} not found in model_input.")


    # --- FORCE LABELS TO BINARY 0/1 ---
    # We may have tri-state labels {-1, 0, 1} from the pipeline.
    # For this model we want:
    #   up (y > 0)      -> 1
    #   flat/down (<=0) -> 0

    raw_unique = np.unique(df[label_col].dropna().to_numpy())
    print(f"[train_xgboost] Raw label values in {label_col}: {raw_unique}")

    df[label_col] = (df[label_col] > 0).astype(int)

    bin_unique = np.unique(df[label_col].dropna().to_numpy())
    print(f"[train_xgboost] After binarisation, labels in {label_col}: {bin_unique}")
    # --- END LABEL BINARISATION ---

    feature_cols = _select_feature_columns(df)
    if not feature_cols:
        raise RuntimeError("No feature columns found after filtering. Check your model_input schema.")


    print(f"[train_xgboost] Using {len(feature_cols)} feature columns.")

    X = df[feature_cols]
    y = df[label_col]

    # Drop rows with NaNs in features or label
    mask = X.notna().all(axis=1) & y.notna()
    if not mask.all():
        dropped = int((~mask).sum())
        print(f"[train_xgboost] Dropping {dropped} rows with NaNs in features/label.")
        X = X[mask]
        y = y[mask]
        if "timestamp" in df.columns:
            df = df.loc[mask]

    X_np = X.to_numpy(dtype=float)
    y_np = y.to_numpy(dtype=int)


    # Detect tri-state / negative labels and collapse to binary:
    #   up   (y > 0)  -> 1
    #   flat/down (y <= 0) -> 0
    unique = np.unique(y_np[~pd.isna(y_np)])
    if (-1 in unique) or (len(unique) > 2):
        print(f"[train_xgboost] Detected multi/tri-state labels {unique}; "
              "collapsing to binary 0/1 via (y > 0).")
        y_np = (y_np > 0).astype(int)
    else:
        y_np = y_np.astype(int)

    X = df[feature_cols]
    y = df[label_col]

    # Drop rows with NaNs in features or label
    mask = X.notna().all(axis=1) & y.notna()
    if not mask.all():
        dropped = int((~mask).sum())
        print(f"[train_xgboost] Dropping {dropped} rows with NaNs in features/label.")
        X = X[mask]
        y = y[mask]
        if "timestamp" in df.columns:
            df = df.loc[mask]

    X_np = X.to_numpy(dtype=float)
    y_np = y.to_numpy(dtype=int)

    n = X_np.shape[0]
    if n < 1000:
        print(f"[WARN] Only {n} samples available after cleaning. This may be too small for a robust model.")

    train_end, val_end = _compute_split_indices(n, args.train_frac, args.val_frac)

    # Split
    X_train, y_train = X_np[:train_end], y_np[:train_end]
    X_val, y_val = X_np[train_end:val_end], y_np[train_end:val_end]
    X_test, y_test = X_np[val_end:], y_np[val_end:]

    print(f"[train_xgboost] Split sizes: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")

    # XGBoost model
    model = XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        min_child_weight=args.min_child_weight,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        objective="binary:logistic",
        n_jobs=args.n_jobs,
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
    )
    eval_set = [(X_train, y_train), (X_val, y_val)]
    
    print("[train_xgboost] Training XGBoost...")

    model.fit(
        X_train,
        y_train,
        eval_set=eval_set,
        verbose=True,
    )

    print("[train_xgboost] Training complete.")


    # Evaluate splits
    metrics_train = _eval_split("train", model, X_train, y_train)
    metrics_val = _eval_split("val", model, X_val, y_val)
    metrics_test = _eval_split("test", model, X_test, y_test)

    # Timestamp ranges for splits (if available)
    ts_ranges: Dict[str, Dict[str, Any]] = {}
    if "timestamp" in df.columns:
        ts = df["timestamp"].reset_index(drop=True)

        def _safe_ts(start: int, end: int) -> Dict[str, Any]:
            if end <= start or start >= len(ts):
                return {"start": None, "end": None}
            return {
                "start": ts.iloc[start].isoformat(),
                "end": ts.iloc[end - 1].isoformat(),
            }

        ts_ranges["train"] = _safe_ts(0, train_end)
        ts_ranges["val"] = _safe_ts(train_end, val_end)
        ts_ranges["test"] = _safe_ts(val_end, len(ts))

    # Build model_id
    model_id = (
        f"xgb_{coin}_1m_{horizon}m_dir_"
        f"fv{int(args.feature_version)}_v{int(args.model_version):03d}"
    )

    coin_dir = models_root / coin
    coin_dir.mkdir(parents=True, exist_ok=True)

    model_path = coin_dir / f"{model_id}.json"
    metrics_path = coin_dir / f"{model_id}_metrics.json"

    # Save model
    model.save_model(model_path)
    print(f"[train_xgboost] Saved model to {model_path}")

    # Save metrics
    metrics: Dict[str, Any] = {
        "model_id": model_id,
        "coin": coin,
        "horizon_min": horizon,
        "feature_version": int(args.feature_version),
        "model_version": int(args.model_version),
        "n_rows_total": int(n),
        "n_features": len(feature_cols),
        "splits": {
            "train": {
                **metrics_train,
                "ts_range": ts_ranges.get("train", {}),
            },
            "val": {
                **metrics_val,
                "ts_range": ts_ranges.get("val", {}),
            },
            "test": {
                **metrics_test,
                "ts_range": ts_ranges.get("test", {}),
            },
        },
        "feature_columns": feature_cols,
        "model_input_path": str(model_input_path),
    }

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"[train_xgboost] Saved metrics to {metrics_path}")
    print("[train_xgboost] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
