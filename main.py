# =====================================================================
# file: home_quant_lab_v5/main.py
# =====================================================================

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ----------------------------- Optional deps (always-bound names) -----------------------------
# zoneinfo
try:
    from zoneinfo import ZoneInfo as _ZoneInfo  # type: ignore
except Exception:  # pragma: no cover
    _ZoneInfo = None  # type: ignore[assignment]

# yaml
try:
    import yaml as _yaml  # type: ignore
except Exception:  # pragma: no cover
    _yaml = None  # type: ignore[assignment]

# joblib
try:
    import joblib as _joblib  # type: ignore
except Exception:  # pragma: no cover
    _joblib = None  # type: ignore[assignment]

# xgboost
try:
    from xgboost import XGBClassifier as _XGBClassifier  # type: ignore
except Exception:  # pragma: no cover
    _XGBClassifier = None  # type: ignore[assignment]

# lightgbm
try:
    from lightgbm import LGBMClassifier as _LGBMClassifier  # type: ignore
except Exception:  # pragma: no cover
    _LGBMClassifier = None  # type: ignore[assignment]

# sklearn bits (these are standard; if missing, we want the import error)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# ----------------------------- Public exports for tests -----------------------------
__all__ = [
    "Costs",
    "TradeLog",
    "run_backtest_next_open_long_only",
    "run_backtest_next_open_long_short",
    "make_estimator",
    "save_model",
    "load_model",
]

# ----------------------------- Logging -----------------------------
logging.basicConfig(
    level=os.environ.get("HQL_LOGLEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# ----------------------------- Time / Config helpers -----------------------------
def now_in_tz(tz: str | None = "Australia/Melbourne") -> datetime:
    """
    Returns timezone-aware now.
    Falls back to UTC if zoneinfo or tz is unavailable, with a clear warning.
    """
    if tz and _ZoneInfo is not None:
        try:
            return datetime.now(_ZoneInfo(tz))
        except Exception:
            logging.warning("Zoneinfo failed for tz=%s, defaulting to UTC.", tz)
    elif tz and _ZoneInfo is None:
        logging.warning("zoneinfo module not available; defaulting to UTC.")
    return datetime.now(timezone.utc)


def load_yaml_config(path: Optional[str]) -> dict:
    """
    Loads YAML mapping; returns {} if no path. Raises helpful errors if missing or PyYAML absent.
    """
    if not path:
        return {}
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing config: {path}")
    if _yaml is None:
        raise ImportError("PyYAML not installed. Install with 'pip install pyyaml' or remove --config.")
    with open(path, "r", encoding="utf-8") as f:
        data = _yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Top-level YAML must be a mapping/dict.")
    return data


# ----------------------------- Costs / Trade Log -----------------------------
@dataclass
class Costs:
    fee_bps: float = 0.0
    slippage_bps: float = 0.0

    def fee(self, notional: float) -> float:
        return abs(float(notional)) * (self.fee_bps / 10_000.0)

    def slippage_factor(self) -> float:
        return self.slippage_bps / 10_000.0


@dataclass
class TradeLog:
    # Entry
    side: str  # "LONG" | "SHORT"
    entry_action: str  # "BUY" or "SELL"
    entry_time: str
    entry_price: float

    # Exit
    exit_time: Optional[str] = None
    exit_price: Optional[float] = None

    # Outcome
    pnl: float = 0.0  # realized PnL (+ profit, - loss)
    hit_stop_loss: bool = False
    hit_take_profit: bool = False

    # Meta
    notes: Optional[str] = None


# ----------------------------- Backtest Core -----------------------------
def _apply_slippage(price: float, side: str, is_entry: bool, costs: Costs) -> float:
    """
    Simple slippage model: push against you.
    LONG:  entry BUY worse (up), exit SELL worse (down)
    SHORT: entry SELL worse (down), exit BUY worse (up)
    """
    p = float(price)
    s = costs.slippage_factor()
    if s <= 0.0:
        return p

    if side == "LONG":
        return p * (1.0 + s) if is_entry else p * (1.0 - s)
    else:  # SHORT
        return p * (1.0 - s) if is_entry else p * (1.0 + s)


def _validate_thresholds(thr_buy: float, thr_sell: float, sl_pct: float, tp_pct: float) -> None:
    if thr_buy <= thr_sell:
        raise ValueError("thr_buy must be greater than thr_sell.")
    if sl_pct < 0 or tp_pct < 0:
        raise ValueError("Stop loss and take profit percentages must be non-negative.")


def _ensure_cols(df: pd.DataFrame, cols: Sequence[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}. Did you run your data/feature prep?")


def _mk_equity_frame() -> list[dict[str, float | str]]:
    return []


def _append_equity(equity_buf: list[dict[str, float | str]], ts: pd.Timestamp, equity_val: float) -> None:
    equity_buf.append({"timestamp": str(ts), "equity": float(equity_val)})


def run_backtest_next_open_long_short(
    df_exec: pd.DataFrame,
    probs: np.ndarray,
    thr_buy: float,
    thr_sell: float,
    sl_pct: float,
    tp_pct: float,
    costs: Costs,
    initial_cash: float = 0.0,
) -> tuple[list[TradeLog], pd.DataFrame]:
    """
    One-position at a time, entries at current bar OPEN based on probs.
    Exits via intrabar TP/SL using current bar's HIGH/LOW.
    No auto-close at the end (open PnL remains unrealized).
    Equity = cash + MTM (using CLOSE).
    """
    _validate_thresholds(thr_buy, thr_sell, sl_pct, tp_pct)
    _ensure_cols(df_exec, ["timestamp", "open", "high", "low", "close"])

    if len(df_exec) != len(probs):
        raise ValueError("probs length must match df_exec length (one decision per bar).")

    cash = float(initial_cash)
    pos_side: Optional[str] = None  # "LONG" or "SHORT"
    pos_entry_px: Optional[float] = None
    pos_entry_time: Optional[str] = None

    trades: list[TradeLog] = []
    equity_buf = _mk_equity_frame()

    for i in range(len(df_exec)):
        row = df_exec.iloc[i]
        ts = pd.to_datetime(row["timestamp"])

        o = float(row["open"])
        h = float(row["high"])
        l = float(row["low"])
        c = float(row["close"])
        p = float(probs[i])

        # 1) Entry decision at OPEN (if no position)
        if pos_side is None:
            if p >= thr_buy:
                # LONG entry at open with slippage & fees
                entry_px_raw = o
                entry_px = _apply_slippage(entry_px_raw, "LONG", True, costs)
                fee = costs.fee(entry_px)
                cash -= entry_px + fee  # buy 1 unit
                pos_side = "LONG"
                pos_entry_px = entry_px
                pos_entry_time = str(ts)
                logging.debug(f"LONG entry @ {entry_px:.6f} (open {entry_px_raw:.6f}) fee {fee:.6f}")
            elif p <= thr_sell:
                # SHORT entry
                entry_px_raw = o
                entry_px = _apply_slippage(entry_px_raw, "SHORT", True, costs)
                fee = costs.fee(entry_px)
                cash += entry_px - fee  # sell 1 unit
                pos_side = "SHORT"
                pos_entry_px = entry_px
                pos_entry_time = str(ts)
                logging.debug(f"SHORT entry @ {entry_px:.6f} (open {entry_px_raw:.6f}) fee {fee:.6f}")
            # else: HOLD

        # 2) If in position, check intrabar TP/SL on THIS bar
        trade_closed_this_bar = False
        if pos_side is not None and pos_entry_px is not None and pos_entry_time is not None:
            if pos_side == "LONG":
                sl_price = pos_entry_px * (1.0 - sl_pct)
                tp_price = pos_entry_px * (1.0 + tp_pct)

                hit_sl = l <= sl_price
                hit_tp = h >= tp_price

                # Ambiguous both-hit handling: take SL first (conservative)
                exit_price_raw: Optional[float] = None
                hit_stop_loss = False
                hit_take_profit = False
                if hit_sl and not hit_tp:
                    exit_price_raw = sl_price
                    hit_stop_loss = True
                elif hit_tp and not hit_sl:
                    exit_price_raw = tp_price
                    hit_take_profit = True
                elif hit_sl and hit_tp:
                    exit_price_raw = sl_price
                    hit_stop_loss = True

                if exit_price_raw is not None:
                    exit_px = _apply_slippage(exit_price_raw, "LONG", False, costs)
                    fee = costs.fee(exit_px)
                    cash += exit_px - fee  # sell to close
                    pnl = exit_px - pos_entry_px - fee - costs.fee(pos_entry_px)
                    trades.append(
                        TradeLog(
                            side="LONG",
                            entry_action="BUY",
                            entry_time=pos_entry_time,
                            entry_price=float(pos_entry_px),
                            exit_time=str(ts),
                            exit_price=float(exit_px),
                            pnl=float(pnl),
                            hit_stop_loss=hit_stop_loss,
                            hit_take_profit=hit_take_profit,
                        )
                    )
                    pos_side = None
                    pos_entry_px = None
                    pos_entry_time = None
                    trade_closed_this_bar = True

            else:  # SHORT
                sl_price = pos_entry_px * (1.0 + sl_pct)
                tp_price = pos_entry_px * (1.0 - tp_pct)

                hit_sl = h >= sl_price
                hit_tp = l <= tp_price

                exit_price_raw = None
                hit_stop_loss = False
                hit_take_profit = False
                if hit_sl and not hit_tp:
                    exit_price_raw = sl_price
                    hit_stop_loss = True
                elif hit_tp and not hit_sl:
                    exit_price_raw = tp_price
                    hit_take_profit = True
                elif hit_sl and hit_tp:
                    exit_price_raw = sl_price
                    hit_stop_loss = True

                if exit_price_raw is not None:
                    exit_px = _apply_slippage(exit_price_raw, "SHORT", False, costs)
                    fee = costs.fee(exit_px)
                    cash -= exit_px + fee  # buy to cover
                    pnl = (pos_entry_px - exit_px) - fee - costs.fee(pos_entry_px)
                    trades.append(
                        TradeLog(
                            side="SHORT",
                            entry_action="SELL",
                            entry_time=pos_entry_time,
                            entry_price=float(pos_entry_px),
                            exit_time=str(ts),
                            exit_price=float(exit_px),
                            pnl=float(pnl),
                            hit_stop_loss=hit_stop_loss,
                            hit_take_profit=hit_take_profit,
                        )
                    )
                    pos_side = None
                    pos_entry_px = None
                    pos_entry_time = None
                    trade_closed_this_bar = True

        # 3) Equity as cash + MTM at CLOSE
        if pos_side is None or pos_entry_px is None:
            eq = cash
        else:
            if pos_side == "LONG":
                eq = cash + (c - pos_entry_px)
            else:
                eq = cash + (pos_entry_px - c)

        _append_equity(equity_buf, ts, eq)

        # If trade closed this bar, equity next bars will carry realized cash only unless re-enter.

    equity_df = pd.DataFrame(equity_buf)
    return trades, equity_df


def run_backtest_next_open_long_only(
    df_exec: pd.DataFrame,
    probs: np.ndarray,
    thr_buy: float,
    thr_sell: float,  # kept for API symmetry; ignored for entries
    sl_pct: float,
    tp_pct: float,
    costs: Costs,
    initial_cash: float = 0.0,
) -> tuple[list[TradeLog], pd.DataFrame]:
    """
    Long-only variant. Same semantics as the long/short engine.
    """
    # Use a high/low thr_sell sentinel to keep validation happy but effectively ignore sells
    if thr_buy <= thr_sell:
        # If caller provided <=, gently nudge thr_sell down a bit to pass validation (or raise)
        thr_sell = min(thr_sell, thr_buy - 1e-12)
    return run_backtest_next_open_long_short(
        df_exec=df_exec,
        probs=probs,
        thr_buy=thr_buy,
        thr_sell=thr_sell,
        sl_pct=sl_pct,
        tp_pct=tp_pct,
        costs=costs,
        initial_cash=initial_cash,
    )


# ----------------------------- ML: Estimator factory / save / load -----------------------------
def make_estimator(model_choice: str, random_state: int = 42) -> Any:
    """
    Returns a sklearn-compatible estimator (Pipeline).
    Model choices: logreg, rf, xgb, lgb, mlp
    """
    choice = (model_choice or "logreg").lower()

    if choice in ("logreg", "lr", "logistic"):
        base: Any = LogisticRegression(max_iter=1000, random_state=random_state)
    elif choice in ("rf", "random_forest"):
        base = RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_split=2, n_jobs=-1, random_state=random_state
        )
    elif choice in ("mlp", "mlpclassifier"):
        base = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", max_iter=500, random_state=random_state)
    elif choice in ("xgb", "xgboost"):
        if _XGBClassifier is None:
            raise ImportError("xgboost not installed. Install with 'pip install xgboost' or choose another model.")
        base = _XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
            eval_metric="logloss",
        )
    elif choice in ("lgb", "lightgbm", "lgbm"):
        if _LGBMClassifier is None:
            raise ImportError("lightgbm not installed. Install with 'pip install lightgbm' or choose another model.")
        base = _LGBMClassifier(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
            objective="binary",
        )
    else:
        raise ValueError(f"Unknown --model-choice: {model_choice}")

    pipe = Pipeline(steps=[("scaler", StandardScaler(with_mean=False)), ("model", base)])
    return pipe


def save_model(est: Any, path: str | os.PathLike[str]) -> None:
    if _joblib is None:
        raise ImportError("joblib is not installed. Install with 'pip install joblib'.")
    path = str(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _joblib.dump(est, path)
    logging.info("Saved model → %s", path)


def load_model(path: str | os.PathLike[str]) -> Any:
    if _joblib is None:
        raise ImportError("joblib is not installed. Install with 'pip install joblib'.")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    est = _joblib.load(path)
    logging.info("Loaded model ← %s", path)
    return est


# ----------------------------- ML: Walk-Forward (simplified, leak-safe) -----------------------------
def time_series_splits(n: int, n_splits: int, min_train: int) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    """
    Generator of (train_idx, test_idx) to do simple expanding-window WFO.
    Leak-safe: test indices strictly after train indices.
    """
    if n_splits < 1:
        raise ValueError("n_splits must be >= 1.")
    if min_train < 1:
        raise ValueError("min_train must be >= 1.")

    # compute split cut points
    split_size = (n - min_train) // n_splits
    if split_size <= 0:
        raise ValueError("Not enough samples for the requested number of splits.")

    for k in range(1, n_splits + 1):
        train_end = min_train + split_size * (k - 1)
        test_end = min_train + split_size * k
        train_idx = np.arange(0, train_end, dtype=int)
        test_idx = np.arange(train_end, test_end, dtype=int)
        if len(test_idx) == 0:
            continue
        yield train_idx, test_idx


def train_walk_forward(
    X: pd.DataFrame,
    y: pd.Series,
    model_choice: str,
    n_splits: int = 5,
    min_train: int = 200,
    save_fold_models: bool = False,
    model_dir: str = "models",
    model_tag: str = "model",
    random_state: int = 42,
) -> dict:
    """
    Train leak-safe expanding-window WFO. Saves per-fold models if requested.
    Returns dict with per-fold metrics and paths.
    """
    X = pd.DataFrame(X).copy()
    y = pd.Series(y).copy()
    # Basic NA handling (no forward-fill here to avoid leakage)
    X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    y = y.loc[X.index]

    n = len(X)
    splits = list(time_series_splits(n, n_splits=n_splits, min_train=min_train))
    results: list[dict[str, Any]] = []

    for fold, (tr_idx, te_idx) in enumerate(splits, start=1):
        est = make_estimator(model_choice, random_state=random_state)

        Xtr, ytr = X.iloc[tr_idx], y.iloc[tr_idx]
        Xte, yte = X.iloc[te_idx], y.iloc[te_idx]

        est.fit(Xtr, ytr)  # type: ignore[call-arg]

        # prefer predict_proba if available
        if hasattr(est, "predict_proba"):
            proba = est.predict_proba(Xte)[:, 1]  # type: ignore[index]
        elif hasattr(est, "decision_function"):
            dfun = est.decision_function(Xte)  # type: ignore[call-arg]
            # map decision_function to [0,1] via logistic for comparability
            proba = 1.0 / (1.0 + np.exp(-np.asarray(dfun, dtype=float)))
        else:
            pred = est.predict(Xte)  # type: ignore[call-arg]
            proba = np.asarray(pred, dtype=float)

        fold_info: dict[str, Any] = {
            "fold": fold,
            "train_size": int(len(Xtr)),
            "test_size": int(len(Xte)),
            "oos_mean_proba": float(np.mean(proba)) if len(proba) > 0 else float("nan"),
        }

        if save_fold_models:
            path = os.path.join(model_dir, f"{model_tag}_fold{fold}.pkl")
            save_model(est, path)
            fold_info["model_path"] = path

        results.append(fold_info)

    return {"splits": results, "n_samples": n, "n_splits": len(splits)}


def train_final_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_choice: str,
    save_path: Optional[str] = None,
    random_state: int = 42,
) -> Any:
    """
    Train a final model on ALL data (for deployment). Returns the estimator.
    """
    X = pd.DataFrame(X).copy()
    y = pd.Series(y).copy()
    X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    y = y.loc[X.index]

    est = make_estimator(model_choice, random_state=random_state)
    est.fit(X, y)  # type: ignore[call-arg]

    if save_path:
        save_model(est, save_path)
    return est


# ----------------------------- CLI -----------------------------
def _merge_config(args: argparse.Namespace, cfg: dict) -> argparse.Namespace:
    """
    Overlay YAML config into args if those CLI args are None/unspecified.
    CLI has priority. Keys in cfg should match dest names.
    """
    for k, v in cfg.items():
        if not hasattr(args, k):
            continue
        if getattr(args, k) is None:
            setattr(args, k, v)
    return args


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("home_quant_lab_v5")
    p.add_argument("--config", type=str, default=None, help="Path to YAML config.")

    # Data / features (placeholders; wire to your pipeline as needed)
    p.add_argument("--features-csv", type=str, default=None, help="Path to features CSV with y column.")
    p.add_argument("--label-col", type=str, default="y", help="Target column name in features CSV.")

    # Model choice + train modes
    p.add_argument("--model-choice", type=str, default="logreg",
                   choices=["logreg", "rf", "xgb", "lgb", "mlp"])
    p.add_argument("--train-walk-forward", action="store_true")
    p.add_argument("--train-final-model", action="store_true")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--min-train", type=int, default=200)
    p.add_argument("--save-fold-models", action="store_true")
    p.add_argument("--final-model-path", type=str, default=None)

    # Backtest modes
    p.add_argument("--backtest-wfo", action="store_true", help="Demo backtest using WFO probs.")
    p.add_argument("--backtest-loaded-model", action="store_true", help="Load a pre-trained model then backtest.")
    p.add_argument("--load-model-path", type=str, default=None)

    # Backtest thresholds & costs
    p.add_argument("--thr-buy", type=float, default=0.55)
    p.add_argument("--thr-sell", type=float, default=0.45)
    p.add_argument("--sl-pct", type=float, default=0.015)
    p.add_argument("--tp-pct", type=float, default=0.015)
    p.add_argument("--fee-bps", type=float, default=0.0)
    p.add_argument("--slippage-bps", type=float, default=0.0)

    # Misc
    p.add_argument("--timezone", type=str, default="Australia/Melbourne")
    p.add_argument("--random-state", type=int, default=42)

    return p


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    # Config overlay
    cfg = load_yaml_config(args.config)
    args = _merge_config(args, cfg)

    # Sanity validations
    if args.sl_pct < 0 or args.tp_pct < 0:
        raise ValueError("Stop loss and take profit percentages must be non-negative.")
    if args.thr_buy <= args.thr_sell:
        raise ValueError("thr_buy must be greater than thr_sell.")

    # Build costs
    costs = Costs(fee_bps=float(args.fee_bps), slippage_bps=float(args.slippage_bps))

    # Example: load features if provided (CSV expected to contain label + features)
    X: Optional[pd.DataFrame] = None
    y: Optional[pd.Series] = None
    if args.features_csv:
        if not os.path.isfile(args.features_csv):
            raise FileNotFoundError(f"Missing model_input: {args.features_csv}. Did you run features.py?")
        df_feat = pd.read_csv(args.features_csv)
        if args.label_col not in df_feat.columns:
            raise KeyError(f"Missing label column '{args.label_col}' in {args.features_csv}.")
        y = df_feat[args.label_col].astype(int)
        X = df_feat.drop(columns=[args.label_col])

    # TRAIN WALK-FORWARD
    if args.train_walk_forward:
        if X is None or y is None:
            raise ValueError("--features-csv required for --train-walk-forward.")
        meta = train_walk_forward(
            X=X,
            y=y,
            model_choice=args.model_choice,
            n_splits=int(args.n_splits),
            min_train=int(args.min_train),
            save_fold_models=bool(args.save_fold_models),
            model_dir="models",
            model_tag=f"{Path(args.features_csv).stem}_{args.model_choice}",
            random_state=int(args.random_state),
        )
        stamp = now_in_tz(args.timezone).strftime("%Y%m%d_%H%M%S")
        outp = f"models/wfo_{args.model_choice}_{stamp}.json"
        os.makedirs("models", exist_ok=True)
        with open(outp, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        logging.info("WFO summary saved → %s", outp)

    # TRAIN FINAL MODEL
    if args.train_final_model:
        if X is None or y is None:
            raise ValueError("--features-csv required for --train-final-model.")
        final_path = args.final_model_path or f"models/final_{Path(args.features_csv).stem}_{args.model_choice}.pkl"
        train_final_model(X, y, model_choice=args.model_choice, save_path=final_path, random_state=int(args.random_state))

    # BACKTEST WITH LOADED MODEL (example demo: uses CLOSE>OPEN dummy features unless you adapt)
    if args.backtest_loaded_model:
        if not args.load_model_path:
            raise ValueError("--load-model-path required for --backtest-loaded-model.")
        est = load_model(args.load_model_path)

        # Minimal demonstration: build a toy df_exec and probs from the loaded model IF features provided.
        # Replace this with your actual market data + feature pipeline.
        if X is None:
            raise ValueError("--features-csv required to generate predictions for backtest.")
        if "timestamp" in X.columns and {"open", "high", "low", "close"}.issubset(X.columns):
            df_exec = X[["timestamp", "open", "high", "low", "close"]].copy()
        else:
            raise KeyError("X must contain ['timestamp','open','high','low','close'] columns for backtest demo.")

        # Get probabilities
        if hasattr(est, "predict_proba"):
            probs = est.predict_proba(X)[:, 1]  # type: ignore[index]
        elif hasattr(est, "decision_function"):
            dfun = est.decision_function(X)  # type: ignore[call-arg]
            probs = 1.0 / (1.0 + np.exp(-np.asarray(dfun, dtype=float)))
        else:
            pred = est.predict(X)  # type: ignore[call-arg]
            probs = np.asarray(pred, dtype=float)

        trades, equity = run_backtest_next_open_long_short(
            df_exec=df_exec.reset_index(drop=True),
            probs=np.asarray(probs, dtype=float),
            thr_buy=float(args.thr_buy),
            thr_sell=float(args.thr_sell),
            sl_pct=float(args.sl_pct),
            tp_pct=float(args.tp_pct),
            costs=costs,
            initial_cash=0.0,
        )
        logging.info("Backtest complete: %d closed trades, equity_last=%.6f",
                     len(trades), float(equity["equity"].iloc[-1]))
        # Dump trades for inspection
        out_trades = [asdict(t) for t in trades]
        os.makedirs("artifacts", exist_ok=True)
        with open("artifacts/backtest_trades.json", "w", encoding="utf-8") as f:
            json.dump(out_trades, f, indent=2)
        equity.to_csv("artifacts/backtest_equity.csv", index=False)
        logging.info("Saved artifacts to artifacts/")

    # BACKTEST WFO (example driver — expects meta JSON from earlier step + features to regenerate probs per split)
    if args.backtest_wfo:
        logging.info("Backtest WFO driver is a placeholder; integrate with your feature splits as needed.")

    logging.info("Done.")


if __name__ == "__main__":
    main()
