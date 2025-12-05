# path: ml/build_multitimeframe_features.py
# Reads raw from:
#   {RAW_DIR}/raw/(SYMBOL)_USD/{parquet,csv}/(SYMBOL)_{tf}_raw.{parquet,csv}
#   OR: {RAW_DIR}/raw/{symbol_lower}_usd/{symbol_lower}_usd_{tf}_raw.{parquet,csv}
# Writes features to FLAT layout:
#   {OUT_DIR}/processed/features/{symbol_lower}_usd/{symbol_lower}_usd_{tf}_features.{parquet,csv}

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Literal, cast

import numpy as np
import pandas as pd

# ------------------------------
# Symbol/timeframe normalization
# ------------------------------

def _base_symbol(symbol: str) -> str:
    s = symbol.strip().upper().replace("-", "/").replace("_", "/")
    parts = [p for p in s.split("/") if p]
    return parts[0] if parts else s

def _norm_tf(tf: str) -> str:
    t = tf.strip().lower()
    table = {
        "1": "1m", "1m": "1m", "1min": "1m", "1minute": "1m",
        "5m": "5m", "5min": "5m", "5minute": "5m",
        "1h": "1h", "1hour": "1h", "60min": "1h",
        "4h": "4h", "4hour": "4h",
        "1d": "1d", "1day": "1d", "day": "1d",
    }
    return table.get(t, t)

# ------------------------------
# I/O paths (raw in, processed out)
# ------------------------------

def _raw_paths(raw_root: Path, symbol: str, tf: str) -> Tuple[Path, Path]:
    base = _base_symbol(symbol)
    sym_upper = base.upper()
    tf_tok = _norm_tf(tf)
    root = raw_root / "raw" / f"{sym_upper}_USD"
    pq = root / "parquet" / f"{sym_upper}_{tf_tok}_raw.parquet"
    csv = root / "csv" / f"{sym_upper}_{tf_tok}_raw.csv"
    return pq, csv

def _processed_features_dir(out_root: Path, symbol: str, tf: str) -> Tuple[Path, str, str, str]:
    # FLAT layout: processed/features/{sym_lower}_usd/
    base = _base_symbol(symbol)
    sym_upper = base.upper()
    sym_lower = base.lower()
    tf_tok = _norm_tf(tf)
    d = out_root / "processed" / "features" / f"{sym_lower}_usd"
    d.mkdir(parents=True, exist_ok=True)
    return d, sym_upper, sym_lower, tf_tok

# ------------------------------
# Feature engineering
# ------------------------------

def safe_pct_change(s: pd.Series, periods: int) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return s.pct_change(periods=periods)

def _rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(length, min_periods=length).mean()
    avg_loss = loss.rolling(length, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute returns/volatility/trend/RSI/microstructure/time-of-day + simple forward label.
    """
    out = df.copy()

    close = pd.to_numeric(out["close"], errors="coerce")
    vol = pd.to_numeric(out["volume"], errors="coerce")

    # Returns
    out["returns_1"] = safe_pct_change(close, 1)
    out["returns_5"] = safe_pct_change(close, 5)
    out["returns_15"] = safe_pct_change(close, 15)
    out["returns_60"] = safe_pct_change(close, 60)
    out["returns_240"] = safe_pct_change(close, 240)
    out["returns_1440"] = safe_pct_change(close, 1440)

    # Volatility / volume regime
    ret_1 = close.pct_change()
    out["rolling_vol_20"] = ret_1.rolling(20).std()
    vol_mean_20 = vol.rolling(20, min_periods=20).mean()
    vol_std_20 = vol.rolling(20, min_periods=20).std()
    out["volume_zscore_20"] = (vol - vol_mean_20) / vol_std_20.replace(0.0, np.nan)

    # Trend / EMA
    ema_20 = close.ewm(span=20, adjust=False, min_periods=20).mean()
    ema_50 = close.ewm(span=50, adjust=False, min_periods=50).mean()
    out["ema_20"] = ema_20
    out["ema_50"] = ema_50
    out["dist_close_ema20"] = (close - ema_20) / ema_20.replace(0.0, np.nan)
    out["dist_ema20_ema50"] = (ema_20 - ema_50) / ema_50.replace(0.0, np.nan)

    # RSI
    out["rsi_14"] = _rsi(close, 14)

    # Microstructure proxies
    high = pd.to_numeric(out["high"], errors="coerce")
    low = pd.to_numeric(out["low"], errors="coerce")
    open_ = pd.to_numeric(out["open"], errors="coerce")
    out["spread_proxy_pct"] = (high - low) / close.replace(0.0, np.nan)
    out["slippage_proxy"] = (high - low) / open_.replace(0.0, np.nan)

    # Simple forward proxies/labels
    out["expected_move_3"] = out["close"].shift(-3) / out["close"] - 1.0
    out["is_trade_viable"] = ((out["expected_move_3"].abs() > 0.009) & (out["spread_proxy_pct"] < 0.002)).astype(int)

    # Time-of-day / calendar
    idx = out.index
    if isinstance(idx, pd.DatetimeIndex):
        minute_of_day = idx.hour * 60 + idx.minute
        angle = 2.0 * np.pi * (minute_of_day / 1440.0)
        out["minute_of_day_sin"] = np.sin(angle)
        out["minute_of_day_cos"] = np.cos(angle)
        out["day_of_week"] = idx.weekday
    else:
        out["minute_of_day_sin"] = np.nan
        out["minute_of_day_cos"] = np.nan
        out["day_of_week"] = np.nan

    # Forward 15m targets
    out["y_ret_fwd_15m"] = close.shift(-15) / close - 1.0
    thr = 0.001
    ret_fwd = out["y_ret_fwd_15m"]
    out["y_dir_fwd_15m"] = np.where(ret_fwd > thr, 1, np.where(ret_fwd < -thr, -1, 0))

    return out

# ------------------------------
# Load raw candles
# ------------------------------

def _load_raw(raw_root: Path, symbol: str, tf: str) -> pd.DataFrame:
    """
    Load raw OHLCV for a given symbol/timeframe.
    """
    # Original upper-case layout
    pq, csv = _raw_paths(raw_root, symbol, tf)

    # Lowercase alt layout (your screenshot)
    base = _base_symbol(symbol)
    sym_lower = base.lower()
    tf_tok = _norm_tf(tf)
    alt_dir = raw_root / "raw" / f"{sym_lower}_usd"
    alt_prefix = f"{sym_lower}_usd_{tf_tok}_raw"
    alt_pq = alt_dir / f"{alt_prefix}.parquet"
    alt_csv = alt_dir / f"{alt_prefix}.csv"

    if pq.exists():
        df = pd.read_parquet(pq)
    elif csv.exists():
        df = pd.read_csv(csv)
    elif alt_pq.exists():
        df = pd.read_parquet(alt_pq)
    elif alt_csv.exists():
        df = pd.read_csv(alt_csv)
    else:
        raise FileNotFoundError(
            "No raw data found for "
            f"{symbol} {tf} at any of:\n"
            f"  {pq}\n  {csv}\n  {alt_pq}\n  {alt_csv}"
        )

    # Normalize to UTC index for feature logic
    if "timestamp" in df.columns:
        idx = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.drop(columns=["timestamp"])
        df.index = idx
        df.index.name = "timestamp"
    else:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Raw data must include 'timestamp' or have a DatetimeIndex.")
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

    req = ["open", "high", "low", "close", "volume"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Raw data missing required columns: {missing}")

    return df.sort_index()

# ------------------------------
# Writers
# ------------------------------

ParquetComp = Literal["snappy", "gzip", "brotli", "lz4", "zstd"]

def _write_features(
    df_feat: pd.DataFrame,
    out_dir: Path,
    sym_lower: str,
    tf_tok: str,
    *,
    write_csv: bool,
    parquet_compression: Optional[ParquetComp],
) -> Tuple[Path, Optional[Path]]:
    # filename base now includes _usd after the symbol
    base = f"{sym_lower}_usd_{tf_tok}_features"  # <-- changed
    pq = out_dir / f"{base}.parquet"
    if parquet_compression is None:
        df_feat.to_parquet(pq, index=True)
    else:
        df_feat.to_parquet(pq, index=True, compression=parquet_compression)
    csv_path: Optional[Path] = None
    if write_csv:
        csv_path = out_dir / f"{base}.csv"
        df_feat.reset_index().to_csv(csv_path, index=False)
    return pq, csv_path

# ------------------------------
# CLI
# ------------------------------

@dataclass
class Config:
    symbols: List[str]
    timeframes: List[str]
    out_dir: Path
    raw_dir: Path
    write_csv: bool
    write_parquet: bool
    parquet_compression: Optional[ParquetComp]
    target_tz: Optional[str]

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build features and write to flat processed layout.")
    p.add_argument("--symbols", nargs="+", required=True, help='Symbols like "BTC_USD", "BTC/USD". USD is enforced for paths.')
    p.add_argument("--timeframes", nargs="+", help="e.g., 1m 5m 1h 4h 1d")
    p.add_argument("--timeframe", nargs="+", help="alias of --timeframes")
    p.add_argument("--out-dir", default="data", help="Root folder for processed outputs (and raw if --raw-dir omitted)")
    p.add_argument("--raw-dir", default=None, help="Root folder for raw inputs; defaults to --out-dir when omitted")
    p.add_argument("--no-csv", action="store_true", help="Skip writing CSV")
    p.add_argument("--no-parquet", action="store_true", help="Skip writing Parquet")
    p.add_argument("--parquet-compression", choices=["snappy", "gzip", "brotli", "lz4", "zstd", "none"], default="snappy")
    p.add_argument("--target-tz", default=None, help="Convert timestamp index to this timezone before writing (e.g., 'Australia/Melbourne')")
    return p.parse_args()

def _lock_usd(symbol: str) -> str:
    return f"{_base_symbol(symbol).upper()}/USD"

def main() -> int:
    args = parse_args()

    timeframes = args.timeframes or args.timeframe
    if not timeframes:
        raise SystemExit("Provide --timeframes or --timeframe with at least one value (e.g., 1m).")

    comp: Optional[ParquetComp] = None if args.parquet_compression == "none" else cast(ParquetComp, args.parquet_compression)
    out_dir = Path(args.out_dir).resolve()
    raw_dir = Path(args.raw_dir).resolve() if args.raw_dir else out_dir

    cfg = Config(
        symbols=args.symbols,
        timeframes=timeframes,
        out_dir=out_dir,
        raw_dir=raw_dir,
        write_csv=not args.no_csv,
        write_parquet=not args.no_parquet,
        parquet_compression=comp,
        target_tz=args.target_tz,
    )
    if not cfg.write_parquet and not cfg.write_csv:
        raise SystemExit("Both outputs disabled; enable at least one of Parquet or CSV.")

    for sym_in in cfg.symbols:
        sym = _lock_usd(sym_in)
        for tf in cfg.timeframes:
            raw = _load_raw(cfg.raw_dir, sym, tf)
            feats = engineer_features(raw)
            if cfg.target_tz:
                # why: outputs in local time if requested; internal logic stays UTC
                feats = feats.tz_convert(cfg.target_tz)

            feat_dir, sym_upper, sym_lower, tf_tok = _processed_features_dir(cfg.out_dir, sym, tf)

            if cfg.write_parquet:
                _write_features(
                    feats,
                    feat_dir,
                    sym_lower,
                    tf_tok,
                    write_csv=cfg.write_csv,
                    parquet_compression=cfg.parquet_compression,
                )
            else:
                base = f"{sym_lower}_usd_{tf_tok}_features"  # <-- changed
                csv_path = feat_dir / f"{base}.csv"
                feats.reset_index().to_csv(csv_path, index=False)

            print(f"[features] {sym_upper} {tf_tok} → {feat_dir}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
