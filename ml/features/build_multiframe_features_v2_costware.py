# path: ml/build_multitimeframe_features.py
# Reads raw candles from:
#   {RAW_DIR}/raw/(SYMBOL)_USD/{parquet,csv}/(SYMBOL)_{tf}_raw.{parquet,csv}
#   OR: {RAW_DIR}/raw/{symbol_lower}_usd/{symbol_lower}_usd_{tf}_raw.{parquet,csv}
# Writes features to FLAT layout:
#   {OUT_DIR}/processed/features/{symbol_lower}_usd/{symbol_lower}_usd_{tf}_features.{parquet,csv}
#
# This is a thin CLI wrapper around `ml.features.engineer_features`, which
# holds the *canonical* feature and label logic (including cost-aware labels).

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Literal, cast

import pandas as pd

from .features import engineer_features  # canonical implementation


# ------------------------------
# Symbol / timeframe normalization
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
        "15m": "15m", "15min": "15m",
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
# Load raw candles
# ------------------------------

def _load_raw(raw_root: Path, symbol: str, tf: str) -> pd.DataFrame:
    """
    Load raw OHLCV for a given symbol/timeframe from the expected locations.
    """
    # Canonical upper-case layout
    pq, csv = _raw_paths(raw_root, symbol, tf)

    # Lowercase alt layout (for flexibility)
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

    # Normalise to UTC index
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
    base = f"{sym_lower}_usd_{tf_tok}_features"
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
# CLI config & argument parsing
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
    # engineer_features config
    fee_bps_per_side: float
    extra_slippage_bps: float
    forward_horizon_bars: int
    return_lags: Tuple[int, ...]
    vol_windows: Tuple[int, ...]
    rsi_window: int
    expected_move_horizon: int
    include_proxies: bool
    include_trade_flag: bool
    trade_move_threshold: float
    trade_spread_threshold: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build features and write to flat processed layout (using ml.features.engineer_features).")
    p.add_argument(
        "--symbols",
        nargs="+",
        required=True,
        help='Symbols like "BTC_USD", "BTC/USD". USD is enforced for paths.',
    )
    p.add_argument(
        "--timeframes",
        nargs="+",
        help="Timeframes like 1m 5m 15m 1h 4h 1d.",
    )
    p.add_argument(
        "--timeframe",
        nargs="+",
        help="Alias for --timeframes (for backwards compatibility).",
    )
    p.add_argument(
        "--out-dir",
        default="data",
        help="Root folder for processed outputs (and raw if --raw-dir omitted).",
    )
    p.add_argument(
        "--raw-dir",
        default=None,
        help="Root folder for raw inputs; defaults to --out-dir when omitted.",
    )
    p.add_argument(
        "--no-csv",
        action="store_true",
        help="Skip writing CSV outputs.",
    )
    p.add_argument(
        "--no-parquet",
        action="store_true",
        help="Skip writing Parquet outputs.",
    )
    p.add_argument(
        "--parquet-compression",
        choices=["snappy", "gzip", "brotli", "lz4", "zstd", "none"],
        default="snappy",
    )
    p.add_argument(
        "--target-tz",
        default=None,
        help="Convert timestamp index to this timezone before writing (e.g., 'Australia/Melbourne').",
    )
    # cost-aware label config
    p.add_argument(
        "--fee-bps-per-side",
        type=float,
        default=25.0,
        help=(
            "Trading fee in basis points per side (entry & exit). "
            "Default 25.0 (Alpaca crypto Tier 1 taker ≈ 0.25% per leg)."
        ),
    )
    p.add_argument(
        "--extra-slippage-bps",
        type=float,
        default=0.0,
        help="Extra constant slippage per round-trip in basis points.",
    )
    p.add_argument(
        "--forward-horizon-bars",
        type=int,
        default=15,
        help="Number of bars for forward return/label horizon. Default 15.",
    )
    # feature-engineering config passthrough
    p.add_argument(
        "--return-lags",
        nargs="+",
        type=int,
        default=[1, 5, 20],
        help="Return lags in bars, e.g. 1 5 20.",
    )
    p.add_argument(
        "--vol-windows",
        nargs="+",
        type=int,
        default=[20],
        help="Volatility windows in bars, e.g. 20 60.",
    )
    p.add_argument(
        "--rsi-window",
        type=int,
        default=14,
        help="RSI lookback window in bars. Default 14.",
    )
    p.add_argument(
        "--expected-move-horizon",
        type=int,
        default=3,
        help="Horizon in bars for expected_move_* feature. Default 3.",
    )
    p.add_argument(
        "--no-proxies",
        action="store_true",
        help="Disable microstructure proxies (spread/slippage).",
    )
    p.add_argument(
        "--no-trade-flag",
        action="store_true",
        help="Disable is_trade_viable flag.",
    )
    p.add_argument(
        "--trade-move-threshold",
        type=float,
        default=0.009,
        help="Absolute move threshold for is_trade_viable.",
    )
    p.add_argument(
        "--trade-spread-threshold",
        type=float,
        default=0.002,
        help="Max spread_proxy_pct for is_trade_viable.",
    )
    return p.parse_args()


def _lock_usd(symbol: str) -> str:
    return f"{_base_symbol(symbol).upper()}/USD"


def main() -> int:
    args = parse_args()

    timeframes = args.timeframes or args.timeframe
    if not timeframes:
        raise SystemExit("Provide --timeframes or --timeframe with at least one value (e.g., 1m).")

    comp: Optional[ParquetComp] = None if args.parquet_compression == "none" else cast(
        ParquetComp, args.parquet_compression
    )
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
        fee_bps_per_side=float(args.fee_bps_per_side),
        extra_slippage_bps=float(args.extra_slippage_bps),
        forward_horizon_bars=int(args.forward_horizon_bars),
        return_lags=tuple(args.return_lags),
        vol_windows=tuple(args.vol_windows),
        rsi_window=int(args.rsi_window),
        expected_move_horizon=int(args.expected_move_horizon),
        include_proxies=not args.no_proxies,
        include_trade_flag=not args.no_trade_flag,
        trade_move_threshold=float(args.trade_move_threshold),
        trade_spread_threshold=float(args.trade_spread_threshold),
    )

    if not cfg.write_parquet and not cfg.write_csv:
        raise SystemExit("Both outputs disabled; enable at least one of Parquet or CSV.")

    for sym_in in cfg.symbols:
        sym = _lock_usd(sym_in)
        for tf in cfg.timeframes:
            raw = _load_raw(cfg.raw_dir, sym, tf)

            feats = engineer_features(
                raw,
                return_lags=cfg.return_lags,
                vol_windows=cfg.vol_windows,
                rsi_window=cfg.rsi_window,
                expected_move_horizon=cfg.expected_move_horizon,
                include_proxies=cfg.include_proxies,
                include_trade_flag=cfg.include_trade_flag,
                trade_move_threshold=cfg.trade_move_threshold,
                trade_spread_threshold=cfg.trade_spread_threshold,
                forward_horizon_bars=cfg.forward_horizon_bars,
                fee_bps_per_side=cfg.fee_bps_per_side,
                extra_slippage_bps=cfg.extra_slippage_bps,
            )

            if cfg.target_tz:
                # Outputs in local time if requested; internal logic stays UTC.
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
                base = f"{sym_lower}_usd_{tf_tok}_features"
                csv_path = feat_dir / f"{base}.csv"
                feats.reset_index().to_csv(csv_path, index=False)

            print(f"[features] {sym_upper} {tf_tok} → {feat_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
