# path: ml/features.py
# Feature engineering with USD-locked outputs, typed config handling, CLI args,
# JSON config, README + metadata, and optional data parquet.

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union, cast

import numpy as np
import pandas as pd


# ------------------------------
# Path & symbol helpers
# ------------------------------

def _symbol_variants(symbol: str) -> List[str]:
    clean = re.sub(r"\s+", "", symbol)
    parts = re.split(r"[/\-_]", clean.upper())
    if len(parts) == 2:
        base, quote = parts
    else:
        base, quote = clean.upper(), "USD"
    variants = [
        f"{base}-{quote}",
        f"{base}_{quote}",
        f"{base}{quote}",
        f"{base}-{quote}".lower(),
        f"{base}_{quote}".lower(),
        f"{base}{quote}".lower(),
    ]
    return list(dict.fromkeys(variants))


def _base_symbol(symbol: str) -> str:
    """Extract base asset only; outputs are locked to USD regardless of input."""
    s = re.sub(r"\s+", "", symbol)
    parts = re.split(r"[/\-_]", s)
    return parts[0] if parts and parts[0] else s


def _norm_tf(tf: str) -> str:
    """Normalize timeframe token (e.g., '1m', '5m', '1h')."""
    t = tf.strip().lower()
    t = t.replace("minute", "m").replace("min", "m").replace("hour", "h").replace("hr", "h")
    m = re.fullmatch(r"(\d+)\s*([mhds])", t)
    if m:
        return f"{m.group(1)}{m.group(2)}"
    table = {
        "1min": "1m", "3min": "3m", "5min": "5m", "15min": "15m", "30min": "30m",
        "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m",
        "60min": "1h", "1h": "1h", "2h": "2h", "4h": "4h",
        "1d": "1d",
    }
    return table.get(t, t)


def _candidate_files(raw_dir: Path, symbol_variants: List[str], timeframe: str) -> Iterable[Path]:
    for v in symbol_variants:
        yield raw_dir / f"{v}_{timeframe}.parquet"
        yield raw_dir / f"{v}_{timeframe}.csv"
        yield raw_dir / timeframe / f"{v}.parquet"
        yield raw_dir / timeframe / f"{v}.csv"
    for v in symbol_variants:
        sym_dir = raw_dir / v
        if sym_dir.is_dir():
            yield sym_dir / f"{timeframe}.parquet"
            yield sym_dir / f"{timeframe}.csv"
    for v in symbol_variants:
        sym_dir = raw_dir / v
        if sym_dir.is_dir():
            for ext in ("parquet", "csv"):
                yield from sym_dir.glob(f"*{timeframe}*.{ext}")
    for ext in ("parquet", "csv"):
        yield from raw_dir.rglob(f"*{timeframe}*.{ext}")


def find_raw_file(raw_dir: Path, symbol: str, timeframe: str) -> Path:
    variants = _symbol_variants(symbol)
    for cand in _candidate_files(raw_dir, variants, timeframe):
        if cand.exists():
            return cand
    raise FileNotFoundError(
        f"Could not find raw OHLCV for symbol='{symbol}' timeframe='{timeframe}' in '{raw_dir}'."
    )


# ------------------------------
# IO & time utilities
# ------------------------------

def _read_parquet_or_csv(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file extension for {path}")


def _ensure_timestamp_index(
    df: pd.DataFrame,
    target_tz: str = "UTC",
    timestamp_col_candidates: Tuple[str, ...] = ("timestamp", "date", "datetime"),
) -> pd.DataFrame:
    ts_col: Optional[str] = None
    for c in timestamp_col_candidates:
        if c in df.columns:
            ts_col = c
            break

    if ts_col is None and isinstance(df.index, pd.DatetimeIndex):
        ts_idx: pd.DatetimeIndex = pd.DatetimeIndex(df.index)
        ts_idx = ts_idx.tz_localize(target_tz) if ts_idx.tz is None else ts_idx.tz_convert(target_tz)
        out = df.copy()
        out.index = ts_idx
        out.index.name = "timestamp"
        out.sort_index(inplace=True)
        return out

    if ts_col is None:
        raise ValueError("No datetime info found (need DatetimeIndex or timestamp/date/datetime column).")

    ts_series = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    if ts_series.isna().any():
        raise ValueError(f"Found {int(ts_series.isna().sum())} unparsable timestamps in '{ts_col}'.")
    ts_series = ts_series.dt.tz_convert(target_tz)
    out = df.drop(columns=[ts_col]).copy()
    out.index = pd.DatetimeIndex(ts_series, name="timestamp")
    out.sort_index(inplace=True)
    return out


def load_raw_ohlcv(raw_dir: Path, symbol: str, timeframe: str, target_tz: str) -> Tuple[pd.DataFrame, Path]:
    """Return dataframe and the resolved source path; recorded for metadata."""
    src_path = find_raw_file(raw_dir, symbol, timeframe)
    df = _read_parquet_or_csv(src_path)

    lower = {c.lower(): c for c in df.columns}
    rename: Dict[str, str] = {}
    for want in ["open", "high", "low", "close", "volume", "timestamp", "date", "datetime", "time", "ts"]:
        if want not in df.columns:
            for k in list(lower.keys()):
                if k == want:
                    rename[lower[k]] = want
    for alias in ("time", "ts"):
        if alias in df.columns and "timestamp" not in df.columns:
            rename[alias] = "timestamp"

    if rename:
        df = df.rename(columns=rename)

    df = _ensure_timestamp_index(df, target_tz=target_tz)

    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required OHLC columns {missing} in {src_path}")

    front = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    rest = [c for c in df.columns if c not in front]
    return df[front + rest].copy(), src_path


# ------------------------------
# Core math helpers
# ------------------------------

def safe_pct_change(s: pd.Series, periods: int = 1) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return s.pct_change(periods=periods)


def safe_div(numer: pd.Series, denom: pd.Series) -> pd.Series:
    denom = pd.to_numeric(denom, errors="coerce").replace(0.0, np.nan)
    numer = pd.to_numeric(numer, errors="coerce")
    return numer / denom


# ------------------------------
# Feature primitives
# ------------------------------

def compute_returns(close: pd.Series, lags: Sequence[int]) -> pd.DataFrame:
    out = {f"returns_{p}": safe_pct_change(close, p) for p in lags}
    return pd.DataFrame(out, index=close.index)


def compute_volatility(close: pd.Series, windows: Sequence[int]) -> pd.DataFrame:
    pct = close.pct_change()
    out = {f"rolling_vol_{w}": pct.rolling(w, min_periods=w).std() for w in windows}
    return pd.DataFrame(out, index=close.index)


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = pd.to_numeric(close, errors="coerce").diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(window, min_periods=window).mean()
    avg_loss = loss.rolling(window, min_periods=window).mean()
    rs = safe_div(avg_gain, avg_loss)
    return 100.0 - (100.0 / (1.0 + rs))


def compute_microstructure_proxies(
    high: pd.Series, low: pd.Series, open_: pd.Series, close: pd.Series
) -> pd.DataFrame:
    rng = pd.to_numeric(high, errors="coerce") - pd.to_numeric(low, errors="coerce")
    spread_proxy_pct = safe_div(rng, close)
    slippage_proxy = safe_div(rng, open_)
    return pd.DataFrame({"spread_proxy_pct": spread_proxy_pct, "slippage_proxy": slippage_proxy}, index=close.index)


def compute_expected_move(close: pd.Series, horizon: int = 3) -> pd.Series:
    c = pd.to_numeric(close, errors="coerce")
    return c.shift(-horizon) / c - 1.0


def compute_trade_viability(
    expected_move: pd.Series,
    spread_proxy_pct: pd.Series,
    move_threshold: float = 0.009,
    spread_threshold: float = 0.002,
) -> pd.Series:
    cond = (expected_move.abs() > move_threshold) & (spread_proxy_pct < spread_threshold)
    return cond.astype(int)


# ------------------------------
# Unified feature engineer
# ------------------------------

def engineer_features(
    df: pd.DataFrame,
    *,
    return_lags: Sequence[int] = (1, 5, 20),
    vol_windows: Sequence[int] = (20,),
    rsi_window: int = 14,
    expected_move_horizon: int = 3,
    include_proxies: bool = True,
    include_trade_flag: bool = True,
    trade_move_threshold: float = 0.009,
    trade_spread_threshold: float = 0.002,
) -> pd.DataFrame:
    out = df.copy()
    open_ = pd.to_numeric(out["open"], errors="coerce")
    high = pd.to_numeric(out["high"], errors="coerce")
    low = pd.to_numeric(out["low"], errors="coerce")
    close = pd.to_numeric(out["close"], errors="coerce")

    out = out.join(compute_returns(close, return_lags))
    out = out.join(compute_volatility(close, vol_windows))
    out["rsi_14"] = compute_rsi(close, window=rsi_window)

    proxies = None
    if include_proxies:
        proxies = compute_microstructure_proxies(high, low, open_, close)
        out = out.join(proxies)

    out["expected_move_3"] = compute_expected_move(close, horizon=expected_move_horizon)

    if include_trade_flag:
        spread_series = proxies["spread_proxy_pct"] if proxies is not None else safe_div(high - low, close)
        out["is_trade_viable"] = compute_trade_viability(
            expected_move=out["expected_move_3"],
            spread_proxy_pct=spread_series,
            move_threshold=trade_move_threshold,
            spread_threshold=trade_spread_threshold,
        )
    return out


# ------------------------------
# Output helpers: paths, hashing, docs
# ------------------------------

def _target_dir(out_root: Path, symbol: str, timeframe: str) -> Tuple[Path, str, str, str]:
    base = _base_symbol(symbol)
    sym_upper = base.upper()
    sym_lower = base.lower()
    tf_token = _norm_tf(timeframe)
    target_dir = out_root / "processed" / "features" / f"{sym_upper}_USD" / f"{sym_lower}_{tf_token}_features"
    target_dir.mkdir(parents=True, exist_ok=True)  # ensure dir exists
    return target_dir, sym_upper, sym_lower, tf_token


def _sha256_file(path: Path, chunk_size: int = 2**20) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None  # why: allow runs without read permission


def _write_readme(target_dir: Path, readme_text: str) -> None:
    (target_dir / "README.md").write_text(readme_text, encoding="utf-8")


def _write_meta(target_dir: Path, meta: Dict[str, object]) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    meta_path = target_dir / f"run_{stamp}.json"
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    return meta_path


def _render_readme(symbol_upper: str, symbol_lower: str, tf_token: str, settings: Dict[str, object], meta: Dict[str, object]) -> str:
    # capture run settings for backtests/repro
    lines = [
        f"# {symbol_upper}_USD / {symbol_lower}_{tf_token}_features",
        "",
        "This folder contains engineered features and optional cleaned data for the specified run.",
        "",
        "## Files",
        f"- `{symbol_lower}_{tf_token}_features.parquet`",
        f"- `{symbol_lower}_{tf_token}_features.csv`",
        f"- `{symbol_lower}_{tf_token}_data.parquet` (optional)",
        "- `run_*.json` (run metadata snapshots)",
        "",
        "## Settings",
        "```json",
        json.dumps(settings, indent=2, sort_keys=True),
        "```",
        "",
        "## Metadata (latest run snapshot)",
        "```json",
        json.dumps(meta, indent=2, sort_keys=True),
        "```",
        "",
        "## Notes",
        "- Output path is locked to `_USD` regardless of the input quote.",
        "- Timestamps are stored in the configured target timezone.",
    ]
    return "\n".join(lines)


# ------------------------------
# Outputs: features + optional data parquet + docs
# ------------------------------

def write_outputs(
    symbol: str,
    timeframe: str,
    df_features: pd.DataFrame,
    out_dir: Path,
    dropna: bool = False,
    *,
    df_data: Optional[pd.DataFrame] = None,
    settings: Optional[Dict[str, object]] = None,
    meta: Optional[Dict[str, object]] = None,
    write_readme: bool = True,
) -> Path:
    """
    Writes to the USD-locked directory:
      {out_dir}/processed/features/(SYMBOL)_USD/(symbol)_{tf}_features/
        - (symbol)_{tf}_features.parquet
        - (symbol)_{tf}_features.csv
        - (symbol)_{tf}_data.parquet      [if df_data is provided]
        - README.md                        [documenting settings + metadata]
        - run_YYYYMMDDTHHMMSSZ.json        [metadata snapshot]
    Returns the features .parquet path.
    """
    target_dir, sym_upper, sym_lower, tf_token = _target_dir(out_dir, symbol, timeframe)

    basename = f"{sym_lower}_{tf_token}_features"
    pq_path = target_dir / f"{basename}.parquet"
    csv_path = target_dir / f"{basename}.csv"

    out_df = df_features.dropna() if dropna else df_features
    out_df.to_parquet(pq_path)
    out_df.reset_index().to_csv(csv_path, index=False)

    if df_data is not None:
        data_path = target_dir / f"{sym_lower}_{tf_token}_data.parquet"
        df_data.to_parquet(data_path)

    _settings = settings or {}
    _meta = meta or {}
    if write_readme:
        readme_text = _render_readme(sym_upper, sym_lower, tf_token, _settings, _meta)
        _write_readme(target_dir, readme_text)
    _write_meta(target_dir, _meta)

    return pq_path


# ------------------------------
# Config handling (JSON + CLI merge) with strict typing
# ------------------------------

def _parse_int_list(s: Optional[str], default: Sequence[int]) -> List[int]:
    if not s:
        return list(default)
    return [int(x.strip()) for x in s.split(",") if x.strip() != ""]


def _load_json_config(path: Optional[str]) -> Dict[str, object]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        return cast(Dict[str, object], json.load(f))


def _as_str(v: object, default: str) -> str:
    if isinstance(v, (str, Path)):
        return str(v)
    return default


def _as_bool(v: object, default: bool) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        vt = v.strip().lower()
        if vt in {"1", "true", "yes", "y", "on"}:
            return True
        if vt in {"0", "false", "no", "n", "off"}:
            return False
    return default


def _as_int(v: object, default: int) -> int:
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, str):
        try:
            return int(v.strip())
        except ValueError:
            return default
    return default


def _as_float(v: object, default: float) -> float:
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v.strip())
        except ValueError:
            return default
    return default


def _as_str_list(v: object, default: Sequence[str]) -> List[str]:
    if isinstance(v, (list, tuple)):
        out: List[str] = []
        for item in v:
            if isinstance(item, (str, Path)):
                out.append(str(item))
        return out if out else list(default)
    if isinstance(v, str):
        return [v]
    return list(default)


def _as_int_seq(v: object, default: Sequence[int]) -> List[int]:
    if isinstance(v, (list, tuple)):
        out: List[int] = []
        for item in v:
            if isinstance(item, int):
                out.append(item)
            elif isinstance(item, float) and item.is_integer():
                out.append(int(item))
            elif isinstance(item, str):
                try:
                    out.append(int(item.strip()))
                except ValueError:
                    pass
        return out if out else list(default)
    if isinstance(v, str):
        return _parse_int_list(v, default)
    return list(default)


@dataclass
class MergedConfig:
    symbols: List[str]
    timeframes: List[str]
    raw_dir: str
    out_dir: str
    target_tz: str
    dropna: bool
    rsi_window: int
    expected_move_horizon: int
    return_lags: List[int]
    vol_windows: List[int]
    trade_move_threshold: float
    trade_spread_threshold: float
    include_proxies: bool
    include_trade_flag: bool
    skip_data: bool
    config_source: Optional[str]


def _merge_config(cli: argparse.Namespace, cfg: Dict[str, object]) -> MergedConfig:
    # Symbols & timeframes
    sym_cli = _as_str_list(getattr(cli, "symbols", None), [])
    tf_cli = _as_str_list(getattr(cli, "timeframes", None), [])
    sym_cfg = _as_str_list(cfg.get("symbols", []), [])
    tf_cfg = _as_str_list(cfg.get("timeframes", []), [])
    symbols = sym_cli or sym_cfg
    timeframes = tf_cli or tf_cfg

    # General
    raw_dir = _as_str(getattr(cli, "raw_dir", None), _as_str(cfg.get("raw_dir", "data"), "data"))
    out_dir = _as_str(getattr(cli, "out_dir", None), _as_str(cfg.get("out_dir", "data"), "data"))
    target_tz = _as_str(getattr(cli, "target_tz", None), _as_str(cfg.get("target_tz", "UTC"), "UTC"))
    dropna = bool(getattr(cli, "dropna", False)) or _as_bool(cfg.get("dropna", False), False)

    # Feature args
    rsi_window = _as_int(getattr(cli, "rsi", None), _as_int(cfg.get("rsi_window", 14), 14))
    expected_move_horizon = _as_int(getattr(cli, "em", None), _as_int(cfg.get("expected_move_horizon", 3), 3))
    return_lags = _parse_int_list(getattr(cli, "returns", None), _as_int_seq(cfg.get("return_lags", [1, 5, 20]), [1, 5, 20]))
    vol_windows = _parse_int_list(getattr(cli, "vol", None), _as_int_seq(cfg.get("vol_windows", [20]), [20]))
    trade_move_threshold = _as_float(getattr(cli, "move_th", None), _as_float(cfg.get("trade_move_threshold", 0.009), 0.009))
    trade_spread_threshold = _as_float(getattr(cli, "spread_th", None), _as_float(cfg.get("trade_spread_threshold", 0.002), 0.002))
    include_proxies = not bool(getattr(cli, "no_proxies", False))
    if "include_proxies" in cfg:
        include_proxies = _as_bool(cfg.get("include_proxies", include_proxies), include_proxies)
    include_trade_flag = not bool(getattr(cli, "no_flag", False))
    if "include_trade_flag" in cfg:
        include_trade_flag = _as_bool(cfg.get("include_trade_flag", include_trade_flag), include_trade_flag)

    # Outputs
    skip_data = bool(getattr(cli, "skip_data", False))
    if "skip_data" in cfg:
        skip_data = _as_bool(cfg.get("skip_data", skip_data), skip_data)

    return MergedConfig(
        symbols=symbols,
        timeframes=timeframes,
        raw_dir=raw_dir,
        out_dir=out_dir,
        target_tz=target_tz,
        dropna=dropna,
        rsi_window=rsi_window,
        expected_move_horizon=expected_move_horizon,
        return_lags=return_lags,
        vol_windows=vol_windows,
        trade_move_threshold=trade_move_threshold,
        trade_spread_threshold=trade_spread_threshold,
        include_proxies=include_proxies,
        include_trade_flag=include_trade_flag,
        skip_data=skip_data,
        config_source=getattr(cli, "config", None),
    )


# ------------------------------
# CLI
# ------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate ML features from OHLCV data.")
    p.add_argument("--symbols", nargs="+", help='Symbols like "BTC/USD" or "BTC_USD". If --config provided, optional.')
    p.add_argument("--timeframes", nargs="+", help="e.g., 1m 5m 1h 1d (case-insensitive). If --config provided, optional.")
    p.add_argument("--raw-dir", default=str(Path("data").resolve()), help="Root folder with raw OHLCV.")
    p.add_argument("--out-dir", default=str(Path("data").resolve()), help="Root folder for outputs (we append processed/features).")
    p.add_argument("--target-tz", default="UTC", help="e.g., 'Australia/Melbourne'.")
    p.add_argument("--dropna", action="store_true", help="Drop NA rows after feature engineering.")
    # Feature args
    p.add_argument("--rsi", type=int, help="RSI window (default: 14).")
    p.add_argument("--em", type=int, help="Expected move horizon in bars (default: 3).")
    p.add_argument("--returns", type=str, help="Comma list of return lags, e.g. '1,5,20'.")
    p.add_argument("--vol", type=str, help="Comma list of volatility windows, e.g. '20,50'.")
    p.add_argument("--move-th", dest="move_th", type=float, help="Trade viability absolute move threshold (default: 0.009).")
    p.add_argument("--spread-th", dest="spread_th", type=float, help="Trade viability spread threshold (default: 0.002).")
    p.add_argument("--no-proxies", action="store_true", help="Disable spread/slippage proxies.")
    p.add_argument("--no-flag", action="store_true", help="Disable is_trade_viable flag.")
    # Outputs
    p.add_argument("--skip-data", action="store_true", help="Skip writing *_data.parquet.")
    # Config
    p.add_argument("--config", type=str, help="Optional JSON config path for batch jobs.")
    return p.parse_args()


# ------------------------------
# Main
# ------------------------------

def main() -> int:
    cli_args = parse_args()

    cfg = _load_json_config(getattr(cli_args, "config", None))
    merged = _merge_config(cli_args, cfg)

    if not merged.symbols or not merged.timeframes:
        raise SystemExit("No symbols or timeframes provided. Use --symbols/--timeframes or --config.")

    raw_dir = Path(merged.raw_dir).resolve()
    out_dir = Path(merged.out_dir).resolve()
    target_tz = merged.target_tz
    dropna = merged.dropna

    # Static settings snapshot
    base_settings: Dict[str, object] = {
        "rsi_window": merged.rsi_window,
        "expected_move_horizon": merged.expected_move_horizon,
        "return_lags": merged.return_lags,
        "vol_windows": merged.vol_windows,
        "trade_move_threshold": merged.trade_move_threshold,
        "trade_spread_threshold": merged.trade_spread_threshold,
        "include_proxies": merged.include_proxies,
        "include_trade_flag": merged.include_trade_flag,
        "dropna_after_features": dropna,
        "target_timezone": target_tz,
        "config_source": merged.config_source or None,
        "cli_invocation": " ".join(["python", Path(__file__).name] + sys.argv[1:]),  # why: reproducibility
    }

    for symbol in merged.symbols:
        for tf in merged.timeframes:
            df, src_path = load_raw_ohlcv(raw_dir, symbol, tf, target_tz=target_tz)
            input_hash = _sha256_file(src_path)
            input_mtime = datetime.fromtimestamp(src_path.stat().st_mtime, tz=timezone.utc).isoformat()

            feats = engineer_features(
                df,
                return_lags=merged.return_lags,
                vol_windows=merged.vol_windows,
                rsi_window=merged.rsi_window,
                expected_move_horizon=merged.expected_move_horizon,
                include_proxies=merged.include_proxies,
                include_trade_flag=merged.include_trade_flag,
                trade_move_threshold=merged.trade_move_threshold,
                trade_spread_threshold=merged.trade_spread_threshold,
            )

            now_utc = datetime.now(timezone.utc).isoformat()
            meta: Dict[str, object] = {
                "run_time_utc": now_utc,
                "python": platform.python_version(),
                "pandas": pd.__version__,
                "numpy": np.__version__,
                "raw_input_file": str(src_path),
                "raw_input_sha256": input_hash,
                "raw_input_mtime_utc": input_mtime,
                "symbol": symbol,
                "timeframe": tf,
                "normalized_timeframe": _norm_tf(tf),
                "output_root": str(out_dir),
            }

            settings = dict(base_settings)
            settings.update({"symbol": symbol, "timeframe": tf})

            pq_path = write_outputs(
                symbol=symbol,
                timeframe=tf,
                df_features=feats,
                out_dir=out_dir,
                dropna=dropna,
                df_data=None if merged.skip_data else df,
                settings=settings,
                meta=meta,
                write_readme=True,
            )
            print(f"[features] Wrote {symbol} {tf} → {pq_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
