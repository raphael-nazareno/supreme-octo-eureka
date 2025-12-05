# path: ml/features.py
"""
Feature engineering utilities for crypto OHLCV data.

This module provides a single canonical `engineer_features` function that
takes a time-indexed OHLCV DataFrame and returns a new DataFrame with:

- multi-horizon returns & volatility
- EMA-based trend features
- RSI
- microstructure proxies (spread & slippage)
- time-of-day & day-of-week encodings
- forward 15-bar *gross* returns & direction labels
- forward 15-bar *net* returns & direction labels after estimated costs

It is deliberately model-agnostic: the output can be used to:
- build training sets for XGBoost / random forests / neural nets
- feed into backtesters
- feed into live signal generation

Cost-aware labels use the following ingredients:

- `fee_bps_per_side`:   trading fee in basis points for entry and exit.
- `spread_proxy_pct`:   (high - low) / close
- `slippage_proxy`:     (high - low) / open
- `extra_slippage_bps`: extra constant slippage on top of proxies.

The resulting columns:

- y_fee_rate           : constant round-trip fee rate (decimal)
- y_micro_cost_rate    : spread + slippage proxies + constant slippage
- y_cost_rate_total    : total estimated round-trip cost
- y_ret_net_fwd_15m    : 15-bar net return (gross - total_cost_rate)
- y_dir_net_fwd_15m    : {-1, 0, +1} based on net_ret and a small band

The "15m" in the names assumes you're using 1-minute bars. For other
timeframes the horizon is still "15 bars", which you can interpret
accordingly (e.g. 15 * 5m = 75 minutes) if you choose to keep the same
labels. The horizon length is configurable.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

# Alpaca crypto maker/taker fee schedule (October 2024).
# Rates are *per trade* as decimal fractions (0.0025 = 0.25%).
ALPACA_CRYPTO_FEE_SCHEDULE = [
    # (min_volume, max_volume, maker_rate, taker_rate)
    (0.0,          100_000.0,      0.0015, 0.0025),  # Tier 1
    (100_000.0,    500_000.0,      0.0012, 0.0022),  # Tier 2
    (500_000.0,  1_000_000.0,      0.0010, 0.0020),  # Tier 3
    (1_000_000.0,10_000_000.0,     0.0008, 0.0018),  # Tier 4
    (10_000_000.0,25_000_000.0,    0.0005, 0.0015),  # Tier 5
    (25_000_000.0,50_000_000.0,    0.0002, 0.0013),  # Tier 6
    (50_000_000.0,100_000_000.0,   0.0002, 0.0012),  # Tier 7
    (100_000_000.0, float("inf"),  0.0000, 0.0010),  # Tier 8
]


def alpaca_crypto_fee_bps(volume_30d_usd: float, maker: bool = False) -> float:
    """
    Map 30-day crypto volume (USD) to Alpaca maker/taker fee in *basis points* per trade.

    Example:
        # Tier 1 taker (0-100k) -> 0.25% per trade = 25 bps
        fee_bps = alpaca_crypto_fee_bps(50_000, maker=False)
    """
    v = float(volume_30d_usd)
    for lo, hi, maker_rate, taker_rate in ALPACA_CRYPTO_FEE_SCHEDULE:
        if lo <= v < hi:
            rate = maker_rate if maker else taker_rate
            return rate * 10_000.0  # decimal -> bps

    # Fallback to highest tier if something weird happens
    _, _, maker_rate, taker_rate = ALPACA_CRYPTO_FEE_SCHEDULE[-1]
    return (maker_rate if maker else taker_rate) * 10_000.0

def safe_pct_change(s: pd.Series, periods: int) -> pd.Series:
    """
    Robust percentage change that tolerates non-numeric input.
    """
    s = pd.to_numeric(s, errors="coerce")
    return s.pct_change(periods=periods)


def _rsi(series: pd.Series, length: int = 14) -> pd.Series:
    """
    Classic Wilder RSI implementation on closing prices.
    """
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(length, min_periods=length).mean()
    avg_loss = loss.rolling(length, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


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
    # --- cost-aware label config ---
    forward_horizon_bars: int = 15,
    fee_bps_per_side: float = 10.0,
    extra_slippage_bps: float = 0.0,
    net_label_band: float = 0.001,
) -> pd.DataFrame:

    """
    Engineer a rich set of features and cost-aware forward labels.

    Parameters
    ----------
    df :
        Input OHLCV DataFrame with columns at least:
        ['open', 'high', 'low', 'close', 'volume'] and a DatetimeIndex.
    return_lags :
        Lags (in bars) for percentage returns. For example (1, 5, 20)
        produces 'returns_1', 'returns_5', 'returns_20'.
    vol_windows :
        Window sizes for rolling volatility over 1-bar returns.
        For example (20,) produces 'rolling_vol_20'.
    rsi_window :
        Lookback window for RSI in bars.
    expected_move_horizon :
        Horizon (bars) for the `expected_move_*` feature.
    include_proxies :
        Whether to compute spread/slippage proxies.
    include_trade_flag :
        Whether to compute `is_trade_viable` flag.
    trade_move_threshold :
        Absolute move threshold used for `is_trade_viable`.
    trade_spread_threshold :
        Maximum allowed `spread_proxy_pct` for `is_trade_viable`.
    forward_horizon_bars :
        Horizon in bars for forward gross/net return labels.
        Defaults to 15 (e.g. 15 minutes on 1m data).
    fee_bps_per_side :
        Trading fee in basis points per side (entry *and* exit).
    extra_slippage_bps :
        Extra constant slippage per round-trip in basis points,
        on top of the spread/slippage proxies.
    net_label_band :
        Small dead-zone band around zero for net-direction labels.

    Returns
    -------
    DataFrame
        Copy of input with added feature and label columns.
    """
    out = df.copy()

    # ---- Basic numeric series ----
    open_ = pd.to_numeric(out["open"], errors="coerce")
    high = pd.to_numeric(out["high"], errors="coerce")
    low = pd.to_numeric(out["low"], errors="coerce")
    close = pd.to_numeric(out["close"], errors="coerce")
    vol = pd.to_numeric(out["volume"], errors="coerce")

    # ---- Returns ----
    for lag in return_lags:
        col = f"returns_{lag}"
        out[col] = safe_pct_change(close, lag)

    # Preserve specific 1/5/20 names expected by tests even if
    # return_lags has been customised.
    if 1 not in return_lags:
        out["returns_1"] = safe_pct_change(close, 1)
    if 5 not in return_lags:
        out["returns_5"] = safe_pct_change(close, 5)
    if 20 not in return_lags:
        out["returns_20"] = safe_pct_change(close, 20)

    # ---- Volatility over 1-bar returns ----
    ret_1 = close.pct_change()
    for w in vol_windows:
        col = f"rolling_vol_{w}"
        out[col] = ret_1.rolling(w).std()

    # Always provide rolling_vol_20 specifically.
    if 20 not in vol_windows:
        out["rolling_vol_20"] = ret_1.rolling(20).std()

        # ---- Volume regime ----
    vol_mean_20 = vol.rolling(20, min_periods=20).mean()
    vol_std_20 = vol.rolling(20, min_periods=20).std()
    out["volume_zscore_20"] = (vol - vol_mean_20) / vol_std_20.replace(0.0, np.nan)

        # ---- Trade count & VWAP (if present) ----
    # Make sure Pylance sees these as Series, not floats.
    if "trade_count" in out.columns:
        trade_count: pd.Series = pd.to_numeric(out["trade_count"], errors="coerce")
    else:
        trade_count = pd.Series(np.nan, index=out.index, dtype="float64")

    if "vwap" in out.columns:
        vwap: pd.Series = pd.to_numeric(out["vwap"], errors="coerce")
    else:
        vwap = pd.Series(np.nan, index=out.index, dtype="float64")

    # Keep them in the output
    out["trade_count"] = trade_count
    out["vwap"] = vwap

    # Average trade size (volume per trade)
    avg_trade_size = vol / trade_count.replace({0: np.nan})
    out["avg_trade_size"] = avg_trade_size

    # Trade count regime (z-score over 20 bars)
    tc_mean_20 = trade_count.rolling(20, min_periods=20).mean()
    tc_std_20 = trade_count.rolling(20, min_periods=20).std()
    out["trade_count_zscore_20"] = (trade_count - tc_mean_20) / tc_std_20.replace(0.0, np.nan)

    # VWAP-based features
    out["dist_close_vwap"] = (close - vwap) / vwap.replace(0.0, np.nan)
    out["vwap_returns_1"] = vwap.pct_change()




    # ---- Trend / moving averages ----
    ema_20 = close.ewm(span=20, adjust=False, min_periods=20).mean()
    ema_50 = close.ewm(span=50, adjust=False, min_periods=50).mean()
    out["ema_20"] = ema_20
    out["ema_50"] = ema_50
    out["dist_close_ema20"] = (close - ema_20) / ema_20.replace(0.0, np.nan)
    out["dist_ema20_ema50"] = (ema_20 - ema_50) / ema_50.replace(0.0, np.nan)

    # ---- RSI ----
    out[f"rsi_{rsi_window}"] = _rsi(close, rsi_window)
    # Also expose fixed rsi_14 name for convenience / tests.
    if rsi_window != 14:
        out["rsi_14"] = _rsi(close, 14)
    else:
        out["rsi_14"] = out[f"rsi_{rsi_window}"]

    # ---- Microstructure proxies ----
    if include_proxies:
        rng = high - low
        out["spread_proxy_pct"] = rng / close.replace(0.0, np.nan)
        out["slippage_proxy"] = rng / open_.replace(0.0, np.nan)
    else:
        out["spread_proxy_pct"] = np.nan
        out["slippage_proxy"] = np.nan

    # ---- Expected move proxy ----
    # simple forward percentage move over `expected_move_horizon` bars
    out[f"expected_move_{expected_move_horizon}"] = (
        close.shift(-expected_move_horizon) / close - 1.0
    )
    # Preserve expected_move_3 name specifically
    if expected_move_horizon != 3:
        out["expected_move_3"] = close.shift(-3) / close - 1.0

    # ---- is_trade_viable flag ----
    if include_trade_flag:
        expected_move = out.get(
            "expected_move_3",
            close.shift(-expected_move_horizon) / close - 1.0,
        )
        spread = out["spread_proxy_pct"]
        out["is_trade_viable"] = (
            (expected_move.abs() > trade_move_threshold)
            & (spread < trade_spread_threshold)
        ).astype(int)
    else:
        out["is_trade_viable"] = 0

    # ---- Time-of-day & calendar encodings ----
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

    # ---- Forward labels: gross 15-bar return & direction ----
    gross_ret = close.shift(-forward_horizon_bars) / close - 1.0
    out["y_ret_fwd_15m"] = gross_ret

    thr = net_label_band
    out["y_dir_fwd_15m"] = np.where(
        gross_ret > thr,
        1,
        np.where(gross_ret < -thr, -1, 0),
    )

    # ---- Cost-aware labels ----

    # Fee component: round-trip decimal rate
    fee_rate = (2.0 * float(fee_bps_per_side)) / 10_000.0
    out["y_fee_rate"] = fee_rate

    # Microstructure component: spread + slippage proxies + constant slippage
    spread_rate = out["spread_proxy_pct"].clip(lower=0.0).fillna(0.0)
    slip_proxy_rate = out["slippage_proxy"].clip(lower=0.0).fillna(0.0)
    extra_slip_rate = float(extra_slippage_bps) / 10_000.0

    micro_cost_rate = spread_rate + slip_proxy_rate + extra_slip_rate
    out["y_micro_cost_rate"] = micro_cost_rate

    # Total estimated round-trip cost
    total_cost_rate = fee_rate + micro_cost_rate
    out["y_cost_rate_total"] = total_cost_rate

    # Net forward return after estimated costs
    out["y_ret_net_fwd_15m"] = gross_ret - total_cost_rate

    net_ret = out["y_ret_net_fwd_15m"]
    out["y_dir_net_fwd_15m"] = np.where(
        net_ret > thr,
        1,
        np.where(net_ret < -thr, -1, 0),
    )

    return out
