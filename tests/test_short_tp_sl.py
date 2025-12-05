# =====================================================================
# file: home_quant_lab_v5/tests/test_short_tp_sl.py
# =====================================================================

import numpy as np
import pandas as pd

from main import (
    run_backtest_next_open_long_short,
    Costs,
)

def _df_from_ohlc(ts, o, h, l, c):
    return pd.DataFrame({
        "timestamp": pd.to_datetime(ts),
        "open": o, "high": h, "low": l, "close": c
    })

# ---------------------------------------------------------
# SHORT TP test
# ---------------------------------------------------------
def test_short_hits_take_profit_next_open():
    df = _df_from_ohlc(
        ["2025-01-01 00:00:00","2025-01-01 00:01:00","2025-01-01 00:02:00"],
        [101.0, 100.0, 100.0],
        [102.0, 101.0, 101.0],
        [100.5,  99.0,  99.0],
        [101.0, 100.0, 100.0],
    )

    probs = np.array([0.1, 0.5], dtype=float)  # SELL then HOLD
    costs = Costs(fee_bps=0.0, slippage_bps=0.0)

    trades, equity = run_backtest_next_open_long_short(
        df_exec=df.iloc[1:].reset_index(drop=True),
        probs=probs,
        thr_buy=0.55,
        thr_sell=0.45,
        sl_pct=0.005,     # 0.5% SL
        tp_pct=0.010,     # 1% TP → 100*(1-0.01)=99
        costs=costs,
        initial_cash=0.0,
    )

    # ---- TRADE ASSERTIONS ----
    assert len(trades) == 1
    t = trades[0]
    assert t.side == "SHORT"
    assert t.entry_action == "SELL"
    assert t.hit_take_profit
    assert not t.hit_stop_loss
    assert abs(t.pnl - 1.0) < 1e-6

    # ---- CASH + EQUITY ASSERTIONS ----
    # Short entry at 100 increases cash by +100
    # Exit at 99 reduces cash by -99 → final cash = +1
    assert abs(equity["equity"].iloc[-1] - 1.0) < 1e-6


# ---------------------------------------------------------
# SHORT SL test
# ---------------------------------------------------------
def test_short_hits_stop_loss_next_open():
    df = _df_from_ohlc(
        ["2025-01-01 00:00:00","2025-01-01 00:01:00","2025-01-01 00:02:00"],
        [101.0, 100.0, 100.0],
        [102.0, 101.5, 101.0],   # hits SL = 100*(1+0.015)=101.5
        [100.5,  98.5,  99.0],
        [101.0, 100.0, 100.0],
    )

    probs = np.array([0.1, 0.5], dtype=float)
    costs = Costs(fee_bps=0.0, slippage_bps=0.0)

    trades, equity = run_backtest_next_open_long_short(
        df_exec=df.iloc[1:].reset_index(drop=True),
        probs=probs,
        thr_buy=0.55,
        thr_sell=0.45,
        sl_pct=0.015,   # 1.5% SL
        tp_pct=0.01,
        costs=costs,
        initial_cash=0.0,
    )

    # ---- TRADE ASSERTIONS ----
    assert len(trades) == 1
    t = trades[0]
    assert t.side == "SHORT"
    assert t.entry_action == "SELL"
    assert t.hit_stop_loss
    assert not t.hit_take_profit
    assert abs(t.pnl + 1.5) < 1e-6  # -1.5 pnl

    # ---- CASH + EQUITY ASSERTIONS ----
    # Cash flow:
    #   Entry short @100 → +100
    #   Exit @101.5 → -101.5
    # Final cash = -1.5
    assert abs(equity["equity"].iloc[-1] + 1.5) < 1e-6
