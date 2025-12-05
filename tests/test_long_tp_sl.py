# =====================================================================
# file: home_quant_lab_v5/tests/test_long_tp_sl.py
# =====================================================================

import numpy as np
import pandas as pd

from main import (
    run_backtest_next_open_long_only,
    Costs,
)

def _df(ts, o, h, l, c):
    return pd.DataFrame({
        "timestamp": pd.to_datetime(ts),
        "open": o, "high": h, "low": l, "close": c
    })

def test_long_take_profit():
    df = _df(
        ["2025-01-01 00:00:00","2025-01-01 00:01:00"],
        [100.0, 100.0],
        [101.5, 101.5],  # hits TP=100*(1+0.015)=101.5
        [99.0, 99.0],
        [100.0, 100.0],
    )

    probs = np.array([0.9], dtype=float)  # BUY
    costs = Costs(fee_bps=0, slippage_bps=0)

    trades, equity = run_backtest_next_open_long_only(
        df_exec=df.iloc[1:].reset_index(drop=True),
        probs=probs,
        thr_buy=0.55,
        thr_sell=0.45,
        sl_pct=0.005,
        tp_pct=0.015,
        costs=costs,
        initial_cash=0.0,
    )

    assert len(trades) == 1
    t = trades[0]

    assert t.hit_take_profit
    assert not t.hit_stop_loss
    # Long profit: 101.5 - 100.0 = +1.5
    assert abs(t.pnl - 1.5) < 1e-6
    assert abs(equity["equity"].iloc[-1] - 1.5) < 1e-6


def test_long_stop_loss():
    df = _df(
        ["2025-01-01 00:00:00","2025-01-01 00:01:00"],
        [100.0, 100.0],
        [101.0, 101.0],
        [98.5, 98.5],  # hits SL=100*(1-0.015)=98.5
        [100.0, 100.0],
    )

    probs = np.array([0.9])
    costs = Costs(fee_bps=0, slippage_bps=0)

    trades, equity = run_backtest_next_open_long_only(
        df_exec=df.iloc[1:].reset_index(drop=True),
        probs=probs,
        thr_buy=0.55,
        thr_sell=0.45,
        sl_pct=0.015,
        tp_pct=0.010,
        costs=costs,
        initial_cash=0.0,
    )

    assert len(trades) == 1
    t = trades[0]

    assert t.hit_stop_loss
    assert not t.hit_take_profit
    # Long loss: 98.5 - 100 = -1.5
    assert abs(t.pnl + 1.5) < 1e-6
    assert abs(equity["equity"].iloc[-1] + 1.5) < 1e-6
