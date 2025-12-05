# =====================================================================
# file: home_quant_lab_v5/tests/test_equity_cash_flow.py
# =====================================================================

import numpy as np
import pandas as pd

from main import (
    run_backtest_next_open_long_short,
    Costs,
)

def test_equity_tracks_open_short_position():
    """
    Ensures mark-to-market equity updates correctly across bars.
    """
    df = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2025-01-01 00:00:00",
            "2025-01-01 00:01:00",
            "2025-01-01 00:02:00",
            "2025-01-01 00:03:00",
        ]),
        "open":  [101, 100, 99, 98],
        "high":  [102, 101, 100, 99],
        "low":   [100, 99, 98, 97],
        "close": [101, 100, 99, 98],
    })

    # SELL → HOLD → HOLD
    probs = np.array([0.1, 0.5, 0.5])
    costs = Costs(fee_bps=0, slippage_bps=0)

    trades, equity = run_backtest_next_open_long_short(
        df_exec=df.iloc[1:].reset_index(drop=True),
        probs=probs,
        thr_buy=0.55,
        thr_sell=0.45,
        sl_pct=0.05,
        tp_pct=0.05,
        costs=costs,
        initial_cash=0.0,
    )

    # Position opened at 100 on bar1
    # Equity at each bar = cash + MTM:
    #   Bar1: entry, cash=+100, MTM=0         → 100
    #   Bar2: close=99 → MTM=+1              → 101
    #   Bar3: close=98 → MTM=+2              → 102

    eq = equity["equity"].values
    assert abs(eq[0] - 100) < 1e-6
    assert abs(eq[1] - 101) < 1e-6
    assert abs(eq[2] - 102) < 1e-6

    # No auto-exit happened → no trades should exist
    assert len(trades) == 0
