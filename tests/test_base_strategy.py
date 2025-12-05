# home_quant_lab_v3/tests/test_base_strategy.py
import unittest
from datetime import datetime
from core.types import Bar, Timeframe, StrategyContext, Position
from strategy.rule_based import CandleUpBuyStrategy
from core.types import Action

class TestBaseStrategy(unittest.TestCase):
    def setUp(self):
        self.symbol = "BTC/USD"
        self.strategy = CandleUpBuyStrategy([self.symbol], Timeframe.M1)
        self.context = StrategyContext(
            account_equity=10000.0,
            cash=10000.0,
            open_positions={},
            recent_bars={},
            config={},
        )

    def _bar(self, o: float, h: float, l: float, c: float):
        return Bar(
            symbol=self.symbol,
            timestamp=datetime.utcnow(),
            open=o, high=h, low=l, close=c,
            volume=100.0,
            timeframe=Timeframe.M1,
            features={},
        )

    def test_buy_on_green_candle(self):
        bar = self._bar(100.0, 105.0, 99.0, 101.0)
        sig = self.strategy.generate_signal(bar, self.context)
        self.assertEqual(sig.action, Action.BUY)
        self.assertEqual(sig.size, 1.0)

    def test_hold_on_red_candle(self):
        bar = self._bar(100.0, 101.0, 95.0, 98.0)
        sig = self.strategy.generate_signal(bar, self.context)
        self.assertEqual(sig.action, Action.HOLD)

if __name__ == "__main__":
    unittest.main()