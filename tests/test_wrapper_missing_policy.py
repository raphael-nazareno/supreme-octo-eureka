# home_quant_lab_v3/tests/test_wrapper_missing_policy.py
import unittest
from ml.model_wrapper import ModelWrapper, WrapperConfig
from core.types import Action

class TestWrapperMissingPolicy(unittest.TestCase):
    def test_skip_policy_returns_hold(self):
        cfg = WrapperConfig(feature_order=["sma_fast","sma_slow","close"], missing_policy="skip")
        w = ModelWrapper(cfg, model=None)
        action, conf, tp, sl = w.predict_action({"sma_fast": 10.0})  # missing others
        self.assertEqual(action, Action.HOLD)
        self.assertEqual(conf, 0.0)

    def test_zero_policy_imputes_and_returns_action(self):
        cfg = WrapperConfig(feature_order=["sma_fast","sma_slow","close"], missing_policy="zero", default_tp_pct=0.02, default_sl_pct=0.01)
        w = ModelWrapper(cfg, model=None)
        action, conf, tp, sl = w.predict_action({"sma_fast": 10.0, "sma_slow": 9.0, "close": 100.0})
        self.assertEqual(action, Action.BUY)
        self.assertIsNotNone(tp); self.assertIsNotNone(sl)

if __name__ == "__main__":
    unittest.main()
