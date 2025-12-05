# path: tests/test_features.py
from __future__ import annotations
import argparse, importlib, importlib.util, json, sys
from pathlib import Path
import numpy as np, pandas as pd, pytest

def _load_builder():
    for modname in ("ml.build_multitimeframe_features","data.build_multitimeframe_features","build_multitimeframe_features"):
        try: return importlib.import_module(modname)
        except Exception: pass
    for alt in (Path("ml/build_multitimeframe_features.py"), Path("data/build_multitimeframe_features.py")):
        if alt.exists():
            spec = importlib.util.spec_from_file_location("build_multitimeframe_features", alt)
            assert spec and spec.loader
            mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
            sys.modules[spec.name] = mod  # important for dataclasses in Py3.14
            spec.loader.exec_module(mod)  # type: ignore[assignment]
            return mod
    pytest.skip("build_multitimeframe_features module not found")

@pytest.fixture(scope="module")
def F():
    return _load_builder()

@pytest.fixture()
def parquet_stub(monkeypatch):
    def _stub(self: pd.DataFrame, path, *args, **kwargs):
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
        meta = {"rows": int(self.shape[0]), "cols": list(map(str, self.columns))}
        p.write_text(json.dumps(meta), encoding="utf-8")
    monkeypatch.setattr(pd.DataFrame, "to_parquet", _stub, raising=True)

def _mk_raw(n: int = 24) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="min", tz="UTC")
    base = 100 + np.arange(n, dtype=float)
    return pd.DataFrame({"timestamp": idx, "open": base+0.1, "high": base+0.2, "low": base-0.2, "close": base, "volume": np.ones(n)})

def _ns_for_main(tmp_path: Path, symbols, timeframes) -> argparse.Namespace:
    return argparse.Namespace(
        symbols=symbols, timeframes=timeframes, timeframe=None,
        out_dir=str(tmp_path / "data"), raw_dir=str(tmp_path / "data"),
        no_csv=False, no_parquet=False, parquet_compression="snappy", target_tz=None
    )

def _write_raw_csv(root: Path, symbol_upper: str, tf_tok: str, df: pd.DataFrame) -> Path:
    raw_dir = root / "data" / "raw" / f"{symbol_upper}_USD" / "csv"; raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{symbol_upper}_{tf_tok}_raw.csv"; df.to_csv(path, index=False); return path

@pytest.mark.parametrize("symbol_in,tf_in,base,tf_tok", [
    ("BTC_USD","1m","BTC","1m"),
    ("ETH/USDT","1MIN","ETH","1m"),  # USD lock & TF normalization
])
def test_flat_layout_outputs(tmp_path: Path, parquet_stub, F, symbol_in, tf_in, base, tf_tok, monkeypatch):
    raw_df = _mk_raw(n=24); _write_raw_csv(tmp_path, base, tf_tok, raw_df)
    ns = _ns_for_main(tmp_path, [symbol_in], [tf_in]); monkeypatch.setattr(F, "parse_args", lambda: ns, raising=True)
    rc = F.main(); assert rc == 0

    base_lower = base.lower()
    root = Path(ns.out_dir) / "processed" / "features" / f"{base_lower}_usd"
    pq_path = root / f"{base_lower}_usd_{tf_tok}_features.parquet"
    csv_path = root / f"{base_lower}_usd_{tf_tok}_features.csv"
    assert pq_path.exists(); assert csv_path.exists(); assert csv_path.stat().st_size > 0

    meta = json.loads(pq_path.read_text(encoding="utf-8"))
    for col in ["open","high","low","close","volume"]:
        assert col in meta.get("cols", [])

def test_engineer_features_columns(F):
    raw = _mk_raw(n=40)
    df = raw.set_index(pd.to_datetime(raw["timestamp"], utc=True)).drop(columns=["timestamp"])
    out = F.engineer_features(df)
    cols = set(out.columns)

    # Stable required features
    required = {
        "returns_1", "returns_5",
        "rolling_vol_20", "rsi_14",
        "spread_proxy_pct", "slippage_proxy",
        "expected_move_3", "is_trade_viable",
    }
    assert required.issubset(cols)

    # Flexible: accept any of these medium-horizon returns depending on config
    assert {"returns_15", "returns_20", "returns_60"}.intersection(cols), "Missing a medium-horizon returns column"
