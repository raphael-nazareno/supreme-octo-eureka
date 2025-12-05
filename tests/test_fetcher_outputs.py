# path: tests/test_fetcher_outputs.py
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"websockets\.legacy")

import argparse
import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"websockets\.legacy")



def _load_fetcher():
    for modname in ("data.fetcher_alpaca", "fetcher_alpaca"):
        try:
            return importlib.import_module(modname)
        except Exception:
            pass
    for alt in (Path("data/fetcher_alpaca.py"), Path("fetcher_alpaca.py")):
        if alt.exists():
            spec = importlib.util.spec_from_file_location("fetcher_alpaca", alt)
            assert spec and spec.loader
            mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)  # type: ignore[assignment]
            return mod
    pytest.skip("fetcher_alpaca module not found")


@pytest.fixture(scope="session")
def F():
    return _load_fetcher()


def _make_fake_chunk(n: int, tf: str, symbol="BTC/USD") -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="min", tz="UTC")
    base_price = 100.0
    data = {
        "timestamp": idx,
        "symbol": [symbol] * n,
        "open": base_price + np.arange(n) * 0.1,
        "high": base_price + np.arange(n) * 0.2,
        "low": base_price + np.arange(n) * 0.05,
        "close": base_price + np.arange(n) * 0.15,
        "volume": np.ones(n) * 10.0,
        "trade_count": np.ones(n, dtype=int),
        "vwap": base_price + np.arange(n) * 0.12,
    }
    return pd.DataFrame(data)


@pytest.fixture
def parquet_stub(F, monkeypatch):
    def fake_parquet_writer(df: pd.DataFrame, path: Path, compression: Any) -> None:
        text_lines = [
            f"rows={len(df)}",
            "cols=" + ",".join(df.columns.astype(str)),
        ]
        path.write_text("\n".join(text_lines), encoding="utf-8")
    monkeypatch.setattr(F, "_atomic_write_parquet", fake_parquet_writer, raising=True)
    return None


def _ns_for_main(tmp_root: Path, symbols, timeframes) -> argparse.Namespace:
    return argparse.Namespace(
        symbols=list(symbols),
        timeframes=list(timeframes),
        years=1,
        start=None,
        end=None,
        output_dir=str(tmp_root),
        log_level="INFO",
        resume=False,
        reindex="off",
        workers=1,
        compression="snappy",
        retries=1,
        chunk_sleep=0.0,
        downcast=False,
    )


@pytest.mark.parametrize("symbol_in,tf,expected_symbol", [
    ("BTC_USD", "1m", "BTC/USD"),
    ("ETH/USDT", "5m", "ETH/USD"),
])
def test_fetcher_writes_both_files_in_required_layout(tmp_path: Path, parquet_stub, F, symbol_in, tf, expected_symbol, monkeypatch):
    fake = _make_fake_chunk(n=12, tf=tf, symbol=expected_symbol).copy()

    def fake_fetch_symbol_tf(**kwargs) -> pd.DataFrame:
        return fake

    monkeypatch.setattr(F, "fetch_symbol_tf", fake_fetch_symbol_tf, raising=True)

    ns = _ns_for_main(tmp_path, [symbol_in], [tf])
    monkeypatch.setattr(F, "parse_args", lambda: ns, raising=True)

    rc = F.main()
    assert rc == 0

    sym_usd, base = F._lock_usd(symbol_in)
    paths = F._output_paths(Path(ns.output_dir), sym_usd, tf)

    csv_path = paths.csv_path
    pq_path = paths.parquet_path

    assert csv_path.exists(), f"Missing CSV at {csv_path}"
    assert pq_path.exists(), f"Missing Parquet at {pq_path}"
    assert csv_path.stat().st_size > 0, "CSV is empty"
    assert pq_path.stat().st_size > 0, "Parquet file is empty"

    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    required_cols = {"timestamp", "symbol", "open", "high", "low", "close", "volume"}
    assert required_cols.issubset(set(df.columns))
    assert df["symbol"].str.endswith("/USD").all()
    assert df["timestamp"].is_monotonic_increasing

    meta_text = pq_path.read_text(encoding="utf-8")
    assert str(fake.shape[0]) in meta_text
    for col in ["timestamp", "open", "close", "symbol"]:
        assert col in meta_text
