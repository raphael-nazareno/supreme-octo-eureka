# path: tests/test_trained_xgb_models.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest


MODELS_ROOT = Path("ml/models/XGBoost")


@pytest.mark.parametrize("coin", ["btc_usd", "eth_usd", "sol_usd"])
def test_xgb_model_and_metrics_exist_for_coin(coin: str) -> None:
    """
    Check that, for each coin, there is at least one trained XGBoost model
    and a matching *_metrics.json file in the expected directory:
        ml/models/XGBoost/<coin>/
    """
    coin_dir = MODELS_ROOT / coin
    assert coin_dir.is_dir(), f"Model directory for {coin} not found: {coin_dir}"

    # Look for metrics files for any horizon/version of the XGB model
    metrics_files: List[Path] = sorted(
        coin_dir.glob(f"xgb_{coin}_1m_*m_dir_fv*_v*_metrics.json")
    )

    assert metrics_files, (
        f"No metrics files found for {coin} under {coin_dir}. "
        f"Did you run train_xgboost for this coin?"
    )

    # Take the most recently modified metrics file as the "current" one
    metrics_path = max(metrics_files, key=lambda p: p.stat().st_mtime)

    # Derive model path by stripping the "_metrics" suffix
    name = metrics_path.name
    assert name.endswith("_metrics.json"), f"Unexpected metrics filename: {name}"
    model_filename = name.replace("_metrics", "")
    model_path = metrics_path.with_name(model_filename)

    assert model_path.exists(), (
        f"Model file not found for metrics {metrics_path.name}. "
        f"Expected model file: {model_path}"
    )


@pytest.mark.parametrize("coin", ["btc_usd", "eth_usd", "sol_usd"])
def test_xgb_metrics_schema_and_paths(coin: str) -> None:
    """
    Validate the contents of the *_metrics.json:
      - required top-level keys are present,
      - splits/train/val/test blocks exist,
      - model_input_path exists on disk,
      - feature_columns is a non-empty list.
    """
    coin_dir = MODELS_ROOT / coin
    assert coin_dir.is_dir(), f"Model directory for {coin} not found: {coin_dir}"

    metrics_files: List[Path] = sorted(
        coin_dir.glob(f"xgb_{coin}_1m_*m_dir_fv*_v*_metrics.json")
    )
    assert metrics_files, f"No metrics files found for {coin} under {coin_dir}"

    metrics_path = max(metrics_files, key=lambda p: p.stat().st_mtime)

    with metrics_path.open("r", encoding="utf-8") as f:
        metrics: Dict[str, Any] = json.load(f)

    # Basic required keys
    for key in [
        "model_id",
        "coin",
        "horizon_min",
        "feature_version",
        "model_version",
        "n_rows_total",
        "n_features",
        "splits",
        "feature_columns",
        "model_input_path",
    ]:
        assert key in metrics, f"Missing key '{key}' in metrics file: {metrics_path}"

    # coin and model_id sanity
    assert metrics["coin"] == coin, (
        f"Metrics 'coin'={metrics['coin']} does not match expected coin={coin}"
    )
    assert isinstance(metrics["model_id"], str) and metrics["model_id"], "model_id must be a non-empty string"

    # Split blocks for train/val/test
    splits = metrics["splits"]
    for split_name in ["train", "val", "test"]:
        assert split_name in splits, f"Missing '{split_name}' split in metrics.splits"
        split = splits[split_name]
        assert "n_samples" in split, f"Missing n_samples in metrics.splits['{split_name}']"
        assert "accuracy" in split, f"Missing accuracy in metrics.splits['{split_name}']"
        # We don't require n_samples > 0 (e.g. tiny datasets), but if >0 then accuracy shouldn't be None
        if split["n_samples"] > 0:
            assert split["accuracy"] is not None, (
                f"Non-empty split '{split_name}' has no accuracy in metrics."
            )

    # feature_columns sanity
    feature_cols = metrics["feature_columns"]
    assert isinstance(feature_cols, list), "feature_columns must be a list"
    assert feature_cols, "feature_columns list is empty; trainer should record used feature columns."

    # model_input_path should exist
    mip = Path(metrics["model_input_path"])
    assert mip.exists(), (
        f"model_input_path recorded in metrics does not exist on disk: {mip}"
    )
