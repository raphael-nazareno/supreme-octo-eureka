# Experiment: 1m features → 15m & 60m crypto XGBoost (cost-aware)

## TL;DR

- Built a full **cost-aware pipeline** for BTC/ETH/SOL:
  - 1m OHLCV + microstructure features → multitimeframe features
  - cost-aware labels (`y_ret_net_fwd_*`, `y_dir_net_fwd_*`)
  - model_input (per-coin, per-horizon)
  - XGBoost (directional, probability output)
  - cost-aware backtest v2 (fees + slippage)
- Tested **15m** and **60m** forward horizons across **BTC/ETH/SOL** and thresholds `0.50–0.70`.
- Under realistic costs (Alpaca-like taker fees + slippage proxies), there is **no clearly tradeable edge**:
  - 15m: almost flat PnL after costs, tiny positive bumps at best.
  - 60m: consistently **negative** PnL across all thresholds and most coins.
- Conclusion: this particular combo of **features + model + horizon** is **not worth trading live**. The pipeline itself is solid and should be reused for new ideas.

---

## 1. Setup

- **Universe**: `btc_usd`, `eth_usd`, `sol_usd`
- **Data**:
  - 1m crypto bars from Alpaca (multi-year)
  - Features computed on 1m bars (price, volume, technicals, spread/slippage proxies, etc.)
- **Labels**:
  - `y_ret_fwd_{H}m`, `y_dir_fwd_{H}m` (gross simple returns + ternary direction)
  - `y_ret_net_fwd_{H}m`, `y_dir_net_fwd_{H}m` (after:
    - per-side fee ≈ 25 bps (0.25%), round-trip ≈ 0.5%
    - spread + slippage proxies
    - optional extra slippage bps)
- **Model input**:
  - Per-coin CSVs, e.g. `ml/models/XGBoost/btc_usd/btc_usd_model_input_15m.csv`
  - Contains:
    - feature columns (27 for these runs)
    - gross labels (`y_ret_fwd_*`, `y_dir_fwd_*`)
    - net labels (`y_ret_net_fwd_*`, `y_dir_net_fwd_*`)
    - cost columns (`y_cost_rate_total_*`, etc.)
- **Model**:
  - XGBoost classifier on **direction of net return**:
    - `label_mode = "net"` → `y_dir_net_fwd_{H}m`
    - `feature_version = 2`
    - Reasonable XGB hyperparams (`n_estimators=500`, `max_depth=6`, etc.)
- **Backtest v2**:
  - Uses **net labels** and simulates execution with:
    - `fee_bps_per_side = 25.0`
    - `use_proxy_slippage = True`
  - Risk: `risk_frac = 0.01` (1% of equity per trade)
  - Thresholds: trade when `p_up >= threshold` (per horizon).

---

## 2. 15m horizon results (cost-aware, `label_mode = net`)

### BTC/USD (15m net)

- Thresholds tested: `0.50, 0.55, 0.60, 0.65, 0.70`
- Behaviour:
  - Trade count: **39 → 9** as threshold increases.
  - Win rate: **~0.36–0.50** depending on threshold.
  - Total return: **≈ 0.0** at all thresholds (slightly negative or flat).
- Interpretation:
  - Model finds almost no net-positive 15m opportunities after costs.
  - Tightening threshold just reduces trade count without revealing a profitable band.

### ETH/USD (15m net)

- Thresholds: `0.50–0.70`
- Behaviour:
  - Trade count: **91 → 7**.
  - Win rate: **~0.52–0.86**, best around `0.55–0.60`.
  - Total return: small positive bumps (**~0.1–0.2%** over full test) at low thresholds, but close to noise.
- Interpretation:
  - Slight hint of edge at low thresholds, but magnitude is tiny once costs are included.

### SOL/USD (15m net)

- Thresholds: `0.50–0.70`
- Behaviour:
  - Trade count: **98 → 7**.
  - Win rate: **~0.54–0.58** for mid thresholds.
  - Total return: **+0.1–0.2%** range, again very small.
- Interpretation:
  - Similar to ETH: maybe a weak signal, but not clearly tradeable after fees/slippage.

**Overall 15m conclusion**:  
Under realistic execution costs, the 15m horizon looks **flat to slightly positive at best**. After thousands of 15m bars, total PnL is effectively zero. No clear, robust edge emerges.

---

## 3. 60m horizon results (cost-aware, `label_mode = net`)

For 60m, we reused the same feature set but changed `horizon_min = 60` for labels + model_input, then trained `xgb_*_1m_60m_dir_fv2_v001` and backtested with the same cost model.

### BTC/USD (60m net)

- Thresholds: `0.50–0.70`
- Trade counts (for one run):  
  - `thr=0.50` → **2060 trades**, win ≈ **0.084**, total_ret ≈ **−0.122**
  - `thr=0.70` → **843 trades**, win ≈ **0.116**, total_ret ≈ **−0.052**
- Interpretation:
  - The model is **systematically wrong** on BTC at 60m horizon:
    - Very high trade count.
    - Very low win rate (~8–12%).
    - Consistent, sizeable losses across thresholds.

### ETH/USD (60m net)

- Thresholds: `0.50–0.70`
- Trade counts: **169 → 22**
- Win rate: **~0.43–0.49**
- Total return: **~−0.1% to −0.3%**, i.e. small but negative.
- Interpretation:
  - ETH 60m is basically a **coin flip or slightly losing** after costs; no profitable threshold band appears.

### SOL/USD (60m net)

- Thresholds: `0.50–0.70`
- Trade counts: **268 → 25**
- Win rate: **~0.38–0.48**
- Total return: **clearly negative** (≈ −0.1% to −0.9% range).
- Interpretation:
  - Similar to BTC: no sign of positive net edge; slippage + fees overwhelm whatever the model is doing.

**Overall 60m conclusion**:  
At 60m horizon, the current feature set + model architecture is **not predictive enough** to beat realistic crypto trading costs. BTC is particularly bad (strongly negative PnL and very low win rate), ETH/SOL are mildly losing across a wide threshold range.

---

## 4. Global conclusion

- The **engineering stack** is in good shape:
  - Clean separation of raw data → features → labels → model_input → train → backtest.
  - Cost-aware labels and cost-aware backtester.
  - Multiple horizons and thresholds easily configurable.
- The **strategy idea**, as implemented here, is not good enough:
  - 1m TA-style features targeting 15–60m crypto net returns,
  - XGBoost directional classifier,
  - basic cost model (fees + slippage proxies).

**Call:** mark this as **“no-trade / research dead-end (v2)”** and do not deploy live.  
Reuse the pipeline for other horizons, assets, and strategy concepts.

---

## 5. Possible next directions

Some directions to explore (reusing this pipeline):

1. **Longer horizons (4h / 1D)**:
   - Aim for larger moves relative to costs.
   - Fewer trades, more swing-style behaviour.

2. **Different assets / markets**:
   - FX (e.g., GBP/USD bounce strategy),
   - Index futures or ETFs where costs are lower and behaviour more stable.

3. **Different feature families**:
   - Regime features (trend / volatility / volume regimes),
   - Cross-sectional features (relative strength across assets),
   - Order book / depth / funding data if available.

4. **Risk filters / regime filters**:
   - Only allow trades during high-vol / trending regimes.
   - Avoid choppy or low-liquidity periods where slippage dominates.

This experiment is still a success: it demonstrates a robust pipeline and provides clear evidence that this particular idea is not worth trading.
