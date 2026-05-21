# Portfolio Backtest — Technical Documentation

> **Project**: Taller B3-T4 — Deep Learning for Financial Time Series  
> **Section**: 11_Portfolio — Model-Based Portfolio Strategy Evaluation  
> **Authors**: MIAX Master's Programme  
> **Last updated**: May 2026

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Objectives](#2-objectives)
3. [Data Sources & Processing Pipeline](#3-data-sources--processing-pipeline)
4. [Methodology — From Model Output to Portfolio Returns](#4-methodology--from-model-output-to-portfolio-returns)
5. [Investment Strategies](#5-investment-strategies)
6. [Benchmarks](#6-benchmarks)
7. [Rolling Offset Robustness Analysis](#7-rolling-offset-robustness-analysis)
8. [Folder Structure](#8-folder-structure)
9. [Notebook Execution Pipeline](#9-notebook-execution-pipeline)
10. [Output Files & Interpretation](#10-output-files--interpretation)
11. [Model Architecture Auto-Detection](#11-model-architecture-auto-detection)
12. [Assumptions](#12-assumptions)
13. [Strengths](#13-strengths)
14. [Limitations & Known Weaknesses](#14-limitations--known-weaknesses)
15. [Glossary](#15-glossary)

---

## 1. Executive Summary

This section of the project evaluates whether the predictive signal from deep learning models trained on fractionally differentiated log-prices can generate economic value when translated into a portfolio investment strategy.

The core question is: **does a model that minimises MAE on frac-diff predictions also produce useful portfolio tilts?**

The investigation compares three model-driven strategies (equal-weight, take-profit, and proportional sizing) against two passive benchmarks (buy-and-hold equal-weight, and periodically rebalanced equal-weight), across multiple models (CNN, MLP, RNN) and multiple starting dates (rolling offsets).

---

## 2. Objectives

The portfolio section has three goals, in order of priority:

**Primary**: Test whether the investigation-track models (trained on frac-diff of log(price), d=0.40) produce a signal with economic value, measured as portfolio return relative to a naïve equal-weight benchmark.

**Secondary**: Compare how different model architectures (CNN, MLP, RNN) and input window lengths (V_IN = 5, 10, 30, 90) translate into portfolio performance, and whether lower test MAE correlates with better portfolio returns.

**Tertiary**: Evaluate the sensitivity of results to the choice of starting date using rolling offsets, and assess whether the model's predicted return magnitudes (not just rankings) contain useful information via proportional sizing.

---

## 3. Data Sources & Processing Pipeline

### 3.1 Data Source

All price data is sourced from **Yahoo Finance** via the `yfinance` Python library. The asset universe consists of **23 S&P 500 constituents** selected during the training phase of the project (see `dataset_utils.py` for the canonical list).

```
TICKERS = ['AEP', 'BA', 'CAT', 'CNP', 'CVX', 'DIS', 'DTE', 'ED', 'GD', 'GE',
           'HON', 'HPQ', 'IBM', 'IP', 'JNJ', 'KO', 'KR', 'MMM', 'MO', 'MRK',
           'MSI', 'PG', 'XOM']
```

### 3.2 Processing Pipeline

The data flows through five stages from raw prices to portfolio decisions:

```
┌──────────────────┐
│  Yahoo Finance   │  Adjusted close prices, ~60 years of history
│  (yfinance)      │  Downloaded per-ticker to avoid API issues
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Log Prices      │  log(P_t) for each asset
│                  │  Needed for FFD application and inversion
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Fractional      │  FFD(d=0.40, threshold=1e-5)
│  Differentiation │  L = 1,458 retained weights
│                  │  Produces stationary series preserving long memory
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  StandardScaler  │  Fit ONLY on the training partition (X_tr)
│                  │  Applied to the V_IN-day input window
│                  │  Reconstructed identically for each V_IN/V_OUT
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Model Inference │  Input: (1, V_IN, 23) scaled frac-diff window
│                  │  Output: (23,) predicted avg daily frac-diff
│                  │  One forward pass per rebalancing date
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  FFD Inversion   │  Converts predicted frac-diff → expected returns
│                  │  See Section 4.2 for full methodology
└──────────────────┘
```

### 3.3 Download Strategy

Due to known issues with `yfinance` v1.2+ returning empty DataFrames when downloading multiple tickers simultaneously, the data loader downloads each ticker individually with retry logic and exponential backoff. This is implemented in `data_loader.py`.

### 3.4 Lookback Requirements

The FFD operator with d=0.40 and threshold=1e-5 retains L=1,458 weights. The FFD inversion step requires at least L historical log-prices before the rebalancing date. To satisfy this, 10 years of price history prior to the backtest start date are downloaded.

---

## 4. Methodology — From Model Output to Portfolio Returns

### 4.1 What the Model Outputs

The model outputs a vector of 23 values, one per asset. Each value represents the **predicted average daily frac-diff(log(P), d=0.40)** over the next V_OUT=90 trading days.

This is NOT a return. It is a filtered version of the log-price series that sits between "the log-price itself" (d=0) and "the daily log-return" (d=1).

### 4.2 FFD Inversion — Converting to Expected Returns

The fractional differentiation operator applied to log-prices is:

```
y_t = Σ_{k=0}^{L-1} w_k · x_{t-k}
```

where `y_t` is the frac-diff value, `x_t = log(P_t)`, and `w_0 = 1`.

Since `w_0 = 1`, we can solve for the log-price:

```
x_t = y_t - Σ_{k=1}^{L-1} w_k · x_{t-k}
```

**The inversion procedure** at each rebalancing date:

```
┌─────────────────────────────────────────────────────────────────┐
│                    FFD INVERSION PROCEDURE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  INPUTS:                                                        │
│    ŷ = model's predicted avg daily frac-diff  (23 values)      │
│    x_{t}, x_{t-1}, ..., x_{t-L+1} = known historical log-prices│
│    w_0, w_1, ..., w_{L-1} = FFD weights                        │
│    H = 90 (horizon)                                             │
│                                                                 │
│  CONSTANT-DAILY ASSUMPTION:                                     │
│    Each future day's frac-diff = ŷ (the predicted average)     │
│                                                                 │
│  FORWARD ITERATION (for step = 1 to H):                        │
│    x̂_{t+step} = ŷ - Σ_{k=1}^{L-1} w_k · x_{t+step-k}        │
│                                                                 │
│    where x values before t use actual history,                  │
│    and x values after t use previously computed x̂              │
│                                                                 │
│  OUTPUT:                                                        │
│    expected_return = exp(x̂_{t+H} - x_t) - 1                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Why this is superior to the naïve approximation**: The naïve approach `expected_return ≈ exp(90 × ŷ) - 1` treats the cumulative sum of frac-diff as a proxy for log-return change. But `Σ y_t` is the Δ⁻¹ operator, while inverting Δ^{0.4} requires Δ^{-0.4}` (a fractional integral). The recursive inversion above is mathematically exact given the constant-daily assumption.

**Weakness of the constant-daily assumption**: The model predicts the *average*, so we assume each day equals that average. In reality, the frac-diff varies daily. This affects the magnitude of the expected return (because the inversion is nonlinear) but preserves the ranking of assets (higher predicted average → higher expected return).

---

## 5. Investment Strategies

All strategies share the same selection logic and differ only in position sizing and exit rules.

### 5.1 Selection Logic (Common to All)

At each rebalancing date (every V_OUT=90 trading days):

1. Run model inference → get 23 expected returns via FFD inversion
2. Rank assets by expected return (descending)
3. Select the top TOP_K=5 assets, but only those with **positive** expected return
4. If fewer than 5 are positive, invest in those that are; remainder stays in cash

### 5.2 Strategy A — Equal Weight, Hold to Rebalance

```
  Selection: Top-K positive assets
  Sizing:    1/K per asset (1/5 = 20% each)
  Exit:      Hold until next rebalancing date
  Cash:      Unused slots earn CASH_RATE (default 0%)
```

This is the cleanest test of the model's ranking ability. If the model correctly identifies which assets will outperform, Strategy A should beat the benchmarks.

### 5.3 Strategy B — Equal Weight with Take-Profit

```
  Selection: Same as A
  Sizing:    Same as A (1/K per asset)
  Exit:      Close position if cumulative return since entry ≥ expected return
             Freed capital moves to cash until next rebalance
  Cash:      Same as A
```

Strategy B tests whether the model's predicted magnitudes are useful as profit targets. It introduces a **structural asymmetry**: winners are capped (closed at target) while losers run for the full window. In trending markets, this reduces returns.

### 5.4 Strategy C — Proportional Weight, Hold to Rebalance

```
  Selection: Same as A
  Sizing:    weight_i = expected_return_i / Σ(expected_returns of selected assets)
             100% of capital allocated (no cash unless zero assets are positive)
  Exit:      Hold until next rebalancing date
```

Strategy C is the most aggressive test of the model's signal. It allocates more capital to assets the model is most bullish on. If expected returns have predictive value beyond just ranking, Strategy C should outperform A. However, concentration risk is high — if only one asset has positive expected return, 100% goes into that single position.

### 5.5 Strategy Comparison Summary

```
┌──────────────┬──────────────┬──────────────┬──────────────────────┐
│              │  Sizing      │  Exit Rule   │  What It Tests       │
├──────────────┼──────────────┼──────────────┼──────────────────────┤
│ Strategy A   │ Equal (1/K)  │ Hold         │ Ranking ability      │
│ Strategy B   │ Equal (1/K)  │ Take-profit  │ Magnitude as target  │
│ Strategy C   │ Proportional │ Hold         │ Magnitude for sizing │
└──────────────┴──────────────┴──────────────┴──────────────────────┘
```

---

## 6. Benchmarks

### 6.1 BM1 — Equal Weight, Fixed (Buy and Hold)

Set 1/23 weight per asset on day 1. Never rebalance. Weights drift naturally with price changes. By the end of the period, the portfolio is overweight in assets that performed well and underweight in those that didn't.

**What it represents**: a completely passive investor who buys all 23 assets equally and does nothing.

### 6.2 BM2 — Equal Weight, Rebalanced with Drift

Reset to 1/23 per asset every 90 trading days (matching the strategy rebalancing frequency). Between rebalancing dates, weights drift with prices.

**What it represents**: an investor who periodically rebalances to equal weight but makes no active bets on any asset.

### 6.3 Why Two Benchmarks

BM1 captures momentum (winners get larger weights over time). BM2 is anti-momentum (sells winners, buys losers at each rebalance). Comparing the strategies against both benchmarks reveals whether the model's signal adds value beyond what passive approaches capture.

---

## 7. Rolling Offset Robustness Analysis

### 7.1 Motivation

A single backtest starting on a specific date can produce misleading results. The portfolio's performance is sensitive to which assets are selected at each rebalancing date, which depends on the exact starting date.

### 7.2 Methodology

The full backtest is run 5 times with different starting offsets:

```
  Offset 0:   Start at trading_day[0]   (e.g., 2025-01-02)
  Offset 5:   Start at trading_day[5]   (e.g., 2025-01-10)
  Offset 10:  Start at trading_day[10]  (e.g., 2025-01-17)
  Offset 15:  Start at trading_day[15]  (e.g., 2025-01-27)
  Offset 20:  Start at trading_day[20]  (e.g., 2025-02-03)
```

Each offset produces different rebalancing dates, different model input windows, different predictions, and therefore different portfolio outcomes.

### 7.3 Reported Metrics

For each model, results are reported as:
- **Individual offset returns**: full transparency on each run
- **Average (AVG)**: the central estimate of performance
- **Standard deviation (STD)**: how sensitive the results are to the starting date

A model that performs well on average but with high STD is less reliable than one with moderate average but low STD.

### 7.4 Computational Cost

Each offset requires ~3-4 model forward passes (one per rebalancing date). With 5 offsets, that's ~15-20 forward passes per model — negligible on GPU. The total batch runtime for 4 models × 5 offsets ≈ 2 minutes on Colab.

---

## 8. Folder Structure

```
Taller4_DL_MIAX/
│
├── 01_src_compartido/
│   ├── dataset_utils.py          ← TICKERS, load_data, create_dataset, etc.
│   └── metrics_utils.py          ← calc_mae_all, BestRunTracker
│
├── 08_results/
│   └── checkpoints/              ← Trained model weights
│       ├── inv_cnn_vin10_vout90_best.weights.h5
│       ├── inv_mlp_vin5_vout90_best.weights.h5
│       ├── inv_mlp_vin90_vout90_best.weights.h5
│       └── inv_rnn_vin30_vout90_best.weights.h5
│
└── 11_Portfolio/                  ← THIS SECTION
    │
    │  ── Shared code ──
    ├── config.py                  Config & model catalog
    ├── model_builder.py           Auto-detect arch from .h5, build & load
    ├── frac_diff_utils.py         FFD weights, forward transform, inversion
    ├── data_loader.py             yfinance download, trading dates, log prices
    │
    │  ── Notebooks ──
    ├── 01_diagnostics.ipynb       Verify each building block (optional)
    ├── 04_full_pipeline.ipynb     Single model: inference + backtest + charts
    ├── 05_batch_runner.ipynb      All models: auto-discover, run, save
    ├── 06_comparison.ipynb        Cross-model comparison & visualisation
    │
    │  ── Per-model results (generated by batch runner) ──
    ├── inv_cnn_vin10_vout90/
    │   ├── model_meta.json        Architecture, MAE, avg returns, alerts
    │   ├── summary_returns.csv    Per-offset total returns + AVG + STD
    │   ├── nav_series_offset0.csv Daily NAV for all strategies (offset=0)
    │   ├── window_detail.csv      Asset-level expected vs actual per window
    │   ├── cumulative_returns.png Chart: NAV curves for all strategies
    │   └── scatter_exp_vs_actual.png  Chart: prediction scatter per window
    │
    ├── inv_mlp_vin5_vout90/       (same structure)
    ├── inv_mlp_vin90_vout90/      (same structure)
    ├── inv_rnn_vin30_vout90/      (same structure)
    │
    │  ── Cross-model comparison (generated by 06_comparison) ──
    ├── comparison_summary.csv
    ├── comparison_prediction_quality.csv
    ├── comparison_table.png
    ├── comparison_cumulative.png
    ├── comparison_bars.png
    ├── comparison_exp_vs_actual.png
    │
    └── TECHNICAL_DOCUMENTATION.md  ← THIS FILE
```

---

## 9. Notebook Execution Pipeline

The notebooks should be executed in order. Steps 1 and 2 are optional diagnostics.

```
┌─────────────────────────────────────────────────────────────────┐
│                    EXECUTION PIPELINE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Step 1 (optional):  01_diagnostics.ipynb                       │
│    ► Verify paths, config, FFD weights, price download          │
│    ► Test model loading and inference on one date               │
│    ► Run on: Colab                                              │
│                                                                 │
│  Step 2 (optional):  04_full_pipeline.ipynb                     │
│    ► Full pipeline for a SINGLE model                           │
│    ► Useful for debugging / exploring one model in detail       │
│    ► Run on: Colab                                              │
│                                                                 │
│  Step 3 (required):  05_batch_runner.ipynb                      │
│    ► Auto-discovers all checkpoints in 08_results/checkpoints/  │
│    ► Runs full pipeline for each (inference + backtest)         │
│    ► Generates per-model result folders                         │
│    ► Robust: try/except per model, failures logged not fatal    │
│    ► Run on: Colab (requires TensorFlow)                        │
│    ► Runtime: ~30s per model × 5 offsets                        │
│                                                                 │
│  Step 4 (required):  06_comparison.ipynb                        │
│    ► Reads model_meta.json from each result folder              │
│    ► Produces cross-model comparison tables and charts          │
│    ► No TensorFlow required (reads CSV/JSON only)               │
│    ► Run on: Colab or local                                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 9.1 Adding New Models

To test a new model:

1. Train the model in the investigation track (frac-diff data, d=0.40)
2. Save the checkpoint to `08_results/checkpoints/inv_{arch}_vin{V_IN}_vout90_best.weights.h5`
3. Re-run `05_batch_runner.ipynb` — it will automatically discover and process the new checkpoint
4. Re-run `06_comparison.ipynb` to update the cross-model comparison

No code changes required. The architecture is auto-detected from the checkpoint file.

---

## 10. Output Files & Interpretation

### 10.1 `model_meta.json`

Contains all metadata for reproducibility: architecture type, V_IN, V_OUT, d_frac, TOP_K, offsets used, MAE on train/val/test, average returns across offsets, and per-offset detailed results including alerts.

### 10.2 `summary_returns.csv`

One row per offset plus AVG and STD rows. Columns: A, B, C, BM1, BM2 (total returns).

**How to interpret**: If AVG(A) > AVG(BM1), the model's ranking signal adds value over passive equal-weight. If AVG(C) > AVG(A), the model's predicted magnitudes contain useful information beyond rankings.

### 10.3 `window_detail.csv`

Asset-level granularity for offset=0. Each row is one selected asset in one rebalancing window. Columns: window, rebalancing date, end date, ticker, weight_eq (Strategy A weight), weight_prop (Strategy C weight), expected return, actual return, direction_correct (1/0).

**How to interpret**: The `direction_correct` column shows whether the model correctly predicted the sign of each asset's return. The difference between `exp_return` and `actual_return` shows prediction error at the individual asset level.

### 10.4 `nav_series_offset0.csv`

Daily portfolio value (NAV) for all 5 portfolios from offset=0. Index is the trading date, columns are A, B, C, BM1, BM2. Initial value = 1.0 for all.

### 10.5 Chart: `cumulative_returns.png`

Overlay of all 5 portfolio NAV curves. Strategies in solid lines (blue, orange, green), benchmarks in dashed (gray). Vertical dotted lines mark rebalancing dates.

### 10.6 Chart: `scatter_exp_vs_actual.png`

One panel per rebalancing window. Each dot is one of the 23 assets. X-axis = expected return (from model), Y-axis = actual return. The diagonal line represents perfect prediction. Spearman rho printed in each panel title.

**How to interpret**: Points near the diagonal indicate good prediction. Points in the upper-left or lower-right quadrants indicate directional errors (predicted positive but went negative, or vice versa). The correlation value summarises overall ranking quality.

---

## 11. Model Architecture Auto-Detection

A key engineering challenge was that each model checkpoint has a different architecture (different layer sizes, different number of layers). The `model_builder.py` module solves this by reading the `.h5` file directly.

### 11.1 How It Works

```
┌──────────────────┐
│  .h5 checkpoint  │
│  file on disk    │
└────────┬─────────┘
         │  h5py reads layer names + weight shapes
         ▼
┌──────────────────┐
│  _read_h5_layers │  Returns: {layer_name: [shape1, shape2, ...], ...}
│                  │  Note: h5 stores layers ALPHABETICALLY, not in
│                  │  network order!
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  _extract_       │  Classifies each layer by name and weight shapes:
│  architecture    │    Conv1D:  (kernel, in_ch, filters) + (filters,)
│                  │    Dense:   (in_features, out_features) + (out,)
│                  │    LSTM:    (input_dim, 4×units) + (units, 4×units)
│                  │    BatchNorm: 4 × (features,)
│                  │
│                  │  Chains dense layers by data flow:
│                  │    Dense(128→64) → Dense(64→23) [not alphabetical]
│                  │
│                  │  Detects architecture type:
│                  │    has conv + lstm + concat → mixto
│                  │    has conv → cnn
│                  │    has lstm/gru → rnn
│                  │    else → mlp
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  _build_{arch}   │  Builds model in CORRECT layer order using
│                  │  extracted sizes (not h5 order).
│                  │
│                  │  tf.keras.backend.clear_session() called first
│                  │  to reset layer name counters.
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  model.load_     │  Weights load successfully because layer names
│  weights(path)   │  and shapes match exactly.
└──────────────────┘
```

### 11.2 The Dense Chain Problem

Dense layers in the h5 file might be stored as `dense`, `dense_1`, `dense_2` in alphabetical order, but the network order is determined by data flow: `Dense(2070→256) → Dense(256→128) → Dense(128→23)`. The `_chain_denses` function traces this by matching each layer's `out_features` to the next layer's `in_features`.

### 11.3 The Keras Naming Problem

When building multiple Sequential models in the same Python session, Keras auto-increments layer names globally. The first model gets `dense`, `dense_1`; the second gets `dense_2`, `dense_3`. But the checkpoint expects `dense`, `dense_1`. Solution: call `tf.keras.backend.clear_session()` before each model build.

---

## 12. Assumptions

The following assumptions are made explicitly:

| # | Assumption | Impact |
|---|---|---|
| 1 | Trade at closing price on rebalancing date | No slippage, no market impact |
| 2 | No transaction costs | Overestimates returns, especially for Strategy B |
| 3 | Model is frozen (no retraining during backtest) | Realistic — models are trained once and deployed |
| 4 | Constant-daily frac-diff assumption in FFD inversion | Affects magnitude of expected returns, not ranking |
| 5 | Cash earns 0% | Conservative (actual risk-free rate ~4-5% in 2025) |
| 6 | Dividends included via adjusted close prices | Consistent treatment across all portfolios |
| 7 | Survivorship bias (23 current S&P 500 members) | All assets survived the period — no delisting events |
| 8 | The scaler is reconstructed (not saved from training) | Exact match if using same dataset_utils pipeline |

---

## 13. Strengths

**Methodological rigour**: The FFD inversion is mathematically grounded — it uses the exact recursive relationship of the fractional differentiation operator rather than a crude approximation. The 90-day prediction horizon matches the holding period exactly.

**Robustness testing**: Rolling offsets (5 starting dates per model) reduce the risk of cherry-picking a favourable period. Results are reported as averages with standard deviations.

**Architecture-agnostic**: The `model_builder.py` auto-detection system means any new checkpoint can be tested without writing new code. This enables rapid experimentation.

**Clean separation of concerns**: Inference (TensorFlow-dependent) and analysis (pure Python) can run on different machines. The batch runner is fault-tolerant (try/except per model).

**Honest evaluation**: The comparison includes three strategies testing different aspects of the model's signal (ranking, magnitude as target, magnitude for sizing) against two benchmarks that represent passive alternatives.

---

## 14. Limitations & Known Weaknesses

### 14.1 Statistical Limitations

**Small sample size**: With only 3-4 rebalancing windows per backtest and 23 assets, Spearman correlations are unreliable (p > 0.50 in most cases). A single lucky or unlucky window can dominate the results.

**Single market regime**: The backtest covers 2025-2026, a period of [describe market conditions]. Results may not generalise to bear markets, high-volatility regimes, or recessions.

**Last window truncation**: The final rebalancing window may have fewer than 90 trading days if the data ends before the full horizon completes. This is flagged with alerts but affects comparability across windows.

### 14.2 Methodological Limitations

**No risk management**: The strategies do not consider volatility, correlation, or drawdown constraints. Strategy C can put 100% into a single asset.

**No transaction costs**: In practice, quarterly rebalancing of 5-position portfolios would incur bid-ask spreads and commissions, reducing returns by approximately 0.1-0.3% per rebalance.

**Survivorship bias**: The 23 assets are all current S&P 500 constituents. Companies that were delisted, merged, or fell out of the index during the sample period are excluded.

**FFD inversion approximation**: The constant-daily assumption in the inversion means predicted return magnitudes are approximate. This primarily affects Strategy C (proportional sizing) and Strategy B (take-profit thresholds).

### 14.3 Strategy-Specific Limitations

**Strategy B structural asymmetry**: Take-profit cuts winners while letting losers run for the full window. This creates negative skew in the return distribution. In trending markets (like 2025), this systematically underperforms Strategy A. The results confirm this across all models.

**Strategy C concentration risk**: When few assets have positive expected return, Strategy C concentrates heavily. In one observed window, 100% of capital went into a single stock (MRK), which happened to return +43%. This is not a sustainable strategy — it worked by luck, not by design.

---

## 15. Glossary

| Term | Definition |
|---|---|
| **FFD** | Fixed-width window Fractional Differentiation (López de Prado, 2018) |
| **d** | Fractional differentiation order (d=0.40 in this project) |
| **L** | Number of retained FFD weights (L=1,458 for d=0.40, threshold=1e-5) |
| **V_IN** | Input window size in trading days (varies by model: 5, 10, 30, 90) |
| **V_OUT** | Output/prediction horizon in trading days (90 for all models) |
| **TOP_K** | Number of top assets selected per rebalancing (default: 5) |
| **MAE** | Mean Absolute Error — the model's training loss function |
| **NAV** | Net Asset Value — the portfolio's total value normalised to 1.0 at start |
| **Rebalancing date** | Date when the portfolio is reconstituted based on new predictions |
| **Rolling offset** | Shifting the start date by N trading days to test sensitivity |
| **Spearman rho** | Rank correlation between predicted and actual returns |
| **Directional accuracy** | Percentage of assets where sign(predicted) = sign(actual) |

---

*End of Technical Documentation*
