# DSA5205 Project 2 — Systematic Multi-Strategy Asset Allocation on NVIDIA (NVDA)

**Course:** DSA5205 Machine Learning in Quantitative Finance  
**Group:** Group 28  
**Team Members:** Yu Jie, Marco, Diana  
**Academic Year:** 2026/2027 — National University of Singapore  

---

## 1. Executive Summary & Strategy Overview

This repository houses the reproducible quantitative trading research codebase for **DSA5205 Project 2**. We investigate the 9-year market cycle of **NVIDIA Corporation (`NVDA`)** (2015–2023) benchmarked against **SPY** (S&P 500 ETF) and unconstrained **NVDA Buy & Hold**.

Rather than testing a single monolithic model, Group 28 deploys a **parallel 3-hypothesis architecture** investigating three foundational market anomalies:

1. **Extension 1 (Yu Jie — Daily Mean Reversion / "Buy the Dip"):**  
   Predicts next-day corrective price rebounds using short return lags ($R_t, R_{t-1}, R_{t-2}$) and moving average deviations ($\text{Dist}_{\text{MA10}}, \text{Dist}_{\text{MA20}}$) regularized by Ridge regression with chronological `TimeSeriesSplit` cross-validation.
2. **Extension 2 (Marco — Macro Trend Following & Regularized Momentum):**  
   Evaluates a non-parametric 200-day Simple Moving Average (SMA 200) structural regime filter alongside machine learning momentum models ($L_2$ Ridge and $L_1$ Lasso) to avoid secular bear markets (such as the 2022 tech drawdown).
3. **Extension 3 (Diana — Dynamic Volatility-Targeted Position Sizing):**  
   Addresses the "volatility drag paradox" ($g \approx \mu - \frac{1}{2}\sigma^2$) by dynamically scaling daily position exposure $w_t = \min\left(1.0, \ \frac{\sigma_{\text{target}}}{\sigma_{t, 20\text{d}}}\right)$ inversely with 20-day realized volatility, preserving capital during market panics without margin borrowing.

All strategies are evaluated across the exact same **449 out-of-sample trading days** (2022-03-17 to 2023-12-28) with a realistic **10 basis points (0.10%)** transaction fee deducted on every unit of portfolio turnover.

---

## 2. Environment Setup

### Prerequisites
* Python 3.10+ (tested on Python 3.10, 3.11, and 3.12)
* Git

### Installation
Clone the repository and set up a clean virtual environment:

```bash
# Clone the repository (or navigate to project directory)
cd dsa5205-project-2

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install locked dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Reproduction & Execution Guide

### A. Run Master Consolidated Showdown (All Strategies Side-by-Side)
To run the full 5-way out-of-sample showdown across all three member extensions and both benchmarks:

```bash
python scripts/run_consolidated_showdown.py
```
**Expected Output:**
* Prints the comparative performance matrix (Annualized Return, CAGR, Annualized Volatility, Sharpe Ratio, Maximum Drawdown, and Annualized Turnover).
* Generates three publication figures in `figures/`:
  - `figures/consolidated_showdown_equity.png` (Semi-log cumulative wealth trajectories)
  - `figures/consolidated_showdown_drawdown.png` (Underwater drawdown profiles)
  - `figures/consolidated_risk_return_tradeoff.png` (Risk-return frontier with Sharpe ratio lines)

---

### B. Run Extension 1 Pipeline (Yu Jie — Ridge Mean Reversion)
```bash
python main.py
```
**Expected Output:**
* Ingests data from `yfinance` with auto-splits and dividend adjustments.
* Chronological split: 1,795 train/val days (2015–2021) and 449 test days (2022–2023).
* 5-fold `TimeSeriesSplit` selects optimal $\alpha = 1000.0$.
* Outputs out-of-sample performance: Strategy Return ~45.96%, Volatility ~54.06%, Sharpe ~0.85.

To generate Extension 1 tear sheets:
```bash
python scripts/split_performance.py
```

---

### C. Run Extension 2 Pipeline (Marco — Macro Trend & ML Momentum)
```bash
python scripts/run_extension2_momentum.py
```
**Expected Output:**
* Backtests Rule-Based Trend (SMA 200), Ridge Momentum ($L_2$), and Lasso Momentum ($L_1$) against NVDA and SPY benchmarks.
* Generates regime and performance figures in `figures/Ext2_*.png`.

---

### D. Run Extension 3 Pipeline (Diana — Volatility-Targeted Position Sizing)
```bash
python src/strategy_volatility.py
```
**Expected Output:**
* Evaluates dynamic inverse-volatility weighting ($\sigma_{\text{target}} = 40\%$).
* Runs sensitivity ablation across $\sigma_{\text{target}} \in [20\%, 60\%]$.
* Outputs performance: Strategy Return ~47.63%, CAGR ~44.68%, Volatility ~42.14%, Sharpe ~1.13, Max Drawdown ~-42.45% (vs -60.80% for unconstrained NVDA).
* Generates figures in `figures/ext3_*.png`.

---

### E. Run Automated Test Suite
To verify numerical guards (e.g. zero-volatility division protection, leverage caps $\le 1.0$, transaction fee deductions):

```bash
pytest
```
**Expected Output:** All unit tests pass cleanly.

---

## 4. Standardized Out-of-Sample Performance Showdown

All metrics below are computed on the identical 449 out-of-sample trading days (**2022-03-17 to 2023-12-28**) with **10 bps (0.0010)** transaction cost friction and **$100,000** initial capital:

| Strategy / Benchmark | Contributor | Ann. Return | CAGR | Ann. Vol | Sharpe Ratio | Max Drawdown | Turnover |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ext 2a: SMA 200 Macro Trend** | Marco | **63.83%** | **70.36%** | **37.88%** | **1.68** | **-19.30%** | **2.81** |
| **Ext 3: Volatility Targeting 40%** | Diana | **47.63%** | **44.68%** | **42.14%** | **1.13** | **-42.45%** | **5.10** |
| **NVDA Buy & Hold Benchmark** | Baseline | 54.06% | 42.28% | 55.49% | 0.97 | -60.80% | 0.00 |
| **Ext 1: Daily Dip-Buying (Ridge)** | Yu Jie | 45.96% | 32.31% | 54.06% | 0.85 | -59.40% | 39.85 |
| **SPY ETF Market Benchmark** | Baseline | 7.78% | 5.48% | 19.18% | 0.41 | -22.09% | 0.00 |
| **Ext 2c: Lasso Momentum ($L_1$)** | Marco | 14.20% | 5.56% | 42.52% | 0.33 | -44.61% | 29.18 |
| **Ext 2b: Ridge Momentum ($L_2$)** | Marco | 12.19% | 3.39% | 42.67% | 0.29 | -48.35% | 30.31 |

---

## 5. Repository Structure

```text
dsa5205-project-2/
├── README.md                           <- Master reproducible execution guide
├── requirements.txt                     <- Pinned environment dependencies
├── pytest.ini                          <- Pytest test configuration
├── main.py                             <- Pipeline entry point (Extension 1)
├── src/
│   ├── __init__.py
│   ├── data.py                         <- Shared split-adjusted data ingestion via yfinance
│   ├── features.py                     <- Zero-lookahead feature and target return engineering
│   ├── model.py                        <- Expanding TimeSeriesSplit CV and Ridge model selection
│   ├── backtest.py                     <- Rebalancing execution with 10 bps transaction costs
│   ├── evaluate.py                     <- Out-of-sample performance and risk metric calculators
│   └── strategy_volatility.py          <- Extension 3: Dynamic Volatility Targeting (Diana)
├── scripts/
│   ├── run_consolidated_showdown.py    <- Master 5-way showdown harness across all extensions
│   ├── run_extension2_momentum.py      <- Extension 2: Macro Trend & Regularized Momentum (Marco)
│   ├── split_performance.py            <- Generates detailed tear-sheet charts
│   ├── split_eda.py                    <- Generates exploratory data analysis charts
│   └── generate_split_report.py        <- Generates markdown tear-sheet report
├── tests/
│   ├── __init__.py
│   └── test_strategy_volatility.py     <- Unit tests for boundary guards, clamping, and fee drag
├── figures/                            <- Publication-ready 300 DPI figures for LaTeX
└── report/
    ├── Group_28_Report_Final.tex       <- Complete 25-page academic report in LaTeX
    └── Group_28_Report_Final.pdf       <- Compiled submission PDF
```

---

## 6. Key Guardrails & Quantitative Rigor

* **Zero Lookahead Bias:** Feature values at close $t$ only incorporate data known up to time $t$. Returns earned on positions established at close $t$ use forward returns $R_{t+1} = \frac{P_{t+1}-P_t}{P_t}$ via `shift(-1)`.
* **Standardized Holdout Period:** All strategies are tested on the exact same 449 calendar dates (2022-03-17 to 2023-12-28).
* **Friction Integrity:** Every position change incurs $10\text{ bps} \times |\Delta w_t|$, penalizing high-frequency churn.
* **No Leverage:** Positions $w_t$ are bounded in $[0.0, 1.0]$.
