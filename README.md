# Group 28 Project 2: Mean Reversion ("Buy the Dip") Strategy

This repository contains the code for Group 28's quantitative trading strategy assignment.
The strategy implements a Ridge regression model to predict next-day mean reversion in NVDA based on short-term past returns and moving average distances.

## Setup Instructions

1. Ensure you have Python 3.9+ installed.
2. Create and activate a virtual environment (recommended):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Code

To execute the end-to-end pipeline (Data Fetch -> Feature Engineering -> Train/Val/Test Split -> Ridge Regression -> Backtesting -> Evaluation):

```bash
python main.py
```

### Expected Output
The script will output the Train/Val/Test split sizes, the selected `alpha` for Ridge Regression, and the annualized performance metrics for the Strategy, NVDA Buy & Hold, and SPY Buy & Hold.

It will also generate a `performance_chart.png` in the root directory showing the cumulative portfolio value over the Out-of-Sample period.

## Code Structure
- `main.py`: Entry point for the pipeline.
- `src/data.py`: Fetches historical data using `yfinance`.
- `src/features.py`: Computes returns, moving average distances, and target labels.
- `src/model.py`: Handles chronological data splitting, feature scaling, and Ridge training/validation.
- `src/backtest.py`: Simulates daily trading, applies 10 bps transaction costs, and tracks portfolio value.
- `src/evaluate.py`: Calculates Annualized Return, Volatility, and Sharpe ratios, and plots the results.
