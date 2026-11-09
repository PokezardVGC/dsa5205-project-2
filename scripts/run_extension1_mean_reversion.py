r"""Extension 1 Standalone Pipeline: Daily Dip-Buying via Ridge Regression.

DSA5205 Machine Learning in Quantitative Finance — Project 2
Group 28 (Yu Jie — Strategy 1)

This script executes the complete standalone pipeline for Strategy 1:
1. Unified Data Ingestion for NVDA and SPY (2015–2023)
2. Past-Only Feature Engineering (return lags & moving average distances)
3. Model Training & Hyperparameter Selection via 5-Fold TimeSeriesSplit
4. Out-of-Sample Backtesting with 10 bps transaction costs
5. Annualized Performance Evaluation
"""

import sys
import warnings
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import fetch_data
from src.features import engineer_features
from src.model import train_and_select_model
from src.backtest import backtest_strategy
from src.evaluate import evaluate_and_plot


def run_strategy1():
    """Run Strategy 1 standalone pipeline."""
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    ASSET = "NVDA"
    BENCHMARK = "SPY"
    START_DATE = "2015-01-01"
    END_DATE = "2024-01-01"

    print("=" * 80)
    print("DSA5205 Project 2 — Strategy 1: Daily Dip-Buying (Yu Jie)")
    print("=" * 80)

    # 1. Fetch Data
    df_raw = fetch_data(ASSET, BENCHMARK, START_DATE, END_DATE)

    # 2. Engineer Features
    df_features, feature_names = engineer_features(df_raw)

    # 3. Model Training & Selection
    best_model, predictions, test_index = train_and_select_model(df_features, feature_names)

    # 4. Backtest Strategy
    df_bt = backtest_strategy(df_features, test_index, predictions)

    # 5. Evaluate and Plot
    evaluate_and_plot(df_bt)


if __name__ == "__main__":
    run_strategy1()
