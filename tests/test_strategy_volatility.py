"""Unit tests for Extension 3: Volatility-Targeted Position Sizing Strategy."""

import numpy as np
import pandas as pd
import pytest

from src.strategy_volatility import (
    compute_volatility_features,
    calculate_performance_metrics,
    run_volatility_strategy,
)


def test_compute_volatility_features_bounds():
    """Verify position weights are strictly clamped between 0.0 and 1.0."""
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    # Generate prices with varying volatility
    np.random.seed(42)
    returns = np.random.normal(0, 0.05, len(dates))
    prices = 100 * np.cumprod(1 + returns)
    spy = 300 * np.cumprod(1 + returns * 0.5)

    df = pd.DataFrame({"Price": prices, "SPY": spy}, index=dates)
    df_feat = compute_volatility_features(df, vol_window=20, target_vol=0.40)

    # Weights must be bounded
    valid_positions = df_feat["Position"].dropna()
    assert (valid_positions >= 0.0).all(), "Positions must be non-negative"
    assert (valid_positions <= 1.0).all(), "Positions must not exceed 1.0 (no leverage)"


def test_zero_volatility_numerical_guard():
    """Verify epsilon guard prevents division by zero when price is flat."""
    dates = pd.date_range("2020-01-01", periods=30, freq="B")
    flat_prices = np.full(len(dates), 100.0)

    df = pd.DataFrame({"Price": flat_prices, "SPY": flat_prices}, index=dates)
    df_feat = compute_volatility_features(df, vol_window=20, target_vol=0.40)

    # Position should be clamped to 1.0 and finite (no NaN or inf)
    last_pos = df_feat["Position"].iloc[-1]
    assert np.isfinite(last_pos), "Position must be finite even with 0 volatility"
    assert last_pos == 1.0, "Position should be clamped to 1.0 during tranquil zero-vol regime"


def test_performance_metrics_consistency():
    """Verify Sharpe ratio and CAGR formulas."""
    daily_returns = pd.Series([0.01, -0.005, 0.02, -0.01, 0.015] * 50)
    initial_capital = 100000.0
    cum_wealth = initial_capital * (1 + daily_returns).cumprod()

    metrics = calculate_performance_metrics(
        returns=daily_returns,
        cum_wealth=cum_wealth,
        risk_free_rate=0.0,
        trading_days=252,
    )

    assert metrics["Annualized_Return"] > 0
    assert metrics["Annualized_Vol"] > 0
    assert metrics["Sharpe_Ratio"] == pytest.approx(
        metrics["Annualized_Return"] / metrics["Annualized_Vol"], rel=1e-5
    )
    assert metrics["Max_Drawdown"] <= 0.0, "Drawdown must be non-positive"


def test_transaction_cost_deduction():
    """Verify transaction cost is deducted upon turnover."""
    dates = pd.date_range("2020-01-01", periods=50, freq="B")
    prices = 100 + np.sin(np.linspace(0, 10, len(dates))) * 10
    spy = prices * 2

    df = pd.DataFrame({"Price": prices, "SPY": spy}, index=dates)
    df_bt, metrics = run_volatility_strategy(
        df, target_vol=0.40, transaction_cost=0.0010, split_ratio=0.5
    )

    assert "Fee_Drag" in df_bt.columns
    assert (df_bt["Fee_Drag"] >= 0.0).all(), "Fee drag must be non-negative"
    assert df_bt["Fee_Drag"].sum() > 0, "Active rebalancing should accumulate fees"
