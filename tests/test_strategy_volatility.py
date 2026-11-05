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
    """Verify Sharpe ratio, terminal wealth identity, and CAGR invariance."""
    daily_returns = pd.Series([0.01, -0.005, 0.02, -0.01, 0.015] * 50)
    initial_capital = 100000.0
    cum_wealth = initial_capital * (1 + daily_returns).cumprod()

    metrics = calculate_performance_metrics(
        returns=daily_returns,
        cum_wealth=cum_wealth,
        risk_free_rate=0.0,
        trading_days=252,
        initial_capital=initial_capital,
    )

    assert metrics["Annualized_Return"] > 0
    assert metrics["Annualized_Vol"] > 0
    assert metrics["Sharpe_Ratio"] == pytest.approx(
        metrics["Annualized_Return"] / metrics["Annualized_Vol"], rel=1e-5
    )
    assert metrics["Max_Drawdown"] <= 0.0, "Drawdown must be non-positive"

    # Strict invariant: Terminal wealth must equal W_0 * prod(1 + r)
    terminal_wealth = cum_wealth.iloc[-1]
    expected_terminal = initial_capital * np.prod(1.0 + daily_returns)
    assert terminal_wealth == pytest.approx(expected_terminal, rel=1e-6)

    # Strict invariant: CAGR compounded over n days must exactly reconstruct terminal wealth
    n_days = len(daily_returns)
    reconstructed_wealth = initial_capital * ((1.0 + metrics["CAGR"]) ** (n_days / 252))
    assert reconstructed_wealth == pytest.approx(terminal_wealth, rel=1e-5)


def test_cagr_first_day_nonzero_invariance():
    """Verify CAGR correctly accounts for Day 1 return without off-by-one dilution."""
    initial_capital = 100000.0

    # Scenario A: Day 1 +10%, followed by 9 flat days
    returns_pos = pd.Series([0.10] + [0.0] * 9)
    wealth_pos = initial_capital * (1 + returns_pos).cumprod()
    m_pos = calculate_performance_metrics(returns_pos, wealth_pos, initial_capital=initial_capital)
    expected_cagr_pos = (1.10) ** (252 / 10) - 1.0
    assert m_pos["CAGR"] == pytest.approx(expected_cagr_pos, rel=1e-5)
    assert m_pos["Max_Drawdown"] == 0.0

    # Scenario B: Day 1 -10%, followed by 9 flat days
    returns_neg = pd.Series([-0.10] + [0.0] * 9)
    wealth_neg = initial_capital * (1 + returns_neg).cumprod()
    m_neg = calculate_performance_metrics(returns_neg, wealth_neg, initial_capital=initial_capital)
    expected_cagr_neg = (0.90) ** (252 / 10) - 1.0
    assert m_neg["CAGR"] == pytest.approx(expected_cagr_neg, rel=1e-5)
    # Day 1 loss from initial capital must be registered as -10% Max Drawdown
    assert m_neg["Max_Drawdown"] == pytest.approx(-0.10, rel=1e-5)


def test_cagr_constant_and_zero_returns():
    """Verify edge cases: zero returns and constant drift."""
    initial_capital = 100000.0

    # Zero returns across 50 days -> CAGR must be exactly 0.0%
    zero_ret = pd.Series([0.0] * 50)
    zero_w = initial_capital * (1 + zero_ret).cumprod()
    m_zero = calculate_performance_metrics(zero_ret, zero_w, initial_capital=initial_capital)
    assert m_zero["CAGR"] == pytest.approx(0.0, abs=1e-9)
    assert m_zero["Max_Drawdown"] == pytest.approx(0.0, abs=1e-9)

    # Constant 10 bps daily return
    const_ret = pd.Series([0.0010] * 252)
    const_w = initial_capital * (1 + const_ret).cumprod()
    m_const = calculate_performance_metrics(const_ret, const_w, initial_capital=initial_capital)
    expected_cagr_const = (1.0010 ** 252) - 1.0
    assert m_const["CAGR"] == pytest.approx(expected_cagr_const, rel=1e-5)


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
