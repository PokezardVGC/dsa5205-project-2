"""Unit tests for pipeline timing, feature engineering, and backtest integrity."""

import numpy as np
import pandas as pd
import pytest

from src.features import engineer_features
from src.backtest import backtest_strategy


def test_zero_lookahead_and_alignment():
    """Verify Target_Next_Return corresponds exactly to next-day forward return."""
    dates = pd.date_range("2020-01-01", periods=50, freq="B")
    prices = [100.0 * (1.02 ** i) for i in range(len(dates))]
    spy = [200.0 * (1.01 ** i) for i in range(len(dates))]
    df_raw = pd.DataFrame({"Price": prices, "SPY": spy}, index=dates)

    df_feat, features = engineer_features(df_raw)

    # Next return must match forward pct_change
    for i in range(len(df_feat) - 1):
        p_curr = df_feat["Price"].iloc[i]
        p_next = df_feat["Price"].iloc[i + 1]
        expected_ret = (p_next - p_curr) / p_curr
        assert df_feat["Target_Next_Return"].iloc[i] == pytest.approx(expected_ret, rel=1e-5)


def test_backtest_preserves_test_length():
    """Verify backtest_strategy does not drop the final day of the test set."""
    dates = pd.date_range("2022-01-01", periods=100, freq="B")
    np.random.seed(42)
    rets = np.random.normal(0.001, 0.02, len(dates))
    prices = 100 * np.cumprod(1 + rets)
    spy_rets = np.random.normal(0.0005, 0.01, len(dates))
    spy = 300 * np.cumprod(1 + spy_rets)

    df_raw = pd.DataFrame({"Price": prices, "SPY": spy}, index=dates)
    df_feat, features = engineer_features(df_raw)

    test_idx = df_feat.index[50:]
    preds = np.random.normal(0.001, 0.01, len(test_idx))

    df_bt = backtest_strategy(df_feat, test_index=test_idx, predictions=preds)

    assert len(df_bt) == len(test_idx), f"Backtest dropped rows! Expected {len(test_idx)}, got {len(df_bt)}"
    assert not df_bt["Strategy_Return"].isna().any(), "Strategy_Return contains NaNs"
    assert not df_bt["SPY_Return"].isna().any(), "SPY_Return contains NaNs"
    assert df_bt.index[-1] == test_idx[-1], "Final date was truncated"
