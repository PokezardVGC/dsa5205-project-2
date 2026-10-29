r"""Consolidated Multi-Strategy Showdown Harness.

DSA5205 Machine Learning in Quantitative Finance — Project 2
Group 28 (Yu Jie, Marco, Diana)

This script executes the master out-of-sample showdown across all strategies:
1. Extension 1: Daily Dip-Buying via Ridge Regression (Yu Jie)
2. Extension 2a: Macro Trend Following via 200-Day SMA Filter (Marco)
3. Extension 2b: ML Momentum via Ridge L2 Regularization (Marco)
4. Extension 2c: ML Momentum via Lasso L1 Regularization (Marco)
5. Extension 3: Volatility-Targeted Position Sizing (\sigma_target = 40%) (Diana)
6. Unconstrained NVDA Buy & Hold Benchmark
7. SPY ETF Benchmark

All models are evaluated on the exact same 449 out-of-sample trading days
(2022-03-17 to 2023-12-28) with uniform 10 bps (0.0010) transaction cost friction.
"""

from pathlib import Path
import sys
from typing import Dict, Any, Tuple, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import fetch_data
from src.features import engineer_features
from src.model import train_and_select_model
from src.backtest import backtest_strategy
from src.strategy_volatility import run_volatility_strategy, calculate_performance_metrics

FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def run_showdown(
    initial_capital: float = 100000.0,
    transaction_cost: float = 0.0010,
    target_vol: float = 0.40,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """Execute all strategies on the identical out-of-sample window and compute metrics."""
    print("=" * 80)
    print("DSA5205 Project 2 — Master Multi-Strategy Showdown")
    print(f"Group 28 | Initial Capital: ${initial_capital:,.2f} | Friction: {transaction_cost * 10000:.0f} bps")
    print("=" * 80)

    # 1. Shared Data Ingestion
    print("\n[1/6] Ingesting unified market data for NVDA and SPY (2015–2023)...")
    df_raw = fetch_data("NVDA", "SPY", "2015-01-01", "2024-01-01")

    # 2. Extension 1: Ridge Dip-Buying (Yu Jie)
    print("\n[2/6] Running Extension 1: Daily Dip-Buying via Ridge Regression (Yu Jie)...")
    df_features, f_names = engineer_features(df_raw)
    _, preds_ext1, test_idx = train_and_select_model(df_features, f_names)
    df_ext1 = backtest_strategy(
        df_features,
        test_index=test_idx,
        predictions=preds_ext1,
        initial_capital=initial_capital,
        transaction_cost=transaction_cost,
    )

    # 3. Extension 2: Macro Trend & ML Momentum (Marco)
    print("\n[3/6] Running Extension 2: Macro Trend & Regularized Momentum (Marco)...")
    df_m = df_raw.copy()
    df_m["Return"] = df_m["Price"].pct_change()
    df_m["Target_Next_Return"] = df_m["Return"].shift(-1)
    df_m["SMA_50"] = df_m["Price"].rolling(50).mean()
    df_m["SMA_200"] = df_m["Price"].rolling(200).mean()
    df_m["Dist_SMA50"] = (df_m["Price"] - df_m["SMA_50"]) / df_m["SMA_50"]
    df_m["Dist_SMA200"] = (df_m["Price"] - df_m["SMA_200"]) / df_m["SMA_200"]
    df_m["Mom_10d"] = df_m["Price"].pct_change(10)
    df_m["Mom_20d"] = df_m["Price"].pct_change(20)

    m_features = ["Dist_SMA50", "Dist_SMA200", "Mom_10d", "Mom_20d"]
    train_m = df_m.loc[df_m.index < test_idx[0]].dropna().copy()
    test_m = df_m.loc[test_idx].copy()

    # 2a: 200-day SMA Trend Filter
    test_m["Bull_Regime"] = test_m["Price"] > test_m["SMA_200"]
    test_m["Pos_2a"] = np.where(test_m["Bull_Regime"], 1.0, 0.0)
    test_m["Chg_2a"] = test_m["Pos_2a"].diff().fillna(test_m["Pos_2a"].iloc[0]).abs()
    test_m["Ret_2a"] = (test_m["Pos_2a"] * test_m["Target_Next_Return"]) - (test_m["Chg_2a"] * transaction_cost)
    test_m["Cum_2a"] = initial_capital * (1.0 + test_m["Ret_2a"]).cumprod()

    # 2b & 2c: Ridge and Lasso Momentum
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(train_m[m_features])
    y_tr = train_m["Target_Next_Return"].clip(-0.5, 0.5)
    X_te = scaler.transform(test_m[m_features])

    ridge_m = Ridge(alpha=10.0).fit(X_tr, y_tr)
    test_m["Pos_2b"] = np.where(ridge_m.predict(X_te) > y_tr.mean(), 1.0, 0.0)
    test_m["Chg_2b"] = test_m["Pos_2b"].diff().fillna(test_m["Pos_2b"].iloc[0]).abs()
    test_m["Ret_2b"] = (test_m["Pos_2b"] * test_m["Target_Next_Return"]) - (test_m["Chg_2b"] * transaction_cost)
    test_m["Cum_2b"] = initial_capital * (1.0 + test_m["Ret_2b"]).cumprod()

    lasso_m = Lasso(alpha=0.00001, random_state=42, max_iter=10000).fit(X_tr, y_tr)
    test_m["Pos_2c"] = np.where(lasso_m.predict(X_te) > y_tr.mean(), 1.0, 0.0)
    test_m["Chg_2c"] = test_m["Pos_2c"].diff().fillna(test_m["Pos_2c"].iloc[0]).abs()
    test_m["Ret_2c"] = (test_m["Pos_2c"] * test_m["Target_Next_Return"]) - (test_m["Chg_2c"] * transaction_cost)
    test_m["Cum_2c"] = initial_capital * (1.0 + test_m["Ret_2c"]).cumprod()

    # 4. Extension 3: Volatility-Targeted Position Sizing (Diana)
    print("\n[4/6] Running Extension 3: Volatility-Targeted Sizing (Diana)...")
    df_ext3, _ = run_volatility_strategy(
        df_raw=df_raw,
        target_vol=target_vol,
        vol_window=20,
        transaction_cost=transaction_cost,
        test_index=test_idx,
        initial_capital=initial_capital,
    )

    # 5. Compile Showdown DataFrame
    print("\n[5/6] Aligning time series and compiling performance metrics...")
    df_showdown = pd.DataFrame(index=test_idx)
    df_showdown["Ext1_Dip_Buying"] = df_ext1["Cum_Strategy"]
    df_showdown["Ext2a_SMA200_Trend"] = test_m["Cum_2a"]
    df_showdown["Ext2b_Ridge_Mom"] = test_m["Cum_2b"]
    df_showdown["Ext2c_Lasso_Mom"] = test_m["Cum_2c"]
    df_showdown["Ext3_Vol_Target"] = df_ext3["Cum_Strategy"]
    df_showdown["NVDA_Buy_Hold"] = df_ext3["Cum_BuyHold"]
    df_showdown["SPY_Benchmark"] = df_ext3["Cum_SPY"]

    # Compute standardized performance dictionary
    results: Dict[str, Dict[str, float]] = {
        "Ext 1: Daily Dip-Buying (Yu Jie)": calculate_performance_metrics(
            df_ext1["Strategy_Return"], df_ext1["Cum_Strategy"], df_ext1["Position_Change"]
        ),
        "Ext 2a: SMA 200 Macro Trend (Marco)": calculate_performance_metrics(
            test_m["Ret_2a"], test_m["Cum_2a"], test_m["Chg_2a"]
        ),
        "Ext 2b: Ridge Momentum (Marco)": calculate_performance_metrics(
            test_m["Ret_2b"], test_m["Cum_2b"], test_m["Chg_2b"]
        ),
        "Ext 2c: Lasso Momentum (Marco)": calculate_performance_metrics(
            test_m["Ret_2c"], test_m["Cum_2c"], test_m["Chg_2c"]
        ),
        "Ext 3: Volatility Targeting 40% (Diana)": calculate_performance_metrics(
            df_ext3["Strategy_Return"], df_ext3["Cum_Strategy"], df_ext3["Position_Change"]
        ),
        "NVDA Buy & Hold Benchmark": calculate_performance_metrics(
            df_ext3["BuyHold_Return"], df_ext3["Cum_BuyHold"], pd.Series(0.0, index=test_idx)
        ),
        "SPY ETF Benchmark": calculate_performance_metrics(
            df_ext3["SPY_Return"], df_ext3["Cum_SPY"], pd.Series(0.0, index=test_idx)
        ),
    }

    # Print summary table
    print("\n" + "=" * 95)
    print(
        f"{'Strategy / Benchmark':<38} | {'Ann. Ret':<9} | {'CAGR':<9} | {'Ann. Vol':<9} | {'Sharpe':<7} | {'MaxDD':<8} | {'Turnover':<8}"
    )
    print("-" * 95)
    for name, m in results.items():
        print(
            f"{name:<38} | {m['Annualized_Return']*100:6.2f}%   | {m['CAGR']*100:6.2f}%   | {m['Annualized_Vol']*100:6.2f}%   | {m['Sharpe_Ratio']:6.2f}  | {m['Max_Drawdown']*100:6.2f}%  | {m['Annualized_Turnover']:6.2f}"
        )
    print("=" * 95)

    return df_showdown, results


def plot_showdown_figures(df_showdown: pd.DataFrame, results: Dict[str, Dict[str, float]]) -> None:
    """Generate high-resolution comparative figures for the final report."""
    print("\n[6/6] Generating publication-grade comparison figures in figures/...")
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = {
        "Ext1_Dip_Buying": "#1f77b4",        # Blue
        "Ext2a_SMA200_Trend": "#2ca02c",     # Green
        "Ext2b_Ridge_Mom": "#9467bd",        # Purple
        "Ext2c_Lasso_Mom": "#8c564b",        # Brown
        "Ext3_Vol_Target": "#d62728",        # Red
        "NVDA_Buy_Hold": "#ff7f0e",          # Orange
        "SPY_Benchmark": "#7f7f7f",          # Gray
    }

    labels = {
        "Ext1_Dip_Buying": "Ext 1: Dip-Buying (Yu Jie)",
        "Ext2a_SMA200_Trend": "Ext 2a: SMA 200 Trend (Marco)",
        "Ext2b_Ridge_Mom": "Ext 2b: Ridge Momentum (Marco)",
        "Ext2c_Lasso_Mom": "Ext 2c: Lasso Momentum (Marco)",
        "Ext3_Vol_Target": "Ext 3: Volatility Targeting (Diana)",
        "NVDA_Buy_Hold": "NVDA Buy & Hold Benchmark",
        "SPY_Benchmark": "SPY ETF Benchmark",
    }

    # -------------------------------------------------------------
    # Chart 1: Master Cumulative Wealth Curves (Semi-Log Scale)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    for col in df_showdown.columns:
        lw = 2.5 if col in ["Ext3_Vol_Target", "Ext2a_SMA200_Trend"] else (2.0 if "Benchmark" in labels[col] or "NVDA" in labels[col] else 1.5)
        ls = "--" if "Benchmark" in labels[col] or "NVDA" in labels[col] else "-"
        ax.plot(
            df_showdown.index,
            df_showdown[col],
            label=labels.get(col, col),
            color=colors.get(col, "#000000"),
            linewidth=lw,
            linestyle=ls,
            alpha=0.9 if lw >= 2.0 else 0.65,
        )

    ax.set_yscale("log")
    ax.set_title("Figure 10: Master Multi-Strategy Cumulative Wealth (Log Scale, 2022–2023)", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Portfolio Value ($) [Initial = $100,000]", fontsize=11, fontweight="bold")
    ax.set_xlabel("Trading Date (Out-of-Sample Window: 449 Days)", fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", frameon=True, fontsize=9)
    plt.tight_layout()
    cum_path = FIGURES_DIR / "consolidated_showdown_equity.png"
    plt.savefig(cum_path)
    plt.close()
    print(f"  -> Saved cumulative wealth showdown: {cum_path}")

    # -------------------------------------------------------------
    # Chart 2: Underwater Drawdown Profile Overlay
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
    primary_strats = ["Ext2a_SMA200_Trend", "Ext3_Vol_Target", "Ext1_Dip_Buying", "NVDA_Buy_Hold", "SPY_Benchmark"]
    for col in primary_strats:
        roll_max = df_showdown[col].cummax()
        dd = (df_showdown[col] - roll_max) / roll_max * 100.0
        lw = 2.2 if col in ["Ext3_Vol_Target", "Ext2a_SMA200_Trend"] else 1.5
        ls = "--" if col in ["NVDA_Buy_Hold", "SPY_Benchmark"] else "-"
        ax.plot(df_showdown.index, dd, label=labels.get(col, col), color=colors.get(col), linewidth=lw, linestyle=ls)

    ax.set_title("Figure 11: Master Multi-Strategy Underwater Drawdown Profile (2022–2023)", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Drawdown (%)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Trading Date", fontsize=11, fontweight="bold")
    ax.axhline(-20, color="gray", linestyle=":", alpha=0.5, label="-20% Bear Threshold")
    ax.legend(loc="lower left", frameon=True, fontsize=9)
    plt.tight_layout()
    dd_path = FIGURES_DIR / "consolidated_showdown_drawdown.png"
    plt.savefig(dd_path)
    plt.close()
    print(f"  -> Saved drawdown profile showdown: {dd_path}")

    # -------------------------------------------------------------
    # Chart 3: Risk-Return Frontier (Annualized Vol vs Annualized Return)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    for name, m in results.items():
        vol = m["Annualized_Vol"] * 100.0
        ret = m["Annualized_Return"] * 100.0
        sharpe = m["Sharpe_Ratio"]
        
        # Color mapping by strategy family
        if "Diana" in name:
            c = "#d62728"
            marker = "o"
            s = 180
        elif "SMA 200" in name:
            c = "#2ca02c"
            marker = "s"
            s = 160
        elif "Yu Jie" in name:
            c = "#1f77b4"
            marker = "^"
            s = 140
        elif "NVDA" in name:
            c = "#ff7f0e"
            marker = "D"
            s = 150
        elif "SPY" in name:
            c = "#7f7f7f"
            marker = "P"
            s = 130
        else:
            c = "#8c564b"
            marker = "x"
            s = 110

        edge = "black" if marker != "x" else None
        ax.scatter(vol, ret, color=c, s=s, marker=marker, edgecolors=edge, linewidth=1.2, zorder=5)
        ax.annotate(
            f"{name.split('(')[0].strip()}\n(Sharpe: {sharpe:.2f})",
            (vol, ret),
            textcoords="offset points",
            xytext=(10, 5),
            fontsize=8.5,
            fontweight="bold" if ("Diana" in name or "SMA" in name) else "normal",
        )

    ax.set_title("Figure 12: Risk-Return Tradeoff Frontier (Out-of-Sample 2022–2023)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Annualized Volatility (%)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Annualized Arithmetic Return (%)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    rr_path = FIGURES_DIR / "consolidated_risk_return_tradeoff.png"
    plt.savefig(rr_path)
    plt.close()
    print(f"  -> Saved risk-return frontier: {rr_path}")


def main() -> None:
    df_showdown, results = run_showdown()
    plot_showdown_figures(df_showdown, results)
    print("\nShowdown complete! All figures and metrics successfully produced.")


if __name__ == "__main__":
    main()
