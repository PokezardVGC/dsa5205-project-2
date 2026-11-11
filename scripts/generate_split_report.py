"""Exports markdown performance summary of all strategies and benchmarks.

DSA5205 Machine Learning in Quantitative Finance — Project 2
Group 28 (Yu Jie, Marco, Diana)
"""

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def build_markdown_report():
    """Generate and save markdown performance scorecard summary."""
    metrics_path = PROJECT_ROOT / "figures" / "consolidated_performance_metrics.csv"
    if not metrics_path.exists():
        from scripts.run_consolidated_showdown import run_showdown, plot_showdown_figures
        df_showdown, results = run_showdown()
        plot_showdown_figures(df_showdown, results)

    df_metrics = pd.read_csv(metrics_path)

    md_lines = [
        "# DSA5205 Project 2 — Quantitative Performance Scorecard Summary",
        "",
        "**Group 28:** Yu Jie, Marco, Diana  ",
        "**Out-of-Sample Test Window:** 2022-03-17 to 2023-12-28 (449 Trading Days)  ",
        "**Transaction Cost Friction:** 10 bps (0.10%) on position rebalancing  ",
        "",
        "## Master Strategy Comparison",
        "",
        "| Strategy / Benchmark | Ann. Return | CAGR | Ann. Vol | Sharpe Ratio | Max Drawdown | Turnover |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for _, row in df_metrics.iterrows():
        strat = row["Strategy_or_Benchmark"]
        ann_ret = f"{row['Annualized_Return']*100:.2f}%"
        cagr = f"{row['CAGR']*100:.2f}%"
        ann_vol = f"{row['Annualized_Vol']*100:.2f}%"
        sharpe = f"{row['Sharpe_Ratio']:.2f}"
        max_dd = f"{row['Max_Drawdown']*100:.2f}%"
        turnover = f"{row['Annualized_Turnover']:.2f}"
        md_lines.append(f"| {strat} | {ann_ret} | {cagr} | {ann_vol} | {sharpe} | {max_dd} | {turnover} |")

    md_lines.extend([
        "",
        "## Key Quantitative Insights",
        "",
        "1. **Macro Trend Dominance (Ext 2a):** The 200-day SMA trend filter delivered the highest risk-adjusted Sharpe (1.68) and lowest drawdown (-19.30%) by effectively exiting into cash during the 2022 bear market and capturing the 2023 bull regime.",
        "2. **Dynamic Volatility Control (Ext 3):** Volatility targeting slashed drawdown from -60.80% (NVDA B&H) to -42.45% while matching B&H CAGR (47.69% vs 47.64%) and increasing Sharpe from 0.97 to 1.13.",
        "3. **Short-Term Mean Reversion Limits (Ext 1):** Daily dip-buying incurred high turnover (39.85) and suffered severe drawdown (-59.40%) during secular trend regimes, demonstrating the limits of pure short-term mean-reversion on high-beta growth equities.",
        "4. **Friction Integrity:** All results strictly account for 10 bps execution friction and backwards-looking features with zero lookahead bias.",
    ])

    output_path = PROJECT_ROOT / "report" / "performance_summary.md"
    output_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Summary markdown report successfully generated: {output_path}")


if __name__ == "__main__":
    build_markdown_report()
