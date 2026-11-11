# DSA5205 Project 2 — Quantitative Performance Scorecard Summary

**Group 28:** Yu Jie, Marco, Diana  
**Out-of-Sample Test Window:** 2022-03-17 to 2023-12-28 (449 Trading Days)  
**Transaction Cost Friction:** 10 bps (0.10%) on position rebalancing  

## Master Strategy Comparison

| Strategy / Benchmark | Ann. Return | CAGR | Ann. Vol | Sharpe Ratio | Max Drawdown | Turnover |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Ext 1: Daily Dip-Buying (Yu Jie) | 45.96% | 37.22% | 54.06% | 0.85 | -59.40% | 39.85 |
| Ext 2a: SMA 200 Macro Trend (Marco) | 63.83% | 76.68% | 37.88% | 1.68 | -19.30% | 2.81 |
| Ext 2b: Ridge Momentum (Marco) | 12.19% | 3.39% | 42.67% | 0.29 | -48.35% | 30.31 |
| Ext 2c: Lasso Momentum (Marco) | 14.20% | 5.56% | 42.52% | 0.33 | -44.61% | 29.18 |
| Ext 3: Volatility Targeting 40% (Diana) | 47.63% | 47.69% | 42.14% | 1.13 | -42.45% | 5.10 |
| NVDA Buy & Hold Benchmark | 54.06% | 47.64% | 55.49% | 0.97 | -60.80% | 0.00 |
| SPY ETF Benchmark | 7.78% | 6.13% | 19.18% | 0.41 | -22.09% | 0.00 |

## Key Quantitative Insights

1. **Macro Trend Dominance (Ext 2a):** The 200-day SMA trend filter delivered the highest risk-adjusted Sharpe (1.68) and lowest drawdown (-19.30%) by effectively exiting into cash during the 2022 bear market and capturing the 2023 bull regime.
2. **Dynamic Volatility Control (Ext 3):** Volatility targeting slashed drawdown from -60.80% (NVDA B&H) to -42.45% while matching B&H CAGR (47.69% vs 47.64%) and increasing Sharpe from 0.97 to 1.13.
3. **Short-Term Mean Reversion Limits (Ext 1):** Daily dip-buying incurred high turnover (39.85) and suffered severe drawdown (-59.40%) during secular trend regimes, demonstrating the limits of pure short-term mean-reversion on high-beta growth equities.
4. **Friction Integrity:** All results strictly account for 10 bps execution friction and backwards-looking features with zero lookahead bias.
