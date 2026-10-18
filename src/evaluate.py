import numpy as np
import matplotlib.pyplot as plt

def calculate_metrics(returns):
    ann_return = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol != 0 else 0
    return ann_return, ann_vol, sharpe

def evaluate_and_plot(df_bt):
    strat_ret, strat_vol, strat_sharpe = calculate_metrics(df_bt['Strategy_Return'])
    bh_ret, bh_vol, bh_sharpe = calculate_metrics(df_bt['BuyHold_Return'])
    spy_ret, spy_vol, spy_sharpe = calculate_metrics(df_bt['SPY_Return'])

    print("\n--- Out-of-Sample Performance (Annualized) ---")
    print(f"Strategy   -> Return: {strat_ret*100:.2f}%, Volatility: {strat_vol*100:.2f}%, Sharpe: {strat_sharpe:.2f}")
    print(f"B&H (NVDA) -> Return: {bh_ret*100:.2f}%, Volatility: {bh_vol*100:.2f}%, Sharpe: {bh_sharpe:.2f}")
    print(f"B&H (SPY)  -> Return: {spy_ret*100:.2f}%, Volatility: {spy_vol*100:.2f}%, Sharpe: {spy_sharpe:.2f}")

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(df_bt.index, df_bt['Cum_Strategy'], label='Buy The Dip Strategy', color='blue')
    plt.plot(df_bt.index, df_bt['Cum_BuyHold'], label='Buy & Hold (NVDA)', color='green')
    plt.plot(df_bt.index, df_bt['Cum_SPY'], label='Buy & Hold (SPY)', color='orange', alpha=0.7)
    plt.title('Out-of-Sample Cumulative Portfolio Value')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)
    plt.savefig('performance_chart.png')
    print("\nSaved chart to performance_chart.png")
