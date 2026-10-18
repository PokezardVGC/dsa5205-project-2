import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data import fetch_data
from src.features import engineer_features
from src.model import train_and_select_model
from src.backtest import backtest_strategy

def calculate_metrics(returns):
    ann_return = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol != 0 else 0
    return ann_return, ann_vol, sharpe

def plot_hero_cumulative_underwater(df_bt):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
    
    # Cumulative Returns (Log Scale)
    ax1.plot(df_bt.index, df_bt['Cum_Strategy'], label='Strategy (Buy The Dip)', color='#1f77b4', linewidth=2)
    ax1.plot(df_bt.index, df_bt['Cum_BuyHold'], label='NVDA Buy & Hold', color='#2ca02c', alpha=0.7)
    ax1.plot(df_bt.index, df_bt['Cum_SPY'], label='SPY Benchmark', color='#7f7f7f', alpha=0.7)
    ax1.set_yscale('log')
    ax1.set_title('Out-of-Sample Cumulative Returns (Log Scale)')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend()
    
    # Underwater Plot (Drawdown)
    roll_max_strat = df_bt['Cum_Strategy'].cummax()
    dd_strat = (df_bt['Cum_Strategy'] - roll_max_strat) / roll_max_strat
    
    roll_max_spy = df_bt['Cum_SPY'].cummax()
    dd_spy = (df_bt['Cum_SPY'] - roll_max_spy) / roll_max_spy
    
    ax2.fill_between(df_bt.index, dd_strat, 0, color='#1f77b4', alpha=0.3, label='Strategy Drawdown')
    ax2.plot(df_bt.index, dd_spy, color='#7f7f7f', alpha=0.5, linewidth=1, label='SPY Drawdown')
    ax2.set_title('Drawdown Profile (Underwater Chart)')
    ax2.set_ylabel('Drawdown (%)')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('perf_hero.png', dpi=300)
    print("Saved perf_hero.png")

def plot_rolling_metrics(df_bt):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Rolling 126-day (6-month) Sharpe Ratio
    window = 126
    roll_ret = df_bt['Strategy_Return'].rolling(window).mean() * 252
    roll_vol = df_bt['Strategy_Return'].rolling(window).std() * np.sqrt(252)
    roll_sharpe = roll_ret / roll_vol
    
    spy_roll_ret = df_bt['SPY_Return'].rolling(window).mean() * 252
    spy_roll_vol = df_bt['SPY_Return'].rolling(window).std() * np.sqrt(252)
    spy_roll_sharpe = spy_roll_ret / spy_roll_vol
    
    ax1.plot(df_bt.index, roll_sharpe, label='Strategy 6M Rolling Sharpe', color='purple')
    ax1.plot(df_bt.index, spy_roll_sharpe, label='SPY 6M Rolling Sharpe', color='grey', alpha=0.5)
    ax1.axhline(0, color='black', linestyle='--')
    ax1.set_title('Rolling 6-Month Sharpe Ratio')
    ax1.set_ylabel('Sharpe Ratio')
    ax1.legend()
    
    # Rolling Beta to SPY
    cov = df_bt['Strategy_Return'].rolling(window).cov(df_bt['SPY_Return'])
    var = df_bt['SPY_Return'].rolling(window).var()
    roll_beta = cov / var
    
    ax2.plot(df_bt.index, roll_beta, label='Strategy 6M Rolling Beta', color='darkorange')
    ax2.axhline(1, color='red', linestyle='--', label='Beta = 1 (Market Neutrality Line)')
    ax2.axhline(0, color='black', linestyle='-')
    ax2.set_title('Rolling 6-Month Beta to Benchmark')
    ax2.set_ylabel('Beta')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('perf_rolling_risk.png', dpi=300)
    print("Saved perf_rolling_risk.png")

def plot_trade_analytics(df_bt):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Only look at days where the strategy actually took a position
    active_days = df_bt[df_bt['Position'] == 1]['Strategy_Return']
    
    sns.histplot(active_days, kde=True, bins=50, color='teal', alpha=0.6, ax=ax)
    ax.axvline(0, color='black', linestyle='--')
    ax.axvline(active_days.mean(), color='red', linestyle='-', label=f'Mean Return: {active_days.mean()*100:.2f}%')
    
    win_rate = (active_days > 0).mean() * 100
    
    stat_text = f"Win Rate: {win_rate:.1f}%\nTotal Active Days: {len(active_days)}"
    ax.text(0.05, 0.95, stat_text, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_title('Distribution of Returns on Active Trading Days')
    ax.set_xlabel('Daily Return')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('perf_trade_dist.png', dpi=300)
    print("Saved perf_trade_dist.png")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    
    df_raw = fetch_data('NVDA', 'SPY', '2015-01-01', '2024-01-01')
    df_features, feature_names = engineer_features(df_raw)
    best_model, predictions, test_index = train_and_select_model(df_features, feature_names)
    df_bt = backtest_strategy(df_features, test_index, predictions)
    
    plot_hero_cumulative_underwater(df_bt)
    plot_rolling_metrics(df_bt)
    plot_trade_analytics(df_bt)
