import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from src.data import fetch_data
from src.features import engineer_features
from src.model import train_and_select_model
from src.backtest import backtest_strategy

def generate_split_perf_charts():
    df_raw = fetch_data('NVDA', 'SPY', '2015-01-01', '2024-01-01')
    df_features, feature_names = engineer_features(df_raw)
    best_model, predictions, test_index = train_and_select_model(df_features, feature_names)
    df_bt = backtest_strategy(df_features, test_index, predictions)
    
    plt.style.use('seaborn-v0_8-whitegrid')

    # 1. Cumulative
    plt.figure(figsize=(10, 5))
    plt.plot(df_bt.index, df_bt['Cum_Strategy'], label='Strategy (Buy The Dip)', color='#1f77b4', linewidth=2)
    plt.plot(df_bt.index, df_bt['Cum_BuyHold'], label='NVDA Buy & Hold', color='#2ca02c', alpha=0.7)
    plt.plot(df_bt.index, df_bt['Cum_SPY'], label='SPY Benchmark', color='#7f7f7f', alpha=0.7)
    plt.yscale('log')
    plt.title('Figure 8: Out-of-Sample Cumulative Returns (Log Scale)')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('perf_1_cumulative.png', dpi=300)
    plt.close()

    # 2. Underwater Strategy
    roll_max_strat = df_bt['Cum_Strategy'].cummax()
    dd_strat = (df_bt['Cum_Strategy'] - roll_max_strat) / roll_max_strat
    roll_max_spy = df_bt['Cum_SPY'].cummax()
    dd_spy = (df_bt['Cum_SPY'] - roll_max_spy) / roll_max_spy

    plt.figure(figsize=(10, 4))
    plt.fill_between(df_bt.index, dd_strat, 0, color='#1f77b4', alpha=0.3, label='Strategy Drawdown')
    plt.plot(df_bt.index, dd_spy, color='#7f7f7f', alpha=0.5, linewidth=1, label='SPY Drawdown')
    plt.title('Figure 9: Strategy vs Benchmark Drawdown Profile')
    plt.ylabel('Drawdown (%)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('perf_2_underwater.png', dpi=300)
    plt.close()

    # 3. Rolling Sharpe
    window = 126
    roll_ret = df_bt['Strategy_Return'].rolling(window).mean() * 252
    roll_vol = df_bt['Strategy_Return'].rolling(window).std() * np.sqrt(252)
    roll_sharpe = roll_ret / roll_vol
    spy_roll_ret = df_bt['SPY_Return'].rolling(window).mean() * 252
    spy_roll_vol = df_bt['SPY_Return'].rolling(window).std() * np.sqrt(252)
    spy_roll_sharpe = spy_roll_ret / spy_roll_vol

    plt.figure(figsize=(10, 4))
    plt.plot(df_bt.index, roll_sharpe, label='Strategy 6M Rolling Sharpe', color='purple')
    plt.plot(df_bt.index, spy_roll_sharpe, label='SPY 6M Rolling Sharpe', color='grey', alpha=0.5)
    plt.axhline(0, color='black', linestyle='--')
    plt.title('Figure 10: Rolling 6-Month Sharpe Ratio')
    plt.ylabel('Sharpe Ratio')
    plt.legend()
    plt.tight_layout()
    plt.savefig('perf_3_sharpe.png', dpi=300)
    plt.close()

    # 4. Rolling Beta
    cov = df_bt['Strategy_Return'].rolling(window).cov(df_bt['SPY_Return'])
    var = df_bt['SPY_Return'].rolling(window).var()
    roll_beta = cov / var

    plt.figure(figsize=(10, 4))
    plt.plot(df_bt.index, roll_beta, label='Strategy 6M Rolling Beta', color='darkorange')
    plt.axhline(1, color='red', linestyle='--', label='Beta = 1')
    plt.axhline(0, color='black', linestyle='-')
    plt.title('Figure 11: Rolling 6-Month Beta to SPY')
    plt.ylabel('Beta')
    plt.legend()
    plt.tight_layout()
    plt.savefig('perf_4_beta.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    generate_split_perf_charts()
