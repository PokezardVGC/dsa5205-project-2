"""Extension 2: Moving Average and Momentum Strategy (Marco)
Compares Rule-Based Macro Trend (SMA 200) vs. Machine Learning Momentum (Ridge L2 & Lasso L1)
on NVDA vs. SPY (2015-2023) with 10 bps transaction costs.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler

# Set reproducibility seed
np.random.seed(42)

def main():
    print("=" * 70)
    print("DSA5205 Project 2 — Extension 2: Macro Trend & Momentum Strategy")
    print("=" * 70)
    
    # 1. Ingest Data
    print("\n[1/5] Fetching daily price data for NVDA and SPY (2015–2023)...")
    raw_data = yf.download(['NVDA', 'SPY'], start='2015-01-01', end='2023-12-31', progress=False)
    
    if isinstance(raw_data.columns, pd.MultiIndex):
        if 'Close' in raw_data.columns.levels[0]:
            df = raw_data['Close'].copy()
        else:
            df = raw_data.xs('Close', axis=1, level=0).copy()
    else:
        df = raw_data[['NVDA', 'SPY']].copy()
        
    df = df.rename(columns={'NVDA': 'Price'}).dropna()
    print(f"Loaded {len(df)} daily bars from {df.index[0].date()} to {df.index[-1].date()}")
    
    # 2. Features & Target
    print("\n[2/5] Engineering momentum features and forward target...")
    df['Return'] = df['Price'].pct_change()
    df['Target_Next_Return'] = df['Return'].shift(-1)
    
    df['SMA_50'] = df['Price'].rolling(50).mean()
    df['SMA_200'] = df['Price'].rolling(200).mean()
    df['Dist_SMA50'] = (df['Price'] - df['SMA_50']) / df['SMA_50']
    df['Dist_SMA200'] = (df['Price'] - df['SMA_200']) / df['SMA_200']
    df['Mom_10d'] = df['Price'].pct_change(10)
    df['Mom_20d'] = df['Price'].pct_change(20)
    
    df = df.dropna().copy()
    features = ['Dist_SMA50', 'Dist_SMA200', 'Mom_10d', 'Mom_20d']
    print(f"Active sample size after warm-up: {len(df)} bars")
    
    # 3. Chronological 80/20 Split
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    TRANSACTION_COST = 0.0010  # 10 bps
    INITIAL_CAPITAL = 100000.0
    print(f"In-sample bars: {len(train_df)} | Out-of-sample bars: {len(test_df)}")
    print(f"Out-of-sample test window: {test_df.index[0].date()} to {test_df.index[-1].date()}")
    
    # 4. Strategy Backtests
    print("\n[3/5] Running backtests across Extension 2a, 2b, and 2c...")
    
    # Ext 2a: Rule-based Trend (SMA 200)
    test_df['Bull_Regime'] = test_df['Price'] > test_df['SMA_200']
    test_df['Pos_2a'] = np.where(test_df['Bull_Regime'], 1.0, 0.0)
    test_df['Chg_2a'] = test_df['Pos_2a'].diff().fillna(test_df['Pos_2a'].iloc[0]).abs()
    test_df['Ret_2a'] = (test_df['Pos_2a'] * test_df['Target_Next_Return']) - (test_df['Chg_2a'] * TRANSACTION_COST)
    test_df['Cum_2a'] = INITIAL_CAPITAL * (1 + test_df['Ret_2a']).cumprod()
    
    # Ext 2b (Ridge) and 2c (Lasso)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[features])
    y_train = train_df['Target_Next_Return'].clip(-0.5, 0.5)
    X_test = scaler.transform(test_df[features])
    
    ridge_model = Ridge(alpha=10.0).fit(X_train, y_train)
    test_df['Pred_Ridge'] = ridge_model.predict(X_test)
    test_df['Pos_2b'] = np.where(test_df['Pred_Ridge'] > y_train.mean(), 1.0, 0.0)
    test_df['Chg_2b'] = test_df['Pos_2b'].diff().fillna(test_df['Pos_2b'].iloc[0]).abs()
    test_df['Ret_2b'] = (test_df['Pos_2b'] * test_df['Target_Next_Return']) - (test_df['Chg_2b'] * TRANSACTION_COST)
    test_df['Cum_2b'] = INITIAL_CAPITAL * (1 + test_df['Ret_2b']).cumprod()
    
    lasso_model = Lasso(alpha=0.00001, random_state=42, max_iter=10000).fit(X_train, y_train)
    test_df['Pred_Lasso'] = lasso_model.predict(X_test)
    test_df['Pos_2c'] = np.where(test_df['Pred_Lasso'] > y_train.mean(), 1.0, 0.0)
    test_df['Chg_2c'] = test_df['Pos_2c'].diff().fillna(test_df['Pos_2c'].iloc[0]).abs()
    test_df['Ret_2c'] = (test_df['Pos_2c'] * test_df['Target_Next_Return']) - (test_df['Chg_2c'] * TRANSACTION_COST)
    test_df['Cum_2c'] = INITIAL_CAPITAL * (1 + test_df['Ret_2c']).cumprod()
    
    # Benchmarks
    test_df['Cum_BuyHold'] = INITIAL_CAPITAL * (1 + test_df['Target_Next_Return']).cumprod()
    test_df['Cum_SPY'] = INITIAL_CAPITAL * (1 + test_df['SPY'].pct_change().shift(-1)).cumprod()
    
    # 5. Performance Metrics
    print("\n[4/5] Computing out-of-sample risk and return metrics...")
    def calc_metrics(returns, cum_series):
        ann_ret = returns.mean() * 252
        ann_vol = returns.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        dd = (cum_series - cum_series.cummax()) / cum_series.cummax()
        max_dd = dd.min()
        return ann_ret, ann_vol, sharpe, max_dd

    ret_2a, vol_2a, sr_2a, dd_2a = calc_metrics(test_df['Ret_2a'], test_df['Cum_2a'])
    ret_2b, vol_2b, sr_2b, dd_2b = calc_metrics(test_df['Ret_2b'], test_df['Cum_2b'])
    ret_2c, vol_2c, sr_2c, dd_2c = calc_metrics(test_df['Ret_2c'], test_df['Cum_2c'])
    ret_bh, vol_bh, sr_bh, dd_bh = calc_metrics(test_df['Target_Next_Return'], test_df['Cum_BuyHold'])
    ret_spy, vol_spy, sr_spy, dd_spy = calc_metrics(test_df['SPY'].pct_change().shift(-1), test_df['Cum_SPY'])

    summary_table = pd.DataFrame({
        'Strategy / Model': [
            'Ext 2a: Rule-Based Trend (SMA 200)',
            'Ext 2c: ML Momentum (Lasso L1)',
            'Ext 2b: ML Momentum (Ridge L2)',
            'NVDA Buy & Hold (Asset Benchmark)',
            'SPY Buy & Hold (Market Benchmark)'
        ],
        'Annualized Return': [f"{ret_2a*100:.2f}%", f"{ret_2c*100:.2f}%", f"{ret_2b*100:.2f}%", f"{ret_bh*100:.2f}%", f"{ret_spy*100:.2f}%"],
        'Annualized Volatility': [f"{vol_2a*100:.2f}%", f"{vol_2c*100:.2f}%", f"{vol_2b*100:.2f}%", f"{vol_bh*100:.2f}%", f"{vol_spy*100:.2f}%"],
        'Sharpe Ratio': [f"{sr_2a:.2f}", f"{sr_2c:.2f}", f"{sr_2b:.2f}", f"{sr_bh:.2f}", f"{sr_spy:.2f}"],
        'Max Drawdown': [f"{dd_2a*100:.2f}%", f"{dd_2c*100:.2f}%", f"{dd_2b*100:.2f}%", f"{dd_bh*100:.2f}%", f"{dd_spy*100:.2f}%"]
    })

    print("\n" + "=" * 80)
    print(summary_table.to_string(index=False))
    print("=" * 80)
    
    # 6. Render Visualizations
    print("\n[5/5] Generating standalone visualization figures...")
    os.makedirs('figures', exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid')
    
    df_full = df.copy()
    df_full['Bull_Regime'] = df_full['Price'] > df_full['SMA_200']
    df_full['Pos_2a'] = np.where(df_full['Bull_Regime'], 1.0, 0.0)
    df_full['Chg_2a'] = df_full['Pos_2a'].diff().fillna(df_full['Pos_2a'].iloc[0]).abs()
    df_full['Ret_2a'] = (df_full['Pos_2a'] * df_full['Target_Next_Return']) - (df_full['Chg_2a'] * 0.0010)

    df_vis = df_full.loc['2021-01-01':'2023-12-28'].copy()
    df_vis['Cum_2a'] = 100000 * (1 + df_vis['Ret_2a']).cumprod()
    df_vis['Cum_BuyHold'] = 100000 * (1 + df_vis['Target_Next_Return']).cumprod()
    df_vis['Cum_SPY'] = 100000 * (1 + df_vis['SPY'].pct_change().shift(-1)).cumprod()
    df_vis['DD_2a'] = (df_vis['Cum_2a'] - df_vis['Cum_2a'].cummax()) / df_vis['Cum_2a'].cummax()
    df_vis['DD_BuyHold'] = (df_vis['Cum_BuyHold'] - df_vis['Cum_BuyHold'].cummax()) / df_vis['Cum_BuyHold'].cummax()

    split_date = test_df.index[0]

    # Chart 1
    plt.figure(figsize=(12, 5))
    plt.plot(df_vis.index, df_vis['Price'], label='NVDA Price', color='#2ca02c', linewidth=2.0)
    plt.plot(df_vis.index, df_vis['SMA_200'], label='200-Day Macro Trend (SMA 200)', color='#d62728', linewidth=2.0, linestyle='--')
    plt.fill_between(df_vis.index, df_vis['Price'].min(), df_vis['Price'].max(), 
                     where=(df_vis['Pos_2a'] == 0), color='#7f7f7f', alpha=0.15, label='Ext 2a Bear Regime (100% Cash)')
    plt.fill_between(df_vis.index, df_vis['Price'].min(), df_vis['Price'].max(), 
                     where=(df_vis['Pos_2a'] == 1), color='#2ca02c', alpha=0.10, label='Ext 2a Bull Regime (100% Long NVDA)')
    plt.axvline(split_date, color='purple', linestyle=':', linewidth=2, label=f'Test Split Start ({split_date.date()})')
    
    sell_date = pd.Timestamp('2022-04-07')
    sell_price = df_vis.loc[sell_date, 'Price']
    plt.annotate(f'Ext 2a SELL Signal (Exit to Cash)\n2022-04-07 @ ${sell_price:.2f}', 
                 xy=(sell_date, sell_price), xytext=(sell_date - pd.Timedelta(days=160), sell_price + 10),
                 arrowprops=dict(facecolor='red', shrink=0.08, width=2, headwidth=7),
                 fontsize=9.5, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='red', lw=1.5))

    buy_date = pd.Timestamp('2023-01-13')
    buy_price = df_vis.loc[buy_date, 'Price']
    plt.annotate(f'Ext 2a BUY Signal (Reclaim Trend)\n2023-01-13 @ ${buy_price:.2f}', 
                 xy=(buy_date, buy_price), xytext=(buy_date - pd.Timedelta(days=140), buy_price - 14),
                 arrowprops=dict(facecolor='green', shrink=0.08, width=2, headwidth=7),
                 fontsize=9.5, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='green', lw=1.5))

    plt.ylabel('Price ($)', fontsize=11)
    plt.xlabel('Date', fontsize=11)
    plt.title('Figure 1: Extension 2a Macro Trend Identification — Price vs. 200-Day SMA (2021–2023)', fontsize=12, fontweight='bold')
    plt.legend(loc='upper left', frameon=True)
    plt.tight_layout()
    plt.savefig('figures/Ext2_chart1_macro_regime_and_signals.png', dpi=300)
    plt.close()

    # Chart 2
    plt.figure(figsize=(12, 5))
    plt.plot(test_df.index, test_df['Cum_2a'], label='Ext 2a: Rule Trend (SMA 200)', color='#1f77b4', linewidth=2.5)
    plt.plot(test_df.index, test_df['Cum_2c'], label='Ext 2c: ML Momentum (Lasso L1)', color='#ff7f0e', linewidth=2.0, linestyle='-')
    plt.plot(test_df.index, test_df['Cum_2b'], label='Ext 2b: ML Momentum (Ridge L2)', color='#9467bd', linewidth=2.0, linestyle='-.')
    plt.plot(test_df.index, test_df['Cum_BuyHold'], label='NVDA Buy & Hold Benchmark', color='#2ca02c', alpha=0.6, linestyle='--')
    plt.plot(test_df.index, test_df['Cum_SPY'], label='SPY Market Benchmark', color='#7f7f7f', alpha=0.8)
    plt.yscale('log')
    plt.ylabel('Portfolio Value ($)', fontsize=11)
    plt.xlabel('Date', fontsize=11)
    plt.title('Figure 2: Out-of-Sample Cumulative Wealth Trajectory ($100k Base, Log Scale)', fontsize=12, fontweight='bold')
    plt.legend(loc='upper left', frameon=True)
    plt.tight_layout()
    plt.savefig('figures/Ext2_chart2_cumulative_wealth.png', dpi=300)
    plt.close()

    # Chart 3
    plt.figure(figsize=(12, 5))
    plt.fill_between(df_vis.index, df_vis['DD_2a'] * 100, 0, color='#1f77b4', alpha=0.35, label='Trend Strategy Drawdown')
    plt.plot(df_vis.index, df_vis['DD_BuyHold'] * 100, color='#d62728', alpha=0.6, linestyle=':', label='NVDA Buy & Hold Drawdown (-66.3% Peak Crash)')
    plt.axvline(split_date, color='purple', linestyle=':', linewidth=2, label=f'Test Split Start ({split_date.date()})')
    plt.axhline(-20, color='black', linestyle='--', alpha=0.4, label='Bear Market Threshold (-20%)')
    plt.ylabel('Drawdown (%)', fontsize=11)
    plt.xlabel('Date', fontsize=11)
    plt.title('Figure 3: Underwater Drawdown Profile Across the Full 2021–2023 Cycle', fontsize=12, fontweight='bold')
    plt.legend(loc='lower left', frameon=True)
    plt.tight_layout()
    plt.savefig('figures/Ext2_chart3_underwater_drawdown.png', dpi=300)
    plt.close()

    print("Execution complete. Output figures saved in figures/ directory.")

if __name__ == '__main__':
    main()
