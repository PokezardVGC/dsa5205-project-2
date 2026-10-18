import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from scipy import stats

def fetch_data(asset, benchmark, start_date, end_date):
    df_asset = yf.download(asset, start=start_date, end=end_date, auto_adjust=True, progress=False)
    df_bench = yf.download(benchmark, start=start_date, end=end_date, auto_adjust=True, progress=False)
    df = pd.DataFrame()
    df['Price'] = df_asset['Close']
    df['SPY'] = df_bench['Close']
    df = df.dropna()
    return df

def hurst_exponent(ts, max_lag=20):
    lags = range(2, max_lag)
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0]*2.0

def plot_price_and_regimes(df, asset='NVDA'):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [3, 1, 1]})
    
    # 1. Price, SMA, and Drawdowns
    df['SMA_50'] = df['Price'].rolling(50).mean()
    df['SMA_200'] = df['Price'].rolling(200).mean()
    
    ax1.plot(df.index, df['Price'], label=f'{asset} Price', color='#2ca02c')
    ax1.plot(df.index, df['SMA_50'], label='50-day SMA', color='orange', alpha=0.7)
    ax1.plot(df.index, df['SMA_200'], label='200-day SMA', color='red', alpha=0.7)
    ax1.set_yscale('log')
    ax1.set_ylabel('Price (Log Scale)')
    ax1.set_title(f'{asset} Price, Moving Averages, and Volatility Regimes')
    ax1.legend()

    # 2. Z-Score (The Dip Signal)
    df['Z_Score_50'] = (df['Price'] - df['SMA_50']) / df['Price'].rolling(50).std()
    ax2.plot(df.index, df['Z_Score_50'], color='purple', label='50-day Z-Score')
    ax2.axhline(0, color='black', linestyle='--')
    ax2.axhline(-2, color='red', linestyle='--', alpha=0.5, label='Dip Threshold (-2σ)')
    ax2.fill_between(df.index, df['Z_Score_50'], -2, where=(df['Z_Score_50'] < -2), color='red', alpha=0.3)
    ax2.set_ylabel('Z-Score')
    ax2.legend()
    
    # 3. Rolling Hurst Exponent
    df['Hurst_60'] = df['Price'].rolling(window=60).apply(lambda x: hurst_exponent(x.values), raw=False)
    ax3.plot(df.index, df['Hurst_60'], color='teal', label='60-day Rolling Hurst Exponent')
    ax3.axhline(0.5, color='red', linestyle='--')
    ax3.fill_between(df.index, df['Hurst_60'], 0.5, where=(df['Hurst_60'] < 0.5), color='green', alpha=0.3, label='Mean Reverting Regime (H<0.5)')
    ax3.set_ylabel('Hurst Exponent')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig('eda_price_improved.png', dpi=300)
    print("Saved eda_price_improved.png")

def plot_return_distributions(df, asset='NVDA'):
    plt.style.use('seaborn-v0_8-whitegrid')
    df['Ret'] = df['Price'].pct_change().dropna()
    df['SPY_Ret'] = df['SPY'].pct_change().dropna()
    df = df.dropna()

    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.histplot(df['Ret'], kde=True, stat='density', color='#2ca02c', alpha=0.5, label=f'{asset} Returns', ax=ax, bins=100)
    sns.histplot(df['SPY_Ret'], kde=True, stat='density', color='#1f77b4', alpha=0.5, label='SPY Returns', ax=ax, bins=100)
    
    # Stats for NVDA
    mean, std = df['Ret'].mean(), df['Ret'].std()
    skew, kurt = stats.skew(df['Ret']), stats.kurtosis(df['Ret'])
    
    ax.axvline(mean - 2*std, color='red', linestyle='--', label=f'{asset} -2σ Dip')
    
    stat_text = f"{asset} Stats:\nMean: {mean*100:.2f}%\nVol (Ann): {std*np.sqrt(252)*100:.2f}%\nSkewness: {skew:.2f}\nKurtosis (Excess): {kurt:.2f}"
    ax.text(0.05, 0.95, stat_text, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_title(f'Daily Return Distribution: {asset} vs SPY (Tail Risk & Leptokurtosis)')
    ax.set_xlabel('Daily Return')
    ax.legend()
    plt.savefig('eda_dist_improved.png', dpi=300)
    print("Saved eda_dist_improved.png")

def plot_drawdowns_and_risk(df, asset='NVDA'):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [2, 1, 1]})
    
    # 1. Underwater / Drawdown Plot
    roll_max = df['Price'].cummax()
    df['Drawdown'] = (df['Price'] - roll_max) / roll_max
    
    ax1.plot(df.index, df['Drawdown'], color='red')
    ax1.fill_between(df.index, df['Drawdown'], 0, color='red', alpha=0.3)
    ax1.axhline(-0.10, color='orange', linestyle='--', alpha=0.7, label='-10% Correction')
    ax1.axhline(-0.20, color='black', linestyle='--', alpha=0.7, label='-20% Bear Market')
    ax1.set_ylabel('Drawdown (%)')
    ax1.set_title(f'{asset} Drawdown Profile and Risk Metrics')
    ax1.legend()
    
    # 2. Rolling Correlation
    df['Ret'] = df['Price'].pct_change()
    df['SPY_Ret'] = df['SPY'].pct_change()
    df['Corr_60'] = df['Ret'].rolling(60).corr(df['SPY_Ret'])
    
    ax2.plot(df.index, df['Corr_60'], color='purple')
    ax2.axhline(df['Corr_60'].mean(), color='black', linestyle='--', label='Mean Correlation')
    ax2.set_ylabel('60-day Rolling Correlation')
    ax2.legend()
    
    # 3. Rolling Volatility
    df['Vol_60'] = df['Ret'].rolling(60).std() * np.sqrt(252)
    ax3.plot(df.index, df['Vol_60'], color='darkorange')
    ax3.set_ylabel('Annualized 60-day Volatility')
    
    plt.tight_layout()
    plt.savefig('eda_drawdown_improved.png', dpi=300)
    print("Saved eda_drawdown_improved.png")

if __name__ == "__main__":
    df = fetch_data('NVDA', 'SPY', '2015-01-01', '2024-01-01')
    plot_price_and_regimes(df)
    plot_return_distributions(df)
    plot_drawdowns_and_risk(df)

