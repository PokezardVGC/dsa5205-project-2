import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import fetch_data

def hurst_exponent(ts, max_lag=20):
    lags = range(2, max_lag)
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0]*2.0

def generate_split_eda_charts():
    df = fetch_data('NVDA', 'SPY', '2015-01-01', '2024-01-01')
    
    # Pre-calc
    df['SMA_50'] = df['Price'].rolling(50).mean()
    df['SMA_200'] = df['Price'].rolling(200).mean()
    df['Z_Score_50'] = (df['Price'] - df['SMA_50']) / df['Price'].rolling(50).std()
    df['Hurst_60'] = df['Price'].rolling(window=60).apply(lambda x: hurst_exponent(x.values), raw=False)
    
    roll_max = df['Price'].cummax()
    df['Drawdown'] = (df['Price'] - roll_max) / roll_max
    
    df['Ret'] = df['Price'].pct_change()
    df['SPY_Ret'] = df['SPY'].pct_change()
    df['Corr_60'] = df['Ret'].rolling(60).corr(df['SPY_Ret'])
    df['Vol_60'] = df['Ret'].rolling(60).std() * np.sqrt(252)

    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. Price
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df['Price'], label='NVDA Price', color='#2ca02c')
    plt.plot(df.index, df['SMA_50'], label='50-day SMA', color='orange', alpha=0.7)
    plt.plot(df.index, df['SMA_200'], label='200-day SMA', color='red', alpha=0.7)
    plt.yscale('log')
    plt.ylabel('Price (Log Scale)')
    plt.title('Figure 1: NVDA Price and Moving Averages')
    plt.legend()
    plt.tight_layout()
    plt.savefig('figures/eda_1_price.png', dpi=300)
    plt.close()

    # 2. Z-Score
    plt.figure(figsize=(10, 4))
    plt.plot(df.index, df['Z_Score_50'], color='purple', label='50-day Z-Score')
    plt.axhline(0, color='black', linestyle='--')
    plt.axhline(-2, color='red', linestyle='--', alpha=0.5, label='Dip Threshold (-2σ)')
    plt.fill_between(df.index, df['Z_Score_50'], -2, where=(df['Z_Score_50'] < -2), color='red', alpha=0.3)
    plt.ylabel('Z-Score')
    plt.title('Figure 2: Rolling 50-day Z-Score (Dip Signal)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('figures/eda_2_zscore.png', dpi=300)
    plt.close()

    # 3. Hurst
    plt.figure(figsize=(10, 4))
    plt.plot(df.index, df['Hurst_60'], color='teal', label='60-day Rolling Hurst Exponent')
    plt.axhline(0.5, color='red', linestyle='--')
    plt.fill_between(df.index, df['Hurst_60'], 0.5, where=(df['Hurst_60'] < 0.5), color='green', alpha=0.3, label='Mean Reverting Regime')
    plt.ylabel('Hurst Exponent')
    plt.title('Figure 3: Rolling Hurst Exponent')
    plt.legend()
    plt.tight_layout()
    plt.savefig('figures/eda_3_hurst.png', dpi=300)
    plt.close()

    # 4. Underwater Drawdown
    plt.figure(figsize=(10, 4))
    plt.plot(df.index, df['Drawdown'], color='red')
    plt.fill_between(df.index, df['Drawdown'], 0, color='red', alpha=0.3)
    plt.axhline(-0.10, color='orange', linestyle='--', alpha=0.7, label='-10% Correction')
    plt.axhline(-0.20, color='black', linestyle='--', alpha=0.7, label='-20% Bear Market')
    plt.ylabel('Drawdown (%)')
    plt.title('Figure 4: NVDA Drawdown Profile')
    plt.legend()
    plt.tight_layout()
    plt.savefig('figures/eda_4_drawdown.png', dpi=300)
    plt.close()

    # 5. Correlation
    plt.figure(figsize=(10, 4))
    plt.plot(df.index, df['Corr_60'], color='purple', label='60-day Rolling Correlation')
    plt.axhline(df['Corr_60'].mean(), color='black', linestyle='--', label='Mean Correlation')
    plt.ylabel('Correlation')
    plt.title('Figure 5: Rolling Correlation with SPY')
    plt.legend()
    plt.tight_layout()
    plt.savefig('figures/eda_5_corr.png', dpi=300)
    plt.close()

    # 6. Volatility
    plt.figure(figsize=(10, 4))
    plt.plot(df.index, df['Vol_60'], color='darkorange', label='60-day Volatility')
    plt.ylabel('Annualized Volatility')
    plt.title('Figure 6: Rolling Annualized Volatility')
    plt.legend()
    plt.tight_layout()
    plt.savefig('figures/eda_6_vol.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    generate_split_eda_charts()
