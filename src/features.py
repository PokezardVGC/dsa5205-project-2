import pandas as pd
import numpy as np

def engineer_features(df):
    """
    Creates predictive features (Buy the Dip signals) and target labels.
    """
    print("Engineering features...")
    
    # Daily returns (Today's close vs Yesterday's close)
    df['Return'] = df['Price'].pct_change()

    # Past returns (Today's, Yesterday's, etc.)
    # Note: df['Return'] is already known at today's close
    df['Ret_0'] = df['Return']
    df['Ret_Lag1'] = df['Return'].shift(1)
    df['Ret_Lag2'] = df['Return'].shift(2)
    df['Ret_Lag5'] = df['Price'].pct_change(5).shift(1) # Return over the 5 days up to yesterday

    # Moving Average Distances (How far is the price from its recent average?)
    df['MA_10'] = df['Price'].rolling(window=10).mean()
    df['Dist_MA10'] = (df['Price'] - df['MA_10']) / df['MA_10']

    df['MA_20'] = df['Price'].rolling(window=20).mean()
    df['Dist_MA20'] = (df['Price'] - df['MA_20']) / df['MA_20']

    # Label Creation: We want to predict tomorrow's return
    df['Target_Next_Return'] = df['Return'].shift(-1)

    # Drop NaN and infinite values created by lagging and rolling windows
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    # Clip extreme outlier returns to ensure numerical stability during model training
    features = ['Ret_0', 'Ret_Lag1', 'Ret_Lag2', 'Ret_Lag5', 'Dist_MA10', 'Dist_MA20']
    df.loc[:, features] = df[features].clip(-0.5, 0.5)

    # Ensure no infinite values are left before proceeding
    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(axis=1)]
    
    return df, features
