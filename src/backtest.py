import numpy as np

def backtest_strategy(df, test_index, predictions, initial_capital=100000, transaction_cost=0.0010):
    """
    Simulates trading based on model predictions.
    Transaction costs set to 10 bps to account for realistic slippage.
    """
    print("\nRunning Backtest on Out-of-Sample (Test) Data...")

    # Create a backtest dataframe specifically for the test period
    df_bt = df.loc[test_index].copy()
    df_bt['Predicted_Return'] = predictions

    # Trading Rule:
    # Strategy executes Market-On-Close (MOC) orders based on signals generated
    # right before close. Position is long (1) when predicted return is positive, else flat (0).
    df_bt['Position'] = np.where(df_bt['Predicted_Return'] > 0, 1, 0)
    
    # Calculate position changes to apply transaction costs
    # First row is entry if Position == 1, so fillna(0) on diff is correct if initial position is 0
    df_bt['Position_Change'] = df_bt['Position'].diff().fillna(df_bt['Position'].iloc[0]).abs()

    # Calculate returns
    # Strategy Return = Position * Actual_Next_Return - Transaction_Costs
    df_bt['Strategy_Return'] = (df_bt['Position'] * df_bt['Target_Next_Return']) - (df_bt['Position_Change'] * transaction_cost)

    # Buy and hold benchmark return
    df_bt['BuyHold_Return'] = df_bt['Target_Next_Return']

    # Calculate cumulative returns
    df_bt['Cum_Strategy'] = initial_capital * (1 + df_bt['Strategy_Return']).cumprod()
    df_bt['Cum_BuyHold'] = initial_capital * (1 + df_bt['BuyHold_Return']).cumprod()
    
    # Benchmark SPY next-period return aligned with execution horizon
    df_bt['SPY_Return'] = df_bt['SPY'].pct_change().shift(-1)
    df_bt['Cum_SPY'] = initial_capital * (1 + df_bt['SPY_Return']).cumprod()
    # Drop the last row since shift(-1) creates a NaN
    df_bt = df_bt.dropna()

    return df_bt
