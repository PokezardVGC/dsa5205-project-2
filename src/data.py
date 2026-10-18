import yfinance as yf
import pandas as pd
import numpy as np

def fetch_data(asset, benchmark, start_date, end_date):
    """
    Fetches adjusted close prices for the target asset and benchmark.
    yfinance auto_adjust=True handles splits and dividends.
    """
    print(f"Fetching data for {asset} and {benchmark}...")
    df_asset = yf.download(asset, start=start_date, end=end_date, auto_adjust=True, progress=False)
    df_bench = yf.download(benchmark, start=start_date, end=end_date, auto_adjust=True, progress=False)

    df = pd.DataFrame()
    df['Price'] = df_asset['Close']
    df['SPY'] = df_bench['Close']
    
    # Drop rows where either asset or benchmark is missing
    df = df.dropna()
    
    return df
