import warnings
from src.data import fetch_data
from src.features import engineer_features
from src.model import train_and_select_model
from src.backtest import backtest_strategy
from src.evaluate import evaluate_and_plot

def main():
    # Suppress runtime warnings from division by zero in numpy
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    ASSET = 'NVDA'
    BENCHMARK = 'SPY'
    START_DATE = '2015-01-01'
    END_DATE = '2024-01-01'

    # 1. Fetch Data
    df_raw = fetch_data(ASSET, BENCHMARK, START_DATE, END_DATE)

    # 2. Engineer Features
    df_features, feature_names = engineer_features(df_raw)

    # 3. Model Training & Selection
    best_model, predictions, test_index = train_and_select_model(df_features, feature_names)
    
    # 4. Backtest Strategy
    df_bt = backtest_strategy(df_features, test_index, predictions)

    # 5. Evaluate and Plot
    evaluate_and_plot(df_bt)

if __name__ == "__main__":
    main()
