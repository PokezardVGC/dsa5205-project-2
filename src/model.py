import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

def train_and_select_model(df, features):
    """
    Splits data chronologically, scales features, and trains a Ridge regression model
    using TimeSeriesSplit cross-validation for hyperparameter tuning.
    """
    print("Splitting data and training model...")
    X = df[features]
    y = df['Target_Next_Return'].clip(-0.5, 0.5)

    # Time series split (chronological): Train 80%, Test 20%
    # We will use TimeSeriesSplit for cross-validation within the Train set
    train_size = int(len(df) * 0.8)
    
    X_train_val = X.iloc[:train_size]
    y_train_val = y.iloc[:train_size]

    X_test = X.iloc[train_size:]
    y_test = y.iloc[train_size:]

    print(f"Train/Val size: {len(X_train_val)}, Test size: {len(X_test)}")

    # We use Ridge to handle multicollinearity and prevent overfitting
    param_grid = [0.1, 1.0, 10.0, 100.0, 1000.0]
    
    best_alpha = None
    best_cv_score = -float('inf')
    best_model = None

    # Implement TimeSeriesSplit (expanding window) for robust hyperparameter tuning
    tscv = TimeSeriesSplit(n_splits=5)
    
    for alpha in param_grid:
        scores = []
        for train_index, val_index in tscv.split(X_train_val):
            X_tr, X_v = X_train_val.iloc[train_index], X_train_val.iloc[val_index]
            y_tr, y_v = y_train_val.iloc[train_index], y_train_val.iloc[val_index]
            
            # Scale features strictly within the fold to prevent leakage
            fold_scaler = StandardScaler()
            X_tr_scaled = fold_scaler.fit_transform(X_tr)
            X_v_scaled = fold_scaler.transform(X_v)
            
            model = Ridge(alpha=alpha)
            model.fit(X_tr_scaled, y_tr)
            scores.append(model.score(X_v_scaled, y_v))
            
        avg_score = np.mean(scores)
        if avg_score > best_cv_score:
            best_cv_score = avg_score
            best_alpha = alpha
            
    # Re-instantiate the model with the best alpha
    best_model = Ridge(alpha=best_alpha)
    print(f"Best Alpha selected: {best_alpha}")

    # Retrain best model on the entire Train+Val set
    # Fit scaler on combined Train+Val set to avoid lookahead/test leakage
    final_scaler = StandardScaler()
    X_train_val_scaled = final_scaler.fit_transform(X_train_val)
    
    best_model.fit(X_train_val_scaled, y_train_val)
    
    # Scale test set using the FINAL scaler
    X_test_scaled = final_scaler.transform(X_test)

    # Generate predictions on the test set
    predictions = best_model.predict(X_test_scaled)
    
    return best_model, predictions, X_test.index
