# Import necessary modules and functions
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.market_data_loader import load_spy_multi_timeframes, get_vix_futures, get_all_iv
from data.economic_data_loader import load_economic_data
from data.sentiment_data_loader import get_news_sentiment
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import pandas as pd
import numpy as np

# Load the custom data loader function
def load_data():
    spy_df = load_spy_multi_timeframes()  # Load SPY data
    vix_df = get_vix_futures()            # Load VIX data
    iv_data = get_all_iv()                # Load IV data
    gdp_data, cpi_data = load_economic_data()  # Load economic data
    avg_sentiment, articles = get_news_sentiment()  # Fetch news sentiment data

    # Add a NaN check function to detect data issues
    def check_nan_in_data(data, name):
        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            if data.isna().sum().sum() > 0:
                print(f"Warning: {name} contains NaN values.")
            else:
                print(f"{name} contains no NaN values.")
        else:
            print(f"{name} is not a DataFrame or Series.")

    for timeframe, df in spy_df.items():
        check_nan_in_data(df, f"SPY data ({timeframe})")
    
    check_nan_in_data(vix_df, "VIX data")
    check_nan_in_data(iv_data, "IV data")
    check_nan_in_data(gdp_data, "GDP data")
    check_nan_in_data(cpi_data, "CPI data")

    data = {
        'spy_data': spy_df,
        'vix_data': vix_df,
        'iv_data': iv_data,
        'gdp_data': gdp_data,
        'cpi_data': cpi_data,
        'sentiment_data': avg_sentiment,
        'articles': articles
    }

    print("Final data structure:")
    for key, value in data.items():
        if isinstance(value, pd.DataFrame) or isinstance(value, pd.Series):
            print(f"{key}: {value.shape}")
        else:
            print(f"{key}: {type(value)}")

    return data

# Prepare the data for training and testing
def prepare_data():
    data = load_data()
    spy_5m_data = data['spy_data']['5m']

    # Ensure 'Date' column exists and is properly set
    if 'Date' in spy_5m_data.columns:
        spy_5m_data['Date'] = pd.to_datetime(spy_5m_data['Date'])
        spy_5m_data.set_index('Date', inplace=True)

    # Add cyclical encoding for time features
    spy_5m_data['day_of_week'] = spy_5m_data.index.dayofweek
    spy_5m_data['hour'] = spy_5m_data.index.hour
    spy_5m_data['minute'] = spy_5m_data.index.minute
    spy_5m_data['hour_sin'] = np.sin(2 * np.pi * spy_5m_data['hour'] / 24)
    spy_5m_data['hour_cos'] = np.cos(2 * np.pi * spy_5m_data['hour'] / 24)
    spy_5m_data['minute_sin'] = np.sin(2 * np.pi * spy_5m_data['minute'] / 60)
    spy_5m_data['minute_cos'] = np.cos(2 * np.pi * spy_5m_data['minute'] / 60)

    # Ensure VIX data is properly indexed and resampled
    vix_data = data['vix_data']
    if 'Date' in vix_data.columns:
        vix_data['Date'] = pd.to_datetime(vix_data['Date'])
        vix_data.set_index('Date', inplace=True)

    vix_data_resampled = vix_data.resample('5min').ffill().reindex(spy_5m_data.index)

    # Define features and target
    features = spy_5m_data.drop('Close', axis=1)
    target = spy_5m_data['Close']

    # Combine with VIX data if needed
    combined_data = pd.merge(features, vix_data_resampled, left_index=True, right_index=True, how='left')

    # Fill missing values
    combined_data.fillna(method='ffill', inplace=True)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(combined_data, target, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

# Hyperparameter tuning functions
def hypertune_random_forest(X_train, y_train):
    rf_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    rf = RandomForestRegressor()
    rf_cv = GridSearchCV(estimator=rf, param_grid=rf_grid, cv=3, scoring='neg_mean_squared_error')
    rf_cv.fit(X_train, y_train)
    print("Best parameters for RandomForest:", rf_cv.best_params_)
    return rf_cv.best_estimator_

def hypertune_xgboost(X_train, y_train):
    xgb_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.3]
    }
    xgboost = xgb.XGBRegressor()
    xgb_cv = GridSearchCV(estimator=xgboost, param_grid=xgb_grid, cv=3, scoring='neg_mean_squared_error')
    xgb_cv.fit(X_train, y_train)
    print("Best parameters for XGBoost:", xgb_cv.best_params_)
    return xgb_cv.best_estimator_

def hypertune_gradient_boosting(X_train, y_train):
    gb_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    gb = GradientBoostingRegressor()
    gb_cv = GridSearchCV(estimator=gb, param_grid=gb_grid, cv=3, scoring='neg_mean_squared_error')
    gb_cv.fit(X_train, y_train)
    print("Best parameters for Gradient Boosting:", gb_cv.best_params_)
    return gb_cv.best_estimator_

# LSTM Hypertuning
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasRegressor

def build_lstm_model(units=50, dropout=0.2, learning_rate=0.01):
    model = Sequential()
    model.add(LSTM(units=units, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(dropout))
    model.add(LSTM(units=units))
    model.add(Dense(1))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

def hypertune_lstm(X_train, y_train):
    print("Tuning LSTM...")
    lstm = KerasRegressor(build_fn=build_lstm_model, epochs=50, batch_size=32, verbose=0)

    lstm_grid = {
        'units': [50, 100, 150],
        'dropout': [0.2, 0.3, 0.4],
        'learning_rate': [0.001, 0.01, 0.02],
        'batch_size': [32, 64, 128],
        'epochs': [50, 100]
    }

    lstm_cv = GridSearchCV(estimator=lstm, param_grid=lstm_grid, cv=3, scoring='neg_mean_squared_error')
    lstm_cv.fit(X_train, y_train)

    print("Best parameters for LSTM:", lstm_cv.best_params_)
    return lstm_cv.best_estimator_

# Evaluate model performance using MAE and RMSE
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    return mae, rmse

# Main hypertuning and evaluation process
X_train, X_test, y_train, y_test = prepare_data()

# Hypertune RandomForest, XGBoost, GradientBoosting, and LSTM
rf_model = hypertune_random_forest(X_train, y_train)
xgb_model = hypertune_xgboost(X_train, y_train)
gb_model = hypertune_gradient_boosting(X_train, y_train)
lstm_model = hypertune_lstm(X_train, y_train)

# Evaluate each model's performance
rf_mae, rf_rmse = evaluate_model(rf_model, X_test, y_test)
xgb_mae, xgb_rmse = evaluate_model(xgb_model, X_test, y_test)
gb_mae, gb_rmse = evaluate_model(gb_model, X_test, y_test)

# LSTM evaluation needs special handling due to its Keras structure
lstm_mae, lstm_rmse = evaluate_model(lstm_model, X_test, y_test)

# Store and display the results for comparison
results = pd.DataFrame({
    'Model': ['RandomForest', 'XGBoost', 'GradientBoosting', 'LSTM'],
    'MAE': [rf_mae, xgb_mae, gb_mae, lstm_mae],
    'RMSE': [rf_rmse, xgb_rmse, gb_rmse, lstm_rmse]
})

print("Model Evaluation Results:")
print(results)

# Select the best model based on RMSE or MAE
best_model_idx = results['RMSE'].idxmin()  # Change to 'MAE' if you prefer MAE
best_model_name = results.iloc[best_model_idx]['Model']
print(f"The best performing model is {best_model_name} with RMSE: {results.iloc[best_model_idx]['RMSE']}")

