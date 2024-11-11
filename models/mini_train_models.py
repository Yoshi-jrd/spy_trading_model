import os
import pickle
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from dynamic_weight_optimizer import optimize_weights

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load and preprocess data with interpolation, synthetic adjustments, and feature engineering
def load_and_preprocess_data():
    # Load 1-hour and 1-day SPY data
    with open("local_data/historical_data.pickle", 'rb') as f:
        data = pickle.load(f)
    spy_data_1h = data['spy_data']['1h']
    spy_data_1d = data['spy_data']['1d']
    
    # Interpolate to create 5-minute intervals from 1-hour data
    spy_data_5m = interpolate_to_5min(spy_data_1h)
    
    # Concatenate interpolated 5-minute data with actual 5-minute data
    spy_data_real_5m = data['spy_data'].get('5m', pd.DataFrame())
    spy_data_combined = pd.concat([spy_data_5m, spy_data_real_5m], ignore_index=True).drop_duplicates().reset_index(drop=True)
    
    # Final preprocessing and adding synthetic variations
    spy_data_combined = spy_data_combined.dropna().reset_index(drop=True)
    spy_data_combined = add_synthetic_variations(spy_data_combined)
    
    # Mapping Impulse_Color to numerical values
    spy_data_combined['Impulse_Color'] = spy_data_combined['Impulse_Color'].map({'red': -1, 'green': 1, 'gray': 0})

    # Feature engineering: Add lagged features and moving averages
    for lag in [1, 2, 3]:
        spy_data_combined[f'Close_lag_{lag}'] = spy_data_combined['Close'].shift(lag)
        spy_data_combined[f'RSI_lag_{lag}'] = spy_data_combined['RSI'].shift(lag)
        spy_data_combined[f'MACD_Histogram_lag_{lag}'] = spy_data_combined['MACD_Histogram'].shift(lag)
    
    for window in [5, 10, 15]:
        spy_data_combined[f'Close_ma_{window}'] = spy_data_combined['Close'].rolling(window).mean()
        spy_data_combined[f'RSI_ma_{window}'] = spy_data_combined['RSI'].rolling(window).mean()
        spy_data_combined[f'MACD_Histogram_ma_{window}'] = spy_data_combined['MACD_Histogram'].rolling(window).mean()

    # Feature engineering: Add seasonal patterns
    spy_data_combined['day_of_week'] = pd.to_datetime(spy_data_combined['datetime']).dt.dayofweek
    spy_data_combined['hour_of_day'] = pd.to_datetime(spy_data_combined['datetime']).dt.hour
    
    # Dropping any rows with NaN values after feature engineering
    spy_data_combined = spy_data_combined.dropna().reset_index(drop=True)
    
    logger.info(f"Extended data with {len(spy_data_combined)} rows, including engineered features.")
    
    return spy_data_combined

# Interpolate 1-hour data to 5-minute intervals
def interpolate_to_5min(spy_data_1h):
    # Resample data to create 5-minute intervals
    spy_data_1h = spy_data_1h.set_index('datetime').resample('5T').interpolate(method='linear').reset_index()
    logger.info(f"Interpolated data to 5-minute intervals. New length: {len(spy_data_1h)}")
    return spy_data_1h

# Add synthetic noise to simulated 5-minute data for realism
def add_synthetic_variations(df, noise_level=0.005):
    np.random.seed(0)
    df['Close'] *= 1 + np.random.normal(0, noise_level, len(df))
    df['Open'] *= 1 + np.random.normal(0, noise_level, len(df))
    df['High'] *= 1 + np.random.normal(0, noise_level, len(df))
    df['Low'] *= 1 + np.random.normal(0, noise_level, len(df))
    return df

# Evaluation function
def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    avg_diff = np.mean(np.abs(y_pred - y_true))
    return mae, rmse, avg_diff

from sklearn.linear_model import Ridge

# Train and stack models with meta-model for interval-specific predictions
def train_and_stack_models(X_train, y_train, X_val, y_val, interval_name):
    # Define base models
    rf_model = RandomForestRegressor(n_estimators=200, max_depth=12)
    gb_model = GradientBoostingRegressor(n_estimators=200, max_depth=4)
    xgb_model = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.1)
    et_model = ExtraTreesRegressor(n_estimators=150, max_depth=10)
    cb_model = CatBoostRegressor(iterations=150, depth=6, learning_rate=0.1, verbose=0)

    # Train base models
    rf_model.fit(X_train, y_train)
    gb_model.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)
    et_model.fit(X_train, y_train)
    cb_model.fit(X_train, y_train)

    # Get predictions from each base model
    predictions_matrix = np.column_stack([
        rf_model.predict(X_val),
        gb_model.predict(X_val),
        xgb_model.predict(X_val),
        et_model.predict(X_val),
        cb_model.predict(X_val)
    ])
    
    # Meta-model to find optimal weights
    ridge_meta_model = Ridge(alpha=1.0)
    ridge_meta_model.fit(predictions_matrix, y_val)
    stacked_predictions = ridge_meta_model.predict(predictions_matrix)
    
    # Log and return the results
    logger.info(f"Optimal weights for {interval_name}: {ridge_meta_model.coef_}")
    return stacked_predictions, ridge_meta_model.coef_

# Main function with rolling predictions over 30 days
def mini_train_models():
    spy_data = load_and_preprocess_data()
    
    # Prediction intervals (in steps of 5 minutes)
    prediction_intervals = {
        '24h': 288,
        '48h': 576,
        '72h': 864,
        '96h': 1152,
        '168h': 2016
    }

    results_summary = {}
    real_prices = []
    predictions_by_interval = {interval: [] for interval in prediction_intervals}

    start_day = 0
    end_day = 30
    daily_step = 288  # 5-minute intervals in one day
    
    for day in range(start_day, end_day):
        start_idx = day * daily_step
        train_data = spy_data.iloc[:start_idx]
        
        if len(train_data) == 0:
            logger.warning(f"Insufficient training data on day {day}. Skipping to next day.")
            continue
        
        # Prepare feature matrix X and target y
        X = train_data[['MACD_Histogram', 'RSI', 'UpperBand', 'LowerBand', 'ATR', 'ADX', 'EMA9', 'EMA21', 'Impulse_Color']]
        y = train_data['Close']
        
        for interval_name, shift_steps in prediction_intervals.items():
            if start_idx + shift_steps >= len(spy_data):
                logger.info(f"End of data reached at {day}-day for interval {interval_name}.")
                break

            y_val = spy_data['Close'].shift(-shift_steps).iloc[start_idx:start_idx + shift_steps].dropna()
            X_val = spy_data[['MACD_Histogram', 'RSI', 'UpperBand', 'LowerBand', 'ATR', 'ADX', 'EMA9', 'EMA21', 'Impulse_Color']].iloc[start_idx:start_idx + shift_steps]
            
            # Ensure consistent length for X_val and y_val
            min_len = min(len(X_val), len(y_val))
            X_val, y_val = X_val.iloc[:min_len], y_val.iloc[:min_len]
            
            if X_val.empty or y_val.empty:
                logger.warning(f"Insufficient validation data for interval {interval_name} on day {day}. Skipping.")
                continue
            
            # Train models and apply meta-model stacking
            stacked_predictions, _ = train_and_stack_models(X, y, X_val, y_val, interval_name)

            # Store predictions and actuals
            predictions_by_interval[interval_name].extend(stacked_predictions)
            if interval_name == '24h':
                real_prices.extend(y_val)

    # Evaluation
    for interval_name, predictions in predictions_by_interval.items():
        real_prices_np = np.array(real_prices[:len(predictions)])  # Align real prices length
        predictions_np = np.array(predictions[:len(real_prices_np)]) 
        
        mae, rmse, avg_diff = evaluate_model(real_prices_np, predictions_np)
        results_summary[interval_name] = {
            'MAE': mae,
            'RMSE': rmse,
            'Avg Difference': avg_diff
        }
        logger.info(f"{interval_name} - MAE: {mae}, RMSE: {rmse}, Avg Difference: {avg_diff}")

    return results_summary

# Run the model
if __name__ == "__main__":
    summary = mini_train_models()
    for interval, metrics in summary.items():
        logger.info(f"{interval} - MAE: {metrics['MAE']}, RMSE: {metrics['RMSE']}, Avg Difference: {metrics['Avg Difference']}")
