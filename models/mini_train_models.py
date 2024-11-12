import os
import pickle
import numpy as np
import pandas as pd
import logging
import tensorflow as tf
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from dynamic_weight_optimizer import optimize_weights
from evaluate_model import evaluate_model

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Path to save best parameters
BEST_PARAMS_PATH = "best_model_params.json"

# Load previous best parameters if available
def load_best_params():
    if os.path.exists(BEST_PARAMS_PATH):
        with open(BEST_PARAMS_PATH, 'r') as f:
            return json.load(f)
    return None

# Save best parameters to JSON
def save_best_params(params):
    with open(BEST_PARAMS_PATH, 'w') as f:
        json.dump(params, f)

# Early stopping based on performance deterioration
def early_stopping_check(current_mae, current_rmse, threshold=0.1):
    best_params = load_best_params()
    if best_params:
        if current_mae > best_params['MAE'] * (1 + threshold) or current_rmse > best_params['RMSE'] * (1 + threshold):
            logger.info("Performance deteriorated. Using previously saved best parameters.")
            return best_params
    return None

# LSTM model definition
def build_lstm(sequence_length, feature_count):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(sequence_length, feature_count)),
        tf.keras.layers.LSTM(64, return_sequences=False),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Load and preprocess 15-minute data
def load_and_preprocess_data():
    with open("local_data/historical_data.pickle", 'rb') as f:
        data = pickle.load(f)
    
    # Load only 15-minute data
    spy_data_15m = data['spy_data']['15m']

    # Add suffixes for clarity and consistency in feature names
    spy_data_15m = spy_data_15m.add_suffix('_15m')
    spy_data_15m.rename(columns={'datetime_15m': 'datetime'}, inplace=True)

    # Feature Engineering
    spy_data_15m['Impulse_Color_15m'] = spy_data_15m['Impulse_Color_15m'].map({'red': -1, 'green': 1, 'gray': 0})

    # Generate lag and moving average features based on 15-minute data
    for lag in [1, 2, 3]:
        spy_data_15m[f'Close_15m_lag_{lag}'] = spy_data_15m['Close_15m'].shift(lag)
    for window in [5, 10, 15]:
        spy_data_15m[f'Close_15m_ma_{window}'] = spy_data_15m['Close_15m'].rolling(window).mean()

    spy_data_15m['day_of_week'] = pd.to_datetime(spy_data_15m['datetime']).dt.dayofweek
    spy_data_15m['hour_of_day'] = pd.to_datetime(spy_data_15m['datetime']).dt.hour

    spy_data_15m = spy_data_15m.dropna().reset_index(drop=True)

    logger.info(f"Processed data with {len(spy_data_15m)} rows and features based on 15-minute intervals.")
    return spy_data_15m

# Updated prepare_lstm_data with debugging and shape checks
def prepare_lstm_data(df, sequence_length=48):
    if len(df) < sequence_length:
        raise ValueError("Dataframe length is smaller than sequence length; cannot prepare sequences.")
        
    scaler = MinMaxScaler()
    close_scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['Close_15m', 'MACD_Histogram_15m', 'RSI_15m', 'UpperBand_15m', 'LowerBand_15m']])
    scaled_close = close_scaler.fit_transform(df[['Close_15m']])

    sequences, targets = [], []
    for i in range(len(scaled_data) - sequence_length):
        sequences.append(scaled_data[i:i + sequence_length])
        targets.append(scaled_close[i + sequence_length, 0])

    sequences = np.array(sequences)
    targets = np.array(targets).reshape(-1, 1)
    
    # Debugging shapes
    logger.info(f"Sequences shape: {sequences.shape}, Targets shape: {targets.shape}")
    return sequences, targets, close_scaler

# Updated train_and_stack_models_with_lstm with model creation moved outside the loop
def train_and_stack_models_with_lstm(original_data, y_train, X_val, y_val, interval_name, sequence_length=48):
    # Data preparation
    X_train_seq, y_train_seq, close_scaler = prepare_lstm_data(original_data, sequence_length)
    X_val_seq, y_val_seq, _ = prepare_lstm_data(original_data.iloc[:len(X_val)], sequence_length)
    
    # Ensure data shapes are as expected
    assert X_train_seq.shape[1:] == (sequence_length, 5), f"Expected shape (batch, {sequence_length}, 5), got {X_train_seq.shape}"
    assert X_val_seq.shape[1:] == (sequence_length, 5), f"Expected shape (batch, {sequence_length}, 5), got {X_val_seq.shape}"

    # Build and fit LSTM model
    lstm_model = build_lstm(sequence_length, X_train_seq.shape[2])
    lstm_model.fit(X_train_seq, y_train_seq, epochs=10, batch_size=32, validation_data=(X_val_seq, y_val_seq), verbose=0)

    # LSTM Predictions
    lstm_predictions = lstm_model.predict(X_val_seq).flatten()
    lstm_predictions = close_scaler.inverse_transform(lstm_predictions.reshape(-1, 1)).flatten()

    # Train base models
    rf_model = RandomForestRegressor(n_estimators=200, max_depth=12)
    gb_model = GradientBoostingRegressor(n_estimators=200, max_depth=4)
    xgb_model = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.1)
    et_model = ExtraTreesRegressor(n_estimators=150, max_depth=10)
    cb_model = CatBoostRegressor(iterations=150, depth=6, learning_rate=0.1, verbose=0)

    rf_model.fit(X_val, y_val)
    gb_model.fit(X_val, y_val)
    xgb_model.fit(X_val, y_val)
    et_model.fit(X_val, y_val)
    cb_model.fit(X_val, y_val)

    # Stack predictions
    base_predictions = [rf_model.predict(X_val), gb_model.predict(X_val), xgb_model.predict(X_val), et_model.predict(X_val), cb_model.predict(X_val)]
    min_length = len(lstm_predictions)
    base_predictions = [pred[:min_length] for pred in base_predictions]
    y_val = y_val[:min_length]
    predictions_matrix = np.column_stack(base_predictions + [lstm_predictions])

    # Ridge meta-model for stacked predictions
    ridge_meta_model = Ridge(alpha=1.0)
    ridge_meta_model.fit(predictions_matrix, y_val)
    stacked_predictions = ridge_meta_model.predict(predictions_matrix)

    # Calculate metrics and early stopping check
    mae, rmse, avg_diff = evaluate_model(y_val, stacked_predictions)
    best_params = early_stopping_check(mae, rmse)

    if not best_params:
        save_best_params({"MAE": mae, "RMSE": rmse, "weights": ridge_meta_model.coef_.tolist()})
        logger.info(f"New best parameters saved for interval {interval_name}")

    logger.info(f"Optimal weights for {interval_name} with LSTM: {ridge_meta_model.coef_}")
    return stacked_predictions, ridge_meta_model.coef_

# Main training function with predictions
def mini_train_models():
    spy_data = load_and_preprocess_data()

    # Use suffixed column names based on the timeframe (15m suffix is primary for features)
    original_features_data = spy_data[['Close_15m', 'MACD_Histogram_15m', 'RSI_15m', 'UpperBand_15m', 'LowerBand_15m']]
    
    prediction_intervals = {'24h': 96, '48h': 192, '72h': 288, '96h': 384, '168h': 672}

    results_summary = {}
    real_prices = []
    predictions_by_interval = {interval: [] for interval in prediction_intervals}

    start_day = 0
    end_day = 30
    daily_step = 96

    for day in range(start_day, end_day):
        start_idx = day * daily_step
        train_data = spy_data.iloc[:start_idx]
        
        if len(train_data) == 0:
            logger.warning(f"Insufficient training data on day {day}. Skipping to next day.")
            continue

        # Training data columns
        X = train_data[['MACD_Histogram_15m', 'RSI_15m', 'UpperBand_15m', 'LowerBand_15m', 'ATR_15m', 'ADX_15m', 'EMA9_15m', 'EMA21_15m', 'Impulse_Color_15m']]
        y = train_data['Close_15m']

        for interval_name, shift_steps in prediction_intervals.items():
            # Ensure sufficient data is available for the specified interval
            if start_idx + shift_steps >= len(spy_data):
                logger.info(f"End of data reached at {day}-day for interval {interval_name}.")
                break

            # Validation data for the specified interval
            y_val = spy_data['Close_15m'].shift(-shift_steps).iloc[start_idx:start_idx + shift_steps].dropna()
            X_val = spy_data[['MACD_Histogram_15m', 'RSI_15m', 'UpperBand_15m', 'LowerBand_15m', 'ATR_15m', 'ADX_15m', 'EMA9_15m', 'EMA21_15m', 'Impulse_Color_15m']].iloc[start_idx:start_idx + shift_steps]
            
            # Ensure consistent length between X_val and y_val
            min_len = min(len(X_val), len(y_val))
            X_val, y_val = X_val.iloc[:min_len], y_val.iloc[:min_len]
            
            # Check if X_val has enough rows for sequence length
            if len(X_val) < 48:  # sequence_length is 48
                logger.warning(f"Insufficient validation data for interval {interval_name} on day {day}. Required: 48 rows, found: {len(X_val)}. Skipping this interval.")
                continue
            
            # Train and stack models, including LSTM for predictions
            stacked_predictions, _ = train_and_stack_models_with_lstm(original_features_data, y, X_val, y_val, interval_name=interval_name)

            # Store predictions and actual values
            predictions_by_interval[interval_name].extend(stacked_predictions)
            if interval_name == '24h':
                real_prices.extend(y_val)

    # Final evaluation and summarizing results
    for interval_name, predictions in predictions_by_interval.items():
        real_prices_np = np.array(real_prices[:len(predictions)])  # Align lengths
        predictions_np = np.array(predictions[:len(real_prices_np)])

        # Check for empty arrays before evaluating
        if real_prices_np.size == 0 or predictions_np.size == 0:
            logger.warning(f"No data available for interval {interval_name} - skipping evaluation.")
            continue

        mae, rmse, avg_diff = evaluate_model(real_prices_np, predictions_np)
        results_summary[interval_name] = {
            'MAE': mae,
            'RMSE': rmse,
            'Avg Difference': avg_diff
        }
        logger.info(f"{interval_name} - MAE: {mae}, RMSE: {rmse}, Avg Difference: {avg_diff}")

    return results_summary

if __name__ == "__main__":
    summary = mini_train_models()
    for interval, metrics in summary.items():
        logger.info(f"{interval} - MAE: {metrics['MAE']}, RMSE: {metrics['RMSE']}, Avg Difference: {metrics['Avg Difference']}")

