# Import necessary modules and functions
import sys
import os
import logging
import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from keras.optimizers import Adam
from scikeras.wrappers import KerasRegressor
from keras.callbacks import EarlyStopping

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load data from historical_data.pickle
def load_pickle_data(file_path='local_data/historical_data.pickle'):
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        logger.info("Data loaded successfully from historical_data.pickle")
        return data
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return None

# Prepare data using the pickle file
def prepare_data(timeframes=['5m', '15m', '1h', '1d']):
    data = load_pickle_data()
    if data is None:
        return None, None, None, None

    combined_data = []
    for timeframe in timeframes:
        spy_data = data.get('spy_data', {}).get(timeframe, pd.DataFrame())
        if spy_data.empty:
            logger.warning(f"SPY {timeframe} data is missing or empty in the pickle file.")
            continue
        
        if 'datetime' in spy_data.columns:
            spy_data['datetime'] = pd.to_datetime(spy_data['datetime'], errors='coerce')
            spy_data.set_index('datetime', inplace=True)

        color_mapping = {'red': -1, 'green': 1, 'gray': 0}
        spy_data['Impulse_Color'] = spy_data['Impulse_Color'].map(color_mapping)

        spy_data['day_of_week'] = spy_data.index.dayofweek
        spy_data['hour'] = spy_data.index.hour
        spy_data['minute'] = spy_data.index.minute
        spy_data['hour_sin'] = np.sin(2 * np.pi * spy_data['hour'] / 24)
        spy_data['hour_cos'] = np.cos(2 * np.pi * spy_data['hour'] / 24)
        spy_data['minute_sin'] = np.sin(2 * np.pi * spy_data['minute'] / 60)
        spy_data['minute_cos'] = np.cos(2 * np.pi * spy_data['minute'] / 60)

        combined_data.append(spy_data)

    if not combined_data:
        logger.error("No valid SPY data available for the specified timeframes.")
        return None, None, None, None

    all_timeframes_data = pd.concat(combined_data)
    features = all_timeframes_data.drop(columns=['Close', 'datetime'], errors='ignore').select_dtypes(include=[np.number])
    target = all_timeframes_data['Close']

    # Handle NaNs in the feature set
    features = features.ffill().bfill()  # Forward and backward fill
    target = target.ffill().bfill()  # Forward and backward fill

    # Split data for training/testing
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    # Confirm there are no NaNs after filling
    if X_train.isna().sum().sum() > 0 or y_train.isna().sum() > 0:
        logger.error("NaN values remain in the training data after filling.")
    else:
        logger.info("NaN handling complete; no NaNs present in training data.")

    return X_train, X_test, y_train, y_test

# Model hypertuning functions
def hypertune_random_forest(X_train, y_train):
    rf_grid = {'n_estimators': [100, 200, 300], 'max_depth': [10, 20, None],
               'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4], 'max_features': ['sqrt', 'log2']}
    rf = RandomForestRegressor()
    rf_cv = GridSearchCV(estimator=rf, param_grid=rf_grid, cv=3, scoring='neg_mean_squared_error')
    rf_cv.fit(X_train, y_train)
    logger.info(f"Best parameters for RandomForest: {rf_cv.best_params_}")
    return rf_cv.best_params_

def hypertune_xgboost(X_train, y_train):
    xgb_grid = {'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.1, 0.2], 
                'max_depth': [3, 5, 7], 'min_child_weight': [1, 3, 5], 'gamma': [0, 0.1, 0.3]}
    xgboost = xgb.XGBRegressor()
    xgb_cv = GridSearchCV(estimator=xgboost, param_grid=xgb_grid, cv=3, scoring='neg_mean_squared_error')
    xgb_cv.fit(X_train, y_train)
    logger.info(f"Best parameters for XGBoost: {xgb_cv.best_params_}")
    return xgb_cv.best_params_

def hypertune_gradient_boosting(X_train, y_train):
    gb_grid = {'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.1, 0.2],
               'max_depth': [3, 5, 7], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
    gb = GradientBoostingRegressor()
    gb_cv = GridSearchCV(estimator=gb, param_grid=gb_grid, cv=3, scoring='neg_mean_squared_error')
    gb_cv.fit(X_train, y_train)
    logger.info(f"Best parameters for GradientBoosting: {gb_cv.best_params_}")
    return gb_cv.best_params_

# LSTM hypertuning
def build_lstm_model(units=50, dropout=0.2, learning_rate=0.01, input_shape=(1, 1)):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(units=units, return_sequences=True),
        Dropout(dropout),
        LSTM(units=units),
        Dense(1)
    ])
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

def hypertune_lstm(X_train, y_train):
    print("Tuning LSTM...")

    try:
        lstm = KerasRegressor(
            model=build_lstm_model,
            model__input_shape=(X_train.shape[1], 1)
        )
        
        # Define the parameter grid for GridSearchCV
        lstm_grid = {
            'model__units': [50, 100, 150],
            'model__dropout': [0.2, 0.3, 0.4],
            'model__learning_rate': [0.001, 0.01, 0.02],
            'batch_size': [32, 64, 128],
            'epochs': [50]
        }

        # Use GridSearchCV to find the best hyperparameters
        lstm_cv = GridSearchCV(estimator=lstm, param_grid=lstm_grid, cv=3, scoring='neg_mean_squared_error', verbose=2)
        lstm_cv.fit(X_train, y_train)

        best_params = lstm_cv.best_params_
        logger.info(f"Best parameters for LSTM: {best_params} with loss: {-lstm_cv.best_score_}")
        return best_params

    except Exception as e:
        logger.error(f"Error in LSTM hypertuning: {e}")
        return None

# Main function
def main():
    X_train, X_test, y_train, y_test = prepare_data()

    # Load existing best parameters if they exist
    try:
        with open('best_params.pkl', 'rb') as f:
            best_params = pickle.load(f)
    except (FileNotFoundError, EOFError):
        best_params = {}

    # Hypertune each model and update best_params if run
    # best_params['RandomForest'] = hypertune_random_forest(X_train, y_train)  # Uncomment to run
    # best_params['XGBoost'] = hypertune_xgboost(X_train, y_train)  # Uncomment to run
    # best_params['GradientBoosting'] = hypertune_gradient_boosting(X_train, y_train)  # Uncomment to run
    best_params['LSTM'] = hypertune_lstm(X_train, y_train)

    # Save updated best parameters back to the file
    with open('best_params.pkl', 'wb') as f:
        pickle.dump(best_params, f)
    logger.info("Best parameters saved to best_params.pkl")

    # Model evaluation summary
    for model_name, params in best_params.items():
        logger.info(f"{model_name} best parameters: {params}")

if __name__ == "__main__":
    main()
