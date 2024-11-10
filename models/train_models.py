import os
import sys
import csv
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from scikeras.wrappers import KerasRegressor
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import models and utilities
from random_forest_model import train_random_forest
from xgboost_model import train_xgboost
from gradient_boosting_model import train_gradient_boosting
from lstm_model import build_lstm_model
from model_utils import compute_confidence_interval
from data.data_loader import load_existing_data
from data.indicator_calculator import calculate_indicators
from evaluate_model import evaluate_model  # Importing evaluate_model for model evaluation

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Paths
best_params_file = "best_params.pkl"
model_save_dir = "saved_models"
os.makedirs(model_save_dir, exist_ok=True)

# Load best parameters
def load_best_params():
    if os.path.exists(best_params_file):
        try:
            with open(best_params_file, 'rb') as f:
                return pickle.load(f)
        except EOFError:
            logger.warning("best_params.pkl is empty or corrupted. Using default parameters.")
    return {
        'RandomForest': {'n_estimators': 200, 'max_depth': 20, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt'},
        'XGBoost': {'n_estimators': 100, 'learning_rate': 0.2, 'max_depth': 3, 'min_child_weight': 3, 'gamma': 0},
        'GradientBoosting': {'n_estimators': 300, 'learning_rate': 0.1, 'max_depth': 3, 'min_samples_split': 10, 'min_samples_leaf': 1},
        'LSTM': {'model__units': 100, 'model__dropout': 0.2, 'model__learning_rate': 0.01, 'batch_size': 32, 'epochs': 50}
    }

# Save best parameters
def save_best_params(best_params):
    with open(best_params_file, 'wb') as f:
        pickle.dump(best_params, f)

# Train and evaluate a model with logging and confidence interval output
def train_and_evaluate_model(model, X_train, y_train, X_val, y_val, model_name, timeframe):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mae, rmse, conf_intervals = [], [], []

    for train_index, val_index in kf.split(X_train):
        X_fold_train, X_fold_val = X_train.iloc[train_index], X_train.iloc[val_index]
        y_fold_train, y_fold_val = y_train.iloc[train_index], y_train.iloc[val_index]
        model.fit(X_fold_train, y_fold_train)
        predictions = model.predict(X_fold_val)
        
        # Evaluate and log performance
        fold_mae, fold_rmse = evaluate_model(y_fold_val, predictions)
        fold_lower, fold_upper = compute_confidence_interval(predictions, confidence_level=0.75)
        mae.append(fold_mae)
        rmse.append(fold_rmse)
        conf_intervals.append((fold_lower, fold_upper))
    
    avg_mae, avg_rmse = np.mean(mae), np.mean(rmse)
    lower, upper = np.mean([ci[0] for ci in conf_intervals]), np.mean([ci[1] for ci in conf_intervals])
    logger.info(f"{model_name} - {timeframe}-hour Forward - MAE: {avg_mae}, RMSE: {avg_rmse}, CI: ({lower}, {upper})")
    return avg_mae, avg_rmse, lower, upper

# Initialize best parameters and load data
best_params = load_best_params()
data = load_existing_data()
spy_data = data['spy_data']

# Timeframes for multiple forward predictions
timeframes = [12, 24, 36, 48, 72, 96]

# Training and Evaluation Loop
predictions_summary = []  # Store predictions for visualization
for timeframe in timeframes:
    for tf_name, spy_df in spy_data.items():
        logger.info(f"Training models on {tf_name} timeframe for {timeframe}-hour forward prediction")
        spy_df = calculate_indicators(spy_df)
        spy_df['Impulse_Color'] = spy_df['Impulse_Color'].map({'red': -1, 'green': 1, 'gray': 0})

        # Feature and Target Setup, including augmented feature if available
        selected_features = ['MACD_Histogram', 'RSI', 'UpperBand', 'LowerBand', 'ATR', 'ADX', 'EMA9', 'EMA21', 'Impulse_Color']
        if 'Close_augmented' in spy_df.columns:
            selected_features.append('Close_augmented')
        X = spy_df[selected_features]
        y = spy_df['Close'].shift(-timeframe).dropna()  # Target shifted by `timeframe` steps
        X = X.iloc[:-timeframe]  # Align features with shifted target
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # RandomForest Model Training
        rf_model = RandomForestRegressor(**best_params['RandomForest'])
        rf_mae, rf_rmse, rf_lower, rf_upper = train_and_evaluate_model(
            rf_model, X_train, y_train, X_val, y_val, "RandomForest", timeframe)

        # XGBoost Model Training
        xgb_model = train_xgboost(X_train, y_train, **best_params['XGBoost'])
        xgb_mae, xgb_rmse, xgb_lower, xgb_upper = train_and_evaluate_model(
            xgb_model, X_train, y_train, X_val, y_val, "XGBoost", timeframe)

        # Gradient Boosting Model Training
        gb_model = train_gradient_boosting(X_train, y_train, **best_params['GradientBoosting'])
        gb_mae, gb_rmse, gb_lower, gb_upper = train_and_evaluate_model(
            gb_model, X_train, y_train, X_val, y_val, "GradientBoosting", timeframe)

        # LSTM Model Training
        try:
            X_train_reshaped = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
            X_val_reshaped = X_val.values.reshape((X_val.shape[0], 1, X_val.shape[1]))
            
            lstm_params = best_params.get('LSTM', {})
            lstm_model = KerasRegressor(
                model=build_lstm_model,
                model__input_shape=(1, X_train.shape[1]),
                model__units=lstm_params.get('model__units', 100),
                model__dropout=lstm_params.get('model__dropout', 0.2),
                model__learning_rate=lstm_params.get('model__learning_rate', 0.01),
                batch_size=lstm_params.get('batch_size', 32),
                epochs=lstm_params.get('epochs', 50),
                verbose=0
            )
            lstm_model.fit(X_train_reshaped, y_train)
            lstm_predictions = lstm_model.predict(X_val_reshaped).flatten()
            lstm_lower, lstm_upper = compute_confidence_interval(lstm_predictions, confidence_level=0.75)
        except Exception as e:
            logger.error(f"Error training LSTM model on {tf_name} timeframe: {e}")

        # Model Stacking with Meta-Model
        meta_X_train = np.column_stack([rf_mae, xgb_mae, gb_mae, lstm_predictions])
        meta_y_train = y_val
        meta_model = LinearRegression().fit(meta_X_train, meta_y_train)
        stacked_predictions = meta_model.predict(meta_X_train)

        # Write training results to CSV
        with open('model_training_results.csv', mode='a') as file:
            writer = csv.writer(file)
            writer.writerow(["Stacked Model", timeframe, np.mean(stacked_predictions), rf_lower, rf_upper])

# Save updated best parameters
save_best_params(best_params)

# Visualization: Plot predictions with confidence intervals and actual values
plt.figure(figsize=(14, 10))
for idx, (predictions, lower, upper, actual, model_name, timeframe) in enumerate(predictions_summary, 1):
    plt.subplot(3, 2, idx)
    plt.plot(actual.index, actual, label="Actual Price", color='black')
    plt.plot(actual.index, predictions, label="Predicted Price", color='blue')
    plt.fill_between(actual.index, lower, upper, color='blue', alpha=0.2, label="75% Confidence Interval")
    plt.title(f"{model_name} Predictions vs Actual ({timeframe}-hour forecast)")
    plt.xlabel("Date")
    plt.ylabel("SPY Price")
    plt.legend()
    if idx >= 6:
        break  # Limit to 6 plots to keep visualization clear

plt.tight_layout()
plt.show()
