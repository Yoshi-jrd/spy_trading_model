import os
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from scikeras.wrappers import KerasRegressor
import logging

# Import models and utilities
from random_forest_model import train_random_forest
from xgboost_model import train_xgboost
from gradient_boosting_model import train_gradient_boosting
from lstm_model import build_lstm_model
from evaluate_model import compute_confidence_interval, evaluate_model
from dynamic_weight_optimizer import optimize_weights  # Import weight optimizer

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Paths
best_params_file = "best_params.pkl"
data_file = "local_data/historical_data.pickle"
model_save_dir = "saved_models"
results_file = "model_results_summary.json"
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
        'LSTM': {'model__units': 100, 'model__dropout': 0.2, 'learning_rate': 0.01, 'batch_size': 32, 'epochs': 50}
    }

# Save best parameters
def save_best_params(best_params):
    with open(best_params_file, 'wb') as f:
        pickle.dump(best_params, f)

# Load and preprocess data
def load_and_preprocess_data():
    try:
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        logger.info("Data loaded successfully from historical_data.pickle")
    except (FileNotFoundError, KeyError, pickle.UnpicklingError) as e:
        logger.error(f"Error loading data from {data_file}: {e}")
        raise
    
    spy_data = data['spy_data']
    color_mapping = {'red': -1, 'green': 1, 'gray': 0}
    for tf, df in spy_data.items():
        if 'Impulse_Color' in df.columns:
            df['Impulse_Color'] = df['Impulse_Color'].map(color_mapping)
            logger.info(f"Converted Impulse_Color in {tf} data to numerical values")
    return spy_data

def save_best_weights(timeframe, weights):
    with open(f"best_weights_{timeframe}.pkl", 'wb') as f:
        pickle.dump(weights, f)

# Train and evaluate a model
def train_and_evaluate_model(model, X_train, y_train, X_val, y_val, model_name, timeframe):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mae, rmse, conf_intervals, all_predictions = [], [], [], []

    for fold, (train_index, val_index) in enumerate(kf.split(X_train), 1):
        X_fold_train, X_fold_val = X_train.iloc[train_index], X_train.iloc[val_index]
        y_fold_train, y_fold_val = y_train.iloc[train_index], y_train.iloc[val_index]
        model.fit(X_fold_train, y_fold_train)
        predictions = model.predict(X_fold_val)
        
        fold_mae, fold_rmse = evaluate_model(y_fold_val, predictions)
        fold_lower, fold_upper = compute_confidence_interval(predictions, confidence_level=0.75)
        mae.append(fold_mae)
        rmse.append(fold_rmse)
        conf_intervals.append((fold_lower, fold_upper))
        all_predictions.extend(predictions)
        
        logger.info(f"{model_name} - Fold {fold}/{kf.n_splits} - MAE: {fold_mae}, RMSE: {fold_rmse}, CI: ({fold_lower}, {fold_upper})")

    avg_mae, avg_rmse = np.mean(mae), np.mean(rmse)
    lower, upper = np.mean([ci[0] for ci in conf_intervals]), np.mean([ci[1] for ci in conf_intervals])
    logger.info(f"{model_name} - {timeframe}-hour Forward - Avg MAE: {avg_mae}, Avg RMSE: {avg_rmse}, CI: ({lower}, {upper})")
    return avg_mae, avg_rmse, lower, upper, np.array(all_predictions)

# Main code execution
best_params = load_best_params()
spy_data = load_and_preprocess_data()
timeframes = [12, 24, 36, 48, 72, 96]
results_summary = {}
predictions_summary = []  # Store predictions for plotting

# Training and Evaluation Loop
for timeframe in timeframes:
    for tf_name, spy_df in spy_data.items():
        logger.info(f"Starting training on {tf_name} for {timeframe}-hour forward prediction")
        
        spy_df = spy_df.dropna().reset_index(drop=True)  # Ensure data integrity
        selected_features = ['MACD_Histogram', 'RSI', 'UpperBand', 'LowerBand', 'ATR', 'ADX', 'EMA9', 'EMA21', 'Impulse_Color']
        if 'Close_augmented' in spy_df.columns:
            selected_features.append('Close_augmented')
        
        X = spy_df[selected_features]
        y = spy_df['Close'].shift(-timeframe).dropna()
        X = X.iloc[:-timeframe]
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        rf_model = RandomForestRegressor(**best_params['RandomForest'])
        rf_mae, rf_rmse, rf_lower, rf_upper, rf_predictions = train_and_evaluate_model(
            rf_model, X_train, y_train, X_val, y_val, "RandomForest", timeframe)
        
        xgb_model = train_xgboost(X_train, y_train, **best_params['XGBoost'])
        xgb_mae, xgb_rmse, xgb_lower, xgb_upper, xgb_predictions = train_and_evaluate_model(
            xgb_model, X_train, y_train, X_val, y_val, "XGBoost", timeframe)
        
        gb_model = train_gradient_boosting(X_train, y_train, **best_params['GradientBoosting'])
        gb_mae, gb_rmse, gb_lower, gb_upper, gb_predictions = train_and_evaluate_model(
            gb_model, X_train, y_train, X_val, y_val, "GradientBoosting", timeframe)
        
        # LSTM Training
        try:
            X_train_reshaped = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
            X_val_reshaped = X_val.values.reshape((X_val.shape[0], 1, X_val.shape[1]))
            lstm_params = best_params['LSTM']
            lstm_model = KerasRegressor(
                model=build_lstm_model,
                model__input_shape=(1, X_train.shape[1]),
                model__units=lstm_params['model__units'],
                model__dropout=lstm_params['model__dropout'],
                batch_size=lstm_params['batch_size'],
                epochs=lstm_params['epochs'],
                verbose=0
            )
            lstm_model.fit(X_train_reshaped, y_train)
            lstm_predictions = lstm_model.predict(X_val_reshaped).flatten()
            lstm_lower, lstm_upper = compute_confidence_interval(lstm_predictions, confidence_level=0.75)
        except Exception as e:
            logger.error(f"Error training LSTM model on {tf_name} timeframe: {e}")
            lstm_predictions = np.full_like(rf_predictions, np.nan)

        if np.isnan(lstm_predictions).any():
            logger.warning("NaN values detected in LSTM predictions; substituting with zeros.")
            lstm_predictions = np.nan_to_num(lstm_predictions)

        # Model Stacking with Dynamic Weight Optimization
        try:
            min_length = min(len(rf_predictions), len(xgb_predictions), len(gb_predictions), len(lstm_predictions), len(y_val))
            rf_predictions = rf_predictions[:min_length]
            xgb_predictions = xgb_predictions[:min_length]
            gb_predictions = gb_predictions[:min_length]
            lstm_predictions = lstm_predictions[:min_length]
            y_val = y_val[:min_length]

            model_predictions = np.column_stack([rf_predictions, xgb_predictions, gb_predictions, lstm_predictions])
            best_weights = optimize_weights(model_predictions, y_val)
            save_best_weights(timeframe, best_weights)
            stacked_predictions = np.dot(model_predictions, best_weights)  # Weighted sum of predictions
            
            stacked_mae, stacked_rmse = evaluate_model(y_val, stacked_predictions)
            lower, upper = compute_confidence_interval(stacked_predictions, confidence_level=0.75)
            logger.info(f"Stacked Model - {timeframe}h: MAE: {stacked_mae}, RMSE: {stacked_rmse}, CI: ({lower}, {upper}), Weights: {best_weights}")

            # Append results to summary dictionary
            results_summary[f"{timeframe}h"] = {
                "RandomForest": {"MAE": rf_mae, "RMSE": rf_rmse, "Confidence Interval": (rf_lower, rf_upper)},
                "XGBoost": {"MAE": xgb_mae, "RMSE": xgb_rmse, "Confidence Interval": (xgb_lower, xgb_upper)},
                "GradientBoosting": {"MAE": gb_mae, "RMSE": gb_rmse, "Confidence Interval": (gb_lower, gb_upper)},
                "LSTM": {"Confidence Interval": (lstm_lower, lstm_upper)},
                "Stacked Model": {"MAE": stacked_mae, "RMSE": stacked_rmse, "Confidence Interval": (lower, upper), "Weights": best_weights.tolist()}
            }

            # Add to plot summary
            predictions_summary.append((stacked_predictions, y_val, f"{timeframe}-hour Stacked Model", lower, upper))
        
        except Exception as e:
            logger.error(f"Error in dynamic weight optimization for {tf_name} timeframe: {e}")

# Save results summary to JSON file
with open(results_file, 'w') as f:
    json.dump(results_summary, f, indent=4)
logger.info(f"Results summary saved to {results_file}")

# Plotting predictions and confidence intervals
try:
    plt.figure(figsize=(18, 12))
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']
    n_plots = len(predictions_summary)
    
    for idx, (predictions, actual, label, lower, upper) in enumerate(predictions_summary):
        plt.subplot((n_plots + 1) // 2, 2, idx + 1)
        plt.plot(range(len(actual)), actual, label="Actual Price", color='black', linewidth=1.5)
        plt.plot(range(len(predictions)), predictions, label=f"{label} Prediction", color=colors[idx % len(colors)], linewidth=1.2)
        plt.fill_between(range(len(actual)), lower, upper, color=colors[idx % len(colors)], alpha=0.1, label="Confidence Interval")
        plt.title(f"{label} vs Actual Price")
        plt.xlabel("Data Points")
        plt.ylabel("SPY Price")
        plt.legend()

    plt.tight_layout()
    plt.savefig("predictions_summary.png")
    plt.show()

except Exception as e:
    logger.error(f"Error in plotting predictions: {e}")
