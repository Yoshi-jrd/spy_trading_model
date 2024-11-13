import os
import numpy as np
import pandas as pd
import json
import logging
from data_preparation import load_data, preprocess_data, prepare_lstm_data
from model_definitions import build_lstm, build_random_forest, build_gradient_boosting, build_xgboost, build_extra_trees, build_catboost, build_ridge
from evaluate_model import evaluate_model

# Set up config path dynamically
config_path = os.path.join(os.path.dirname(__file__), 'config.json')
with open(config_path, 'r') as f:
    config = json.load(f)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Out-of-Sample Evaluation Function
def out_of_sample_evaluation(test_data, intervals, update_frequency=15):
    """Evaluate model accuracy on out-of-sample data with walk-forward testing."""
    
    results_summary = {interval: {"predictions": [], "actuals": []} for interval in intervals}
    
    # Initialize walk-forward testing loop
    sequence_length = config['lstm']['sequence_length']
    for start_idx in range(0, len(test_data) - max(intervals.values()), update_frequency):
        
        # Prepare training data for prediction at current start index
        current_data = test_data.iloc[:start_idx + sequence_length]
        if len(current_data) < sequence_length:
            continue
        
        # Generate LSTM sequence and other model inputs
        X_train_seq, _, close_scaler = prepare_lstm_data(current_data)
        X_last_timestep = X_train_seq[-1, -1, :]  # Last timestep features for traditional models
        
        # Train and predict using each model
        lstm_model = build_lstm(sequence_length, X_train_seq.shape[2])
        lstm_model.fit(X_train_seq, close_scaler.inverse_transform(X_train_seq[:, -1, 0].reshape(-1, 1)), epochs=config['lstm']['epochs'], batch_size=config['lstm']['batch_size'], verbose=0)
        lstm_prediction = close_scaler.inverse_transform(lstm_model.predict(X_train_seq[-1:])).flatten()
        
        # Initialize traditional models with only the last timestep
        models = {
            "rf": build_random_forest(),
            "gb": build_gradient_boosting(),
            "xgb": build_xgboost(),
            "et": build_extra_trees(),
            "cb": build_catboost(),
            "ridge": build_ridge()
        }
        
        # Record each interval’s predictions
        for interval_name, interval_steps in intervals.items():
            # Get target data for the future interval
            target_idx = start_idx + interval_steps
            if target_idx >= len(test_data):
                continue
            
            actual_price = test_data['Close_15m'].iloc[target_idx]
            
            # Stack predictions from traditional models with LSTM prediction
            base_predictions = []
            for name, model in models.items():
                if name == "ridge":
                    continue  # Ridge will use stacked predictions
                model.fit(X_last_timestep.reshape(1, -1), [actual_price])  # Use current price for supervised signal
                base_predictions.append(model.predict(X_last_timestep.reshape(1, -1))[0])
            
            # Add LSTM prediction
            base_predictions.append(lstm_prediction[0])
            
            # Stack predictions for Ridge meta-model
            ridge_meta_model = build_ridge()
            stacked_prediction = ridge_meta_model.fit(np.array(base_predictions).reshape(1, -1), [actual_price]).predict(np.array(base_predictions).reshape(1, -1))[0]
            
            # Record prediction and actual price for this interval
            results_summary[interval_name]["predictions"].append(stacked_prediction)
            results_summary[interval_name]["actuals"].append(actual_price)
            
            logger.info(f"{interval_name} - Prediction: {stacked_prediction}, Actual: {actual_price}")
    
    # Calculate and summarize errors for each interval
    for interval_name, results in results_summary.items():
        if len(results["predictions"]) > 0:
            predictions = np.array(results["predictions"])
            actuals = np.array(results["actuals"])
            mae, rmse, avg_diff = evaluate_model(actuals, predictions)
            logger.info(f"Out-of-Sample {interval_name} - MAE: {mae}, RMSE: {rmse}, Avg Difference: {avg_diff}")
        else:
            logger.warning(f"No predictions recorded for interval {interval_name}.")
    
    return results_summary

# Main function for running out-of-sample evaluation
def main():
    # Load and preprocess test data
    test_data = load_data()
    test_data = preprocess_data(test_data)
    
    # Prediction intervals
    intervals = {
        '24h': 96,
        '48h': 192,
        '72h': 288,
        '96h': 384,
        '168h': 672
    }
    
    # Run out-of-sample evaluation
    results_summary = out_of_sample_evaluation(test_data, intervals)

if __name__ == "__main__":
    main()
