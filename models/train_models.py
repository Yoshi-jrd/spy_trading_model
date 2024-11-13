import os
import logging
import numpy as np
import json
from data_preparation import load_data, preprocess_data, prepare_lstm_data
from model_definitions import build_lstm, build_random_forest, build_gradient_boosting, build_xgboost, build_extra_trees, build_catboost, build_ridge
from evaluate_model import evaluate_model
from dynamic_weight_optimizer import optimize_weights
import matplotlib.pyplot as plt
from data_processing.data_augmentation import create_year_of_15min_data  # New import

# Set the path to config.json based on the current file's location
config_path = os.path.join(os.path.dirname(__file__), 'config.json')
with open(config_path, 'r') as f:
    config = json.load(f)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def plot_predictions_vs_actual(predictions, actual_prices, interval_name):
    """
    Plot model predictions against actual SPY prices for verification.

    Parameters:
    - predictions: Array of predicted prices from the model.
    - actual_prices: Array of actual SPY prices for the same intervals.
    - interval_name: String representing the prediction interval (e.g., "24h").
    """
    plt.figure(figsize=(10, 5))
    plt.plot(actual_prices, label='Actual Prices', color='blue')
    plt.plot(predictions, label='Predicted Prices', color='red')
    plt.title(f'Predicted vs. Actual Prices - {interval_name} Interval')
    plt.xlabel('Time Step')
    plt.ylabel('SPY Price')
    plt.legend()
    plt.show()

def train_and_stack_models(spy_data, interval_name, interval_steps):
    """Trains and stacks models with LSTM for the specified interval with adjusted market hours."""
    # Prepare LSTM data with adjusted interval steps
    X_train_seq, y_train_seq, close_scaler = prepare_lstm_data(spy_data, interval_steps=interval_steps)

    # Build and train LSTM model
    lstm_model = build_lstm(sequence_length=config['lstm']['sequence_length'], feature_count=X_train_seq.shape[2])
    lstm_model.fit(X_train_seq, y_train_seq, epochs=config['lstm']['epochs'], batch_size=config['lstm']['batch_size'], verbose=0)
    
    # LSTM Predictions
    lstm_predictions = close_scaler.inverse_transform(lstm_model.predict(X_train_seq)).flatten()

    # Convert sequence data to a suitable format for other models
    X_train_2d = X_train_seq[:, -1, :]  # Extract last timestep

    # Train other models
    rf_model = build_random_forest().fit(X_train_2d, y_train_seq.ravel())
    gb_model = build_gradient_boosting().fit(X_train_2d, y_train_seq.ravel())
    xgb_model = build_xgboost().fit(X_train_2d, y_train_seq.ravel())
    et_model = build_extra_trees().fit(X_train_2d, y_train_seq.ravel())
    cb_model = build_catboost().fit(X_train_2d, y_train_seq.ravel())

    # Stack predictions
    base_predictions = [
        rf_model.predict(X_train_2d),
        gb_model.predict(X_train_2d),
        xgb_model.predict(X_train_2d),
        et_model.predict(X_train_2d),
        cb_model.predict(X_train_2d),
        lstm_predictions
    ]
    predictions_matrix = np.column_stack(base_predictions)

    # Meta-model stacking with Ridge
    ridge_meta_model = build_ridge()
    ridge_meta_model.fit(predictions_matrix, y_train_seq)

    # Calculate metrics
    stacked_predictions = ridge_meta_model.predict(predictions_matrix)
    mae, rmse, avg_diff = evaluate_model(y_train_seq, stacked_predictions)
    logger.info(f"{interval_name} - MAE: {mae}, RMSE: {rmse}, Avg Difference: {avg_diff}")

    return stacked_predictions, y_train_seq

def main():
    """Main function to train and evaluate models for each interval."""
    spy_data = create_year_of_15min_data()  # Use external function to create a year of 15-minute data
    spy_data = preprocess_data(spy_data)

    # Loop over adjusted intervals based on market hours
    for interval_name, interval_steps in config['intervals_market_hours'].items():
        logger.info(f"Training for interval: {interval_name} with steps: {interval_steps}")
        predictions, actual_prices = train_and_stack_models(spy_data, interval_name, interval_steps)

        # Plot predictions vs actual prices for verification
        plot_predictions_vs_actual(predictions, actual_prices, interval_name)

if __name__ == "__main__":
    main()
