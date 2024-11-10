# backtest.py
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from models.train_models import load_best_params, evaluate_model
from model_utils import compute_confidence_interval
from data.data_loader import load_existing_data

# Load historical data
data = load_existing_data()
spy_data = data['spy_data']

# Define the forward timeframes (in hours) to predict
forward_hours = [12, 24, 36, 48, 72, 96]

# Main Backtest Functionality
for timeframe, spy_df in spy_data.items():
    print(f"\nRunning backtest for {timeframe} timeframe")

    # Extract features and initialize results storage
    X = spy_df[['MACD_Histogram', 'RSI', 'UpperBand', 'LowerBand', 'ATR', 'ADX', 'EMA9', 'EMA21', 'Impulse_Color']].values
    predictions_summary = {}

    # Loop through each forward timeframe and predict
    for hours in forward_hours:
        y = spy_df['Close'].shift(-hours).dropna().values  # Shift target to create forward prediction
        X_adj = X[:len(y)]  # Align X with shifted target

        # Split data into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X_adj, y, test_size=0.2, random_state=42)

        # Load model, predict, and calculate confidence intervals
        predictions, lower_bound, upper_bound = make_predictions(X_val, best_params)
        mae, rmse = evaluate_model(y_val, predictions)
        
        within_conf = np.mean((y_val >= lower_bound) & (y_val <= upper_bound))
        predictions_summary[hours] = {
            'predictions': predictions,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'actual': y_val,
            'mae': mae,
            'rmse': rmse,
            'within_confidence': within_conf
        }

    # Plotting results for each forward timeframe
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"Multi-Timeframe Forward Predictions Backtest for {timeframe}")

    for idx, hours in enumerate(forward_hours):
        ax = axs[idx // 3, idx % 3]
        summary = predictions_summary[hours]
        
        ax.plot(summary['actual'], label="Actual Price", color="black")
        ax.plot(summary['predictions'], label=f"Predicted Price ({hours}h)", color="blue")
        ax.fill_between(range(len(summary['predictions'])), summary['lower_bound'], summary['upper_bound'], color="blue", alpha=0.2, label="75% Confidence Interval")
        
        ax.set_title(f"{hours}-Hour Forward Prediction\nMAE: {summary['mae']:.2f}, RMSE: {summary['rmse']:.2f}, % Within CI: {summary['within_confidence']*100:.2f}%")
        ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
