import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
from models.train_models import train_and_save_models, cross_validate_models, stack_predictions, load_model_sklearn, load_keras_model, evaluate_model

# Load the historical data from pickle file
def load_historical_data():
    file_path = os.path.join('local_data', 'historical_data.pickle')
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    else:
        print(f"No historical data found at {file_path}")
        return None

# --- Backtest Function ---
def backtest_trading_strategy(X, y, spy_data, trade_type="credit_spread", trade_days=2, evaluation_days=1):
    trades = []
    actual_prices = []
    predicted_prices = []
    
    # Step 1: Train or load base models
    rf_model, xgb_model, gb_model, lstm_model = train_and_save_models(X, y)

    # Step 2: Cross-validate base models and get stacking predictions
    base_models = {'RandomForest': rf_model, 'XGBoost': xgb_model, 'GradientBoosting': gb_model}
    rf_predictions, xgb_predictions, gb_oof_predictions = cross_validate_models(X, y, base_models)

    # Step 3: Combine stacking predictions using adjustable weights
    stacking_predictions = stack_predictions(rf_predictions, xgb_predictions, gb_oof_predictions,
                                             rf_weight=0.4, xgb_weight=0.3, gb_weight=0.3)

    # Step 4: Generate LSTM predictions and ensure they are 1D
    lstm_predictions = lstm_model.predict(X.values.reshape((X.shape[0], 1, X.shape[1])))
    lstm_predictions = lstm_predictions.flatten()  # Ensure predictions are 1D

    # Step 5: Combine LSTM and stacking predictions
    combined_predictions = stack_predictions(stacking_predictions, lstm_predictions, rf_weight=0.5, xgb_weight=0.25, gb_weight=0.25)

    # Track exact prices and confidence intervals
    for i in range(len(combined_predictions) - trade_days - evaluation_days):
        open_price = spy_data['Close'].iloc[i + evaluation_days]
        close_price = spy_data['Close'].iloc[i + evaluation_days + trade_days]

        # Predicted movement -> transform to price prediction
        predicted_price = combined_predictions[i]

        # Confidence interval (e.g., 80% confidence interval)
        prediction_std = np.std([rf_predictions[i], xgb_predictions[i], gb_oof_predictions[i], lstm_predictions[i]])
        lower_bound = predicted_price - 1.28 * prediction_std  # 80% confidence interval
        upper_bound = predicted_price + 1.28 * prediction_std

        # Store actual and predicted prices for error calculation
        actual_prices.append(close_price)
        predicted_prices.append(predicted_price)

        # Store trade details
        trade = {
            'Trade Type': trade_type,
            'SPY Price (Open)': open_price,
            'SPY Price (Close)': close_price,
            'Predicted SPY Price (2 days)': predicted_price,
            '80% Confidence Lower Bound': lower_bound,
            '80% Confidence Upper Bound': upper_bound,
            'Trade Success': lower_bound <= close_price <= upper_bound
        }
        trades.append(trade)

    # Check the lengths of actual_prices and predicted_prices before calculating error
    print(f"Final actual_prices length: {len(actual_prices)}")
    print(f"Final predicted_prices length: {len(predicted_prices)}")

    # Calculate error metrics (e.g., MAE and RMSE)
    if len(actual_prices) == len(predicted_prices):
        mae = mean_absolute_error(actual_prices, predicted_prices)
        rmse = sqrt(mean_squared_error(actual_prices, predicted_prices))
        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"Root Mean Squared Error (RMSE): {rmse}")
    else:
        print(f"Error: Mismatched lengths - actual_prices: {len(actual_prices)}, predicted_prices: {len(predicted_prices)}")
    
    # Create DataFrame of trades
    trades_df = pd.DataFrame(trades)
    
    return trades_df

def run_backtest():
    # Step 1: Load the historical data from the pickle file
    data = load_historical_data()
    if data is None:
        return

    # Step 2: Extract X, y, and spy_data from the loaded historical data
    # X is a feature matrix with technical indicators, sentiment, and closing prices
    X = data['spy_data']['1h'][['MACD', 'RSI', '%K', '%D', 'ATR', 'PlusDI', 'MinusDI', 'EMA9', 'EMA21', 'MFI', 'Close']]

    # y represents the sentiment or target variable
    spy_data = data['spy_data']['1h']  # SPY data for 1-hour timeframe
    spy_data['Sentiment'] = (spy_data['EMA9'] > spy_data['EMA21']).astype(int)  # Binary sentiment calculation
    y = spy_data['Sentiment']  # Use calculated sentiment as target variable

    # Step 3: Run the backtest
    trades_df = backtest_trading_strategy(X, y, spy_data, 
                                          trade_type="credit_spread", 
                                          trade_days=2, 
                                          evaluation_days=1)

    # Step 4: Print results (first few rows)
    print(trades_df.head())

if __name__ == '__main__':
    run_backtest()
