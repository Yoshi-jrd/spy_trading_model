import numpy as np
import pandas as pd
from sklearn.utils import resample
from datetime import timedelta
import random
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.data_loader import load_data  # Use the data loader here
import warnings
warnings.filterwarnings("ignore")
from strategies.simulate_trade import simulate_trade
from run_models_on_timeframes import train_lstm, run_stacking_model, load_data_for_timeframes
from strategies.apply_risk_management import apply_risk_management
from summarize_backtest import summarize_backtest
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# Synthetic Data Randomization
def randomize_synthetic_data(X, y, num_samples=1000):
    """
    Generates randomized synthetic data to simulate different market conditions.
    """
    X_synthetic, y_synthetic = resample(X, y, replace=True, n_samples=num_samples, random_state=random.randint(0, 1000))
    return X_synthetic, y_synthetic

# --- Full Backtest Function ---
def backtest_trading_strategy(X, y, base_models, lstm_model, spy_data, trade_type="credit_spread", trade_days=2, risk_management=None, evaluation_days=1):
    """
    Backtests an options trading strategy using a given model on historical and synthetic data.
    """
    trades = []

    # Add a print statement to show the first few rows of spy_data
    print("SPY Data Passed into Backtest (first 5 rows):")
    print(spy_data.head())  
    
    print("SPY Data Columns:")
    print(spy_data.columns)

    # Ensure 'Datetime' or 'Date' column exists in spy_data
    if 'Datetime' in spy_data.columns:
        dates = pd.to_datetime(spy_data['Datetime'])
        print("Using 'Datetime' column for date parsing.")
    elif 'Date' in spy_data.columns:
        dates = pd.to_datetime(spy_data['Date'])
        print("Using 'Date' column for date parsing.")
    else:
        print("Columns in spy_data:", spy_data.columns)  # This will print all available columns if no 'Datetime' or 'Date'
        raise KeyError("'Datetime' or 'Date' column not found in spy_data")
    
    spy_prices = spy_data['Close']  # Use 'Close' prices from the selected SPY timeframe data

    # Handle missing data by interpolation
    spy_data.interpolate(method='time', inplace=True)

    # Reshape data for LSTM if it's being used
    X_reshaped = X.to_numpy().reshape((X.shape[0], 1, X.shape[1])) if lstm_model else X

    # Generate predictions from base models and LSTM
    rf_predictions = base_models['RandomForest'].predict_proba(X)[:, 1].reshape(-1, 1)
    xgb_predictions = base_models['XGBoost'].predict_proba(X)[:, 1].reshape(-1, 1)
    gb_predictions = base_models['GradientBoosting'].predict_proba(X)[:, 1].reshape(-1, 1)

    lstm_predictions = lstm_model.predict(X_reshaped).flatten() if lstm_model else np.zeros(len(X))

    # Combine predictions using a weighted approach
    predictions = (0.4 * rf_predictions + 0.3 * xgb_predictions + 0.2 * gb_predictions + 0.1 * lstm_predictions)

    for i in range(len(predictions) - trade_days - evaluation_days):
        open_price = spy_prices.iloc[i + evaluation_days]
        expiry_date = dates.iloc[i + evaluation_days] + timedelta(days=trade_days)

        prediction_confidence = predictions[i]
        prediction_error = abs(prediction_confidence - y.iloc[i])

        # Simulate a trade based on the prediction
        close_price = spy_prices.iloc[i + evaluation_days + trade_days]
        trade_outcome, profit_loss, risk_reward = simulate_trade(open_price, close_price, trade_type, prediction_confidence)

        # Apply risk management
        if risk_management:
            profit_loss, trade_outcome = apply_risk_management(profit_loss, risk_management)

        # Collect trade details
        trade = {
            'Trade Type': trade_type,
            'Strike Price (Open)': open_price,
            'Strike Price (Close)': close_price,
            'Open Date': dates.iloc[i + evaluation_days],
            'Expiry Date': expiry_date,
            'Days Evaluated': evaluation_days,
            'Risk/Reward': risk_reward,
            'Profit/Loss': profit_loss,
            'SPY Price (Open)': open_price,
            'SPY Price (Close)': close_price,
            'Successful Trade': trade_outcome,
            'Prediction Confidence': prediction_confidence,
            'Prediction Error': prediction_error
        }
        trades.append(trade)

    # Create a summary dataframe
    trades_df = pd.DataFrame(trades)
    return trades_df


# --- Run Full Backtest ---
def run_backtest():
    """
    Main function to run backtest on historical and synthetic data.
    """
    timeframes = ['5m', '15m', '1h', '1d']
    data_dict = load_data_for_timeframes(timeframes)

    for timeframe, (X, y, spy_data) in data_dict.items():
        print(f"Backtesting for {timeframe} timeframe...")
        print(f"Spy data structure for {timeframe} timeframe:\n")
        print(spy_data.head())  # Print first few rows to confirm structure

        # Initialize base models
        base_models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', verbosity=0),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }

        # Randomize synthetic data
        X_synthetic, y_synthetic = randomize_synthetic_data(X, y)

        # Train base models
        print(f"Training base models for {timeframe} timeframe...")
        for model_name, model in base_models.items():
            model.fit(X_synthetic, y_synthetic)  # Train each model on the synthetic data

        # Train the LSTM model if the timeframe is 1h or 1d
        lstm_model = None
        if timeframe in ['1h', '1d']:
            print(f"Training LSTM model for {timeframe} timeframe...")
            lstm_model = train_lstm(X_synthetic, y_synthetic)

        # Run backtest with risk management
        risk_management = {'stop_loss': -0.1, 'take_profit': 0.2}  # Example risk management
        trades_df = backtest_trading_strategy(X_synthetic, y_synthetic, base_models, lstm_model, spy_data, trade_days=2, risk_management=risk_management, evaluation_days=3)

        # Summarize the backtest
        summary_df = summarize_backtest(trades_df)
        print(f"Summary for {timeframe}:\n{summary_df}\n")
        print(f"Trade Details for {timeframe}:\n{trades_df.head()}\n")


# --- Execute the backtest ---
if __name__ == '__main__':
    run_backtest()
