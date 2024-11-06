# Add the parent directory to the system path
import os
import sys
import pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import your models and utilities
from random_forest_model import train_random_forest
from xgboost_model import train_xgboost
from gradient_boosting_model import train_gradient_boosting
from lstm_model import build_lstm_model, train_lstm  # Import the LSTM build function
from model_utils import evaluate_model  # Import the evaluation function
from data.data_loader import load_existing_data  # Import data from local pickle data storage.
from data.indicator_calculator import calculate_indicators
from sklearn.ensemble import RandomForestRegressor
from scikeras.wrappers import KerasRegressor

# Define paths for the best parameters file
best_params_file = "best_params.pkl"

# Helper function to load best parameters
def load_best_params():
    if os.path.exists(best_params_file):
        with open(best_params_file, 'rb') as f:
            return pickle.load(f)
    else:
        # Default best parameters if the file doesn't exist
        return {
            'RandomForest': {'n_estimators': 200, 'max_depth': 20, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt'},
            'XGBoost': {'n_estimators': 100, 'learning_rate': 0.2, 'max_depth': 3, 'min_child_weight': 3, 'gamma': 0},
            'GradientBoosting': {'n_estimators': 300, 'learning_rate': 0.1, 'max_depth': 3, 'min_samples_split': 10, 'min_samples_leaf': 1}
        }

# Helper function to save best parameters
def save_best_params(best_params):
    with open(best_params_file, 'wb') as f:
        pickle.dump(best_params, f)

# Initialize a dictionary to store the best RMSE for each model
best_rmse = {
    'RandomForest': float('inf'),  # Initially set to infinity, so any RMSE will be an improvement
    'XGBoost': float('inf'),
    'GradientBoosting': float('inf'),
    'LSTM': float('inf')
}

# Function to compare and update the best parameters dynamically
def update_best_params(model_name, current_rmse, current_params):
    global best_rmse, best_params

    # Check if the current model's RMSE is better than the previously stored best RMSE
    if current_rmse < best_rmse[model_name]:
        print(f"New best RMSE for {model_name}: {current_rmse}. Updating best parameters...")
        best_rmse[model_name] = current_rmse  # Update the best RMSE
        best_params[model_name] = current_params  # Update the best parameters

# Load the best parameters
best_params = load_best_params()

# Load the data from the pickle file
data = load_existing_data()

# Accessing SPY data (this contains multiple timeframes like '5m', '15m', '1h', '1d')
spy_data = data['spy_data']

# Loop through each timeframe in spy_data
for timeframe, spy_df in spy_data.items():
    print(f"\n--- Training models on {timeframe} timeframe ---")

    # Step 1: Calculate the indicators and add them to the dataframe
    spy_df = calculate_indicators(spy_df)

    # Step 2: Prepare the features (X_train) and target (y_train)
    X_train = spy_df.drop(columns=['Close', 'datetime'])  # Features
    y_train = spy_df['Close']  # Target

    # -----------------------------
    # Train and Hypertune each model
    # -----------------------------

    # RandomForest Model with best parameters
    print(f"Training RandomForest Model on {timeframe} timeframe...")
    rf_model = RandomForestRegressor(**best_params['RandomForest'])
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_train)
    rf_mae, rf_rmse = evaluate_model(y_train, rf_predictions)
    print(f"RandomForest - {timeframe} - MAE: {rf_mae}, RMSE: {rf_rmse}")

    # Update best parameters for RandomForest if the new RMSE is better
    update_best_params('RandomForest', rf_rmse, rf_model.get_params())

    # XGBoost Model with best parameters
    print(f"Training XGBoost Model on {timeframe} timeframe...")
    xgb_model = train_xgboost(X_train, y_train, **best_params['XGBoost'])
    xgb_predictions = xgb_model.predict(X_train)
    xgb_mae, xgb_rmse = evaluate_model(y_train, xgb_predictions)
    print(f"XGBoost - {timeframe} - MAE: {xgb_mae}, RMSE: {xgb_rmse}")

    # Update best parameters for XGBoost if the new RMSE is better
    update_best_params('XGBoost', xgb_rmse, xgb_model.get_params())

    # Gradient Boosting Model with best parameters
    print(f"Training Gradient Boosting Model on {timeframe} timeframe...")
    gb_model = train_gradient_boosting(X_train, y_train, **best_params['GradientBoosting'])
    gb_predictions = gb_model.predict(X_train)
    gb_mae, gb_rmse = evaluate_model(y_train, gb_predictions)
    print(f"GradientBoosting - {timeframe} - MAE: {gb_mae}, RMSE: {gb_rmse}")

    # Update best parameters for GradientBoosting if the new RMSE is better
    update_best_params('GradientBoosting', gb_rmse, gb_model.get_params())

    # LSTM Model with fixed parameter passing
    print(f"Training LSTM Model on {timeframe} timeframe...")
    lstm_model = KerasRegressor(build_fn=build_lstm_model, dropout=0.2, epochs=50, batch_size=32, verbose=0)
    lstm_model.fit(X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1])), y_train)
    lstm_predictions = lstm_model.predict(X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1])))
    lstm_predictions = lstm_predictions.flatten()  # Flatten the LSTM predictions
    lstm_mae, lstm_rmse = evaluate_model(y_train, lstm_predictions)
    print(f"LSTM - {timeframe} - MAE: {lstm_mae}, RMSE: {lstm_rmse}")

    # Update best parameters for LSTM if the new RMSE is better
    update_best_params('LSTM', lstm_rmse, lstm_model.get_params())

# Save updated best parameters at the end of the run
save_best_params(best_params)
