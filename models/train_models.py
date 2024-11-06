# Add the parent directory to the system path
import os
import sys
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import models and utilities
from random_forest_model import train_random_forest
from xgboost_model import train_xgboost
from gradient_boosting_model import train_gradient_boosting
from lstm_model import build_lstm_model, train_lstm
from model_utils import evaluate_model
from data.data_loader import load_existing_data
from data.indicator_calculator import calculate_indicators
from sklearn.ensemble import RandomForestRegressor
from scikeras.wrappers import KerasRegressor

# Define paths for the best parameters file and model save path
best_params_file = "best_params.pkl"
model_save_dir = "saved_models"

# Ensure the model save directory exists
os.makedirs(model_save_dir, exist_ok=True)

# Helper function to load best parameters
def load_best_params():
    if os.path.exists(best_params_file):
        try:
            with open(best_params_file, 'rb') as f:
                return pickle.load(f)
        except EOFError:
            print("Warning: best_params.pkl is empty or corrupted. Resetting to default parameters.")
    # Default parameters if file is missing or corrupted
    return {
        'RandomForest': {'n_estimators': 200, 'max_depth': 20, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt'},
        'XGBoost': {'n_estimators': 100, 'learning_rate': 0.2, 'max_depth': 3, 'min_child_weight': 3, 'gamma': 0},
        'GradientBoosting': {'n_estimators': 300, 'learning_rate': 0.1, 'max_depth': 3, 'min_samples_split': 10, 'min_samples_leaf': 1}
    }

# Helper function to save best parameters
def save_best_params(best_params):
    with open(best_params_file, 'wb') as f:
        pickle.dump(best_params, f)

# Load best parameters and initialize best RMSE tracker
best_params = load_best_params()
best_rmse = {'RandomForest': float('inf'), 'XGBoost': float('inf'), 'GradientBoosting': float('inf'), 'LSTM': float('inf')}

# Load the data
data = load_existing_data()
spy_data = data['spy_data']

# Function to update best parameters dynamically and save models
def update_best_params_and_save_model(model_name, model, current_rmse, current_params):
    global best_rmse, best_params
    if current_rmse < best_rmse[model_name]:
        print(f"New best RMSE for {model_name}: {current_rmse}. Updating best parameters and saving model...")
        best_rmse[model_name] = current_rmse
        best_params[model_name] = current_params
        model_save_path = os.path.join(model_save_dir, f"{model_name}_{timeframe}_model.pkl")
        with open(model_save_path, 'wb') as f:
            pickle.dump(model, f)

# Training loop for each timeframe in SPY data
for timeframe, spy_df in spy_data.items():
    print(f"\n--- Training models on {timeframe} timeframe ---")
    spy_df = calculate_indicators(spy_df)

    # Convert 'Impulse_Color' to numeric values for model training
    color_mapping = {'red': -1, 'green': 1, 'gray': 0}
    spy_df['Impulse_Color'] = spy_df['Impulse_Color'].map(color_mapping)

    # Prepare features and target, select only numeric columns
    X = spy_df.drop(columns=['Close', 'datetime']).select_dtypes(include=[np.number])  # Only numeric columns
    y = spy_df['Close']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Debugging: Check if X_train contains only numeric columns
    print(f"Columns in X_train after color conversion and numeric filtering: {X_train.columns}")

    # -----------------------------
    # Train and Hypertune each model
    # -----------------------------

    # RandomForest Model
    print(f"Training RandomForest Model on {timeframe} timeframe...")
    rf_model = RandomForestRegressor(**best_params['RandomForest'])
    rf_model.fit(X_train, y_train)
    rf_val_predictions = rf_model.predict(X_val)
    rf_mae, rf_rmse = evaluate_model(y_val, rf_val_predictions)
    print(f"RandomForest - {timeframe} - MAE: {rf_mae}, RMSE: {rf_rmse}")
    update_best_params_and_save_model('RandomForest', rf_model, rf_rmse, rf_model.get_params())

    # XGBoost Model
    print(f"Training XGBoost Model on {timeframe} timeframe...")
    xgb_model = train_xgboost(X_train, y_train, **best_params['XGBoost'])
    xgb_val_predictions = xgb_model.predict(X_val)
    xgb_mae, xgb_rmse = evaluate_model(y_val, xgb_val_predictions)
    print(f"XGBoost - {timeframe} - MAE: {xgb_mae}, RMSE: {xgb_rmse}")
    update_best_params_and_save_model('XGBoost', xgb_model, xgb_rmse, xgb_model.get_params())

    # Gradient Boosting Model
    print(f"Training Gradient Boosting Model on {timeframe} timeframe...")
    gb_model = train_gradient_boosting(X_train, y_train, **best_params['GradientBoosting'])
    gb_val_predictions = gb_model.predict(X_val)
    gb_mae, gb_rmse = evaluate_model(y_val, gb_val_predictions)
    print(f"GradientBoosting - {timeframe} - MAE: {gb_mae}, RMSE: {gb_rmse}")
    update_best_params_and_save_model('GradientBoosting', gb_model, gb_rmse, gb_model.get_params())

# LSTM Model
print(f"Training LSTM Model on {timeframe} timeframe...")
try:
    # Determine input shape
    X_train_reshaped = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_val_reshaped = X_val.values.reshape((X_val.shape[0], 1, X_val.shape[1]))
    input_shape = (X_train_reshaped.shape[1], X_train_reshaped.shape[2])

    # Debugging shapes
    print(f"X_train reshaped for LSTM: {X_train_reshaped.shape}")
    print(f"X_val reshaped for LSTM: {X_val_reshaped.shape}")

    # Initialize LSTM model with input shape as a keyword argument
    lstm_model = KerasRegressor(
        model=build_lstm_model,
        model__input_shape=input_shape,  # Pass input_shape as a keyword argument to the model
        epochs=50,
        batch_size=32,
        verbose=0
    )

    # Fit the model
    lstm_model.fit(X_train_reshaped, y_train)
    lstm_val_predictions = lstm_model.predict(X_val_reshaped).flatten()

    # Evaluate and log LSTM performance
    lstm_mae, lstm_rmse = evaluate_model(y_val, lstm_val_predictions)
    print(f"LSTM - {timeframe} - MAE: {lstm_mae}, RMSE: {lstm_rmse}")
    update_best_params_and_save_model('LSTM', lstm_model, lstm_rmse, lstm_model.get_params())
except Exception as e:
    print(f"Error training LSTM model on {timeframe}: {e}")

# Save best parameters after training
save_best_params(best_params)
