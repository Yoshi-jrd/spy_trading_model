import logging
from train_models import train_and_evaluate_model, load_best_params, save_best_params
from data.data_loader import load_existing_data
from evaluate_model import evaluate_model, evaluate_multiple_timeframes
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from scikeras.wrappers import KerasRegressor
from lstm_model import build_lstm_model
import numpy as np
from sklearn.linear_model import LinearRegression
from model_utils import compute_confidence_interval
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def run_evaluation_workflow():
    # Step 1: Load data and best parameters
    data = load_existing_data()
    spy_data = data['spy_data']
    best_params = load_best_params()
    
    # Initialize a list to store evaluation results across timeframes
    predictions_summary = []
    timeframes = [12, 24, 36, 48, 72, 96]

    # Step 2: Train and Evaluate Models Across Timeframes
    for timeframe in timeframes:
        for tf_name, spy_df in spy_data.items():
            logger.info(f"Evaluating models for {tf_name} timeframe, {timeframe}-hour forward prediction")

            # Define features and target
            selected_features = ['MACD_Histogram', 'RSI', 'UpperBand', 'LowerBand', 'ATR', 'ADX', 'EMA9', 'EMA21', 'Impulse_Color']
            X = spy_df[selected_features]
            y = spy_df['Close'].shift(-timeframe).dropna()
            X = X.iloc[:-timeframe]
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

            # Initialize models with best parameters
            rf_model = RandomForestRegressor(**best_params['RandomForest'])
            xgb_model = XGBRegressor(**best_params['XGBoost'])
            gb_model = GradientBoostingRegressor(**best_params['GradientBoosting'])
            lstm_model = KerasRegressor(
                model=build_lstm_model,
                model__input_shape=(1, X_train.shape[1]),
                **best_params['LSTM']
            )

            # Train and evaluate each model
            rf_mae, rf_rmse, rf_lower, rf_upper, rf_predictions = train_and_evaluate_model(
                rf_model, X_train, y_train, X_val, y_val, "RandomForest", timeframe)
            xgb_mae, xgb_rmse, xgb_lower, xgb_upper, xgb_predictions = train_and_evaluate_model(
                xgb_model, X_train, y_train, X_val, y_val, "XGBoost", timeframe)
            gb_mae, gb_rmse, gb_lower, gb_upper, gb_predictions = train_and_evaluate_model(
                gb_model, X_train, y_train, X_val, y_val, "GradientBoosting", timeframe)

            # LSTM evaluation with reshaped data
            X_train_reshaped = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
            X_val_reshaped = X_val.values.reshape((X_val.shape[0], 1, X_val.shape[1]))
            lstm_model.fit(X_train_reshaped, y_train)
            lstm_predictions = lstm_model.predict(X_val_reshaped).flatten()
            lstm_mae, lstm_rmse = evaluate_model(y_val, lstm_predictions)
            lstm_lower, lstm_upper = compute_confidence_interval(lstm_predictions, confidence_level=0.75)

            # Log model-specific performance
            logger.info(f"RandomForest - {timeframe}h: MAE: {rf_mae}, RMSE: {rf_rmse}, CI: ({rf_lower}, {rf_upper})")
            logger.info(f"XGBoost - {timeframe}h: MAE: {xgb_mae}, RMSE: {xgb_rmse}, CI: ({xgb_lower}, {xgb_upper})")
            logger.info(f"GradientBoosting - {timeframe}h: MAE: {gb_mae}, RMSE: {gb_rmse}, CI: ({gb_lower}, {gb_upper})")
            logger.info(f"LSTM - {timeframe}h: MAE: {lstm_mae}, RMSE: {lstm_rmse}, CI: ({lstm_lower}, {lstm_upper})")

            # Step 3: Stack model predictions and evaluate
            min_length = min(len(rf_predictions), len(xgb_predictions), len(gb_predictions), len(lstm_predictions))
            meta_X_train = np.column_stack([
                rf_predictions[:min_length],
                xgb_predictions[:min_length],
                gb_predictions[:min_length],
                lstm_predictions[:min_length]
            ])
            meta_y_train = y_val[:min_length]
            meta_model = LinearRegression().fit(meta_X_train, meta_y_train)
            stacked_predictions = meta_model.predict(meta_X_train)
            stacked_mae, stacked_rmse = evaluate_model(meta_y_train, stacked_predictions)
            lower, upper = compute_confidence_interval(stacked_predictions, confidence_level=0.75)

            # Store evaluation results for the current timeframe
            predictions_summary.append((stacked_predictions, meta_y_train, "Stacked Model", timeframe))

            # Log stacked model performance
            logger.info(f"Stacked Model - {timeframe}h: MAE: {stacked_mae}, RMSE: {stacked_rmse}, Confidence Interval: ({lower}, {upper})")

    # Step 4: Summarize evaluation results across all timeframes
    summary = evaluate_multiple_timeframes(predictions_summary)
    logger.info(f"Summary across timeframes: {summary}")

if __name__ == "__main__":
    run_evaluation_workflow()
