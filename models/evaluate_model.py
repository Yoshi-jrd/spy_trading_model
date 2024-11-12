# evaluate_model.py

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def evaluate_model(y_true, y_pred):
    """
    Evaluate model performance using MAE, RMSE, and Average Difference.
    
    Args:
        y_true (np.array or pd.Series): Actual values.
        y_pred (np.array or pd.Series): Predicted values.
        
    Returns:
        tuple: MAE, RMSE, and Average Difference values.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    avg_diff = np.mean(np.abs(y_true - y_pred))
    
    logger.info(f"Evaluation Metrics - MAE: {mae}, RMSE: {rmse}, Average Difference: {avg_diff}")
    return mae, rmse, avg_diff

def compute_confidence_interval(predictions, confidence_level=0.75):
    """
    Compute the confidence interval for predictions.
    
    Args:
        predictions (np.array): Array of predictions.
        confidence_level (float): Confidence level (e.g., 0.75 for 75% confidence interval).
        
    Returns:
        tuple: Lower and upper bounds of the confidence interval.
    """
    mean_pred = np.mean(predictions)
    std_dev = np.std(predictions)
    margin_of_error = std_dev * confidence_level
    
    lower_bound = mean_pred - margin_of_error
    upper_bound = mean_pred + margin_of_error
    
    logger.info(f"Confidence Interval ({confidence_level * 100}%): ({lower_bound}, {upper_bound})")
    return lower_bound, upper_bound

def evaluate_multiple_timeframes(predictions_summary):
    """
    Evaluate model predictions across multiple timeframes and output summarized results.
    
    Args:
        predictions_summary (list): A list containing tuples with (predictions, actual values, model_name, timeframe).
        
    Returns:
        dict: A summary of MAE, RMSE, Average Difference, and confidence intervals for each model and timeframe.
    """
    summary = {}
    for predictions, y_true, model_name, timeframe in predictions_summary:
        mae, rmse, avg_diff = evaluate_model(y_true, predictions)
        lower, upper = compute_confidence_interval(predictions)
        
        summary_key = f"{model_name}_{timeframe}h"
        summary[summary_key] = {
            "MAE": mae,
            "RMSE": rmse,
            "Average Difference": avg_diff,
            "Confidence Interval": (lower, upper)
        }
        
        logger.info(f"{summary_key} - MAE: {mae}, RMSE: {rmse}, Average Difference: {avg_diff}, Confidence Interval: ({lower}, {upper})")
    
    return summary

# For testing or as a standalone script
if __name__ == "__main__":
    import numpy as np

    # Simulate actual values and predictions
    y_true = np.array([100, 102, 98, 101, 99, 103, 105])
    y_pred = np.array([101, 101, 97, 100, 100, 102, 106])

    # Evaluate single set of predictions
    mae, rmse, avg_diff = evaluate_model(y_true, y_pred)
    lower, upper = compute_confidence_interval(y_pred)
    
    # Output example for logging
    logger.info(f"Single Evaluation - MAE: {mae}, RMSE: {rmse}, Average Difference: {avg_diff}, CI: ({lower}, {upper})")
    
    # Example for multiple timeframes
    predictions_summary = [
        (y_pred, y_true, "RandomForest", 12),
        (y_pred, y_true, "XGBoost", 24),
    ]
    summary = evaluate_multiple_timeframes(predictions_summary)
    logger.info(f"Summary across timeframes: {summary}")
