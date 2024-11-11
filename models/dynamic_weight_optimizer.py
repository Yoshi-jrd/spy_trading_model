import logging
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def objective_function(weights, model_predictions, y_true, metric='mae'):
    """
    Objective function to minimize the error by optimizing weights for stacked predictions.
    """
    # Compute weighted predictions
    weighted_preds = np.sum(weights * model_predictions, axis=1)
    weighted_preds = np.nan_to_num(weighted_preds)  # Handle NaNs in weighted_preds

    # Calculate MAE and RMSE
    mae = mean_absolute_error(y_true, weighted_preds)
    rmse = np.sqrt(mean_squared_error(y_true, weighted_preds))
    
    # Choose the metric for optimization
    if metric == 'mae':
        return mae
    elif metric == 'rmse':
        return rmse
    elif metric == 'hybrid':
        return 0.5 * mae + 0.5 * rmse
    else:
        raise ValueError("Invalid metric specified. Choose 'mae', 'rmse', or 'hybrid'.")

def optimize_weights(model_predictions, y_true, initial_weights=None, metric='mae'):
    """
    Optimizes weights for stacking models to minimize error metrics on validation set.
    """
    num_models = model_predictions.shape[1]
    if initial_weights is None:
        initial_weights = np.ones(num_models) / num_models  # Start with equal weights

    # Constraints: weights must sum to 1
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0, 1)] * num_models  # Weights between 0 and 1

    # Try SLSQP first for efficiency; fallback to COBYLA if needed
    result = minimize(objective_function, initial_weights, args=(model_predictions, y_true, metric),
                      bounds=bounds, constraints=constraints, method='SLSQP')
    
    if not result.success:
        # Fallback to COBYLA if SLSQP fails
        result = minimize(objective_function, initial_weights, args=(model_predictions, y_true, metric),
                          bounds=bounds, constraints=constraints, method='COBYLA')

    if result.success:
        logger.info("Optimization successful.")
    else:
        logger.warning("Optimization completed with issues: " + result.message)

    return result.x
