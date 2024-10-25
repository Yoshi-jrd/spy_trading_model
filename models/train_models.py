import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from scikeras.wrappers import KerasRegressor
print("KerasRegressor imported successfully")
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ---------------------------------------------------------
# Save and Load Functions for Scikit-learn and Keras Models
# ---------------------------------------------------------

def save_model(model, model_name):
    joblib.dump(model, f'{model_name}.pkl')
    print(f"Model {model_name} saved to disk.")

def save_keras_model(model, model_name):
    model.save(f'{model_name}.h5')
    print(f"Keras model {model_name} saved to disk.")

def load_model_sklearn(model_name):
    try:
        model = joblib.load(f'{model_name}.pkl')
        print(f"Model {model_name} loaded from disk.")
        return model
    except FileNotFoundError:
        print(f"No saved model found for {model_name}, training from scratch.")
        return None

def load_keras_model(model_name):
    try:
        model = load_model(f'{model_name}.h5')
        print(f"Keras model {model_name} loaded from disk.")
        return model
    except FileNotFoundError:
        print(f"No saved Keras model found for {model_name}, training from scratch.")
        return None

# ---------------------------------------------------------
# Hypertuning Functions
# ---------------------------------------------------------

def hypertune_random_forest(X_train, y_train):
    print("Tuning RandomForest...")
    rf_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }

    rf = RandomForestRegressor(random_state=42)
    rf_cv = GridSearchCV(estimator=rf, param_grid=rf_grid, cv=3, scoring='neg_mean_squared_error')
    rf_cv.fit(X_train, y_train)

    print("Best parameters for RandomForest:", rf_cv.best_params_)
    return rf_cv.best_estimator_

def hypertune_xgboost(X_train, y_train):
    print("Tuning XGBoost...")
    xgb_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.3]
    }

    xgboost = XGBRegressor(eval_metric='rmse', verbosity=0)
    xgb_cv = GridSearchCV(estimator=xgboost, param_grid=xgb_grid, cv=3, scoring='neg_mean_squared_error')
    xgb_cv.fit(X_train, y_train)

    print("Best parameters for XGBoost:", xgb_cv.best_params_)
    return xgb_cv.best_estimator_

def hypertune_gradient_boosting(X_train, y_train):
    print("Tuning Gradient Boosting...")
    gb_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    gb = GradientBoostingRegressor(random_state=42)
    gb_cv = GridSearchCV(estimator=gb, param_grid=gb_grid, cv=3, scoring='neg_mean_squared_error')
    gb_cv.fit(X_train, y_train)

    print("Best parameters for Gradient Boosting:", gb_cv.best_params_)
    return gb_cv.best_estimator_

def build_lstm_model(units=50, dropout=0.2, learning_rate=0.01):
    model = Sequential()
    model.add(LSTM(units=units, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(dropout))
    model.add(LSTM(units=units))
    model.add(Dense(1))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

def hypertune_lstm(X_train, y_train):
    print("Tuning LSTM...")
    lstm = KerasRegressor(build_fn=build_lstm_model, epochs=50, batch_size=32, verbose=0)

    lstm_grid = {
        'units': [50, 100, 150],
        'dropout': [0.2, 0.3, 0.4],
        'learning_rate': [0.001, 0.01, 0.02],
        'batch_size': [32, 64, 128],
        'epochs': [50, 100]
    }

    lstm_cv = GridSearchCV(estimator=lstm, param_grid=lstm_grid, cv=3, scoring='neg_mean_squared_error')
    lstm_cv.fit(X_train, y_train)

    print("Best parameters for LSTM:", lstm_cv.best_params_)
    return lstm_cv.best_estimator_

# ---------------------------------------------------------
# Train, Save, and Evaluate Models
# ---------------------------------------------------------

def train_and_save_models(X_train, y_train):
    # Load or Hypertune RandomForest
    rf_model = load_model_sklearn('RandomForest')
    if rf_model is None:
        rf_model = hypertune_random_forest(X_train, y_train)
        save_model(rf_model, 'RandomForest')

    # Load or Hypertune XGBoost
    xgb_model = load_model_sklearn('XGBoost')
    if xgb_model is None:
        xgb_model = hypertune_xgboost(X_train, y_train)
        save_model(xgb_model, 'XGBoost')

    # Load or Hypertune GradientBoosting
    gb_model = load_model_sklearn('GradientBoosting')
    if gb_model is None:
        gb_model = hypertune_gradient_boosting(X_train, y_train)
        save_model(gb_model, 'GradientBoosting')

    # Load or Hypertune LSTM
    lstm_model = load_keras_model('LSTM')
    if lstm_model is None:
        lstm_model = hypertune_lstm(X_train, y_train)
        save_keras_model(lstm_model, 'LSTM')

    return rf_model, xgb_model, gb_model, lstm_model

# ---------------------------------------------------------
# Cross-Validation Function
# ---------------------------------------------------------

def cross_validate_models(X, y, base_models, k=5, use_timeseries_split=False):
    if use_timeseries_split:
        cv = TimeSeriesSplit(n_splits=k)
    else:
        cv = KFold(n_splits=k)

    rf_oof_predictions = np.zeros((X.shape[0]))
    xgb_oof_predictions = np.zeros((X.shape[0]))
    gb_oof_predictions = np.zeros((X.shape[0]))

    for train_index, test_index in cv.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        base_models['RandomForest'].fit(X_train, y_train)
        rf_oof_predictions[test_index] = base_models['RandomForest'].predict(X_test).reshape(-1)

        base_models['XGBoost'].fit(X_train, y_train)
        xgb_oof_predictions[test_index] = base_models['XGBoost'].predict(X_test).reshape(-1)

        base_models['GradientBoosting'].fit(X_train, y_train)
        gb_oof_predictions[test_index] = base_models['GradientBoosting'].predict(X_test).reshape(-1)

    return rf_oof_predictions, xgb_oof_predictions, gb_oof_predictions

# ---------------------------------------------------------
# Stacking Function
# ---------------------------------------------------------

def stack_predictions(rf_predictions, xgb_predictions, gb_predictions, rf_weight=0.33, xgb_weight=0.33, gb_weight=0.34):
    assert rf_predictions.shape == xgb_predictions.shape == gb_predictions.shape
    # Weighted average of the base model predictions
    stacking_predictions = (rf_weight * rf_predictions + 
                            xgb_weight * xgb_predictions + 
                            gb_weight * gb_predictions)
    
    return stacking_predictions

# ---------------------------------------------------------
# Model Evaluation Function
# ---------------------------------------------------------

def evaluate_model(y_true, predictions):
    mae = mean_absolute_error(y_true, predictions)
    rmse = mean_squared_error(y_true, predictions, squared=False)
    return mae, rmse

# ---------------------------------------------------------
# Hypertuning Workflow
# ---------------------------------------------------------

def hypertune_models(X_train, y_train):
    rf_model = hypertune_random_forest(X_train, y_train)
    xgb_model = hypertune_xgboost(X_train, y_train)
    gb_model = hypertune_gradient_boosting(X_train, y_train)
    lstm_model = hypertune_lstm(X_train, y_train)

    return rf_model, xgb_model, gb_model, lstm_model


# ---------------------------------------------------------
# Example to Combine Everything and Evaluate Models
# ---------------------------------------------------------

def run_full_training_workflow(X_train, y_train, X_test, y_test):
    # Train and save models (RandomForest, XGBoost, GradientBoosting, and LSTM)
    rf_model, xgb_model, gb_model, lstm_model = train_and_save_models(X_train, y_train)

    # Perform stacking with RandomForest, XGBoost, and GradientBoosting
    base_models = {'RandomForest': rf_model, 'XGBoost': xgb_model, 'GradientBoosting': gb_model}
    rf_oof_predictions, xgb_oof_predictions, gb_oof_predictions = cross_validate_models(X_train, y_train, base_models)
    
    # Stack predictions from base models
    stacking_predictions = stack_predictions(rf_oof_predictions, xgb_oof_predictions, gb_oof_predictions)

    # Generate predictions using LSTM model (reshape X_train for LSTM)
    lstm_predictions_train = lstm_model.predict(X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1])))
    lstm_predictions_train = lstm_predictions_train.flatten()  # Flatten LSTM predictions to 1D

    # Combine stacking predictions with LSTM predictions for final predictions
    combined_predictions_train = stack_predictions(stacking_predictions, lstm_predictions_train, rf_weight=0.5, xgb_weight=0.25, gb_weight=0.25)

    # Evaluate the combined model (RandomForest, XGBoost, GradientBoosting, and LSTM) on the training data
    train_mae, train_rmse = evaluate_model(y_train, combined_predictions_train)
    print(f"Training Set - Combined Model MAE: {train_mae}, RMSE: {train_rmse}")

    # Now generate predictions on the test set
    rf_predictions_test = rf_model.predict(X_test)
    xgb_predictions_test = xgb_model.predict(X_test)
    gb_predictions_test = gb_model.predict(X_test)

    # Stack test set predictions from base models
    stacking_predictions_test = stack_predictions(rf_predictions_test, xgb_predictions_test, gb_predictions_test)

    # Generate LSTM predictions on the test set (reshape X_test for LSTM)
    lstm_predictions_test = lstm_model.predict(X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1])))
    lstm_predictions_test = lstm_predictions_test.flatten()

    # Combine stacking predictions and LSTM predictions on test set
    combined_predictions_test = stack_predictions(stacking_predictions_test, lstm_predictions_test, rf_weight=0.5, xgb_weight=0.25, gb_weight=0.25)

    # Evaluate the combined model on the test set
    test_mae, test_rmse = evaluate_model(y_test, combined_predictions_test)
    print(f"Test Set - Combined Model MAE: {test_mae}, RMSE: {test_rmse}")

