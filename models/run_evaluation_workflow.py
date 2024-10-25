# Import necessary functions from train_models.py
from train_models import hypertune_models, evaluate_model, stack_predictions, cross_validate_models
from data.data_loader import load_data  # Assuming you have a data loader to pull your SPY data

def run_evaluation_workflow():
    # Step 1: Load data (assuming load_data function exists and returns X_train, X_test, y_train, y_test)
    X_train, X_test, y_train, y_test = load_data()  # This should return the preprocessed datasets
    
    # Step 2: Hypertune models
    print("Starting hypertuning...")
    rf_model, xgb_model, gb_model, lstm_model = hypertune_models(X_train, y_train)
    print("Hypertuning complete.")

    # Step 3: Cross-validation for out-of-fold predictions
    base_models = {'RandomForest': rf_model, 'XGBoost': xgb_model, 'GradientBoosting': gb_model}
    rf_oof_predictions, xgb_oof_predictions, gb_oof_predictions = cross_validate_models(X_train, y_train, base_models)

    # Step 4: Stack predictions
    stacking_predictions = stack_predictions(rf_oof_predictions, xgb_oof_predictions, gb_oof_predictions)

    # Step 5: Evaluate performance on the training set
    train_mae, train_rmse = evaluate_model(y_train, stacking_predictions)
    print(f"Training Set - Stacked Model MAE: {train_mae}, RMSE: {train_rmse}")

    # Step 6: Generate predictions on the test set
    rf_predictions_test = rf_model.predict(X_test)
    xgb_predictions_test = xgb_model.predict(X_test)
    gb_predictions_test = gb_model.predict(X_test)

    # Step 7: Stack predictions for the test set
    stacking_predictions_test = stack_predictions(rf_predictions_test, xgb_predictions_test, gb_predictions_test)

    # Step 8: Evaluate performance on the test set
    test_mae, test_rmse = evaluate_model(y_test, stacking_predictions_test)
    print(f"Test Set - Stacked Model MAE: {test_mae}, RMSE: {test_rmse}")

# Call the function to run the evaluation
if __name__ == "__main__":
    run_evaluation_workflow()
