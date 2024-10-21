import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold

# Function to train Random Forest
def train_random_forest(X, y, n_estimators=100):
    rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    rf_model.fit(X, y)
    return rf_model

# Function to train XGBoost
def train_xgboost(X, y):
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', verbosity=0)
    xgb_model.fit(X, y)
    return xgb_model

# Function to train Gradient Boosting
def train_gradient_boosting(X, y, n_estimators=100):
    gb_model = GradientBoostingClassifier(n_estimators=n_estimators, random_state=42)
    gb_model.fit(X, y)
    return gb_model

# Cross-validation to get out-of-fold (OOF) predictions for each model
def cross_validate_models(X, y, base_models, k=10):
    kf = StratifiedKFold(n_splits=k)
    
    # Initialize arrays to hold out-of-fold predictions for each model
    rf_oof_predictions = np.zeros((X.shape[0], len(np.unique(y))))
    xgb_oof_predictions = np.zeros((X.shape[0], len(np.unique(y))))
    gb_oof_predictions = np.zeros((X.shape[0], len(np.unique(y))))
    
    # Perform k-fold cross-validation
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Train RandomForest and store OOF predictions
        base_models['RandomForest'].fit(X_train, y_train)
        rf_oof_predictions[test_index] = base_models['RandomForest'].predict_proba(X_test)
        
        # Train XGBoost and store OOF predictions
        base_models['XGBoost'].fit(X_train, y_train)
        xgb_oof_predictions[test_index] = base_models['XGBoost'].predict_proba(X_test)
        
        # Train GradientBoosting and store OOF predictions
        base_models['GradientBoosting'].fit(X_train, y_train)
        gb_oof_predictions[test_index] = base_models['GradientBoosting'].predict_proba(X_test)
    
    # Return the out-of-fold predictions for each model
    return rf_oof_predictions, xgb_oof_predictions, gb_oof_predictions

# Stacking function to combine predictions with adjustable weights
def stack_predictions(rf_predictions, xgb_predictions, gb_predictions, rf_weight=0.33, xgb_weight=0.33, gb_weight=0.34):
    # Ensure all predictions are the same shape
    assert rf_predictions.shape == xgb_predictions.shape == gb_predictions.shape
    
    # Weighted average of the base model predictions
    stacking_predictions = (rf_weight * rf_predictions + 
                            xgb_weight * xgb_predictions + 
                            gb_weight * gb_predictions)
    
    return stacking_predictions
