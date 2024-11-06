from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

def hypertune_xgboost(X_train, y_train, **kwargs):
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

def train_xgboost(X_train, y_train, **kwargs):
    if kwargs:  # If specific parameters are provided, use them
        xgb_model = XGBRegressor(**kwargs)
    else:  # Otherwise, run hypertuning
        xgb_model = hypertune_xgboost(X_train, y_train)
    xgb_model.fit(X_train, y_train)
    return xgb_model
