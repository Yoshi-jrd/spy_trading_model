from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

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

def train_random_forest(X_train, y_train, **kwargs):
    if kwargs:  # Use provided parameters if available
        rf_model = RandomForestRegressor(**kwargs)
    else:  # Otherwise, run hypertuning
        rf_model = hypertune_random_forest(X_train, y_train)
    rf_model.fit(X_train, y_train)
    return rf_model
