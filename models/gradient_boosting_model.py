from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

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

def train_gradient_boosting(X_train, y_train):
    gb_model = hypertune_gradient_boosting(X_train, y_train)
    return gb_model
