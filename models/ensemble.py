import numpy as np
import pandas as pd
from data.data_loader import load_spy_multi_timeframes, calculate_indicators
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE

# Load Data
def load_data_for_model():
    spy_data = load_spy_multi_timeframes()
    spy_data_df = spy_data['1d']
    spy_data_df = calculate_indicators(spy_data_df)
    features = ['MACD', 'RSI', '%K', '%D', 'ATR', 'PlusDI', 'MinusDI', 'EMA9', 'EMA21', 'MFI']
    X = spy_data_df[features]
    threshold = 0.1
    spy_data_df['Sentiment'] = np.where(spy_data_df['EMA9'] > spy_data_df['EMA21'] + threshold, 1,
                                        np.where(spy_data_df['EMA9'] < spy_data_df['EMA21'] - threshold, 0, 2))
    y = spy_data_df['Sentiment']
    combined_df = pd.concat([X, y], axis=1).dropna()
    X = combined_df[features]
    y = combined_df['Sentiment']
    print(f"X shape after cleaning: {X.shape}")
    print(f"y shape after cleaning: {y.shape}")
    print("Class distribution in y (Sentiment):")
    print(y.value_counts())
    return X, y

# XGBoost Model Function
def xgboost_model(X_train, X_test, y_train, y_test):
    xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, eval_metric='mlogloss', early_stopping_rounds=10)
    xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    return xgb.predict(X_test)

# Random Forest Model Function
def random_forest_model(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_split=10, min_samples_leaf=1)
    rf.fit(X_train, y_train)
    return rf.predict(X_test)

# Stacking Model Function
def stacking_model(X, y):
    rare_class_threshold = 5
    class_counts = y.value_counts()
    valid_classes = class_counts[class_counts > rare_class_threshold].index
    X = X[y.isin(valid_classes)]
    y = y[y.isin(valid_classes)]
    print(f"Filtered class distribution in y (after removing rare classes):")
    print(y.value_counts())
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    print(f"Shape after SMOTE resampling: X={X_resampled.shape}, y={y_resampled.shape}")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    xgb_model = XGBClassifier(eval_metric='mlogloss')
    meta_model = LogisticRegression()
    kf = StratifiedKFold(n_splits=5)
    accuracies, precisions, recalls = [], [], []
    for train_index, test_index in kf.split(X_resampled, y_resampled):
        X_train, X_test = X_resampled.iloc[train_index], X_resampled.iloc[test_index]
        y_train, y_test = y_resampled.iloc[train_index], y_resampled.iloc[test_index]
        rf_model.fit(X_train, y_train)
        xgb_model.fit(X_train, y_train)
        rf_pred = rf_model.predict_proba(X_test)[:, 1]
        xgb_pred = xgb_model.predict_proba(X_test)[:, 1]
        stacked_test = np.column_stack((rf_pred, xgb_pred))
        meta_model.fit(stacked_test, y_test)
        final_predictions = meta_model.predict(stacked_test)
        accuracy = accuracy_score(y_test, final_predictions)
        precision = precision_score(y_test, final_predictions, average='weighted')
        recall = recall_score(y_test, final_predictions, average='weighted')
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        print(f"Fold Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}")
    avg_accuracy = np.mean(accuracies)
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    print(f"Stacking Model Accuracy: {avg_accuracy}")
    print(f"Stacking Model Precision: {avg_precision}")
    print(f"Stacking Model Recall: {avg_recall}")

# You can add more ensemble methods like bagging, boosting, etc., into this file as needed.
