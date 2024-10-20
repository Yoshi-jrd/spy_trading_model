import firebase_admin
from firebase_admin import credentials, storage
import numpy as np
import pandas as pd
from data.data_loader import load_spy_multi_timeframes, calculate_indicators
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE

# Load Data for Timeframes
def load_data_for_timeframes(timeframes):
    spy_data_load = load_spy_multi_timeframes()  # Load the multi-timeframe SPY data
    data_dict = {}  # Create an empty dictionary to store the results for each timeframe
    
    for timeframe in timeframes:
        # Select the data for the specific timeframe
        spy_data = spy_data_load[timeframe]

        # Calculate indicators
        spy_data = calculate_indicators(spy_data)

        # Define the features
        features = ['MACD', 'RSI', '%K', '%D', 'ATR', 'PlusDI', 'MinusDI', 'EMA9', 'EMA21', 'MFI']
        X = spy_data[features]

        # Prepare sentiment target (y)
        threshold = 0.1
        spy_data['Sentiment'] = np.where(spy_data['EMA9'] > spy_data['EMA21'] + threshold, 1,
                                         np.where(spy_data['EMA9'] < spy_data['EMA21'] - threshold, 0, 2))
        y = spy_data['Sentiment']

        # Handle missing values
        combined_df = pd.concat([X, y], axis=1).dropna()
        X = combined_df[features]
        y = combined_df['Sentiment']

        # Filter out small classes before SMOTE
        class_counts = y.value_counts()
        min_class_size = 6  # Minimum number of samples required for SMOTE
        valid_classes = class_counts[class_counts >= min_class_size].index

        # Filter the data to remove classes with too few samples
        X_filtered = X[y.isin(valid_classes)]
        y_filtered = y[y.isin(valid_classes)]

        print(f"Filtered class distribution for {timeframe}:")
        print(y_filtered.value_counts())

        # Apply SMOTE to balance classes
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_filtered, y_filtered)

        # Store the resampled data in the dictionary
        data_dict[timeframe] = (X_resampled, y_resampled)
    
    return data_dict

# --- Run Base Models ---
def run_base_models(X_train, X_test, y_train, y_test):
    results = {}

    # RandomForest for Daily Timeframe
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    rf_precision = precision_score(y_test, rf_pred, average='weighted')
    rf_recall = recall_score(y_test, rf_pred, average='weighted')
    # Return the RandomForest model object along with metrics
    results['RandomForest'] = (rf_model, rf_accuracy, rf_precision, rf_recall)

    # XGBoost for Shorter Timeframes
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_accuracy = accuracy_score(y_test, xgb_pred)
    xgb_precision = precision_score(y_test, xgb_pred, average='weighted')
    xgb_recall = recall_score(y_test, xgb_pred, average='weighted')
    # Return the XGBoost model object along with metrics
    results['XGBoost'] = (xgb_model, xgb_accuracy, xgb_precision, xgb_recall)

    return results

# --- Stacking Model ---
# --- Improved Stacking Model ---
def run_stacking_model(X, y, base_models, meta_model=None):
    """
    Implements an enhanced stacking model with cross-validation-based stacking
    using Random Forest and XGBoost as base models and GradientBoosting as the meta-model.
    """
    if meta_model is None:
        meta_model = GradientBoostingClassifier()  # Use Gradient Boosting as the meta-model by default
    
    # Get the number of unique classes in y
    n_classes = len(np.unique(y))
    
    # Prepare cross-validation folds
    kf = StratifiedKFold(n_splits=5)
    
    # Arrays to store out-of-fold predictions for stacking
    rf_oof_predictions = np.zeros((X.shape[0], n_classes))  # Adjusted to match the number of classes
    xgb_oof_predictions = np.zeros((X.shape[0], n_classes))  # Adjusted to match the number of classes
    
    # Store final predictions for the test set
    final_predictions = np.zeros(X.shape[0])
    
    # Perform cross-validation
    for train_index, test_index in kf.split(X, y):
        # Split the data into training and testing sets
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Get base models (Random Forest and XGBoost)
        rf_model, xgb_model = base_models['RandomForest'], base_models['XGBoost']
        
        # Train Random Forest and XGBoost
        rf_model.fit(X_train, y_train)
        xgb_model.fit(X_train, y_train)
        
        # Get out-of-fold predictions (probabilities) for stacking
        rf_oof_predictions[test_index, :] = rf_model.predict_proba(X_test)  # Adjust to handle 2 or 3 classes
        xgb_oof_predictions[test_index, :] = xgb_model.predict_proba(X_test)  # Adjust to handle 2 or 3 classes
    
    # Stack the out-of-fold predictions
    stacked_predictions = np.column_stack((
        rf_oof_predictions[:, 1],  # Use the probabilities for class 1
        xgb_oof_predictions[:, 1]  # Use the probabilities for class 1
    ))
    
    # Train the meta-model on the stacked predictions
    meta_model.fit(stacked_predictions, y)
    
    # Cross-validation final predictions
    final_accuracy = []
    final_precision = []
    final_recall = []
    
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        rf_model, xgb_model = base_models['RandomForest'], base_models['XGBoost']
        
        rf_test_pred = rf_model.predict_proba(X_test)[:, 1]  # Probabilities for class 1
        xgb_test_pred = xgb_model.predict_proba(X_test)[:, 1]  # Probabilities for class 1
        
        stacked_test = np.column_stack((rf_test_pred, xgb_test_pred))
        
        # Make predictions using the meta-model
        final_pred = meta_model.predict(stacked_test)
        
        # Calculate metrics
        acc = accuracy_score(y_test, final_pred)
        prec = precision_score(y_test, final_pred, average='weighted')
        rec = recall_score(y_test, final_pred, average='weighted')
        
        final_accuracy.append(acc)
        final_precision.append(prec)
        final_recall.append(rec)
    
    # Return average metrics over folds
    avg_accuracy = np.mean(final_accuracy)
    avg_precision = np.mean(final_precision)
    avg_recall = np.mean(final_recall)
    
    return avg_accuracy, avg_precision, avg_recall

# --- LSTM Model (Placeholder for LSTM Integration) ---
def train_lstm(X, y):
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X)

    # Reshape for LSTM input
    X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(X_reshaped.shape[1], X_reshaped.shape[2])))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    model.fit(X_reshaped, y, epochs=50, batch_size=32, verbose=1)

    return model

# --- Updated Main Function ---
def run_model():
    # Define the timeframes to run the models for
    timeframes = ['5m', '15m', '1h', '1d']

    # Load data for all timeframes and store in a dictionary
    data_dict = load_data_for_timeframes(timeframes)

    # Iterate through each timeframe and run models
    for timeframe, (X, y) in data_dict.items():
        print(f"Running models for {timeframe} timeframe...")

        # Define base models
        base_models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        }
        
        # Run stacking model (No need to split X_train/X_test or y_train/y_test manually)
        stacking_accuracy, stacking_precision, stacking_recall = run_stacking_model(X, y, base_models)
        
        # Output stacking model results
        print(f"Stacking Model for {timeframe} Timeframe: Accuracy = {stacking_accuracy}, Precision = {stacking_precision}, Recall = {stacking_recall}")

        # Future integration: Train LSTM or Reinforcement Learning
        if timeframe == '1d':
            print(f"Training LSTM for 2-7 day predictive modeling on {timeframe} timeframe...")
            lstm_model = train_lstm(X, y)


# --- Load Data for All Timeframes ---


# Execute the model
if __name__ == '__main__':
    run_model()
