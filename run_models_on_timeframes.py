import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from firebase_admin import credentials, initialize_app, storage
from data.data_loader import load_spy_multi_timeframes, calculate_indicators
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import mean_absolute_error, accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

# Suppress warnings and TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

# Firebase Initialization (Optional)
def initialize_firebase():
    cred = credentials.Certificate("firebase_credentials.json")
    initialize_app(cred, {'storageBucket': 'gs://spymodel-e731d.appspot.com'})
    print("Firebase initialized.")

# --- Load Data for Timeframes ---
def load_data_for_timeframes(timeframes):
    spy_data_load = load_spy_multi_timeframes()  # Load the multi-timeframe SPY data
    data_dict = {}  # Dictionary to store the results for each timeframe

    for timeframe in timeframes:
        spy_data = spy_data_load[timeframe]
        
        # Ensure 'Close' is part of the features or saved separately
        features = ['MACD', 'RSI', '%K', '%D', 'ATR', 'PlusDI', 'MinusDI', 'EMA9', 'EMA21', 'MFI', 'Close']  # Add 'Close' to features
        spy_data = calculate_indicators(spy_data)

        X = spy_data[features]

        threshold = 0.1
        spy_data['Sentiment'] = np.where(spy_data['EMA9'] > spy_data['EMA21'] + threshold, 1,
                                         np.where(spy_data['EMA9'] < spy_data['EMA21'] - threshold, 0, 2))
        y = spy_data['Sentiment']

        combined_df = pd.concat([X, y], axis=1).dropna()
        X, y = combined_df[features], combined_df['Sentiment']

        class_counts = y.value_counts()
        valid_classes = class_counts[class_counts >= 6].index
        X_filtered, y_filtered = X[y.isin(valid_classes)], y[y.isin(valid_classes)]

        print(f"Filtered class distribution for {timeframe}:")
        print(y_filtered.value_counts())

        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_filtered, y_filtered)

        data_dict[timeframe] = (X_resampled, y_resampled, spy_data[['Close']])  # Return 'Close' prices as well
    
    return data_dict

# --- Stacking Model ---
def run_stacking_model(X, y, base_models, meta_model=None):
    if meta_model is None:
        meta_model = GradientBoostingClassifier()

    kf = StratifiedKFold(n_splits=10)
    n_classes = len(np.unique(y))
    rf_oof_predictions = np.zeros((X.shape[0], n_classes))
    xgb_oof_predictions = np.zeros((X.shape[0], n_classes))

    for train_index, test_index in kf.split(X, y):
        X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
        rf_model, xgb_model = base_models['RandomForest'], base_models['XGBoost']

        rf_model.fit(X_train, y_train)
        xgb_model.fit(X_train, y_train)

        rf_oof_predictions[test_index, :] = rf_model.predict_proba(X_test)
        xgb_oof_predictions[test_index, :] = xgb_model.predict_proba(X_test)

    stacked_predictions = np.column_stack((rf_oof_predictions, xgb_oof_predictions))
    meta_model.fit(stacked_predictions, y)

    accuracy_scores, precision_scores, recall_scores = [], [], []

    for train_index, test_index in kf.split(X, y):
        X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
        rf_test_pred = base_models['RandomForest'].predict_proba(X_test)
        xgb_test_pred = base_models['XGBoost'].predict_proba(X_test)

        stacked_test = np.column_stack((rf_test_pred, xgb_test_pred))
        final_pred = meta_model.predict(stacked_test)

        accuracy_scores.append(accuracy_score(y_test, final_pred))
        precision_scores.append(precision_score(y_test, final_pred, average='weighted'))
        recall_scores.append(recall_score(y_test, final_pred, average='weighted'))

    print(f"Cross-Validation Results:\n Accuracy: {np.mean(accuracy_scores)}\n Precision: {np.mean(precision_scores)}\n Recall: {np.mean(recall_scores)}")
    return np.mean(accuracy_scores), np.mean(precision_scores), np.mean(recall_scores)

# --- LSTM Model ---
def train_lstm(X, y, num_units=50, learning_rate=0.001, batch_size=32, epochs=50):
    scaler_X, scaler_y = MinMaxScaler(), MinMaxScaler()
    X_scaled, y_scaled = scaler_X.fit_transform(X), scaler_y.fit_transform(y.values.reshape(-1, 1))

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
    X_train_reshaped = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test_reshaped = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    model = Sequential()
    model.add(Input(shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])))
    model.add(LSTM(num_units, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train_reshaped, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test_reshaped, y_test), callbacks=[early_stopping], verbose=1)

    y_pred = model.predict(X_test_reshaped)
    y_pred_inversed = scaler_y.inverse_transform(y_pred)
    y_test_inversed = scaler_y.inverse_transform(y_test)

    mae = mean_absolute_error(y_test_inversed, y_pred_inversed)
    accuracy = accuracy_score(np.round(y_test_inversed), np.round(y_pred_inversed))
    precision = precision_score(np.round(y_test_inversed), np.round(y_pred_inversed), average='weighted')
    recall = recall_score(np.round(y_test_inversed), np.round(y_pred_inversed), average='weighted')

    print(f"LSTM Evaluation - MAE: {mae}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}")

    plt.figure(figsize=(10, 6))
    plt.plot(y_test_inversed, label='Actual SPY Price', color='blue', linewidth=2)
    plt.plot(y_pred_inversed, label='Predicted SPY Price', color='red', linestyle='--', linewidth=2)
    plt.title('LSTM Predicted vs Actual SPY Price', fontsize=16)
    plt.xlabel('Time Steps', fontsize=14)
    plt.ylabel('Price', fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.7)
    plt.ylim([min(min(y_test_inversed), min(y_pred_inversed)) * 0.95, max(max(y_test_inversed), max(y_pred_inversed)) * 1.05])
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

    return model

# --- Generate Synthetic Data ---
def generate_synthetic_data(X, y, num_samples=1000):
    return resample(X, y, replace=True, n_samples=num_samples, random_state=42)

# --- Main Function ---
def run_model():
    timeframes = ['5m', '15m', '1h', '1d']
    data_dict = load_data_for_timeframes(timeframes)

    for timeframe, (X, y) in data_dict.items():
        print(f"Running models for {timeframe} timeframe...")

        # Base models for stacking
        base_models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', verbosity=0),
            'GradientBoosting': GradientBoostingClassifier(random_state=42)
        }

        # Generate synthetic data
        X_synthetic, y_synthetic = generate_synthetic_data(X, y)

        # Run stacking model and calculate its accuracy
        stacking_accuracy, stacking_precision, stacking_recall = run_stacking_model(X_synthetic, y_synthetic, base_models)
        print(f"Stacking Model for {timeframe} Timeframe: Accuracy = {stacking_accuracy}, Precision = {stacking_precision}, Recall = {stacking_recall}")

        # If it's 1h or 1d, augment LSTM with predictions from stacking model
        if timeframe in ['1h', '1d']:
            print(f"Training LSTM for {timeframe} timeframe...")

            # Get predictions from the base models
            rf_model = base_models['RandomForest']
            xgb_model = base_models['XGBoost']
            gb_model = base_models['GradientBoosting']

            rf_predictions = rf_model.predict_proba(X)[:, 1].reshape(-1, 1)
            xgb_predictions = xgb_model.predict_proba(X)[:, 1].reshape(-1, 1)
            gb_predictions = gb_model.predict_proba(X)[:, 1].reshape(-1, 1)

            # Augment the input for LSTM with these predictions
            augmented_X = np.hstack((X, rf_predictions, xgb_predictions, gb_predictions))

            # Train the LSTM with the augmented data and get LSTM's accuracy
            lstm_model = train_lstm(augmented_X, y)
            lstm_accuracy = evaluate_lstm(lstm_model, augmented_X, y)

            # Calculate weights based on performance
            total_accuracy = stacking_accuracy + lstm_accuracy
            weight_stacked = stacking_accuracy / total_accuracy
            weight_lstm = lstm_accuracy / total_accuracy

            print(f"Weights - Stacked Model: {weight_stacked}, LSTM: {weight_lstm}")

            # Combine the predictions using the calculated weights
            final_predictions = (weight_stacked * stacked_predictions) + (weight_lstm * lstm_predictions)

# --- Execute the model ---
if __name__ == '__main__':
    run_model()

