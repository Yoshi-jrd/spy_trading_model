import sys
import os

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import xgboost as xgb
from data.data_loader import load_data
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from models.random_forest import MarketSentimentClassifier
from sklearn.model_selection import train_test_split

def backtest_model():
    # Load the SPY data using the data loader
    data = load_data()

    # Access the 1-hour timeframe data from 'spy_data'
    spy_data_df = data['spy_data']['1d']

    # Check if 'Sentiment' exists in the DataFrame
    if 'Sentiment' not in spy_data_df.columns:
        # Calculate Sentiment based on EMA crossover
        threshold = 0.1  # Neutral threshold
        spy_data_df['Sentiment'] = np.where(spy_data_df['EMA9'] > spy_data_df['EMA21'] + threshold, 1, 
                                            np.where(spy_data_df['EMA9'] < spy_data_df['EMA21'] - threshold, 0, 2))

    # Prepare features and labels
    X = spy_data_df[['MACD', 'RSI', '%K', '%D', 'ATR', 'PlusDI', 'MinusDI', 'EMA9', 'EMA21', 'MFI']]
    y = spy_data_df['Sentiment']  # Use the 'Sentiment' column as the labels

    # Normalize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # TimeSeriesSplit for backtesting
    tscv = TimeSeriesSplit(n_splits=5)

    accuracy_scores = []
    precision_scores = []
    recall_scores = []

    for train_index, test_index in tscv.split(X_scaled):
        # Split the data into training and testing sets based on time
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]  # Use .iloc to access the correct rows
        
        # Train the model
        classifier = MarketSentimentClassifier()
        classifier.train(X_train, y_train)

        # Make predictions
        y_pred = classifier.predict(X_test)

        # Calculate performance metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
        recall = recall_score(y_test, y_pred, average='weighted')

        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)

        print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}")

    # Average metrics over all splits
    avg_accuracy = sum(accuracy_scores) / len(accuracy_scores)
    avg_precision = sum(precision_scores) / len(precision_scores)
    avg_recall = sum(recall_scores) / len(recall_scores)

    print(f"\nBacktesting Results:\nAverage Accuracy: {avg_accuracy}\nAverage Precision: {avg_precision}\nAverage Recall: {avg_recall}")

# Call the backtesting function
backtest_model()
