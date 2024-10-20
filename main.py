import numpy as np
import pandas as pd
from models.random_forest import MarketSentimentClassifier
from data.data_loader import load_data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# Load the data once
data = load_data()

# Extract SPY data and features
spy_data = data['spy_data']['1h']  # Use 1-hour data

# Use all technical indicators as features
X = spy_data[['MACD', 'RSI', 'UpperBand', 'LowerBand', '%K', '%D', 'ATR', 'PlusDI', 'MinusDI', 'EMA9', 'EMA21', 'OBV', 'MFI']]

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create sentiment labels (0 = bearish, 1 = bullish, -1 = neutral)
#spy_data['Sentiment'] = np.where(spy_data['Close'].diff() > 0.5, 1, np.where(spy_data['Close'].diff() < -0.5, -1, 0))
#y = spy_data['Sentiment']

# Refine labeling using EMA crossover (bullish when EMA9 crosses above EMA21)
spy_data['Sentiment'] = np.where(spy_data['EMA9'] > spy_data['EMA21'], 1, 0)
y = spy_data['Sentiment']

# Function to train and test the model
def run_random_forest(X_scaled, y):
    classifier = MarketSentimentClassifier()

    # Define hyperparameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],  # Number of trees
        'max_depth': [None, 10, 20, 30],  # Maximum depth of the tree
        'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split a node
        'min_samples_leaf': [1, 2, 4]     # Minimum number of samples required at a leaf node
    }
    
    # Grid Search with cross-validation
    grid_search = GridSearchCV(classifier.model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_scaled, y)
    
    # Best parameters and accuracy
    print("Best Parameters:", grid_search.best_params_)
    print("Best Accuracy with Grid Search:", grid_search.best_score_)

    # Set the classifier to the best estimator found by GridSearchCV
    classifier.model = grid_search.best_estimator_
    
    # Train and predict using the optimized model
    classifier.train(X_scaled, y)

    # Example prediction for the latest data point
    prediction = classifier.predict(X_scaled[-1].reshape(1, -1))
    print(f"Market Sentiment Prediction: {prediction}")

# Call the function to run Random Forest with hyperparameter tuning
run_random_forest(X_scaled, y)

"""
    # Test and print key data components for inspection
    print("SPY Data (Multi-timeframe):")
    print(data['spy_data']['1h'].head())  # Example: print 1-hour SPY data

    print("\nVIX Data:")
    print(data['vix_data'].head())  # Print VIX data

    print("\nImplied Volatility (IV):")
    print(data['iv_data'])  # Print calculated IV

    print("\nEconomic Data:")
    print(data['economic_data'])  # Print economic data

    print("\nNews Sentiment Data:")
    print(data['sentiment_data'])  # Print news sentiment data
"""