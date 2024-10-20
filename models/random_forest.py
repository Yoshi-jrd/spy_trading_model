import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class MarketSentimentClassifier:
    def __init__(self):
        # Initialize Random Forest with 100 trees
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def train(self, X, y):
        # Split data into training and test sets (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        self.model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = self.model.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Random Forest Accuracy: {accuracy}")

    def predict(self, X):
        # Predict market sentiment
        return self.model.predict(X)
