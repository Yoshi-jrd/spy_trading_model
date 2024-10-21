import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score

def train_lstm_model(X, y, num_units=50, learning_rate=0.001, batch_size=32, epochs=50):
    model = Sequential()
    model.add(Input(shape=(X.shape[1], X.shape[2])))
    model.add(LSTM(num_units, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])

    model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1)
    return model

def combine_predictions(stacking_predictions, lstm_predictions, stacking_weight=0.7, lstm_weight=0.3):
    final_predictions = (stacking_weight * stacking_predictions) + (lstm_weight * lstm_predictions)
    return final_predictions

def evaluate_model_performance(y_true, y_pred):
    accuracy = accuracy_score(y_true, np.round(y_pred))
    precision = precision_score(y_true, np.round(y_pred), average='weighted')
    recall = recall_score(y_true, np.round(y_pred), average='weighted')
    return accuracy, precision, recall
