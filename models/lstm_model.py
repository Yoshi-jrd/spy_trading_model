from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV

def build_lstm_model(units=50, dropout=0.2, learning_rate=0.01):
    model = Sequential()
    model.add(LSTM(units=units, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(dropout))
    model.add(LSTM(units=units))
    model.add(Dense(1))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

def hypertune_lstm(X_train, y_train):
    print("Tuning LSTM...")
    lstm = KerasRegressor(build_fn=build_lstm_model, epochs=50, batch_size=32, verbose=0)

    lstm_grid = {
        'units': [50, 100, 150],
        'dropout': [0.2, 0.3, 0.4],
        'learning_rate': [0.001, 0.01, 0.02],
        'batch_size': [32, 64, 128],
        'epochs': [50, 100]
    }

    lstm_cv = GridSearchCV(estimator=lstm, param_grid=lstm_grid, cv=3, scoring='neg_mean_squared_error')
    lstm_cv.fit(X_train, y_train)

    print("Best parameters for LSTM:", lstm_cv.best_params_)
    return lstm_cv.best_estimator_

def train_lstm(X_train, y_train):
    lstm_model = hypertune_lstm(X_train, y_train)
    return lstm_model
