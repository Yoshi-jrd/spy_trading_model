from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV

def build_lstm_model(input_shape, units=50, dropout=0.2, learning_rate=0.01):
    model = Sequential()
    model.add(LSTM(units=units, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout))
    model.add(LSTM(units=units))
    model.add(Dense(1))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

def hypertune_lstm(X_train, y_train, input_shape):
    print("Tuning LSTM...")
    lstm = KerasRegressor(model=build_lstm_model, input_shape=input_shape, verbose=0)

    lstm_grid = {
        'model__units': [50, 100, 150],
        'model__dropout': [0.2, 0.3, 0.4],
        'model__learning_rate': [0.001, 0.01, 0.02],
        'batch_size': [32, 64, 128],
        'epochs': [50, 100]
    }

    lstm_cv = GridSearchCV(estimator=lstm, param_grid=lstm_grid, cv=3, scoring='neg_mean_squared_error')
    lstm_cv.fit(X_train, y_train)

    print("Best parameters for LSTM:", lstm_cv.best_params_)
    return lstm_cv.best_estimator_

def train_lstm(X_train, y_train, input_shape):
    # Reshape X_train and X_val to be 3D for LSTM input
    X_train_reshaped = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    
    # Define the model with the correct input shape
    lstm_model = KerasRegressor(
        model=build_lstm_model,
        input_shape=(1, X_train.shape[1]),  # 1 time step, number of features
        epochs=50,
        batch_size=32,
        verbose=0
    )

    # Fit the model
    lstm_model.fit(X_train_reshaped, y_train)

    return lstm_model
