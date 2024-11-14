import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from data_processing.data_augmentation import create_year_of_15min_data

# Forward-shift parameter for a 24-hour trading horizon
shift_steps = 10
pred_shift = 0  # Shifts predictions by 25 steps backward

# Load and preprocess the data
def load_data():
    data = create_year_of_15min_data()
    return data

def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data['Close_scaled'] = scaler.fit_transform(data[['Close_15m']])
    return data, scaler

# Add indicators (reusing the function structure for simplicity)
def add_indicators(data):
    # Add EMAs
    data['EMA9'] = data['Close_15m'].ewm(span=9, adjust=False).mean()
    data['EMA21'] = data['Close_15m'].ewm(span=21, adjust=False).mean()
    data['EMA50'] = data['Close_15m'].ewm(span=50, adjust=False).mean()
    data['EMA200'] = data['Close_15m'].ewm(span=200, adjust=False).mean()

    # Add Impulse MACD
    ema12 = data['Close_15m'].ewm(span=12, adjust=False).mean()
    ema26 = data['Close_15m'].ewm(span=26, adjust=False).mean()
    data['MACD'] = ema12 - ema26
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
    data['Impulse_MACD'] = np.where(data['MACD_Histogram'] > 0, 1, np.where(data['MACD_Histogram'] < 0, -1, 0))

    # Add RSI-normalized Bollinger Bands
    data['RSI'] = data['Close_15m'].rolling(window=14).apply(lambda x: 100 - (100 / (1 + (x.diff().clip(lower=0).sum() / x.diff().clip(upper=0).abs().sum()))), raw=False)
    data['SMA20'] = data['Close_15m'].rolling(window=20).mean()
    data['UpperBand'] = data['SMA20'] + (data['Close_15m'].rolling(window=20).std() * 2)
    data['LowerBand'] = data['SMA20'] - (data['Close_15m'].rolling(window=20).std() * 2)

    # Add Money Flow Index (MFI)
    data['Typical_Price'] = (data['Close_15m'] + data['High'] + data['Low']) / 3
    data['Money_Flow'] = data['Typical_Price'] * data['Volume']
    data['Positive_MF'] = np.where(data['Typical_Price'] > data['Typical_Price'].shift(1), data['Money_Flow'], 0)
    data['Negative_MF'] = np.where(data['Typical_Price'] < data['Typical_Price'].shift(1), data['Money_Flow'], 0)
    data['MFR'] = data['Positive_MF'].rolling(window=14).sum() / data['Negative_MF'].rolling(window=14).sum()
    data['MFI'] = 100 - (100 / (1 + data['MFR']))

    # Add ADX
    data['TR'] = data[['High', 'Low', 'Close_15m']].max(axis=1) - data[['High', 'Low', 'Close_15m']].min(axis=1)
    data['ATR'] = data['TR'].rolling(window=14).mean()
    data['PlusDM'] = np.where((data['High'] - data['High'].shift(1)) > (data['Low'].shift(1) - data['Low']), data['High'] - data['High'].shift(1), 0)
    data['MinusDM'] = np.where((data['Low'].shift(1) - data['Low']) > (data['High'] - data['High'].shift(1)), data['Low'].shift(1) - data['Low'], 0)
    data['PlusDI'] = 100 * (data['PlusDM'] / data['ATR']).rolling(window=14).mean()
    data['MinusDI'] = 100 * (data['MinusDM'] / data['ATR']).rolling(window=14).mean()
    data['DX'] = (abs(data['PlusDI'] - data['MinusDI']) / abs(data['PlusDI'] + data['MinusDI'])) * 100
    data['ADX'] = data['DX'].rolling(window=14).mean()

    return data

# Prepare data with alternating sequence lengths and padding
def prepare_data_with_alternating_steps(data, max_sequence_length=14):
    sequences = []
    targets = []
    sequence_length = 13  # Start with 13 steps
    step_counter = 0

    shifted_target = data['Close_scaled'].shift(-shift_steps).dropna()
    data = data.iloc[:-shift_steps]

    for i in range(len(data) - max_sequence_length):
        sequence = data['Close_scaled'].iloc[i:i + sequence_length].values
        
        # Pad sequence to max length if needed
        if len(sequence) < max_sequence_length:
            sequence = np.pad(sequence, (0, max_sequence_length - len(sequence)), 'constant')
        
        target = shifted_target.iloc[i + sequence_length - 1]
        sequences.append(sequence)
        targets.append(target)

        # Alternate between 13 and 14 steps
        step_counter += 1
        sequence_length = 14 if step_counter % 2 == 0 else 13

    return np.array(sequences), np.array(targets)

# Build and train the LSTM model
def build_and_train_lstm(train_X, train_y, sequence_length, epochs=10, batch_size=32):
    model = Sequential([
        LSTM(50, input_shape=(sequence_length, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, verbose=1)
    return model

# Evaluate the model and plot results with shifted predictions
def evaluate_and_plot(model, scaler, test_X, test_y, pred_shift=25):
    predictions = model.predict(test_X)
    predictions = scaler.inverse_transform(predictions)
    actual_prices = scaler.inverse_transform(test_y.reshape(-1, 1))

    # Apply the shift to predictions
    shifted_predictions = np.roll(predictions, -pred_shift)
    shifted_predictions[-pred_shift:] = np.nan  # Fill shifted end values with NaN

    mae = mean_absolute_error(actual_prices[:-pred_shift], shifted_predictions[:-pred_shift])
    rmse = np.sqrt(mean_squared_error(actual_prices[:-pred_shift], shifted_predictions[:-pred_shift]))
    avg_diff = np.mean(np.abs(shifted_predictions[:-pred_shift] - actual_prices[:-pred_shift]))

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(actual_prices, label='Actual Prices', color='blue')
    plt.plot(shifted_predictions, label='Shifted Predicted Prices', color='red')
    plt.title('LSTM Predicted vs. Actual Prices - 24h Interval with Shifted Predictions')
    plt.xlabel('Time Step')
    plt.ylabel('SPY Price')
    plt.legend()
    plt.show()

    print(f"Shifted MAE: {mae:.4f}, RMSE: {rmse:.4f}, Average Difference: {avg_diff:.4f}")

# Main execution
data = load_data()
data = add_indicators(data)
data, scaler = preprocess_data(data)
max_sequence_length = 14  # Max length for alternating between 13 and 14 steps

# Prepare data with alternating steps and padding
train_X, train_y = prepare_data_with_alternating_steps(data, max_sequence_length)

# Reshape for LSTM input
train_X = train_X.reshape(train_X.shape[0], train_X.shape[1], 1)

# Split into training and testing
split = int(len(train_X) * 0.8)
test_X, test_y = train_X[split:], train_y[split:]
train_X, train_y = train_X[:split], train_y[:split]

# Build and train LSTM
lstm_model = build_and_train_lstm(train_X, train_y, max_sequence_length)

# Evaluate on the test set with shifted predictions
evaluate_and_plot(lstm_model, scaler, test_X, test_y, pred_shift=25)
