import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import json

# Set the path to config.json based on the current file's location
config_path = os.path.join(os.path.dirname(__file__), 'config.json')
with open(config_path, 'r') as f:
    config = json.load(f)


def load_data():
    """Load and preprocess data from pickle file."""
    with open(config['paths']['historical_data'], 'rb') as f:
        data = pickle.load(f)
    
    spy_data_15m = data['spy_data']['15m']
    spy_data_15m = spy_data_15m.add_suffix('_15m')
    spy_data_15m.rename(columns={'datetime_15m': 'datetime'}, inplace=True)
    return spy_data_15m

def preprocess_data(spy_data_15m):
    """Feature engineering and preprocessing."""
    spy_data_15m['Impulse_Color_15m'] = spy_data_15m['Impulse_Color_15m'].map({'red': -1, 'green': 1, 'gray': 0})
    for lag in [1, 2, 3]:
        spy_data_15m[f'Close_15m_lag_{lag}'] = spy_data_15m['Close_15m'].shift(lag)
    for window in [5, 10, 15]:
        spy_data_15m[f'Close_15m_ma_{window}'] = spy_data_15m['Close_15m'].rolling(window).mean()
    
    spy_data_15m['day_of_week'] = pd.to_datetime(spy_data_15m['datetime']).dt.dayofweek
    spy_data_15m['hour_of_day'] = pd.to_datetime(spy_data_15m['datetime']).dt.hour
    spy_data_15m = spy_data_15m.dropna().reset_index(drop=True)
    return spy_data_15m

def prepare_lstm_data(df):
    """Prepare sequences and targets for LSTM."""
    sequence_length = config['lstm']['sequence_length']
    scaler = MinMaxScaler()
    close_scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['Close_15m', 'MACD_Histogram_15m', 'RSI_15m', 'UpperBand_15m', 'LowerBand_15m']])
    scaled_close = close_scaler.fit_transform(df[['Close_15m']])
    
    sequences, targets = [], []
    for i in range(len(scaled_data) - sequence_length):
        sequences.append(scaled_data[i:i + sequence_length])
        targets.append(scaled_close[i + sequence_length, 0])
    
    return np.array(sequences), np.array(targets).reshape(-1, 1), close_scaler
