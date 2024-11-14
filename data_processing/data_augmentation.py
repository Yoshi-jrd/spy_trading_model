# data_processing/data_augmentation.py
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from data.data_loader import load_data

logger = logging.getLogger(__name__)

def ensure_numeric(df):
    """Convert all columns in a DataFrame to numeric types, if possible."""
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def add_features(spy_data):
    """Adds required features to the SPY data for compatibility with train_models."""
    # Rename `SPY_Price` to `Close_15m` for consistency
    spy_data = spy_data.rename(columns={'SPY_Price': 'Close_15m'})
    print("Initial data after renaming SPY_Price to Close_15m:")
    print(spy_data.head())
    print(spy_data.tail())

    # Impulse Color Calculation
    spy_data['Impulse_Color_15m'] = np.where(
        spy_data['Close_15m'].diff() > 0, 'green', 
        np.where(spy_data['Close_15m'].diff() < 0, 'red', 'gray')
    )

    # Lagged Close Prices
    for lag in [1, 2, 3]:
        spy_data[f'Close_15m_lag_{lag}'] = spy_data['Close_15m'].shift(lag)

    # Moving Averages
    for window in [5, 10, 15]:
        spy_data[f'Close_15m_ma_{window}'] = spy_data['Close_15m'].rolling(window).mean()

    # Fill initial NaN values from lagged and moving average columns
    spy_data[['Close_15m_lag_1', 'Close_15m_lag_2', 'Close_15m_lag_3']] = spy_data[['Close_15m_lag_1', 'Close_15m_lag_2', 'Close_15m_lag_3']].fillna(method='bfill')
    spy_data[['Close_15m_ma_5', 'Close_15m_ma_10', 'Close_15m_ma_15']] = spy_data[['Close_15m_ma_5', 'Close_15m_ma_10', 'Close_15m_ma_15']].fillna(method='bfill')

    # Date-Based Features
    spy_data['day_of_week'] = spy_data.index.dayofweek
    spy_data['hour_of_day'] = spy_data.index.hour

    # MACD Histogram
    short_ema = spy_data['Close_15m'].ewm(span=12, adjust=False).mean()
    long_ema = spy_data['Close_15m'].ewm(span=26, adjust=False).mean()
    spy_data['MACD_Histogram_15m'] = short_ema - long_ema

    # RSI Calculation
    delta = spy_data['Close_15m'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = -np.where(delta < 0, delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=1).mean()
    rs = avg_gain / avg_loss
    spy_data['RSI_15m'] = 100 - (100 / (1 + rs))

    # Fill NaN values in RSI, initially with backfill, then with zero for remaining NaNs
    spy_data['RSI_15m'] = spy_data['RSI_15m'].fillna(method='bfill').fillna(0)

    # Bollinger Bands
    rolling_mean = spy_data['Close_15m'].rolling(window=20).mean()
    rolling_std = spy_data['Close_15m'].rolling(window=20).std()
    spy_data['UpperBand_15m'] = rolling_mean + (rolling_std * 2)
    spy_data['LowerBand_15m'] = rolling_mean - (rolling_std * 2)
    spy_data[['UpperBand_15m', 'LowerBand_15m']] = spy_data[['UpperBand_15m', 'LowerBand_15m']].fillna(method='bfill').fillna(0)

    # Final fallback to zero for any NaNs not resolved by backfill/forward-fill
    spy_data = spy_data.fillna(0)

    print("\nFinal data after processing:")
    print(spy_data.head())
    print(spy_data.tail())

    return spy_data


def create_year_of_15min_data():
    """Creates a complete year's worth of 15-minute SPY data for market hours only (9:30 am - 4:00 pm ET, Mon - Fri)."""
    data_15min = load_data('15min')  
    data_1hr = ensure_numeric(load_data('1hr'))  
    data_1d = ensure_numeric(load_data('1d'))    

    for df in [data_15min, data_1hr, data_1d]:
        if 'Close' in df.columns:
            df.rename(columns={'Close': 'SPY_Price'}, inplace=True)
        elif 'Price' in df.columns:
            df.rename(columns={'Price': 'SPY_Price'}, inplace=True)

    combined_data = data_15min.copy()

    data_1hr = data_1hr.resample('15min').interpolate(method='linear')
    combined_data = pd.concat([combined_data, data_1hr])

    data_1d = data_1d.resample('15min').interpolate(method='linear')
    combined_data = pd.concat([combined_data, data_1d])

    combined_data = combined_data[~combined_data.index.duplicated(keep='last')]

    start_date = combined_data.index.max() - pd.Timedelta(days=365)
    combined_data = combined_data.loc[start_date:]

    full_index = pd.date_range(
        start=combined_data.index[0],
        end=combined_data.index[-1],
        freq='15min'
    ).to_series()
    full_index = full_index[full_index.index.dayofweek < 5]
    full_index = full_index.between_time('09:30', '16:00')

    combined_data = combined_data.reindex(full_index)
    combined_data['SPY_Price'] = combined_data['SPY_Price'].interpolate(method='linear')

    # Verify that SPY_Price is populated
    if combined_data['SPY_Price'].isnull().any():
        logger.warning("SPY_Price has NaN values after interpolation.")
    else:
        logger.info("SPY_Price data is complete.")

    # Plot SPY_Price before feature engineering for verification
    plt.figure(figsize=(12, 6))
    plt.plot(combined_data['SPY_Price'], label='SPY Price', color='blue')
    plt.title("Generated 1 Year of 15-Minute SPY Data (Market Hours Only)")
    plt.xlabel("Date")
    plt.ylabel("SPY Price")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Add required features for compatibility with train_models
    combined_data = add_features(combined_data)

    # Log and plot the final result to verify
    logger.info("Generated a full year of 15-minute SPY data with required features for market hours only.")
    
    plt.figure(figsize=(12, 6))
    plt.plot(combined_data['Close_15m'], label='SPY Price (Processed)', color='green')
    plt.title("Generated 1 Year of 15-Minute SPY Data with Features (Market Hours Only)")
    plt.xlabel("Date")
    plt.ylabel("SPY Price")
    plt.legend()
    plt.grid(True)
    plt.show()

    return combined_data

# Example usage if run directly
if __name__ == "__main__":
    create_year_of_15min_data()
