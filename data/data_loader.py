import os
import sys
import pickle
from datetime import datetime
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.market_data_loader import load_spy_multi_timeframes, get_vix_futures, get_all_iv
from data.economic_data_loader import load_economic_data
from data.sentiment_data_loader import get_news_sentiment
from data.indicator_calculator import calculate_indicators

def import_data():
    """
    Import data from different sources: SPY, VIX, IV, GDP, CPI, Sentiment, and Articles.
    """
    print("Starting data import...")
    spy_data = load_spy_multi_timeframes()
    vix_data = get_vix_futures()
    iv_data = get_all_iv()
    gdp_data, cpi_data = load_economic_data()
    sentiment_data, articles = get_news_sentiment()

    print("\n--- Data Import Overview ---")
    print("SPY timeframes:", list(spy_data.keys()))
    print("VIX columns:", vix_data.columns)
    print("IV columns:", iv_data.columns)
    print(f"GDP data shape: {gdp_data.shape}, CPI data shape: {cpi_data.shape}")
    print("Sentiment data:", sentiment_data if isinstance(sentiment_data, pd.DataFrame) else f"Value: {sentiment_data}")
    print("Articles data:", articles.head() if isinstance(articles, pd.DataFrame) else articles)

    return spy_data, vix_data, iv_data, gdp_data, cpi_data, sentiment_data, articles

def ensure_datetime(df, possible_col_names):
    """
    Ensure the DataFrame has a 'datetime' column.
    """
    for col_name in possible_col_names:
        if col_name in df.columns:
            df['datetime'] = pd.to_datetime(df[col_name], errors='coerce')
            df.drop(columns=[col_name], inplace=True)
            df['datetime'] = df['datetime'].ffill().bfill().fillna(pd.Timestamp.now())
            return df
    raise ValueError(f"DataFrame missing expected datetime columns {possible_col_names}")

def clean_and_normalize(spy_data, vix_data, iv_data, gdp_data, cpi_data, sentiment_data, articles):
    """
    Clean and normalize data, ensuring:
    - Consistent 'datetime' column.
    - No missing or invalid values (NaN, NaT).
    """
    spy_data_cleaned = {}
    for tf, df in spy_data.items():
        print(f"\nCleaning SPY {tf} data")
        df = ensure_datetime(df, ['Date', 'Datetime'] if tf == '1d' else ['Datetime'])
        df = calculate_indicators(df)

        # Check indicators for NaNs or all-zero values
        for col in ['RSI', 'UpperBand', 'LowerBand', 'ATR', 'ADX', 'MACD_Histogram']:
            if df[col].isna().any() or (df[col] == 0).all():
                print(f"Warning: {col} in {tf} timeframe has NaNs or all-zero values.")
        
        spy_data_cleaned[tf] = df.ffill().bfill()
        print(f"SPY {tf} data cleaned. Shape: {df.shape}")

    # Clean and normalize other data sources
    vix_data = ensure_datetime(vix_data, ['Date', 'Datetime']).drop(columns=['Volume', 'Dividends', 'Stock Splits'], errors='ignore').ffill()
    iv_data = ensure_datetime(iv_data, ['expiry']).ffill()
    gdp_data, cpi_data = gdp_data.ffill().bfill().reset_index(), cpi_data.ffill().bfill().reset_index()
    gdp_data.columns, cpi_data.columns = ['datetime', 'value'], ['datetime', 'value']
    sentiment_data_cleaned = pd.DataFrame({'datetime': [datetime.now().strftime('%Y-%m-%d')], 'sentiment_value': [sentiment_data]})
    articles_cleaned = pd.DataFrame(articles.get('articles', [])).assign(datetime=lambda df: pd.to_datetime(df.get('publishedAt', pd.Timestamp.now()), errors='coerce').ffill().bfill())

    datasets = {
        'spy_data': spy_data_cleaned,
        'vix_data': vix_data,
        'iv_data': iv_data,
        'gdp_data': gdp_data,
        'cpi_data': cpi_data,
        'sentiment_data': sentiment_data_cleaned,
        'articles': articles_cleaned
    }
    
    print("\n--- Cleaned Data Summary ---")
    for key, data in datasets.items():
        if isinstance(data, dict):
            for sub_key, sub_data in data.items():
                print(f"{key} ({sub_key}): Shape {sub_data.shape}")
                print(sub_data.head(2), "\n")
        else:
            print(f"{key}: Shape {data.shape}")
            print(data.head(2), "\n")
    
    return datasets

def augment_data(spy_df, noise_level=0.01):
    """
    Perform data augmentation on SPY data to simulate slight market variations.
    """
    print("Augmenting data with noise level:", noise_level)
    spy_df['Close_augmented'] = spy_df['Close'] * (1 + np.random.normal(0, noise_level, len(spy_df)))
    return spy_df

def apply_augmentations(spy_data):
    """
    Apply augmentations to each SPY timeframe data and append results.
    """
    augmented_spy_data = {}
    for tf, df in spy_data.items():
        print(f"Applying augmentations to {tf} timeframe data")
        df_augmented = augment_data(df)
        augmented_spy_data[tf] = df_augmented
    return augmented_spy_data

def load_existing_data(filename='local_data/historical_data.pickle'):
    """
    Load existing historical data from a pickle file.
    """
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        print(f"Existing data loaded from {filename}.")
        return data
    print(f"{filename} not found.")
    return None

def append_new_data(existing_data, new_data):
    """
    Append new data to the existing data, ensuring a continuous historical record.
    """
    for key, new_df in new_data.items():
        if isinstance(new_df, dict):
            for sub_key, sub_df in new_df.items():
                existing_data.setdefault(key, {}).setdefault(sub_key, pd.DataFrame())
                existing_data[key][sub_key] = pd.concat([existing_data[key][sub_key], sub_df]).drop_duplicates('datetime').sort_values('datetime')
        else:
            existing_data.setdefault(key, pd.DataFrame())
            existing_data[key] = pd.concat([existing_data[key], new_df]).drop_duplicates('datetime').sort_values('datetime')
    return existing_data

def save_to_pickle(data, filename='local_data/historical_data.pickle'):
    """
    Save data to the specified pickle file.
    """
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"\nData saved to {filename}.")

if __name__ == "__main__":
    # Step 1: Import data
    spy_data, vix_data, iv_data, gdp_data, cpi_data, sentiment_data, articles = import_data()

    # Step 2: Clean and normalize data
    cleaned_data = clean_and_normalize(spy_data, vix_data, iv_data, gdp_data, cpi_data, sentiment_data, articles)

    # Step 3: Apply augmentations to SPY data
    augmented_data = apply_augmentations(cleaned_data['spy_data'])
    cleaned_data['spy_data'] = augmented_data  # Integrate augmented data

    # Step 4: Load existing data if available
    existing_data = load_existing_data()

    # Step 5: Append new data
    updated_data = append_new_data(existing_data, cleaned_data) if existing_data else cleaned_data
    print("\n--- Updated Data Overview ---")
    for key, data in updated_data.items():
        if isinstance(data, dict):
            for sub_key, sub_data in data.items():
                print(f"{key} ({sub_key}): Shape {sub_data.shape}")
                print(sub_data.head(2), "\n")
        else:
            print(f"{key}: Shape {data.shape}")
            print(data.head(2), "\n")

    # Step 6: Save updated data to pickle
    save_to_pickle(updated_data)
