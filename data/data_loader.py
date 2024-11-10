import os
import sys
import pickle
from datetime import datetime
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.market_data_loader import load_spy_multi_timeframes, get_vix_futures, get_all_iv
from data.economic_data_loader import load_economic_data
from data.sentiment_data_loader import get_news_sentiment

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

    print("\n--- Imported Data Preview ---")
    print("SPY data keys (timeframes):", spy_data.keys())
    print("VIX data columns:", vix_data.columns)
    print("IV data columns:", iv_data.columns)
    print("GDP data shape:", gdp_data.shape)
    print("CPI data shape:", cpi_data.shape)
    
    if isinstance(sentiment_data, pd.DataFrame):
        print("Sentiment data preview:", sentiment_data.head())
    else:
        print("Sentiment data value:", sentiment_data)
    
    if isinstance(articles, pd.DataFrame):
        print("Articles preview:", articles.head())
    else:
        print("Articles:", articles)

    return spy_data, vix_data, iv_data, gdp_data, cpi_data, sentiment_data, articles

def ensure_datetime(df, possible_col_names):
    """
    Ensure that the DataFrame has a consistent 'datetime' column.
    """
    found_col = None
    for col_name in possible_col_names:
        if col_name in df.columns:
            found_col = col_name
            break

    if found_col:
        df['datetime'] = pd.to_datetime(df[found_col], errors='coerce')
        df.drop(columns=[found_col], inplace=True)
    else:
        print(f"Available columns: {df.columns}")
        raise ValueError(f"DataFrame does not contain any of the columns {possible_col_names} for datetime")

    df['datetime'] = df['datetime'].ffill().bfill().fillna(pd.Timestamp.now())
    return df

def clean_and_normalize(spy_data, vix_data, iv_data, gdp_data, cpi_data, sentiment_data, articles):
    """
    Clean and normalize data, ensuring:
    1. Consistent 'datetime' column.
    2. No missing or invalid values (NaN, NaT).
    """
    spy_data_cleaned = {}
    for tf, df in spy_data.items():
        print(f"\nCleaning SPY data for {tf} timeframe")
        if tf == '1d':
            df = ensure_datetime(df, ['Date', 'Datetime'])
        else:
            df = ensure_datetime(df, ['Datetime'])
        
        df = df.ffill().bfill()
        spy_data_cleaned[tf] = df
        print(f"SPY {tf} data cleaned. Shape: {df.shape}")

    print("\nCleaning VIX data")
    vix_data = ensure_datetime(vix_data, ['Date', 'Datetime'])
    vix_data = vix_data.ffill()
    print("VIX data cleaned. Shape:", vix_data.shape)

    print("\nCleaning IV data")
    iv_data = ensure_datetime(iv_data, ['expiry'])
    iv_data = iv_data.ffill()
    print("IV data cleaned. Shape:", iv_data.shape)

    print("\nCleaning GDP and CPI data")
    gdp_data = pd.DataFrame(gdp_data).reset_index()
    gdp_data.columns = ['datetime', 'value']
    cpi_data = pd.DataFrame(cpi_data).reset_index()
    cpi_data.columns = ['datetime', 'value']
    gdp_data = gdp_data.ffill().bfill()
    cpi_data = cpi_data.ffill().bfill()
    print("GDP data cleaned. Shape:", gdp_data.shape)
    print("CPI data cleaned. Shape:", cpi_data.shape)

    print("\nCleaning Sentiment data")
    current_date = datetime.now().strftime('%Y-%m-%d')
    sentiment_data_cleaned = pd.DataFrame({'datetime': [current_date], 'sentiment_value': [sentiment_data]})
    print("Sentiment data cleaned. Shape:", sentiment_data_cleaned.shape)

    print("\nCleaning Articles data")
    articles_cleaned = pd.DataFrame(articles.get('articles', []))
    
    if 'publishedAt' in articles_cleaned.columns:
        articles_cleaned['datetime'] = pd.to_datetime(articles_cleaned['publishedAt'], errors='coerce')
        articles_cleaned['datetime'] = articles_cleaned['datetime'].ffill().bfill()
    else:
        articles_cleaned['datetime'] = current_date
    print("Articles data cleaned. Shape:", articles_cleaned.shape)

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
                print(f"{key} ({sub_key}): {sub_data.shape}")
        else:
            print(f"{key}: {data.shape}")
    
    return datasets

def load_existing_data(filename='local_data/historical_data.pickle'):
    """
    Load existing historical data from a pickle file.
    """
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            existing_data = pickle.load(f)
        print(f"\nExisting data loaded from {filename}.")
        return existing_data
    else:
        print(f"{filename} not found. No existing data to load.")
        return None

def append_new_data(existing_data, new_data):
    """
    Append only new data to the existing data, ensuring a continuous historical record.
    """
    for key, df_new in new_data.items():
        if isinstance(df_new, dict):
            for sub_key, sub_df_new in df_new.items():
                if key in existing_data and sub_key in existing_data[key]:
                    existing_data[key][sub_key] = pd.concat(
                        [existing_data[key][sub_key], sub_df_new]
                    ).drop_duplicates(subset='datetime').sort_values(by='datetime')
                else:
                    if key not in existing_data:
                        existing_data[key] = {}
                    existing_data[key][sub_key] = sub_df_new
        else:
            if key in existing_data:
                existing_data[key] = pd.concat(
                    [existing_data[key], df_new]
                ).drop_duplicates(subset='datetime').sort_values(by='datetime')
            else:
                existing_data[key] = df_new
    return existing_data

def save_to_pickle(data, filename='local_data/historical_data.pickle'):
    """
    Save the updated data to the pickle file.
    """
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"\nData saved to {filename}.")

if __name__ == "__main__":
    # Step 1: Import the data
    spy_data, vix_data, iv_data, gdp_data, cpi_data, sentiment_data, articles = import_data()

    # Step 2: Clean and normalize the data
    cleaned_data = clean_and_normalize(spy_data, vix_data, iv_data, gdp_data, cpi_data, sentiment_data, articles)

    # Step 3: Load existing data if available
    existing_data = load_existing_data()

    if existing_data:
        # Step 4: Append the new data to the existing data
        updated_data = append_new_data(existing_data, cleaned_data)
    else:
        # If no existing data, use the new data directly
        updated_data = cleaned_data

    # Step 5: Save the updated data to the pickle file
    save_to_pickle(updated_data)
