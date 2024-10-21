import os
import sys
import pickle  # Import pickle for saving data
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from data.market_data_loader import load_spy_multi_timeframes, get_vix_futures, get_all_iv
from data.economic_data_loader import load_economic_data
from data.sentiment_data_loader import get_news_sentiment

def preprocess_data(df, name="data"):
    """
    Clean and preprocess the data by handling NaN values and ensuring the right shape.
    """
    # Fill NaN values and drop any remaining NaNs
    df = df.ffill().dropna()

    # Check for any remaining NaN values
    if df.isna().sum().sum() == 0:
        print(f"{name} is clean with shape {df.shape}")
    else:
        print(f"Warning: {name} still contains NaNs after preprocessing.")

    return df

def load_cleaned_data():
    """
    Load and clean all the necessary data for the model.
    """
    # Load data using your existing functions
    spy_data = load_spy_multi_timeframes()
    vix_data = get_vix_futures()
    iv_data = get_all_iv()
    gdp_data, cpi_data = load_economic_data()
    sentiment_data, articles = get_news_sentiment()

    # Preprocess each DataFrame
    cleaned_data = {
        'spy_data': {tf: preprocess_data(spy_data[tf], f"SPY {tf}") for tf in spy_data},
        'vix_data': preprocess_data(vix_data, "VIX"),
        'iv_data': preprocess_data(iv_data, "IV"),
        'gdp_data': preprocess_data(gdp_data, "GDP"),
        'cpi_data': preprocess_data(cpi_data, "CPI"),
        'sentiment_data': sentiment_data,  # Assuming sentiment_data is not a DataFrame
        'articles': articles  # Assuming articles are not a DataFrame
    }

    # Print summary for review
    print("Data cleaning complete. Here's a summary of the cleaned data:")
    for key, value in cleaned_data.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, pd.DataFrame) or isinstance(sub_value, pd.Series):
                    print(f"{key} - {sub_key}: {sub_value.shape}")
                else:
                    print(f"{key} - {sub_key}: {type(sub_value)}")
        elif isinstance(value, pd.DataFrame) or isinstance(value, pd.Series):
            print(f"{key}: {value.shape}")
        else:
            print(f"{key}: {type(value)}")

    return cleaned_data

from data.data_loader import load_cleaned_data
# from utils.data_storage import append_data_to_pickle, upload_to_firebase  # Commented out for now

def append_data_to_pickle(new_data):
    """
    Save the cleaned data to a pickle file.
    """
    file_path = os.path.join('local_data', "historical_data.pickle")

    # Load the existing data if the pickle file exists
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            existing_data = pickle.load(f)
        print(f"Loaded existing historical data from {file_path}")
    else:
        existing_data = {}
        print(f"No existing data found, creating a new file.")

    # Append only new data
    for key, value in new_data.items():
        if key in existing_data:
            if isinstance(existing_data[key], pd.DataFrame):
                combined_data = pd.concat([existing_data[key], value])
                existing_data[key] = combined_data.drop_duplicates()  # Ensures only unique rows remain
            else:
                print(f"Warning: {key} is not a DataFrame.")
        else:
            existing_data[key] = value

    # Save the updated data back to the pickle file
    with open(file_path, 'wb') as f:
        pickle.dump(existing_data, f)
    print(f"Appended new historical data and saved to {file_path}")

def review_data():
    """
    Review cleaned data before passing it to data storage.
    """
    cleaned_data = load_cleaned_data()  # Load and clean the data
    
    # Prompt for review confirmation
    proceed = input("Do you want to proceed with saving this data? (y/n): ")
    if proceed.lower() == 'y':
        append_data_to_pickle(cleaned_data)  # Save the cleaned data
        local_file_path = os.path.join('local_data', "historical_data.pickle")
        # upload_to_firebase(local_file_path, "data/historical_data.pickle")  # Upload the pickle file (commented out for now)
    else:
        print("Data saving and uploading aborted.")

if __name__ == '__main__':
    review_data()
