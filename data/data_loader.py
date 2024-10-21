from data.market_data_loader import load_spy_multi_timeframes, get_vix_futures, get_all_iv
from data.economic_data_loader import load_economic_data
from data.sentiment_data_loader import get_news_sentiment
import pandas as pd

def load_data():
    spy_df = load_spy_multi_timeframes()  # Load SPY data
    vix_df = get_vix_futures()            # Load VIX data
    iv_data = get_all_iv()                # Load IV data
    gdp_data, cpi_data = load_economic_data()  # Load economic data
    avg_sentiment, articles = get_news_sentiment()  # Fetch news sentiment data

    def check_nan_in_data(data, name):
        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            if data.isna().sum().sum() > 0:
                print(f"Warning: {name} contains NaN values.")
            else:
                print(f"{name} contains no NaN values.")
        else:
            print(f"{name} is not a DataFrame or Series.")

    for timeframe, df in spy_df.items():
        check_nan_in_data(df, f"SPY data ({timeframe})")
    
    check_nan_in_data(vix_df, "VIX data")
    check_nan_in_data(iv_data, "IV data")
    check_nan_in_data(gdp_data, "GDP data")
    check_nan_in_data(cpi_data, "CPI data")

    # Structure for final data dictionary
    data = {
        'spy_data': spy_df,
        'vix_data': vix_df,
        'iv_data': iv_data,
        'gdp_data': gdp_data,
        'cpi_data': cpi_data,
        'sentiment_data': avg_sentiment,
        'articles': articles
    }

    # Print the structure of the data for review
    print("Final data structure:")
    for key, value in data.items():
        if isinstance(value, pd.DataFrame) or isinstance(value, pd.Series):
            print(f"{key}: {value.shape}")  # Print shape for DataFrame/Series
        else:
            print(f"{key}: {type(value)}")  # Print type for non-DataFrame/Series

    return data

# Example of how to call this function:
if __name__ == "__main__":
    loaded_data = load_data()

