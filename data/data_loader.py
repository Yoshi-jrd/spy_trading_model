import yfinance as yf
import pandas as pd
import csv
from fredapi import Fred
from newsapi import NewsApiClient
from datetime import datetime, timedelta
from textblob import TextBlob

# Core function for calculating indicators
def calculate_indicators(df):
    """
    Calculates all necessary indicators for a given DataFrame.
    """
    # MACD
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False, min_periods=1).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False, min_periods=1).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False, min_periods=1).mean()

    # RSI
    delta = df['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df['SMA20'] = df['Close'].rolling(window=20, min_periods=1).mean()
    df['UpperBand'] = df['SMA20'] + 2 * df['Close'].rolling(window=20, min_periods=1).std()
    df['LowerBand'] = df['SMA20'] - 2 * df['Close'].rolling(window=20, min_periods=1).std()

    # Stochastic Oscillator
    low_min = df['Low'].rolling(window=14, min_periods=1).min()
    high_max = df['High'].rolling(window=14, min_periods=1).max()
    df['%K'] = (df['Close'] - low_min) * 100 / (high_max - low_min)
    df['%D'] = df['%K'].rolling(window=3, min_periods=1).mean()

    # ATR
    df['PrevClose'] = df['Close'].shift(1)
    df['TR'] = df[['High', 'Low', 'PrevClose']].apply(
        lambda row: max(row['High'] - row['Low'], abs(row['High'] - row['PrevClose']), abs(row['Low'] - row['PrevClose'])), axis=1)
    df['ATR'] = df['TR'].rolling(window=14, min_periods=1).mean()

    # ADX
    plus_dm = df['High'].diff().clip(lower=0)
    minus_dm = df['Low'].diff().clip(upper=0).abs()
    tr14 = df['TR'].rolling(window=14, min_periods=1).sum()
    plus_dm14 = plus_dm.rolling(window=14, min_periods=1).sum()
    minus_dm14 = minus_dm.rolling(window=14, min_periods=1).sum()
    df['PlusDI'] = 100 * (plus_dm14 / tr14)
    df['MinusDI'] = 100 * (minus_dm14 / tr14)
    df['ADX'] = 100 * (abs(df['PlusDI'] - df['MinusDI']) / (df['PlusDI'] + df['MinusDI'])).rolling(window=14, min_periods=1).mean()

    # EMA Crossovers
    df['EMA9'] = df['Close'].ewm(span=9, adjust=False, min_periods=1).mean()
    df['EMA21'] = df['Close'].ewm(span=21, adjust=False, min_periods=1).mean()

    # OBV
    df['OBV'] = (df['Volume'] * ((df['Close'] > df['Close'].shift(1)) * 2 - 1)).cumsum()

    # MFI
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    mf = tp * df['Volume']
    pos_mf = mf.where(tp > tp.shift(1), 0).rolling(window=14, min_periods=1).sum()
    neg_mf = mf.where(tp < tp.shift(1), 0).rolling(window=14, min_periods=1).sum()
    df['MFI'] = 100 - (100 / (1 + pos_mf / neg_mf))

    return df

# Load SPY data for multiple timeframes and calculate indicators
def load_spy_multi_timeframes():
    timeframes = {
        "5m": calculate_indicators(load_spy_data("1mo", "5m")),
        "15m": calculate_indicators(load_spy_data("1mo", "15m")),
        "1h": calculate_indicators(load_spy_data("6mo", "1h")),
        "1d": calculate_indicators(load_spy_data("1y", "1d")),
    }
    return timeframes

# Load SPY historical data
def load_spy_data(period="1mo", interval="1h"):
    spy = yf.Ticker("SPY")
    spy_data = spy.history(period=period, interval=interval)
    spy_data.reset_index(inplace=True)  # Reset index for easier manipulation
    return spy_data

# VIX Data Loader
def get_vix_futures():
    vix = yf.Ticker("^VIX")  # VIX ticker symbol
    vix_data = vix.history(period="1mo", interval="1d")
    vix_data.reset_index(inplace=True)
    return vix_data

# Implied Volatility (IV) Data Loader
def get_filtered_iv(expiry="2024-10-18", strike_range=0.1):
    spy = yf.Ticker("SPY")
    opt = spy.option_chain(expiry)
    current_price = spy.history(period="1d")['Close'].iloc[-1]
    atm_calls = opt.calls[(abs(opt.calls['strike'] - current_price) / current_price) < strike_range]
    atm_puts = opt.puts[(abs(opt.puts['strike'] - current_price) / current_price) < strike_range]
    avg_iv_calls = atm_calls['impliedVolatility'].mean()
    avg_iv_puts = atm_puts['impliedVolatility'].mean()
    avg_iv = (avg_iv_calls + avg_iv_puts) / 2
    return avg_iv

# Economic Data Loader (FRED)
def load_economic_data():
    fred = Fred(api_key="b4405278e8f6f6f01934fbae7d1953e3")  # Add your FRED API key here
    gdp_data = fred.get_series('GDP')
    cpi_data = fred.get_series('CPIAUCSL')
    return gdp_data, cpi_data

# Function to read the last row in the CSV
def get_last_saved_date(csv_file='sentiment_history.csv'):
    try:
        with open(csv_file, mode='r') as file:
            reader = csv.reader(file)
            rows = list(reader)
            if len(rows) > 0:
                last_row = rows[-1]
                last_date = datetime.strptime(last_row[0], '%Y-%m-%d %H:%M:%S')
                return last_date
            return None
    except FileNotFoundError:
        return None

# Function to append sentiment score to CSV if the date is new
def save_sentiment_score(score, csv_file='sentiment_history.csv'):
    last_saved_date = get_last_saved_date(csv_file)
    current_date = datetime.now()
    
    # Check if the current date (without time) matches the last saved date
    if last_saved_date is None or current_date.date() != last_saved_date.date():
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([current_date.strftime('%Y-%m-%d %H:%M:%S'), score])
            print(f"Sentiment score saved for {current_date.strftime('%Y-%m-%d')}.")
    else:
        print(f"Data for {current_date.strftime('%Y-%m-%d')} already exists, skipping save.")

# News Sentiment Analysis
def get_news_sentiment(query="SPY stock market sentiment OR investor confidence", days_ago=30):
    print(datetime.now())
    print(datetime.now()-timedelta(days=days_ago))
    newsapi = NewsApiClient(api_key="458f35cd1d794a4f9640174d1aa548d9")  # Add your News API key here
    to_date = datetime.now().strftime('%Y-%m-%d')  # Current date
    from_date = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')
    articles = newsapi.get_everything(
        q=query,
        from_param=from_date,
        to=to_date,
        language="en",
        sort_by="relevancy"
    )
    sentiments = [TextBlob(article['title']).sentiment.polarity for article in articles['articles']]
    avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
    save_sentiment_score(avg_sentiment)
    return avg_sentiment, articles['articles']

# Load all the required data
def load_data():
    spy_df = load_spy_multi_timeframes()  # Load SPY data
    vix_df = get_vix_futures()            # Load VIX data
    iv_data = get_filtered_iv()           # Load IV data
    econ_data = load_economic_data()      # Load economic data
    sentiment_data = get_news_sentiment() # Fetch news sentiment data

    # Return all the data in a structured format
    data = {
        'spy_data': spy_df,
        'vix_data': vix_df,
        'iv_data': iv_data,
        'economic_data': econ_data,
        'sentiment_data': sentiment_data
    }

    return data
