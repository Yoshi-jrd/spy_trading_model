import yfinance as yf
import pandas as pd
from data.indicator_calculator import calculate_indicators

def load_spy_data_with_indicators(period="1mo", interval="1h"):
    print(f"Loading SPY data with period='{period}' and interval='{interval}'...")
    spy = yf.Ticker("SPY")
    spy_data = spy.history(period=period, interval=interval)
    
    if spy_data.empty:
        print("Warning: SPY data is empty!")
    else:
        print(f"SPY data loaded. Shape: {spy_data.shape}")

    spy_data.reset_index(inplace=True)
    print("Applying indicator calculations...")
    spy_data_with_indicators = calculate_indicators(spy_data)
    print("Indicators applied successfully.")

    return spy_data_with_indicators

def load_spy_multi_timeframes():
    print("Loading SPY data for multiple timeframes...")
    timeframes = {
        "5m": load_spy_data_with_indicators("1mo", "5m"),
        "15m": load_spy_data_with_indicators("1mo", "15m"),
        "1h": load_spy_data_with_indicators("6mo", "1h"),
        "1d": load_spy_data_with_indicators("1y", "1d"),
    }
    
    for tf, df in timeframes.items():
        if df.empty:
            print(f"Warning: SPY data for {tf} timeframe is empty.")
        else:
            print(f"SPY data for {tf} timeframe loaded. Shape: {df.shape}")
            print(df[['MACD_Histogram', 'RSI', 'UpperBand', 'LowerBand', 'ATR', 'ADX', 'Impulse_Color']].head())
    
    return timeframes

def get_vix_futures():
    print("Loading VIX futures data...")
    vix = yf.Ticker("^VIX")
    vix_data = vix.history(period="1mo", interval="1d")
    vix_data.reset_index(inplace=True)
    vix_data.ffill(inplace=True)
    vix_data.fillna(0, inplace=True)
    print("VIX data loaded and cleaned.")
    return vix_data

def get_all_iv(strike_range=0.1):
    print("Loading IV data for SPY options...")
    spy = yf.Ticker("SPY")
    available_expirations = spy.options
    iv_data = []

    current_price = spy.history(period="1d")['Close'].iloc[-1]

    for expiry in available_expirations:
        opt = spy.option_chain(expiry)
        atm_calls = opt.calls[(abs(opt.calls['strike'] - current_price) / current_price) < strike_range]
        atm_puts = opt.puts[(abs(opt.puts['strike'] - current_price) / current_price) < strike_range]

        avg_iv_calls = atm_calls['impliedVolatility'].mean()
        avg_iv_puts = atm_puts['impliedVolatility'].mean()
        avg_iv = (avg_iv_calls + avg_iv_puts) / 2

        iv_data.append({
            'expiry': expiry,
            'avg_iv_calls': avg_iv_calls,
            'avg_iv_puts': avg_iv_puts,
            'avg_iv': avg_iv
        })

    iv_df = pd.DataFrame(iv_data)
    iv_df.ffill(inplace=True)
    iv_df.fillna(0, inplace=True)
    print("IV data loaded and processed.")
    return iv_df
