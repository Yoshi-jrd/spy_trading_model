import yfinance as yf
import pandas as pd

def get_spy_vix_correlation(spy_data, vix_data, lookback=10):
    """
    Calculate the rolling correlation between SPY and VIX returns.
    Ensure timezone consistency and aligned indexes.
    """
    # Convert VIX data to timezone-naive
    if vix_data.index.tz is not None:
        vix_data.index = vix_data.index.tz_localize(None)
    
    # Convert SPY data to timezone-naive
    if spy_data.index.tz is not None:
        spy_data.index = spy_data.index.tz_localize(None)

    # Align indexes if needed
    common_index = spy_data.index.intersection(vix_data.index)
    spy_data = spy_data.reindex(common_index)
    vix_data = vix_data.reindex(common_index)

    spy_data['SPY_Returns'] = spy_data['Close'].pct_change()
    vix_data['VIX_Returns'] = vix_data['Close'].pct_change()

    correlation = spy_data['SPY_Returns'].rolling(window=lookback).corr(vix_data['VIX_Returns'])
    return correlation




def main():
    # Fetch SPY and VIX price data
    spy_data = yf.download("SPY", period="1mo", interval="1d")
    vix_data = yf.download("^VIX", period="1mo", interval="1d")
    
    # Calculate SPY-VIX correlation
    correlation = get_spy_vix_correlation(spy_data, vix_data, lookback=10)
    print(correlation.tail())  # Print last few correlation values

if __name__ == "__main__":
    main()
