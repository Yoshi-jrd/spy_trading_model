import numpy as np
import yfinance as yf

def get_hv(df, lookback=10):
    """
    Calculate Historical Volatility (HV) based on SPY's closing prices.
    """
    df['Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    hv = df['Returns'].rolling(window=lookback).std() * np.sqrt(252)  # Annualized HV
    print(f"Historical Volatility (HV) over {lookback} days: {hv.iloc[-1]}")
    return hv

def main():
    # Fetch SPY price data
    spy_data = yf.download("SPY", period="1mo", interval="1d")
    
    # Get Historical Volatility
    hv = get_hv(spy_data, lookback=10)
    print(hv.tail())  # Print last few HV values for review

if __name__ == "__main__":
    main()
