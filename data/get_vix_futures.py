import yfinance as yf

def get_vix_futures():
    """
    Fetch VIX futures prices and use the index as the Date.
    """
    vix = yf.Ticker("^VIX")
    vix_futures = vix.history(period="1mo", interval="15m")
    
    # Use the index as the Date, no need to create a duplicate 'Date' column
    print(vix_futures[['Open', 'High', 'Low', 'Close']].tail())  # Print relevant VIX data
    return vix_futures



def main():
    # Get VIX futures data
    vix_futures = get_vix_futures()
    print(vix_futures.tail())  # Print latest VIX futures data

if __name__ == "__main__":
    main()
