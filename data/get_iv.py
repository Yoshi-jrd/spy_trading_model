import yfinance as yf

def get_filtered_iv(expiry="2024-10-18", strike_range=0.12):
    """
    Get Implied Volatility (IV) for near at-the-money (ATM) options expiring soon.
    Filters options within a given range (strike_range) of the current SPY price.
    """
    spy = yf.Ticker("SPY")
    opt = spy.option_chain(expiry)
    
    # Get SPY's current price
    current_price = spy.history(period="1d")['Close'].iloc[-1]  # Use a longer period to get a better price average
    
    # Filter for near-ATM options (strike within strike_range of the current price)
    atm_calls = opt.calls[(abs(opt.calls['strike'] - current_price) / current_price) < strike_range]
    atm_puts = opt.puts[(abs(opt.puts['strike'] - current_price) / current_price) < strike_range]

    # Exclude deep OTM options and missing IV values
    atm_calls = atm_calls[atm_calls['impliedVolatility'] > 0].dropna(subset=['impliedVolatility'])
    atm_puts = atm_puts[atm_puts['impliedVolatility'] > 0].dropna(subset=['impliedVolatility'])

    # Calculate average IV for both calls and puts
    avg_iv_calls = atm_calls['impliedVolatility'].mean()
    avg_iv_puts = atm_puts['impliedVolatility'].mean()
    
    # Calculate combined average IV
    avg_iv = (avg_iv_calls + avg_iv_puts) / 2

    print(f"Filtered Call IV: {avg_iv_calls}")
    print(f"Filtered Put IV: {avg_iv_puts}")
    print(f"Filtered Average IV (Call & Put): {avg_iv}")

    return avg_iv


def main():
    # Get Implied Volatility data
    iv = get_filtered_iv()
    print(f"Test Implied Volatility: {iv}")

if __name__ == "__main__":
    main()
