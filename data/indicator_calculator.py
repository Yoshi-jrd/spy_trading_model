import pandas as pd

def calculate_indicators(df):
    """
    Calculate prioritized indicators for 24-96 hour trade predictions.
    Focuses on high-impact indicators only, streamlining calculations.
    """
    df = df.copy()

    # MACD - Focus on Histogram as a key momentum indicator
    df['EMA12'] = df['Close'].ewm(span=8, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=21, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    print("\n--- MACD Histogram (Key Momentum Indicator) ---")
    print(df[['MACD', 'MACD_Signal', 'MACD_Histogram']].head())

    # RSI - Key Overbought/Oversold Indicator
    delta = df['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Ensure no division by zero in RSI calculation
    df['RSI'].fillna(0, inplace=True)
    if (df['RSI'] == 0).all():
        print("Warning: All RSI values are zero.")

    # Bollinger Bands for volatility analysis
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['UpperBand'] = df['SMA20'] + 2 * df['Close'].rolling(window=20).std()
    df['LowerBand'] = df['SMA20'] - 2 * df['Close'].rolling(window=20).std()
    
    print("\n--- Bollinger Bands ---")
    print(df[['SMA20', 'UpperBand', 'LowerBand']].head())

    # ATR - Average True Range for volatility assessment
    df['PrevClose'] = df['Close'].shift(1)
    df['TR'] = df[['High', 'Low', 'PrevClose']].apply(
        lambda row: max(row['High'] - row['Low'], abs(row['High'] - row['PrevClose']), abs(row['Low'] - row['PrevClose'])), axis=1
    )
    df['ATR'] = df['TR'].rolling(window=14).mean()
    
    print("\n--- ATR Calculation ---")
    print(df[['TR', 'ATR']].head())
    if (df['ATR'] == 0).all():
        print("Warning: All ATR values are zero.")

    # ADX - Trend strength indicator
    plus_dm = df['High'].diff().clip(lower=0)
    minus_dm = df['Low'].diff().clip(upper=0).abs()
    tr14 = df['TR'].rolling(window=14).sum()
    plus_dm14 = plus_dm.rolling(window=14).sum()
    minus_dm14 = minus_dm.rolling(window=14).sum()
    df['PlusDI'] = 100 * (plus_dm14 / tr14)
    df['MinusDI'] = 100 * (minus_dm14 / tr14)
    df['ADX'] = 100 * (abs(df['PlusDI'] - df['MinusDI']) / (df['PlusDI'] + df['MinusDI'])).rolling(window=14).mean()

    # Handle potential NaNs or zeros in ADX calculations
    df['ADX'].fillna(0, inplace=True)
    if (df['ADX'] == 0).all():
        print("Warning: All ADX values are zero.")
    
    print("\n--- ADX Calculation ---")
    print(df[['PlusDI', 'MinusDI', 'ADX']].head())

    # EMA Crossovers - For trend direction
    df['EMA9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()
    
    print("\n--- EMA Crossovers Preview ---")
    print(df[['EMA9', 'EMA21']].head())

    # MFI (Money Flow Index) for money flow assessment
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    mf = tp * df['Volume']
    pos_mf = mf.where(tp > tp.shift(1), 0).rolling(window=14).sum()
    neg_mf = mf.where(tp < tp.shift(1), 0).rolling(window=14).sum()
    df['MFI'] = 100 - (100 / (1 + pos_mf / neg_mf))
    
    # Handle NaN in MFI and verify values
    df['MFI'].fillna(0, inplace=True)
    if (df['MFI'] == 0).all():
        print("Warning: All MFI values are zero.")
    
    # Impulse MACD and RSI - LazyBear (Key for bullish/bearish conditions)
    df['Impulse_MACD'] = df['MACD']
    df['Impulse_RSI'] = df['RSI']
    
    # Assign colors for the Impulse MACD, helpful for visual sentiment representation
    df['Impulse_Color'] = 'gray'  # Neutral by default
    df.loc[(df['MACD_Histogram'] > 0) & (df['RSI'] > df['RSI'].shift(1)), 'Impulse_Color'] = 'green'  # Bullish
    df.loc[(df['MACD_Histogram'] < 0) & (df['RSI'] < df['RSI'].shift(1)), 'Impulse_Color'] = 'red'  # Bearish

    # Forward fill and fill NaN with 0 to handle any missing values
    df.ffill(inplace=True)
    df.fillna(0, inplace=True)

    print("\n--- Indicator Calculation Complete ---")
    print(df[['MACD_Histogram', 'RSI', 'UpperBand', 'LowerBand', 'ATR', 'ADX', 'Impulse_Color']].head())

    return df
