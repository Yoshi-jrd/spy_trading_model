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

    # Print for validation of MACD values
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

    # Bollinger Bands for volatility analysis
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['UpperBand'] = df['SMA20'] + 2 * df['Close'].rolling(window=20).std()
    df['LowerBand'] = df['SMA20'] - 2 * df['Close'].rolling(window=20).std()

    # ATR - Average True Range for volatility assessment
    df['PrevClose'] = df['Close'].shift(1)
    df['TR'] = df[['High', 'Low', 'PrevClose']].apply(
        lambda row: max(row['High'] - row['Low'], abs(row['High'] - row['PrevClose']), abs(row['Low'] - row['PrevClose'])), axis=1
    )
    df['ATR'] = df['TR'].rolling(window=14).mean()

    # ADX - Trend strength indicator
    plus_dm = df['High'].diff().clip(lower=0)
    minus_dm = df['Low'].diff().clip(upper=0).abs()
    tr14 = df['TR'].rolling(window=14).sum()
    plus_dm14 = plus_dm.rolling(window=14).sum()
    minus_dm14 = minus_dm.rolling(window=14).sum()
    df['PlusDI'] = 100 * (plus_dm14 / tr14)
    df['MinusDI'] = 100 * (minus_dm14 / tr14)
    df['ADX'] = 100 * (abs(df['PlusDI'] - df['MinusDI']) / (df['PlusDI'] + df['MinusDI'])).rolling(window=14).mean()

    # EMA Crossovers - For trend direction
    df['EMA9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()

    # Print for EMA crossover verification
    print("\n--- EMA Crossovers Preview ---")
    print(df[['EMA9', 'EMA21']].head())

    # MFI (Money Flow Index) for money flow assessment
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    mf = tp * df['Volume']
    pos_mf = mf.where(tp > tp.shift(1), 0).rolling(window=14).sum()
    neg_mf = mf.where(tp < tp.shift(1), 0).rolling(window=14).sum()
    df['MFI'] = 100 - (100 / (1 + pos_mf / neg_mf))

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
