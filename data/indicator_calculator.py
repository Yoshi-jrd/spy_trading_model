import pandas as pd

def calculate_indicators(df):
    df = df.copy()

    # MACD
    df['EMA12'] = df['Close'].ewm(span=8, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=21, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']

    # Print to verify MACD_Histogram values
    print("\n--- MACD Histogram Preview ---")
    print(df[['MACD', 'MACD_Signal', 'MACD_Histogram']].head())

    # RSI
    delta = df['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['UpperBand'] = df['SMA20'] + 2 * df['Close'].rolling(window=20).std()
    df['LowerBand'] = df['SMA20'] - 2 * df['Close'].rolling(window=20).std()

    # Stochastic Oscillator
    low_min = df['Low'].rolling(window=14).min()
    high_max = df['High'].rolling(window=14).max()
    df['%K'] = (df['Close'] - low_min) * 100 / (high_max - low_min)
    df['%D'] = df['%K'].rolling(window=3).mean()

    # ATR
    df['PrevClose'] = df['Close'].shift(1)
    df['TR'] = df[['High', 'Low', 'PrevClose']].apply(
        lambda row: max(row['High'] - row['Low'], abs(row['High'] - row['PrevClose']), abs(row['Low'] - row['PrevClose'])), axis=1
    )
    df['ATR'] = df['TR'].rolling(window=14).mean()

    # ADX
    plus_dm = df['High'].diff().clip(lower=0)
    minus_dm = df['Low'].diff().clip(upper=0).abs()
    tr14 = df['TR'].rolling(window=14).sum()
    plus_dm14 = plus_dm.rolling(window=14).sum()
    minus_dm14 = minus_dm.rolling(window=14).sum()
    df['PlusDI'] = 100 * (plus_dm14 / tr14)
    df['MinusDI'] = 100 * (minus_dm14 / tr14)
    df['ADX'] = 100 * (abs(df['PlusDI'] - df['MinusDI']) / (df['PlusDI'] + df['MinusDI'])).rolling(window=14).mean()

    # EMA Crossovers
    df['EMA9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()

    # OBV (On Balance Volume)
    df['OBV'] = (df['Volume'] * ((df['Close'] > df['Close'].shift(1)) * 2 - 1)).cumsum()

    # MFI (Money Flow Index)
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    mf = tp * df['Volume']
    pos_mf = mf.where(tp > tp.shift(1), 0).rolling(window=14).sum()
    neg_mf = mf.where(tp < tp.shift(1), 0).rolling(window=14).sum()
    df['MFI'] = 100 - (100 / (1 + pos_mf / neg_mf))

    # Impulse MACD and RSI (LazyBear)
    df['Impulse_MACD'] = df['MACD']
    df['Impulse_RSI'] = df['RSI']

    # Calculate color signal for the Impulse MACD using MACD_Histogram
    df['Impulse_Color'] = 'gray'  # Neutral by default
    df.loc[(df['MACD_Histogram'] > 0) & (df['RSI'] > df['RSI'].shift(1)), 'Impulse_Color'] = 'green'  # Bullish
    df.loc[(df['MACD_Histogram'] < 0) & (df['RSI'] < df['RSI'].shift(1)), 'Impulse_Color'] = 'red'  # Bearish

    # Forward fill missing values and fill NaN with 0
    df.ffill(inplace=True)
    df.fillna(0, inplace=True)

    return df
