import pandas as pd
import numpy as np

def generate_trade_signals(df):
    """
    Analyzes the data for various options strategies based on indicator logic.
    
    Parameters:
    - df: DataFrame containing SPY price data and calculated indicators ('MACD', 'RSI', 'UpperBand', 'LowerBand', 'ATR', 'EMA9', 'EMA21', etc.)
    
    Returns:
    - DataFrame: A DataFrame containing the signals for different options strategies.
    """

    # Create a DataFrame to store signals
    signals = pd.DataFrame(index=df.index)

    # 1. Debit Spread Signals (Directional)
    signals['debit_spread_call'] = np.where(
        (df['EMA9'] > df['EMA21']) & (df['MACD'] > df['MACD_Signal']) & (df['RSI'] < 70), 1, 0
    )
    
    signals['debit_spread_put'] = np.where(
        (df['EMA9'] < df['EMA21']) & (df['MACD'] < df['MACD_Signal']) & (df['RSI'] > 30), 1, 0
    )

    # 2. Credit Spread Signals (Range-Bound)
    signals['credit_spread_put'] = np.where(
        (df['Close'] > df['LowerBand']) & (df['RSI'] > 40) & (df['EMA9'] > df['EMA21']) & (df['ATR'] < df['ATR'].rolling(window=10).mean()), 1, 0
    )
    
    signals['credit_spread_call'] = np.where(
        (df['Close'] < df['UpperBand']) & (df['RSI'] < 60) & (df['EMA9'] < df['EMA21']) & (df['ATR'] < df['ATR'].rolling(window=10).mean()), 1, 0
    )

    # 3. Condor Signals (Range-Bound, Low Volatility)
    signals['iron_condor'] = np.where(
        (df['UpperBand'] - df['LowerBand'] < df['ATR'] * 2) & (df['ATR'] < df['ATR'].rolling(window=10).mean()), 1, 0
    )

    # 4. Straddle/Strangle Signals (Volatility Plays)
    signals['straddle'] = np.where(
        (df['UpperBand'] - df['LowerBand'] > df['ATR'] * 2) & (df['RSI'] > 40) & (df['RSI'] < 60), 1, 0
    )
    
    signals['strangle'] = np.where(
        (df['UpperBand'] - df['LowerBand'] > df['ATR'] * 2) & (df['RSI'] > 40) & (df['RSI'] < 60), 1, 0
    )

    # 5. Butterfly Signals (Neutral, Low Volatility)
    signals['butterfly'] = np.where(
        (df['UpperBand'] - df['LowerBand'] < df['ATR']) & (df['RSI'] > 40) & (df['RSI'] < 60), 1, 0
    )

    # Return the final signals DataFrame
    return signals
