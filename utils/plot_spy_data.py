import pickle
import pandas as pd
import matplotlib.pyplot as plt
import os

def load_pickle_data(file_path='local_data/historical_data.pickle'):
    """
    Load historical data from a pickle file.
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    print("Data loaded successfully.")
    return data

def plot_spy_data(data, timeframe):
    """
    Plot SPY 'Close' price data for a specific timeframe.
    """
    spy_data = data.get('spy_data', {}).get(timeframe, pd.DataFrame())
    
    if spy_data.empty:
        print(f"No data available for timeframe: {timeframe}")
        return
    
    # Ensure datetime is properly formatted for plotting
    spy_data['datetime'] = pd.to_datetime(spy_data['datetime'], errors='coerce')
    spy_data.set_index('datetime', inplace=True)
    
    plt.figure(figsize=(12, 6))
    plt.plot(spy_data.index, spy_data['Close'], label=f'SPY Close Price - {timeframe}', color='blue')
    plt.title(f'SPY Close Price ({timeframe})')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # Load data
    data = load_pickle_data()

    if data is None:
        print("No data loaded. Exiting...")
        return
    
    # Plot for 15-minute and 1-hour timeframes
    print("Plotting 15-minute data...")
    plot_spy_data(data, '15m')
    
    print("Plotting 1-hour data...")
    plot_spy_data(data, '1h')

if __name__ == "__main__":
    main()
