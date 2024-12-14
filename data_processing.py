import pandas as pd
import numpy as np

def calculate_indicators(data):
    # Ensure relevant columns are numeric
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
    data['High'] = pd.to_numeric(data['High'], errors='coerce')
    data['Low'] = pd.to_numeric(data['Low'], errors='coerce')
    data = data.dropna(subset=['Close', 'High', 'Low'])  # Drop rows with NaN values in relevant columns

    # Calculate RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # Calculate MACD
    fast_ema = data['Close'].ewm(span=12, adjust=False).mean()
    slow_ema = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = fast_ema - slow_ema
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

    # Calculate Bollinger Bands
    rolling_mean = data['Close'].rolling(window=20).mean()
    rolling_std = data['Close'].rolling(window=20).std()
    data['Bollinger_High'] = rolling_mean + (rolling_std * 2)
    data['Bollinger_Low'] = rolling_mean - (rolling_std * 2)

    # Calculate Stochastic Oscillator
    low_min = data['Low'].rolling(window=14).min()
    high_max = data['High'].rolling(window=14).max()
    data['Stochastic_K'] = 100 * ((data['Close'] - low_min) / (high_max - low_min))

    # Calculate Average True Range (ATR)
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    tr = high_low.combine(high_close, max).combine(low_close, max)
    data['ATR'] = tr.rolling(window=14).mean()

    return data

if __name__ == "__main__":
    ticker = 'RELIANCE.NS'
    data = pd.read_csv(f'{ticker}_historical_data.csv', parse_dates=True, index_col=0)

    data = calculate_indicators(data)
    data.to_csv(f'{ticker}_data_with_indicators.csv')
    print(data.tail())
