import pandas as pd
import numpy as np

def generate_signals(data):
    signals = pd.DataFrame(index=data.index)
    signals['RSI'] = data['RSI']
    signals['MACD'] = data['MACD']
    signals['MACD_Signal'] = data['MACD_Signal']
    signals['Stochastic_K'] = data['Stochastic_K']
    signals['ATR'] = data['ATR']
    signals['Buy'] = np.where((signals['RSI'] < 30) & (signals['MACD'] > signals['MACD_Signal']), 1, 0)
    signals['Sell'] = np.where((signals['RSI'] > 70) & (signals['MACD'] < signals['MACD_Signal']), 1, 0)
    return signals

if __name__ == "__main__":
    ticker = 'RELIANCE.NS'
    data = pd.read_csv(f'{ticker}_data_with_indicators.csv', parse_dates=True, index_col=0)

    signals = generate_signals(data)

    signals.to_csv(f'{ticker}_buy_sell_signals.csv')
    print(signals.tail())
