import pandas as pd
import matplotlib.pyplot as plt

def plot_signals(data, signals, ticker):
    plt.figure(figsize=(14, 10))

    # Plot closing price with buy/sell signals and Bollinger Bands
    plt.subplot(4, 1, 1)
    plt.plot(data['Close'], label='Close Price', color='blue')
    plt.plot(data['Bollinger_High'], label='Bollinger High', linestyle='--', color='red')
    plt.plot(data['Bollinger_Low'], label='Bollinger Low', linestyle='--', color='green')
    plt.scatter(signals.loc[signals['Buy'] == 1].index, data['Close'][signals['Buy'] == 1], label='Buy Signal', marker='^', color='green', alpha=1)
    plt.scatter(signals.loc[signals['Sell'] == 1].index, data['Close'][signals['Sell'] == 1], label='Sell Signal', marker='v', color='red', alpha=1)
    plt.title(f'{ticker} - Stock Price with Bollinger Bands and Buy/Sell Signals')
    plt.legend()

    # Plot RSI
    plt.subplot(4, 1, 2)
    plt.plot(signals['RSI'], label='RSI', color='orange')
    plt.axhline(70, color='red', linestyle='--')
    plt.axhline(30, color='green', linestyle='--')
    plt.title(f'{ticker} - RSI')
    plt.legend()

    # Plot MACD and Signal Line
    plt.subplot(4, 1, 3)
    plt.plot(data['MACD'], label='MACD', color='purple')
    plt.plot(data['MACD_Signal'], label='MACD Signal Line', color='red')
    plt.title(f'{ticker} - MACD')
    plt.legend()

    # Plot Stochastic Oscillator
    plt.subplot(4, 1, 4)
    plt.plot(data['Stochastic_K'], label='Stochastic %K', color='magenta')
    plt.axhline(80, color='red', linestyle='--')
    plt.axhline(20, color='green', linestyle='--')
    plt.title(f'{ticker} - Stochastic Oscillator')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{ticker}_enhanced_signals_plot.png')  # Save the plot as an image file
    plt.close()  # Close the plot to free up memory

if __name__ == "__main__":
    ticker = 'RELIANCE.NS'
    stock_data = pd.read_csv(f'{ticker}_data_with_indicators.csv', parse_dates=True, index_col=0)
    signals = pd.read_csv(f'{ticker}_buy_sell_signals.csv', parse_dates=True, index_col=0)
    
    plot_signals(stock_data, signals, ticker)
