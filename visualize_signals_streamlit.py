import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

def plot_signals(data, signals, ticker):
    fig, axs = plt.subplots(4, 1, figsize=(14, 10))

    # Plot closing price with buy/sell signals and Bollinger Bands
    axs[0].plot(data['Close'], label='Close Price', color='blue')
    axs[0].plot(data['Bollinger_High'], label='Bollinger High', linestyle='--', color='red')
    axs[0].plot(data['Bollinger_Low'], label='Bollinger Low', linestyle='--', color='green')
    axs[0].scatter(signals.loc[signals['Buy'] == 1].index, data['Close'][signals['Buy'] == 1], label='Buy Signal', marker='^', color='green', alpha=1)
    axs[0].scatter(signals.loc[signals['Sell'] == 1].index, data['Close'][signals['Sell'] == 1], label='Sell Signal', marker='v', color='red', alpha=1)
    axs[0].set_title(f'{ticker} - Stock Price with Bollinger Bands and Buy/Sell Signals')
    axs[0].legend()

    # Plot RSI
    axs[1].plot(signals['RSI'], label='RSI', color='orange')
    axs[1].axhline(70, color='red', linestyle='--')
    axs[1].axhline(30, color='green', linestyle='--')
    axs[1].set_title(f'{ticker} - RSI')
    axs[1].legend()

    # Plot MACD and Signal Line
    axs[2].plot(data['MACD'], label='MACD', color='purple')
    axs[2].plot(data['MACD_Signal'], label='MACD Signal Line', color='red')
    axs[2].set_title(f'{ticker} - MACD')
    axs[2].legend()

    # Plot Stochastic Oscillator
    axs[3].plot(data['Stochastic_K'], label='Stochastic %K', color='magenta')
    axs[3].axhline(80, color='red', linestyle='--')
    axs[3].axhline(20, color='green', linestyle='--')
    axs[3].set_title(f'{ticker} - Stochastic Oscillator')
    axs[3].legend()

    plt.tight_layout()
    st.pyplot(fig)  # Display the plot in Streamlit

if __name__ == "__main__":
    st.title('Stock Analysis and Buy/Sell Signal Visualization')

    ticker = 'RELIANCE.NS'
    stock_data = pd.read_csv(f'{ticker}_data_with_indicators.csv', parse_dates=True, index_col=0)
    signals = pd.read_csv(f'{ticker}_buy_sell_signals.csv', parse_dates=True, index_col=0)
    
    plot_signals(stock_data, signals, ticker)
