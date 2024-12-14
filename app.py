import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import gc

# Function to fetch stock data (cached)
@st.cache_data
def download_stock_data(ticker, start_date, end_date, interval):
    return yf.download(ticker, start=start_date, end=end_date, interval=interval)

# Function to fetch news headlines
def fetch_news_headlines(ticker):
    try:
        api_key = "4dd57367c6984cca936e8e5c60ca5d70"
        query = ticker.split(".")[0]  # Simplify query (e.g., "RELIANCE" from "RELIANCE.NS")
        api_url = f"https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&apiKey={api_key}"

        response = requests.get(api_url)
        if response.status_code == 200:
            articles = response.json().get("articles", [])
            if not articles:
                st.warning("No news headlines found for the given query.")
                return pd.DataFrame()

            # Extract relevant fields
            headlines = [{"Date": article["publishedAt"], "Headline": article["title"]} for article in articles]
            return pd.DataFrame(headlines)
        else:
            st.error(f"API Error: {response.status_code} - {response.reason}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"An error occurred while fetching news: {e}")
        return pd.DataFrame()

# Function to calculate sentiment using VADER
def analyze_sentiment_vader(headlines_df):
    if headlines_df.empty:
        return None
    sid = SentimentIntensityAnalyzer()
    headlines_df["Sentiment"] = headlines_df["Headline"].apply(lambda x: sid.polarity_scores(x)['compound'])
    return headlines_df

# Function to calculate RSI
def calculate_rsi(data, window=14):
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to calculate Bollinger Bands
def calculate_bollinger_bands(data, window=20):
    rolling_mean = data['Close'].rolling(window=window, min_periods=1).mean()
    rolling_std = data['Close'].rolling(window=window, min_periods=1).std()

    data['Bollinger_High'] = rolling_mean + (rolling_std * 2)
    data['Bollinger_Low'] = rolling_mean - (rolling_std * 2)

# Function to calculate MACD
def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data['Close'].ewm(span=short_window, min_periods=1, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_window, min_periods=1, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, min_periods=1, adjust=False).mean()
    return macd, signal

# Function to prepare LSTM data
def prepare_lstm_data(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps, 0])
        y.append(data[i + n_steps, 0])
    return np.array(X), np.array(y)

# Function to plot data
def plot_signals(data, signals, ticker):
    fig, axs = plt.subplots(5, figsize=(12, 20))

    # Plot Close Price
    axs[0].plot(data['Close'], label='Close Price', color='blue')
    axs[0].set_title(f'{ticker} Close Price')
    axs[0].legend()

    # Plot Bollinger Bands
    axs[1].plot(data['Close'], label='Close Price', color='blue')
    axs[1].plot(data['Bollinger_High'], label='Bollinger High', linestyle='--', color='red')
    axs[1].plot(data['Bollinger_Low'], label='Bollinger Low', linestyle='--', color='green')
    axs[1].set_title(f'{ticker} Bollinger Bands')
    axs[1].legend()

    # Plot RSI
    axs[2].plot(signals['RSI'], label='RSI', color='orange')
    axs[2].axhline(70, linestyle='--', color='red', label='Overbought')
    axs[2].axhline(30, linestyle='--', color='green', label='Oversold')
    axs[2].set_title(f'{ticker} RSI')
    axs[2].legend()

    # Plot MACD
    axs[3].plot(signals['MACD'], label='MACD', color='blue')
    axs[3].plot(signals['Signal'], label='Signal', linestyle='--', color='red')
    axs[3].set_title(f'{ticker} MACD')
    axs[3].legend()

    # Plot Predictions
    if 'Predictions' in data.columns:
        axs[4].plot(data['Close'], label='Actual Price', color='blue')
        axs[4].plot(data['Predictions'], label='Predicted Price', linestyle='--', color='purple')
        axs[4].set_title(f'{ticker} Actual vs Predicted Prices')
        axs[4].legend()

    st.pyplot(fig)

# Streamlit app
def main():
    st.set_page_config(page_title="AI-Powered Stock Analysis Tool", layout="wide")
    st.title("AI-Powered Stock Analysis and Prediction Tool for NSE and BSE")
    st.image(r"C:\\STOCKS_THREADS\\stock.png", use_container_width=False)

    # Inputs for stock ticker and date range
    ticker = st.text_input("Enter Stock Ticker (e.g., RELIANCE.NS or SBIN.BO)", "RELIANCE.NS")
    start_date = st.date_input("Start Date", pd.to_datetime("2020-01-01"))
    end_date = st.date_input("End Date", pd.to_datetime("2024-12-13"))

    # Sliders for customization
    rsi_window = st.slider("RSI Window", 10, 30, 14)
    bollinger_window = st.slider("Bollinger Bands Window", 10, 50, 20)
    macd_short_window = st.slider("MACD Short Window", 5, 15, 12)
    macd_long_window = st.slider("MACD Long Window", 15, 30, 26)
    macd_signal_window = st.slider("MACD Signal Window", 5, 15, 9)
    n_steps = st.slider("LSTM Steps", 10, 100, 50)

    st.markdown("<hr>", unsafe_allow_html=True)

    if st.button("Analyze Stock"):
        try:
            # Fetch stock data
            data = download_stock_data(ticker, start_date, end_date, interval="1d")
            if data.empty:
                st.error("No stock data found. Please check the ticker and date range.")
                return

            st.write("### Fetched Stock Data")
            st.write(data.tail())

            # Calculate Bollinger Bands, RSI, and MACD
            calculate_bollinger_bands(data, window=bollinger_window)
            signals = pd.DataFrame()
            signals['RSI'] = calculate_rsi(data, window=rsi_window)
            signals['MACD'], signals['Signal'] = calculate_macd(data, short_window=macd_short_window, 
                                                                long_window=macd_long_window, 
                                                                signal_window=macd_signal_window)

            st.write("### Processed Signals")
            st.write(signals.tail())

            # Fetch news and analyze sentiment
            headlines = fetch_news_headlines(ticker)
            if not headlines.empty:
                headlines = analyze_sentiment_vader(headlines)
                st.write("### News Headlines with Sentiment")
                st.write(headlines)

            # LSTM Model for prediction
            close_data = data[['Close']].values.astype('float32')
            scaler = MinMaxScaler(feature_range=(0, 1))
            close_data_scaled = scaler.fit_transform(close_data)

            X, y = prepare_lstm_data(close_data_scaled, n_steps)
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))

            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(n_steps, 1)),
                LSTM(50),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(X, y, epochs=10, batch_size=32, verbose=0)

            # Predict the next day
            last_50_days = close_data_scaled[-n_steps:]
            last_50_days = np.reshape(last_50_days, (1, n_steps, 1))
            predicted_price = scaler.inverse_transform(model.predict(last_50_days))
            st.write(f"### Predicted Next Day Close Price: {predicted_price[0][0]:.2f}")

            # Plot everything
            predictions = scaler.inverse_transform(model.predict(X))
            data['Predictions'] = np.nan
            data.iloc[n_steps:len(predictions) + n_steps, data.columns.get_loc('Predictions')] = predictions.flatten()
            plot_signals(data, signals, ticker)

            # Cleanup
            del data, close_data_scaled, predictions
            gc.collect()

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
