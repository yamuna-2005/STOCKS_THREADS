import yfinance as yf
import pandas as pd
from time import sleep
from requests.exceptions import ConnectionError

def fetch_historical_data(ticker, start_date, end_date, retries=3):
    for attempt in range(retries):
        try:
            stock_data = yf.download(ticker, start=start_date, end=end_date)
            stock_data.to_csv(f'{ticker}_historical_data.csv')
            return stock_data
        except ConnectionError as e:
            if attempt < retries - 1:
                sleep(5)
                continue
            else:
                print(f"Failed to fetch historical data: {e}")
                return pd.DataFrame()

def fetch_realtime_data(ticker, retries=3):
    for attempt in range(retries):
        try:
            stock_data = yf.download(ticker, period="1d", interval="1m")
            return stock_data
        except ConnectionError as e:
            if attempt < retries - 1:
                sleep(5)
                continue
            else:
                print(f"Failed to fetch real-time data: {e}")
                return pd.DataFrame()

if __name__ == "__main__":
    ticker = 'RELIANCE.NS'
    start_date = '2020-01-01'
    end_date = '2023-12-31'
    
    historical_data = fetch_historical_data(ticker, start_date, end_date)
    realtime_data = fetch_realtime_data(ticker)
    
    if not historical_data.empty:
        print(historical_data.tail())
    else:
        print("Historical data fetch failed.")

    if not realtime_data.empty:
        print(realtime_data.tail())
    else:
        print("Real-time data fetch failed.")
