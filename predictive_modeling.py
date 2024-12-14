import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def train_lstm_model(data):
    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data[['Close']])

    # Prepare data for LSTM
    X, y = [], []
    for i in range(60, len(data_scaled)):
        X.append(data_scaled[i-60:i, 0])
        y.append(data_scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=50, batch_size=32)

    return model, scaler

def predict_stock_price(model, scaler, data):
    # Predict future stock prices
    last_60_days = data[-60:].values
    last_60_days_scaled = scaler.transform(last_60_days)

    X_test = []
    X_test.append(last_60_days_scaled)
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    predicted_price = model.predict(X_test)
    predicted_price = scaler.inverse_transform(predicted_price)

    return predicted_price

if __name__ == "__main__":
    ticker = 'RELIANCE.NS'
    data = pd.read_csv(f'{ticker}_data_with_indicators.csv', parse_dates=True, index_col=0)

    # Train LSTM model
    model, scaler = train_lstm_model(data)

    # Predict future price
    predicted_price = predict_stock_price(model, scaler, data[['Close']])
    print(f'Predicted next day close price: {predicted_price[0][0]}')
