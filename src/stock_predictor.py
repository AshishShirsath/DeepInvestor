import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import date

def load_stock_model(model_path='models/Stocks2_new.h5'):
    return load_model(model_path)

def fetch_stock_data(stock, start, end):
    return yf.download(stock, start, end)

def preprocess_stock_data(data):
    return data[['Close', 'High', 'Low']]

def scale_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler, scaler.fit_transform(data)

def prepare_test_data(data, train_size):
    data_train = data.iloc[:train_size]
    data_test = data.iloc[train_size:]
    past_100_days = data_train.tail(100)
    return pd.concat([past_100_days, data_test], ignore_index=True)

def predict_stock_prices(model, data_test_scaled, scaler, future_days=30):
    future_predictions = []
    future_dates = pd.date_range(start=date.today(), periods=future_days + 1)[1:]
    X_input = data_test_scaled[-100:].reshape(1, 100, 3)
    for _ in range(future_days):
        pred = model.predict(X_input)[0]
        pred *= np.random.uniform(0.98, 1.02, size=pred.shape)
        close_pred = pred[0]
        high_pred = max(pred[1], close_pred * np.random.uniform(1.01, 1.05))
        low_pred = min(pred[2], close_pred * np.random.uniform(0.95, 0.99))
        future_predictions.append([close_pred, high_pred, low_pred])
        new_real_data = np.vstack((data_test_scaled[-99:], pred))
        X_input = new_real_data.reshape(1, 100, 3)
    future_predictions = np.array(future_predictions)
    future_predictions = scaler.inverse_transform(future_predictions)
    return pd.DataFrame(future_predictions, columns=['Close', 'High', 'Low'], index=future_dates)