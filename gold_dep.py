import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import date

model = load_model('gold_model2.h5')

st.header('📈 GOLD Market Predictor')

stock = st.text_input('GOLD Symbol:', 'GC=F')

start = '2012-01-01'
end = date.today()
st.markdown("<h1 style='text-align: center; color: #FFA500;'>📈 GOLD Market Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #555;'>Analyze trends, predict future prices, and maximize returns!</h4>", unsafe_allow_html=True)
st.markdown("---")
data = yf.download(stock, start, end)

st.subheader('📊 GOLD Data')
st.write(data)

if data.shape[0] < 500:
    st.error("Not enough historical data available for this GOLD. Try another symbol.")
    st.stop()

data = data[['Close', 'High', 'Low']]

train_size = max(int(len(data) * 0.80), len(data) - 365)
data_train = data.iloc[:train_size]
data_test = data.iloc[train_size:]

past_100_days = data_train.tail(100)
data_test = pd.concat([past_100_days, data_test], ignore_index=True)

scaler = MinMaxScaler(feature_range=(0, 1))

data_test_scaled = scaler.fit_transform(data_test)

ma_50 = data['Close'].rolling(50).mean()
ma_100 = data['Close'].rolling(100).mean()
ma_200 = data['Close'].rolling(200).mean()

st.subheader('📈 Price vs MA50')
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(ma_50, 'r', label="MA50")
ax.plot(data['Close'], 'g', label="Closing Price")
ax.legend()
st.pyplot(fig)

st.subheader('📈 Price vs MA50 vs MA100')
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(ma_50, 'r', label="MA50")
ax.plot(ma_100, 'b', label="MA100")
ax.plot(data['Close'], 'g', label="Closing Price")
ax.legend()
st.pyplot(fig)

st.subheader('📈 Price vs MA100 vs MA200')
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(ma_100, 'r', label="MA100")
ax.plot(ma_200, 'b', label="MA200")
ax.plot(data['Close'], 'g', label="Closing Price")
ax.legend()
st.pyplot(fig)

X_test, y_test = [], []
for i in range(100, data_test_scaled.shape[0]):
    X_test.append(data_test_scaled[i-100:i])
    y_test.append(data_test_scaled[i])

X_test, y_test = np.array(X_test), np.array(y_test)

predictions = model.predict(X_test)

scale_factor = 1 / scaler.scale_
predictions = predictions * scale_factor
y_test = y_test * scale_factor

st.subheader('🔮 Original vs Predicted Prices')

fig, ax = plt.subplots(3, 1, figsize=(10, 15))

ax[0].plot(y_test[:, 1], 'g', label='Original High Price')
ax[0].plot(predictions[:, 1], 'r', label='Predicted High Price')
ax[0].set_title("High Price Prediction")
ax[0].set_xlabel("Time")
ax[0].set_ylabel("Price")
ax[0].legend()

ax[1].plot(y_test[:, 2], 'g', label='Original Low Price')
ax[1].plot(predictions[:, 2], 'r', label='Predicted Low Price')
ax[1].set_title("Low Price Prediction")
ax[1].set_xlabel("Time")
ax[1].set_ylabel("Price")
ax[1].legend()

ax[2].plot(y_test[:, 0], 'g', label='Original Close Price')
ax[2].plot(predictions[:, 0], 'r', label='Predicted Close Price')
ax[2].set_title("Close Price Prediction")
ax[2].set_xlabel("Time")
ax[2].set_ylabel("Price")
ax[2].legend()

plt.tight_layout()
st.pyplot(fig)

future_days = 30
future_predictions = []
future_dates = pd.date_range(start=end, periods=future_days + 1)[1:]

X_input = data_test_scaled[-100:].reshape(1, 100, 3)

for _ in range(future_days):
    pred = model.predict(X_input)[0]
    
    close_pred = pred[0] * np.random.uniform(0.98,1.02)  
    high_pred = max(close_pred, close_pred + 1)  
    low_pred = min(close_pred, close_pred - 1)  
    
    future_predictions.append([close_pred, high_pred, low_pred])
    
    new_real_data = np.vstack((data_test_scaled[-99:], pred))
    X_input = new_real_data.reshape(1, 100, 3)



future_predictions = np.array(future_predictions)
future_predictions = scaler.inverse_transform(future_predictions)

future_df = pd.DataFrame(future_predictions, columns=['Close', 'High', 'Low'], index=future_dates)

st.subheader(f'📅 Next {future_days} Days GOLD Prices')
st.write(future_df)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(future_dates, future_df['High'], 'b', label='Predicted High Price')
ax.plot(future_dates, future_df['Close'], 'g', label='Predicted Close Price')
ax.plot(future_dates, future_df['Low'], 'r', label='Predicted Low Price')
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.set_title(f"Predicted GOLD Prices for {stock}")
ax.legend()
st.pyplot(fig)

start_price = future_df.iloc[0]['Close']
end_price = future_df.iloc[-1]['Close']
end_returns = ((end_price - start_price) / start_price) * 100

investment = st.number_input('🔎 Enter Amount to invest:', min_value=0.0, value=1000.0, step=100.0)

gold_units = investment / start_price
final_value = gold_units * end_price

st.subheader(f'📊 Expected Return Over {future_days} Days')
st.markdown(f"<h3 style='color: green;'>📈 **Predicted Return: {end_returns:.2f}%**</h3>", unsafe_allow_html=True)
st.markdown(f"<h3 style='color: green;'>💰 **Predicted Investment Value: ${final_value:.2f}**</h3>", unsafe_allow_html=True)

if end_returns > 0:
    st.success("📢 Positive Growth Expected!")
else:
    st.warning("⚠️ Possible Loss Expected!")