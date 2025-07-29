import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Stock Market Predictor", layout="wide")

model = load_model('Stocks2_new.h5')

st.markdown("<h1 style='text-align: center; color: #FFA500;'>📈 Stock Market Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #555;'>Analyze trends, predict future prices, and maximize returns!</h4>", unsafe_allow_html=True)
st.markdown("---")

stock = st.text_input('🔎 Enter Stock Symbol:', 'GOOG')

start = '2012-01-01'
end = '2025-03-22'

with st.spinner("Fetching stock data..."):
    data = yf.download(stock, start, end)

if data.shape[0] < 500:
    st.error("Not enough historical data available for this stock. Try another symbol.")
    st.stop()

st.subheader(f"📊 Stock Data for {stock}")
st.dataframe(data.tail(10), use_container_width=True)

data = data[['Close', 'High', 'Low']]
train_size = max(int(len(data) * 0.80), len(data) - 365)
data_train = data.iloc[:train_size]
data_test = data.iloc[train_size:]

past_100_days = data_train.tail(100)
data_test = pd.concat([past_100_days, data_test], ignore_index=True)

scaler = MinMaxScaler(feature_range=(0, 1))
if data_test.shape[0] > 0:
    data_test_scaled = scaler.fit_transform(data_test)
else:
    st.error("Not enough test data to scale. Try a different date range or stock symbol.")
    st.stop()

ma_50 = data['Close'].rolling(50).mean()
ma_100 = data['Close'].rolling(100).mean()
ma_200 = data['Close'].rolling(200).mean()

col1, col2 = st.columns(2)

with col1:
    st.subheader('📈 Price vs MA50')
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(ma_50, 'r', label="MA50")
    ax.plot(data['Close'], 'g', label="Closing Price")
    ax.legend()
    st.pyplot(fig)

with col2:
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

if X_test.shape[0] == 0:
    st.error("Error: X_test is empty. Please check stock symbol and date range.")
    st.stop()

predictions = model.predict(X_test)
scale_factor = 1 / scaler.scale_
predictions = predictions * scale_factor
y_test = y_test * scale_factor

st.subheader('🔮 Actual vs Predicted Prices')

fig, ax = plt.subplots(3, 1, figsize=(10, 15))
ax[0].plot(y_test[:, 1], 'g', label='Original High Price')
ax[0].plot(predictions[:, 1], 'r', label='Predicted High Price')
ax[0].set_title("High Price Prediction")
ax[0].legend()

ax[1].plot(y_test[:, 2], 'g', label='Original Low Price')
ax[1].plot(predictions[:, 2], 'r', label='Predicted Low Price')
ax[1].set_title("Low Price Prediction")
ax[1].legend()

ax[2].plot(y_test[:, 0], 'g', label='Original Close Price')
ax[2].plot(predictions[:, 0], 'r', label='Predicted Close Price')
ax[2].set_title("Close Price Prediction")
ax[2].legend()

st.pyplot(fig)

future_days = 30
future_predictions = []
future_dates = pd.date_range(start=end, periods=future_days + 1)[1:]

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
future_df = pd.DataFrame(future_predictions, columns=['Close', 'High', 'Low'], index=future_dates)

st.subheader(f'📅 Next {future_days} Days Stock Prices')
st.dataframe(future_df.style.set_properties(**{'background-color': '#fff5e6', 'color': '#000'}))

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(future_dates, future_df['High'], 'b', label='Predicted High Price')
ax.plot(future_dates, future_df['Close'], 'g', label='Predicted Close Price')
ax.plot(future_dates, future_df['Low'], 'r', label='Predicted Low Price')
ax.set_title(f"📈 Predicted Stock Prices for {stock}")
ax.legend()
st.pyplot(fig)

start_price = future_df.iloc[0]['Close']
end_price = future_df.iloc[-1]['Close']
end_returns = ((end_price - start_price) / start_price) * 100

investment = st.number_input('🔎 Enter Amount to invest:', min_value=0.0, value=1000.0, step=100.0)


return_money= investment+((end_price - start_price)*31)

st.subheader(f'📊 Expected Return Over {future_days} Days')
st.markdown(f"<h3 style='color: green;'>📈 **Predicted Return: {end_returns:.2f}%**</h3>", unsafe_allow_html=True)
st.markdown(f"<h3 style='color: green;'>📈 **Predicted Return: {return_money:.2f}**</h3>", unsafe_allow_html=True)
if end_returns > 0:
    st.success("📢 Positive Growth Expected!")
else:
    st.warning("⚠️ Possible Loss Expected!")

st.markdown("---")

with st.expander("📚 Unit I – Probability Analysis"):
    daily_returns = data['Close'].pct_change().dropna()
    mean_return = daily_returns.mean()
    std_return = daily_returns.std()
    prob_increase = 1 - norm.cdf(0, loc=mean_return, scale=std_return)
    prob_decrease = norm.cdf(0, loc=mean_return, scale=std_return)
    st.markdown(f"📌 **Probability of Price Increase Tomorrow:** {float(prob_increase):.2%}")
    st.markdown(f"📌 **Probability of Price Decrease Tomorrow:** {float(prob_decrease):.2%}")


with st.expander("📚 Unit II – Sampling Theory"):
    sample = daily_returns.sample(50)
    sample_mean = float(sample.mean()) 
    sample_std = float(sample.std())    
    conf_int = stats.norm.interval(0.95, loc=sample_mean, scale=sample_std / np.sqrt(len(sample)))
    st.markdown(f"📌 **Sample Mean Return (50 days):** {sample_mean:.4f}")
    st.markdown(f"📌 **95% Confidence Interval:** ({conf_int[0]:.4f}, {conf_int[1]:.4f})")




with st.expander("📚 Unit III – Hypothesis Testing"):
    t_stat, p_val = stats.ttest_1samp(daily_returns, 0)
    t_stat = float(t_stat)
    p_val = float(p_val)
    
    st.markdown(f"📌 **t-Statistic:** {t_stat:.4f}")
    st.markdown(f"📌 **p-Value:** {p_val:.4f}")
    
    if p_val < 0.05:
        st.success("Reject Null Hypothesis: Mean return is significantly different from 0.")
    else:
        st.warning("Fail to Reject Null Hypothesis: Mean return is not significantly different from 0.")


with st.expander("📚 Unit IV – Correlation & Regression"):
    corr = data[['Close', 'High']].corr().iloc[0, 1]
    st.markdown(f"📌 **Correlation (Close vs High):** {corr:.4f}")

    X_lr = data['Close'].values.reshape(-1, 1)
    y_lr = data['High'].values

    model_lr = LinearRegression().fit(X_lr, y_lr)
    slope = float(model_lr.coef_[0])        
    intercept = float(model_lr.intercept_)   

    st.markdown(f"📌 **Regression Equation:** High = {slope:.2f} * Close + {intercept:.2f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(data['Close'], data['High'], alpha=0.5, label='Data')
    ax.plot(data['Close'], model_lr.predict(X_lr), color='red', label='Regression Line')
    ax.set_xlabel("Close Price")
    ax.set_ylabel("High Price")
    ax.set_title("Linear Regression: Close vs High")
    ax.legend()
    st.pyplot(fig)

st.markdown("<h5 style='text-align: center; color: #888;'>🔍 Data Source: Yahoo Finance | 📊 Model: LSTM | 📅 Predictions for 30 Days</h5>", unsafe_allow_html=True)

