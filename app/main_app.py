import streamlit as st
from datetime import date
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.stock_predictor import (
    load_stock_model, fetch_stock_data, preprocess_stock_data,
    prepare_test_data, scale_data, predict_stock_prices
)
from src.gold_predictor import (
    load_gold_model, fetch_gold_data, preprocess_data as preprocess_gold_data,
    prepare_test_data as prepare_gold_test_data, scale_data as scale_gold_data,
    predict_gold_prices, adjust_gold_fluctuation
)
from src.bond_calculator import calculate_bond_return


st.set_page_config(page_title="Investment Portfolio Predictor", layout="wide")
st.markdown(
    "<h1 style='text-align: center; color: #FFA500;'>💰 Investment Portfolio Predictor</h1>",
    unsafe_allow_html=True
)


col1, col2, col3 = st.columns(3)

with col1:
    stock_symbol = st.text_input('📈 Stock Symbol:', 'GOOG')
    stock_investment = st.number_input(
        '💵 Stock Investment Amount ($):', min_value=0.0, value=1000.0, step=100.0
    )

with col2:
    gold_symbol = st.text_input('🏆 Gold Symbol:', 'GC=F')
    gold_investment = st.number_input(
        '💰 Gold Investment Amount ($):', min_value=0.0, value=1000.0, step=100.0
    )

with col3:
    bond_principal = st.number_input(
        '📜 Bond Investment Amount ($):', min_value=0.0, value=1000.0, step=100.0
    )
    bond_rate = st.number_input(
        '📊 Bond Interest Rate (%):', min_value=0.0, value=5.0, step=0.1
    )
    bond_months = st.number_input(
        '⏳ Bond Duration (Months):', min_value=1, value=6, step=1
    )
    compounding = st.radio('🔄 Compounding?', ['No', 'Yes'])


if st.button('📊 Predict and Calculate'):
    stock_model = load_stock_model()
    gold_model = load_gold_model()

    start, end = '2012-01-01', date.today()

    stock_data = fetch_stock_data(stock_symbol, start, end)
    gold_data = fetch_gold_data(gold_symbol, start, end)

    if stock_data.empty:
        st.error(f"❌ No stock data found for symbol: {stock_symbol}")
        st.stop()
    if gold_data.empty:
        st.error(f"❌ No gold data found for symbol: {gold_symbol}")
        st.stop()

    stock_data = preprocess_stock_data(stock_data)
    gold_data = preprocess_gold_data(gold_data)

    stock_train_size = max(int(len(stock_data) * 0.80), len(stock_data) - 365)
    gold_train_size = max(int(len(gold_data) * 0.80), len(gold_data) - 365)

    stock_test = prepare_test_data(stock_data, stock_train_size)
    gold_test = prepare_gold_test_data(gold_data, gold_train_size)

    if stock_test.empty or len(stock_test) < 101:
        st.error("❌ Not enough stock data to generate predictions (need at least 101 rows).")
        st.stop()
    if gold_test.empty or len(gold_test) < 101:
        st.error("❌ Not enough gold data to generate predictions (need at least 101 rows).")
        st.stop()

    stock_scaler, stock_test_scaled = scale_data(stock_test)
    gold_scaler, gold_test_scaled = scale_gold_data(gold_test)

    future_days = 30
    stock_future_df = predict_stock_prices(stock_model, stock_test_scaled, stock_scaler, future_days)
    gold_predictions = predict_gold_prices(gold_model, gold_test_scaled, gold_scaler, future_days)
    gold_predictions = adjust_gold_fluctuation(gold_predictions)

    stock_units = stock_investment / stock_future_df.iloc[0]['Close']
    stock_final_value = stock_units * stock_future_df.iloc[-1]['Close']

    gold_units = gold_investment / gold_predictions[0][0]
    gold_final_value = gold_units * gold_predictions[-1][0]

    bond_final_value = calculate_bond_return(bond_principal, bond_rate, bond_months, compounding == 'Yes')

    total_initial = stock_investment + gold_investment + bond_principal
    total_final = stock_final_value + gold_final_value + bond_final_value
    total_return = ((total_final - total_initial) / total_initial) * 100

    st.subheader('📊 Investment Summary')
    st.write(f'*Stock Final Value:* ${stock_final_value:,.2f}')
    st.write(f'*Gold Final Value:* ${gold_final_value:,.2f}')
    st.write(f'*Bond Final Value:* ${bond_final_value:,.2f}')
    st.write(f'*Total Final Value:* ${total_final:,.2f}')
    st.write(f'*Total Portfolio Return:* {total_return:.2f}%')

    if total_return > 0:
        st.success("📈 Positive Growth Expected!")
    else:
        st.warning("⚠ Possible Loss Expected!")
