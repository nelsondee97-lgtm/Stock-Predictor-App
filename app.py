import streamlit as st
import joblib
import numpy as np
import yfinance as yf
st.set_page_config(page_title="Stock Predictor", layout="wide")

st.title("📈 Stock Movement Predictor")
st.markdown("### Smart predictions powered by Machine Learning")
st.markdown("---")
# Load model
model = joblib.load("model.pkl")

# User input
ticker = st.text_input("Enter Stock Ticker (e.g. AAPL, TSLA)", "AAPL")

# Fetch data (single source of truth)
data = yf.download(ticker, period="5d")

# Chart
if not data.empty:
   latest = data.iloc[-1]

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Price Chart")
    st.line_chart(data["Close"])

with col2:
    st.subheader("📋 Latest Data")
    st.dataframe(data.tail(1))
    
   # Extract values correctly from yfinance MultiIndex
open_val = float(latest[("Open", ticker)])
high_val = float(latest[("High", ticker)])
low_val = float(latest[("Low", ticker)])
volume_val = float(latest[("Volume", ticker)])

# Check missing values
if any(np.isnan([open_val, high_val, low_val, volume_val])):
    st.error("⚠️ Missing data — cannot make prediction")
else:
    input_data = np.array([[open_val, high_val, low_val, volume_val]])

   if st.button("Predict"):
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    if prediction[0] == 1:
        st.success(f"📈 UP (Confidence: {probability:.2f})")
    else:
        st.error(f"📉 DOWN (Confidence: {1 - probability:.2f})")
