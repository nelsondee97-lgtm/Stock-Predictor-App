import streamlit as st
import joblib
import numpy as np
import yfinance as yf
import pandas as pd

# Page config
st.set_page_config(page_title="StockSage Pro", layout="wide")

st.title("📈 Stock-Movement-Predictor")
st.markdown("AI-powered stock movement prediction with real-time indicators")
st.markdown("---")

# Load model
model = joblib.load("stock.pkl")

# User input
ticker = st.text_input("Enter Stock Ticker", "AAPL")

# Fetch data
data = yf.download(ticker, period="3mo", auto_adjust=True)

if data.empty:
    st.error("Invalid ticker or no data found")
    st.stop()

# =========================
# 📊 FEATURE ENGINEERING
# =========================
data["Return"] = data["Close"].pct_change()
data["MA_5"] = data["Close"].rolling(5).mean()
data["MA_10"] = data["Close"].rolling(10).mean()
data["Volatility"] = data["Return"].rolling(5).std()

data = data.dropna()

# =========================
# 📈 VISUALIZATION
# =========================
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Price & Moving Averages")
    st.line_chart(data[["Close", "MA_5", "MA_10"]])

with col2:
    st.subheader("📉 Volatility")
    st.line_chart(data["Volatility"])

# =========================
# 🎯 LATEST DATA FOR PREDICTION
# =========================
latest = data.iloc[-1]

input_data = np.array([[
    latest["Return"],
    latest["MA_5"],
    latest["MA_10"],
    latest["Volatility"]
]])

# =========================
# 🔮 PREDICTION
# =========================
if st.button("Predict Next Day Movement"):

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("📊 Prediction Result")

    if prediction[0] == 1:
        st.success(f"📈 UP (Confidence: {probability:.2f})")
    else:
        st.error(f"📉 DOWN (Confidence: {1 - probability:.2f})")

    # =========================
    # 🧠 INSIGHTS
    # =========================
    st.markdown("### 🧠 Market Insight")

    if latest["MA_5"] > latest["MA_10"]:
        st.write("📊 Short-term trend is bullish")
    else:
        st.write("📉 Short-term trend is bearish")

    if latest["Volatility"] > data["Volatility"].mean():
        st.write("⚠️ Market is currently volatile")
    else:
        st.write("✅ Market volatility is stable")
