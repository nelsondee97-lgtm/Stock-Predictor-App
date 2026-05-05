import streamlit as st
import joblib
import numpy as np
import yfinance as yf
import pandas as pd

# =========================
# ⚙️ CONFIG
# =========================
st.set_page_config(page_title="StockSage Pro", layout="wide")

st.title("📈 StockSage Pro")
st.markdown("### 📊 Real-Time Market Intelligence Dashboard")
st.markdown("Predict stock direction using trend indicators and volatility analysis")
st.markdown("---")

# =========================
# 📦 LOAD MODEL
# =========================
model = joblib.load("stock.pkl")

# =========================
# 📊 USER INPUT
# =========================
popular_tickers = ["AAPL", "TSLA", "MSFT", "GOOG", "AMZN"]
ticker = st.selectbox("Select Stock", popular_tickers)

# =========================
# 📡 FETCH DATA
# =========================
data = yf.download(ticker, period="3mo", auto_adjust=True)

if data.empty:
    st.error("Invalid ticker or no data found")
    st.stop()

# Fix MultiIndex
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# =========================
# 🧠 FEATURE ENGINEERING
# =========================
data["Return"] = data["Close"].pct_change()
data["MA_5"] = data["Close"].rolling(5).mean()
data["MA_10"] = data["Close"].rolling(10).mean()
data["Volatility"] = data["Return"].rolling(5).std()

data = data.dropna()

if data.empty:
    st.error("Not enough data")
    st.stop()

# =========================
# 🎯 LATEST DATA
# =========================
latest = data.iloc[-1]

# =========================
# 📊 METRICS DASHBOARD
# =========================
colA, colB, colC, colD = st.columns(4)

colA.metric("Close", round(latest["Close"], 2))
colB.metric("MA_5", round(latest["MA_5"], 2))
colC.metric("MA_10", round(latest["MA_10"], 2))
colD.metric("Volatility", round(latest["Volatility"], 4))

# =========================
# 📈 VISUALS
# =========================
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Price Trends")
    st.line_chart(data[["Close", "MA_5", "MA_10"]])

with col2:
    st.subheader("📉 Volatility")
    st.line_chart(data["Volatility"])

# =========================
# 🔮 PREDICTION BUTTON
# =========================
if st.button("🚀 Predict Market Direction"):

    input_data = np.array([[
        latest["Return"],
        latest["MA_5"],
        latest["MA_10"],
        latest["Volatility"]
    ]])

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("📊 Prediction Result")

    # =========================
    # 💰 BUY / SELL / HOLD
    # =========================
    if prediction[0] == 1 and probability > 0.6:
        decision = "BUY 📈"
        st.success(f"{decision} (Confidence: {probability:.2f})")

    elif prediction[0] == 1:
        decision = "HOLD 🤏"
        st.info(f"{decision} (Weak Uptrend | {probability:.2f})")

    elif prediction[0] == 0 and probability > 0.6:
        decision = "SELL 📉"
        st.error(f"{decision} (Confidence: {1 - probability:.2f})")

    else:
        decision = "HOLD 🤏"
        st.info(f"{decision} (Uncertain Market | {1 - probability:.2f})")

    # =========================
    # 📊 CONFIDENCE BAR
    # =========================
    confidence = probability if prediction[0] == 1 else (1 - probability)

    st.markdown("### 📊 Confidence Level")
    st.progress(int(confidence * 100))
    st.write(f"Confidence Score: {confidence:.2f}")

    # =========================
    # 🧠 INSIGHTS
    # =========================
    st.markdown("### 🧠 Market Insight")

    if latest["MA_5"] > latest["MA_10"]:
        st.write("📊 Short-term trend is bullish")
    else:
        st.write("📉 Short-term trend is bearish")

    if latest["Volatility"] > data["Volatility"].mean():
        st.write("⚠️ Market is volatile")
    else:
        st.write("✅ Market is stable")

    # =========================
    # 📄 DOWNLOAD REPORT
    # =========================
    report = f"""
Stock Report

Ticker: {ticker}
Decision: {decision}
Confidence: {confidence:.2f}

Close: {latest["Close"]}
MA5: {latest["MA_5"]}
MA10: {latest["MA_10"]}
Volatility: {latest["Volatility"]}
"""

    st.download_button("📥 Download Report", report, file_name="stock_report.txt")

# =========================
# 📊 MULTI-STOCK COMPARISON
# =========================
st.markdown("---")
st.subheader("📊 Compare Stocks")

compare = st.multiselect("Select stocks to compare", popular_tickers, default=["AAPL", "TSLA"])

if compare:
    df_compare = yf.download(compare, period="3mo", auto_adjust=True)["Close"]
    st.line_chart(df_compare)
