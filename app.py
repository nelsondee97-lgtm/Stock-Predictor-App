import streamlit as st
import joblib
import numpy as np
import yfinance as yf

st.title("📈 Stock Movement Predictor")

# Load model
model = joblib.load("model.pkl")

# User input
ticker = st.text_input("Enter Stock Ticker (e.g. AAPL, TSLA)", "AAPL")

# Fetch data (single source of truth)
data = yf.download(ticker, period="5d")

# Chart
if not data.empty:
    st.subheader("📊 Stock Price Chart")
    st.line_chart(data["Close"])

    latest = data.iloc[-1]

    st.subheader("Latest Stock Data")
    st.write(latest)

    # Clean values
    open_val = float(latest["Open"])
    high_val = float(latest["High"])
    low_val = float(latest["Low"])
    volume_val = float(latest["Volume"])

    # Check missing values
    if any(np.isnan([open_val, high_val, low_val, volume_val])):
        st.error("⚠️ Missing data — cannot make prediction")
    else:
        input_data = np.array([[open_val, high_val, low_val, volume_val]])

        if st.button("Predict"):
            prediction = model.predict(input_data)

            if prediction[0] == 1:
                st.success("📈 Price likely to go UP tomorrow")
            else:
                st.error("📉 Price likely to go DOWN tomorrow")

else:
    st.warning("No data found for this ticker")
