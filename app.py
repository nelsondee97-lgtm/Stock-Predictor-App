import streamlit as st
import joblib
import numpy as np
import yfinance as yf

# Load model
model = joblib.load("model.pkl")

st.title("📈 Stock Movement Predictor")

# User selects stock
ticker = st.text_input("Enter Stock Ticker (e.g. AAPL, TSLA)", "AAPL")

# Fetch data
data = yf.download(ticker, period="5d")

if not data.empty:
    latest = data.iloc[-1]

    st.subheader("Latest Stock Data")
    st.write(latest)

    # Prepare input for model
    input_data = np.array([[
        latest["Open"],
        latest["High"],
        latest["Low"],
        latest["Volume"]
    ]])

    if st.button("Predict"):
        prediction = model.predict(input_data)

        if prediction[0] == 1:
            st.success("📈 Price likely to go UP tomorrow")
        else:
            st.error("📉 Price likely to go DOWN tomorrow")
else:
    st.warning("Invalid ticker or no data found")
