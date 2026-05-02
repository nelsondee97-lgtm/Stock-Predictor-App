import streamlit as st
import joblib
import numpy as np

model = joblib.load("model.pkl")

st.title("📈 Stock Movement Predictor")

st.write("Enter today's stock data to predict tomorrow's movement.")

open_price = st.number_input("Open Price")
high_price = st.number_input("High Price")
low_price = st.number_input("Low Price")
volume = st.number_input("Volume")

if st.button("Predict"):
    data = np.array([[open_price, high_price, low_price, volume]])
    prediction = model.predict(data)

    if prediction[0] == 1:
        st.success("📈 Price will go UP")
    else:
        st.error("📉 Price will go DOWN")
