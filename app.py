import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Title and description
st.title("Stock Price Prediction App with Pre-Trained Model")
st.write("""
This app predicts stock prices using a pre-trained deep learning model.
""")

# Input for stock ticker
ticker = st.text_input("Enter Stock Ticker Symbol (e.g., AAPL, MSFT):", "AAPL")

# Date range selection
start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = st.date_input("End Date", value=pd.to_datetime("2023-01-01"))

# Load pre-trained model
model_path = st.text_input("C:/Users/khadi/Desktop/stockPrediction/Stock Predictions Model.keras", "Stock Predictions Model.keras")

if st.button("Fetch Data and Predict"):
    try:
        # Load the model
        model = load_model(model_path)
        st.success("Model loaded successfully!")
        
        # Fetch stock data
        data = yf.download(ticker, start=start_date, end=end_date)
        if not data.empty:
            st.write(f"Showing data for {ticker}:")
            st.dataframe(data.tail())

            # Data preparation
            data['Date'] = data.index
            data['Days'] = (data['Date'] - data['Date'].min()).dt.days
            X = data[['Days']]
            y = data['Close']

            # Normalize the data (if required by the model)
            X_scaled = X / X.max()
            
            # Predict
            predictions = model.predict(X_scaled)

            # Plot results
            plt.figure(figsize=(10, 6))
            plt.plot(data['Date'], y, label='Actual Prices', color='blue')
            plt.plot(data['Date'], predictions, label='Predicted Prices', color='red')
            plt.xlabel("Date")
            plt.ylabel("Stock Price")
            plt.title(f"{ticker} Stock Price Prediction")
            plt.legend()
            st.pyplot(plt)

            # Future prediction
            future_days = st.number_input("Enter number of days to predict future prices:", min_value=1, max_value=365)
            if st.button("Predict Future Prices"):
                future_dates = pd.DataFrame({'Days': [data['Days'].max() + i for i in range(1, future_days + 1)]})
                future_dates_scaled = future_dates / future_dates.max()
                future_prices = model.predict(future_dates_scaled)
                st.write("Predicted Prices for the next days:")
                st.dataframe(pd.DataFrame({'Days': future_dates['Days'], 'Predicted Prices': future_prices.flatten()}))
        else:
            st.error("No data found. Please check the ticker symbol or date range.")
    except Exception as e:
        st.error(f"An error occurred: {e}")