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
model_path = st.text_input("Enter Model Path (e.g., Stock Predictions Model.keras):", 
                           "Stock Predictions Model.keras")

# Initialize session state to manage the model and data
if "model" not in st.session_state:
    st.session_state.model = None
if "data" not in st.session_state:
    st.session_state.data = None

# Button to fetch data and predict
if st.button("Fetch Data and Predict"):
    try:
        # Load the model
        st.session_state.model = load_model(model_path)
        st.success("Model loaded successfully!")
        
        # Fetch stock data
        st.session_state.data = yf.download(ticker, start=start_date, end=end_date)
        if not st.session_state.data.empty:
            st.write(f"Showing data for {ticker}:")
            st.dataframe(st.session_state.data.tail())

            # Data preparation
            st.session_state.data['Date'] = st.session_state.data.index
            st.session_state.data['Days'] = (st.session_state.data['Date'] - st.session_state.data['Date'].min()).dt.days
            X = st.session_state.data[['Days']]
            y = st.session_state.data['Close']

            # Normalize the data (if required by the model)
            X_scaled = X / X.max()
            
            # Predict
            predictions = st.session_state.model.predict(X_scaled)

            # Plot results
            plt.figure(figsize=(10, 6))
            plt.plot(st.session_state.data['Date'], y, label='Actual Prices', color='blue')
            plt.plot(st.session_state.data['Date'], predictions, label='Predicted Prices', color='red')
            plt.xlabel("Date")
            plt.ylabel("Stock Price")
            plt.title(f"{ticker} Stock Price Prediction")
            plt.legend()
            st.pyplot(plt)
        else:
            st.error("No data found. Please check the ticker symbol or date range.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Future prediction block
future_days = st.number_input("Enter number of days to predict future prices:", min_value=1, max_value=365)

if st.button("Predict Future Prices"):
    try:
        # Ensure model and data are loaded
        if st.session_state.model is not None and st.session_state.data is not None:
            # Generate future dates
            future_dates = pd.DataFrame({'Days': [st.session_state.data['Days'].max() + i for i in range(1, future_days + 1)]})
            
            # Scale future dates using the same max value from training
            max_days = st.session_state.data['Days'].max()  # Max Days from original data
            future_dates_scaled = future_dates / max_days
            
            # Predict future prices
            future_prices = st.session_state.model.predict(future_dates_scaled)
            
            # Display results
            st.write("Predicted Prices for the next days:")
            st.dataframe(pd.DataFrame({'Days': future_dates['Days'], 'Predicted Prices': future_prices.flatten()}))
        else:
            st.error("Please load the model and fetch data first.")
    except Exception as e:
        st.error(f"An error occurred during future price prediction: {e}")
