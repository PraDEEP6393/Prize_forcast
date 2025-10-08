import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import matplotlib.pyplot as plt

# --------------------------
# Streamlit Configuration
# --------------------------
st.set_page_config(page_title="Stock Trend Predictor", layout="wide")
st.title("📈 LSTM + DMA Stock Trend Predictor")

# --------------------------
# Load model and scaler dynamically
# --------------------------
@st.cache_resource
def load_model_and_scaler(stock_name):
    model_scaler_paths = {
        "Tata Motors": ("model_tatamotors.keras", "scaler_tatamotors.pkl"),
        "Reliance": ("model_reliance.keras", "scaler_reliance.pkl"),
        "Infosys": ("model_infosys.keras", "scaler_infosys.pkl"),
        "HDFC Bank": ("model_hdfc.keras", "scaler_hdfc.pkl")
    }

    if stock_name not in model_scaler_paths:
        raise ValueError("Model and scaler not available for this stock yet.")

    model_path, scaler_path = model_scaler_paths[stock_name]
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

# --------------------------
# Download stock data
# --------------------------
@st.cache_data
def download_stock(ticker):
    df = yf.download(ticker, start="2015-01-01")
    df['50DMA'] = df['Close'].rolling(window=50).mean()
    df['200DMA'] = df['Close'].rolling(window=200).mean()
    df.dropna(inplace=True)
    return df

# --------------------------
# Detect trend based on DMAs
# --------------------------
def detect_trend(latest_row):
    close_price = float(latest_row['Close'])
    dma50 = float(latest_row['50DMA'])
    dma200 = float(latest_row['200DMA'])
    
    if dma50 > dma200 and close_price > dma50:
        return "📈 Strong Uptrend"
    elif dma50 > dma200 and close_price < dma50:
        return "🟡 Weak Uptrend"
    elif dma50 < dma200 and close_price < dma50:
        return "📉 Strong Downtrend"
    elif dma50 < dma200 and close_price > dma50:
        return "🟠 Weak Downtrend"
    else:
        return "⚪ No clear trend / Possible reversal"

# --------------------------
# Predict tomorrow's price
# --------------------------
def predict_tomorrow(model, scaler, df):
    features = ['Close', '50DMA', '200DMA']
    last_60 = df[features].tail(60)
    if len(last_60) < 60:
        return None

    scaled = scaler.transform(last_60.values)
    X = scaled.reshape(1, scaled.shape[0], scaled.shape[1])
    pred_scaled = model.predict(X)
    
    # Handle scaler shape mismatch safely
    try:
        pred_full = np.concatenate((pred_scaled, np.zeros((pred_scaled.shape[0], 2))), axis=1)
        pred = scaler.inverse_transform(pred_full)[:, 0]
    except Exception:
        pred = scaler.inverse_transform(pred_scaled)[:, 0]
        
    return float(pred[0])

# --------------------------
# Streamlit UI
# --------------------------

# Dropdown for stock selection
stock_name = st.selectbox(
    "Select Stock",
    ["Tata Motors", "Reliance", "Infosys", "HDFC Bank"],
    index=0,
    help="Choose a stock to analyze and predict its trend"
)

# Map stock to ticker symbol
ticker_map = {
    "Tata Motors": "TATAMOTORS.NS",
    "Reliance": "RELIANCE.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS"
}
ticker = ticker_map[stock_name]

# Run Button
if st.button("🔍 Load Data & Predict"):
    try:
        with st.spinner(f"Fetching {stock_name} data..."):
            df = download_stock(ticker)

        if df.empty:
            st.error("No data found for this ticker.")
        else:
            latest = df.iloc[-1]
            trend = detect_trend(latest)

            # Display metrics
            latest_close = float(latest['Close'])
            dma50 = float(latest['50DMA'])
            dma200 = float(latest['200DMA'])

            col1, col2, col3 = st.columns(3)
            col1.metric("Close Price", f"₹{latest_close:.2f}")
            col2.metric("50 DMA", f"₹{dma50:.2f}")
            col3.metric("200 DMA", f"₹{dma200:.2f}")

            st.info(f"**Current Trend:** {trend}")

            # Load model and predict
            with st.spinner("Loading model and predicting..."):
                model, scaler = load_model_and_scaler(stock_name)
                tomorrow_pred = predict_tomorrow(model, scaler, df)

            if tomorrow_pred is not None:
                tomorrow_date = df.index[-1] + pd.Timedelta(days=1)
                st.metric("Predicted Close Price (Tomorrow)", f"₹{tomorrow_pred:.2f}")

                # Plotting
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(df.index, df['Close'], label='Close', color='blue')
                ax.plot(df.index, df['50DMA'], '--', label='50DMA', color='orange')
                ax.plot(df.index, df['200DMA'], ':', label='200DMA', color='red')
                ax.set_title(f"{ticker} Close Price & DMA Trends")
                ax.set_xlabel("Date")
                ax.set_ylabel("Price (₹)")
                ax.legend()
                st.pyplot(fig)
            else:
                st.warning("Not enough data to predict tomorrow (need at least 60 days).")

    except Exception as e:
        st.error("An error occurred:")
        st.exception(e)
