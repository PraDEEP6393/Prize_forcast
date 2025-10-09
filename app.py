import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Stock Trend Predictor", layout="wide")

# --------------------------
# Load model and scaler
# --------------------------
@st.cache_resource
def load_model_and_scaler():
    model = load_model("model.keras") 
    scaler = joblib.load("scaler.pkl")
    return model, scaler

# --------------------------
# Download stock data
# --------------------------
@st.cache_data
def download_stock(ticker, period="5y"):
    #df = yf.download(ticker, start="2015-01-01", end=pd.Timestamp.today().strftime('%Y-%m-%d'))
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
    if dma50 > dma200 and close_price > dma50 and close_price > dma200:
        return "Strong Uptrend"
    elif dma50 > dma200 and close_price < dma50 and close_price > dma200:
        return "Weak Uptrend"
    elif dma50 > dma200 and close_price < dma50 and close_price < dma200:
        return "Wait for confirmation! Trend reversal"
    elif dma50 < dma200 and close_price < dma50 and close_price < dma200:
        return "Strong Downtrend"
    elif dma50 < dma200 and close_price > dma50 and close_price < dma200:
        return "Weak Downtrend"
    elif dma50 < dma200 and close_price > dma50 and close_price > dma200:
        return "Wait for confirmation! Trend reversal"
    else:
        return "No clear trend"

# --------------------------
# Predict tomorrow's price
# --------------------------
def predict_tomorrow(model, scaler, df):
    features = ['Close','50DMA','200DMA']
    last_60 = df[features].tail(60)
    if len(last_60) < 60:
        return None
    scaled = scaler.transform(last_60.values)
    X = scaled.reshape(1, scaled.shape[0], scaled.shape[1])
    pred_scaled = model.predict(X)
    pred_full = np.concatenate((pred_scaled, np.zeros((pred_scaled.shape[0],2))), axis=1)
    pred = scaler.inverse_transform(pred_full)[:,0]
    return float(pred[0])

# --------------------------
# Streamlit UI
# --------------------------
st.title("📈 LSTM + DMA Stock Trend Predictor")

ticker = st.text_input("Enter Ticker (Yahoo Finance)", value="TATAMOTORS.NS")
#period = st.selectbox("Select historical period", ["1y","2y","3y","5y","10y"], index=3)
run = st.button("Load Data & Predict")

if run:
    try:
        #df = download_stock(ticker, period=period)
        df = download_stock(ticker)
        if df.empty:
            st.error("No data found for this ticker.")
        else:
            latest = df.iloc[-1]
            trend = detect_trend(latest)
            st.metric("Latest Close Price", f"₹{latest['Close'].iloc[0]:.2f}")
            st.metric("50 DMA Price", f"₹{latest['50DMA'].iloc[0]:.2f}")
            st.metric("200 DMA Price", f"₹{latest['200DMA'].iloc[0]:.2f}")
            st.info(f"Trend: {trend}")

            model, scaler = load_model_and_scaler()
            tomorrow_pred = predict_tomorrow(model, scaler, df)
            tomorrow_date = df.index[-1] + pd.Timedelta(days=1)
            if tomorrow_pred:
                st.metric("Predicted Close Price of Tomorrow", f"₹{tomorrow_pred:.2f}")

                # Plot historical close + DMAs
                fig, ax = plt.subplots(figsize=(10,4))
                ax.plot(df.index, df['Close'], label='Close')
                ax.plot(df.index, df['50DMA'], linestyle='--', label='50DMA')
                ax.plot(df.index, df['200DMA'], linestyle=':', label='200DMA')
                tomorrow_date = df.index[-1] + pd.Timedelta(days=1)
                ax.legend()
                ax.set_title(f"{ticker} Close Price & DMAs")
                ax.set_xlabel("Date")
                ax.set_ylabel("Price")
                st.pyplot(fig)
                #st.info(f"Date: {tomorrow_date}")
            else:
                st.warning("Not enough data to predict tomorrow (need at least 60 days).")
    except Exception as e:
        st.exception(e)
