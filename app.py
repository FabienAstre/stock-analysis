import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, EMAIndicator

st.set_page_config(page_title="Stock Analysis App", layout="wide")

st.title("📊 Stock Analysis App")

# --- Sidebar ---
st.sidebar.header("Settings")
tickers_input = st.sidebar.text_input("Enter Stock Tickers (comma-separated)", "AAPL,MSFT")
period = st.sidebar.selectbox("Select Historical Period", ["1mo", "3mo", "6mo", "1y", "5y"])

tickers = [t.strip().upper() for t in tickers_input.split(",")]

# --- Fetch and Display Data ---
for ticker in tickers:
    st.header(f"{ticker} Stock Data")
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        
        st.subheader("Price Chart")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], name='Close'))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Technical Indicators")
        hist['SMA20'] = SMAIndicator(hist['Close'], 20).sma_indicator()
        hist['EMA20'] = EMAIndicator(hist['Close'], 20).ema_indicator()
        hist['RSI14'] = RSIIndicator(hist['Close'], 14).rsi()
        
        st.line_chart(hist[['Close', 'SMA20', 'EMA20']])
        st.line_chart(hist['RSI14'])
        
        st.subheader("Key Info")
        st.write({
            "Market Cap": stock.info.get("marketCap"),
            "P/E Ratio": stock.info.get("trailingPE"),
            "Dividend Yield": stock.info.get("dividendYield"),
            "52 Week High": stock.info.get("fiftyTwoWeekHigh"),
            "52 Week Low": stock.info.get("fiftyTwoWeekLow")
        })
        
    except Exception as e:
        st.error(f"Error fetching {ticker}: {e}")
