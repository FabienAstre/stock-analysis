import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, EMAIndicator
import plotly.graph_objects as go

st.set_page_config(page_title="Stock Analysis App", layout="wide")
st.title("📊 Advanced Stock Analysis & Prediction App")

# --- Sidebar ---
st.sidebar.header("Settings")
tickers_input = st.sidebar.text_input("Enter Stock Tickers (comma-separated)", "AAPL,MSFT")
period = st.sidebar.selectbox("Select Historical Period", ["1mo", "3mo", "6mo", "1y", "5y"])
tickers = [t.strip().upper() for t in tickers_input.split(",")]

# --- Loop through tickers ---
for ticker in tickers:
    st.header(f"{ticker} Analysis")
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)

        if hist.empty:
            st.warning("No historical data available.")
            continue

        # --- Price Chart ---
        st.subheader("Price Chart")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], name='Close'))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        # --- Technical Indicators ---
        st.subheader("Technical Indicators & Explanation")
        hist['SMA20'] = SMAIndicator(hist['Close'], 20).sma_indicator()
        hist['EMA20'] = EMAIndicator(hist['Close'], 20).ema_indicator()
        hist['RSI14'] = RSIIndicator(hist['Close'], 14).rsi()
        st.line_chart(hist[['Close', 'SMA20', 'EMA20']])
        st.line_chart(hist['RSI14'])
        st.write("""
        **Interpretation:**  
        - **SMA/EMA:** Shows the average price trend; EMA reacts faster to recent changes.  
        - **RSI:** >70 = overbought, <30 = oversold. Indicates potential reversal points.
        """)

        # --- Medallion-style Prediction ---
        st.subheader("📈 Short-Term Price Prediction")
        lookback = 30
        if len(hist) >= lookback:
            hist['Day'] = np.arange(len(hist))
            X = hist['Day'].values[-lookback:].reshape(-1,1)
            y = hist['Close'].values[-lookback:]
            model = LinearRegression()
            model.fit(X, y)
            next_day = np.array([[X[-1][0] + 1]])
            predicted_price = model.predict(next_day)[0]
            current_price = hist['Close'].iloc[-1]
            st.write(f"Predicted next-day close: **${predicted_price:.2f}**")
            if predicted_price > current_price * 1.002:
                st.success("Signal: BUY ✅")
            elif predicted_price < current_price * 0.998:
                st.error("Signal: SELL ❌")
            else:
                st.info("Signal: HOLD ⏸️")
        else:
            st.warning("Not enough data for prediction (needs at least 30 days).")
        st.write("**Interpretation:** Linear regression forecasts the next-day price based on recent trends. Compare predicted vs current price for potential signals.")

        # --- Anomaly Trading ---
        st.subheader("⚡ Anomaly Trading Signals")
        hist['Returns'] = hist['Close'].pct_change()
        hist['ZScore'] = (hist['Returns'] - hist['Returns'].mean()) / hist['Returns'].std()
        anomalies = hist[abs(hist['ZScore']) > 2]
        st.dataframe(anomalies[['Close','Returns','ZScore']])
        st.write("**Interpretation:** Z-Score > 2 indicates unusual price movement. Could be a short-term trading opportunity.")

        # --- Déjà Vue Trading ---
        st.subheader("🔁 Déjà Vue Trading Signals")
        pattern_length = 5
        last_pattern = hist['Close'].values[-pattern_length:]
        matches = []
        for i in range(len(hist)-pattern_length-1):
            hist_pattern = hist['Close'].values[i:i+pattern_length]
            similarity = np.corrcoef(last_pattern, hist_pattern)[0,1]
            if similarity > 0.95:
                matches.append((i, hist.index[i], similarity))
        st.dataframe(matches, columns=['Index','Date','Similarity'])
        st.write("**Interpretation:** Detects repeating historical price patterns. High similarity may suggest history could repeat.")

        # --- Trending & Reversion Signals ---
        st.subheader("📊 Trending & Mean-Reversion Signals")
        hist['SMA50'] = hist['Close'].rolling(50).mean()
        hist['SMA200'] = hist['Close'].rolling(200).mean()
        hist['Trend'] = np.where(hist['SMA50'] > hist['SMA200'], 'Uptrend', 'Downtrend')
        st.line_chart(hist[['Close','SMA50','SMA200']])
        st.write("Latest Trend Signal:", hist['Trend'].iloc[-1])
        hist['Deviation'] = (hist['Close'] - hist['SMA50']) / hist['SMA50']
        if hist['Deviation'].iloc[-1] > 0.03:
            st.warning("Price above SMA50 by >3%: potential mean-reversion")
        elif hist['Deviation'].iloc[-1] < -0.03:
            st.success("Price below SMA50 by >3%: potential mean-reversion")
        st.write("**Interpretation:** Trend analysis shows overall market direction. Deviation indicates potential pullback or rebound.")

        # --- Price to Tangible Book ---
        st.subheader("💰 Price to Tangible Book (PTB)")
        try:
            book_value = stock.info.get('bookValue')
            intangible = stock.info.get('intangibleAssets', 0)
            ptb = stock.info['currentPrice'] / (book_value - intangible)
            st.write(f"Price to Tangible Book: **{ptb:.2f}**")
            st.write("**Interpretation:** PTB < 1 may indicate undervalued relative to tangible assets.")
        except:
            st.write("PTB not available")

        # --- Situational Analysis ---
        st.subheader("🧐 Situational Analysis Trading")
        rsi = RSIIndicator(hist['Close'], 14).rsi().iloc[-1]
        vol_percent = (hist['Close'].iloc[-1] - hist['Close'].iloc[-20:].mean()) / hist['Close'].iloc[-20:].mean() * 100
        if rsi > 70:
            st.warning("RSI indicates overbought")
        elif rsi < 30:
            st.success("RSI indicates oversold")
        if hist['Close'].iloc[-1] >= stock.info.get('fiftyTwoWeekHigh',0) * 0.98:
            st.info("Price near 52-week high")
        elif hist['Close'].iloc[-1] <= stock.info.get('fiftyTwoWeekLow',0) * 1.02:
            st.info("Price near 52-week low")
        if abs(vol_percent) > 5:
            st.write(f"Significant deviation from 20-day mean: {vol_percent:.2f}%")
        st.write("**Interpretation:** Situational analysis combines RSI, price relative to 52-week high/low, and recent volatility to find potential trading opportunities.")

    except Exception as e:
        st.error(f"Error fetching {ticker}: {e}")
