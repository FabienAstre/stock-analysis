import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.volatility import BollingerBands
import plotly.graph_objects as go

st.set_page_config(page_title="Advanced Stock Analysis App", layout="wide")
st.title("📊 Advanced Stock Analysis & Trading Guidance App")

# --- Sidebar ---
st.sidebar.header("Settings")
tickers_input = st.sidebar.text_input("Enter Stock Tickers (comma-separated)", "AAPL,MSFT")
period = st.sidebar.selectbox("Select Historical Period", ["1mo", "3mo", "6mo", "1y", "5y"])
tickers = [t.strip().upper() for t in tickers_input.split(",")]

# --- Simulated Sentiment Score ---
def get_sentiment_score(ticker):
    try:
        return round(np.random.uniform(-1,1), 2)
    except:
        return None

# --- Loop through tickers ---
for ticker in tickers:
    st.header(f"{ticker} Analysis")
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)

        if hist.empty:
            st.warning("No historical data available.")
            continue

        # --- Latest Price ---
        latest_price = hist['Close'].iloc[-1]
        st.metric(label="Latest Closing Price", value=f"${latest_price:.2f}")

        # --- Candlestick + Bollinger + MACD ---
        st.subheader("Candlestick Chart + Bollinger Bands + MACD")
        hist['SMA20'] = SMAIndicator(hist['Close'], 20).sma_indicator()
        bb = BollingerBands(hist['Close'], 20, 2)
        hist['BB_upper'] = bb.bollinger_hband()
        hist['BB_lower'] = bb.bollinger_lband()
        macd_indicator = MACD(hist['Close'])
        hist['MACD'] = macd_indicator.macd()
        hist['MACD_signal'] = macd_indicator.macd_signal()

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=hist.index,
            open=hist['Open'],
            high=hist['High'],
            low=hist['Low'],
            close=hist['Close'],
            name='Candlestick'
        ))
        fig.add_trace(go.Scatter(x=hist.index, y=hist['BB_upper'], line=dict(color='orange'), name='BB Upper'))
        fig.add_trace(go.Scatter(x=hist.index, y=hist['BB_lower'], line=dict(color='orange'), name='BB Lower'))
        fig.update_layout(height=500, title=f"{ticker} Candlestick + Bollinger Bands", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        **Interpretation & Actions:**  
        - Price near **upper Bollinger Band** → potential overbought → consider caution or sell.  
        - Price near **lower Bollinger Band** → potential oversold → consider buy.  
        - **MACD line crosses above signal** → bullish trend → consider buying.  
        - **MACD line crosses below signal** → bearish trend → consider selling or holding off.
        """)

        # --- Advanced Prediction + Anomaly ---
        st.subheader("Advanced Prediction & Anomaly Signals")
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

            hist['Returns'] = hist['Close'].pct_change()
            hist['ZScore'] = (hist['Returns'] - hist['Returns'].mean()) / hist['Returns'].std()
            is_anomaly = abs(hist['ZScore'].iloc[-1]) > 2

            st.write(f"Predicted next-day close: **${predicted_price:.2f}**")
            if predicted_price > current_price * 1.002:
                st.success("Signal: BUY ✅")
            elif predicted_price < current_price * 0.998:
                st.error("Signal: SELL ❌")
            else:
                st.info("Signal: HOLD ⏸️")
            if is_anomaly:
                st.warning("Anomaly detected: unusual price movement ⚡")
        else:
            st.warning("Not enough data for prediction.")

        st.markdown("""
        **Interpretation & Actions:**  
        - Prediction > current price → potential buy  
        - Prediction < current price → potential sell  
        - Large Z-score anomaly → market behaving unusually, use caution
        """)

        # --- Déjà Vue Trading ---
        st.subheader("🔁 Déjà Vue Trading Signals")
        pattern_length = 5
        threshold_similarity = 0.95
        last_pattern = hist['Close'].values[-pattern_length:]
        matches = []
        for i in range(len(hist) - pattern_length):
            hist_pattern = hist['Close'].values[i:i + pattern_length]
            similarity = np.corrcoef(last_pattern, hist_pattern)[0, 1]
            if similarity >= threshold_similarity:
                matches.append((i, hist.index[i], similarity))
        if matches:
            matches_df = pd.DataFrame(matches, columns=['Index', 'Date', 'Similarity'])
            st.dataframe(matches_df)
        else:
            st.info("No similar historical patterns found.")

        st.markdown("""
        **Interpretation & Actions:**  
        - High similarity patterns → history may repeat  
        - If past pattern led to gains → potential buy  
        - If past pattern led to losses → potential caution or sell  
        - Always combine with trend, RSI, and volatility indicators
        """)

        # --- Trending & Mean-Reversion ---
        st.subheader("Trending & Mean-Reversion Signals")
        hist['SMA50'] = hist['Close'].rolling(50).mean()
        hist['SMA200'] = hist['Close'].rolling(200).mean()
        hist['Trend'] = np.where(hist['SMA50'] > hist['SMA200'], 'Uptrend', 'Downtrend')
        st.write("Latest Trend Signal:", hist['Trend'].iloc[-1])
        hist['Deviation'] = (hist['Close'] - hist['SMA50']) / hist['SMA50']
        if hist['Deviation'].iloc[-1] > 0.03:
            st.warning("Price above SMA50 by >3% → potential pullback")
        elif hist['Deviation'].iloc[-1] < -0.03:
            st.success("Price below SMA50 by >3% → potential bounce")

        st.markdown("""
        **Interpretation & Actions:**  
        - Uptrend + slight pullback → consider buying  
        - Downtrend + slight bounce → consider short-term profit  
        - Large deviation → watch for mean-reversion
        """)

        # --- Price to Tangible Book ---
        st.subheader("💰 Price to Tangible Book (PTB)")
        try:
            book_value = stock.info.get('bookValue')
            intangible = stock.info.get('intangibleAssets', 0)
            ptb = stock.info['currentPrice'] / (book_value - intangible)
            st.write(f"Price to Tangible Book: **{ptb:.2f}**")
            if ptb < 1:
                st.success("PTB < 1 → potentially undervalued")
            else:
                st.info("PTB > 1 → consider valuation")
        except:
            st.write("PTB not available")

        st.markdown("""
        **Interpretation & Actions:**  
        - PTB < 1 + bullish indicators → consider buying  
        - PTB > 1 + bearish indicators → consider caution or selling
        """)

        # --- Situational Analysis & Sentiment ---
        st.subheader("Situational Analysis & Sentiment Score")
        rsi = RSIIndicator(hist['Close'], 14).rsi().iloc[-1]
        vol_percent = (hist['Close'].iloc[-1] - hist['Close'].iloc[-20:].mean()) / hist['Close'].iloc[-20:].mean() * 100
        sentiment_score = get_sentiment_score(ticker)
        st.write(f"Simulated Sentiment Score: {sentiment_score} (-1 negative → +1 positive)")

        if rsi > 70:
            st.warning("RSI > 70 → overbought")
        elif rsi < 30:
            st.success("RSI < 30 → oversold")
        if hist['Close'].iloc[-1] >= stock.info.get('fiftyTwoWeekHigh',0) * 0.98:
            st.info("Price near 52-week high")
        elif hist['Close'].iloc[-1] <= stock.info.get('fiftyTwoWeekLow',0) * 1.02:
            st.info("Price near 52-week low")
        if abs(vol_percent) > 5:
            st.write(f"Significant deviation from 20-day mean: {vol_percent:.2f}%")

        st.markdown("""
        **Interpretation & Actions:**  
        - Combine RSI, volatility, 52-week levels, and sentiment  
        - Positive sentiment + oversold → strong buy opportunity  
        - Negative sentiment + overbought → consider selling  
        - Significant deviation → monitor for reversals
        """)

    except Exception as e:
        st.error(f"Error processing {ticker}: {e}")
