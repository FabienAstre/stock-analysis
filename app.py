import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.volatility import BollingerBands
import plotly.graph_objects as go

# --- Page config ---
st.set_page_config(page_title="Advanced Stock Analysis App", layout="wide")
st.title("📊 Advanced Stock Analysis & Trading Guidance App")

# --- Sidebar ---
st.sidebar.header("Settings")
tickers_input = st.sidebar.text_input("Enter Stock Tickers (comma-separated)", "AAPL,MSFT")
period = st.sidebar.selectbox("Select Historical Period", ["1mo", "3mo", "6mo", "1y", "5y"])
tickers = [t.strip().upper() for t in tickers_input.split(",")]

# --- Chart Indicator Selection ---
st.sidebar.header("Chart Options")
show_sma = st.sidebar.checkbox("Show SMA50 & SMA200", value=True)
show_bb = st.sidebar.checkbox("Show Bollinger Bands", value=True)
show_macd = st.sidebar.checkbox("Show MACD", value=True)
show_deja = st.sidebar.checkbox("Show Déjà Vue Patterns", value=True)

# --- Simulated Sentiment Score ---
def get_sentiment_score(ticker):
    try:
        return round(np.random.uniform(-1, 1), 2)
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

        latest_price = hist['Close'].iloc[-1]
        st.metric(label="Latest Closing Price", value=f"${latest_price:.2f}")

        # --- Technical Indicators ---
        hist['SMA50'] = SMAIndicator(hist['Close'], 50).sma_indicator()
        hist['SMA200'] = SMAIndicator(hist['Close'], 200).sma_indicator()
        hist['SMA20'] = SMAIndicator(hist['Close'], 20).sma_indicator()

        # Bollinger Bands
        bb = BollingerBands(hist['Close'], 20, 2)
        hist['BB_upper'] = bb.bollinger_hband()
        hist['BB_lower'] = bb.bollinger_lband()

        # MACD
        macd_indicator = MACD(hist['Close'])
        hist['MACD'] = macd_indicator.macd()
        hist['MACD_signal'] = macd_indicator.macd_signal()
        hist['MACD_cross'] = np.where(hist['MACD'] > hist['MACD_signal'], 'bullish', 'bearish')

        # RSI
        rsi = RSIIndicator(hist['Close'], 14).rsi().iloc[-1]

        # Déjà Vue
        pattern_length = 5
        last_pattern = hist['Close'].values[-pattern_length:]
        matches = []
        for i in range(len(hist) - pattern_length - 1):
            hist_pattern = hist['Close'].values[i:i+pattern_length]
            similarity = np.corrcoef(last_pattern, hist_pattern)[0,1]
            if similarity > 0.95:
                matches.append((i, hist.index[i], similarity))

        # --- Interactive Candlestick Chart ---
        st.subheader("Candlestick Chart with Selected Indicators")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=hist.index,
            open=hist['Open'],
            high=hist['High'],
            low=hist['Low'],
            close=hist['Close'],
            name='Candlestick'
        ))

        if show_sma:
            fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA50'], line=dict(color='blue'), name='SMA50'))
            fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA200'], line=dict(color='red'), name='SMA200'))

        if show_bb:
            fig.add_trace(go.Scatter(x=hist.index, y=hist['BB_upper'], line=dict(color='orange'), name='BB Upper'))
            fig.add_trace(go.Scatter(x=hist.index, y=hist['BB_lower'], line=dict(color='orange'), name='BB Lower'))

        if show_macd:
            # Add MACD line below in a new y-axis
            fig.add_trace(go.Scatter(x=hist.index, y=hist['MACD'], line=dict(color='green'), name='MACD'))
            fig.add_trace(go.Scatter(x=hist.index, y=hist['MACD_signal'], line=dict(color='black'), name='MACD Signal'))

        if show_deja and matches:
            for match in matches:
                fig.add_vline(x=match[1], line_width=1, line_dash="dash", line_color="purple")

        fig.update_layout(height=600, xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig, use_container_width=True)

        # --- Trend & Mean Reversion ---
        trend = "Uptrend" if hist['SMA50'].iloc[-1] > hist['SMA200'].iloc[-1] else "Downtrend"
        st.write(f"**Trend (SMA50 vs SMA200):** {trend}")

        # --- PTB ---
        try:
            ptb = stock.info.get('priceToBook', None)
        except:
            ptb = None
        st.write(f"**Price-to-Tangible Book (PTB):** {ptb if ptb else 'N/A'}")

        # --- Sentiment ---
        sentiment_score = get_sentiment_score(ticker)
        st.write(f"**Simulated Sentiment Score:** {sentiment_score}")

        # --- Deviation ---
        deviation = (hist['Close'].iloc[-1] - hist['SMA50'].iloc[-1]) / hist['SMA50'].iloc[-1]

        # --- Advanced Prediction & Anomaly Signals ---
        st.subheader("Advanced Prediction & Anomaly Signals")
        lookback = 30
        if len(hist) >= lookback:
            hist['Day'] = np.arange(len(hist))
            X = hist['Day'].values[-lookback:].reshape(-1,1)
            y = hist['Close'].values[-lookback:]
            model = LinearRegression()
            model.fit(X,y)
            next_day = np.array([[X[-1][0]+1]])
            predicted_price = model.predict(next_day)[0]
            current_price = hist['Close'].iloc[-1]

            hist['Returns'] = hist['Close'].pct_change()
            hist['ZScore'] = (hist['Returns'] - hist['Returns'].mean())/hist['Returns'].std()
            is_anomaly = abs(hist['ZScore'].iloc[-1])>2

            # Determine signal
            if predicted_price > current_price*1.002:
                signal = "BUY ✅"
            elif predicted_price < current_price*0.998:
                signal = "SELL ❌"
            else:
                signal = "HOLD ⏸️"

            st.write(f"Predicted next-day close: **${predicted_price:.2f}**")
            st.write(f"Signal: {signal}")

            interpretation = []
            if predicted_price > current_price:
                interpretation.append(f"Predicted price (${predicted_price:.2f}) higher than current → potential buy")
            elif predicted_price < current_price:
                interpretation.append(f"Predicted price (${predicted_price:.2f}) lower than current → potential sell")
            else:
                interpretation.append(f"Predicted price roughly equal → hold")
            if is_anomaly:
                interpretation.append("Large Z-score anomaly detected → use caution")
            for item in interpretation:
                st.write("- " + item)
        else:
            st.warning("Not enough data for prediction (requires at least 30 days).")

        # --- Dynamic Key Reasons ---
        reasons = []
        reasons.append(f"Trend: {trend}")
        if rsi < 30:
            reasons.append("RSI oversold → potential buy")
        elif rsi > 70:
            reasons.append("RSI overbought → potential sell")
        if hist['MACD_cross'].iloc[-1] == 'bullish':
            reasons.append("MACD bullish crossover")
        else:
            reasons.append("MACD bearish crossover")
        if hist['Close'].iloc[-1] < hist['BB_lower'].iloc[-1]:
            reasons.append("Price below lower Bollinger → potential buy")
        elif hist['Close'].iloc[-1] > hist['BB_upper'].iloc[-1]:
            reasons.append("Price above upper Bollinger → potential sell")
        if matches:
            reasons.append("Déjà Vue pattern found")
        if ptb:
            if ptb < 1:
                reasons.append("PTB < 1 → undervalued")
            elif ptb > 2:
                reasons.append("PTB > 2 → overvalued")
        if sentiment_score:
            if sentiment_score > 0.3:
                reasons.append("Positive sentiment")
            elif sentiment_score < -0.3:
                reasons.append("Negative sentiment")
        if deviation < -0.03:
            reasons.append("Price below SMA50 → potential bounce")
        elif deviation > 0.03:
            reasons.append("Price above SMA50 → potential pullback")

        st.subheader("📌 Key Reasons")
        for r in reasons:
            st.write("- " + r)

        # --- Summary & Action Guidance ---
        score = 0
        if signal == "BUY ✅":
            score += 1
        elif signal == "SELL ❌":
            score -= 1
        if deviation < -0.03 or rsi < 30 or (hist['Close'].iloc[-1] < hist['BB_lower'].iloc[-1]):
            score += 1
        if deviation > 0.03 or rsi > 70 or (hist['Close'].iloc[-1] > hist['BB_upper'].iloc[-1]):
            score -= 1

        if score > 0:
            overall = "BUY ✅"
        elif score < 0:
            overall = "SELL ❌"
        else:
            overall = "HOLD ⏸️"

        st.subheader("📌 Overall Recommendation")
        st.markdown(f"**{overall}**")
        st.markdown("Based on combined indicators: trend, SMA, MACD, Bollinger, RSI, predicted price, Déjà Vue, PTB, sentiment, deviation.")

    except Exception as e:
        st.error(f"Error processing {ticker}: {e}")
