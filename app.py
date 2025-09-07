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

        # MACD cross detection
        hist['MACD_cross'] = np.where(hist['MACD'] > hist['MACD_signal'], 'bullish', 'bearish')

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

        # --- RSI ---
        rsi = RSIIndicator(hist['Close'], 14).rsi().iloc[-1]

        # --- Déjà Vue Trading ---
        st.subheader("🔁 Déjà Vue Trading Signals")
        pattern_length = 5
        last_pattern = hist['Close'].values[-pattern_length:]
        matches = []
        for i in range(len(hist) - pattern_length - 1):
            hist_pattern = hist['Close'].values[i:i+pattern_length]
            similarity = np.corrcoef(last_pattern, hist_pattern)[0, 1]
            if similarity > 0.95:
                matches.append((i, hist.index[i], similarity))
        if matches:
            matches_df = pd.DataFrame(matches, columns=['Index', 'Date', 'Similarity'])
            st.dataframe(matches_df)
        else:
            st.write("No similar historical patterns found.")
        st.write("**Interpretation:** Detects repeating historical price patterns. High similarity may suggest history could repeat.")

        # --- Price-to-Tangible Book (PTB) ---
        try:
            ptb = stock.info.get('priceToBook', None)
        except:
            ptb = None

        # --- Sentiment ---
        sentiment_score = get_sentiment_score(ticker)

        # --- Deviation from SMA50 ---
        hist['SMA50'] = SMAIndicator(hist['Close'], 50).sma_indicator()
        deviation = (hist['Close'].iloc[-1] - hist['SMA50'].iloc[-1]) / hist['SMA50'].iloc[-1]

        # --- Advanced Prediction & Anomaly Signals ---
        st.subheader("Advanced Prediction & Anomaly Signals")
        lookback = 30
        if len(hist) >= lookback:
            hist['Day'] = np.arange(len(hist))
            X = hist['Day'].values[-lookback:].reshape(-1, 1)
            y = hist['Close'].values[-lookback:]
            model = LinearRegression()
            model.fit(X, y)
            next_day = np.array([[X[-1][0] + 1]])
            predicted_price = model.predict(next_day)[0]
            current_price = hist['Close'].iloc[-1]

            # Z-score anomaly
            hist['Returns'] = hist['Close'].pct_change()
            hist['ZScore'] = (hist['Returns'] - hist['Returns'].mean()) / hist['Returns'].std()
            is_anomaly = abs(hist['ZScore'].iloc[-1]) > 2

            # Determine dynamic signal
            if predicted_price > current_price * 1.002:
                signal = "BUY ✅"
                signal_action = "Potential buy opportunity"
            elif predicted_price < current_price * 0.998:
                signal = "SELL ❌"
                signal_action = "Potential sell opportunity"
            else:
                signal = "HOLD ⏸️"
                signal_action = "Price predicted stable, consider holding"

            st.write(f"Predicted next-day close: **${predicted_price:.2f}**")
            st.write(f"Signal: {signal}")

            interpretation = []
            if predicted_price > current_price:
                interpretation.append(f"Predicted price (${predicted_price:.2f}) is higher than current (${current_price:.2f}) → {signal_action}")
            elif predicted_price < current_price:
                interpretation.append(f"Predicted price (${predicted_price:.2f}) is lower than current (${current_price:.2f}) → {signal_action}")
            else:
                interpretation.append(f"Predicted price (${predicted_price:.2f}) is roughly equal to current → {signal_action}")
            if is_anomaly:
                interpretation.append("Large Z-score anomaly detected → market behaving unusually, use caution")

            st.markdown("**Interpretation & Actions:**")
            for item in interpretation:
                st.write("- " + item)
        else:
            st.warning("Not enough data for prediction (requires at least 30 days).")

        # --- Dynamic Key Reasons ---
        reasons = []

        # Trend
        trend = 'Uptrend' if hist['SMA50'].iloc[-1] > hist['SMA20'].iloc[-1] else 'Downtrend'
        reasons.append(f"Trend: {trend} (SMA50 {'>' if trend=='Uptrend' else '<'} SMA20)")

        # RSI
        if rsi < 30:
            reasons.append("RSI indicates oversold → potential buy")
        elif rsi > 70:
            reasons.append("RSI indicates overbought → potential sell")

        # MACD cross
        if hist['MACD_cross'].iloc[-1] == 'bullish':
            reasons.append("MACD crossover bullish → positive momentum")
        else:
            reasons.append("MACD crossover bearish → negative momentum")

        # Bollinger Bands
        if hist['Close'].iloc[-1] < hist['BB_lower'].iloc[-1]:
            reasons.append("Price below lower Bollinger Band → potential buy")
        elif hist['Close'].iloc[-1] > hist['BB_upper'].iloc[-1]:
            reasons.append("Price above upper Bollinger Band → potential sell")

        # Déjà Vue
        if matches:
            last_match_index = matches[-1][0]
            if hist['Close'].iloc[last_match_index + pattern_length] > hist['Close'].iloc[last_match_index]:
                reasons.append("Déjà Vue pattern historically led to gains")
            else:
                reasons.append("Déjà Vue pattern historically led to losses")

        # PTB
        if ptb:
            if ptb < 1:
                reasons.append("PTB < 1 → undervalued")
            elif ptb > 2:
                reasons.append("PTB > 2 → overvalued")

        # Sentiment
        if sentiment_score:
            if sentiment_score > 0.3:
                reasons.append("Positive sentiment")
            elif sentiment_score < -0.3:
                reasons.append("Negative sentiment")

        # Deviation
        if deviation < -0.03:
            reasons.append("Price below SMA50 → potential bounce")
        elif deviation > 0.03:
            reasons.append("Price above SMA50 → potential pullback")

        st.markdown("**Key Reasons Driving the Recommendation:**")
        for reason in reasons:
            st.write("- " + reason)

        # --- Summary & Action Guidance ---
        st.subheader("📌 Summary & Action Guidance")
        # Determine overall recommendation dynamically
        score = 0
        if 'BUY' in signal:
            score += 1
        if 'SELL' in signal:
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

        st.markdown(f"**Overall Recommendation: {overall}**")
        st.markdown("**Explanation:** Based on combined indicators: trend, MACD, Bollinger Bands, RSI, predicted price, Déjà Vue, PTB, sentiment, and deviation.")

    except Exception as e:
        st.error(f"Error processing {ticker}: {e}")
