import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
from ta.volatility import BollingerBands
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from prophet import Prophet
import plotly.graph_objects as go

# --- Page config ---
st.set_page_config(page_title="📊 Advanced Stock Analysis & Prediction App", layout="wide")
st.title("📊 Advanced Stock Analysis & Prediction App")

# --- Sidebar ---
st.sidebar.header("Settings")
tickers_input = st.sidebar.text_input("Enter Stock Tickers (comma-separated)", "MSFT,AAPL")
period = st.sidebar.selectbox("Select Historical Period", ["1mo", "3mo", "6mo", "1y", "5y"])
tickers = [t.strip().upper() for t in tickers_input.split(",")]

st.sidebar.header("Chart Options")
show_sma = st.sidebar.checkbox("Show SMA50 & SMA200", value=True)
show_ema = st.sidebar.checkbox("Show EMA20 & EMA50", value=True)
show_bb = st.sidebar.checkbox("Show Bollinger Bands", value=True)
show_macd = st.sidebar.checkbox("Show MACD", value=True)
show_adx = st.sidebar.checkbox("Show ADX", value=True)
show_deja = st.sidebar.checkbox("Show Déjà Vue Patterns", value=True)

# --- Loop through tickers ---
for ticker in tickers:
    st.header(f"{ticker} Analysis")

    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        if hist.empty:
            st.warning("No historical data available.")
            continue
    except Exception as e:
        st.warning(f"Error fetching {ticker}: {e}")
        continue

    # --- Latest Price ---
    latest_price = hist['Close'].iloc[-1]
    st.subheader("Latest Closing Price")
    st.metric(label="Price", value=f"${latest_price:.2f}")
    st.markdown("**Explanation:** Shows the most recent market closing price for quick reference.")

    # --- Indicators ---
    hist['SMA50'] = SMAIndicator(hist['Close'], 50).sma_indicator()
    hist['SMA200'] = SMAIndicator(hist['Close'], 200).sma_indicator()
    hist['EMA20'] = EMAIndicator(hist['Close'], 20).ema_indicator()
    hist['EMA50'] = EMAIndicator(hist['Close'], 50).ema_indicator()
    bb = BollingerBands(hist['Close'], 20, 2)
    hist['BB_upper'] = bb.bollinger_hband()
    hist['BB_lower'] = bb.bollinger_lband()
    macd_indicator = MACD(hist['Close'])
    hist['MACD'] = macd_indicator.macd()
    hist['MACD_signal'] = macd_indicator.macd_signal()
    hist['MACD_cross'] = np.where(hist['MACD'] > hist['MACD_signal'], 'bullish', 'bearish')
    rsi = RSIIndicator(hist['Close'], 14).rsi().iloc[-1]
    adx = ADXIndicator(hist['High'], hist['Low'], hist['Close'], 14).adx().iloc[-1]
    hist['VolumeTrend'] = hist['Volume'].rolling(5).mean()  # Simple volume trend

    # --- Déjà Vue Patterns ---
    pattern_length = 5
    last_pattern = hist['Close'].values[-pattern_length:]
    matches = []
    for i in range(len(hist) - pattern_length - 1):
        hist_pattern = hist['Close'].values[i:i+pattern_length]
        similarity = np.corrcoef(last_pattern, hist_pattern)[0,1]
        if similarity > 0.95:
            matches.append((i, hist.index[i], similarity))

    # --- Candlestick Chart + Indicators ---
    st.subheader("Candlestick Chart + Indicators")
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
    if show_ema:
        fig.add_trace(go.Scatter(x=hist.index, y=hist['EMA20'], line=dict(color='green', dash='dot'), name='EMA20'))
        fig.add_trace(go.Scatter(x=hist.index, y=hist['EMA50'], line=dict(color='purple', dash='dot'), name='EMA50'))
    if show_bb:
        fig.add_trace(go.Scatter(x=hist.index, y=hist['BB_upper'], line=dict(color='orange'), name='BB Upper'))
        fig.add_trace(go.Scatter(x=hist.index, y=hist['BB_lower'], line=dict(color='orange'), name='BB Lower'))
    if show_macd:
        fig.add_trace(go.Scatter(x=hist.index, y=hist['MACD'], line=dict(color='green'), name='MACD'))
        fig.add_trace(go.Scatter(x=hist.index, y=hist['MACD_signal'], line=dict(color='black'), name='MACD Signal'))
    if show_deja and matches:
        for match in matches:
            fig.add_vline(x=match[1], line_width=1, line_dash="dash", line_color="purple")
    fig.update_layout(height=500, xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Explanation:** Candlestick chart shows price action. SMAs, EMAs, MACD, and Bollinger Bands help identify trends and volatility. Purple lines indicate repeating historical patterns (Déjà Vue).")

    # --- Advanced Prediction & Anomaly Signals ---
    st.subheader("Advanced Prediction Signals")
    signal = "HOLD ⏸️"
    predicted_price = None
    lookback = 60
    is_anomaly = False
    conf_interval = None

    try:
        close_series = hist['Close'][-lookback:].dropna()
        if len(close_series) >= lookback:
            current_price = hist['Close'].iloc[-1]

            # --- ARIMA ---
            arima_pred = None
            try:
                model = ARIMA(close_series, order=(5,1,0))
                model_fit = model.fit()
                forecast = model_fit.forecast(steps=1)
                arima_pred = forecast[0]
            except:
                pass

            # --- Linear Regression ---
            lr_pred = None
            try:
                X = np.arange(len(close_series)).reshape(-1,1)
                y = close_series.values
                lr_model = LinearRegression()
                lr_model.fit(X, y)
                lr_pred = lr_model.predict(np.array([[len(close_series)]]))[0]
            except:
                pass

            # --- Prophet Forecast ---
            prophet_pred = None
            try:
                df_prophet = close_series.reset_index()
                df_prophet.columns = ['ds','y']
                prophet = Prophet(daily_seasonality=True, interval_width=0.95)
                prophet.fit(df_prophet)
                future = prophet.make_future_dataframe(periods=1)
                forecast = prophet.predict(future)
                prophet_pred = forecast['yhat'].iloc[-1]
                conf_interval = (forecast['yhat_lower'].iloc[-1], forecast['yhat_upper'].iloc[-1])
            except:
                pass

            # --- Combine Predictions (Ensemble) ---
            preds = [p for p in [arima_pred, lr_pred, prophet_pred] if p is not None]
            if preds:
                predicted_price = np.mean(preds)

            # --- Generate Signal ---
            if predicted_price:
                if predicted_price > current_price*1.002:
                    signal = "BUY ✅"
                elif predicted_price < current_price*0.998:
                    signal = "SELL ❌"

                st.write(f"Predicted next-day close: **${predicted_price:.2f}**")
                if conf_interval:
                    st.write(f"(95% CI: ${conf_interval[0]:.2f} – ${conf_interval[1]:.2f})")
                st.write(f"Signal: {signal}")

            # --- Z-score anomaly ---
            hist['Returns'] = hist['Close'].pct_change()
            hist['ZScore'] = (hist['Returns'] - hist['Returns'].mean()) / hist['Returns'].std()
            is_anomaly = abs(hist['ZScore'].iloc[-1]) > 2
            if is_anomaly:
                st.write("⚡ Anomaly detected: unusual price movement")

            st.markdown("**Explanation:** Uses an ensemble of ARIMA, Linear Regression, and Prophet forecasts. Detects anomalies with Z-score analysis.")

        else:
            st.warning("Not enough valid data for prediction (requires at least 60 non-NaN closing prices).")

    except Exception as e:
        st.warning(f"Prediction module encountered an error but other modules will continue: {e}")

    # --- Déjà Vue Signals ---
    st.subheader("🔁 Déjà Vue Trading Signals")
    if matches:
        deja_results = []
        for i, date, similarity in matches:
            if i + pattern_length + 5 < len(hist):
                future_prices = hist['Close'].iloc[i+pattern_length:i+pattern_length+5]
                trend_return = (future_prices.iloc[-1]-future_prices.iloc[0])/future_prices.iloc[0]
                if trend_return > 0.02: trend = "Uptrend 📈"
                elif trend_return < -0.02: trend = "Downtrend 📉"
                else: trend = "Flat ➖"
            else:
                trend = "N/A"
            deja_results.append({"Date": date.date(), "Similarity": f"{similarity:.2%}", "Trend After Pattern": trend})
        deja_df = pd.DataFrame(deja_results)
        st.dataframe(deja_df)
    else:
        st.write("No similar historical patterns found.")
    st.markdown("**Explanation:** Shows historical patterns that repeat (Déjà Vue) and the trend that followed them.")

    # --- Trending & Mean-Reversion ---
    st.subheader("📊 Trending & Mean-Reversion Signals")
    trend = "Uptrend" if hist['SMA50'].iloc[-1] > hist['SMA200'].iloc[-1] else "Downtrend"
    deviation = (hist['Close'].iloc[-1]-hist['SMA50'].iloc[-1])/hist['SMA50'].iloc[-1]
    st.write(f"Latest Trend Signal: {trend}")
    st.write(f"RSI: {rsi:.2f} | ADX: {adx:.2f} | Volume Trend: {hist['VolumeTrend'].iloc[-1]:.0f}")
    st.markdown("**Explanation:** SMA cross identifies long-term trend. RSI identifies overbought/oversold. ADX measures trend strength. Volume trend confirms market participation.")

    # --- Key Reasons & Overall Recommendation ---
    reasons = []
    reasons.append(f"Trend: {trend}")
    if rsi < 30: reasons.append("RSI oversold → potential buy")
    elif rsi > 70: reasons.append("RSI overbought → potential sell")
    if hist['MACD_cross'].iloc[-1] == 'bullish': reasons.append("MACD bullish crossover")
    else: reasons.append("MACD bearish crossover")
    if hist['Close'].iloc[-1] < hist['BB_lower'].iloc[-1]:
        reasons.append("Price below lower Bollinger → potential buy")
    elif hist['Close'].iloc[-1] > hist['BB_upper'].iloc[-1]:
        reasons.append("Price above upper Bollinger → potential sell")
    if matches: reasons.append("Déjà Vue pattern found")
    if deviation < -0.03: reasons.append("Price below SMA50 → potential bounce")
    elif deviation > 0.03: reasons.append("Price above SMA50 → potential pullback")
    if predicted_price: reasons.append(f"Prediction signal: {signal}")

    st.subheader("📌 Key Reasons")
    for r in reasons:
        st.write("- " + r)

    # --- Overall Recommendation ---
    weights = {
        "prediction": 0.5,
        "rsi": 0.2,
        "macd": 0.1,
        "bollinger": 0.1,
        "adx": 0.1
    }
    score = 0
    if signal == "BUY ✅": score += weights["prediction"]
    elif signal == "SELL ❌": score -= weights["prediction"]

    if rsi < 30: score += weights["rsi"]
    elif rsi > 70: score -= weights["rsi"]

    score += weights["macd"] if hist['MACD_cross'].iloc[-1]=='bullish' else -weights["macd"]
    if deviation < -0.03: score += weights["bollinger"]
    elif deviation > 0.03: score -= weights["bollinger"]
    score += weights["adx"] if adx > 20 else -weights["adx"]

    if score > 0: overall = "BUY ✅"
    elif score < 0: overall = "SELL ❌"
    else: overall = "HOLD ⏸️"

    st.subheader("📌 Overall Recommendation")
    st.markdown(f"**{overall}**")
    st.markdown("**Explanation:** Combines prediction ensemble, technical indicators, trend strength, and deviation from SMA50 to give a reliable trading signal.")
