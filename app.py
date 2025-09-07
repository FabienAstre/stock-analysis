import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, MACD, ADXIndicator
from ta.volatility import BollingerBands
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from prophet import Prophet
import plotly.graph_objects as go

# --- Page config ---
st.set_page_config(page_title="📊 Advanced Stock Analysis & AI Prediction App", layout="wide")
st.title("📊 Advanced Stock Analysis & AI Prediction App")

# --- Sidebar ---
st.sidebar.header("Settings")
tickers_input = st.sidebar.text_input("Enter Stock Tickers (comma-separated)", "MSFT,AAPL")
period = st.sidebar.selectbox("Select Historical Period", ["1mo", "3mo", "6mo", "1y", "5y"])
tickers = [t.strip().upper() for t in tickers_input.split(",")]

st.sidebar.header("Chart Options")
show_sma = st.sidebar.checkbox("Show SMA50 & SMA200", value=True)
show_bb = st.sidebar.checkbox("Show Bollinger Bands", value=True)
show_macd = st.sidebar.checkbox("Show MACD", value=True)
show_deja = st.sidebar.checkbox("Show Déjà Vue Patterns", value=True)

# --- Loop through tickers ---
for ticker in tickers:
    st.header(f"{ticker} Analysis")

    # --- Fetch historical data ---
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

    # --- Indicators ---
    hist['SMA50'] = SMAIndicator(hist['Close'], 50).sma_indicator()
    hist['SMA200'] = SMAIndicator(hist['Close'], 200).sma_indicator()
    hist['SMA20'] = SMAIndicator(hist['Close'], 20).sma_indicator()
    bb = BollingerBands(hist['Close'], 20, 2)
    hist['BB_upper'] = bb.bollinger_hband()
    hist['BB_lower'] = bb.bollinger_lband()
    macd_indicator = MACD(hist['Close'])
    hist['MACD'] = macd_indicator.macd()
    hist['MACD_signal'] = macd_indicator.macd_signal()
    hist['MACD_cross'] = np.where(hist['MACD'] > hist['MACD_signal'], 'bullish', 'bearish')
    rsi_value = RSIIndicator(hist['Close'], 14).rsi().iloc[-1]
    adx_indicator = ADXIndicator(hist['High'], hist['Low'], hist['Close'], 14)
    hist['ADX'] = adx_indicator.adx()

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

    # --- Advanced Prediction & Anomaly Signals ---
    st.subheader("Advanced Prediction Signals")
    signal = "HOLD ⏸️"
    predicted_price = None
    lookback = 60
    is_anomaly = False
    conf_interval = None

    close_series = hist['Close'][-lookback:].dropna()
    if len(close_series) >= lookback:
        current_price = hist['Close'].iloc[-1]

        # ARIMA
        arima_pred = None
        try:
            model = ARIMA(close_series, order=(5,1,0))
            model_fit = model.fit()
            arima_pred = model_fit.forecast(steps=1)[0]
        except:
            pass

        # Linear Regression
        lr_pred = None
        try:
            X = np.arange(len(close_series)).reshape(-1,1)
            y = close_series.values
            lr_model = LinearRegression()
            lr_model.fit(X, y)
            lr_pred = lr_model.predict(np.array([[len(close_series)]]))[0]
        except:
            pass

        # Prophet
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

        # Ensemble
        preds = [p for p in [arima_pred, lr_pred, prophet_pred] if p is not None]
        if preds:
            predicted_price = np.mean(preds)

        if predicted_price:
            if predicted_price > current_price*1.002: signal = "BUY ✅"
            elif predicted_price < current_price*0.998: signal = "SELL ❌"

            st.write(f"Predicted next-day close: **${predicted_price:.2f}**")
            if conf_interval:
                st.write(f"(95% CI: ${conf_interval[0]:.2f} – ${conf_interval[1]:.2f})")
            st.write(f"Signal: {signal}")

        # Z-score anomaly
        hist['Returns'] = hist['Close'].pct_change()
        hist['ZScore'] = (hist['Returns'] - hist['Returns'].mean()) / hist['Returns'].std()
        is_anomaly = abs(hist['ZScore'].iloc[-1]) > 2
        if is_anomaly:
            st.write("⚡ Anomaly detected: unusual price movement")

        st.markdown("**Explanation:** Combines ARIMA, Linear Regression, Prophet, and anomaly detection for robust prediction.")

    else:
        st.warning("Not enough data for prediction (requires at least 60 non-NaN closing prices).")

      
    # --- Déjà Vue Signals ---
    st.subheader("🔁 Déjà Vue Trading Signals")
    if matches:
        deja_results = []
        for i, date, similarity in matches:
            if i + pattern_length + 5 < len(hist):
                future_prices = hist['Close'].iloc[i+pattern_length:i+pattern_length+5]
                trend_return = (future_prices.iloc[-1]-future_prices.iloc[0])/future_prices.iloc[0]
                if trend_return > 0.02: trend_val = "Uptrend 📈"
                elif trend_return < -0.02: trend_val = "Downtrend 📉"
                else: trend_val = "Flat ➖"
            else:
                trend_val = "N/A"
            deja_results.append({"Date": date.date(), "Similarity": f"{similarity:.2%}", "Trend After Pattern": trend_val})
        deja_df = pd.DataFrame(deja_results)
        st.dataframe(deja_df)
    else:
        st.write("No similar historical patterns found.")
    st.markdown("**Explanation:** Shows repeating patterns, similarity, and subsequent trend direction.")

 # --- Trending & Mean-Reversion Signals ---
st.subheader("📊 Trending & Mean-Reversion Signals")

# --- Calculate indicators ---
hist['EMA10'] = EMAIndicator(hist['Close'], 10).ema_indicator()
hist['ADX'] = ADXIndicator(hist['High'], hist['Low'], hist['Close'], 14).adx()
hist['Volume_SMA20'] = hist['Volume'].rolling(20).mean()

# --- Determine trend ---
trend_signal = "Uptrend" if hist['EMA10'].iloc[-1] > hist['SMA50'].iloc[-1] else "Downtrend"

# --- RSI ---
rsi = RSIIndicator(hist['Close'], 14).rsi().iloc[-1]

# --- ADX ---
adx = hist['ADX'].iloc[-1]

# --- Volume trend ---
volume_trend = hist['Volume_SMA20'].iloc[-1]

# --- Display values ---
st.write(f"Latest Trend Signal: {trend_signal}")
st.write(f"RSI: {rsi:.2f} | ADX: {adx:.2f} | Volume Trend (SMA20): {volume_trend:.0f}")

# --- Explanation ---
st.markdown("""
**Explanation:**  
- **EMA10 vs SMA50:** Short-term EMA above SMA50 indicates short-term bullish momentum; below indicates bearish.  
- **RSI:** Measures overbought (>70) or oversold (<30) conditions.  
- **ADX:** Measures trend strength; >25 indicates strong trend, <20 weak trend.  
- **Volume Trend (SMA20):** Rising volume confirms market participation supporting the trend.
""")

# --- Plot improved chart ---
import plotly.graph_objects as go

fig_trend = go.Figure()

# Price candlestick
fig_trend.add_trace(go.Candlestick(
    x=hist.index,
    open=hist['Open'],
    high=hist['High'],
    low=hist['Low'],
    close=hist['Close'],
    name='Candlestick'
))

# SMA50, SMA200
fig_trend.add_trace(go.Scatter(x=hist.index, y=hist['SMA50'], line=dict(color='blue', width=1), name='SMA50'))
fig_trend.add_trace(go.Scatter(x=hist.index, y=hist['SMA200'], line=dict(color='red', width=1), name='SMA200'))

# EMA10
fig_trend.add_trace(go.Scatter(x=hist.index, y=hist['EMA10'], line=dict(color='green', width=1, dash='dot'), name='EMA10'))

# Volume as bar chart
fig_trend.add_trace(go.Bar(
    x=hist.index,
    y=hist['Volume'],
    marker_color='lightgrey',
    name='Volume',
    yaxis='y2'
))

# Layout with secondary y-axis for volume
fig_trend.update_layout(
    height=500,
    xaxis_title='Date',
    yaxis_title='Price',
    yaxis2=dict(
        overlaying='y',
        side='right',
        title='Volume',
        showgrid=False
    ),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig_trend, use_container_width=True)


    # --- Price to Tangible Book ---
    st.subheader("💰 Price to Tangible Book (PTB)")
    try:
        ptb = stock.info.get('priceToBook', None)
    except:
        ptb = None
    st.write(f"Price to Tangible Book: {ptb if ptb else 'N/A'}")
    st.markdown("**Explanation:** PTB <1 may indicate undervalued relative to tangible assets.")

    # --- Key Reasons & Overall Recommendation ---
    reasons = [f"Trend: {trend_signal}"]
    if rsi_value < 30: reasons.append("RSI oversold → potential buy")
    elif rsi_value > 70: reasons.append("RSI overbought → potential sell")
    if hist['MACD_cross'].iloc[-1] == 'bullish': reasons.append("MACD bullish crossover")
    else: reasons.append("MACD bearish crossover")
    if hist['Close'].iloc[-1] < hist['BB_lower'].iloc[-1]: reasons.append("Price below lower Bollinger → potential buy")
    elif hist['Close'].iloc[-1] > hist['BB_upper'].iloc[-1]: reasons.append("Price above upper Bollinger → potential sell")
    if matches: reasons.append("Déjà Vue pattern found")
    if ptb:
        if ptb < 1: reasons.append("PTB <1 → undervalued")
        elif ptb > 2: reasons.append("PTB >2 → overvalued")
    if deviation < -0.03: reasons.append("Price below SMA50 → potential bounce")
    elif deviation > 0.03: reasons.append("Price above SMA50 → potential pullback")
    if predicted_price: reasons.append(f"Prediction signal: {signal}")

    st.subheader("📌 Key Reasons")
    for r in reasons:
        st.write("- " + r)

    # --- Overall Recommendation (Weighted AI-like) ---
    weights = {"prediction":0.5, "rsi":0.2, "macd":0.1, "bollinger":0.1}
    score = 0
    if signal == "BUY ✅": score += weights["prediction"]
    elif signal == "SELL ❌": score -= weights["prediction"]
    if rsi_value < 30: score += weights["rsi"]
    elif rsi_value > 70: score -= weights["rsi"]
    score += weights["macd"] if hist['MACD_cross'].iloc[-1]=='bullish' else -weights["macd"]
    if deviation < -0.03: score += weights["bollinger"]
    elif deviation > 0.03: score -= weights["bollinger"]

    if score > 0: overall = "BUY ✅"
    elif score < 0: overall = "SELL ❌"
    else: overall = "HOLD ⏸️"

    st.subheader("📌 Overall Recommendation")
    st.markdown(f"**{overall}**")
    st.markdown("This concludes the analysis for this stock.")
