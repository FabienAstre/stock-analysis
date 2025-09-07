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
from plotly.subplots import make_subplots

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
show_ema = st.sidebar.checkbox("Show EMA10", value=True)
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
    hist['EMA10'] = EMAIndicator(hist['Close'], 10).ema_indicator()
    bb = BollingerBands(hist['Close'], 20, 2)
    hist['BB_upper'] = bb.bollinger_hband()
    hist['BB_lower'] = bb.bollinger_lband()
    macd = MACD(hist['Close'])
    hist['MACD'] = macd.macd()
    hist['MACD_signal'] = macd.macd_signal()
    hist['MACD_cross'] = np.where(hist['MACD'] > hist['MACD_signal'], 'bullish','bearish')
    rsi = RSIIndicator(hist['Close'],14).rsi().iloc[-1]
    adx = ADXIndicator(hist['High'], hist['Low'], hist['Close'], 14).adx().iloc[-1]
    hist['Volume_SMA20'] = hist['Volume'].rolling(20).mean()

    # --- Déjà Vue Patterns ---
    pattern_len = 5
    last_pattern = hist['Close'].values[-pattern_len:]
    matches = []
    for i in range(len(hist)-pattern_len-1):
        pattern = hist['Close'].values[i:i+pattern_len]
        corr = np.corrcoef(last_pattern, pattern)[0,1]
        if corr>0.95:
            matches.append((i,hist.index[i],corr))

    # --- Prediction & Anomaly ---
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

        # Z-score anomaly
        hist['Returns'] = hist['Close'].pct_change()
        hist['ZScore'] = (hist['Returns'] - hist['Returns'].mean()) / hist['Returns'].std()
        is_anomaly = abs(hist['ZScore'].iloc[-1]) > 2

    # --- Plot chart ---
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.05,
        specs=[[{"secondary_y": True}], [{}]]
    )

    # Candlestick + Indicators on row 1
    fig.add_trace(go.Candlestick(
        x=hist.index,
        open=hist['Open'],
        high=hist['High'],
        low=hist['Low'],
        close=hist['Close'],
        name='Price',
        increasing_line_color='green',
        decreasing_line_color='red'
    ), row=1, col=1)

    # SMA & EMA
    fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA50'], line=dict(color='blue', width=2), name='SMA50'), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA200'], line=dict(color='red', width=2), name='SMA200'), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=hist['EMA10'], line=dict(color='green', width=2, dash='dot'), name='EMA10'), row=1, col=1)

    # Bollinger Bands
    fig.add_trace(go.Scatter(x=hist.index, y=hist['BB_upper'], line=dict(color='orange', width=1, dash='dash'), name='BB Upper'), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=hist['BB_lower'], line=dict(color='orange', width=1, dash='dash'), name='BB Lower'), row=1, col=1)

    # Déjà Vue patterns
    if show_deja:
        for match in matches:
            fig.add_vline(x=match[1], line_width=1, line_dash="dot", line_color="purple")

    # Anomalies
    if is_anomaly:
        fig.add_trace(go.Scatter(
            x=[hist.index[-1]],
            y=[hist['Close'].iloc[-1]],
            mode='markers',
            marker=dict(color='red', size=10, symbol='diamond'),
            name='Anomaly'
        ), row=1, col=1)

    # Volume bars on secondary y-axis
    fig.add_trace(go.Bar(
        x=hist.index,
        y=hist['Volume'],
        name='Volume',
        marker_color='lightgrey',
        opacity=0.4
    ), row=1, col=1, secondary_y=True)

    # MACD subplot on row 2
    fig.add_trace(go.Scatter(x=hist.index, y=hist['MACD'], line=dict(color='green', width=2), name='MACD'), row=2, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=hist['MACD_signal'], line=dict(color='black', width=2, dash='dot'), name='MACD Signal'), row=2, col=1)

    fig.update_layout(
        height=700,
        title=f"{ticker} Price Chart + Indicators + MACD",
        xaxis_title="Date",
        yaxis_title="Price",
        yaxis2=dict(
            overlaying='y',
            side='right',
            title='Volume',
            showgrid=False
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)

    # --- Trending & Mean-Reversion Signals ---
    st.subheader("📊 Trending & Mean-Reversion Signals")
    trend_signal = "Uptrend" if hist['EMA10'].iloc[-1] > hist['SMA50'].iloc[-1] else "Downtrend"
    volume_trend = hist['Volume_SMA20'].iloc[-1]
    st.write(f"Latest Trend Signal: {trend_signal}")
    st.write(f"RSI: {rsi:.2f} | ADX: {adx:.2f} | Volume Trend (SMA20): {volume_trend:.0f}")
    st.markdown("""
    **Explanation:**  
    - **EMA10 vs SMA50:** Short-term EMA above SMA50 indicates bullish momentum; below indicates bearish.  
    - **RSI:** Measures overbought (>70) or oversold (<30).  
    - **ADX:** >25 strong trend, <20 weak trend.  
    - **Volume Trend (SMA20):** Rising volume confirms market participation.
    """)

    # --- Price to Tangible Book ---
    st.subheader("💰 Price to Tangible Book (PTB)")
    ptb = stock.info.get('priceToBook', 'N/A')
    st.write(f"Price to Tangible Book: {ptb}")
    st.markdown("**Explanation:** PTB <1 may indicate undervalued relative to tangible assets.")

    # --- Key Reasons & Recommendation ---
    reasons = [f"Trend: {trend_signal}"]
    if rsi < 30: reasons.append("RSI oversold → potential buy")
    elif rsi > 70: reasons.append("RSI overbought → potential sell")
    if hist['MACD_cross'].iloc[-1] == 'bullish': reasons.append("MACD bullish crossover")
    else: reasons.append("MACD bearish crossover")
    if hist['Close'].iloc[-1] < hist['BB_lower'].iloc[-1]: reasons.append("Price below lower Bollinger → potential buy")
    elif hist['Close'].iloc[-1] > hist['BB_upper'].iloc[-1]: reasons.append("Price above upper Bollinger → potential sell")
    if matches: reasons.append("Déjà Vue pattern found")
    if ptb != 'N/A':
        if ptb < 1: reasons.append("PTB <1 → undervalued")
        elif ptb > 2: reasons.append("PTB >2 → overvalued")
    if predicted_price: reasons.append(f"Prediction signal: {signal}")

    st.subheader("📌 Key Reasons")
    for r in reasons:
        st.write("- " + r)

    st.subheader("📌 Overall Recommendation")
    st.markdown(f"**{signal}**")
    st.markdown("This concludes the analysis for this stock.")
