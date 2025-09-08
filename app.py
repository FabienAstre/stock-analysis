import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
from ta.volatility import BollingerBands
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression

# Prophet may not be available in all environments; import lazily
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Page config ---
st.set_page_config(page_title="📊 Advanced Stock Analysis & AI Prediction App (Lovable)", layout="wide")
st.title("📊 Advanced Stock Analysis & AI Prediction App — Lovable Edition")

# --- Sidebar ---
st.sidebar.header("Settings")
tickers_input = st.sidebar.text_input("Enter Stock Tickers (comma-separated)", "MSFT,AAPL")
period = st.sidebar.selectbox("Select Historical Period", ["1mo", "3mo", "6mo", "1y", "5y"], index=3)
user_sma_short = st.sidebar.number_input("SMA short period", min_value=5, max_value=200, value=50)
user_sma_long = st.sidebar.number_input("SMA long period", min_value=20, max_value=400, value=200)
user_ema = st.sidebar.number_input("EMA period", min_value=5, max_value=200, value=10)
show_deja = st.sidebar.checkbox("Show Déjà Vue Patterns", value=True)
use_prophet = st.sidebar.checkbox("Use Prophet (if available)", value=PROPHET_AVAILABLE and True)

tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

# --- Caching data fetch ---
@st.cache_data
def get_history(ticker: str, period: str) -> pd.DataFrame:
    ticker_obj = yf.Ticker(ticker)
    hist = ticker_obj.history(period=period)
    hist.index = pd.to_datetime(hist.index)
    return hist

# indicator computation
def compute_indicators(hist: pd.DataFrame):
    h = hist.copy()
    try:
        h[f'SMA{user_sma_short}'] = SMAIndicator(h['Close'], user_sma_short).sma_indicator()
    except Exception:
        h[f'SMA{user_sma_short}'] = np.nan
    try:
        h[f'SMA{user_sma_long}'] = SMAIndicator(h['Close'], user_sma_long).sma_indicator()
    except Exception:
        h[f'SMA{user_sma_long}'] = np.nan
    try:
        h[f'EMA{user_ema}'] = EMAIndicator(h['Close'], user_ema).ema_indicator()
    except Exception:
        h[f'EMA{user_ema}'] = np.nan
    try:
        bb = BollingerBands(h['Close'], 20, 2)
        h['BB_upper'] = bb.bollinger_hband()
        h['BB_lower'] = bb.bollinger_lband()
    except Exception:
        h['BB_upper'] = np.nan
        h['BB_lower'] = np.nan
    try:
        macd = MACD(h['Close'])
        h['MACD'] = macd.macd()
        h['MACD_signal'] = macd.macd_signal()
        h['MACD_cross'] = np.where(h['MACD'] > h['MACD_signal'], 'bullish', 'bearish')
    except Exception:
        h['MACD'] = np.nan
        h['MACD_signal'] = np.nan
        h['MACD_cross'] = 'unknown'
    try:
        h['RSI'] = RSIIndicator(h['Close'], 14).rsi()
    except Exception:
        h['RSI'] = np.nan
    try:
        h['ADX'] = ADXIndicator(h['High'], h['Low'], h['Close'], 14).adx()
    except Exception:
        h['ADX'] = np.nan
    h['Volume_SMA20'] = h['Volume'].rolling(20).mean()
    return h

# Déjà Vue detection
def deja_vue_matches(h: pd.DataFrame, pattern_len: int = 5, corr_thresh: float = 0.95):
    matches = []
    if len(h) <= pattern_len * 2:
        return matches
    last_pattern = h['Close'].values[-pattern_len:]
    for i in range(len(h)-pattern_len-1):
        pattern = h['Close'].values[i:i+pattern_len]
        if np.std(pattern) == 0 or np.std(last_pattern) == 0:
            continue
        corr = np.corrcoef(last_pattern, pattern)[0,1]
        if corr > corr_thresh:
            matches.append((i, h.index[i], corr))
    return matches

# Prediction ensemble
def predict_price_ensemble(h: pd.DataFrame, lookback: int = 60, use_prophet_flag: bool = False):
    result = {'predicted_price': None, 'signal': 'HOLD ⏸️', 'conf_interval': None}
    close_series = h['Close'].dropna()
    if len(close_series) < lookback:
        return result
    window = close_series[-lookback:]
    current_price = window.iloc[-1]
    preds = []
    try:
        arima_model = ARIMA(window, order=(5,1,0)).fit()
        arima_pred = arima_model.forecast(steps=1)[0]
        preds.append(arima_pred)
    except Exception:
        pass
    try:
        X = np.arange(len(window)).reshape(-1,1)
        lr = LinearRegression().fit(X, window.values)
        lr_pred = lr.predict(np.array([[len(window)]]))[0]
        preds.append(lr_pred)
    except Exception:
        pass
    if use_prophet_flag and PROPHET_AVAILABLE:
        try:
            dfp = window.reset_index()
            dfp.columns = ['ds','y']
            m = Prophet(daily_seasonality=True, interval_width=0.90)
            m.fit(dfp)
            future = m.make_future_dataframe(periods=1)
            fc = m.predict(future)
            prophet_pred = fc['yhat'].iloc[-1]
            preds.append(prophet_pred)
            result['conf_interval'] = (fc['yhat_lower'].iloc[-1], fc['yhat_upper'].iloc[-1])
        except Exception:
            pass
    if preds:
        predicted_price = float(np.mean(preds))
        result['predicted_price'] = predicted_price
        if predicted_price > current_price * 1.002:
            result['signal'] = 'BUY ✅'
        elif predicted_price < current_price * 0.998:
            result['signal'] = 'SELL ❌'
    return result

# ----- Main loop -----
if not tickers:
    st.warning("Please enter at least one ticker symbol.")

for ticker in tickers:
    with st.spinner(f"Fetching {ticker} data..."):
        hist = get_history(ticker, period)
    st.header(f"{ticker} Analysis")
    if hist.empty:
        st.warning(f"No historical data available for {ticker}.")
        continue

    hist = compute_indicators(hist)

    latest_price = hist['Close'].iloc[-1]
    st.subheader("Latest Closing Price")
    st.metric(label="Price", value=f"${latest_price:.2f}")

    matches = deja_vue_matches(hist)
    pred = predict_price_ensemble(hist, lookback=60, use_prophet_flag=use_prophet)

    hist['Returns'] = hist['Close'].pct_change()
    hist['ZScore'] = (hist['Returns'] - hist['Returns'].mean()) / (hist['Returns'].std() if hist['Returns'].std() != 0 else 1)
    is_anomaly = abs(hist['ZScore'].iloc[-1]) > 2

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.55, 0.15, 0.2], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name='Price'), row=1, col=1)
    try:
        fig.add_trace(go.Scatter(x=hist.index, y=hist[f'SMA{user_sma_short}'], name=f'SMA{user_sma_short}'), row=1, col=1)
        fig.add_trace(go.Scatter(x=hist.index, y=hist[f'SMA{user_sma_long}'], name=f'SMA{user_sma_long}'), row=1, col=1)
        fig.add_trace(go.Scatter(x=hist.index, y=hist[f'EMA{user_ema}'], name=f'EMA{user_ema}', line=dict(dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=hist.index, y=hist['BB_upper'], name='BB Upper', line=dict(dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=hist.index, y=hist['BB_lower'], name='BB Lower', line=dict(dash='dash')), row=1, col=1)
    except Exception:
        pass
    if show_deja and matches:
        for match in matches:
            fig.add_vline(x=match[1], line_width=1, line_dash="dot", line_color="purple")
    if is_anomaly:
        fig.add_trace(go.Scatter(x=[hist.index[-1]], y=[hist['Close'].iloc[-1]], mode='markers', marker=dict(size=10, symbol='diamond'), name='Anomaly'), row=1, col=1)
    fig.add_trace(go.Bar(x=hist.index, y=hist['Volume'], name='Volume', opacity=0.5), row=2, col=1)
    try:
        fig.add_trace(go.Scatter(x=hist.index, y=hist['Volume_SMA20'], name='Vol SMA20'), row=2, col=1)
    except Exception:
        pass
    try:
        fig.add_trace(go.Scatter(x=hist.index, y=hist['MACD'], name='MACD'), row=3, col=1)
        fig.add_trace(go.Scatter(x=hist.index, y=hist['MACD_signal'], name='MACD Signal', line=dict(dash='dot')), row=3, col=1)
    except Exception:
        pass
    fig.update_layout(height=800, title_text=f"{ticker} Price Chart + Indicators")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Trending & Mean-Reversion Signals")
    try:
        trend_signal = "Uptrend" if hist[f'EMA{user_ema}'].iloc[-1] > hist[f'SMA{user_sma_short}'].iloc[-1] else "Downtrend"
    except Exception:
        trend_signal = "Unknown"
    rsi_val = float(hist['RSI'].iloc[-1]) if not pd.isna(hist['RSI'].iloc[-1]) else None
    adx_val = float(hist['ADX'].iloc[-1]) if not pd.isna(hist['ADX'].iloc[-1]) else None
    vol_trend = float(hist['Volume_SMA20'].iloc[-1]) if not pd.isna(hist['Volume_SMA20'].iloc[-1]) else None

    st.write(f"Latest Trend Signal: {trend_signal}")
    st.write(f"RSI: {rsi_val if rsi_val is not None else 'N/A'} | ADX: {adx_val if adx_val is not None else 'N/A'} | Volume Trend (SMA20): {vol_trend if vol_trend is not None else 'N/A'}")
    st.markdown("""
    **Explanation:**  
    - **EMA vs SMA:** Short-term EMA above SMA indicates bullish momentum; below indicates bearish.  
    - **RSI:** Measures overbought (>70) or oversold (<30).  
    - **ADX:** >25 strong trend, <20 weak trend.  
    - **Volume Trend (SMA20):** Rising volume confirms market participation.
    """)

    st.subheader("💰 Price to Tangible Book (PTB)")
    try:
        t = yf.Ticker(ticker)
        info = t.fast_info if hasattr(t, 'fast_info') else {}
        ptb = info.get('priceToBook') or 'N/A'
    except Exception:
        ptb = 'N/A'
    st.write(f"Price to Tangible Book: {ptb}")
    st.markdown("**Explanation:** PTB <1 may indicate undervalued relative to tangible assets.")

    reasons = [f"Trend: {trend_signal}"]
    if rsi_val is not None:
        if rsi_val < 30:
            reasons.append("RSI oversold → potential buy")
        elif rsi_val > 70:
            reasons.append("RSI overbought → potential sell")
    try:
        macd_cross = hist['MACD_cross'].iloc[-1]
        if macd_cross == 'bullish':
            reasons.append("MACD bullish crossover")
        elif macd_cross == 'bearish':
            reasons.append("MACD bearish crossover")
    except Exception:
        pass
    try:
        if hist['Close'].iloc[-1] < hist['BB_lower'].iloc[-1]:
            reasons.append("Price below lower Bollinger → potential buy")
        elif hist['Close'].iloc[-1] > hist['BB_upper'].iloc[-1]:
            reasons.append("Price above upper Bollinger → potential sell")
    except Exception:
        pass
    if matches:
        reasons.append("Déjà Vue pattern found")
    if ptb != 'N/A':
        try:
            ptb_val = float(ptb)
            if ptb_val < 1:
                reasons.append("PTB <1 → undervalued")
            elif ptb_val > 2:
                reasons.append("PTB >2 → overvalued")
        except Exception:
            pass
    if pred['predicted_price'] is not None:
        reasons.append(f"Prediction signal: {pred['signal']}")

    st.subheader("📌 Key Reasons")
    for r in reasons:
        st.write("- " + r)

    st.subheader("📌 Overall Recommendation")
    st.markdown(f"**{pred['signal']}**")

    if use_prophet and not PROPHET_AVAILABLE:
        st.warning("Prophet requested but not available in this environment. Uncheck 'Use Prophet' or install prophet/cmdstanpy.")

    st.markdown("This concludes the analysis for this stock.")
