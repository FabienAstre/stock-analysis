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
st.set_page_config(page_title="📊 Advanced Stock Analysis & AI Prediction App", layout="wide")
st.title("📊 Advanced Stock Analysis & AI Prediction App")

# --- Sidebar ---
st.sidebar.header("Settings")
tickers_input = st.sidebar.text_input("Enter Stock Tickers (comma-separated)", "NVDA,MSFT")
period = st.sidebar.selectbox("Select Historical Period", ["1mo","3mo","6mo","1y","5y"])
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

    # --- Fetch data ---
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        if hist.empty:
            st.warning("No historical data available.")
            continue
    except Exception as e:
        st.warning(f"Error fetching {ticker}: {e}")
        continue

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

    # --- Déjà Vue ---
    pattern_len = 5
    last_pattern = hist['Close'].values[-pattern_len:]
    matches = []
    for i in range(len(hist)-pattern_len-1):
        pattern = hist['Close'].values[i:i+pattern_len]
        corr = np.corrcoef(last_pattern, pattern)[0,1]
        if corr>0.95: matches.append((i,hist.index[i],corr))

    # --- Candlestick + indicators ---
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
        fig.add_trace(go.Scatter(x=hist.index, y=hist['EMA10'], line=dict(color='green', dash='dot'), name='EMA10'))
    if show_bb:
        fig.add_trace(go.Scatter(x=hist.index, y=hist['BB_upper'], line=dict(color='orange'), name='BB Upper'))
        fig.add_trace(go.Scatter(x=hist.index, y=hist['BB_lower'], line=dict(color='orange'), name='BB Lower'))
    if show_macd:
        fig.add_trace(go.Scatter(x=hist.index, y=hist['MACD'], line=dict(color='purple'), name='MACD'))
        fig.add_trace(go.Scatter(x=hist.index, y=hist['MACD_signal'], line=dict(color='black'), name='MACD Signal'))
    if show_deja and matches:
        for _, date, _ in matches:
            fig.add_vline(x=date, line_width=1, line_dash="dash", line_color="magenta")
    fig.update_layout(height=500, xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig, use_container_width=True)

    # --- Prediction & anomaly ---
    st.subheader("Advanced Prediction Signals")
    signal = "HOLD ⏸️"
    predicted_price = None
    lookback = 60
    hist_close = hist['Close'][-lookback:].dropna()
    if len(hist_close)>=lookback:
        X = np.arange(len(hist_close)).reshape(-1,1)
        y = hist_close.values

        # Linear Regression
        lr_pred = LinearRegression().fit(X,y).predict(np.array([[len(hist_close)]]))[0]

        # ARIMA
        try:
            arima_pred = ARIMA(hist_close, order=(5,1,0)).fit().forecast(steps=1)[0]
        except:
            arima_pred = None

        # Prophet
        try:
            df_prop = hist_close.reset_index()
            df_prop.columns = ['ds','y']
            prophet_model = Prophet(daily_seasonality=True)
            prophet_model.fit(df_prop)
            future = prophet_model.make_future_dataframe(periods=1)
            prophet_pred = prophet_model.predict(future)['yhat'].iloc[-1]
        except:
            prophet_pred = None

        preds = [p for p in [lr_pred,arima_pred,prophet_pred] if p is not None]
        if preds: predicted_price = np.mean(preds)
        current_price = hist['Close'].iloc[-1]
        if predicted_price:
            if predicted_price>current_price*1.002: signal="BUY ✅"
            elif predicted_price<current_price*0.998: signal="SELL ❌"
            st.write(f"Predicted next-day close: **${predicted_price:.2f}**")
            st.write(f"Signal: {signal}")

        # Anomaly detection
        hist['Returns'] = hist['Close'].pct_change()
        hist['ZScore'] = (hist['Returns']-hist['Returns'].mean())/hist['Returns'].std()
        if abs(hist['ZScore'].iloc[-1])>2:
            st.write("⚡ Anomaly detected: unusual price movement")
        st.markdown("**Explanation:** Ensemble of LR, ARIMA, Prophet + anomaly detection for robust short-term prediction.")
    else:
        st.warning("Not enough data for prediction (min 60 days).")

    # --- Déjà Vue signals ---
    st.subheader("🔁 Déjà Vue Trading Signals")
    if matches:
        deja_results=[]
        for i,date,sim in matches:
            trend_val = "N/A"
            if i+pattern_len+5<len(hist):
                fut=hist['Close'].iloc[i+pattern_len:i+pattern_len+5]
                ret=(fut.iloc[-1]-fut.iloc[0])/fut.iloc[0]
                if ret>0.02: trend_val="Uptrend 📈"
                elif ret<-0.02: trend_val="Downtrend 📉"
                else: trend_val="Flat ➖"
            deja_results.append({"Date":date.date(),"Similarity":f"{sim:.2%}","Trend After Pattern":trend_val})
        st.dataframe(pd.DataFrame(deja_results))
    else:
        st.write("No similar historical patterns found.")
    st.markdown("**Explanation:** Shows repeating patterns, similarity, and subsequent trend direction.")

    # --- Trending & Mean-Reversion ---
    st.subheader("📊 Trending & Mean-Reversion Signals")
    trend_signal = "Uptrend" if hist['EMA10'].iloc[-1]>hist['SMA50'].iloc[-1] else "Downtrend"
    st.write(f"Latest Trend Signal: {trend_signal}")
    st.write(f"RSI: {rsi:.2f} | ADX: {adx:.2f} | Volume Trend (SMA20): {hist['Volume_SMA20'].iloc[-1]:.0f}")
    st.markdown("""
**Explanation:**  
- **EMA10 vs SMA50:** Short-term EMA above SMA50 → bullish momentum  
- **RSI:** Measures overbought (>70) or oversold (<30)  
- **ADX:** >25 strong trend, <20 weak trend  
- **Volume Trend (SMA20):** Confirms participation supporting trend
""")

    # --- PTB ---
    st.subheader("💰 Price to Tangible Book (PTB)")
    try: ptb=stock.info.get('priceToBook',None)
    except: ptb=None
    st.write(f"Price to Tangible Book: {ptb if ptb else 'N/A'}")
    st.markdown("**Explanation:** PTB <1 undervalued; PTB>2 overvalued")

    # --- Key Reasons & Overall ---
    reasons = [f"Trend: {trend_signal}"]
    if rsi<30: reasons.append("RSI oversold → potential buy")
    elif rsi>70: reasons.append("RSI overbought → potential sell")
    reasons.append(f"MACD crossover: {hist['MACD_cross'].iloc[-1]}")
    if hist['Close'].iloc[-1]<hist['BB_lower'].iloc[-1]: reasons.append("Price below lower Bollinger → potential buy")
    elif hist['Close'].iloc[-1]>hist['BB_upper'].iloc[-1]: reasons.append("Price above upper Bollinger → potential sell")
    if matches: reasons.append("Déjà Vue pattern found")
    if ptb:
        if ptb<1: reasons.append("PTB <1 → undervalued")
        elif ptb>2: reasons.append("PTB >2 → overvalued")
    if predicted_price: reasons.append(f"Prediction signal: {signal}")

    st.subheader("📌 Key Reasons")
    for r in reasons: st.write("- "+r)

    # --- Overall Recommendation ---
    score=0
    weights={"prediction":0.5,"rsi":0.2,"macd":0.1,"bollinger":0.1}
    if signal=="BUY ✅": score+=weights["prediction"]
    elif signal=="SELL ❌": score-=weights["prediction"]
    if rsi<30: score+=weights["rsi"]
    elif rsi>70: score-=weights["rsi"]
    score+=weights["macd"] if hist['MACD_cross'].iloc[-1]=="bullish" else -weights["macd"]
    if hist['Close'].iloc[-1]<hist['BB_lower'].iloc[-1]: score+=weights["bollinger"]
    elif hist['Close'].iloc[-1]>hist['BB_upper'].iloc[-1]: score-=weights["bollinger"]
    overall="BUY ✅" if score>0 else "SELL ❌" if score<0 else "HOLD ⏸️"
    st.subheader("📌 Overall Recommendation")
    st.markdown(f"**{overall}**")
    st.markdown("This concludes the analysis for this stock.")
