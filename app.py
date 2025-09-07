import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objs as go
from utils.data_fetch import fetch_price_data

# ================================
# 📊 Stock Screener & Advanced Analyzer
# ================================
st.title("📊 Stock Screener & Advanced Analyzer")

# === User Inputs ===
tickers = st.text_input("Enter tickers (comma-separated)", "AAPL, MSFT, TSLA").split(",")
period = st.selectbox("Select historical period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)

df_summary = []

# ================================
# 🔧 Helper Functions
# ================================
def compute_rsi(series, window=14):
    if series.empty or len(series) < window:
        return None
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None

def compute_zscore(series, window=20):
    if series.empty or len(series) < window:
        return None
    mean = series.rolling(window).mean().iloc[-1]
    std = series.rolling(window).std().iloc[-1]
    if std == 0 or pd.isna(std):
        return None
    return float((series.iloc[-1] - mean) / std)

def compute_macd(series, fast=12, slow=26, signal=9):
    if series.empty or len(series) < slow:
        return None, None
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return float(macd.iloc[-1]), float(signal_line.iloc[-1])

def compute_bollinger(series, window=20, num_std=2):
    if series.empty or len(series) < window:
        return None, None, None
    sma = series.rolling(window).mean().iloc[-1]
    std = series.rolling(window).std().iloc[-1]
    return float(sma), float(sma + num_std * std), float(sma - num_std * std)

def anomaly_score(series):
    returns = series.pct_change().dropna()
    if len(returns) < 20:
        return None
    rolling_vol = returns.rolling(20).std().iloc[-1]
    if rolling_vol == 0 or pd.isna(rolling_vol):
        return None
    return float(returns.iloc[-1] / rolling_vol)

def deja_vu_similarity(series, window=10):
    if len(series) < window * 2:
        return None
    recent = series[-window:].pct_change().dropna().values
    best_corr = -1
    for i in range(len(series) - 2 * window):
        past = series[i:i+window].pct_change().dropna().values
        if len(past) == len(recent):
            corr = np.corrcoef(recent, past)[0, 1]
            best_corr = max(best_corr, corr)
    return float(best_corr) if best_corr >= 0 else None

def trending_reversion_signals(series):
    if len(series) < 50:
        return "Insufficient data"
    ma20, ma50 = series.rolling(20).mean().iloc[-1], series.rolling(50).mean().iloc[-1]
    rsi = compute_rsi(series)
    if rsi is None:
        return "Insufficient data"
    if ma20 > ma50 and rsi < 70:
        return "Trending Up"
    elif ma20 < ma50 and rsi > 30:
        return "Trending Down"
    elif rsi >= 70:
        return "Overbought (Reversion risk)"
    elif rsi <= 30:
        return "Oversold (Reversion chance)"
    else:
        return "Neutral"

def situational_analysis(series, fundamentals, rsi, macd, macd_signal):
    if len(series) < 50 or rsi is None or macd is None or macd_signal is None:
        return "Not enough data"

    ma20, ma50 = series.rolling(20).mean().iloc[-1], series.rolling(50).mean().iloc[-1]
    trend = "Bullish" if ma20 > ma50 else "Bearish"

    vol = series.pct_change().rolling(20).std().iloc[-1]
    hist_vol = series.pct_change().rolling(100).std().mean()
    vol_regime = "High" if vol > hist_vol * 1.2 else "Low" if vol < hist_vol * 0.8 else "Normal"

    if macd > macd_signal and rsi < 70:
        momentum = "Strong Up"
    elif macd < macd_signal and rsi > 30:
        momentum = "Strong Down"
    else:
        momentum = "Neutral"

    zscore = compute_zscore(series)
    pe = fundamentals.get("trailingPE", None)
    valuation = "Fair"
    if zscore is not None:
        if zscore < -1 and pe and pe < 20:
            valuation = "Undervalued"
        elif zscore > 1 and pe and pe > 30:
            valuation = "Overvalued"

    if trend == "Bullish" and vol_regime == "Low" and valuation == "Undervalued":
        return "📈 Accumulation / Long Trend Play"
    elif trend == "Bearish" and vol_regime == "High":
        return "⚠️ Avoid / Hedge Risk"
    elif momentum == "Strong Up" and valuation == "Overvalued":
        return "💡 Momentum Long (Short-term)"
    elif momentum == "Strong Down" and valuation == "Undervalued":
        return "🔄 Mean Reversion Buy Setup"
    else:
        return f"{trend} / {vol_regime} Vol — Neutral"

# ================================
# 🔄 Loop through Tickers
# ================================
for ticker in [t.strip().upper() for t in tickers]:
    try:
        data = fetch_price_data(ticker, period)
        if data.empty:
            st.warning(f"No data found for {ticker}")
            continue

        close = data["Close"].dropna()
        if close.empty:
            st.warning(f"No closing price data for {ticker}")
            continue

        current_price = float(close.iloc[-1])

        # === Technicals ===
        rsi = compute_rsi(close)
        zscore = compute_zscore(close)
        macd, macd_signal = compute_macd(close)
        sma20, upper_bb, lower_bb = compute_bollinger(close)

        # === Quant Signals ===
        anomaly = anomaly_score(close)
        deja_corr = deja_vu_similarity(close)
        trend_signal = trending_reversion_signals(close)

        # === Fundamentals ===
        info = yf.Ticker(ticker).info
        situational = situational_analysis(close, info, rsi, macd, macd_signal)

        df_summary.append({
            "Ticker": ticker,
            "Price": current_price,
            "RSI": rsi,
            "Z-Score": zscore,
            "MACD": macd,
            "MACD Signal": macd_signal,
            "Anomaly Score": anomaly,
            "Déjà Vu Corr": deja_corr,
            "Trend/Reversion": trend_signal,
            "Situational Analysis": situational,
            "P/E": info.get("trailingPE"),
            "P/B": info.get("priceToBook"),
            "Dividend Yield": info.get("dividendYield"),
            "Market Cap": info.get("marketCap")
        })

        # === Chart ===
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data["Open"], high=data["High"], low=data["Low"], close=data["Close"],
            name="Price"
        ))

        if len(close) >= 200 and sma20 is not None:
            fig.add_trace(go.Scatter(x=data.index, y=close.rolling(20).mean(), line=dict(color='blue'), name="20MA"))
            fig.add_trace(go.Scatter(x=data.index, y=close.rolling(50).mean(), line=dict(color='orange'), name="50MA"))
            fig.add_trace(go.Scatter(x=data.index, y=close.rolling(200).mean(), line=dict(color='green'), name="200MA"))
            if upper_bb is not None and lower_bb is not None:
                fig.add_trace(go.Scatter(x=data.index, y=[upper_bb]*len(data), line=dict(color='red', dash='dot'), name="Upper BB"))
                fig.add_trace(go.Scatter(x=data.index, y=[lower_bb]*len(data), line=dict(color='red', dash='dot'), name="Lower BB"))

        fig.update_layout(title=f"{ticker} Technical Chart", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error fetching {ticker}: {e}")

# ================================
# 📋 Show Results
# ================================
if df_summary:
    st.subheader("📋 Stock Analysis Summary")
    df = pd.DataFrame(df_summary)
    st.dataframe(df)
    st.download_button("Download CSV", df.to_csv(index=False), "stock_analysis.csv", "text/csv")
