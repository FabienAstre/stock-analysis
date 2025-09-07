# utils/data_fetch.py

import yfinance as yf
import pandas as pd
import numpy as np

def fetch_price_data(ticker, period="1y"):
    """Fetch historical OHLC data from Yahoo Finance."""
    data = yf.download(ticker, period=period)
    return data

def compute_rsi(series, window=14):
    """Compute Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]

def compute_zscore(series, window=20):
    """Compute rolling z-score."""
    if len(series) < window:
        return 0
    return (series.iloc[-1] - series.rolling(window).mean().iloc[-1]) / series.rolling(window).std().iloc[-1]
