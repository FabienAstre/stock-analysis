import yfinance as yf
import pandas as pd
import numpy as np

def fetch_price_data(ticker, period="1y"):
    """Fetch historical OHLC data from Yahoo Finance."""
    data = yf.download(ticker, period=period)
    return data
