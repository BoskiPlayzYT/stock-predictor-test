# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- Data Fetch via Stooq (CSV) with yfinance fallback ---
def fetch_price_data(symbol: str, months: int) -> pd.DataFrame:
    ticker = symbol.lower()
    if not ticker.endswith('.us') and '-' not in symbol:
        ticker = f"{ticker}.us"
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=months*30)
    url = f"https://stooq.com/q/d/l/?s={ticker}&d1={start_dt.strftime('%Y%m%d')}&d2={end_dt.strftime('%Y%m%d')}&i=d"
    try:
        r = requests.get(url, timeout=10)
        df = pd.read_csv(StringIO(r.text), parse_dates=['Date'], index_col='Date')
        return df[['Close']].dropna()
    except Exception:
        pass
    try:
        df = yf.download(symbol, period=f"{months}mo", interval="1d", progress=False)
        return df[['Close']].dropna()
    except Exception:
        return pd.DataFrame()

# --- Fetch financial metrics ---
def fetch_financials(symbol: str):
    rev_val = None
    next_earn = None
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        if 'totalRevenue' in info and info['totalRevenue']:
            rev_val = info['totalRevenue'] / 1e9
    except Exception:
        rev_val = None
    try:
        cal = ticker.calendar
        next_earn = cal.loc['Earnings Date'][0]
    except Exception:
        next_earn = None
    return rev_val, next_earn

# --- News sentiment analysis ---
def fetch_news_sentiment(symbol: str, days: int = 7) -> float:
    analyzer = SentimentIntensityAnalyzer()
    scores = []
    now = datetime.now()
    for i in range(days):
        day = now - timedelta(days=i)
        q = f"{symbol} site:news.google.com after:{day.date()} before:{(day+timedelta(days=1)).date()}"
        url = f"https://www.google.com/search?q={requests.utils.quote(q)}&tbm=nws"
        try:
            r = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=5)
            soup = BeautifulSoup(r.text, 'html.parser')
            heads = [h.get_text() for h in soup.select('.BNeawe.vvjwJb.AP7Wnd')][:5]
            scores.extend([analyzer.polarity_scores(h)['compound'] for h in heads])
        except Exception:
            continue
    return np.mean(scores) if scores else 0.0

# --- Monte Carlo via GBM ---
def monte_carlo_gbm(S0: float, mu: float, sigma: float, days: int, sims: int) -> np.ndarray:
    dt = 1/252
    paths = np.zeros((sims, days+1)); paths[:,0] = S0
    for t in range(1, days+1):
        Z = np.random.standard_normal(sims)
        paths[:,t] = paths[:,t-1] * np.exp((mu-0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z)
    return paths

# --- Streamlit UI ---
st.set_page_config(page_title="AI Stock Predictor", layout="wide")
st.markdown("""
<style>
  .css-1d391kg {background-color: #ffffff !important;}
  .sidebar .sidebar-content {width: 300px;}
  header, footer {visibility: hidden;}
""", unsafe_allow_html=True)

st.sidebar.title("Controls")
symbol = st.sidebar.text_input("Ticker (AAPL, TSLA, BTC-USD)", "AAPL").upper()
months = st.sidebar.slider("History (months)", 1, 24, 6)
sims = st.sidebar.slider("Simulations", 100, 2000, 500)

if st.sidebar.button("Run Prediction"):
    prices = fetch_price_data(symbol, months)
    if prices.empty:
        st.error(f"No price data for '{symbol}'.")
        st.stop()

    # compute parameters
    log_ret = np.log(prices.Close/prices.Close.shift(1)).dropna()
    mu = log_ret.mean()*252; sigma = log_ret.std()*np.sqrt(252)
    S0 = prices.Close.iloc[-1]

    paths = monte_carlo_gbm(S0, mu, sigma, 30, sims)
    median = np.median(paths, axis=0)
    p_up5 = (paths[:,-1]>S0*1.05).mean()*100
    p_down5 = (paths[:,-1]<S0*0.95).mean()*100

    # extra factors
    revenue, next_earn = fetch_financials(symbol)
    sentiment = fetch_news_sentiment(symbol, days=7)

    # header metrics
    st.markdown(f"## {symbol} 30-Day Forecast")
    c1, c2, c3 = st.columns(3)
    c1.metric("Current Price", f"${S0:.2f}")
    c2.metric("Revenue (B)", f"{revenue:.2f}" if revenue else "N/A")
    c3.metric("Sentiment", f"{sentiment:+.2f}")
    if next_earn:
        st.markdown(f"**Next Earnings Date:** {next_earn.date()}")

    # plot
    dates = pd.date_range(prices.index[-1], periods=31, freq='D')
    fig, ax = plt.subplots(figsize=(10,4), facecolor='none')
    fig.patch.set_alpha(0)
    ax.set_facecolor('none')
    # remove borders
    for spine in ax.spines.values():
        spine.set_visible(False)
    # tick colors
    ax.tick_params(colors='#555555')
    ax.plot(prices.index, prices.Close, label="Historical")
    ax.plot(dates, median, '--', label="Forecast")
    ax.fill_between(dates, np.percentile(paths,10,axis=0), np.percentile(paths,90,axis=0), alpha=0.2)
    ax.legend(frameon=False)
    ax.set_ylabel("Price", color='#333333')
    st.pyplot(fig)

    # probabilities
    st.markdown("### Scenario Probabilities")
    st.table(pd.DataFrame({
        'Scenario':['↑ > 5%','±5%','↓ > 5%'],
        'Probability':[f"{p_up5:.1f}%", f"{100-p_up5-p_down5:.1f}%", f"{p_down5:.1f}%"]
    }))

    # assessment
    opinion = "Bullish" if sentiment>0 and p_up5>p_down5 else "Bearish" if sentiment<0 and p_down5>p_up5 else "Neutral"
    st.markdown(f"**Overall Assessment:** {opinion}")
    st.success("Complete!")
