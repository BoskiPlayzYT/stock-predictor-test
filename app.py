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

# --- Utility Functions ---
def fetch_price_data(symbol: str, months: int) -> pd.DataFrame:
    """Fetch historical Close prices via Stooq CSV (fallback to yfinance)."""
    ticker = symbol.lower()
    if not ticker.endswith('.us') and '-' not in symbol:
        ticker += '.us'
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

def fetch_financials(symbol: str):
    """Retrieve revenue and next earnings date."""
    rev_val = None
    next_earn = None
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        rev = info.get('totalRevenue') or ticker.financials.loc['Total Revenue'][0]
        rev_val = rev / 1e9 if rev else None
        cal = ticker.calendar
        next_earn = cal.loc['Earnings Date'][0]
    except Exception:
        pass
    return rev_val, next_earn

def fetch_news_sentiment(symbol: str, days: int = 7) -> float:
    """Aggregate VADER sentiment from Google News."""
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
            scores += [analyzer.polarity_scores(h)['compound'] for h in heads]
        except Exception:
            continue
    return np.mean(scores) if scores else 0.0

def monte_carlo_gbm(S0: float, mu: float, sigma: float, days: int, sims: int) -> np.ndarray:
    """Simulate GBM paths."""
    dt = 1/252
    paths = np.zeros((sims, days+1)); paths[:,0] = S0
    for t in range(1, days+1):
        Z = np.random.standard_normal(sims)
        paths[:,t] = paths[:,t-1] * np.exp((mu-0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z)
    return paths

# --- Technical Indicators ---
def compute_technical(df: pd.DataFrame) -> dict:
    data = {}
    data['MA50'] = df['Close'].rolling(50).mean().iloc[-1]
    data['MA200'] = df['Close'].rolling(200).mean().iloc[-1]
    delta = df['Close'].diff()
    up = delta.clip(lower=0);
    down = -delta.clip(upper=0)
    roll_up = up.rolling(14).mean(); roll_down = down.rolling(14).mean()
    RS = roll_up / roll_down; RSI = 100 - (100 / (1 + RS))
    data['RSI'] = RSI.iloc[-1]
    # Correlation with S&P500
    try:
        sp = yf.download('^GSPC', period=f"{len(df)}d", interval='1d', progress=False)['Close']
        data['Corr_SP500'] = np.corrcoef(df['Close'].pct_change()[1:], sp.pct_change()[1:])[0,1]
    except:
        data['Corr_SP500'] = None
    return data

# --- Streamlit App ---
st.set_page_config(page_title="✅ AI Stock Predictor", layout="wide")
st.markdown("""
<style>
  .css-1d391kg {background-color: #000000 !important;}
  .sidebar .sidebar-content {width: 300px;}
  header, footer {visibility: hidden;}
  .stMetric > div {color: #ffffff !important;}
</style>
""", unsafe_allow_html=True)

st.sidebar.title("Controls")
symbol = st.sidebar.text_input("Ticker", "AAPL").upper()
months = st.sidebar.slider("History (months)", 1, 24, 12)
sims = st.sidebar.slider("Monte Carlo Simulations", 100, 5000, 1000)

if st.sidebar.button("Run"):
    df = fetch_price_data(symbol, months)
    if df.empty:
        st.error(f"No data for '{symbol}'")
        st.stop()

    # Fundamental & sentiment
    revenue, next_earn = fetch_financials(symbol)
    sentiment = fetch_news_sentiment(symbol, days=7)

    # Returns for drift
    log_ret = np.log(df.Close / df.Close.shift(1)).dropna()
    mu = log_ret.mean()*252 + (sentiment * 0.1)
    sigma = log_ret.std()*np.sqrt(252)
    S0 = df.Close.iloc[-1]

    # Simulate
    paths = monte_carlo_gbm(S0, mu, sigma, 30, sims)
    median = np.median(paths, axis=0)
    up5 = (paths[:,-1] > S0*1.05).mean()*100
    down5 = (paths[:,-1] < S0*0.95).mean()*100

    # Tech indicators
    tech = compute_technical(df)
    cross_signal = "Buy" if tech['MA50'] > tech['MA200'] else "Sell"
    rsi_signal = "Oversold" if tech['RSI'] < 30 else ("Overbought" if tech['RSI'] > 70 else "Neutral")

    # Recommendation
    score = up5 - down5 + (10 if cross_signal=='Buy' else -10)
    if score > 30: rec = "Strong Buy"
    elif score > 10: rec = "Buy"
    elif score > -10: rec = "Hold"
    elif score > -30: rec = "Sell"
    else: rec = "Strong Sell"

    # Display metrics
    st.markdown(f"## {symbol} 30-Day Forecast 🎯")
    cols = st.columns(4)
    cols[0].metric("Price", f"${S0:.2f}")
    cols[1].metric("Revenue (B)", f"{revenue:.2f}" if revenue else "N/A")
    cols[2].metric("Sentiment", f"{sentiment:+.2f}")
    cols[3].metric("Recommendation", rec)
    if next_earn: st.markdown(f"**Next Earnings Date:** {next_earn.date()}")

    # Plot
    dates = pd.date_range(df.index[-1], periods=31, freq='D')
    fig, ax = plt.subplots(figsize=(12,6), facecolor='none')
    fig.patch.set_alpha(0)
    ax.set_facecolor('none')
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.tick_params(colors='white')
    ax.plot(df.index, df.Close, color='#1f77b4', label="Historical", linewidth=2)
    ax.plot(dates, median, '--', color='#ff7f0e', label="Forecast", linewidth=2)
    ax.fill_between(dates, np.percentile(paths,10,axis=0), np.percentile(paths,90,axis=0), color='#ff7f0e', alpha=0.2)
    legend = ax.legend(frameon=False)
    for text in legend.get_texts(): text.set_color('white')
    ax.set_ylabel("Price", color='white')
    st.pyplot(fig)

    # Scenario
    st.markdown("### Scenario Probabilities")
    st.table(pd.DataFrame({
        'Scenario': ['↑>5%','±5%','↓>5%'],
        'Probability': [f"{up5:.1f}%", f"{100-up5-down5:.1f}%", f"{down5:.1f}%"]
    }))

    # Technical Signals
    st.markdown("### Technical Signals")
    st.write(f"- 50-day MA vs 200-day MA: {cross_signal}")
    st.write(f"- RSI (14): {tech['RSI']:.1f} ({rsi_signal})")
    if tech['Corr_SP500'] is not None:
        st.write(f"- Correlation w/ S&P500: {tech['Corr_SP500']:.2f}")

    st.success("Analysis Complete!")
