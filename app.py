# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO
from datetime import datetime, timedelta
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import altair as alt

# --- Data Fetch via Stooq CSV with yfinance fallback ---
def fetch_price_data(symbol: str, months: int) -> pd.DataFrame:
    ticker = symbol.lower()
    if '-' not in ticker and not ticker.endswith('.us'):
        ticker += '.us'
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=months*30)
    url = f"https://stooq.com/q/d/l/?s={ticker}&d1={start_dt.strftime('%Y%m%d')}&d2={end_dt.strftime('%Y%m%d')}&i=d"
    try:
        r = requests.get(url, timeout=10)
        df = pd.read_csv(StringIO(r.text), parse_dates=['Date'], index_col='Date')
        return df[['Close']].dropna()
    except:
        pass
    try:
        df = yf.download(symbol, period=f"{months}mo", interval="1d", progress=False)
        return df[['Close']].dropna()
    except:
        return pd.DataFrame()

# --- Fetch fundamentals and news ---
def fetch_financials(symbol: str):
    ticker = yf.Ticker(symbol)
    rev = None; earn = None
    try:
        info = ticker.info
        rev = info.get('totalRevenue')
        if rev:
            rev = rev / 1e9
    except:
        rev = None
    try:
        cal = ticker.calendar
        earn = cal.loc['Earnings Date'][0]
    except:
        earn = None
    return rev, earn

# Safely fetch news headlines and sentiment
def fetch_news_and_sentiment(symbol: str, count:int=5):
    analyzer = SentimentIntensityAnalyzer()
    ticker = yf.Ticker(symbol)
    try:
        news_items = ticker.news[:count]
    except:
        news_items = []
    results = []
    for article in news_items:
        title = article.get('title') or ''
        link = article.get('link') or ''
        if not title or not link:
            continue
        score = analyzer.polarity_scores(title)['compound']
        results.append({'title': title, 'link': link, 'sentiment': score})
    avg_sent = np.mean([r['sentiment'] for r in results]) if results else 0.0
    return results, avg_sent

# --- Monte Carlo via Geometric Brownian Motion ---
def monte_carlo(S0, mu, sigma, days, sims):
    dt = 1/252
    paths = np.zeros((sims, days+1)); paths[:,0] = S0
    for t in range(1, days+1):
        Z = np.random.standard_normal(sims)
        paths[:,t] = paths[:,t-1] * np.exp((mu - 0.5*sigma**2)*dt + sigma * np.sqrt(dt) * Z)
    return paths

# --- User Interface ---
st.set_page_config(page_title="AI Stock Predictor", layout="wide")
st.markdown("""
<style>
  .css-1d391kg {background-color: #000 !important;}
  .sidebar .sidebar-content {width: 300px;}
  header, footer {visibility: hidden;}
  .stMetric > div {color: #fff !important;}
""", unsafe_allow_html=True)

st.sidebar.title("Controls")
symbol = st.sidebar.text_input("Ticker", "AAPL").upper()
months = st.sidebar.slider("History (months)", 1, 24, 6)
sims = st.sidebar.slider("Monte Carlo Sims", 100, 5000, 1000)
others = st.sidebar.multiselect("Additional tickers for news", ["MSFT","AMZN","GOOG","TSLA"], default=["MSFT","AMZN"])

if st.sidebar.button("Run"):
    # Fetch price data
    df = fetch_price_data(symbol, months)
    if df.empty:
        st.error(f"No data for {symbol}")
        st.stop()

    # Fetch fundamentals and news sentiment
    revenue, next_earn = fetch_financials(symbol)
    news_list, news_sent = fetch_news_and_sentiment(symbol)

    # Technical indicators
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA50'] = df['Close'].rolling(50).mean()

    # Calculate returns and Monte Carlo parameters
    log_ret = np.log(df.Close/df.Close.shift(1)).dropna()
    mu = log_ret.mean()*252 + news_sent*0.1
    sigma = log_ret.std()*np.sqrt(252)
    S0 = df.Close.iloc[-1]

    # Run Monte Carlo simulation
    paths = monte_carlo(S0, mu, sigma, 30, sims)
    median = np.median(paths, axis=0)
    up5 = (paths[:,-1] > S0*1.05).mean()*100
    down5 = (paths[:,-1] < S0*0.95).mean()*100
    bias = up5 - down5
    rec = ('Strong Buy' if bias>40 else 'Buy' if bias>20 else 'Hold' if bias>-20 else 'Sell' if bias>-40 else 'Strong Sell')

    # Display header metrics
    st.markdown(f"## {symbol} — {rec} ({bias:.1f}% bias)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Price", f"${S0:.2f}")
    c2.metric("Revenue (B)", f"{revenue:.2f}" if revenue else "N/A")
    c3.metric("News Sentiment", f"{news_sent:+.2f}")
    if next_earn: c4.metric("Next Earnings", next_earn.date())

    # Prepare chart dataset
    fut_dates = pd.date_range(df.index[-1], periods=31, freq='D')
    chart_df = pd.DataFrame({
        'date': np.concatenate([df.index.values, fut_dates.values]),
        'Price': np.concatenate([df.Close.values, [None]*31]),
        'MA20': np.concatenate([df.MA20.values, [None]*31]),
        'MA50': np.concatenate([df.MA50.values, [None]*31]),
        'Forecast': np.concatenate([[None]*len(df), median])
    })

    # Build interactive Altair chart
    base = alt.Chart(chart_df).encode(x='date:T')
    price_line = base.mark_line(color='#1f77b4', strokeWidth=2).encode(y='Price:Q')
    ma20_line = base.mark_line(color='#2ca02c', strokeDash=[5,5]).encode(y='MA20:Q')
    ma50_line = base.mark_line(color='#d62728', strokeDash=[2,2]).encode(y='MA50:Q')
    forecast_line = base.mark_line(color='#ff7f0e', strokeDash=[4,4], strokeWidth=2).encode(y='Forecast:Q')
    band = base.mark_area(color='#ff7f0e', opacity=0.2).encode(y='Forecast:Q', y2='Forecast:Q')
    chart = alt.layer(band, price_line, ma20_line, ma50_line, forecast_line)
    chart = chart.properties(width='container', height=400)
    st.altair_chart(chart, use_container_width=True)

    # Main stock news
    st.markdown(f"## Recent News for {symbol}")
    news_df = pd.DataFrame(news_list)
    news_df['Headline'] = news_df.apply(lambda r: f"[{r.title}]({r.link})", axis=1)
    st.table(news_df[['Headline','sentiment']].rename(columns={'sentiment':'Sentiment'}))

    # News for other tickers
    st.markdown("## News for Others")
    for o in others:
        st.markdown(f"### {o}")
        ol, os = fetch_news_and_sentiment(o)
        odf = pd.DataFrame(ol)
        odf['Headline'] = odf.apply(lambda r: f"[{r.title}]({r.link})", axis=1)
        st.table(odf[['Headline','sentiment']].rename(columns={'sentiment':'Sentiment'}))

    # Scenario probabilities
    st.markdown("## Scenario Probabilities")
    st.write(f"**Up >5%:** {up5:.1f}%  |  **Stable:** {100-up5-down5:.1f}%  |  **Down >5%:** {down5:.1f}%")

    st.success("Analysis complete!")
