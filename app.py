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
from bs4 import BeautifulSoup

# --- Data Fetch via Stooq CSV with yfinance fallback ---
def fetch_price_data(symbol: str, months: int) -> pd.DataFrame:
    ticker = symbol.lower()
    if '-' not in ticker and not ticker.endswith('.us'):
        ticker += '.us'
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=months * 30)
    url = (
        f"https://stooq.com/q/d/l/?s={ticker}"
        f"&d1={start_dt.strftime('%Y%m%d')}"
        f"&d2={end_dt.strftime('%Y%m%d')}&i=d"
    )
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

# --- Fetch fundamentals ---
def fetch_financials(symbol: str):
    ticker = yf.Ticker(symbol)
    rev_val, next_earn = None, None
    try:
        info = ticker.info
        rev = info.get('totalRevenue')
        if rev:
            rev_val = rev / 1e9
    except:
        pass
    try:
        cal = ticker.calendar
        next_earn = cal.loc['Earnings Date'][0]
    except:
        pass
    return rev_val, next_earn

# --- Fetch news and sentiment ---
def fetch_news_and_sentiment(symbol: str, count: int = 5):
    analyzer = SentimentIntensityAnalyzer()
    # Try yfinance news
    results = []
    try:
        items = yf.Ticker(symbol).news[:count]
    except:
        items = []
    # Fallback to Yahoo Finance news page
    if not items:
        url = f"https://finance.yahoo.com/quote/{symbol}/news?p={symbol}"
        try:
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            soup = BeautifulSoup(r.text, 'html.parser')
            cards = soup.select('li.js-stream-content h3 a')[:count]
            items = []
            for a in cards:
                title = a.get_text()
                link = a['href']
                if link.startswith('/'): link = 'https://finance.yahoo.com' + link
                items.append({'title': title, 'link': link})
        except:
            items = []
    # Process headlines
    for art in items:
        title = art.get('title') or art.get('title', '')
        link = art.get('link') or art.get('link', '')
        if title and link:
            score = analyzer.polarity_scores(title)['compound']
            results.append({'title': title, 'link': link, 'sentiment': score})
    avg_sent = np.mean([r['sentiment'] for r in results]) if results else 0.0
    return results, avg_sent

# --- Monte Carlo via Geometric Brownian Motion ---
def monte_carlo(S0, mu, sigma, days, sims):
    dt = 1/252
    paths = np.zeros((sims, days + 1)); paths[:, 0] = S0
    for t in range(1, days + 1):
        Z = np.random.standard_normal(sims)
        paths[:, t] = (
            paths[:, t - 1] *
            np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)
        )
    return paths

# --- Streamlit App ---
st.set_page_config(page_title="AI Stock Predictor", layout="wide")
st.markdown("""
<style>
  .css-1d391kg {background-color: #000 !important;}
  .sidebar .sidebar-content {width: 300px;}
  header, footer {visibility: hidden;}
  .stMetric > div {color: #fff !important;}
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Controls")
symbol = st.sidebar.text_input("Ticker", "AAPL").upper()
months = st.sidebar.slider("History (months)", 1, 24, 6)
sims = st.sidebar.slider("Monte Carlo Sims", 100, 5000, 1000)
others = st.sidebar.multiselect("Other tickers for news", ["MSFT","AMZN","GOOG","TSLA"], default=["MSFT","AMZN"])

if st.sidebar.button("Run Prediction"):
    # Fetch data
    df = fetch_price_data(symbol, months)
    if df.empty:
        st.error(f"No data for {symbol}")
        st.stop()
    revenue, next_earn = fetch_financials(symbol)
    news_list, news_sent = fetch_news_and_sentiment(symbol)

    # Technicals
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA50'] = df['Close'].rolling(50).mean()

    # MC parameters
    log_ret = np.log(df.Close / df.Close.shift(1)).dropna()
    mu = log_ret.mean() * 252 + news_sent * 0.1
    sigma = log_ret.std() * np.sqrt(252)
    S0 = df.Close.iloc[-1]
    days = 30

    # Simulate
    paths = monte_carlo(S0, mu, sigma, days, sims)
    median = np.median(paths, axis=0)
    p10 = np.percentile(paths, 10, axis=0)
    p90 = np.percentile(paths, 90, axis=0)
    up5 = (paths[:, -1] > S0 * 1.05).mean() * 100
    down5 = (paths[:, -1] < S0 * 0.95).mean() * 100
    bias = up5 - down5
    rec = (
        'Strong Buy' if bias > 40 else
        'Buy' if bias > 20 else
        'Hold' if bias > -20 else
        'Sell' if bias > -40 else
        'Strong Sell'
    )

    # Header
    st.markdown(f"## {symbol} — {rec} ({bias:.1f}% bias)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Price", f"${S0:.2f}")
    c2.metric("Revenue (B)", f"{revenue:.2f}" if revenue else "N/A")
    c3.metric("News Sentiment", f"{news_sent:+.2f}")
    if next_earn: c4.metric("Next Earnings", next_earn.date())

    # Prepare chart data
    fut_dates = pd.date_range(df.index[-1], periods=days + 1, freq='D')
    chart_df = pd.DataFrame({
        'date': np.concatenate([df.index.values, fut_dates.values]),
        'Price': np.concatenate([df.Close.values, [np.nan] * (days + 1)]),
        'MA20': np.concatenate([df.MA20.values, [np.nan] * (days + 1)]),
        'MA50': np.concatenate([df.MA50.values, [np.nan] * (days + 1)]),
        'Forecast': np.concatenate([[np.nan] * len(df), median]),
        'P10': np.concatenate([[np.nan] * len(df), p10]),
        'P90': np.concatenate([[np.nan] * len(df), p90])
    })

    # Line series for legend
    line_df = chart_df.melt('date', ['Price', 'MA20', 'MA50', 'Forecast'], var_name='Series', value_name='Value')

    # Y-axis domain
    min_val = min(chart_df['P10'].min(), df.Close.min())
    max_val = max(chart_df['P90'].max(), df.Close.max())
    y_scale = alt.Scale(domain=[min_val * 0.995, max_val * 1.005])

    # Selection for tooltip and crosshair
    selector = alt.selection_single(fields=['date'], nearest=True, on='mouseover', empty='none')
    base = alt.Chart(line_df).encode(x='date:T')
    band = alt.Chart(chart_df).mark_area(color='#ff7f0e', opacity=0.2).encode(
        x='date:T', y='P10:Q', y2='P90:Q')
    lines = base.mark_line().encode(
        y=alt.Y('Value:Q', scale=y_scale),
        color=alt.Color('Series:N', scale=alt.Scale(domain=['Price','MA20','MA50','Forecast'],
                                                  range=['#1f77b4','#2ca02c','#d62728','#ff7f0e'])),
        strokeDash=alt.condition(
            alt.datum.Series=='MA20', alt.value([5,5]),
            alt.condition(
                alt.datum.Series=='MA50', alt.value([2,2]),
                alt.condition(alt.datum.Series=='Forecast', alt.value([4,4]), alt.value([ ]))
            )
        ),
        strokeWidth=alt.condition(alt.datum.Series=='Forecast', alt.value(2), alt.value(1.5))
    )
    rule = base.mark_rule(color='white').encode(
        opacity=alt.condition(selector, alt.value(1), alt.value(0))
    ).add_selection(selector)
    tooltip = base.mark_point(size=0).encode(
        opacity=alt.value(0),
        tooltip=[alt.Tooltip('date:T', title='Date'), alt.Tooltip('Value:Q', title='Value'), alt.Tooltip('Series:N', title='Series')]
    ).add_selection(selector)

    chart = alt.layer(band, lines, rule, tooltip).properties(width='container', height=300)
    chart = chart.configure_axis(labelColor='white', titleColor='white')
    chart = chart.configure_legend(labelColor='white', titleColor='white')
    st.altair_chart(chart, use_container_width=True)

    # Main news
    st.markdown(f"## Recent News for {symbol}")
    if news_list:
        for item in news_list:
            st.markdown(f"- [{item['title']}]({item['link']}) (Sentiment: {item['sentiment']:+.2f})")
    else:
        st.write("No recent news available.")

    # Other tickers news
    st.markdown("## News for Others")
    for o in others:
        st.markdown(f"### {o}")
        ol, osent = fetch_news_and_sentiment(o)
        if ol:
            for it in ol:
                st.markdown(f"- [{it['title']}]({it['link']}) (Sentiment: {it['sentiment']:+.2f})")
        else:
            st.write("No recent news available.")

    # Scenario probabilities
    st.markdown("## Scenario Probabilities")
    st.write(f"**Up >5%:** {up5:.1f}%  |  **Stable:** {100-up5-down5:.1f}%  |  **Down >5%:** {down5:.1f}%")

    st.success("Analysis complete!")
