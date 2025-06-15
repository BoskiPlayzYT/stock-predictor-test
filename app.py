# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO
from datetime import datetime, timedelta
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.graph_objs as go
from bs4 import BeautifulSoup

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

# --- Fetch fundamentals ---
def fetch_financials(symbol: str):
    ticker = yf.Ticker(symbol)
    rev_val = None
    next_earn = None
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

# --- Fetch news and sentiment with fallback to Google News ---
def fetch_news_and_sentiment(symbol: str, count:int=5):
    analyzer = SentimentIntensityAnalyzer()
    results = []
    # Try yfinance news
    try:
        items = yf.Ticker(symbol).news[:count]
    except:
        items = []
    # If no items, fallback scrape Google News
    if not items:
        query = f"{symbol} site:news.google.com"
        url = f"https://www.google.com/search?q={requests.utils.quote(query)}&tbm=nws"
        try:
            r = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=5)
            soup = BeautifulSoup(r.text, 'html.parser')
            headlines = soup.select('.BNeawe.vvjwJb.AP7Wnd')[:count]
            links = [a.parent['href'] for a in headlines]
            items = [{'title': h.get_text(), 'link': l} for h,l in zip(headlines, links)]
        except:
            items = []
    # Process items
    for art in items:
        title = art.get('title','')
        link = art.get('link','')
        if title and link:
            sentiment = analyzer.polarity_scores(title)['compound']
            results.append({'title': title, 'link': link, 'sentiment': sentiment})
    avg_sent = np.mean([r['sentiment'] for r in results]) if results else 0.0
    return results, avg_sent

# --- Monte Carlo via GBM ---
def monte_carlo(S0, mu, sigma, days, sims):
    dt = 1/252
    paths = np.zeros((sims, days+1))
    paths[:,0] = S0
    for t in range(1, days+1):
        Z = np.random.standard_normal(sims)
        paths[:,t] = paths[:,t-1] * np.exp((mu - 0.5*sigma**2)*dt + sigma * np.sqrt(dt) * Z)
    return paths

# --- Streamlit UI ---
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
    # Data fetch
    df = fetch_price_data(symbol, months)
    if df.empty:
        st.error(f"No data for {symbol}")
        st.stop()
    revenue, next_earn = fetch_financials(symbol)
    news_list, news_sent = fetch_news_and_sentiment(symbol)

    # Technicals
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA50'] = df['Close'].rolling(50).mean()

    # Monte Carlo params
    log_ret = np.log(df.Close/df.Close.shift(1)).dropna()
    mu = log_ret.mean()*252 + news_sent*0.1
    sigma = log_ret.std()*np.sqrt(252)
    S0 = df.Close.iloc[-1]

    # Simulate
    days = 30
    paths = monte_carlo(S0, mu, sigma, days, sims)
    median = np.median(paths, axis=0)
    p10 = np.percentile(paths,10,axis=0)
    p90 = np.percentile(paths,90,axis=0)
    up5 = (paths[:,-1]>S0*1.05).mean()*100
    down5 = (paths[:,-1]<S0*0.95).mean()*100
    bias = up5-down5
    rec = ('Strong Buy' if bias>40 else 'Buy' if bias>20 else 'Hold' if bias>-20 else 'Sell' if bias>-40 else 'Strong Sell')

    # Header metrics
    st.markdown(f"## {symbol} — {rec} ({bias:.1f}% bias)")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Price", f"${S0:.2f}")
    c2.metric("Revenue (B)", f"{revenue:.2f}" if revenue else "N/A")
    c3.metric("News Sentiment", f"{news_sent:+.2f}")
    if next_earn: c4.metric("Next Earnings", next_earn.date())

    # Plotly chart
    fut_dates = pd.date_range(df.index[-1], periods=days+1, freq='D')
    fig = go.Figure()
    # Actual and MAs
    fig.add_trace(go.Scatter(x=df.index, y=df.Close, name='Price', line=dict(color='#1f77b4')))
    fig.add_trace(go.Scatter(x=df.index, y=df.MA20, name='MA20', line=dict(color='#2ca02c', dash='dash')))
    fig.add_trace(go.Scatter(x=df.index, y=df.MA50, name='MA50', line=dict(color='#d62728', dash='dot')))
    # Forecast and band
    fig.add_trace(go.Scatter(x=fut_dates, y=median, name='Forecast', line=dict(color='#ff7f0e', dash='dash')))
    fig.add_trace(go.Scatter(x=np.concatenate([fut_dates, fut_dates[::-1]]),
                             y=np.concatenate([p90, p10[::-1]]),
                             fill='toself', fillcolor='rgba(255,127,14,0.2)', line=dict(color='rgba(255,127,14,0)'),
                             hoverinfo='skip', showlegend=True, name='10-90% Band'))
    # Layout
    fig.update_layout(plot_bgcolor='black', paper_bgcolor='black', font_color='white',
                      hovermode='x unified', legend=dict(bgcolor='rgba(0,0,0,0)'),
                      margin=dict(l=40,r=40,t=40,b=40))
    fig.update_xaxes(showspikes=True, spikecolor='white', spikethickness=1, spikedash='dash')
    fig.update_yaxes(showspikes=True, spikecolor='white', spikethickness=1, spikedash='dash')
    st.plotly_chart(fig, use_container_width=True)

    # Main stock news
    st.markdown(f"## Recent News for {symbol}")
    if news_list:
        news_df = pd.DataFrame(news_list)
        news_df['Headline'] = news_df.apply(lambda r: f"[{r['title']}]({r['link']})", axis=1)
        news_df = news_df[['Headline','sentiment']].rename(columns={'sentiment':'Sentiment'})
        st.write(news_df.to_html(escape=False), unsafe_allow_html=True)
    else:
        st.write("No recent news available.")

    # News for other tickers
    st.markdown("## News for Others")
    for o in others:
        st.markdown(f"### {o}")
        ol, os = fetch_news_and_sentiment(o)
        if ol:
            odf = pd.DataFrame(ol)
            odf['Headline'] = odf.apply(lambda r: f"[{r['title']}]({r['link']})", axis=1)
            odf = odf[['Headline','sentiment']].rename(columns={'sentiment':'Sentiment'})
            st.write(odf.to_html(escape=False), unsafe_allow_html=True)
        else:
            st.write("No recent news available.")

    # Scenario probabilities
    st.markdown("## Scenario Probabilities")
    st.write(f"**Up >5%:** {up5:.1f}%  |  **Stable:** {100-up5-down5:.1f}%  |  **Down >5%:** {down5:.1f}%")

    st.success("Analysis complete!")
