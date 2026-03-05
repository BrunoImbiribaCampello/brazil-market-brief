"""
╔══════════════════════════════════════════════════════════╗
║     LIVE MARKET BRIEF — Streamlit Web Dashboard          ║
║     Bruno Imbiriba Campello | UFRJ Finance               ║
║                                                          ║
║  DEPLOY IN 3 STEPS:                                      ║
║    1. Push this file to GitHub                           ║
║    2. Go to share.streamlit.io                           ║
║    3. Connect repo → Deploy → Get public URL             ║
╚══════════════════════════════════════════════════════════╝
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# ── PAGE CONFIG ────────────────────────────────────────────
# This must be the FIRST streamlit command in the file
st.set_page_config(
    page_title="Market Brief | Bruno Campello",
    page_icon="📊",
    layout="wide",                    # use full browser width
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ─────────────────────────────────────────────
# st.markdown() with unsafe_allow_html=True lets you inject CSS
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0A0F2C; }
    
    /* All text white */
    .stApp, .stMarkdown, p, h1, h2, h3, label { color: #F8F6F2 !important; }
    
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #111830; }
    
    /* Metric cards */
    [data-testid="metric-container"] {
        background-color: #111830;
        border: 1px solid #C9A84C;
        border-radius: 8px;
        padding: 12px;
    }
    [data-testid="metric-container"] label { color: #C9A84C !important; font-size: 11px !important; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] { color: #F8F6F2 !important; }
    [data-testid="stMetricDelta"] svg { display: none; }
    
    /* Header */
    .header-box {
        background: linear-gradient(135deg, #0A0F2C 0%, #1A2A5E 100%);
        border-bottom: 3px solid #C9A84C;
        padding: 20px 30px;
        margin-bottom: 20px;
        border-radius: 8px;
    }
    .header-title { color: #C9A84C !important; font-size: 28px; font-weight: 900; letter-spacing: 2px; }
    .header-sub   { color: #AAAAAA !important; font-size: 13px; }
    
    /* Section headers */
    .section-header {
        background-color: #111830;
        border-left: 4px solid #C9A84C;
        padding: 6px 14px;
        margin: 12px 0 8px 0;
        border-radius: 0 4px 4px 0;
        font-weight: 700;
        color: #F8F6F2 !important;
        font-size: 13px;
        letter-spacing: 1px;
    }
    
    /* Positive / negative colours */
    .pos { color: #2D9E6B !important; font-weight: 700; }
    .neg { color: #E03C3C !important; font-weight: 700; }
    
    /* Hide streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }
    
    /* DataFrame styling */
    .stDataFrame { border: 1px solid #C9A84C !important; }
</style>
""", unsafe_allow_html=True)

# ── COLOUR PALETTE (for Plotly) ────────────────────────────
NAVY    = '#0A0F2C'
NAVY2   = '#111830'
GOLD    = '#C9A84C'
GREEN   = '#2D9E6B'
RED     = '#E03C3C'
GRAY    = '#888888'
WHITE   = '#F8F6F2'

STOCK_COLORS = [GOLD, '#4A9EFF', '#9B59B6', GREEN, '#E67E22', RED]

# ── TICKER DEFINITIONS ─────────────────────────────────────
DEFAULT_STOCKS = {
    "PETR4.SA": "Petrobras",
    "VALE3.SA": "Vale",
    "ITUB4.SA": "Itaú Unibanco",
    "BBDC4.SA": "Bradesco",
    "WEGE3.SA": "WEG",
}

INDICES = {
    "^BVSP":    "Ibovespa",
    "^GSPC":    "S&P 500",
    "^DJI":     "Dow Jones",
    "^IXIC":    "Nasdaq",
}

FX_PAIRS = {
    "BRL=X":    "USD/BRL",
    "EURUSD=X": "EUR/USD",
    "GBPUSD=X": "GBP/USD",
}

# ── DATA FETCHING ──────────────────────────────────────────
# @st.cache_data tells Streamlit:
# "Cache this result for 15 minutes — don't re-fetch on every click"
# This is CRITICAL for performance — without it, every interaction
# triggers a full API call

@st.cache_data(ttl=900)   # ttl=900 seconds = 15 minutes
def fetch_stock_data(ticker, period="35d"):
    """
    Fetches price history and calculates key metrics.
    Returns a dict with price, changes, volatility, history.
    
    WHY @st.cache_data:
    Without it: Every time user clicks anything → new API call → slow
    With it:    Data is stored in memory for 15 min → instant response
    """
    try:
        stock = yf.Ticker(ticker)
        hist  = stock.history(period=period)
        
        if len(hist) < 2:
            return None
        
        today      = hist['Close'].iloc[-1]
        yesterday  = hist['Close'].iloc[-2]
        week_ago   = hist['Close'].iloc[-6]  if len(hist) >= 6  else hist['Close'].iloc[0]
        month_ago  = hist['Close'].iloc[-22] if len(hist) >= 22 else hist['Close'].iloc[0]
        
        daily_returns = hist['Close'].pct_change().dropna()
        
        return {
            'price':       round(today, 2),
            'daily_chg':   round((today - yesterday) / yesterday * 100, 2),
            'weekly_chg':  round((today - week_ago)  / week_ago  * 100, 2),
            'monthly_chg': round((today - month_ago) / month_ago * 100, 2),
            'volatility':  round(daily_returns.std() * (252**0.5) * 100, 1),
            'history':     hist['Close'],
            'volume':      int(hist['Volume'].iloc[-1]) if 'Volume' in hist else 0,
            'high_52w':    round(hist['Close'].max(), 2),
            'low_52w':     round(hist['Close'].min(), 2),
        }
    except Exception as e:
        return None

# ── CHART BUILDERS ─────────────────────────────────────────
def make_price_chart(stocks_data, selected_tickers):
    """
    Multi-line indexed price chart.
    
    WHY INDEXED?
    Petrobras trades at R$39, Vale at R$65.
    If you plot both on same axis, Vale dominates and Petrobras looks flat.
    Indexing to 100 puts both on equal footing — you see % change, not price.
    """
    fig = go.Figure()
    
    for (ticker, data), color in zip(
        {k: v for k, v in stocks_data.items() if k in selected_tickers}.items(),
        STOCK_COLORS
    ):
        hist = data['history']
        indexed = (hist / hist.iloc[0]) * 100   # base = 100
        
        # Main price line
        fig.add_trace(go.Scatter(
            x=indexed.index,
            y=indexed.values,
            name=data['name'].split()[0],
            line=dict(color=color, width=2),
            hovertemplate=f'<b>{data["name"]}</b><br>Indexed: %{{y:.1f}}<br>Date: %{{x|%d %b}}<extra></extra>'
        ))
        
        # 30-day MA (dashed)
        ma = indexed.rolling(min(10, len(indexed))).mean()
        fig.add_trace(go.Scatter(
            x=ma.index, y=ma.values,
            name=f'{data["name"].split()[0]} MA',
            line=dict(color=color, width=1, dash='dot'),
            opacity=0.4,
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Baseline at 100
    fig.add_hline(y=100, line_dash='dash', line_color=GRAY, line_width=0.8)
    
    fig.update_layout(
        title=dict(text='Price Performance — Indexed to 100', 
                  font=dict(color=GOLD, size=13), x=0),
        plot_bgcolor=NAVY2, paper_bgcolor=NAVY,
        font=dict(color=WHITE, family='sans-serif'),
        xaxis=dict(gridcolor='#223366', showgrid=True, color=GRAY),
        yaxis=dict(gridcolor='#223366', showgrid=True, color=GRAY,
                  title='Indexed Price (Base=100)'),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=10)),
        margin=dict(l=10, r=10, t=40, b=10),
        height=300,
        hovermode='x unified',
    )
    return fig

def make_returns_chart(stocks_data):
    """Horizontal bar chart: 1d, 7d, 30d returns per stock."""
    tickers = list(stocks_data.keys())
    names   = [d['name'].split()[0] for d in stocks_data.values()]
    
    fig = go.Figure()
    
    periods = [
        ('daily_chg',   '1 Day',  GOLD),
        ('weekly_chg',  '7 Days', '#4A9EFF'),
        ('monthly_chg', '30 Days','#9B59B6'),
    ]
    
    for key, label, color in periods:
        values = [stocks_data[t][key] for t in tickers]
        fig.add_trace(go.Bar(
            name=label,
            x=names,
            y=values,
            marker_color=[GREEN if v >= 0 else RED for v in values],
            marker_opacity=0.9 if key == 'daily_chg' else 0.6,
            text=[f'{v:+.1f}%' for v in values],
            textposition='outside',
            textfont=dict(size=9, color=WHITE),
        ))
        break   # show only 1 period at a time for clarity
    
    # Add buttons to switch between periods
    fig.update_layout(
        updatemenus=[dict(
            type='buttons',
            direction='right',
            x=0, y=1.15,
            buttons=[
                dict(label='1 Day',
                     method='update',
                     args=[{'y': [[stocks_data[t]['daily_chg'] for t in tickers]],
                             'marker.color': [[GREEN if stocks_data[t]['daily_chg'] >= 0 else RED for t in tickers]],
                             'text': [[f'{stocks_data[t]["daily_chg"]:+.1f}%' for t in tickers]]}]),
                dict(label='7 Days',
                     method='update',
                     args=[{'y': [[stocks_data[t]['weekly_chg'] for t in tickers]],
                             'marker.color': [[GREEN if stocks_data[t]['weekly_chg'] >= 0 else RED for t in tickers]],
                             'text': [[f'{stocks_data[t]["weekly_chg"]:+.1f}%' for t in tickers]]}]),
                dict(label='30 Days',
                     method='update',
                     args=[{'y': [[stocks_data[t]['monthly_chg'] for t in tickers]],
                             'marker.color': [[GREEN if stocks_data[t]['monthly_chg'] >= 0 else RED for t in tickers]],
                             'text': [[f'{stocks_data[t]["monthly_chg"]:+.1f}%' for t in tickers]]}]),
            ],
            bgcolor=NAVY2, bordercolor=GOLD, font=dict(color=WHITE),
        )],
        title=dict(text='Returns by Period', font=dict(color=GOLD, size=13), x=0),
        plot_bgcolor=NAVY2, paper_bgcolor=NAVY,
        font=dict(color=WHITE),
        xaxis=dict(gridcolor='#223366', color=GRAY),
        yaxis=dict(gridcolor='#223366', color=GRAY, title='Return %',
                  zeroline=True, zerolinecolor=GRAY, zerolinewidth=1),
        margin=dict(l=10, r=10, t=70, b=10),
        height=300,
        showlegend=False,
    )
    return fig

def make_volatility_gauge(stocks_data):
    """Volatility comparison chart with risk zones."""
    names = [d['name'].split()[0] for d in stocks_data.values()]
    vols  = [d['volatility'] for d in stocks_data.values()]
    
    colors = [GREEN if v < 30 else (GOLD if v < 50 else RED) for v in vols]
    
    fig = go.Figure(go.Bar(
        x=names, y=vols,
        marker_color=colors,
        text=[f'{v:.0f}%' for v in vols],
        textposition='outside',
        textfont=dict(color=WHITE, size=9),
    ))
    
    # Risk zone lines
    fig.add_hline(y=30, line_dash='dash', line_color=GREEN,
                 line_width=1, annotation_text='Low risk threshold',
                 annotation_font_color=GREEN, annotation_font_size=9)
    fig.add_hline(y=50, line_dash='dash', line_color=RED,
                 line_width=1, annotation_text='High risk threshold',
                 annotation_font_color=RED, annotation_font_size=9)
    
    fig.update_layout(
        title=dict(text='Annualised Volatility', font=dict(color=GOLD, size=13), x=0),
        plot_bgcolor=NAVY2, paper_bgcolor=NAVY,
        font=dict(color=WHITE),
        xaxis=dict(color=GRAY),
        yaxis=dict(gridcolor='#223366', color=GRAY, title='Volatility %'),
        margin=dict(l=10, r=10, t=40, b=10),
        height=280,
        showlegend=False,
    )
    return fig

# ═══════════════════════════════════════════════════════════
# MAIN APP LAYOUT
# ═══════════════════════════════════════════════════════════
def main():
    
    # ── HEADER ─────────────────────────────────────────────
    st.markdown(f"""
    <div class="header-box">
        <div class="header-title">📊 DAILY MARKET BRIEF</div>
        <div class="header-sub">
            Live B3 Data  ·  {datetime.now().strftime('%A, %d %B %Y  ·  %H:%M')} BRT  ·  
            Bruno Imbiriba Campello  ·  UFRJ Finance
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ── SIDEBAR ────────────────────────────────────────────
    with st.sidebar:
        st.markdown(f'<div class="section-header">⚙ SETTINGS</div>', unsafe_allow_html=True)
        
        period = st.selectbox(
            'Data Period',
            options=['1mo', '3mo', '6mo', '1y'],
            index=0,
            help='Historical period for trend charts'
        )
        
        st.markdown('<div class="section-header">📌 STOCKS</div>', unsafe_allow_html=True)
        
        # Let user toggle which stocks to show
        selected = {}
        for ticker, name in DEFAULT_STOCKS.items():
            if st.checkbox(name, value=True, key=ticker):
                selected[ticker] = name
        
        st.markdown('<div class="section-header">ℹ ABOUT</div>', unsafe_allow_html=True)
        st.markdown("""
        <small style='color:#888'>
        Data via Yahoo Finance API.<br>
        Refreshes every 15 minutes.<br>
        Not investment advice.<br><br>
        <a href='https://github.com/BrunoImbiribaCampello/brazil-equity-research' 
           style='color:#C9A84C'>📂 GitHub Portfolio</a>
        </small>
        """, unsafe_allow_html=True)
    
    # ── FETCH DATA ─────────────────────────────────────────
    with st.spinner('Fetching live market data...'):
        stocks_data  = {}
        indices_data = {}
        fx_data      = {}
        
        for ticker in selected:
            data = fetch_stock_data(ticker, period)
            if data:
                data['name'] = DEFAULT_STOCKS[ticker]
                stocks_data[ticker] = data
        
        for ticker, name in INDICES.items():
            data = fetch_stock_data(ticker, '5d')
            if data:
                data['name'] = name
                indices_data[ticker] = data
        
        for ticker, name in FX_PAIRS.items():
            data = fetch_stock_data(ticker, '5d')
            if data:
                data['name'] = name
                fx_data[ticker] = data
    
    if not stocks_data:
        st.error('Could not load market data. Please refresh.')
        return
    
    # ── GLOBAL MARKETS BAR ─────────────────────────────────
    st.markdown('<div class="section-header">🌍 GLOBAL MARKETS</div>', unsafe_allow_html=True)
    
    all_global = {**indices_data, **fx_data}
    cols = st.columns(len(all_global))
    
    for col, (ticker, data) in zip(cols, all_global.items()):
        chg   = data['daily_chg']
        arrow = '▲' if chg >= 0 else '▼'
        col.metric(
            label=data['name'],
            value=f"{data['price']:.2f}",
            delta=f"{arrow} {chg:+.2f}%",
        )
    
    # ── STOCK METRICS ROW ──────────────────────────────────
    st.markdown('<div class="section-header">📈 B3 EQUITIES — TODAY</div>', unsafe_allow_html=True)
    
    cols = st.columns(len(stocks_data))
    for col, (ticker, data) in zip(cols, stocks_data.items()):
        chg   = data['daily_chg']
        arrow = '▲' if chg >= 0 else '▼'
        col.metric(
            label=f"{ticker.replace('.SA','')} · {data['name'].split()[0]}",
            value=f"R${data['price']:.2f}",
            delta=f"{arrow} {chg:+.2f}%",
        )
    
    # ── CHARTS ─────────────────────────────────────────────
    st.markdown('<div class="section-header">📊 CHARTS</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.plotly_chart(
            make_price_chart(stocks_data, list(selected.keys())),
            use_container_width=True
        )
    
    with col2:
        st.plotly_chart(
            make_returns_chart(stocks_data),
            use_container_width=True
        )
    
    col3, col4 = st.columns([2, 3])
    
    with col3:
        st.plotly_chart(
            make_volatility_gauge(stocks_data),
            use_container_width=True
        )
    
    with col4:
        # ── DETAIL TABLE ───────────────────────────────────
        st.markdown('<div class="section-header">📋 FULL DATA TABLE</div>', unsafe_allow_html=True)
        
        table_rows = []
        for ticker, data in stocks_data.items():
            def fmt(v):
                arrow = '▲' if v >= 0 else '▼'
                return f'{arrow} {v:+.2f}%'
            
            table_rows.append({
                'Ticker':    ticker.replace('.SA', ''),
                'Company':   data['name'],
                'Price':     f"R${data['price']:.2f}",
                '1 Day':     fmt(data['daily_chg']),
                '7 Days':    fmt(data['weekly_chg']),
                '30 Days':   fmt(data['monthly_chg']),
                'Volatility':f"{data['volatility']:.1f}%",
                '52W High':  f"R${data['high_52w']:.2f}",
                '52W Low':   f"R${data['low_52w']:.2f}",
            })
        
        df = pd.DataFrame(table_rows)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            height=260,
        )
    
    # ── LAST UPDATED ───────────────────────────────────────
    st.markdown(f"""
    <p style='color:#555; font-size:11px; text-align:center; margin-top:16px;'>
        Data from Yahoo Finance API · Auto-refreshes every 15 min · 
        Not investment advice · 
        <a href='https://github.com/BrunoImbiribaCampello' style='color:#C9A84C'>
        github.com/BrunoImbiribaCampello</a>
    </p>
    """, unsafe_allow_html=True)

# ── ENTRY POINT ────────────────────────────────────────────
if __name__ == '__main__':
    main()
