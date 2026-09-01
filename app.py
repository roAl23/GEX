import streamlit as st
import streamlit.components.v1 as components
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time

st.set_page_config(page_title="Deribit GEX Pro Dashboard", layout="wide")

st.title("📈 BTC TradingView & Options GEX/DEX Analytics")

# 1. Spot Price
@st.cache_data(ttl=15)
def get_btc_spot():
    url = "https://www.deribit.com/api/v2/public/get_index_price?index_name=btc_usd"
    res = requests.get(url).json()
    return res['result']['index_price']

# 2. Optionsdaten abrufen
@st.cache_data(ttl=30)
def get_option_data():
    url = "https://www.deribit.com/api/v2/public/get_book_summary_by_currency?currency=BTC&kind=option"
    res = requests.get(url).json()['result']
    return pd.DataFrame(res)

try:
    spot = get_btc_spot()
    df_raw = get_option_data()

    # Preprocessing
    df_raw['expiration'] = df_raw['instrument_name'].apply(lambda x: x.split('-')[1])
    df_raw['strike'] = df_raw['instrument_name'].apply(lambda x: float(x.split('-')[2]))
    df_raw['type'] = df_raw['instrument_name'].apply(lambda x: 'Call' if x.endswith('-C') else 'Put')

    for col in ['open_interest', 'volume', 'gamma', 'delta', 'mark_iv']:
        if col in df_raw.columns:
            df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce').fillna(0)
        else:
            df_raw[col] = 0.0

    # SIDEBAR FILTER
    st.sidebar.header("🗓️ Filter Options")
    expirations = sorted(df_raw['expiration'].unique().tolist())
    selected_exp = st.sidebar.selectbox("Select Expiration Date", ["ALL (Aggregated)"] + expirations)

    if selected_exp != "ALL (Aggregated)":
        df = df_raw[df_raw['expiration'] == selected_exp].copy()
    else:
        df = df_raw.copy()

    # Berechnungen
    df['gex_oi'] = np.where(df['type'] == 'Call', df['gamma'] * df['open_interest'] * spot, -df['gamma'] * df['open_interest'] * spot) / 1e6
    df['dex'] = df['delta'] * df['open_interest'] * spot / 1e6

    # Fokussierung um den Preis (+/- $4.000)
    zoom_margin = 4000
    df_filtered = df[(df['strike'] >= spot - zoom_margin) & (df['strike'] <= spot + zoom_margin)].copy()

    summary = df_filtered.groupby('strike').agg({
        'gex_oi': 'sum',
        'dex': 'sum',
        'open_interest': 'sum'
    }).reset_index()

    # Key Levels
    call_wall = df_filtered[df_filtered['type'] == 'Call'].groupby('strike')['gex_oi'].sum().idxmax() if not df_filtered.empty else spot
    put_wall = df_filtered[df_filtered['type'] == 'Put'].groupby('strike')['gex_oi'].sum().abs().idxmax() if not df_filtered.empty else spot

    summary_sorted = summary.sort_values('strike').copy()
    summary_sorted['cum_gex'] = summary_sorted['gex_oi'].cumsum()
    zero_crossings = summary_sorted[np.sign(summary_sorted['cum_gex']).diff() != 0]
    gamma_flip = zero_crossings['strike'].iloc[0] if len(zero_crossings) > 0 else spot

    atm_strike = summary_sorted.iloc[(summary_sorted['strike'] - spot).abs().argsort()[:1]]['strike'].values[0] if not summary_sorted.empty else spot
    atm_iv = df_filtered[df_filtered['strike'] == atm_strike]['mark_iv'].mean() / 100.0 if not df_filtered.empty else 0.55
    if np.isnan(atm_iv) or atm_iv == 0: atm_iv = 0.55

    # Top Header Metrics
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("BTC Spot Price", f"${spot:,.1f}")
    m2.metric("Gamma Flip", f"${gamma_flip:,.0f}")
    m3.metric("Call Wall", f"${call_wall:,.0f}")
    m4.metric("Put Wall", f"${put_wall:,.0f}")
    m5.metric("ATM IV (Daily)", f"{atm_iv*100:.1f}%")

    st.markdown("---")

    # GRID LAYOUT: Links TradingView, Rechts GEX & DEX Profile
    col_tv, col_gex, col_dex = st.columns([0.5, 0.25, 0.25])

    # SPALTE 1: Native TradingView Widget
    with col_tv:
        st.subheader("📺 TradingView Chart")
        tv_widget_code = """
        <div class="tradingview-widget-container" style="height:650px;width:100%">
          <div id="tradingview_chart" style="height:650px;width:100%"></div>
          <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
          <script type="text/javascript">
          new TradingView.widget(
          {
            "autosize": true,
            "symbol": "DERIBIT:BTCUSDT.P",
            "interval": "60",
            "timezone": "Etc/UTC",
            "theme": "dark",
            "style": "1",
            "locale": "de_DE",
            "toolbar_bg": "#f1f3f6",
            "enable_publishing": false,
            "allow_symbol_change": true,
            "container_id": "tradingview_chart"
          }
          );
          </script>
        </div>
        """
        components.html(tv_widget_code, height=660)

    # SPALTE 2: GEX Profile
    with col_gex:
        st.subheader(f"📊 GEX ({selected_exp})")
        colors_gex = ['#00E676' if x >= 0 else '#FF5252' for x in summary['gex_oi']]
        
        fig_gex = go.Figure()
        fig_gex.add_trace(go.Bar(
            x=summary['gex_oi'],
            y=summary['strike'],
            orientation='h',
            marker_color=colors_gex
        ))
        
        # Spot Line
        fig_gex.add_hline(y=spot, line_dash="solid", line_color="white", annotation_text="SPOT")
        fig_gex.add_hline(y=gamma_flip, line_dash="solid", line_color="yellow", annotation_text="FLIP")

        fig_gex.update_layout(
            template="plotly_dark",
            height=650,
            showlegend=False,
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis=dict(range=[spot - 3500, spot + 3500], tickformat="$,.0f", title="Strike ($)")
        )
        st.plotly_chart(fig_gex, use_container_width=True)

    # SPALTE 3: DEX Profile
    with col_dex:
        st.subheader("🔮 DEX Profile")
        
        fig_dex = go.Figure()
        fig_dex.add_trace(go.Bar(
            x=summary['dex'],
            y=summary['strike'],
            orientation='h',
            marker_color='#AB47BC'
        ))
        
        # Spot Line
        fig_dex.add_hline(y=spot, line_dash="solid", line_color="white", annotation_text="SPOT")

        fig_dex.update_layout(
            template="plotly_dark",
            height=650,
            showlegend=False,
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis=dict(range=[spot - 3500, spot + 3500], tickformat="$,.0f", title="")
        )
        st.plotly_chart(fig_dex, use_container_width=True)

except Exception as e:
    st.error(f"Fehler beim Laden der Daten: {e}")# 3. Optionsdaten abrufen
@st.cache_data(ttl=30)
def get_option_data():
    url = "https://www.deribit.com/api/v2/public/get_book_summary_by_currency?currency=BTC&kind=option"
    res = requests.get(url).json()['result']
    return pd.DataFrame(res)

try:
    spot = get_btc_spot()
    candles = get_btc_candles()
    df_raw = get_option_data()

    # Preprocessing
    df_raw['expiration'] = df_raw['instrument_name'].apply(lambda x: x.split('-')[1])
    df_raw['strike'] = df_raw['instrument_name'].apply(lambda x: float(x.split('-')[2]))
    df_raw['type'] = df_raw['instrument_name'].apply(lambda x: 'Call' if x.endswith('-C') else 'Put')

    for col in ['open_interest', 'volume', 'gamma', 'delta', 'mark_iv']:
        if col in df_raw.columns:
            df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce').fillna(0)
        else:
            df_raw[col] = 0.0

    # SIDEBAR FILTER
    st.sidebar.header("🗓️ Filter Options")
    expirations = sorted(df_raw['expiration'].unique().tolist())
    selected_exp = st.sidebar.selectbox("Select Expiration Date", ["ALL (Aggregated)"] + expirations)

    if selected_exp != "ALL (Aggregated)":
        df = df_raw[df_raw['expiration'] == selected_exp].copy()
    else:
        df = df_raw.copy()

    # Berechnungen
    df['gex_oi'] = np.where(df['type'] == 'Call', df['gamma'] * df['open_interest'] * spot, -df['gamma'] * df['open_interest'] * spot) / 1e6
    df['dex'] = df['delta'] * df['open_interest'] * spot / 1e6

    # Enge Fokussierung um den Preis (+/- $3.500)
    zoom_margin = 3500
    df_filtered = df[(df['strike'] >= spot - zoom_margin) & (df['strike'] <= spot + zoom_margin)].copy()

    summary = df_filtered.groupby('strike').agg({
        'gex_oi': 'sum',
        'dex': 'sum',
        'open_interest': 'sum'
    }).reset_index()

    # Levels
    call_wall = df_filtered[df_filtered['type'] == 'Call'].groupby('strike')['gex_oi'].sum().idxmax() if not df_filtered.empty else spot
    put_wall = df_filtered[df_filtered['type'] == 'Put'].groupby('strike')['gex_oi'].sum().abs().idxmax() if not df_filtered.empty else spot

    summary_sorted = summary.sort_values('strike').copy()
    summary_sorted['cum_gex'] = summary_sorted['gex_oi'].cumsum()
    zero_crossings = summary_sorted[np.sign(summary_sorted['cum_gex']).diff() != 0]
    gamma_flip = zero_crossings['strike'].iloc[0] if len(zero_crossings) > 0 else spot

    atm_strike = summary_sorted.iloc[(summary_sorted['strike'] - spot).abs().argsort()[:1]]['strike'].values[0] if not summary_sorted.empty else spot
    atm_iv = df_filtered[df_filtered['strike'] == atm_strike]['mark_iv'].mean() / 100.0 if not df_filtered.empty else 0.55
    if np.isnan(atm_iv) or atm_iv == 0: atm_iv = 0.55

    exp_move_1sd = spot * atm_iv * np.sqrt(1 / 365.0)
    min_upper, min_lower = spot + exp_move_1sd, spot - exp_move_1sd
    max_upper, max_lower = spot + (2 * exp_move_1sd), spot - (2 * exp_move_1sd)

    # VPVR
    nbins = 30
    price_bins = pd.cut(candles['close'], bins=nbins)
    vpvr = candles.groupby(price_bins, observed=False)['volume'].sum().reset_index()
    vpvr['bin_mid'] = vpvr['close'].apply(lambda x: x.mid)

    # Top Header Metrics
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("BTC Spot Price", f"${spot:,.1f}")
    m2.metric("Gamma Flip", f"${gamma_flip:,.0f}")
    m3.metric("Call Wall", f"${call_wall:,.0f}")
    m4.metric("Put Wall", f"${put_wall:,.0f}")
    m5.metric("ATM IV (Daily)", f"{atm_iv*100:.1f}%")

    st.markdown("---")

    # Layout Setup
    fig = make_subplots(
        rows=1, cols=3, 
        shared_yaxes=True,
        column_widths=[0.5, 0.25, 0.25],
        subplot_titles=("BTC Perpetual 1H (mit VPVR)", f"GEX Profile ({selected_exp})", "DEX Exposure")
    )

    # 1. Candlestick Chart
    fig.add_trace(
        go.Candlestick(
            x=candles['ticks'],
            open=candles['open'],
            high=candles['high'],
            low=candles['low'],
            close=candles['close'],
            name="BTC Price"
        ), row=1, col=1
    )

    # 1b. VPVR Overlay
    fig.add_trace(
        go.Bar(
            x=vpvr['volume'],
            y=vpvr['bin_mid'],
            orientation='h',
            name="Volume Profile",
            marker_color='rgba(255, 255, 255, 0.12)',
            width=200
        ), row=1, col=1
    )

    # 2. GEX Profile
    colors_gex = ['#00E676' if x >= 0 else '#FF5252' for x in summary['gex_oi']]
    fig.add_trace(
        go.Bar(
            x=summary['gex_oi'], 
            y=summary['strike'], 
            orientation='h', 
            name="GEX OI", 
            marker_color=colors_gex,
            width=300
        ), row=1, col=2
    )

    # 3. DEX Profile
    fig.add_trace(
        go.Bar(
            x=summary['dex'], 
            y=summary['strike'], 
            orientation='h', 
            name="DEX", 
            marker_color='#AB47BC',
            width=300
        ), row=1, col=3
    )

    # Levels
    levels = [
        (spot, "SPOT", "white", "solid"),
        (gamma_flip, "FLIP", "yellow", "solid"),
        (min_upper, "+1 SD", "orange", "dash"),
        (min_lower, "-1 SD", "orange", "dash"),
        (max_upper, "+2 SD", "fuchsia", "dash"),
        (max_lower, "-2 SD", "fuchsia", "dash")
    ]

    for lvl_price, lvl_name, lvl_color, lvl_style in levels:
        for c in range(1, 4):
            fig.add_hline(
                y=lvl_price, 
                line_dash=lvl_style, 
                line_color=lvl_color, 
                annotation_text=lvl_name if c == 1 else "", 
                row=1, col=c
            )

    # FIX FÜR DIESEN ERROR: Automatischen Zoom auf den aktuellen Spot-Preis begrenzen (+/- $3.000)
    y_min = spot - 3000
    y_max = spot + 3000

    fig.update_layout(
        template="plotly_dark",
        height=750,
        showlegend=False,
        xaxis_rangeslider_visible=False,
        yaxis=dict(
            title="BTC Preis ($)",
            range=[y_min, y_max],  # Erzwingt den Fokus auf die Relevanz-Zone
            tickformat="$,.0f"
        )
    )

    st.plotly_chart(fig, width="stretch")

except Exception as e:
    st.error(f"Fehler beim Laden der Daten: {e}")# 3. Optionsdaten abrufen
@st.cache_data(ttl=30)
def get_option_data():
    url = "https://www.deribit.com/api/v2/public/get_book_summary_by_currency?currency=BTC&kind=option"
    res = requests.get(url).json()['result']
    return pd.DataFrame(res)

try:
    spot = get_btc_spot()
    candles = get_btc_candles()
    df_raw = get_option_data()

    # Data Preprocessing & Parsing
    # Instrument Format: BTC-2SEP26-69000-C
    df_raw['expiration'] = df_raw['instrument_name'].apply(lambda x: x.split('-')[1])
    df_raw['strike'] = df_raw['instrument_name'].apply(lambda x: float(x.split('-')[2]))
    df_raw['type'] = df_raw['instrument_name'].apply(lambda x: 'Call' if x.endswith('-C') else 'Put')

    for col in ['open_interest', 'volume', 'gamma', 'delta', 'mark_iv']:
        if col in df_raw.columns:
            df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce').fillna(0)
        else:
            df_raw[col] = 0.0

    # --- SIDEBAR: EXPIRATION FILTER ---
    st.sidebar.header("🗓️ Filter Options")
    expirations = sorted(df_raw['expiration'].unique().tolist())
    selected_exp = st.sidebar.selectbox("Select Expiration Date", ["ALL (Aggregated)"] + expirations)

    if selected_exp != "ALL (Aggregated)":
        df = df_raw[df_raw['expiration'] == selected_exp].copy()
    else:
        df = df_raw.copy()

    # GEX & DEX Berechnungen
    df['gex_oi'] = np.where(df['type'] == 'Call', df['gamma'] * df['open_interest'] * spot, -df['gamma'] * df['open_interest'] * spot) / 1e6
    df['dex'] = df['delta'] * df['open_interest'] * spot / 1e6

    # Zoom-Filter um den Spot (+/- $8.000)
    df_filtered = df[(df['strike'] >= spot - 8000) & (df['strike'] <= spot + 8000)].copy()

    summary = df_filtered.groupby('strike').agg({
        'gex_oi': 'sum',
        'dex': 'sum',
        'open_interest': 'sum'
    }).reset_index()

    # Key Levels
    call_wall = df_filtered[df_filtered['type'] == 'Call'].groupby('strike')['gex_oi'].sum().idxmax() if not df_filtered.empty else spot
    put_wall = df_filtered[df_filtered['type'] == 'Put'].groupby('strike')['gex_oi'].sum().abs().idxmax() if not df_filtered.empty else spot

    summary_sorted = summary.sort_values('strike').copy()
    summary_sorted['cum_gex'] = summary_sorted['gex_oi'].cumsum()
    zero_crossings = summary_sorted[np.sign(summary_sorted['cum_gex']).diff() != 0]
    gamma_flip = zero_crossings['strike'].iloc[0] if len(zero_crossings) > 0 else spot

    atm_strike = summary_sorted.iloc[(summary_sorted['strike'] - spot).abs().argsort()[:1]]['strike'].values[0] if not summary_sorted.empty else spot
    atm_iv = df_filtered[df_filtered['strike'] == atm_strike]['mark_iv'].mean() / 100.0 if not df_filtered.empty else 0.55
    if np.isnan(atm_iv) or atm_iv == 0: 
        atm_iv = 0.55

    exp_move_1sd = spot * atm_iv * np.sqrt(1 / 365.0)
    min_upper, min_lower = spot + exp_move_1sd, spot - exp_move_1sd
    max_upper, max_lower = spot + (2 * exp_move_1sd), spot - (2 * exp_move_1sd)

    # VPVR (Volume Profile) aus Kerzendaten berechnen
    nbins = 40
    price_bins = pd.cut(candles['close'], bins=nbins)
    vpvr = candles.groupby(price_bins, observed=False)['volume'].sum().reset_index()
    vpvr['bin_mid'] = vpvr['close'].apply(lambda x: x.mid)

    # Top Metrics Header
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("BTC Spot Price", f"${spot:,.1f}")
    m2.metric("Gamma Flip", f"${gamma_flip:,.0f}")
    m3.metric("Call Wall", f"${call_wall:,.0f}")
    m4.metric("Put Wall", f"${put_wall:,.0f}")
    m5.metric("ATM IV (Daily)", f"{atm_iv*100:.1f}%")

    st.markdown("---")

   import streamlit.components.v1 as components

# TradingView Advanced Chart Widget
tv_widget = """
<!-- TradingView Widget BEGIN -->
<div class="tradingview-widget-container" style="height:100%;width:100%">
  <div id="tradingview_chart" style="height:600px;width:100%"></div>
  <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
  <script type="text/javascript">
  new TradingView.widget(
  {
    "autosize": true,
    "symbol": "DERIBIT:BTCUSDT.P",
    "interval": "60",
    "timezone": "Etc/UTC",
    "theme": "dark",
    "style": "1",
    "locale": "de_DE",
    "toolbar_bg": "#f1f3f6",
    "enable_publishing": false,
    "allow_symbol_change": true,
    "container_id": "tradingview_chart"
  }
  );
  </script>
</div>
<!-- TradingView Widget END -->
"""

# In Spalten aufteilen: Links TradingView, Rechts GEX & DEX Profile
col1, col2, col3 = st.columns([0.5, 0.25, 0.25])

with col1:
    st.subheader("BTC TradingView Chart")
    components.html(tv_widget, height=600)

with col2:
    st.subheader("GEX Profile")
    # Hier der Plotly Bar-Chart für GEX
    # st.plotly_chart(fig_gex, width="stretch")

with col3:
    st.subheader("DEX Profile")
    # Hier der Plotly Bar-Chart für DEX
    # st.plotly_chart(fig_dex, width="stretch")
