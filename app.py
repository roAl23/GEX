import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

st.set_page_config(page_title="Deribit GEX Pro Dashboard", layout="wide")

st.title("📈 BTC Live Candlestick, Volume Profile & Options Analytics")

# 1. Spot Price
@st.cache_data(ttl=15)
def get_btc_spot():
    url = "https://www.deribit.com/api/v2/public/get_index_price?index_name=btc_usd"
    res = requests.get(url).json()
    return res['result']['index_price']

# 2. OHLC Kerzendaten von Deribit abrufen
@st.cache_data(ttl=60)
def get_btc_candles(resolution="60"):
    end_time = int(time.time() * 1000)
    start_time = end_time - (48 * 60 * 60 * 1000) # Letzte 48h für besseres VPVR
    url = f"https://www.deribit.com/api/v2/public/get_tradingview_chart_data?instrument_name=BTC-PERPETUAL&start_timestamp={start_time}&end_timestamp={end_time}&resolution={resolution}"
    res = requests.get(url).json()['result']
    df_candles = pd.DataFrame({
        'ticks': pd.to_datetime(res['ticks'], unit='ms'),
        'open': res['open'],
        'high': res['high'],
        'low': res['low'],
        'close': res['close'],
        'volume': res['volume']
    })
    return df_candles

# 3. Optionsdaten abrufen
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
        df_raw[col] = pd.to_numeric(df_raw.get(col, 0), errors='coerce').fillna(0)

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
    if np.isnan(atm_iv) or atm_iv == 0: atm_iv = 0.55

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

    # Subplots Layout: Col 1 = Kerzen + VPVR Overlay, Col 2 = GEX, Col 3 = DEX
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

    # 1b. Volume Profile Overlay (VPVR) im Kerzenchart auf zweiter X-Achse
    fig.add_trace(
        go.Bar(
            x=vpvr['volume'],
            y=vpvr['bin_mid'],
            orientation='h',
            name="Volume Profile",
            marker_color='rgba(255, 255, 255, 0.15)',
            opacity=0.4
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
            marker_color=colors_gex
        ), row=1, col=2
    )

    # 3. DEX Profile
    fig.add_trace(
        go.Bar(
            x=summary['dex'], 
            y=summary['strike'], 
            orientation='h', 
            name="DEX", 
            marker_color='#AB47BC'
        ), row=1, col=3
    )

    # Preis-Levels über alle Charts legen
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

    # Styling & Achsen
    fig.update_layout(
        template="plotly_dark",
        height=800,
        showlegend=False,
        xaxis_rangeslider_visible=False,
        yaxis=dict(
            title="BTC Preis ($)",
            tickformat="$,.0f",
            dtick=500
        )
    )

    st.plotly_chart(fig, width="stretch")

except Exception as e:
    st.error(f"Fehler beim Laden der Daten: {e}")        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        else:
            df[col] = 0.0

    # Berechnungen pro Strike
    # GEX = Gamma * OI * Spot (Calls positiv, Puts negativ)
    df['gex_oi'] = np.where(df['type'] == 'Call', 
                            df['gamma'] * df['open_interest'] * spot, 
                            -df['gamma'] * df['open_interest'] * spot) / 1e6

    # GEX Volume = Gamma * Volume * Spot
    df['gex_vol'] = np.where(df['type'] == 'Call', 
                             df['gamma'] * df['volume'] * spot, 
                             -df['gamma'] * df['volume'] * spot) / 1e6

    # DEX = Delta * OI * Spot ($ Millions)
    df['dex'] = df['delta'] * df['open_interest'] * spot / 1e6

    # Aggregierung pro Strike um den aktuellen Spot (+/- $15.000)
    df_filtered = df[(df['strike'] >= spot - 15000) & (df['strike'] <= spot + 15000)].copy()
    
    summary = df_filtered.groupby('strike').agg({
        'gex_oi': 'sum',
        'gex_vol': 'sum',
        'dex': 'sum',
        'open_interest': 'sum',
        'volume': 'sum'
    }).reset_index()

    # Core Key Levels berechnen
    call_wall = df_filtered[df_filtered['type'] == 'Call'].groupby('strike')['gex_oi'].sum().idxmax()
    put_wall = df_filtered[df_filtered['type'] == 'Put'].groupby('strike')['gex_oi'].sum().abs().idxmax()
    
    # Cumulatives Gamma zur Ermittlung des Gamma Flips
    summary_sorted = summary.sort_values('strike').copy()
    summary_sorted['cum_gex'] = summary_sorted['gex_oi'].cumsum()
    
    # Gamma Flip (Wo wechselt das kumulierte GEX das Vorzeichen?)
    zero_crossings = summary_sorted[np.sign(summary_sorted['cum_gex']).diff() != 0]
    gamma_flip = zero_crossings['strike'].iloc[0] if len(zero_crossings) > 0 else spot

    # IV & Daily Move Calculation (1 SD / 2 SD)
    atm_strike = summary_sorted.iloc[(summary_sorted['strike'] - spot).abs().argsort()[:1]]['strike'].values[0]
    atm_iv = df_filtered[df_filtered['strike'] == atm_strike]['mark_iv'].mean() / 100.0
    if np.isnan(atm_iv) or atm_iv == 0:
        atm_iv = 0.55  # Fallback IV 55%

    exp_move_1sd = spot * atm_iv * np.sqrt(1 / 365.0)
    min_upper = spot + exp_move_1sd
    min_lower = spot - exp_move_1sd
    max_upper = spot + (2 * exp_move_1sd)
    max_lower = spot - (2 * exp_move_1sd)

    # --- TOP METRICS HEADER ---
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("BTC Spot Price", f"${spot:,.1f}")
    m2.metric("Gamma Flip", f"${gamma_flip:,.0f}")
    m3.metric("Call Wall", f"${call_wall:,.0f}")
    m4.metric("Put Wall", f"${put_wall:,.0f}")
    m5.metric("ATM IV (Daily)", f"{atm_iv*100:.1f}%")

    st.markdown("---")

    # --- SIDEBAR CONTROL & DAILY MOVES ---
    st.sidebar.header("🎯 Key Levels Overview")
    st.sidebar.write(f"**+2 SD Max Upper:** ${max_upper:,.0f}")
    st.sidebar.write(f"**+1 SD Min Upper:** ${min_upper:,.0f}")
    st.sidebar.write(f"**Gamma Flip:** ${gamma_flip:,.0f}")
    st.sidebar.write(f"**-1 SD Min Lower:** ${min_lower:,.0f}")
    st.sidebar.write(f"**-2 SD Max Lower:** ${max_lower:,.0f}")
    
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Display Options")
    show_gex = st.sidebar.checkbox("Show GEX Normal (Orange)", value=True)
    show_gex_vol = st.sidebar.checkbox("Show GEX Volume (Aqua)", value=True)
    show_dex = st.sidebar.checkbox("Show DEX (Purple)", value=True)
    show_oi = st.sidebar.checkbox("Show Open Interest (Yellow)", value=True)

    # --- CHARTS (PROFILE HISTOGRAMS SIDE-BY-SIDE) ---
    fig = make_subplots(rows=1, cols=4, shared_yaxes=True, 
                        subplot_titles=("GEX Normal ($M)", "GEX Volume ($M)", "DEX Exposure ($M)", "Open Interest"))

    # 1. GEX Normal
    if show_gex:
        colors_gex = ['#FF9900' if x >= 0 else '#FF3333' for x in summary['gex_oi']]
        fig.add_trace(go.Bar(x=summary['gex_oi'], y=summary['strike'], orientation='h', name="GEX OI", marker_color=colors_gex), row=1, col=1)

    # 2. GEX Volume
    if show_gex_vol:
        fig.add_trace(go.Bar(x=summary['gex_vol'], y=summary['strike'], orientation='h', name="GEX Vol", marker_color='cyan'), row=1, col=2)

    # 3. DEX
    if show_dex:
        fig.add_trace(go.Bar(x=summary['dex'], y=summary['strike'], orientation='h', name="DEX", marker_color='purple'), row=1, col=3)

    # 4. OI
    if show_oi:
        fig.add_trace(go.Bar(x=summary['open_interest'], y=summary['strike'], orientation='h', name="OI", marker_color='gold'), row=1, col=4)

    # Lines across all plots
    for c in range(1, 5):
        fig.add_hline(y=spot, line_dash="solid", line_color="white", annotation_text="SPOT", row=1, col=c)
        fig.add_hline(y=gamma_flip, line_dash="solid", line_color="yellow", annotation_text="FLIP", row=1, col=c)
        fig.add_hline(y=min_upper, line_dash="dash", line_color="orange", annotation_text="+1 SD", row=1, col=c)
        fig.add_hline(y=min_lower, line_dash="dash", line_color="orange", annotation_text="-1 SD", row=1, col=c)
        fig.add_hline(y=max_upper, line_dash="dash", line_color="fuchsia", annotation_text="+2 SD", row=1, col=c)
        fig.add_hline(y=max_lower, line_dash="dash", line_color="fuchsia", annotation_text="-2 SD", row=1, col=c)

    fig.update_layout(template="plotly_dark", height=750, showlegend=False)
    
    # Aktuelle API-Syntax verwenden
    st.plotly_chart(fig, width="stretch")

except Exception as e:
    st.error(f"Fehler bei der Berechnung der Optionsdaten: {e}")
