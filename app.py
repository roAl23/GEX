import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
from scipy.stats import norm
import plotly.graph_objects as go

st.set_page_config(page_title="Deribit GEX Pro Dashboard", layout="wide")

st.title("📈 BTC TradingView & Options GEX/DEX Analytics Suite")

# --- BLACK-SCHOLES BERECHNUNG FÜR GAMMA & DELTA ---
def calculate_greeks(spot, strike, t_years, iv, option_type):
    if t_years <= 0 or iv <= 0 or spot <= 0 or strike <= 0:
        return 0.0, 0.0
    
    d1 = (np.log(spot / strike) + (0.5 * iv**2) * t_years) / (iv * np.sqrt(t_years))
    gamma = norm.pdf(d1) / (spot * iv * np.sqrt(t_years))
    
    if option_type == 'Call':
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1.0
        
    return gamma, delta

# 1. Spot Price
@st.cache_data(ttl=15)
def get_btc_spot():
    url = "https://www.deribit.com/api/v2/public/get_index_price?index_name=btc_usd"
    res = requests.get(url).json()
    return res['result']['index_price']

# 2. OHLC Kerzen von Deribit
@st.cache_data(ttl=60)
def get_btc_candles(resolution="60"):
    end_time = int(time.time() * 1000)
    start_time = end_time - (72 * 60 * 60 * 1000)
    url = f"https://www.deribit.com/api/v2/public/get_tradingview_chart_data?instrument_name=BTC-PERPETUAL&start_timestamp={start_time}&end_timestamp={end_time}&resolution={resolution}"
    res = requests.get(url).json()['result']
    
    df = pd.DataFrame({
        'ticks': pd.to_datetime(res['ticks'], unit='ms'),
        'open': res['open'],
        'high': res['high'],
        'low': res['low'],
        'close': res['close'],
        'volume': res['volume']
    })
    return df

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

    # Preprocessing
    df_raw['expiration_str'] = df_raw['instrument_name'].apply(lambda x: x.split('-')[1])
    df_raw['strike'] = df_raw['instrument_name'].apply(lambda x: float(x.split('-')[2]))
    df_raw['type'] = df_raw['instrument_name'].apply(lambda x: 'Call' if x.endswith('-C') else 'Put')

    for col in ['open_interest', 'volume', 'mark_iv']:
        if col in df_raw.columns:
            df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce').fillna(0.0)

    # Restlaufzeit in Jahren berechnen
    current_time = time.time()
    def parse_expiration_years(exp_str):
        try:
            exp_date = pd.to_datetime(exp_str, format='%d%b%y')
            exp_timestamp = exp_date.timestamp() + (8 * 3600)
            t_seconds = max(exp_timestamp - current_time, 3600)
            return t_seconds / (365.25 * 84600)
        except:
            return 1.0 / 365.25

    df_raw['t_years'] = df_raw['expiration_str'].apply(parse_expiration_years)

    # Greeks berechnen
    gammas, deltas = [], []
    for idx, row in df_raw.iterrows():
        g, d = calculate_greeks(
            spot=spot,
            strike=row['strike'],
            t_years=row['t_years'],
            iv=row['mark_iv'] / 100.0,
            option_type=row['type']
        )
        gammas.append(g)
        deltas.append(d)

    df_raw['gamma'] = gammas
    df_raw['delta'] = deltas

    # SIDEBAR FILTER
    st.sidebar.header("🗓️ Filter Options")
    expirations = sorted(df_raw['expiration_str'].unique().tolist())
    selected_exp = st.sidebar.selectbox("Select Expiration Date", ["ALL (Aggregated)"] + expirations, index=0)

    if selected_exp != "ALL (Aggregated)":
        df = df_raw[df_raw['expiration_str'] == selected_exp].copy()
    else:
        df = df_raw.copy()

    # Kennzahlen Berechnungen
    df['gex_oi'] = np.where(df['type'] == 'Call', df['gamma'] * df['open_interest'] * spot, -df['gamma'] * df['open_interest'] * spot) / 1e6
    df['dex'] = df['delta'] * df['open_interest'] * spot / 1e6

    zoom_margin = 3500
    y_min, y_max = spot - zoom_margin, spot + zoom_margin
    df_filtered = df[(df['strike'] >= y_min) & (df['strike'] <= y_max)].copy()

    summary = df_filtered.groupby('strike').agg({'gex_oi': 'sum', 'dex': 'sum', 'open_interest': 'sum'}).reset_index()
    summary_sorted = summary.sort_values('strike').copy()
    summary_sorted['cum_gex'] = summary_sorted['gex_oi'].cumsum()
    zero_crossings = summary_sorted[np.sign(summary_sorted['cum_gex']).diff() != 0]
    gamma_flip = zero_crossings['strike'].iloc[0] if len(zero_crossings) > 0 else spot

    call_wall = df_filtered[df_filtered['type'] == 'Call'].groupby('strike')['gex_oi'].sum().idxmax() if not df_filtered.empty else spot
    put_wall = df_filtered[df_filtered['type'] == 'Put'].groupby('strike')['gex_oi'].sum().abs().idxmax() if not df_filtered.empty else spot

    # ATM IV & Expected Move (1 SD / 2 SD)
    atm_strike = summary_sorted.iloc[(summary_sorted['strike'] - spot).abs().argsort()[:1]]['strike'].values[0] if not summary_sorted.empty else spot
    matching_iv = df_filtered[df_filtered['strike'] == atm_strike]['mark_iv'].mean()
    atm_iv = (matching_iv / 100.0) if not np.isnan(matching_iv) and matching_iv > 0 else 0.55

    exp_move_1sd = spot * atm_iv * np.sqrt(1 / 365.0)
    min_upper, min_lower = spot + exp_move_1sd, spot - exp_move_1sd
    max_upper, max_lower = spot + (2 * exp_move_1sd), spot - (2 * exp_move_1sd)

    # Top Metrics Header
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("BTC Spot", f"${spot:,.1f}")
    m2.metric("Gamma Flip", f"${gamma_flip:,.0f}")
    m3.metric("Call Wall", f"${call_wall:,.0f}")
    m4.metric("Put Wall", f"${put_wall:,.0f}")
    m5.metric("±1 SD Move", f"±${exp_move_1sd:,.0f}")

    st.markdown("---")

    # LAYOUT: 3 Spalten (Kerzenchart | GEX Profile | DEX Profile)
    col_chart, col_gex, col_dex = st.columns([0.48, 0.26, 0.26])

    # Gemeinsame Linien-Funktion für den Kerzenchart
    def add_standard_lines(fig):
        lines = [
            (spot, "SPOT", "white", "solid"),
            (gamma_flip, "GAMMA FLIP", "yellow", "solid"),
            (min_upper, "+1 SD", "orange", "dash"),
            (min_lower, "-1 SD", "orange", "dash"),
            (max_upper, "+2 SD", "fuchsia", "dash"),
            (max_lower, "-2 SD", "fuchsia", "dash"),
            (call_wall, "Call Wall", "red", "dot"),
            (put_wall, "Put Wall", "green", "dot")
        ]
        for price, name, color, style in lines:
            fig.add_hline(y=price, line_dash=style, line_color=color, annotation_text=f"{name} ({price:,.0f})")

    # 1. Kerzenchart (Linke Spalte)
    with col_chart:
        st.subheader("📊 BTC Perpetual 1H & Key Levels")
        fig_candle = go.Figure()
        fig_candle.add_trace(go.Candlestick(
            x=candles['ticks'], open=candles['open'], high=candles['high'], low=candles['low'], close=candles['close'], name="BTC"
        ))
        
        # VPVR Volumenprofil als Overlay
        nbins = 25
        price_bins = pd.cut(candles['close'], bins=nbins)
        vpvr = candles.groupby(price_bins, observed=False)['volume'].sum().reset_index()
        vpvr['bin_mid'] = vpvr['close'].apply(lambda x: x.mid)
        
        fig_candle.add_trace(go.Bar(
            x=vpvr['volume'], y=vpvr['bin_mid'], orientation='h', name="Volume", marker_color='rgba(255,255,255,0.1)', width=150
        ))

        add_standard_lines(fig_candle)
        fig_candle.update_layout(
            template="plotly_dark", height=700, showlegend=False, xaxis_rangeslider_visible=False,
            yaxis=dict(range=[y_min, y_max], tickformat="$,.0f", title="Preis ($)")
        )
        st.plotly_chart(fig_candle, use_container_width=True)

    # 2. GEX Profile (Mittlere Spalte)
    with col_gex:
        st.subheader(f"🟠 GEX ({selected_exp})")
        colors_gex = ['#00E676' if x >= 0 else '#FF5252' for x in summary['gex_oi']]
        
        fig_gex = go.Figure()
        fig_gex.add_trace(go.Bar(x=summary['gex_oi'], y=summary['strike'], orientation='h', marker_color=colors_gex, width=200))
        add_standard_lines(fig_gex)
        
        fig_gex.update_layout(
            template="plotly_dark", height=700, showlegend=False,
            yaxis=dict(range=[y_min, y_max], tickformat="$,.0f", showticklabels=False),
            xaxis=dict(title="GEX ($M)")
        )
        st.plotly_chart(fig_gex, use_container_width=True)

    # 3. DEX Profile (Rechte Spalte)
    with col_dex:
        st.subheader("🟣 DEX Profile")
        fig_dex = go.Figure()
        fig_dex.add_trace(go.Bar(x=summary['dex'], y=summary['strike'], orientation='h', marker_color='#AB47BC', width=200))
        add_standard_lines(fig_dex)
        
        fig_dex.update_layout(
            template="plotly_dark", height=700, showlegend=False,
            yaxis=dict(range=[y_min, y_max], tickformat="$,.0f", showticklabels=False),
            xaxis=dict(title="DEX ($M)")
        )
        st.plotly_chart(fig_dex, use_container_width=True)

except Exception as e:
    st.error(f"Fehler beim Laden des Dashboards: {e}")
