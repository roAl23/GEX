import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
from scipy.stats import norm
import plotly.graph_objects as go

st.set_page_config(page_title="Deribit Options Profile Engine Pro", layout="wide")

st.title("📊 Deribit Options Profile Engine + Min/Max Daily Move")

# --- BLACK-SCHOLES BERECHNUNG FÜR GREEKS ---
def calculate_greeks(spot, strike, t_years, iv, option_type):
    if t_years <= 0 or iv <= 0 or spot <= 0 or strike <= 0:
        return 0.0, 0.0, 0.5
    d1 = (np.log(spot / strike) + (0.5 * iv**2) * t_years) / (iv * np.sqrt(t_years))
    gamma = norm.pdf(d1) / (spot * iv * np.sqrt(t_years))
    if option_type == 'Call':
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1.0
    return gamma, delta, norm.cdf(d1)

# 1. Spot Price abrufen
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
    df_raw['expiration_str'] = df_raw['instrument_name'].apply(lambda x: x.split('-')[1])
    df_raw['strike'] = df_raw['instrument_name'].apply(lambda x: float(x.split('-')[2]))
    df_raw['type'] = df_raw['instrument_name'].apply(lambda x: 'Call' if x.endswith('-C') else 'Put')

    for col in ['open_interest', 'volume', 'mark_iv']:
        if col in df_raw.columns:
            df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce').fillna(0.0)

    # Restlaufzeit in Jahren
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
        g, d, _ = calculate_greeks(
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

    # ==========================================
    # SIDEBAR: KONTROLLEN & EINSTELLUNGEN
    # ==========================================
    st.sidebar.header("🎛️ Profile Selection")
    show_gex    = st.sidebar.checkbox("🟠 Show GEX Normal (Gamma OI)", value=True)
    show_gex_vol = st.sidebar.checkbox("🩵 Show GEX Volume (Orderflow)", value=True)
    show_dex    = st.sidebar.checkbox("🟣 Show DEX (Delta Exposure)", value=True)
    show_oi     = st.sidebar.checkbox("🟡 Show Open Interest (Contracts)", value=True)

    st.sidebar.header("🗓️ Expiry Filter")
    expirations = sorted(df_raw['expiration_str'].unique().tolist())
    selected_exp = st.sidebar.selectbox("Select Expiration Date", ["ALL (Aggregated)"] + expirations, index=0)

    st.sidebar.header("📐 Display Range")
    zoom_margin = st.sidebar.slider("Strike Zoom Range (+/- USD)", 1000, 10000, 4000, step=500)

    if selected_exp != "ALL (Aggregated)":
        df = df_raw[df_raw['expiration_str'] == selected_exp].copy()
    else:
        df = df_raw.copy()

    # Kennzahlen für Profile berechnen
    df['gex_normal'] = np.where(df['type'] == 'Call', df['gamma'] * df['open_interest'] * spot, -df['gamma'] * df['open_interest'] * spot) / 1e6
    df['gex_volume'] = np.where(df['type'] == 'Call', df['gamma'] * df['volume'] * spot, -df['gamma'] * df['volume'] * spot) / 1e6
    df['dex_val']    = df['delta'] * df['open_interest'] * spot / 1e6
    df['oi_val']     = df['open_interest']

    y_min, y_max = spot - zoom_margin, spot + zoom_margin
    df_filtered = df[(df['strike'] >= y_min) & (df['strike'] <= y_max)].copy()

    summary = df_filtered.groupby('strike').agg({
        'gex_normal': 'sum',
        'gex_volume': 'sum',
        'dex_val': 'sum',
        'oi_val': 'sum',
        'mark_iv': 'mean'
    }).reset_index()

    # Kern-Key-Levels berechnen
    summary_sorted = summary.sort_values('strike').copy()
    summary_sorted['cum_gex'] = summary_sorted['gex_normal'].cumsum()
    zero_crossings = summary_sorted[np.sign(summary_sorted['cum_gex']).diff() != 0]
    gamma_flip = zero_crossings['strike'].iloc[0] if len(zero_crossings) > 0 else spot

    call_wall = df_filtered[df_filtered['type'] == 'Call'].groupby('strike')['gex_normal'].sum().idxmax() if not df_filtered.empty else spot
    put_wall  = df_filtered[df_filtered['type'] == 'Put'].groupby('strike')['gex_normal'].sum().abs().idxmax() if not df_filtered.empty else spot
    max_pain  = summary_sorted.loc[summary_sorted['oi_val'].idxmax()]['strike'] if not summary_sorted.empty else spot

    # IV & Expected Moves (Min/Max Daily Move 1SD / 2SD)
    atm_strike = summary_sorted.iloc[(summary_sorted['strike'] - spot).abs().argsort()[:1]]['strike'].values[0] if not summary_sorted.empty else spot
    matching_iv = summary_sorted[summary_sorted['strike'] == atm_strike]['mark_iv'].values
    atm_iv = (matching_iv[0] / 100.0) if len(matching_iv) > 0 and not np.isnan(matching_iv[0]) and matching_iv[0] > 0 else 0.55

    exp_move_1sd = spot * atm_iv * np.sqrt(1 / 365.0)
    min_upper, min_lower = spot + exp_move_1sd, spot - exp_move_1sd
    max_upper, max_lower = spot + (2 * exp_move_1sd), spot - (2 * exp_move_1sd)

    # Top Metrics Dashboard
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("BTC Spot", f"${spot:,.1f}")
    m2.metric("Gamma Flip", f"${gamma_flip:,.0f}")
    m3.metric("Intraday Call Wall", f"${call_wall:,.0f}")
    m4.metric("Intraday Put Wall", f"${put_wall:,.0f}")
    m5.metric("Max Pain Pin", f"${max_pain:,.0f}")
    m6.metric("±1 SD Move", f"±${exp_move_1sd:,.0f}")

    st.markdown("---")

    # ==========================================
    # PLOTLY PROFIL-DIAGRAMM (VOLUME PROFILE ENGINE)
    # ==========================================
    fig = go.Figure()

    if show_gex:
        fig.add_trace(go.Bar(
            x=summary['gex_normal'], y=summary['strike'], orientation='h',
            name='🟠 GEX Normal (Gamma OI)', marker_color='#FF9800'
        ))
    if show_gex_vol:
        fig.add_trace(go.Bar(
            x=summary['gex_volume'], y=summary['strike'], orientation='h',
            name='🩵 GEX Volume (Orderflow)', marker_color='#00BCD4'
        ))
    if show_dex:
        fig.add_trace(go.Bar(
            x=summary['dex_val'], y=summary['strike'], orientation='h',
            name='🟣 DEX (Delta Exposure)', marker_color='#9C27B0'
        ))
    if show_oi:
        fig.add_trace(go.Bar(
            x=summary['oi_val'], y=summary['strike'], orientation='h',
            name='🟡 Open Interest (Contracts)', marker_color='#FFEB3B'
        ))

    # Linien für alle Levels aus dem Pine Script einzeichnen
    levels = [
        (spot, "SPOT", "white", "solid", 2),
        (gamma_flip, "GAMMA FLIP", "yellow", "solid", 3),
        (call_wall, "Intraday Call Wall", "red", "dot", 2),
        (put_wall, "Intraday Put Wall", "green", "dot", 2),
        (max_pain, "Max Pain Pin", "purple", "dash", 2),
        (min_upper, "MIN Move Upper (+1 SD)", "orange", "dash", 1),
        (min_lower, "MIN Move Lower (-1 SD)", "orange", "dash", 1),
        (max_upper, "MAX Move Upper (+2 SD)", "fuchsia", "dash", 2),
        (max_lower, "MAX Move Lower (-2 SD)", "fuchsia", "dash", 2)
    ]

    for price, name, color, style, width in levels:
        fig.add_hline(
            y=price, 
            line_dash=style, 
            line_color=color, 
            line_width=width,
            annotation_text=f"{name}: ${price:,.0f}",
            annotation_position="top right"
        )

    fig.update_layout(
        template="plotly_dark",
        height=750,
        barmode='group',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(range=[y_min, y_max], tickformat="$,.0f", title="Strike Price ($)"),
        xaxis=dict(title="Profile Metric Value"),
        margin=dict(l=20, r=20, t=40, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Fehler beim Laden des Dashboards: {e}")    start_time = end_time - (72 * 60 * 60 * 1000)
    url = f"https://www.deribit.com/api/v2/public/get_tradingview_chart_data?instrument_name=BTC-PERPETUAL&start_timestamp={start_time}&end_timestamp={end_time}&resolution={resolution}"
    res = requests.get(url).json()['result']
    df = pd.DataFrame({
        'time': [t // 1000 for t in res['ticks']],
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

    # Restlaufzeit in Jahren
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

    # SIDEBAR: Checkboxen wie im Pine Script
    st.sidebar.header("🎛️ Profile Selection")
    show_gex_normal = st.sidebar.checkbox("🟠 Show GEX Normal (Gamma OI)", value=True)
    show_gex_vol    = st.sidebar.checkbox("🩵 Show GEX Volume (Orderflow)", value=True)
    show_dex        = st.sidebar.checkbox("🟣 Show DEX (Delta Exposure)", value=True)
    show_oi         = st.sidebar.checkbox("🟡 Show Open Interest (Contracts)", value=True)

    st.sidebar.header("🗓️ Filter Expiration")
    expirations = sorted(df_raw['expiration_str'].unique().tolist())
    selected_exp = st.sidebar.selectbox("Select Expiration", ["ALL (Aggregated)"] + expirations, index=0)

    if selected_exp != "ALL (Aggregated)":
        df = df_raw[df_raw['expiration_str'] == selected_exp].copy()
    else:
        df = df_raw.copy()

    # Metriken berechnen
    df['gex_normal'] = np.where(df['type'] == 'Call', df['gamma'] * df['open_interest'] * spot, -df['gamma'] * df['open_interest'] * spot) / 1e6
    df['gex_volume'] = np.where(df['type'] == 'Call', df['gamma'] * df['volume'] * spot, -df['gamma'] * df['volume'] * spot) / 1e6
    df['dex_val']    = df['delta'] * df['open_interest'] * spot / 1e6
    df['oi_val']     = df['open_interest'] / 1000.0 # in Tausend Contracts

    zoom_range = 3500
    y_min, y_max = spot - zoom_range, spot + zoom_range
    df_filtered = df[(df['strike'] >= y_min) & (df['strike'] <= y_max)].copy()

    summary = df_filtered.groupby('strike').agg({
        'gex_normal': 'sum',
        'gex_volume': 'sum',
        'dex_val': 'sum',
        'oi_val': 'sum'
    }).reset_index()

    # Key Levels & Expected Move
    summary_sorted = summary.sort_values('strike').copy()
    summary_sorted['cum_gex'] = summary_sorted['gex_normal'].cumsum()
    zero_crossings = summary_sorted[np.sign(summary_sorted['cum_gex']).diff() != 0]
    gamma_flip = zero_crossings['strike'].iloc[0] if len(zero_crossings) > 0 else spot

    call_wall = df_filtered[df_filtered['type'] == 'Call'].groupby('strike')['gex_normal'].sum().idxmax() if not df_filtered.empty else spot
    put_wall  = df_filtered[df_filtered['type'] == 'Put'].groupby('strike')['gex_normal'].sum().abs().idxmax() if not df_filtered.empty else spot

    # ATM IV & Expected Moves (±1 SD / ±2 SD)
    atm_strike = summary_sorted.iloc[(summary_sorted['strike'] - spot).abs().argsort()[:1]]['strike'].values[0] if not summary_sorted.empty else spot
    matching_iv = df_filtered[df_filtered['strike'] == atm_strike]['mark_iv'].mean()
    atm_iv = (matching_iv / 100.0) if not np.isnan(matching_iv) and matching_iv > 0 else 0.55

    exp_move_1sd = spot * atm_iv * np.sqrt(1 / 365.0)
    min_upper, min_lower = spot + exp_move_1sd, spot - exp_move_1sd
    max_upper, max_lower = spot + (2 * exp_move_1sd), spot - (2 * exp_move_1sd)

    # Top Metrics Bar
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("BTC Spot", f"${spot:,.1f}")
    m2.metric("Gamma Flip", f"${gamma_flip:,.0f}")
    m3.metric("Call Wall", f"${call_wall:,.0f}")
    m4.metric("Put Wall", f"${put_wall:,.0f}")
    m5.metric("±1 SD Move", f"±${exp_move_1sd:,.0f}")

    st.markdown("---")

    # LAYOUT: Links TradingView Chart | Rechts Die 4 Balken-Profile nebeneinander
    col_tv, col_profiles = st.columns([0.45, 0.55])

    with col_tv:
        st.subheader("📺 TradingView Perpetual Chart")
        candle_data = candles[['time', 'open', 'high', 'low', 'close']].to_dict('records')
        volume_data = candles[['time', 'volume', 'open', 'close']].copy()
        volume_data['color'] = np.where(volume_data['close'] >= volume_data['open'], 'rgba(38, 166, 154, 0.5)', 'rgba(239, 83, 80, 0.5)')
        volume_data = volume_data.rename(columns={'volume': 'value'}).drop(columns=['open', 'close']).to_dict('records')

        chart_options = {
            "height": 700,
            "layout": {"background": {"type": "solid", "color": "#131722"}, "textColor": "#d1d4dc"},
            "grid": {"vertLines": {"color": "#2B2B43"}, "horzLines": {"color": "#2B2B43"}},
            "crosshair": {"mode": 0},
            "priceScale": {"borderColor": "#555"},
            "timeScale": {"borderColor": "#555", "timeVisible": True}
        }
        series_candlestick = [{
            "type": "Candlestick",
            "data": candle_data,
            "options": {"upColor": "#26a69a", "downColor": "#ef5350", "borderVisible": False}
        }, {
            "type": "Histogram",
            "data": volume_data,
            "options": {"priceFormat": {"type": "volume"}, "priceScaleId": ""},
            "priceScale": {"scaleMargins": {"top": 0.8, "bottom": 0}}
        }]
        renderLightweightCharts([{"chart": chart_options, "series": series_candlestick}], key="tv_main_chart")

    with col_profiles:
        st.subheader(f"📊 Side-by-Side Profiles ({selected_exp})")
        
        # Plotly Subplots für die aktivierten Profile nebeneinander
        fig = go.Figure()

        if show_gex_normal:
            fig.add_trace(go.Bar(
                x=summary['gex_normal'], y=summary['strike'], orientation='h',
                name='GEX Normal', marker_color='#FF9800'
            ))
        if show_gex_vol:
            fig.add_trace(go.Bar(
                x=summary['gex_volume'], y=summary['strike'], orientation='h',
                name='GEX Volume', marker_color='#00BCD4'
            ))
        if show_dex:
            fig.add_trace(go.Bar(
                x=summary['dex_val'], y=summary['strike'], orientation='h',
                name='DEX', marker_color='#9C27B0'
            ))
        if show_oi:
            fig.add_trace(go.Bar(
                x=summary['oi_val'], y=summary['strike'], orientation='h',
                name='Open Interest', marker_color='#FFEB3B'
            ))

        # Linien für Key Levels und Expected Move einzeichnen
        levels = [
            (spot, "SPOT", "white", "solid"),
            (gamma_flip, "GAMMA FLIP", "yellow", "solid"),
            (min_upper, "+1 SD", "orange", "dash"),
            (min_lower, "-1 SD", "orange", "dash"),
            (max_upper, "+2 SD", "fuchsia", "dash"),
            (max_lower, "-2 SD", "fuchsia", "dash"),
            (call_wall, "Call Wall", "red", "dot"),
            (put_wall, "Put Wall", "green", "dot")
        ]
        for price, name, color, style in levels:
            fig.add_hline(y=price, line_dash=style, line_color=color, annotation_text=f"{name} ({price:,.0f})")

        fig.update_layout(
            template="plotly_dark",
            height=700,
            barmode='group',
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis=dict(range=[y_min, y_max], tickformat="$,.0f", title="Strike Price ($)"),
            xaxis=dict(title="Exposure / Volume Value"),
            margin=dict(l=10, r=10, t=30, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Fehler beim Laden des Dashboards: {e}")
