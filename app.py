import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
from scipy.stats import norm
import plotly.graph_objects as go

# 1. Page Config (Wide Mode für das Terminal-Layout)
st.set_page_config(
    page_title="Deribit Options Profile Engine + Min/Max Daily Move",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- TERMINAL CSS STYLING ---
st.markdown("""
    <style>
    .stApp { background-color: #0b0e14; color: #e6edf3; }
    div.stMetric {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 10px;
        border-radius: 6px;
    }
    div.stMetric label { color: #8b949e !important; }
    </style>
""", unsafe_allow_html=True)

st.markdown("### 📊 Deribit Options Profile Engine + Min/Max Daily Move (Histogram Layout)")

# --- BLACK-SCHOLES GREEKS BERECHNUNG ---
def calculate_greeks(spot, strike, t_years, iv, option_type):
    if t_years <= 0 or iv <= 0 or spot <= 0 or strike <= 0:
        return 0.0, 0.0, 0.5
    d1 = (np.log(spot / strike) + (0.5 * iv**2) * t_years) / (iv * np.sqrt(t_years))
    gamma = norm.pdf(d1) / (spot * iv * np.sqrt(t_years))
    delta = norm.cdf(d1) if option_type == 'Call' else norm.cdf(d1) - 1.0
    return gamma, delta, norm.cdf(d1)

@st.cache_data(ttl=15)
def get_btc_spot():
    try:
        url = "https://www.deribit.com/api/v2/public/get_index_price?index_name=btc_usd"
        return requests.get(url).json()['result']['index_price']
    except:
        return 85000.0

@st.cache_data(ttl=30)
def get_option_data():
    url = "https://www.deribit.com/api/v2/public/get_book_summary_by_currency?currency=BTC&kind=option"
    return pd.DataFrame(requests.get(url).json()['result'])

try:
    spot = get_btc_spot()
    df_raw = get_option_data()

    # Preprocessing der Instrumente
    df_raw['expiration_str'] = df_raw['instrument_name'].apply(lambda x: x.split('-')[1])
    df_raw['strike'] = df_raw['instrument_name'].apply(lambda x: float(x.split('-')[2]))
    df_raw['type'] = df_raw['instrument_name'].apply(lambda x: 'Call' if x.endswith('-C') else 'Put')

    for col in ['open_interest', 'volume', 'mark_iv']:
        if col in df_raw.columns:
            df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce').fillna(0.0)

    current_time = time.time()
    def parse_expiration_years(exp_str):
        try:
            exp_date = pd.to_datetime(exp_str, format='%d%b%y')
            return max(exp_date.timestamp() + 28800 - current_time, 3600) / (365.25 * 86400)
        except:
            return 1.0 / 365.25

    df_raw['t_years'] = df_raw['expiration_str'].apply(parse_expiration_years)

    gammas, deltas = [], []
    for _, row in df_raw.iterrows():
        g, d, _ = calculate_greeks(spot, row['strike'], row['t_years'], row['mark_iv'] / 100.0, row['type'])
        gammas.append(g)
        deltas.append(d)

    df_raw['gamma'] = gammas
    df_raw['delta'] = deltas

    # ==========================================
    # 1. CHECKBOX SELECTION (Aus Pine Script übernommen)
    # ==========================================
    st.sidebar.markdown("### 🎛️ Profile Selection (Histogram)")
    showGex    = st.sidebar.checkbox("🟠 Show GEX Normal (Gamma OI)", value=True)
    showGexVol = st.sidebar.checkbox("🩵 Show GEX Volume (Orderflow)", value=True)
    showDex    = st.sidebar.checkbox("🟣 Show DEX (Delta Exposure)", value=True)
    showOi     = st.sidebar.checkbox("🟡 Show Open Interest (Contracts)", value=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Engine Parameters")
    maxBarWidth = st.sidebar.slider("Max Profile Bar Width / Scale Divisor", 10.0, 500.0, 150.0, step=10.0)
    showDailyMove = st.sidebar.checkbox("Show Daily Move Range Lines", value=True)

    expirations = sorted(df_raw['expiration_str'].unique().tolist())
    selected_exp = st.sidebar.selectbox("🗓️ Expiry Filter", ["ALL (Aggregated)"] + expirations)
    zoom_margin = st.sidebar.slider("📐 Zoom Range (± USD um Spot)", 1000, 15000, 4000, step=500)

    df = df_raw if selected_exp == "ALL (Aggregated)" else df_raw[df_raw['expiration_str'] == selected_exp]

    # --- Kennzahlen analog Pine Script berechnen ---
    df['gex_normal'] = np.where(df['type'] == 'Call', df['gamma'] * df['open_interest'] * spot, -df['gamma'] * df['open_interest'] * spot) / maxBarWidth
    df['gex_volume'] = np.where(df['type'] == 'Call', df['gamma'] * df['volume'] * spot, -df['gamma'] * df['volume'] * spot) / maxBarWidth
    df['dex_val']    = (df['delta'] * df['open_interest'] * spot / 1e6) / 0.3  # Skalierungsfaktor wie im Pine Script
    df['oi_val']     = df['open_interest'] / 10.0

    y_min, y_max = spot - zoom_margin, spot + zoom_margin
    df_filtered = df[(df['strike'] >= y_min) & (df['strike'] <= y_max)]

    summary = df_filtered.groupby('strike').agg({
        'gex_normal': 'sum', 'gex_volume': 'sum', 'dex_val': 'sum', 'oi_val': 'sum', 'mark_iv': 'mean'
    }).reset_index().sort_values('strike')

    # Core Key Levels
    summary['cum_gex'] = summary['gex_normal'].cumsum()
    zero_crossings = summary[np.sign(summary['cum_gex']).diff() != 0]
    gammaFlip = zero_crossings['strike'].iloc[0] if len(zero_crossings) > 0 else spot

    callWallIntra = df_filtered[df_filtered['type'] == 'Call'].groupby('strike')['gex_normal'].sum().idxmax() if not df_filtered.empty else spot
    putWallIntra  = df_filtered[df_filtered['type'] == 'Put'].groupby('strike')['gex_normal'].sum().abs().idxmax() if not df_filtered.empty else spot
    maxPain       = summary.loc[summary['oi_val'].idxmax()]['strike'] if not summary.empty else spot

    # IV & Expected Moves (±1 SD / ±2 SD)
    atm_strike = summary.iloc[(summary['strike'] - spot).abs().argsort()[:1]]['strike'].values[0] if not summary.empty else spot
    matching_iv = summary[summary['strike'] == atm_strike]['mark_iv'].values
    atm_iv = (matching_iv[0] / 100.0) if len(matching_iv) > 0 and not np.isnan(matching_iv[0]) else 0.55
    
    exp_move_1sd = spot * atm_iv * np.sqrt(1 / 365.0)
    minMoveUpper, minMoveLower = spot + exp_move_1sd, spot - exp_move_1sd
    maxMoveUpper, maxMoveLower = spot + (2 * exp_move_1sd), spot - (2 * exp_move_1sd)

    # --- TOP METRIC ROW ---
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("BTC Spot", f"${spot:,.1f}")
    m2.metric("Gamma Flip", f"${gammaFlip:,.0f}")
    m3.metric("Intraday Call Wall", f"${callWallIntra:,.0f}")
    m4.metric("Intraday Put Wall", f"${putWallIntra:,.0f}")
    m5.metric("Max Pain Pin", f"${maxPain:,.0f}")
    m6.metric("±1 SD Move", f"±${exp_move_1sd:,.0f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ==========================================
    # 2. PLOTLY VERTICAL HISTOGRAM (WIE AUF DEM FOTO)
    # ==========================================
    fig = go.Figure()

    # X-Achse sind die Strikes, Y-Achse sind die Werte (Balken nach oben/unten)
    if showGex:
        fig.add_trace(go.Bar(x=summary['strike'], y=summary['gex_normal'], name='🟠 GEX Normal (Gamma OI)', marker_color='#ff9800'))
    if showGexVol:
        fig.add_trace(go.Bar(x=summary['strike'], y=summary['gex_volume'], name='🩵 GEX Volume (Orderflow)', marker_color='#00bcd4'))
    if showDex:
        fig.add_trace(go.Bar(x=summary['strike'], y=summary['dex_val'], name='🟣 DEX (Delta Exposure)', marker_color='#ab47bc'))
    if showOi:
        fig.add_trace(go.Bar(x=summary['strike'], y=summary['oi_val'], name='🟡 Open Interest (Contracts)', marker_color='#ffeb3b'))

    # Vertikale Linien für die Key Levels (entspricht den Linien aus dem Pine-Skript auf der Strike-Achse)
    v_lines = [
        (spot, "SPOT", "#ffffff", "solid", 2),
        (gammaFlip, "GAMMA FLIP", "#ffeb3b", "solid", 3),
        (callWallIntra, "Intraday Call Wall", "#ff5252", "dot", 2),
        (putWallIntra, "Intraday Put Wall", "#66bb6a", "dot", 2),
        (maxPain, "Max Pain Pin", "#ab47bc", "dash", 2),
    ]

    if showDailyMove:
        v_lines.extend([
            (minMoveUpper, "MIN Move Upper (+1 SD)", "#ffa726", "dash", 1),
            (minMoveLower, "MIN Move Lower (-1 SD)", "#ffa726", "dash", 1),
            (maxMoveUpper, "MAX Move Upper (+2 SD)", "#f06292", "dash", 2),
            (maxMoveLower, "MAX Move Lower (-2 SD)", "#f06292", "dash", 2),
        ])

    for x_val, name, color, style, width in v_lines:
        fig.add_vline(
            x=x_val, line_dash=style, line_color=color, line_width=width,
            annotation_text=f"{name} (${x_val:,.0f})", 
            annotation_position="top",
            annotation_font_color=color,
            annotation_font_size=10
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='#0b0e14',
        plot_bgcolor='#11141d',
        height=750,
        barmode='group',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor='rgba(0,0,0,0)'),
        xaxis=dict(range=[y_min, y_max], tickformat="$,.0f", title="Strike Price ($)", gridcolor="#21262d"),
        yaxis=dict(title="Profile Exposure / Volume Bar Value", gridcolor="#21262d"),
        margin=dict(l=40, r=40, t=50, b=40)
    )

    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Fehler beim Verarbeiten der Daten: {e}")
