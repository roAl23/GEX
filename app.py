import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
from scipy.stats import norm
import plotly.graph_objects as go

# Layout auf Wide stellen
st.set_page_config(page_title="Deribit Options Terminal", layout="wide")

# --- CUSTOM CSS FÜR PREMIUM TERMINAL LOOK ---
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #fafafa; }
    .metric-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

st.title("⚡ Deribit Options Profile Engine")

# --- BLACK-SCHOLES BERECHNUNG ---
def calculate_greeks(spot, strike, t_years, iv, option_type):
    if t_years <= 0 or iv <= 0 or spot <= 0 or strike <= 0:
        return 0.0, 0.0, 0.5
    d1 = (np.log(spot / strike) + (0.5 * iv**2) * t_years) / (iv * np.sqrt(t_years))
    gamma = norm.pdf(d1) / (spot * iv * np.sqrt(t_years))
    delta = norm.cdf(d1) if option_type == 'Call' else norm.cdf(d1) - 1.0
    return gamma, delta, norm.cdf(d1)

@st.cache_data(ttl=15)
def get_btc_spot():
    url = "https://www.deribit.com/api/v2/public/get_index_price?index_name=btc_usd"
    return requests.get(url).json()['result']['index_price']

@st.cache_data(ttl=30)
def get_option_data():
    url = "https://www.deribit.com/api/v2/public/get_book_summary_by_currency?currency=BTC&kind=option"
    return pd.DataFrame(requests.get(url).json()['result'])

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

    # SIDEBAR
    st.sidebar.markdown("### 🎛️ Profil-Steuerung")
    show_gex    = st.sidebar.checkbox("🟠 GEX Normal (Gamma OI)", value=True)
    show_gex_vol = st.sidebar.checkbox("🩵 GEX Volume (Orderflow)", value=True)
    show_dex    = st.sidebar.checkbox("🟣 DEX (Delta Exposure)", value=True)
    show_oi     = st.sidebar.checkbox("🟡 Open Interest (Contracts)", value=True)

    expirations = sorted(df_raw['expiration_str'].unique().tolist())
    selected_exp = st.sidebar.selectbox("🗓️ Expiry Filter", ["ALL (Aggregated)"] + expirations)
    zoom_margin = st.sidebar.slider("📐 Zoom Range (± USD)", 1000, 10000, 4000, step=500)

    df = df_raw if selected_exp == "ALL (Aggregated)" else df_raw[df_raw['expiration_str'] == selected_exp]

    df['gex_normal'] = np.where(df['type'] == 'Call', df['gamma'] * df['open_interest'] * spot, -df['gamma'] * df['open_interest'] * spot) / 1e6
    df['gex_volume'] = np.where(df['type'] == 'Call', df['gamma'] * df['volume'] * spot, -df['gamma'] * df['volume'] * spot) / 1e6
    df['dex_val']    = df['delta'] * df['open_interest'] * spot / 1e6
    df['oi_val']     = df['open_interest']

    y_min, y_max = spot - zoom_margin, spot + zoom_margin
    df_filtered = df[(df['strike'] >= y_min) & (df['strike'] <= y_max)]

    summary = df_filtered.groupby('strike').agg({
        'gex_normal': 'sum', 'gex_volume': 'sum', 'dex_val': 'sum', 'oi_val': 'sum', 'mark_iv': 'mean'
    }).reset_index()

    summary_sorted = summary.sort_values('strike')
    summary_sorted['cum_gex'] = summary_sorted['gex_normal'].cumsum()
    zero_crossings = summary_sorted[np.sign(summary_sorted['cum_gex']).diff() != 0]
    gamma_flip = zero_crossings['strike'].iloc[0] if len(zero_crossings) > 0 else spot

    call_wall = df_filtered[df_filtered['type'] == 'Call'].groupby('strike')['gex_normal'].sum().idxmax() if not df_filtered.empty else spot
    put_wall  = df_filtered[df_filtered['type'] == 'Put'].groupby('strike')['gex_normal'].sum().abs().idxmax() if not df_filtered.empty else spot
    max_pain  = summary_sorted.loc[summary_sorted['oi_val'].idxmax()]['strike'] if not summary_sorted.empty else spot

    atm_strike = summary_sorted.iloc[(summary_sorted['strike'] - spot).abs().argsort()[:1]]['strike'].values[0] if not summary_sorted.empty else spot
    matching_iv = summary_sorted[summary_sorted['strike'] == atm_strike]['mark_iv'].values
    atm_iv = (matching_iv[0] / 100.0) if len(matching_iv) > 0 and not np.isnan(matching_iv[0]) else 0.55
    exp_move_1sd = spot * atm_iv * np.sqrt(1 / 365.0)

    # Clean Metrics Row
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("BTC Spot", f"${spot:,.1f}")
    m2.metric("Gamma Flip", f"${gamma_flip:,.0f}")
    m3.metric("Intraday Call Wall", f"${call_wall:,.0f}")
    m4.metric("Intraday Put Wall", f"${put_wall:,.0f}")
    m5.metric("Max Pain Pin", f"${max_pain:,.0f}")
    m6.metric("±1 SD Move", f"±${exp_move_1sd:,.0f}")

    st.markdown("<hr style='border: 1px solid #30363d;'>", unsafe_allow_html=True)

    # --- PLOTLY PROFIL-DIAGRAMM (CLEAN & MODERN) ---
    fig = go.Figure()

    if show_gex:
        fig.add_trace(go.Bar(x=summary['gex_normal'], y=summary['strike'], orientation='h', name='GEX Normal', marker_color='#FF9800'))
    if show_gex_vol:
        fig.add_trace(go.Bar(x=summary['gex_volume'], y=summary['strike'], orientation='h', name='GEX Volume', marker_color='#00BCD4'))
    if show_dex:
        fig.add_trace(go.Bar(x=summary['dex_val'], y=summary['strike'], orientation='h', name='DEX', marker_color='#AB47BC'))
    if show_oi:
        fig.add_trace(go.Bar(x=summary['oi_val'], y=summary['strike'], orientation='h', name='Open Interest', marker_color='#FFEE58'))

    levels = [
        (spot, "SPOT", "#ffffff", "solid", 2),
        (gamma_flip, "GAMMA FLIP", "#ffeb3b", "solid", 3),
        (call_wall, "Call Wall", "#ff5252", "dot", 2),
        (put_wall, "Put Wall", "#66bb6a", "dot", 2),
        (max_pain, "Max Pain", "#ab47bc", "dash", 2),
        (spot + exp_move_1sd, "+1 SD Upper", "#ffa726", "dash", 1),
        (spot - exp_move_1sd, "-1 SD Lower", "#ffa726", "dash", 1),
    ]

    for price, name, color, style, width in levels:
        fig.add_hline(y=price, line_dash=style, line_color=color, line_width=width,
                      annotation_text=f"  {name}: ${price:,.0f}", annotation_position="top right",
                      annotation_font_color=color)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='#0e1117',
        plot_bgcolor='#161b22',
        height=800,
        barmode='group',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor='rgba(0,0,0,0)'),
        yaxis=dict(range=[y_min, y_max], tickformat="$,.0f", title="Strike Price ($)", gridcolor="#30363d"),
        xaxis=dict(title="Metric Exposure Value", gridcolor="#30363d"),
        margin=dict(l=40, r=40, t=40, b=40)
    )

    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Fehler: {e}")
