import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
from scipy.stats import norm
import plotly.graph_objects as go

st.set_page_config(page_title="Deribit Options Profile Engine Pro", layout="wide", initial_sidebar_state="expanded")

# --- CSS FÜR TERMINAL OPTIK ---
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

st.markdown("### 📊 Deribit Options Profile Engine (Live API)")

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
        return 90000.0

@st.cache_data(ttl=30)
def get_option_data():
    url = "https://www.deribit.com/api/v2/public/get_book_summary_by_currency?currency=BTC&kind=option"
    return pd.DataFrame(requests.get(url).json()['result'])

try:
    spot = get_btc_spot()
    df_raw = get_option_data()

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

    # --- SIDEBAR UI ---
    st.sidebar.markdown("### 🎛️ Profile Selection")
    showGex    = st.sidebar.checkbox("🟠 Show GEX Normal (Gamma)", value=True)
    showGexVol = st.sidebar.checkbox("🩵 Show GEX Volume", value=True)
    showDex    = st.sidebar.checkbox("🟣 Show DEX (Delta)", value=True)
    showOi     = st.sidebar.checkbox("🟡 Show Open Interest", value=True)
    
    st.sidebar.markdown("### ⚖️ Visual Scaling (Divisors)")
    div_gex = st.sidebar.number_input("GEX Divisor", value=1.0, step=0.5)
    div_dex = st.sidebar.number_input("DEX Divisor", value=1.0, step=0.1)
    div_oi  = st.sidebar.number_input("OI Divisor", value=100.0, step=10.0)

    st.sidebar.markdown("---")
    showDailyMove = st.sidebar.checkbox("Show Daily Move Range Lines", value=True)
    expirations = sorted(df_raw['expiration_str'].unique().tolist())
    selected_exp = st.sidebar.selectbox("🗓️ Expiry Filter", ["ALL (Aggregated)"] + expirations)
    zoom_margin = st.sidebar.slider("📐 Zoom Range (± USD um Spot)", 1000, 20000, 5000, step=500)

    df = df_raw if selected_exp == "ALL (Aggregated)" else df_raw[df_raw['expiration_str'] == selected_exp]

    # --- KENNZAHLEN LOGIK (Mathematisch korrekt) ---
    contract_multiplier = 1.0  # 1 Kontrakt = 1 BTC bei Deribit
    
    # GEX in Milliarden USD ($), DEX in Millionen USD ($) 
    df['gex_normal'] = np.where(df['type'] == 'Call', df['gamma'] * df['open_interest'] * contract_multiplier * (spot**2), -df['gamma'] * df['open_interest'] * contract_multiplier * (spot**2)) / 1e9
    df['gex_volume'] = np.where(df['type'] == 'Call', df['gamma'] * df['volume'] * contract_multiplier * (spot**2), -df['gamma'] * df['volume'] * contract_multiplier * (spot**2)) / 1e9
    df['dex_val']    = (df['delta'] * df['open_interest'] * contract_multiplier * spot) / 1e6
    df['oi_val']     = df['open_interest']

    y_min, y_max = spot - zoom_margin, spot + zoom_margin
    df_filtered = df[(df['strike'] >= y_min) & (df['strike'] <= y_max)]

    summary = df_filtered.groupby('strike').agg({
        'gex_normal': 'sum', 
        'gex_volume': 'sum', 
        'dex_val': 'sum', 
        'oi_val': 'sum', 
        'mark_iv': 'mean'
    }).reset_index().sort_values('strike')

    # Visuelle Skalierung für Plotly
    summary['gex_norm_scaled'] = summary['gex_normal'] / div_gex
    summary['gex_vol_scaled']  = summary['gex_volume'] / div_gex
    summary['dex_val_scaled']  = summary['dex_val'] / div_dex
    summary['oi_val_scaled']   = summary['oi_val'] / div_oi

    # Call / Put OI trennen (Für echtes Max Pain)
    calls_dict = df_filtered[df_filtered['type'] == 'Call'].groupby('strike')['open_interest'].sum().to_dict()
    puts_dict = df_filtered[df_filtered['type'] == 'Put'].groupby('strike')['open_interest'].sum().to_dict()
    summary['call_oi'] = summary['strike'].map(calls_dict).fillna(0)
    summary['put_oi'] = summary['strike'].map(puts_dict).fillna(0)

    # --- KEY LEVELS ---
    
    # KORRIGIERTER GAMMA FLIP (Ignoriert den fehlerhaften NaN-Bug der ersten Zeile)
    sign_changes = summary[np.sign(summary['gex_normal']).diff().fillna(0) != 0].iloc[1:]
    if not sign_changes.empty:
        gammaFlip = sign_changes.iloc[(sign_changes['strike'] - spot).abs().argsort()[:1]]['strike'].values[0]
    else:
        gammaFlip = spot

    callWallIntra = df_filtered[df_filtered['type'] == 'Call'].groupby('strike')['gex_normal'].sum().idxmax() if not df_filtered.empty else spot
    putWallIntra  = df_filtered[df_filtered['type'] == 'Put'].groupby('strike')['gex_normal'].sum().abs().idxmax() if not df_filtered.empty else spot

    # Echte Max-Pain-Berechnung
    strike_arr = summary['strike'].values
    call_oi_arr = summary['call_oi'].values
    put_oi_arr = summary['put_oi'].values
    pains = []
    for s in strike_arr:
        call_payout = np.where(s > strike_arr, (s - strike_arr) * call_oi_arr, 0).sum()
        put_payout = np.where(strike_arr > s, (strike_arr - s) * put_oi_arr, 0).sum()
        pains.append(call_payout + put_payout)
    maxPain = strike_arr[np.argmin(pains)] if len(pains) > 0 else spot

    # Expected Move
    atm_strike = summary.iloc[(summary['strike'] - spot).abs().argsort()[:1]]['strike'].values[0] if not summary.empty else spot
    matching_iv = summary[summary['strike'] == atm_strike]['mark_iv'].values
    atm_iv = (matching_iv[0] / 100.0) if len(matching_iv) > 0 and not np.isnan(matching_iv[0]) else 0.55
    exp_move_1sd = spot * atm_iv * np.sqrt(1 / 365.0)
    minMoveUpper, minMoveLower = spot + exp_move_1sd, spot - exp_move_1sd
    maxMoveUpper, maxMoveLower = spot + (2 * exp_move_1sd), spot - (2 * exp_move_1sd)

    # --- MACRO BERECHNUNGEN (Für Sidebar Tabelle) ---
    net_gamma = summary['gex_normal'].sum()
    net_vex = summary['gex_volume'].sum()
    gamma_regime = "🟢 Positiv (Low Vol)" if net_gamma > 0 else "🔴 Negativ (High Vol)"
    
    total_call_oi = summary['call_oi'].sum()
    total_put_oi = summary['put_oi'].sum()
    pc_ratio = (total_put_oi / total_call_oi) if total_call_oi > 0 else 0.0

    # ==========================================
    # SIDEBAR TABELLE (Market Overview)
    # ==========================================
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Market Overview")
    
    sidebar_table_html = f"""
    <table style="width:100%; border-collapse: collapse; font-size: 14px; text-align: left; color: #e6edf3;">
        <tr style="border-bottom: 1px solid #30363d;">
            <th style="padding: 8px 0;">Metrik</th>
            <th style="padding: 8px 0; text-align: right;">Wert</th>
        </tr>
        <tr style="border-bottom: 1px solid #30363d;">
            <td style="padding: 8px 0; color: #8b949e;">Net GEX (Mrd. $)</td>
            <td style="padding: 8px 0; text-align: right;">{net_gamma:,.2f}</td>
        </tr>
        <tr style="border-bottom: 1px solid #30363d;">
            <td style="padding: 8px 0; color: #8b949e;">Net VEX (Mrd. $)</td>
            <td style="padding: 8px 0; text-align: right;">{net_vex:,.2f}</td>
        </tr>
        <tr style="border-bottom: 1px solid #30363d;">
            <td style="padding: 8px 0; color: #8b949e;">Regime</td>
            <td style="padding: 8px 0; text-align: right;">{gamma_regime}</td>
        </tr>
        <tr style="border-bottom: 1px solid #30363d;">
            <td style="padding: 8px 0; color: #8b949e;">P/C Ratio</td>
            <td style="padding: 8px 0; text-align: right;">{pc_ratio:.2f}</td>
        </tr>
        <tr style="border-bottom: 1px solid #30363d;">
            <td style="padding: 8px 0; color: #8b949e;">Max Pain</td>
            <td style="padding: 8px 0; text-align: right;">${maxPain:,.0f}</td>
        </tr>
        <tr>
            <td style="padding: 8px 0; color: #8b949e;">Gamma Flip</td>
            <td style="padding: 8px 0; text-align: right;">${gammaFlip:,.0f}</td>
        </tr>
    </table>
    """
    st.sidebar.markdown(sidebar_table_html, unsafe_allow_html=True)

    # --- TOP METRICS ---
    st.markdown("#### 🔑 Key Levels & Spot")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("BTC Spot", f"${spot:,.1f}")
    m2.metric("Intraday Call Wall", f"${callWallIntra:,.0f}")
    m3.metric("Intraday Put Wall", f"${putWallIntra:,.0f}")
    m4.metric("±1 SD Move", f"±${exp_move_1sd:,.0f}")

    st.markdown("<hr style='border: 1px solid #30363d;'>", unsafe_allow_html=True)

    # ==========================================
    # CHART RENDERING (Vertical Histogram)
    # ==========================================
    fig = go.Figure()

    if showGex:
        fig.add_trace(go.Bar(x=summary['strike'], y=summary['gex_norm_scaled'], customdata=summary['gex_normal'], hovertemplate='Strike: $%{x}<br>GEX Normal: %{customdata:,.2f} Mrd.<extra></extra>', name='🟠 GEX Normal', marker_color='#ff9800'))
    if showGexVol:
        fig.add_trace(go.Bar(x=summary['strike'], y=summary['gex_vol_scaled'], customdata=summary['gex_volume'], hovertemplate='Strike: $%{x}<br>GEX Vol: %{customdata:,.2f} Mrd.<extra></extra>', name='🩵 GEX Volume', marker_color='#00bcd4'))
    if showDex:
        fig.add_trace(go.Bar(x=summary['strike'], y=summary['dex_val_scaled'], customdata=summary['dex_val'], hovertemplate='Strike: $%{x}<br>DEX: %{customdata:,.2f} Mio.<extra></extra>', name='🟣 DEX Exposure', marker_color='#ab47bc'))
    if showOi:
        fig.add_trace(go.Bar(x=summary['strike'], y=summary['oi_val_scaled'], customdata=summary['oi_val'], hovertemplate='Strike: $%{x}<br>Open Interest: %{customdata:,.0f}<extra></extra>', name='🟡 Open Interest', marker_color='#ffeb3b'))

    v_lines = [
        (spot, "SPOT", "#ffffff", "solid", 2),
        (gammaFlip, "GAMMA FLIP", "#ffeb3b", "solid", 3),
        (callWallIntra, "Call Wall", "#ff5252", "dot", 2),
        (putWallIntra, "Put Wall", "#66bb6a", "dot", 2),
        (maxPain, "Max Pain", "#ab47bc", "dash", 2),
    ]

    if showDailyMove:
        v_lines.extend([
            (minMoveUpper, "MIN Move (+1 SD)", "#ffa726", "dash", 1),
            (minMoveLower, "MIN Move (-1 SD)", "#ffa726", "dash", 1),
            (maxMoveUpper, "MAX Move (+2 SD)", "#f06292", "dash", 2),
            (maxMoveLower, "MAX Move (-2 SD)", "#f06292", "dash", 2),
        ])

    for x_val, name, color, style, width in v_lines:
        fig.add_vline(x=x_val, line_dash=style, line_color=color, line_width=width, annotation_text=f"{name}", annotation_position="top", annotation_font_color=color, annotation_font_size=11)

    fig.update_layout(
        template="plotly_dark", paper_bgcolor='#0b0e14', plot_bgcolor='#11141d', height=750, barmode='group',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor='rgba(0,0,0,0)'),
        xaxis=dict(range=[y_min, y_max], tickformat="$,.0f", title="Strike Price ($)", gridcolor="#21262d"),
        yaxis=dict(title="Scaled Profile Value (Visual Only)", gridcolor="#21262d", showticklabels=False),
        margin=dict(l=40, r=40, t=50, b=40)
    )

    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Fehler beim Verarbeiten der Daten: {e}")
