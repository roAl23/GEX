import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Deribit GEX Pro Dashboard", layout="wide")

st.title("📊 Deribit Options Analytics Dashboard (GEX, DEX, Levels)")

# 1. Spot-Preis abrufen
@st.cache_data(ttl=15)
def get_btc_spot():
    url = "https://www.deribit.com/api/v2/public/get_index_price?index_name=btc_usd"
    res = requests.get(url).json()
    return res['result']['index_price']

# 2. Alle Options-Marktdaten abrufen
@st.cache_data(ttl=30)
def get_option_data():
    url = "https://www.deribit.com/api/v2/public/get_book_summary_by_currency?currency=BTC&kind=option"
    res = requests.get(url).json()['result']
    df = pd.DataFrame(res)
    return df

try:
    spot = get_btc_spot()
    df = get_option_data()
    
    # Datenaufbereitung & Parsing
    df['strike'] = df['instrument_name'].apply(lambda x: float(x.split('-')[2]))
    df['type'] = df['instrument_name'].apply(lambda x: 'Call' if x.endswith('-C') else 'Put')
    
    # Fehlende Griechen/Werte auf 0 setzen
    for col in ['open_interest', 'volume', 'gamma', 'delta', 'mark_iv']:
        if col in df.columns:
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
