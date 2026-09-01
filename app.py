import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Deribit GEX & Options Dashboard", layout="wide")

st.title("📊 Deribit Options Profile & Daily Move Dashboard")

# 1. Spot Preis von Deribit abrufen
@st.cache_data(ttl=30)
def get_btc_spot():
    url = "https://www.deribit.com/api/v2/public/get_index_price?index_name=btc_usd"
    res = requests.get(url).json()
    return res['result']['index_price']

# 2. Optionsdaten abrufen
@st.cache_data(ttl=60)
def get_option_data():
    url = "https://www.deribit.com/api/v2/public/get_book_summary_by_currency?currency=BTC&kind=option"
    res = requests.get(url).json()['result']
    df = pd.DataFrame(res)
    return df

try:
    spot = get_btc_spot()
    st.metric("BTC Spot Price", f"${spot:,.2f}")
    
    df = get_option_data()
    
    # Strikes und Types extrahieren
    df['strike'] = df['instrument_name'].apply(lambda x: float(x.split('-')[2]))
    df['type'] = df['instrument_name'].apply(lambda x: 'Call' if x.endswith('-C') else 'Put')
    
    # Filter um den aktuellen Spot-Preis herum (+/- $10.000)
    df_filtered = df[(df['strike'] >= spot - 10000) & (df['strike'] <= spot + 10000)].copy()
    
    # Groupings für Open Interest & Volumen
    oi_df = df_filtered.groupby('strike')['open_interest'].sum().reset_index()
    vol_df = df_filtered.groupby('strike')['volume'].sum().reset_index()
    
    # Daily Move Berechnung (Expected Move 1 SD / 2 SD)
    avg_iv = 0.55  # ~55% IV
    exp_move_1sd = spot * avg_iv * np.sqrt(1 / 365.0)
    
    min_upper = spot + exp_move_1sd
    min_lower = spot - exp_move_1sd
    max_upper = spot + (2 * exp_move_1sd)
    max_lower = spot - (2 * exp_move_1sd)
    
    # Sidebar Info
    st.sidebar.header("📊 Daily Move Levels (Expected Range)")
    st.sidebar.write(f"**+2 SD Max Upper:** ${max_upper:,.0f}")
    st.sidebar.write(f"**+1 SD Min Upper:** ${min_upper:,.0f}")
    st.sidebar.write(f"**Spot:** ${spot:,.0f}")
    st.sidebar.write(f"**-1 SD Min Lower:** ${min_lower:,.0f}")
    st.sidebar.write(f"**-2 SD Max Lower:** ${max_lower:,.0f}")
    
    # Charts erstellen
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Open Interest by Strike", "Volume by Strike"))
    
    fig.add_trace(go.Bar(x=oi_df['open_interest'], y=oi_df['strike'], orientation='h', name="Open Interest", marker_color='gold'), row=1, col=1)
    fig.add_trace(go.Bar(x=vol_df['volume'], y=vol_df['strike'], orientation='h', name="Volume", marker_color='cyan'), row=1, col=2)
    
    # Levels einzeichnen
    for col_idx in [1, 2]:
        fig.add_hline(y=spot, line_dash="solid", line_color="white", annotation_text="Spot", row=1, col=col_idx)
        fig.add_hline(y=min_upper, line_dash="dash", line_color="orange", annotation_text="+1 SD Min", row=1, col=col_idx)
        fig.add_hline(y=min_lower, line_dash="dash", line_color="orange", annotation_text="-1 SD Min", row=1, col=col_idx)
        fig.add_hline(y=max_upper, line_dash="dash", line_color="fuchsia", annotation_text="+2 SD Max", row=1, col=col_idx)
        fig.add_hline(y=max_lower, line_dash="dash", line_color="fuchsia", annotation_text="-2 SD Max", row=1, col=col_idx)
        
    fig.update_layout(template="plotly_dark", height=700, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Fehler beim Laden der Live-Daten: {e}")
