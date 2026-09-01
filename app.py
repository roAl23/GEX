{\rtf1\ansi\ansicpg1252\cocoartf2759
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import streamlit as st\
import requests\
import pandas as pd\
import numpy as np\
import plotly.graph_objects as go\
from plotly.subplots import make_subplots\
\
st.set_page_config(page_title="Deribit GEX & Options Dashboard", layout="wide")\
\
st.title("\uc0\u55357 \u56522  Deribit Options Profile & Daily Move Dashboard")\
\
# 1. Spot Preis von Deribit abrufen\
@st.cache_data(ttl=30)\
def get_btc_spot():\
    url = "https://www.deribit.com/api/v2/public/get_index_price?index_name=btc_usd"\
    res = requests.get(url).json()\
    return res['result']['index_price']\
\
# 2. Optionsdaten f\'fcr das n\'e4chste Ablaufdatum abrufen\
@st.cache_data(ttl=60)\
def get_option_data():\
    url = "https://www.deribit.com/api/v2/public/get_book_summary_by_currency?currency=BTC&kind=option"\
    res = requests.get(url).json()['result']\
    df = pd.DataFrame(res)\
    return df\
\
try:\
    spot = get_btc_spot()\
    st.metric("BTC Spot Price", f"$\{spot:,.2f\}")\
    \
    df = get_option_data()\
    \
    # Filtern auf kurzfristige Optionen (z.B. nahe Strikes um Spot)\
    df['strike'] = df['instrument_name'].apply(lambda x: float(x.split('-')[2]))\
    df['type'] = df['instrument_name'].apply(lambda x: 'Call' if x.endswith('-C') else 'Put')\
    \
    # Filter um den aktuellen Spot-Preis herum (+/- $10.000)\
    df_filtered = df[(df['strike'] >= spot - 10000) & (df['strike'] <= spot + 10000)].copy()\
    \
    # Berechnungen\
    # Open Interest & Volumen\
    oi_df = df_filtered.groupby('strike')['open_interest'].sum().reset_index()\
    vol_df = df_filtered.groupby('strike')['volume'].sum().reset_index()\
    \
    # Vereinfachte Annahme f\'fcr t\'e4gliche IV (durchschnittliche Mark IV)\
    avg_iv = 0.55  # ~55% IV\
    exp_move_1sd = spot * avg_iv * np.sqrt(1 / 365.0)\
    \
    min_upper = spot + exp_move_1sd\
    min_lower = spot - exp_move_1sd\
    max_upper = spot + (2 * exp_move_1sd)\
    max_lower = spot - (2 * exp_move_1sd)\
    \
    # Sidebar Info\
    st.sidebar.header("\uc0\u55357 \u56522  Daily Move Levels (Expected Range)")\
    st.sidebar.write(f"**+2 SD Max Upper:** $\{max_upper:,.0f\}")\
    st.sidebar.write(f"**+1 SD Min Upper:** $\{min_upper:,.0f\}")\
    st.sidebar.write(f"**Spot:** $\{spot:,.0f\}")\
    st.sidebar.write(f"**-1 SD Min Lower:** $\{min_lower:,.0f\}")\
    st.sidebar.write(f"**-2 SD Max Lower:** $\{max_lower:,.0f\}")\
    \
    # Chart erstellen\
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Open Interest by Strike", "Volume by Strike"))\
    \
    fig.add_trace(go.Bar(x=oi_df['open_interest'], y=oi_df['strike'], orientation='h', name="Open Interest", marker_color='gold'), row=1, col=1)\
    fig.add_trace(go.Bar(x=vol_df['volume'], y=vol_df['strike'], orientation='h', name="Volume", marker_color='cyan'), row=1, col=2)\
    \
    # Horizontale Linien f\'fcr Min/Max Moves auf beiden Subplots\
    for col_idx in [1, 2]:\
        fig.add_hline(y=spot, line_dash="solid", line_color="white", annotation_text="Spot", row=1, col=col_idx)\
        fig.add_hline(y=min_upper, line_dash="dash", line_color="orange", annotation_text="+1 SD Min", row=1, col=col_idx)\
        fig.add_hline(y=min_lower, line_dash="dash", line_color="orange", annotation_text="-1 SD Min", row=1, col=col_idx)\
        fig.add_hline(y=max_upper, line_dash="dash", line_color="fuchsia", annotation_text="+2 SD Max", row=1, col=col_idx)\
        fig.add_hline(y=max_lower, line_dash="dash", line_color="fuchsia", annotation_text="-2 SD Max", row=1, col=col_idx)\
        \
    fig.update_layout(template="plotly_dark", height=700, showlegend=False)\
    st.plotly_chart(fig, use_container_width=True)\
\
except Exception as e:\
    st.error(f"Fehler beim Laden der Live-Daten: \{e\}")}