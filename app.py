import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
from streamlit_lightweight_charts import renderLightweightCharts

st.set_page_config(page_title="Deribit GEX Pro Dashboard", layout="wide")

st.title("📈 BTC TradingView Chart & Options Analytics")

# 1. Spot Price
@st.cache_data(ttl=15)
def get_btc_spot():
    url = "https://www.deribit.com/api/v2/public/get_index_price?index_name=btc_usd"
    res = requests.get(url).json()
    return res['result']['index_price']

# 2. OHLC Kerzen von Deribit (für TradingView Chart)
@st.cache_data(ttl=60)
def get_btc_candles(resolution="60"):
    end_time = int(time.time() * 1000)
    start_time = end_time - (72 * 60 * 60 * 1000) # 72 Stunden
    url = f"https://www.deribit.com/api/v2/public/get_tradingview_chart_data?instrument_name=BTC-PERPETUAL&start_timestamp={start_time}&end_timestamp={end_time}&resolution={resolution}"
    res = requests.get(url).json()['result']
    
    # TV Lightweight Charts verlangt Timestamp in Sekunden (UNIX)
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
    df_raw['expiration'] = df_raw['instrument_name'].apply(lambda x: x.split('-')[1])
    df_raw['strike'] = df_raw['instrument_name'].apply(lambda x: float(x.split('-')[2]))
    df_raw['type'] = df_raw['instrument_name'].apply(lambda x: 'Call' if x.endswith('-C') else 'Put')

    for col in ['open_interest', 'volume', 'gamma', 'delta', 'mark_iv']:
        if col in df_raw.columns:
            df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce').fillna(0)

    # SIDEBAR FILTER
    st.sidebar.header("🗓️ Filter Options")
    expirations = sorted(df_raw['expiration'].unique().tolist())
    selected_exp = st.sidebar.selectbox("Select Expiration Date", ["ALL (Aggregated)"] + expirations, index=0)

    if selected_exp != "ALL (Aggregated)":
        df = df_raw[df_raw['expiration'] == selected_exp].copy()
    else:
        df = df_raw.copy()

    # GEX Berechnungen
    df['gex_oi'] = np.where(df['type'] == 'Call', df['gamma'] * df['open_interest'] * spot, -df['gamma'] * df['open_interest'] * spot) / 1e6
    
    zoom_margin = 3500
    df_filtered = df[(df['strike'] >= spot - zoom_margin) & (df['strike'] <= spot + zoom_margin)].copy()

    summary = df_filtered.groupby('strike').agg({'gex_oi': 'sum'}).reset_index()
    summary_sorted = summary.sort_values('strike').copy()
    summary_sorted['cum_gex'] = summary_sorted['gex_oi'].cumsum()
    zero_crossings = summary_sorted[np.sign(summary_sorted['cum_gex']).diff() != 0]
    gamma_flip = zero_crossings['strike'].iloc[0] if len(zero_crossings) > 0 else spot

    # Top Metrics Header
    m1, m2, m3 = st.columns(3)
    m1.metric("BTC Spot Price", f"${spot:,.1f}")
    m2.metric("Gamma Flip", f"${gamma_flip:,.0f}")
    m3.metric("Selected Expiration", selected_exp)

    st.markdown("---")

    # TRADINGVIEW LIGHTWEIGHT CHART SETUP
    candle_data = candles[['time', 'open', 'high', 'low', 'close']].to_dict('records')
    volume_data = candles[['time', 'volume', 'open', 'close']].copy()
    volume_data['color'] = np.where(volume_data['close'] >= volume_data['open'], 'rgba(38, 166, 154, 0.5)', 'rgba(239, 83, 80, 0.5)')
    volume_data = volume_data.rename(columns={'volume': 'value'}).drop(columns=['open', 'close']).to_dict('records')

    chart_options = {
        "height": 650,
        "layout": {
            "background": {"type": "solid", "color": "#131722"},
            "textColor": "#d1d4dc"
        },
        "grid": {
            "vertLines": {"color": "#2B2B43"},
            "horzLines": {"color": "#2B2B43"}
        },
        "crosshair": {"mode": 0},
        "priceScale": {"borderColor": "#555"},
        "timeScale": {"borderColor": "#555", "timeVisible": True}
    }

    # Chart Daten-Struktur für TradingView
    series_candlestick = [{
        "type": "Candlestick",
        "data": candle_data,
        "options": {
            "upColor": "#26a69a",
            "downColor": "#ef5350",
            "borderVisible": False,
            "wickUpColor": "#26a69a",
            "wickDownColor": "#ef5350"
        }
    }, {
        "type": "Histogram",
        "data": volume_data,
        "options": {
            "priceFormat": {"type": "volume"},
            "priceScaleId": ""
        },
        "priceScale": {
            "scaleMargins": {"top": 0.8, "bottom": 0}
        }
    }]

    st.subheader("📺 TradingView Lightweight Chart (BTC/USD)")
    renderLightweightCharts([{"chart": chart_options, "series": series_candlestick}], key="tv_chart")

except Exception as e:
    st.error(f"Fehler beim Laden des Dashboards: {e}")
