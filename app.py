import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timezone, timedelta
from scipy.stats import norm
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ──────────────────────────────────────────────
st.set_page_config(page_title="Deribit Options Engine v4.2", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.stApp { background-color: #0b0e14; color: #e6edf3; }
div.stMetric { background-color: #161b22; border: 1px solid #30363d; padding: 10px; border-radius: 6px; }
div.stMetric label { color: #8b949e !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("### 📊 Deribit Options Engine v4.2")
st.caption("Einheiten-sicher · kein Spot-Fallback · bessere ATM-IV · GEX(S) · instrument_name-Join · Customer/Dealer Flow")

# ──────────────────────────────────────────────
# BLACK-SCHOLES
# ──────────────────────────────────────────────
def bs_greeks(spot, strike, t_years, iv, opt_type):
    if t_years <= 1e-12 or iv <= 0 or spot <= 0 or strike <= 0 or np.isnan(t_years):
        return np.nan, np.nan, np.nan
    sqrt_t = np.sqrt(t_years)
    d1 = (np.log(spot / strike) + 0.5 * iv * iv * t_years) / (iv * sqrt_t)
    gamma = norm.pdf(d1) / (spot * iv * sqrt_t)
    delta = norm.cdf(d1) if opt_type == "Call" else norm.cdf(d1) - 1.0
    vega  = spot * sqrt_t * norm.pdf(d1) / 100.0
    return gamma, delta, vega

# ──────────────────────────────────────────────
# DATA
# ──────────────────────────────────────────────
@st.cache_data(ttl=8)
def get_spot():
    """Kein Fallback mehr. Bei Fehler → NaN."""
    try:
        r = requests.get(
            "https://www.deribit.com/api/v2/public/get_index_price?index_name=btc_usd",
            timeout=5
        )
        price = float(r.json()["result"]["index_price"])
        if not np.isfinite(price) or price <= 0:
            return np.nan
        return price
    except Exception:
        return np.nan

@st.cache_data(ttl=120)
def get_instruments():
    try:
        r = requests.get(
            "https://www.deribit.com/api/v2/public/get_instruments",
            params={"currency": "BTC", "kind": "option", "expired": "false"},
            timeout=10
        )
        df = pd.DataFrame(r.json()["result"])
        return df[["instrument_name", "expiration_timestamp", "contract_size"]].copy()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=12)
def get_book_summary():
    r = requests.get(
        "https://www.deribit.com/api/v2/public/get_book_summary_by_currency",
        params={"currency": "BTC", "kind": "option"},
        timeout=10
    )
    return pd.DataFrame(r.json()["result"])

@st.cache_data(ttl=10)
def get_recent_trades(count=800):
    try:
        r = requests.get(
            "https://www.deribit.com/api/v2/public/get_last_trades_by_currency",
            params={"currency": "BTC", "kind": "option", "count": count},
            timeout=8
        )
        return pd.DataFrame(r.json()["result"]["trades"])
    except Exception:
        return pd.DataFrame()

# ──────────────────────────────────────────────
try:
    spot = get_spot()

    # Punkt 2: Kein Fallback – lieber tot als falsch
    if not np.isfinite(spot):
        st.error("❌ BTC Spot konnte nicht von Deribit geladen werden. Dashboard gestoppt.")
        st.stop()

    df_inst = get_instruments()
    df_raw  = get_book_summary()
    now_utc = datetime.now(timezone.utc)
    current_ts = time.time()

    # Instrument Metadata
    if not df_inst.empty:
        df_raw = df_raw.merge(
            df_inst[["instrument_name", "expiration_timestamp", "contract_size"]],
            on="instrument_name", how="left"
        )
        df_raw["contract_size"] = df_raw["contract_size"].fillna(1.0)
    else:
        df_raw["expiration_timestamp"] = np.nan
        df_raw["contract_size"] = 1.0

    df_raw["expiration_str"] = df_raw["instrument_name"].apply(lambda x: x.split("-")[1])
    df_raw["strike"] = df_raw["instrument_name"].apply(lambda x: float(x.split("-")[2]))
    df_raw["type"]   = df_raw["instrument_name"].apply(lambda x: "Call" if x.endswith("-C") else "Put")

    for col in ["open_interest", "volume", "mark_iv"]:
        df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce").fillna(0.0)

    # Echte Restlaufzeit
    def remaining_years(row):
        ts = row.get("expiration_timestamp")
        if pd.isna(ts):
            try:
                exp_date = pd.to_datetime(row["expiration_str"], format="%d%b%y").tz_localize("UTC")
                ts = (exp_date.timestamp() + 8 * 3600) * 1000
            except Exception:
                return np.nan
        remaining = (ts / 1000.0) - current_ts
        return remaining / (365.25 * 86400) if remaining > 0 else np.nan

    df_raw["t_years"] = df_raw.apply(remaining_years, axis=1)
    df_raw["expiry_ts"] = df_raw["expiration_timestamp"].fillna(0) / 1000.0
    df_raw = df_raw[df_raw["t_years"].notna() & (df_raw["t_years"] > 0)].copy()

    # 0DTE = Expiry-Datum == heute UTC
    today = now_utc.date()
    tomorrow = today + timedelta(days=1)

    def dte_label(row):
        if row["expiry_ts"] <= 0:
            return "Unknown"
        exp_date = datetime.fromtimestamp(row["expiry_ts"], tz=timezone.utc).date()
        if exp_date == today:
            return "0DTE"
        if exp_date == tomorrow:
            return "1DTE"
        days = (exp_date - today).days
        if days <= 3:
            return "2-3D"
        if days <= 7:
            return "4-7D"
        if days <= 30:
            return "8-30D"
        return ">30D"

    df_raw["dte_bucket"] = df_raw.apply(dte_label, axis=1)
    df_raw["minutes_to_exp"] = df_raw["t_years"] * 365.25 * 24 * 60

    # Greeks
    g_list, d_list, v_list = [], [], []
    for _, r in df_raw.iterrows():
        g, d, v = bs_greeks(spot, r["strike"], r["t_years"], r["mark_iv"]/100.0, r["type"])
        g_list.append(g); d_list.append(d); v_list.append(v)
    df_raw["gamma"] = g_list
    df_raw["delta"] = d_list
    df_raw["vega"]  = v_list

    # ──────────────────────────────────────────────
    # SIDEBAR
    # ──────────────────────────────────────────────
    st.sidebar.markdown("### 🎛️ Filter")
    show_gex   = st.sidebar.checkbox("🟠 Estimated GEX", True)
    show_oi    = st.sidebar.checkbox("🟡 Open Interest", True)
    show_gex_s = st.sidebar.checkbox("📈 GEX(S) Kurve", True)
    show_flow  = st.sidebar.checkbox("🌊 Flow (Customer / Dealer)", True)

    flow_mode = st.sidebar.radio("Flow-Fenster", ["Trade Count", "Zeitfenster"], index=1)
    if flow_mode == "Trade Count":
        trade_count = st.sidebar.select_slider("Anzahl Trades", [200, 400, 600, 800, 1000], 600)
        time_window_min = None
    else:
        time_window_min = st.sidebar.select_slider("Minuten", [1, 5, 15, 30, 60, 120], 15)
        trade_count = 1000

    div_gex  = st.sidebar.number_input("GEX Divisor", 0.5, 50.0, 1.0, 0.5)
    div_oi   = st.sidebar.number_input("OI Divisor", 10.0, 1000.0, 100.0, 10.0)
    div_flow = st.sidebar.number_input("Flow Divisor", 0.1, 20.0, 1.0, 0.5)

    buckets = ["ALL", "0DTE", "1DTE", "2-3D", "4-7D", "8-30D", ">30D"]
    selected_bucket = st.sidebar.selectbox("📅 DTE Bucket", buckets)
    expirations = sorted(df_raw["expiration_str"].unique().tolist())
    selected_exp = st.sidebar.selectbox("🗓️ Single Expiry", ["— Bucket —"] + expirations)

    zoom = st.sidebar.slider("Zoom ± USD", 2000, 40000, 10000, 500)
    gex_s_pct = st.sidebar.slider("GEX(S) Range ± %", 5, 25, 12, 1)

    # Filter
    if selected_exp != "— Bucket —":
        df = df_raw[df_raw["expiration_str"] == selected_exp].copy()
        label = selected_exp
    else:
        if selected_bucket == "ALL":
            df = df_raw.copy()
            label = "ALL"
        else:
            df = df_raw[df_raw["dte_bucket"] == selected_bucket].copy()
            label = selected_bucket

    if df.empty:
        st.warning("Keine Daten für diesen Filter.")
        st.stop()

    min_minutes = df["minutes_to_exp"].min()
    if min_minutes < 30:
        st.warning(f"⚠️ Sehr kurze Restlaufzeit: {min_minutes:.1f} min. Gamma kann extrem werden und schnell kollabieren.")

    # ──────────────────────────────────────────────
    # ESTIMATED GEX
    # ──────────────────────────────────────────────
    # Bei BTC-Inverse-Optionen sind open_interest und amount bereits in BTC.
    # Keine Extra-Multiplikation mit contract_size.

    df["gex"] = np.where(
        df["type"] == "Call",
         df["gamma"] * df["open_interest"] * (spot**2) * 0.01,
        -df["gamma"] * df["open_interest"] * (spot**2) * 0.01
    ) / 1e6

    df["oi"] = df["open_interest"]

    summary = df.groupby("strike").agg(
        gex=("gex", "sum"),
        oi=("oi", "sum"),
        volume=("volume", "sum"),
        mark_iv=("mark_iv", "mean"),
        t_years=("t_years", "mean"),
        minutes_to_exp=("minutes_to_exp", "min")
    ).reset_index().sort_values("strike")

    call_gamma_wall = df[df["type"]=="Call"].groupby("strike")["gex"].sum().idxmax() if not df.empty else spot
    put_gamma_wall  = df[df["type"]=="Put"].groupby("strike")["gex"].sum().abs().idxmax() if not df.empty else spot

    # Max Pain
    strikes = summary["strike"].values
    call_oi = df[df["type"]=="Call"].groupby("strike")["open_interest"].sum().reindex(strikes, fill_value=0).values
    put_oi  = df[df["type"]=="Put"].groupby("strike")["open_interest"].sum().reindex(strikes, fill_value=0).values
    pains = [np.sum(np.maximum(s-strikes,0)*call_oi) + np.sum(np.maximum(strikes-s,0)*put_oi) for s in strikes]
    max_pain = strikes[np.argmin(pains)] if len(pains) else spot

    # ──────────────────────────────────────────────
    # GEX(S)
    # ──────────────────────────────────────────────
    pct = gex_s_pct / 100.0
    hypo_spots = np.linspace(spot*(1-pct), spot*(1+pct), 101)

    def compute_gex_at(s_hyp):
        net = 0.0
        for _, row in df.iterrows():
            g, _, _ = bs_greeks(s_hyp, row["strike"], row["t_years"], row["mark_iv"]/100.0, row["type"])
            if np.isnan(g):
                continue
            sign = 1.0 if row["type"] == "Call" else -1.0
            net += sign * g * row["open_interest"] * (s_hyp**2) * 0.01
        return net / 1e6

    gex_vals = [compute_gex_at(s) for s in hypo_spots]
    gex_s_df = pd.DataFrame({"spot": hypo_spots, "gex": gex_vals})

    flips = []
    for i in range(1, len(gex_s_df)):
        y0, y1 = gex_s_df["gex"].iloc[i-1], gex_s_df["gex"].iloc[i]
        if y0 * y1 < 0:
            x0, x1 = gex_s_df["spot"].iloc[i-1], gex_s_df["spot"].iloc[i]
            flip = x0 - y0 * (x1 - x0) / (y1 - y0)
            flips.append(flip)

    if flips:
        nearest_flip = min(flips, key=lambda x: abs(x - spot))
        other_flips = sorted([f for f in flips if abs(f - nearest_flip) > 50])
    else:
        nearest_flip = spot
        other_flips = []

    # ──────────────────────────────────────────────
    # FLOW – instrument_name Join + Strike-Parsing (Fix)
    # ──────────────────────────────────────────────
    df_trades = get_recent_trades(trade_count)

    flow_by_strike = pd.DataFrame()
    net_cust_gamma = 0.0
    net_dealer_gamma = 0.0
    net_cust_delta = 0.0

    if not df_trades.empty and "instrument_name" in df_trades.columns:

        # Strike aus instrument_name extrahieren (das fehlte zuvor → KeyError)
        df_trades["strike"] = df_trades["instrument_name"].apply(
            lambda x: float(str(x).split("-")[2]) if len(str(x).split("-")) >= 3 else np.nan
        )
        df_trades = df_trades.dropna(subset=["strike"])

        # Zeitfilter
        if time_window_min is not None and "timestamp" in df_trades.columns:
            cutoff = (current_ts - time_window_min * 60) * 1000
            df_trades = df_trades[df_trades["timestamp"] >= cutoff].copy()

        # Signed Volume (amount ist bereits in BTC)
        df_trades["signed_vol"] = np.where(
            df_trades["direction"].str.lower() == "buy",
            df_trades["amount"],
            -df_trades["amount"]
        )

        # Greek-Join über instrument_name
        greek_map = df.set_index("instrument_name")[["gamma", "delta", "type"]].to_dict("index")

        def attach_greeks(row):
            info = greek_map.get(row["instrument_name"])
            if info is None:
                return np.nan, np.nan, None
            return info["gamma"], info["delta"], info.get("type")

        greeks = df_trades.apply(attach_greeks, axis=1, result_type="expand")
        df_trades["gamma"] = greeks[0]
        df_trades["delta"] = greeks[1]
        df_trades["opt_type"] = greeks[2]
        df_trades = df_trades.dropna(subset=["gamma"])

        # Kein extra contract_size
        df_trades["cust_gamma_flow"] = (
            df_trades["signed_vol"] * df_trades["gamma"] * (spot**2) * 0.01
        ) / 1e6

        df_trades["cust_delta_flow"] = (
            df_trades["signed_vol"] * df_trades["delta"] * spot
        ) / 1e6

        df_trades["dealer_gamma_flow"] = -df_trades["cust_gamma_flow"]

        flow_by_strike = df_trades.groupby("strike").agg(
            signed_vol=("signed_vol", "sum"),
            cust_gamma_flow=("cust_gamma_flow", "sum"),
            dealer_gamma_flow=("dealer_gamma_flow", "sum"),
            cust_delta_flow=("cust_delta_flow", "sum"),
            trade_count=("amount", "count")
        ).reset_index().sort_values("cust_gamma_flow", key=abs, ascending=False)

        net_cust_gamma   = flow_by_strike["cust_gamma_flow"].sum()
        net_dealer_gamma = flow_by_strike["dealer_gamma_flow"].sum()
        net_cust_delta   = flow_by_strike["cust_delta_flow"].sum()

        summary = summary.merge(
            flow_by_strike[["strike", "cust_gamma_flow", "dealer_gamma_flow", "signed_vol"]],
            on="strike", how="left"
        )
        summary["cust_gamma_flow"] = summary["cust_gamma_flow"].fillna(0.0)
        summary["dealer_gamma_flow"] = summary["dealer_gamma_flow"].fillna(0.0)

    # ──────────────────────────────────────────────
    # EXPECTED MOVE – bessere ATM-IV
    # ──────────────────────────────────────────────
    atm_strike = summary.iloc[(summary["strike"] - spot).abs().argsort()[:1]]["strike"].values[0]

    call_ivs = df[(df["type"] == "Call") & (df["strike"] == atm_strike)]["mark_iv"]
    put_ivs  = df[(df["type"] == "Put")  & (df["strike"] == atm_strike)]["mark_iv"]

    if len(call_ivs) > 0 and len(put_ivs) > 0:
        atm_iv = (call_ivs.iloc[0] + put_ivs.iloc[0]) / 2.0 / 100.0
        iv_source = "Call+Put Mid"
    elif len(call_ivs) > 0:
        atm_iv = call_ivs.iloc[0] / 100.0
        iv_source = "Call only"
    elif len(put_ivs) > 0:
        atm_iv = put_ivs.iloc[0] / 100.0
        iv_source = "Put only"
    else:
        nearest_iv = summary.iloc[(summary["strike"] - spot).abs().argsort()[:1]]["mark_iv"].values[0]
        atm_iv = nearest_iv / 100.0
        iv_source = "Nearest Strike"

    t_use = max(df["t_years"].min(), 1e-8) if label in ["0DTE", "1DTE"] else 1 / 365.25
    move_label = "Restlaufzeit" if label in ["0DTE", "1DTE"] else "1-Day"
    exp_1sd = spot * atm_iv * np.sqrt(t_use)
