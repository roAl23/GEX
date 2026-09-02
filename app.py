import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timezone, timedelta
from scipy.stats import norm
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Deribit Options Engine v4.4.1", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.stApp { background-color: #0b0e14; color: #e6edf3; }
div.stMetric { background-color: #161b22; border: 1px solid #30363d; padding: 10px; border-radius: 6px; }
div.stMetric label { color: #8b949e !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("### 📊 Deribit Options Engine v4.4.1")
st.caption(
    "Estimated GEX (Call+/Put−) · Estimated DEX · Vega Exposure · Max Pain · Expected Move · "
    "Static-IV GEX(S) · Aggressor / Counterparty Flow"
)

def bs_greeks(spot, strike, t_years, iv, opt_type):
    if t_years <= 1e-12 or iv <= 0 or spot <= 0 or strike <= 0 or np.isnan(t_years):
        return np.nan, np.nan, np.nan
    sqrt_t = np.sqrt(t_years)
    d1 = (np.log(spot / strike) + 0.5 * iv * iv * t_years) / (iv * sqrt_t)
    gamma = norm.pdf(d1) / (spot * iv * sqrt_t)
    delta = norm.cdf(d1) if opt_type == "Call" else norm.cdf(d1) - 1.0
    vega = spot * sqrt_t * norm.pdf(d1) / 100.0  # pro 1 IV-Punkt
    return gamma, delta, vega

@st.cache_data(ttl=8)
def get_spot():
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

try:
    spot = get_spot()
    if not np.isfinite(spot):
        st.error("BTC Spot konnte nicht geladen werden. Dashboard gestoppt.")
        st.stop()

    df_inst = get_instruments()
    df_raw = get_book_summary()
    now_utc = datetime.now(timezone.utc)
    current_ts = time.time()

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
    df_raw["type"] = df_raw["instrument_name"].apply(lambda x: "Call" if x.endswith("-C") else "Put")

    for col in ["open_interest", "volume", "mark_iv"]:
        df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce").fillna(0.0)

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

    g_list, d_list, v_list = [], [], []
    for _, r in df_raw.iterrows():
        g, d, v = bs_greeks(spot, r["strike"], r["t_years"], r["mark_iv"] / 100.0, r["type"])
        g_list.append(g)
        d_list.append(d)
        v_list.append(v)
    df_raw["gamma"] = g_list
    df_raw["delta"] = d_list
    df_raw["vega"] = v_list

    # ── Sidebar ──
    st.sidebar.markdown("### Filter")
    show_gex = st.sidebar.checkbox("Est. GEX (Call+/Put−)", True)
    show_dex = st.sidebar.checkbox("Est. DEX (OI×Δ×S)", True)
    show_vega_exp = st.sidebar.checkbox("Vega Exposure", False)
    show_oi = st.sidebar.checkbox("Open Interest", True)
    show_gex_s = st.sidebar.checkbox("Static-IV GEX(S)", True)
    show_flow = st.sidebar.checkbox("Aggressor Flow", True)

    flow_mode = st.sidebar.radio("Flow-Fenster", ["Trade Count", "Zeitfenster"], index=1)
    if flow_mode == "Trade Count":
        trade_count = st.sidebar.select_slider("Anzahl Trades", [200, 400, 600, 800, 1000], 600)
        time_window_min = None
    else:
        time_window_min = st.sidebar.select_slider("Minuten", [1, 5, 15, 30, 60, 120], 15)
        trade_count = 1000

    div_gex = st.sidebar.number_input("GEX Divisor", 0.5, 50.0, 1.0, 0.5)
    div_dex = st.sidebar.number_input("DEX Divisor", 0.1, 50.0, 1.0, 0.5)
    div_vega = st.sidebar.number_input("Vega Exp. Divisor", 0.1, 50.0, 1.0, 0.5)
    div_oi = st.sidebar.number_input("OI Divisor", 10.0, 1000.0, 100.0, 10.0)
    div_flow = st.sidebar.number_input("Flow Divisor", 0.1, 20.0, 1.0, 0.5)

    buckets = ["ALL", "0DTE", "1DTE", "2-3D", "4-7D", "8-30D", ">30D"]
    selected_bucket = st.sidebar.selectbox("DTE Bucket", buckets)
    expirations = sorted(df_raw["expiration_str"].unique().tolist())
    selected_exp = st.sidebar.selectbox("Single Expiry", ["— Bucket —"] + expirations)

    zoom = st.sidebar.slider("Zoom USD", 2000, 40000, 10000, 500)
    gex_s_pct = st.sidebar.slider("GEX(S) Range %", 5, 25, 12, 1)

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
        st.warning(f"Sehr kurze Restlaufzeit: {min_minutes:.1f} min. Gamma kann extrem werden.")

    # ── Exposure ──
    df["gex"] = np.where(
        df["type"] == "Call",
        df["gamma"] * df["open_interest"] * (spot ** 2) * 0.01,
        -df["gamma"] * df["open_interest"] * (spot ** 2) * 0.01
    ) / 1e6

    df["dex"] = (df["delta"] * df["open_interest"] * spot) / 1e6
    df["vega_exp"] = (df["vega"] * df["open_interest"]) / 1e6
    df["oi"] = df["open_interest"]

    summary = df.groupby("strike").agg(
        gex=("gex", "sum"),
        dex=("dex", "sum"),
        vega_exp=("vega_exp", "sum"),
        oi=("oi", "sum"),
        volume=("volume", "sum"),
        mark_iv=("mark_iv", "mean"),
        t_years=("t_years", "mean"),
        minutes_to_exp=("minutes_to_exp", "min")
    ).reset_index().sort_values("strike")

    call_gamma_wall = (
        df[df["type"] == "Call"].groupby("strike")["gex"].sum().idxmax()
        if not df.empty else spot
    )
    put_gamma_wall = (
        df[df["type"] == "Put"].groupby("strike")["gex"].sum().abs().idxmax()
        if not df.empty else spot
    )

    # Max Pain
    strikes = summary["strike"].values
    call_oi = df[df["type"] == "Call"].groupby("strike")["open_interest"].sum().reindex(strikes, fill_value=0).values
    put_oi = df[df["type"] == "Put"].groupby("strike")["open_interest"].sum().reindex(strikes, fill_value=0).values
    pains = [
        np.sum(np.maximum(s - strikes, 0) * call_oi) + np.sum(np.maximum(strikes - s, 0) * put_oi)
        for s in strikes
    ]
    max_pain = strikes[np.argmin(pains)] if len(pains) else spot

    # Static-IV GEX(S)
    pct = gex_s_pct / 100.0
    hypo_spots = np.linspace(spot * (1 - pct), spot * (1 + pct), 101)

    def compute_gex_at(s_hyp):
        net = 0.0
        for _, row in df.iterrows():
            g, _, _ = bs_greeks(s_hyp, row["strike"], row["t_years"], row["mark_iv"] / 100.0, row["type"])
            if np.isnan(g):
                continue
            sign = 1.0 if row["type"] == "Call" else -1.0
            net += sign * g * row["open_interest"] * (s_hyp ** 2) * 0.01
        return net / 1e6

    gex_vals = [compute_gex_at(s) for s in hypo_spots]
    gex_s_df = pd.DataFrame({"spot": hypo_spots, "gex": gex_vals})

    flips = []
    for i in range(1, len(gex_s_df)):
        y0 = gex_s_df["gex"].iloc[i - 1]
        y1 = gex_s_df["gex"].iloc[i]
        if y0 * y1 < 0:
            x0 = gex_s_df["spot"].iloc[i - 1]
            x1 = gex_s_df["spot"].iloc[i]
            flip = x0 - y0 * (x1 - x0) / (y1 - y0)
            flips.append(flip)

    if flips:
        nearest_flip = min(flips, key=lambda x: abs(x - spot))
        other_flips = sorted([f for f in flips if abs(f - nearest_flip) > 50])
    else:
        nearest_flip = spot
        other_flips = []

    # Aggressor Flow
    df_trades = get_recent_trades(trade_count)
    flow_by_strike = pd.DataFrame()
    net_agg_gamma = 0.0
    net_cp_gamma = 0.0
    net_agg_delta = 0.0

    if not df_trades.empty and "instrument_name" in df_trades.columns:
        df_trades["strike"] = df_trades["instrument_name"].apply(
            lambda x: float(str(x).split("-")[2]) if len(str(x).split("-")) >= 3 else np.nan
        )
        df_trades = df_trades.dropna(subset=["strike"])

        if time_window_min is not None and "timestamp" in df_trades.columns:
            cutoff = (current_ts - time_window_min * 60) * 1000
            df_trades = df_trades[df_trades["timestamp"] >= cutoff].copy()

        df_trades["signed_vol"] = np.where(
            df_trades["direction"].str.lower() == "buy",
            df_trades["amount"],
            -df_trades["amount"]
        )

        greek_map = df.set_index("instrument_name")[["gamma", "delta", "type"]].to_dict("index")

        def attach_greeks(row):
            info = greek_map.get(row["instrument_name"])
            if info is None:
                return np.nan, np.nan
            return info["gamma"], info["delta"]

        greeks = df_trades.apply(attach_greeks, axis=1, result_type="expand")
        df_trades["gamma"] = greeks[0]
        df_trades["delta"] = greeks[1]
        df_trades = df_trades.dropna(subset=["gamma"])

        df_trades["agg_gamma_flow"] = (
            df_trades["signed_vol"] * df_trades["gamma"] * (spot ** 2) * 0.01
        ) / 1e6
        df_trades["agg_delta_flow"] = (
            df_trades["signed_vol"] * df_trades["delta"] * spot
        ) / 1e6
        df_trades["cp_gamma_flow"] = -df_trades["agg_gamma_flow"]

        flow_by_strike = df_trades.groupby("strike").agg(
            signed_vol=("signed_vol", "sum"),
            agg_gamma_flow=("agg_gamma_flow", "sum"),
            cp_gamma_flow=("cp_gamma_flow", "sum"),
            agg_delta_flow=("agg_delta_flow", "sum"),
            trade_count=("amount", "count")
        ).reset_index().sort_values("agg_gamma_flow", key=abs, ascending=False)

        net_agg_gamma = flow_by_strike["agg_gamma_flow"].sum()
        net_cp_gamma = flow_by_strike["cp_gamma_flow"].sum()
        net_agg_delta = flow_by_strike["agg_delta_flow"].sum()

        summary = summary.merge(
            flow_by_strike[["strike", "agg_gamma_flow", "cp_gamma_flow", "signed_vol"]],
            on="strike", how="left"
        )
        summary["agg_gamma_flow"] = summary["agg_gamma_flow"].fillna(0.0)
        summary["cp_gamma_flow"] = summary["cp_gamma_flow"].fillna(0.0)

    # ATM IV & Expected Move
    atm_strike = summary.iloc[(summary["strike"] - spot).abs().argsort()[:1]]["strike"].values[0]
    call_ivs = df[(df["type"] == "Call") & (df["strike"] == atm_strike)]["mark_iv"]
    put_ivs = df[(df["type"] == "Put") & (df["strike"] == atm_strike)]["mark_iv"]

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

    net_gex = summary["gex"].sum()
    net_dex = summary["dex"].sum()
    net_vega_exp = summary["vega_exp"].sum()
    regime = "Positive Est. GEX" if net_gex > 0 else "Negative Est. GEX"

    # ── Metrics ──
    st.markdown("#### Key Levels & Exposure")
    c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)
    c1.metric("BTC Spot", f"${spot:,.0f}")
    c2.metric("Max Pain", f"${max_pain:,.0f}")
    c3.metric("Gamma Flip*", f"${nearest_flip:,.0f}")
    c4.metric("Call G-Wall", f"${call_gamma_wall:,.0f}")
    c5.metric("Put G-Wall", f"${put_gamma_wall:,.0f}")
    c6.metric(f"±1 SD ({move_label})", f"±${exp_1sd:,.0f}")
    c7.metric("Net Est. GEX", f"{net_gex:+.1f} M")
    c8.metric("Net Est. DEX", f"{net_dex:+.1f} M")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Net Vega Exposure", f"{net_vega_exp:+.2f} M")
    m2.metric("Aggressor Γ-Flow", f"{net_agg_gamma:+.2f} M")
    m3.metric("Est. Counterparty Γ-Flow", f"{net_cp_gamma:+.2f} M")
    m4.metric("Aggressor Δ-Flow", f"{net_agg_delta:+.2f} M")

    st.caption("*Gamma Flip aus Static-IV GEX(S) · Call+/Put−-Annahme")

    if other_flips:
        st.caption("Weitere Static-IV Flips: " + ", ".join([f"${f:,.0f}" for f in other_flips[:4]]))

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"""
**Filter:** `{label}`  
**Max Pain:** `${max_pain:,.0f}`  
**Net Est. GEX:** `{net_gex:+.1f} M$`  
**Net Est. DEX:** `{net_dex:+.1f} M$`  
**Net Vega Exp.:** `{net_vega_exp:+.2f} M$`  
**ATM IV ({iv_source}):** `{atm_iv*100:.1f}%`  
**EM ±1 SD:** `±${exp_1sd:,.0f}`  
**Regime:** {regime}
""")

    # ── Charts ──
    y_min, y_max = spot - zoom, spot + zoom
    sum_zoom = summary[(summary["strike"] >= y_min) & (summary["strike"] <= y_max)].copy()

    n_rows = 2 + (1 if show_gex_s else 0)
    heights = [0.42, 0.28, 0.30] if show_gex_s else [0.55, 0.45]

    fig = make_subplots(
        rows=n_rows, cols=1, shared_xaxes=False, vertical_spacing=0.07,
        row_heights=heights[:n_rows],
        subplot_titles=(
            f"Strike Profile · {label}",
            "Aggressor Gamma Flow (selected filter only)",
            "Static-IV GEX(S)"
        )[:n_rows]
    )

    if show_gex:
        fig.add_trace(go.Bar(
            x=sum_zoom["strike"], y=sum_zoom["gex"] / div_gex,
            customdata=sum_zoom["gex"], name="Est. GEX (Call+/Put−)",
            marker_color="#ff9800",
            hovertemplate="Strike $%{x}<br>Est. GEX %{customdata:,.2f} M$<extra></extra>"
        ), row=1, col=1)

    if show_dex:
        fig.add_trace(go.Bar(
            x=sum_zoom["strike"], y=sum_zoom["dex"] / div_dex,
            customdata=sum_zoom["dex"], name="Est. DEX (OI×Δ×S)",
            marker_color="#ab47bc",
            hovertemplate="Strike $%{x}<br>Est. DEX %{customdata:,.2f} M$<extra></extra>"
        ), row=1, col=1)

    if show_vega_exp:
        fig.add_trace(go.Bar(
            x=sum_zoom["strike"], y=sum_zoom["vega_exp"] / div_vega,
            customdata=sum_zoom["vega_exp"], name="Vega Exposure",
            marker_color="#26c6da",
            hovertemplate="Strike $%{x}<br>Vega Exp. %{customdata:,.3f} M$<extra></extra>"
        ), row=1, col=1)

    if show_oi:
        fig.add_trace(go.Bar(
            x=sum_zoom["strike"], y=sum_zoom["oi"] / div_oi,
            customdata=sum_zoom["oi"], name="OI",
            marker_color="#ffeb3b",
            hovertemplate="Strike $%{x}<br>OI %{customdata:,.0f}<extra></extra>"
        ), row=1, col=1)

    upper_sd = spot + exp_1sd
    lower_sd = spot - exp_1sd

    for val, name, color, style, width in [
        (spot, "SPOT", "#fff", "solid", 2),
        (nearest_flip, "FLIP*", "#ffeb3b", "solid", 2),
        (max_pain, "Max Pain", "#ab47bc", "dash", 2),
        (call_gamma_wall, "Call Wall", "#ff5252", "dot", 2),
        (put_gamma_wall, "Put Wall", "#66bb6a", "dot", 2),
        (upper_sd, "+1 SD", "#ffa726", "dash", 1),
        (lower_sd, "-1 SD", "#ffa726", "dash", 1),
    ]:
        fig.add_vline(
            x=val, line_dash=style, line_color=color, line_width=width,
            annotation_text=name, annotation_font_color=color, row=1, col=1
        )

    if show_flow and "agg_gamma_flow" in sum_zoom.columns:
        colors = ["#00e676" if v >= 0 else "#ff5252" for v in sum_zoom["agg_gamma_flow"]]
        fig.add_trace(go.Bar(
            x=sum_zoom["strike"], y=sum_zoom["agg_gamma_flow"] / div_flow,
            customdata=sum_zoom[["agg_gamma_flow", "cp_gamma_flow"]],
            name="Aggressor Γ-Flow",
            marker_color=colors,
            hovertemplate="Strike $%{x}<br>Agg Γ %{customdata[0]:+.3f}<br>CP Γ %{customdata[1]:+.3f}<extra></extra>"
        ), row=2, col=1)
        fig.add_hline(y=0, line_color="#666", row=2, col=1)
        fig.add_vline(x=spot, line_color="#fff", line_width=1.5, row=2, col=1)
        fig.add_vline(x=nearest_flip, line_color="#ffeb3b", line_width=1.5, row=2, col=1)

    if show_gex_s:
        r = 3 if show_flow else 2
        fig.add_trace(go.Scatter(
            x=gex_s_df["spot"], y=gex_s_df["gex"], mode="lines", name="Static-IV GEX(S)",
            line=dict(color="#00e676", width=2.5),
            fill="tozeroy", fillcolor="rgba(0,230,118,0.12)",
            hovertemplate="Hypo Spot $%{x:,.0f}<br>GEX %{y:,.2f} M$<extra></extra>"
        ), row=r, col=1)
        fig.add_hline(y=0, line_color="#888", row=r, col=1)
        fig.add_vline(x=spot, line_color="#fff", line_width=2, row=r, col=1)
        fig.add_vline(x=nearest_flip, line_color="#ffeb3b", line_width=2.5, row=r, col=1)
        fig.add_vline(x=max_pain, line_color="#ab47bc", line_dash="dash", line_width=1.5, row=r, col=1)
        for f in other_flips[:3]:
            fig.add_vline(x=f, line_color="#ffeb3b", line_dash="dot", line_width=1, row=r, col=1)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b0e14",
        plot_bgcolor="#11141d",
        height=900 if show_gex_s else 700,
        barmode="group",
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
        margin=dict(l=40, r=40, t=70, b=40)
    )
    fig.update_xaxes(tickformat="$,.0f", gridcolor="#21262d")
    fig.update_yaxes(gridcolor="#21262d")
    st.plotly_chart(fig, use_container_width=True)

    # ── Tables ──
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Top Aggressor Gamma Flow")
        st.caption(f"Nur Trades, die zum Filter **{label}** passen · Aggressor = direction buy/sell")
        if not flow_by_strike.empty:
            top = flow_by_strike.head(12).copy()
            top["Dir"] = top["agg_gamma_flow"].apply(lambda x: "Buy" if x > 0 else "Sell")
            st.dataframe(
                top[["strike", "signed_vol", "agg_gamma_flow", "cp_gamma_flow", "agg_delta_flow", "trade_count", "Dir"]]
                .rename(columns={
                    "strike": "Strike",
                    "signed_vol": "Signed Vol",
                    "agg_gamma_flow": "Agg Γ",
                    "cp_gamma_flow": "CP Γ",
                    "agg_delta_flow": "Agg Δ",
                    "trade_count": "#Trades"
                }).style.format({
                    "Strike": "${:,.0f}",
                    "Signed Vol": "{:+.2f}",
                    "Agg Γ": "{:+.3f}",
                    "CP Γ": "{:+.3f}",
                    "Agg Δ": "{:+.2f}"
                }),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("Keine Trades im Fenster / Filter.")

    with col2:
        st.markdown("#### Definitionen & Annahmen")
        st.markdown(f"""
**Exposure**
- **Est. GEX** = γ × OI × S² × 0.01 → modelliertes $ bei +1 % Spot  
  Annahme: **Call + / Put −** (kein beobachtetes Dealer-Gamma)
- **Est. DEX** = Δ × OI × S → Delta-Exposure aus OI  
  **keine** Call+/Put−-Umsignierung
- **Vega Exposure** = ν × OI → **kein Volga / kein VEX**

**Einheiten:** Bei BTC-Inverse sind OI/amount bereits in BTC.  
`contract_size` ≈ 1 → wird **nicht** extra multipliziert.

**Flow**
- **Aggressor** = `direction` buy/sell (nimmt Liquidität)
- **Est. Counterparty** = −Aggressor (Annahme)
- Flow nur für den **gewählten** Bucket/Expiry

**Sonstiges**
- ATM-IV: `{iv_source}`
- Expected Move ±1 SD: ±${exp_1sd:,.0f} ({move_label})
- Max Pain: `${max_pain:,.0f}` (strukturelle Referenz, kein Predictor)
- GEX(S): **Static-IV** (IV bleibt bei hypothetischem Spot konstant)
""")

    st.info(
        "v4.4.1: Ehrliche Nomenklatur — Est. GEX (Call+/Put−) · Est. DEX · Vega Exposure "
        "(nicht VEX) · Aggressor/Counterparty Flow · Static-IV GEX(S). "
        "Research-Dashboard, keine belastbare Dealer-Position."
    )

except Exception as e:
    st.error(f"Fehler: {e}")
    st.exception(e)
