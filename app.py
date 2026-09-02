# ==========================================
    # KORREKTE KENNZAHLEN LOGIK (The 1% GEX Rule)
    # ==========================================
    contract_multiplier = 1.0  
    
    # 1% GEX = Gamma * OI * Spot^2 * 0.01
    # Das zeigt das Gamma-Exposure (in $), wenn BTC sich um 1% bewegt. Skaliert in Millionen USD.
    df['gex_normal'] = np.where(df['type'] == 'Call', 
                                df['gamma'] * df['open_interest'] * (spot**2) * 0.01, 
                                -df['gamma'] * df['open_interest'] * (spot**2) * 0.01) / 1e6
                                
    df['gex_volume'] = np.where(df['type'] == 'Call', 
                                df['gamma'] * df['volume'] * (spot**2) * 0.01, 
                                -df['gamma'] * df['volume'] * (spot**2) * 0.01) / 1e6
                                
    # DEX in Millionen USD ($)
    df['dex_val'] = (df['delta'] * df['open_interest'] * spot) / 1e6
    df['oi_val']  = df['open_interest']

    y_min, y_max = spot - zoom_margin, spot + zoom_margin
    df_filtered = df[(df['strike'] >= y_min) & (df['strike'] <= y_max)]

    summary = df_filtered.groupby('strike').agg({
        'gex_normal': 'sum', 'gex_volume': 'sum', 'dex_val': 'sum', 'oi_val': 'sum', 'mark_iv': 'mean'
    }).reset_index().sort_values('strike')

    # Visuelle Skalierung
    summary['gex_norm_scaled'] = summary['gex_normal'] / div_gex
    summary['gex_vol_scaled']  = summary['gex_volume'] / div_gex
    summary['dex_val_scaled']  = summary['dex_val'] / div_dex
    summary['oi_val_scaled']   = summary['oi_val'] / div_oi

    # Call / Put OI trennen (Für Max Pain)
    calls_dict = df_filtered[df_filtered['type'] == 'Call'].groupby('strike')['open_interest'].sum().to_dict()
    puts_dict = df_filtered[df_filtered['type'] == 'Put'].groupby('strike')['open_interest'].sum().to_dict()
    summary['call_oi'] = summary['strike'].map(calls_dict).fillna(0)
    summary['put_oi'] = summary['strike'].map(puts_dict).fillna(0)

    # --- KEY LEVELS ---
    sign_changes = summary[np.sign(summary['gex_normal']).diff().fillna(0) != 0].iloc[1:]
    gammaFlip = sign_changes.iloc[(sign_changes['strike'] - spot).abs().argsort()[:1]]['strike'].values[0] if not sign_changes.empty else spot
    callWallIntra = df_filtered[df_filtered['type'] == 'Call'].groupby('strike')['gex_normal'].sum().idxmax() if not df_filtered.empty else spot
    putWallIntra  = df_filtered[df_filtered['type'] == 'Put'].groupby('strike')['gex_normal'].sum().abs().idxmax() if not df_filtered.empty else spot

    # Max Pain Logik (Payout-Summe)
    strike_arr, call_oi_arr, put_oi_arr = summary['strike'].values, summary['call_oi'].values, summary['put_oi'].values
    pains = [np.where(s > strike_arr, (s - strike_arr) * call_oi_arr, 0).sum() + np.where(strike_arr > s, (strike_arr - s) * put_oi_arr, 0).sum() for s in strike_arr]
    maxPain = strike_arr[np.argmin(pains)] if len(pains) > 0 else spot

    # ==========================================
    # LOGNORMAL EXPECTED MOVE BERECHNUNG
    # ==========================================
    atm_strike = summary.iloc[(summary['strike'] - spot).abs().argsort()[:1]]['strike'].values[0] if not summary.empty else spot
    
    # IV sauber auslesen (Wenn "ALL", nehmen wir zur Sicherheit Front-Month-Vola für den Daily Move, ansonsten die Expiry-Vola)
    if selected_exp == "ALL (Aggregated)":
        front_exp = df_raw['expiration_str'].min() # Kürzeste Laufzeit
        matching_iv = df_raw[(df_raw['expiration_str'] == front_exp) & (df_raw['strike'] == atm_strike)]['mark_iv'].values
        time_to_use = 1 / 365.0 # Fest auf 1 Tag
        move_label = "1-Day Move"
    else:
        matching_iv = summary[summary['strike'] == atm_strike]['mark_iv'].values
        time_to_use = df_filtered['t_years'].mean() # Gesamte Restlaufzeit der ausgewählten Option!
        move_label = f"Expiry Move"

    atm_iv = (matching_iv[0] / 100.0) if len(matching_iv) > 0 and not np.isnan(matching_iv[0]) else 0.55

    # Lognormal Bounds (Mathematisch korrekt für Krypto!)
    minMoveUpper = spot * np.exp(atm_iv * np.sqrt(time_to_use))
    minMoveLower = spot * np.exp(-atm_iv * np.sqrt(time_to_use))
    
    maxMoveUpper = spot * np.exp(2 * atm_iv * np.sqrt(time_to_use))
    maxMoveLower = spot * np.exp(-2 * atm_iv * np.sqrt(time_to_use))
    
    # Distanz in Dollar (Für die Anzeige im Top-Dashboard)
    exp_move_dollar = (minMoveUpper - spot)
