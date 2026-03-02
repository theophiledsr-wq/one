import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec
import datetime

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="European Monte Carlo Pro", layout="wide")

# --- TITRE ---
st.title("THE FRENCH BUILT TOOL FOR MONTE CARLO PROJECTION FOR THE EUROPEAN INVESTOR")

# --- RÉCUPÉRATION DES TICKERS (Indices Majeurs) ---
@st.cache_data
def get_european_base_list():
    try:
        indices = {
            "CAC 40": ("https://en.wikipedia.org/wiki/CAC_40", "Ticker", ".PA"),
            "DAX 40": ("https://en.wikipedia.org/wiki/DAX", "Ticker", ".DE"),
            "IBEX 35": ("https://en.wikipedia.org/wiki/IBEX_35", "Ticker", ".MC"),
        }
        tickers = []
        for url, col, suffix in indices.values():
            tables = pd.read_html(url)
            for t in tables:
                if col in t.columns:
                    tickers.extend([str(tk).split('.')[0] + suffix for tk in t[col].tolist()])
                    break
        return sorted(list(set(tickers)))
    except:
        return ["AIR.PA", "MC.PA", "SAP.DE", "ASML.AS"]

BASE_TICKERS = get_european_base_list()

# --- BARRE LATÉRALE ---
with st.sidebar:
    st.header("🛒 Portefeuille")
    selected_from_list = st.multiselect("Indices européens :", options=BASE_TICKERS, default=["AIR.PA"])
    
    manual_ticker = st.text_input("Ajout manuel (ex: PUST.PA) :").upper()
    if 'manual_list' not in st.session_state: st.session_state.manual_list = []
    if st.button("➕ Ajouter"):
        if manual_ticker and manual_ticker not in st.session_state.manual_list:
            st.session_state.manual_list.append(manual_ticker)

    final_ticker_list = list(set(selected_from_list + st.session_state.manual_list))
    
    with st.form("simulation_params"):
        st.divider()
        shares_dict = {t: st.number_input(f"Unités {t}", value=10, min_value=1) for t in final_ticker_list}
        st.divider()
        start_date_hist = st.date_input("Historique depuis", datetime.date(2019, 1, 1))
        sim_start_date = st.date_input("Date début simulation", datetime.date(2025, 1, 1))
        n_days_projection = st.number_input("Horizon (jours)", value=150)
        n_simulations = st.number_input("Nb Simulations", value=2000)
        decay_factor = st.slider("Lambda (EWMA)", 0.80, 0.99, 0.94)
        run_sim = st.form_submit_button("🚀 SIMULER")

# --- CALCULS ET AFFICHAGE ---
if run_sim and final_ticker_list:
    with st.spinner("Récupération des prix et analyse..."):
        # 1. Acquisition des données
        data = yf.download(final_ticker_list, start=start_date_hist.strftime("%Y-%m-%d"))['Close']
        if len(final_ticker_list) == 1: data = data.to_frame(final_ticker_list[0])
        data = data.ffill().dropna()
        
        # Récupération des derniers prix pour le tableau récapitulatif
        last_prices = data.iloc[-1]
        
        # 2. Moteur de Simulation (FHS Dynamique)
        returns = np.log(data / data.shift(1)).dropna()
        returns_calib = returns[returns.index < sim_start_date.strftime("%Y-%m-%d")].copy()
        
        start_prices_sim = data.iloc[-1].values
        fixed_shares = np.array([shares_dict[t] for t in final_ticker_list])
        initial_val = np.sum(start_prices_sim * fixed_shares)

        ewma_var = (returns_calib**2).ewm(alpha=(1 - decay_factor), adjust=False).mean()
        hist_vol = np.sqrt(ewma_var)
        std_residuals = (returns_calib / hist_vol).values
        sim_vols = np.tile(hist_vol.iloc[-1].values, (n_simulations, 1))
        price_paths = np.zeros((n_days_projection, n_simulations, len(final_ticker_list)))
        temp_prices = np.tile(start_prices_sim, (n_simulations, 1))

        for t in range(n_days_projection):
            idx = np.random.randint(0, len(std_residuals), size=n_simulations)
            shocks = std_residuals[idx]
            if len(final_ticker_list) == 1: shocks = shocks.reshape(-1, 1)
            daily_ret = shocks * sim_vols
            temp_prices *= np.exp(daily_ret)
            price_paths[t] = temp_prices
            sim_vols = np.sqrt(decay_factor * (sim_vols**2) + (1 - decay_factor) * (daily_ret**2))

        portfolio_sim = np.sum(price_paths * fixed_shares, axis=2) if len(final_ticker_list) > 1 else price_paths[:, :, 0] * fixed_shares[0]
        final_pnl = portfolio_sim[-1, :] - initial_val

        # ==========================================
        # 3. AFFICHAGE DES KPIs (METRICS)
        # ==========================================
        st.subheader("📊 Performance de la Stratégie")
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Valeur Initiale", f"{initial_val:,.2f} €")
        with c2:
            esp_gain = np.mean(final_pnl)
            st.metric("Espérance", f"{esp_gain:+,.2f} €", f"{(esp_gain/initial_val)*100:.2f} %")
        with c3: st.metric("Probabilité Profit", f"{np.mean(final_pnl > 0)*100:.1f} %")
        with c4: st.metric("VaR (95%)", f"{np.percentile(final_pnl, 5):,.2f} €", delta_color="inverse")

        # ==========================================
        # 4. NOUVEAU : TABLEAU DES PRIX DU JOUR
        # ==========================================
        st.write("### 📋 Détails du Portefeuille (Prix du jour)")
        
        df_summary = pd.DataFrame({
            "Ticker": final_ticker_list,
            "Prix Unitaire (€)": [last_prices[t] for t in final_ticker_list],
            "Quantité": [shares_dict[t] for t in final_ticker_list],
            "Valeur de la ligne (€)": [last_prices[t] * shares_dict[t] for t in final_ticker_list]
        })
        
        # Style du tableau pour un rendu propre
        st.dataframe(
            df_summary.style.format({
                "Prix Unitaire (€)": "{:,.2f}",
                "Valeur de la ligne (€)": "{:,.2f}"
            }),
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        # ==========================================
        # 5. GRAPHIQUE MATPLOTLIB ÉPURÉ
        # ==========================================
        plt.rcParams.update({
            "text.color": "#808495", "axes.labelcolor": "#808495", "axes.edgecolor": "#262730",
            "xtick.color": "#808495", "ytick.color": "#808495", "grid.color": "#262730"
        })

        fig = plt.figure(figsize=(16, 7), facecolor='none')
        gs = GridSpec(1, 2, width_ratios=[1.8, 1], wspace=0.2)

        # Trajectoires
        ax1 = fig.add_subplot(gs[0], facecolor='none')
        norm = plt.Normalize(final_pnl.min(), final_pnl.max())
        cmap = plt.cm.RdYlGn
        sample_idx = np.random.choice(n_simulations, min(200, n_simulations), replace=False)
        for i in sample_idx:
            ax1.plot(portfolio_sim[:, i], color=cmap(norm(final_pnl[i])), lw=0.8, alpha=0.3)
        ax1.plot(np.median(portfolio_sim, axis=1), color='white', lw=3, label='Médiane')
        ax1.set_title("TRAJECTOIRES SIMULÉES", fontsize=12, fontweight='bold', color='white')
        ax1.grid(True, ls=':', alpha=0.3)
        ax1.spines[['top', 'right']].set_visible(False)

        # Distribution
        ax2 = fig.add_subplot(gs[1], facecolor='none')
        n_bins, bins, patches = ax2.hist(final_pnl, bins=50, density=True, alpha=0.7)
        for b, p in zip(bins, patches): p.set_facecolor(cmap(norm(b)))
        ax2.axvline(np.percentile(final_pnl, 5), color='#FF4B4B', ls='--', lw=2)
        ax2.set_title("DISTRIBUTION DES P&L", fontsize=12, fontweight='bold', color='white')
        ax2.spines[['top', 'right']].set_visible(False)

        st.pyplot(fig, transparent=True)

# --- FOOTER ---
st.divider()
st.caption("Données fournies par Yahoo Finance. Simulation basée sur la volatilité historique EWMA.")
