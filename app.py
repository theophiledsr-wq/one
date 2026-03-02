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

# --- RÉCUPÉRATION DES TICKERS EUROPÉENS (Indices Majeurs) ---
@st.cache_data
def get_european_base_list():
    tickers = []
    try:
        # On récupère les indices majeurs via Wikipedia pour peupler le menu
        indices = {
            "CAC 40 (Paris)": ("https://en.wikipedia.org/wiki/CAC_40", "Ticker", ".PA"),
            "DAX 40 (Frankfurt)": ("https://en.wikipedia.org/wiki/DAX", "Ticker", ".DE"),
            "IBEX 35 (Madrid)": ("https://en.wikipedia.org/wiki/IBEX_35", "Ticker", ".MC"),
            "FTSE MIB (Milan)": ("https://en.wikipedia.org/wiki/FTSE_MIB", "Ticker", ".MI"),
        }
        
        for name, (url, col, suffix) in indices.items():
            tables = pd.read_html(url)
            for t in tables:
                if col in t.columns:
                    raw_list = t[col].tolist()
                    # Nettoyage et ajout du suffixe si nécessaire
                    cleaned = [str(tk).split('.')[0] + suffix for tk in raw_list]
                    tickers.extend(cleaned)
                    break
        return sorted(list(set(tickers))) # Suppression des doublons
    except:
        # Fallback si le scraping échoue
        return ["AIR.PA", "MC.PA", "SAP.DE", "ASML.AS", "ITX.MC", "UCG.MI"]

BASE_TICKERS = get_european_base_list()

# --- BARRE LATÉRALE ---
with st.sidebar:
    st.header("1. Sélection des Actifs")
    
    # Menu déroulant avec la liste de base
    selected_from_list = st.multiselect(
        "Sélectionnez parmi les indices européens :", 
        options=BASE_TICKERS,
        default=["AIR.PA"] if "AIR.PA" in BASE_TICKERS else []
    )
    
    # NOUVEAU : Bouton pour rajout manuel
    st.subheader("Ajout Manuel")
    manual_ticker = st.text_input("Ajouter un ticker spécifique (ex: PUST.PA, BTC-EUR, GLE.PA) :").upper()
    
    # Gestion de la liste finale (Fusion liste + manuel)
    if 'manual_list' not in st.session_state:
        st.session_state.manual_list = []
        
    if st.button("➕ Ajouter au portefeuille"):
        if manual_ticker and manual_ticker not in st.session_state.manual_list:
            st.session_state.manual_list.append(manual_ticker)
            st.toast(f"Ticker {manual_ticker} ajouté !")

    final_ticker_list = list(set(selected_from_list + st.session_state.manual_list))
    
    if st.button("🗑️ Réinitialiser le manuel"):
        st.session_state.manual_list = []
        st.rerun()

    # Formulaire pour les paramètres restants
    with st.form("simulation_params"):
        st.divider()
        st.header("2. Quantités & Dates")
        
        shares_dict = {}
        for t in final_ticker_list:
            shares_dict[t] = st.number_input(f"Unités pour {t}", value=10, min_value=1)
        
        st.divider()
        start_date_hist = st.date_input("Historique depuis", datetime.date(2019, 1, 1))
        sim_start_date = st.date_input("Date début simulation", datetime.date(2025, 1, 1))
        
        n_days_projection = st.number_input("Horizon (jours)", value=150)
        n_simulations = st.number_input("Nombre de simulations", value=2000)
        decay_factor = st.slider("Lambda (EWMA)", 0.80, 0.99, 0.94)
        
        run_sim = st.form_submit_button("🚀 LANCER LA PROJECTION")

# --- LOGIQUE DE CALCUL (Même moteur puissant que précédemment) ---
if run_sim:
    if not final_ticker_list:
        st.error("Votre portefeuille est vide !")
        st.stop()

    fixed_shares = np.array([shares_dict[t] for t in final_ticker_list])

    with st.spinner(f"Analyse de {len(final_ticker_list)} actifs en cours..."):
        try:
            data = yf.download(final_ticker_list, start=start_date_hist.strftime("%Y-%m-%d"))['Close']
            
            # Gestion format single/multi-ticker
            if len(final_ticker_list) == 1:
                data = data.to_frame(final_ticker_list[0])
            else:
                data = data[final_ticker_list]

            data = data.ffill().dropna()
            returns = np.log(data / data.shift(1)).dropna()

            # Calibration
            sim_start_str = sim_start_date.strftime("%Y-%m-%d")
            returns_calib = returns[returns.index < sim_start_str].copy()
            
            if returns_calib.empty:
                st.error("Pas assez de données historiques pour cette période.")
                st.stop()

            start_prices = data.iloc[-1].values
            initial_val = np.sum(start_prices * fixed_shares)

            # Modèle EWMA
            ewma_var = (returns_calib**2).ewm(alpha=(1 - decay_factor), adjust=False).mean()
            hist_vol = np.sqrt(ewma_var)
            std_residuals = (returns_calib / hist_vol).values

            # Simulation FHS Dynamique
            current_vol = hist_vol.iloc[-1].values 
            sim_vols = np.tile(current_vol, (n_simulations, 1)) 
            price_paths = np.zeros((n_days_projection, n_simulations, len(final_ticker_list)))
            last_prices = np.tile(start_prices, (n_simulations, 1))

            for t in range(n_days_projection):
                idx = np.random.randint(0, len(std_residuals), size=n_simulations)
                shocks = std_residuals[idx]
                if len(final_ticker_list) == 1: shocks = shocks.reshape(-1, 1)
                
                daily_ret = shocks * sim_vols
                last_prices = last_prices * np.exp(daily_ret)
                price_paths[t] = last_prices
                sim_vols = np.sqrt(decay_factor * (sim_vols**2) + (1 - decay_factor) * (daily_ret**2))

            # Portefeuille sim
            if len(final_ticker_list) == 1:
                portfolio_sim = price_paths[:, :, 0] * fixed_shares[0]
            else:
                portfolio_sim = np.sum(price_paths * fixed_shares, axis=2)

            # Analyse
            final_pnl = portfolio_sim[-1, :] - initial_val
            esp_gain = np.mean(final_pnl)
            var_95 = np.percentile(final_pnl, 5)

            # --- VISUALISATION ---
            fig = plt.figure(figsize=(20, 10))
            gs = GridSpec(2, 2, width_ratios=[2.5, 1], height_ratios=[2, 1], hspace=0.3)

            # Graphique 1 : Trajectoires
            ax1 = fig.add_subplot(gs[0, 0])
            norm = plt.Normalize(final_pnl.min(), final_pnl.max())
            cmap = plt.cm.RdYlGn
            for i in np.random.choice(n_simulations, min(150, n_simulations), replace=False):
                ax1.plot(portfolio_sim[:, i], color=cmap(norm(final_pnl[i])), lw=0.6, alpha=0.2)
            ax1.plot(np.median(portfolio_sim, axis=1), color='black', lw=3, label='Médiane')
            ax1.set_title("Projections de la valeur du portefeuille", fontsize=14, fontweight='bold')
            ax1.set_ylabel("Valeur (€)")
            ax1.grid(alpha=0.3)
            ax1.legend()

            # Graphique 2 : Distribution
            ax2 = fig.add_subplot(gs[1, 0])
            ax2.hist(final_pnl, bins=70, color='lightsteelblue', edgecolor='white', density=True)
            ax2.axvline(var_95, color='darkred', ls='--', label=f'VaR 95% : {var_95:.0f}€')
            ax2.axvline(esp_gain, color='darkgreen', ls='-', label=f'Espérance : {esp_gain:.0f}€')
            ax2.set_title("Distribution des Profits/Pertes à l'échéance", fontweight='bold')
            ax2.legend()

            # Panneau Info
            ax_info = fig.add_subplot(gs[:, 1])
            ax_info.axis('off')
            info_txt = (
                f"ANALYSE DU PORTEFEUILLE\n"
                f"--------------------------\n"
                f"Nombre d'actifs : {len(final_ticker_list)}\n"
                f"Valeur de départ : {initial_val:,.2f} €\n"
                f"Horizon : {n_days_projection} jours\n"
                f"Simulations : {n_simulations}\n\n"
                f"STATISTIQUES FINALES\n"
                f"--------------------------\n"
                f"Probabilité de gain : {np.mean(final_pnl > 0)*100:.1f} %\n"
                f"VaR (95%) : {var_95:,.2f} €\n"
                f"Espérance : {esp_gain:,.2f} €"
            )
            ax_info.text(0, 0.95, info_txt, transform=ax_info.transAxes, fontsize=12, 
                         family='monospace', bbox=dict(facecolor='#f0f2f6', alpha=0.9))

            st.pyplot(fig)

        except Exception as e:
            st.error(f"Une erreur est survenue lors du calcul. Vérifiez les tickers saisis manuellement. Détails : {e}")

# --- FOOTER ---
st.divider()
st.caption("Outil d'analyse quantitative - Les suffixes recommandés pour Euronext : Paris (.PA), Milan (.MI), Francfort (.DE), Madrid (.MC), Hambourg (.HM).")
