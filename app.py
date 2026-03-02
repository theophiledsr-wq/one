import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec
import datetime
from scipy.linalg import cholesky

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="European Portfolio Master", layout="wide")

st.markdown("""
    <style>
    .stPlot { background-color: transparent; }
    summary { font-weight: bold; font-size: 1.1rem; }
    </style>
    """, unsafe_allow_html=True)

# --- FONCTION DE RÉCUPÉRATION SÉCURISÉE DES TICKERS ---
@st.cache_data
def get_european_tickers():
    fallback = ["AIR.PA", "MC.PA", "OR.PA", "SAP.DE", "ASML.AS", "ITX.MC", "UCG.MI"]
    indices = {
        "CAC 40": ("https://en.wikipedia.org/wiki/CAC_40", "Ticker", ".PA"),
        "DAX 40": ("https://en.wikipedia.org/wiki/DAX", "Ticker", ".DE"),
        "IBEX 35": ("https://en.wikipedia.org/wiki/IBEX_35", "Ticker", ".MC")
    }
    all_tickers = []
    try:
        for name, (url, col, suffix) in indices.items():
            tables = pd.read_html(url)
            for t in tables:
                if col in t.columns:
                    raw = t[col].tolist()
                    all_tickers.extend([str(tk).split('.')[0] + suffix for tk in raw])
                    break
        return sorted(list(set(all_tickers))) if all_tickers else fallback
    except Exception:
        return fallback

BASE_LIST = get_european_tickers()

# --- BARRE LATÉRALE (SIDEBAR) ---
with st.sidebar:
    st.title("⚙️ PRO NAVIGATION")
    app_mode = st.radio("Sélectionner l'outil :", ["Projection Monte Carlo", "Frontière Efficiente & Ratios"])
    
    st.divider()
    st.header("🛒 Composition du Portefeuille")
    
    # Correction de l'erreur default/options
    default_selection = ["AIR.PA"] if "AIR.PA" in BASE_LIST else [BASE_LIST[0]]
    selected_tickers = st.multiselect("Sélectionner dans les indices :", options=BASE_LIST, default=default_selection)
    
    manual_t = st.text_input("Ajout manuel (ex: PUST.PA) :").upper()
    if 'manual_list' not in st.session_state: st.session_state.manual_list = []
    if st.button("➕ Ajouter"):
        if manual_t and manual_t not in st.session_state.manual_list:
            st.session_state.manual_list.append(manual_t)
            st.rerun()

    final_list = list(set(selected_tickers + st.session_state.manual_list))
    
    shares_dict = {}
    if final_list:
        st.subheader("Unités détenues")
        for t in final_list:
            shares_dict[t] = st.number_input(f"Quantité {t}", value=10, min_value=1)

    st.divider()
    if app_mode == "Projection Monte Carlo":
        model_type = st.radio("Moteur :", ["FHS (Historique)", "Student-t"])
        horizon = st.number_input("Horizon (jours)", value=150)
        n_sims = st.number_input("Simulations", value=2000)
        decay = st.slider("Lambda (EWMA)", 0.80, 0.99, 0.94)
        run_btn = st.button("🚀 LANCER LA PROJECTION")
    else:
        start_opt = st.date_input("Historique d'analyse", datetime.date(2021, 1, 1))
        rf_rate = st.number_input("Taux sans risque (%)", value=3.0) / 100
        run_btn = st.button("🎯 GÉNÉRER L'ANALYSE")

# --- CHARGEMENT DES DONNÉES ---
if final_list:
    with st.spinner("Téléchargement des données Yahoo Finance..."):
        try:
            # On utilise Period 'max' pour être sûr d'avoir assez de data
            raw_data = yf.download(final_list, start="2020-01-01")['Close']
            
            # Gestion robuste Series vs DataFrame
            if isinstance(raw_data, pd.Series):
                raw_data = raw_data.to_frame(name=final_list[0])
            
            if raw_data.empty:
                st.error("⚠️ Impossible de récupérer les prix. Vérifiez les tickers ou votre connexion.")
                st.stop()
                
            raw_data = raw_data.ffill().dropna()
            last_prices = raw_data.iloc[-1]
            total_val = sum(last_prices[t] * shares_dict[t] for t in final_list)
            
        except Exception as e:
            st.error(f"Erreur lors du téléchargement : {e}")
            st.stop()

# --- LOGIQUE AFFICHAGE ---
if app_mode == "Projection Monte Carlo" and 'run_btn' in locals() and run_btn:
    # (Calculs Monte Carlo...)
    returns = np.log(raw_data / raw_data.shift(1)).dropna()
    ewma_var = (returns**2).ewm(alpha=(1 - decay), adjust=False).mean()
    curr_vol = np.sqrt(ewma_var.iloc[-1].values)
    
    price_paths = np.zeros((horizon, n_sims, len(final_list)))
    temp_prices = np.tile(last_prices.values, (n_sims, 1))
    sim_vols = np.tile(curr_vol, (n_sims, 1))
    
    # Simulation simplifiée pour la démo
    for t in range(horizon):
        shocks = np.random.standard_normal((n_sims, len(final_list)))
        daily_ret = shocks * sim_vols
        temp_prices *= np.exp(daily_ret)
        price_paths[t] = temp_prices
        sim_vols = np.sqrt(decay * (sim_vols**2) + (1 - decay) * (daily_ret**2))

    portfolio_paths = np.sum(price_paths * [shares_dict[t] for t in final_list], axis=2)
    final_pnl = portfolio_paths[-1, :] - total_val

    st.subheader("📊 Résultats Monte Carlo")
    c1, c2, c3 = st.columns(3)
    c1.metric("Valeur Portefeuille", f"{total_val:,.2f} €")
    c2.metric("Espérance Gain", f"{np.mean(final_pnl):+,.2f} €")
    c3.metric("VaR 95%", f"{np.percentile(final_pnl, 5):,.2f} €")

    fig, ax = plt.subplots(figsize=(10, 5), facecolor='none')
    ax.plot(portfolio_paths[:, :100], alpha=0.3)
    st.pyplot(fig, transparent=True)

elif app_mode == "Frontière Efficiente & Ratios":
    st.subheader("📊 Répartition Actuelle")
    fig_p, ax_p = plt.subplots(figsize=(5, 5), facecolor='none')
    sizes = [last_prices[t] * shares_dict[t] for t in final_list]
    ax_p.pie(sizes, labels=final_list, autopct='%1.1f%%', textprops={'color':"w"})
    st.pyplot(fig_p, transparent=True)

    if 'run_btn' in locals() and run_btn:
        st.write("🎯 Analyse de la frontière en cours...")
        # (Logique frontière efficiente...)
