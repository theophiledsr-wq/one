import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec
import datetime
from scipy.linalg import cholesky

# --- CONFIGURATION ---
st.set_page_config(page_title="European Portfolio Master Pro", layout="wide")
st.title("THE FRENCH BUILT TOOL FOR STRATEGIC INVESTING")

# --- RÉCUPÉRATION SÉCURISÉE DES TICKERS ---
@st.cache_data
def get_safe_tickers():
    fallback = ["AIR.PA", "MC.PA", "OR.PA", "ASML.AS", "SAP.DE"]
    try:
        # Tentative de scraping (CAC 40 par exemple)
        url = "https://en.wikipedia.org/wiki/CAC_40"
        table = pd.read_html(url)[2]
        tickers = [str(t).replace(" ", "") + ".PA" for t in table['Ticker'].tolist()]
        return sorted(list(set(tickers)))
    except:
        return fallback

BASE_LIST = get_safe_tickers()

# --- BARRE LATÉRALE ---
with st.sidebar:
    st.header("🛒 Portefeuille")
    
    # Correction Erreur 1 : On s'assure que la valeur par défaut existe dans la liste
    default_val = [BASE_LIST[0]] if BASE_LIST else []
    if "AIR.PA" in BASE_LIST: default_val = ["AIR.PA"]
    
    selected_tickers = st.multiselect("Sélectionner dans les indices :", options=BASE_LIST, default=default_val)
    
    manual_t = st.text_input("Ajout manuel (ex: PUST.PA) :").upper()
    if 'manual_list' not in st.session_state: st.session_state.manual_list = []
    if st.button("➕ Ajouter"):
        if manual_t and manual_t not in st.session_state.manual_list:
            st.session_state.manual_list.append(manual_t)
            st.rerun()

    final_list = list(set(selected_tickers + st.session_state.manual_list))
    
    if not final_list:
        st.warning("Ajoutez au moins un actif pour continuer.")
        st.stop()

    shares_dict = {t: st.number_input(f"Quantité {t}", value=10, min_value=1, key=f"q_{t}") for t in final_list}
    
    st.divider()
    app_mode = st.radio("Mode :", ["Monte Carlo", "Optimisation"])
    run_btn = st.button("🚀 LANCER L'ANALYSE")

# --- CHARGEMENT DES DONNÉES ---
@st.cache_data
def load_market_data(tickers):
    # On télécharge les actifs + le S&P 500 pour le benchmark
    all_tickers = list(set(tickers + ["^GSPC"]))
    df = yf.download(all_tickers, start="2019-01-01")['Close']
    
    # Correction Erreur 2 : Toujours forcer en DataFrame même si un seul ticker
    if isinstance(df, pd.Series):
        df = df.to_frame()
    return df.ffill().dropna()

raw_df = load_market_data(final_list)

# Vérification après téléchargement
available_assets = [t for t in final_list if t in raw_df.columns]
if not available_assets:
    st.error("Aucune donnée disponible pour les tickers sélectionnés.")
    st.stop()

data = raw_df[available_assets]
sp500 = raw_df["^GSPC"] if "^GSPC" in raw_df.columns else None

# --- CALCULS DE BASE ---
last_prices = data.iloc[-1]
portfolio_value = sum(last_prices[t] * shares_dict[t] for t in available_assets)
current_weights = np.array([(last_prices[t] * shares_dict[t]) / portfolio_value for t in available_assets])

# --- LOGIQUE MONTE CARLO ---
if app_mode == "Monte Carlo" and run_btn:
    st.subheader(f"📈 Simulation Monte Carlo")
    vol_date = data.index[-1].date()
    st.info(f"⚡ Volatilité EWMA (λ=0.94) fixée au : **{vol_date}**")

    returns = np.log(data / data.shift(1)).dropna()
    # Correction Erreur 4 : Vérifier qu'on a assez de données pour l'échantillonnage
    if len(returns) < 10:
        st.error("Pas assez d'historique pour simuler.")
        st.stop()

    decay = 0.94
    ewma_var = (returns**2).ewm(alpha=(1 - decay), adjust=False).mean()
    curr_vol = np.sqrt(ewma_var.iloc[-1].values)
    
    # Paramètres de simulation
    n_days, n_sims = 150, 2000
    price_paths = np.zeros((n_days, n_sims, len(available_assets)))
    temp_prices = np.tile(last_prices.values, (n_sims, 1))
    sim_vols = np.tile(curr_vol, (n_sims, 1))

    # Simulation FHS simple
    for t in range(n_days):
        shocks = returns.sample(n_sims, replace=True).values / np.sqrt(ewma_var.sample(n_sims).values)
        daily_ret = shocks * sim_vols
        temp_prices *= np.exp(daily_ret)
        price_paths[t] = temp_prices
        sim_vols = np.sqrt(decay * (sim_vols**2) + (1 - decay) * (daily_ret**2))

    portfolio_paths = np.sum(price_paths * [shares_dict[t] for t in available_assets], axis=2)
    final_pnl = portfolio_paths[-1, :] - portfolio_value

    # Graphiques
    fig = plt.figure(figsize=(12, 6), facecolor='none')
    plt.rcParams.update({"text.color": "white", "axes.labelcolor": "white", "xtick.color": "white", "ytick.color": "white"})
    
    gs = GridSpec(1, 2, width_ratios=[2, 1])
    ax1 = fig.add_subplot(gs[0], facecolor='none')
    ax1.plot(portfolio_paths[:, :100], alpha=0.3)
    ax1.set_title("100 Trajectoires Simulées")

    ax2 = fig.add_subplot(gs[1], facecolor='none')
    n, bins, patches = ax2.hist(final_pnl, bins=50, alpha=0.8)
    for b, p in zip(bins, patches):
        p.set_facecolor('red' if b < 0 else 'green')
    ax2.set_title("Distribution des Issues")
    st.pyplot(fig, transparent=True)

# --- LOGIQUE OPTIMISATION ---
elif app_mode == "Optimisation" and run_btn:
    st.subheader("🎯 Frontière Efficiente")
    
    # Camembert de départ
    col1, col2 = st.columns([1, 1])
    with col1:
        fig_pie, ax_pie
