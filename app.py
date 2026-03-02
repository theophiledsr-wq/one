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

# CSS pour le rendu propre
st.markdown("""
    <style>
    .stPlot { background-color: transparent; }
    summary { font-weight: bold; font-size: 1.1rem; }
    </style>
    """, unsafe_allow_html=True)

# --- FONCTION DE RÉCUPÉRATION DES TICKERS (WIKIPEDIA) ---
# --- FONCTION DE RÉCUPÉRATION SÉCURISÉE ---
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
        
        # Si le scraping n'a rien donné, on utilise le fallback
        return sorted(list(set(all_tickers))) if all_tickers else fallback
    except Exception:
        return fallback

BASE_LIST = get_european_tickers()

# --- DANS LA SIDEBAR (Ligne 51 corrigée) ---
# On vérifie que AIR.PA est bien présent avant de le mettre en défaut
default_choice = ["AIR.PA"] if "AIR.PA" in BASE_LIST else [BASE_LIST[0]]

selected_tickers = st.multiselect(
    "Sélectionner dans les indices :", 
    options=BASE_LIST, 
    default=default_choice
)

# --- BARRE LATÉRALE (SIDEBAR) ---
with st.sidebar:
    st.title("⚙️ PRO NAVIGATION")
    app_mode = st.radio("Sélectionner l'outil :", ["Projection Monte Carlo", "Frontière Efficiente & Ratios"])
    
    st.divider()
    st.header("🛒 Composition du Portefeuille")
    selected_tickers = st.multiselect("Sélectionner dans les indices :", options=BASE_LIST, default=["AIR.PA"])
    
    manual_t = st.text_input("Ajout manuel (ex: PUST.PA, BTC-EUR) :").upper()
    if 'manual_list' not in st.session_state: st.session_state.manual_list = []
    if st.button("➕ Ajouter au portefeuille"):
        if manual_t and manual_t not in st.session_state.manual_list:
            st.session_state.manual_list.append(manual_t)
            st.rerun()

    final_list = list(set(selected_tickers + st.session_state.manual_list))
    if st.button("🗑️ Vider le portefeuille"):
        st.session_state.manual_list = []
        st.rerun()

    st.divider()
    # Saisie des unités
    shares_dict = {}
    if final_list:
        st.subheader("Unités détenues")
        for t in final_list:
            shares_dict[t] = st.number_input(f"Quantité {t}", value=10, min_value=1)

    st.divider()
    if app_mode == "Projection Monte Carlo":
        model_type = st.radio("Moteur Mathématique :", ["FHS (Historique)", "Student-t (Fat-Tails)"])
        nu = st.slider("Degrés de liberté (nu)", 3, 30, 5) if "Student" in model_type else 5
        horizon = st.number_input("Horizon (jours)", value=150)
        n_sims = st.number_input("Simulations", value=2000)
        decay = st.slider("Lambda (EWMA)", 0.80, 0.99, 0.94)
        run_btn = st.button("🚀 LANCER LA PROJECTION")
    else:
        start_opt = st.date_input("Historique d'analyse", datetime.date(2021, 1, 1))
        rf_rate = st.number_input("Taux sans risque (%)", value=3.0) / 100
        run_btn = st.button("🎯 GÉNÉRER L'ANALYSE")

# --- CORPS DE L'APPLICATION ---
st.header(f"✨ {app_mode.upper()}")

if not final_list:
    st.info("👈 Veuillez ajouter des actifs dans le menu à gauche pour commencer.")
    st.stop()

# Téléchargement des données communes
with st.spinner("Téléchargement des données Yahoo Finance..."):
    # On prend un peu plus d'historique pour la calibration
    raw_data = yf.download(final_list, start="2019-01-01")['Close']
    if len(final_list) == 1: raw_data = raw_data.to_frame(final_list[0])
    raw_data = raw_data.ffill().dropna()
    last_prices = raw_data.iloc[-1]
    
    # Calcul des poids actuels
    current_values = {t: last_prices[t] * shares_dict[t] for t in final_list}
    total_val = sum(current_values.values())
    current_weights = np.array([current_values[t]/total_val for t in final_list])

# --- MODE 1 : MONTE CARLO ---
if app_mode == "Projection Monte Carlo" and run_btn:
    returns = np.log(raw_data / raw_data.shift(1)).dropna()
    
    # Calibration Volatilité EWMA
    ewma_var = (returns**2).ewm(alpha=(1 - decay), adjust=False).mean()
    curr_vol = np.sqrt(ewma_var.iloc[-1].values)
    
    # Simulation
    price_paths = np.zeros((horizon, n_sims, len(final_list)))
    temp_prices = np.tile(last_prices.values, (n_sims, 1))
    sim_vols = np.tile(curr_vol, (n_sims, 1))
    
    if model_type == "Student-t (Fat-Tails)":
        L = cholesky(returns.corr().values, lower=True)

    for t in range(horizon):
        if model_type == "FHS (Historique)":
            shocks = returns.sample(n_sims, replace=True).values / np.sqrt(ewma_var.sample(n_sims).values)
        else:
            t_samples = np.random.standard_t(df=nu, size=(n_sims, len(final_list)))
            shocks = (t_samples @ L.T) * np.sqrt((nu - 2) / nu)
        
        daily_ret = shocks * sim_vols
        temp_prices *= np.exp(daily_ret)
        price_paths[t] = temp_prices
        sim_vols = np.sqrt(decay * (sim_vols**2) + (1 - decay) * (daily_ret**2))

    portfolio_paths = np.sum(price_paths * [shares_dict[t] for t in final_list], axis=2)
    final_pnl = portfolio_paths[-1, :] - total_val

    # Affichage des KPIs Natifs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Valeur Portefeuille", f"{total_val:,.2f} €")
    col2.metric("Espérance Gain", f"{np.mean(final_pnl):+,.2f} €")
    col3.metric("Probabilité Profit", f"{np.mean(final_pnl > 0)*100:.1f} %")
    col4.metric("VaR 95% (Risque)", f"{np.percentile(final_pnl, 5):,.2f} €")

    # Graphes Monte Carlo
    fig = plt.figure(figsize=(16, 7), facecolor='none')
    gs = GridSpec(1, 2, width_ratios=[1.8, 1])
    norm = plt.Normalize(final_pnl.min(), final_pnl.max())
    cmap = plt.cm.RdYlGn

    ax1 = fig.add_subplot(gs[0], facecolor='none')
    for i in np.random.choice(n_sims, 150):
        ax1.plot(portfolio_paths[:, i], color=cmap(norm(final_pnl[i])), alpha=0.2, lw=0.7)
    ax1.plot(np.median(portfolio_paths, axis=1), color='white', lw=3, label='Médiane')
    ax1.set_title("TRAJECTOIRES (ROUGE = RISQUE / VERT = OPPORTUNITÉ)", color='white', fontweight='bold')
    
    ax2 = fig.add_subplot(gs[1], facecolor='none')
    n, bins, patches = ax2.hist(final_pnl, bins=50, density=True, alpha=0.8)
    for b, p in zip(bins, patches): p.set_facecolor(cmap(norm(b)))
    ax2.set_title("DISTRIBUTION DES PROFITS/PERTES", color='white', fontweight='bold')
    
    st.pyplot(fig, transparent=True)

# --- MODE 2 : FRONTIÈRE EFFICIENTE ---
if app_mode == "Frontière Efficiente & Ratios":
    # 1. Camembert de répartition
    st.subheader("📊 Répartition et Prix du jour")
    c_pie, c_table = st.columns([1, 1.2])
    
    with c_pie:
        fig_p, ax_p = plt.subplots(figsize=(5, 5), facecolor='none')
        ax_p.pie(current_values.values(), labels=final_list, autopct='%1.1f%%', 
                 colors=plt.cm.viridis(np.linspace(0, 1, len(final_list))), textprops={'color':"w"})
        st.pyplot(fig_p, transparent=True)
        
    with c_table:
        df_sum = pd.DataFrame({
            "Ticker": final_list,
            "Prix Unitaire": [last_prices[t] for t in final_list],
            "Quantité": [shares_dict[t] for t in final_list],
            "Valeur (€)": [current_values[t] for t in final_list],
            "Poids (%)": [current_weights[i]*100 for i, t in enumerate(final_list)]
        })
        st.dataframe(df_sum.style.format({"Prix Unitaire": "{:.2f}€", "Valeur (€)": "{:,.2f}€", "Poids (%)": "{:.1f}%"}), hide_index=True)

    if run_btn:
        st.divider()
        # Calcul Frontière
        returns_opt = raw_data[raw_data.index >= pd.Timestamp(start_opt)].pct_change().dropna()
        mean_ret = returns_opt.mean() * 252
        cov_mat = returns_opt.cov() * 252
        
        n_p = 4000
        results = np.zeros((3, n_p))
        for i in range(n_p):
            w = np.random.random(len(final_list))
            w /= np.sum(w)
            results[0,i] = np.sum(mean_ret * w) # Return
            results[1,i] = np.sqrt(np.dot(w.T, np.dot(cov_mat, w))) # Vol
            results[2,i] = (results[0,i] - rf_rate) / results[1,i] # Sharpe

        # Stats Portefeuille Actuel
        curr_ret_ann = np.sum(mean_ret * current_weights)
        curr_vol_ann = np.sqrt(np.dot(current_weights.T, np.dot(cov_mat, current_weights)))
        curr_sharpe = (curr_ret_ann - rf_rate) / curr_vol_ann

        # Ratios supplémentaires (Sortino / Calmar)
        p_rets = (returns_opt * current_weights).sum(axis=1)
        downside = p_rets[p_rets < 0].std() * np.sqrt(252)
        curr_sortino = (curr_ret_ann - rf_rate) / downside
        
        cum_ret = (1 + p_rets).cumprod()
        max_dd = abs(((cum_ret / cum_ret.expanding().max()) - 1).min())
        curr_calmar = curr_ret_ann / max_dd

        # Affichage Ratios
        st.subheader("🎯 Diagnostic d'Efficience")
        r1, r2, r3 = st.columns(3)
        r1.metric("Ratio de Sharpe", f"{curr_sharpe:.2f}", help="Rendement/Risque global")
        r2.metric("Ratio de Sortino", f"{curr_sortino:.2f}", help="Performance vs Risque de baisse")
        r3.metric("Ratio de Calmar", f"{curr_calmar:.2f}", help="Performance vs Pire perte historique")

        # Graphique Frontière
        fig_ef, ax_ef = plt.subplots(figsize=(12, 6), facecolor='none')
        sc = ax_ef.scatter(results[1,:], results[0,:], c=results[2,:], cmap='RdYlGn', alpha=0.4)
        ax_ef.scatter(curr_vol_ann, curr_ret_ann, color='blue', marker='X', s=250, label='VOTRE POSITION')
        ax_ef.set_xlabel("Volatilité Annuelle", color='white')
        ax_ef.set_ylabel("Rendement Annuel", color='white')
        plt.colorbar(sc, label='Ratio de Sharpe')
        ax_ef.legend()
        st.pyplot(fig_ef, transparent=True)

        with st.expander("📚 Définitions des Ratios"):
            st.markdown("""
            - **Sharpe** : Rendement excédentaire divisé par la volatilité totale. Plus il est élevé, plus votre prise de risque est rémunérée.
            - **Sortino** : Identique au Sharpe mais ne punit que la volatilité négative (celle qui fait perdre de l'argent).
            - **Calmar** : Rendement annuel / Maximum Drawdown. Indique si le gain vaut le coup par rapport à la plus grosse chute historique subie.
            """)

st.divider()
st.caption("Outil de simulation quantitative - Données Yahoo Finance - Modèles FHS & Student-t Corrélation par Cholesky.")
