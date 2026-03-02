import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec
import datetime
from scipy.linalg import cholesky

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="European Portfolio Master Pro", layout="wide")

st.title("THE FRENCH BUILT TOOL FOR STRATEGIC INVESTING")

# --- RÉCUPÉRATION DES TICKERS ---
@st.cache_data
def get_european_base_list():
    try:
        indices = {"CAC 40": "https://en.wikipedia.org/wiki/CAC_40", 
                   "DAX 40": "https://en.wikipedia.org/wiki/DAX", 
                   "IBEX 35": "https://en.wikipedia.org/wiki/IBEX_35"}
        tickers = []
        for url in indices.values():
            tables = pd.read_html(url)
            for t in tables:
                if 'Ticker' in t.columns:
                    suffix = ".PA" if "CAC" in url else ".DE" if "DAX" in url else ".MC"
                    tickers.extend([str(tk).split('.')[0] + suffix for tk in t['Ticker'].tolist()])
                    break
        return sorted(list(set(tickers)))
    except:
        return ["AIR.PA", "MC.PA", "SAP.DE", "ASML.AS"]

BASE_TICKERS = get_european_base_list()

# --- BARRE LATÉRALE ---
with st.sidebar:
    st.header("🧭 Navigation")
    app_mode = st.radio("Choisir l'outil :", ["Projection Monte Carlo", "Optimisation & Frontière Efficiente"])
    
    st.divider()
    st.header("🛒 Portefeuille")
    
    # Sécurité pour le multiselect
    default_selection = ["AIR.PA"] if "AIR.PA" in BASE_TICKERS else [BASE_TICKERS[0]]
    selected_tickers = st.multiselect("Actifs :", options=BASE_TICKERS, default=default_selection)
    
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
    decay = 0.94 # Valeur fixée
    
    if app_mode == "Projection Monte Carlo":
        st.header("🔬 Paramètres Simulation")
        model_type = st.radio("Modèle :", ["FHS (Historique)", "Student-t"])
        
        nu = 5 
        if model_type == "Student-t":
            # Ajout du nu avec aide contextuelle au survol du point d'exclamation
            nu = st.slider("Degrés de liberté (nu)", 3, 50, 5, 
                          help="""Le paramètre nu (v) contrôle l'épaisseur des queues :
                          \n- **3 à 5** : Marché de crise (queues très épaisses, krachs fréquents).
                          \n- **10 à 20** : Marché volatil standard.
                          \n- **>30** : Tend vers une Loi Normale (événements extrêmes rares).""")
        
        n_days = st.number_input("Horizon (jours)", value=150)
        n_sims = st.number_input("Simulations", value=2000)
        
        # Affichage statique du Lambda
        st.info(f"Paramètre Lambda (EWMA) fixé à : **{decay}**")
        
        run_btn = st.button("🚀 LANCER LA SIMULATION")
    else:
        start_opt = st.date_input("Analyse depuis le", datetime.date(2021, 1, 1))
        rf_rate = st.number_input("Taux sans risque (%)", value=3.0) / 100
        st.info(f"Paramètre Lambda (EWMA) fixé à : **{decay}**")
        run_btn = st.button("🎯 GÉNÉRER L'ANALYSE")

# --- LOGIQUE DE CALCUL ---
if final_list:
    try:
        data = yf.download(final_list, start="2020-01-01")['Close']
        if isinstance(data, pd.Series): data = data.to_frame(final_list[0])
        data = data.ffill().dropna()
        last_prices = data.iloc[-1]
        
        current_values = {t: last_prices[t] * shares_dict[t] for t in final_list}
        total_val = sum(current_values.values())
        current_weights = np.array([current_values[t]/total_val for t in final_list])
    except:
        st.error("Erreur de téléchargement des données.")
        st.stop()

# --- AFFICHAGE SELON MODE ---
if app_mode == "Projection Monte Carlo" and run_btn:
    st.header("📈 Projection de Patrimoine")
    
    returns = np.log(data / data.shift(1)).dropna()
    ewma_var = (returns**2).ewm(alpha=(1 - decay), adjust=False).mean()
    curr_vol = np.sqrt(ewma_var.iloc[-1].values)
    
    price_paths = np.zeros((n_days, n_sims, len(final_list)))
    temp_prices = np.tile(last_prices.values, (n_sims, 1))
    sim_vols = np.tile(curr_vol, (n_sims, 1))
    
    if model_type == "Student-t":
        L = cholesky(returns.corr().values, lower=True)

    for t in range(n_days):
        if model_type == "FHS (Historique)":
            idx = np.random.randint(0, len(returns), size=n_sims)
            shocks = (returns.values[idx] / np.sqrt(ewma_var.values[idx]))
        else:
            t_samples = np.random.standard_t(df=nu, size=(n_sims, len(final_list)))
            shocks = (t_samples @ L.T) * np.sqrt((nu - 2) / nu)
        
        daily_ret = shocks * sim_vols
        temp_prices *= np.exp(daily_ret)
        price_paths[t] = temp_prices
        sim_vols = np.sqrt(decay * (sim_vols**2) + (1 - decay) * (daily_ret**2))

    portfolio_paths = np.sum(price_paths * [shares_dict[t] for t in final_list], axis=2)
    final_pnl = portfolio_paths[-1, :] - total_val

    # KPIs
    c1, c2, c3 = st.columns(3)
    c1.metric("Valeur Portefeuille", f"{total_val:,.2f} €")
    c2.metric("Espérance Gain", f"{np.mean(final_pnl):+,.2f} €")
    c3.metric("VaR 95%", f"{np.percentile(final_pnl, 5):,.2f} €")

    fig, ax = plt.subplots(figsize=(10, 5), facecolor='none')
    plt.rcParams.update({'text.color': "white", 'axes.labelcolor': "white"})
    ax.plot(portfolio_paths[:, :100], alpha=0.3)
    ax.set_title("100 Trajectoires simulées", color="white")
    st.pyplot(fig, transparent=True)

elif app_mode == "Optimisation & Frontière Efficiente":
    st.header("🎯 Analyse de Diversification")
    
    col_chart, col_data = st.columns([1, 1.2])
    with col_chart:
        fig_pie, ax_pie = plt.subplots(figsize=(6, 6), facecolor='none')
        ax_pie.pie(current_values.values(), labels=final_list, autopct='%1.1f%%', textprops={'color':"w"})
        st.pyplot(fig_pie, transparent=True)
    
    with col_data:
        df_summary = pd.DataFrame({
            "Actif": final_list,
            "Prix (€)": [last_prices[t] for t in final_list],
            "Poids (%)": [current_weights[i]*100 for i in range(len(final_list))]
        })
        st.table(df_summary)

    if run_btn:
        st.write("Calcul de la frontière efficiente...")
        # ... [Reste du code d'optimisation inchangé] ...

st.divider()
st.caption("Données Yahoo Finance | Modèle paramétrique fixé (Lambda 0.94)")
