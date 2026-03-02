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

# --- BANDEAU ANIMÉ RALENTI (TICKER TAPE) ---
def display_animated_ticker():
    indices = {
        "^FCHI": "CAC 40",
        "^GDAXI": "DAX 40",
        "^STOXX50E": "EURO 50",
        "^GSPC": "S&P 500",
        "^IXIC": "NASDAQ",
        "^N225": "NIKKEI",
        "BTC-USD": "BITCOIN",
        "GC=F": "OR"
    }
    
    try:
        # Récupération sur 5 jours pour garantir des données valides (évite les NaN du week-end)
        ticker_data = yf.download(list(indices.keys()), period="5d", progress=False)['Close']
        ticker_data = ticker_data.ffill()
        
        ticker_items = ""
        for ticker, name in indices.items():
            series = ticker_data[ticker].dropna()
            if len(series) >= 2:
                current = series.iloc[-1]
                prev = series.iloc[-2]
                var = ((current - prev) / prev) * 100
                color = "#00ff00" if var >= 0 else "#ff4b4b"
                icon = "▲" if var >= 0 else "▼"
                sign = "+" if var >= 0 else ""
                
                # Utilisation de <b> pour le gras (évite l'affichage des astérisques en HTML)
                ticker_items += f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <b>{name}</b> {current:,.2f} <span style='color:{color};'>{icon} {sign}{var:.2f}%</span> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; |"

        # Contenu triplé pour une boucle infinie sans saut
        full_content = (ticker_items * 3) if ticker_items else "Chargement des données marchés..."

        st.markdown(f"""
            <style>
            @keyframes marquee {{
                0% {{ transform: translateX(0); }}
                100% {{ transform: translateX(-50%); }}
            }}
            .ticker-wrap {{
                width: 100%;
                overflow: hidden;
                background-color: #0e1117;
                padding: 12px 0;
                border-bottom: 2px solid #31333f;
                white-space: nowrap;
            }}
            .ticker-move {{
                display: inline-block;
                white-space: nowrap;
                animation: marquee 100s linear infinite; /* Ralenti à 100s */
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                font-size: 1.1rem;
                color: white;
            }}
            .ticker-move:hover {{
                animation-play-state: paused;
                cursor: pointer;
            }}
            </style>
            <div class="ticker-wrap">
                <div class="ticker-move">
                    {full_content}
                </div>
            </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Erreur flux : {e}")

# Lancement du bandeau
display_animated_ticker()
st.title("THE FRENCH BUILT TOOL FOR STRATEGIC INVESTING")

# --- RÉCUPÉRATION DES TICKERS (WIKIPEDIA + FALLBACK) ---
@st.cache_data
def get_european_base_list():
    try:
        fallback = ["AIR.PA", "MC.PA", "OR.PA", "RMS.PA", "SAP.DE", "ASML.AS", "SIE.DE"]
        indices = {"CAC 40": "https://en.wikipedia.org/wiki/CAC_40", "DAX 40": "https://en.wikipedia.org/wiki/DAX"}
        tickers = []
        for url in indices.values():
            tables = pd.read_html(url)
            for t in tables:
                if 'Ticker' in t.columns:
                    suffix = ".PA" if "CAC" in url else ".DE"
                    tickers.extend([str(tk).split('.')[0] + suffix for tk in t['Ticker'].tolist()])
                    break
        return sorted(list(set(tickers))) if tickers else fallback
    except:
        return ["AIR.PA", "MC.PA", "OR.PA", "SAP.DE", "ASML.AS"]

BASE_LIST = get_european_base_list()

# --- BARRE LATÉRALE ---
with st.sidebar:
    st.header("🧭 Navigation")
    app_mode = st.radio("Outil :", ["Projection Monte Carlo", "Optimisation & Frontière Efficiente"])
    
    st.divider()
    st.header("🛒 Portefeuille")
    
    selected_tickers = st.multiselect("Sélectionner :", options=BASE_LIST, default=[BASE_LIST[0]])
    
    manual_t = st.text_input("Ajout manuel (ex: PUST.PA) :").upper()
    if 'manual_list' not in st.session_state: st.session_state.manual_list = []
    if st.button("➕ Ajouter"):
        if manual_t and manual_t not in st.session_state.manual_list:
            st.session_state.manual_list.append(manual_t)
            st.rerun()

    final_list = list(set(selected_tickers + st.session_state.manual_list))
    if not final_list: st.stop()

    shares_dict = {t: st.number_input(f"Quantité {t}", value=10, min_value=1) for t in final_list}

    st.divider()
    if app_mode == "Projection Monte Carlo":
        model_type = st.radio("Modèle :", ["FHS (Historique)", "Student-t", "GARCH(1,1)"])
        start_mc = st.date_input("Analyser depuis le :", datetime.date(2021, 1, 1), key="date_mc")
        nu_val = st.slider("nu (v)", 3, 50, 5) if model_type == "Student-t" else 5
        n_days = st.number_input("Horizon (jours)", value=150)
        n_sims = st.number_input("Simulations", value=2000)
        run_btn = st.button("🚀 LANCER LA SIMULATION")
    else:
        start_opt = st.date_input("Analyse depuis le", datetime.date(2021, 1, 1), key="date_opt")
        rf_rate = st.number_input("Taux sans risque (%)", value=3.0) / 100
        run_btn = st.button("🎯 GÉNÉRER LA FRONTIÈRE")

# --- CHARGEMENT ---
@st.cache_data
def load_data_portfolio(tickers):
    df = yf.download(tickers + ["^GSPC"], start="2018-01-01")['Close']
    return df.ffill().dropna()

raw_data = load_data_portfolio(final_list)

# --- MODE MONTE CARLO ---
if app_mode == "Projection Monte Carlo" and 'run_btn' in locals() and run_btn:
    data_filtered = raw_data[raw_data.index >= pd.Timestamp(start_mc)][final_list]
    # Calcul des rendements logarithmiques (plus stables pour les simulations)
    returns = np.log(data_filtered / data_filtered.shift(1)).dropna()
    last_prices = data_filtered.iloc[-1]
    total_val = sum(last_prices[t] * shares_dict[t] for t in final_list)
    
    price_paths = np.zeros((n_days, n_sims, len(final_list)))
    temp_prices = np.tile(last_prices.values, (n_sims, 1))
    
    decay = 0.94
    ewma_var = (returns**2).ewm(alpha=(1 - decay), adjust=False).mean()
    sim_vols = np.tile(np.sqrt(ewma_var.iloc[-1].values), (n_sims, 1))

    if model_type == "Student-t":
        L = cholesky(returns.corr().values + np.eye(len(final_list))*1e-8, lower=True)
    
    if model_type == "GARCH(1,1)":
        omega, alpha, beta = 1e-6, 0.05, 0.90

    for t in range(n_days):
        if model_type == "FHS (Historique)":
            std_rets = (returns / np.sqrt(ewma_var.shift(1))).dropna()
            shocks = std_rets.sample(n_sims, replace=True).values
        elif model_type == "Student-t":
            t_samples = np.random.standard_t(df=nu_val, size=(n_sims, len(final_list)))
            shocks = (t_samples @ L.T) * np.sqrt((nu_val - 2) / nu_val)
        else:
            shocks = np.random.normal(0, 1, size=(n_sims, len(final_list)))

        daily_ret = shocks * sim_vols
        temp_prices *= np.exp(daily_ret)
        price_paths[t] = temp_prices
        
        if model_type == "GARCH(1,1)":
            sim_vols = np.sqrt(omega + alpha * (daily_ret**2) + beta * (sim_vols**2))
        else:
            sim_vols = np.sqrt(decay * (sim_vols**2) + (1 - decay) * (daily_ret**2))

    portfolio_paths = np.sum(price_paths * [shares_dict[t] for t in final_list], axis=2)
    final_pnl = portfolio_paths[-1, :] - total_val
    
    st.metric("Issue Médiane", f"{np.median(final_pnl):,.2f} €", f"{(np.median(final_pnl)/total_val)*100:.2f} %")
    st.line_chart(portfolio_paths[:, :100])

# --- MODE OPTIMISATION ---
elif app_mode == "Optimisation & Frontière Efficiente":
    data_opt = raw_data[final_list]
    if run_btn:
        ret_opt = data_opt[data_opt.index >= pd.Timestamp(start_opt)].pct_change().dropna()
        mean_ret, cov_mat = ret_opt.mean() * 252, ret_opt.cov() * 252
        
        results = []
        for _ in range(3000):
            w = np.random.random(len(final_list)); w /= np.sum(w)
            r = np.sum(mean_ret * w)
            v = np.sqrt(w.T @ cov_mat @ w)
            results.append([r, v, (r - rf_rate) / v, w])
        
        df_res = pd.DataFrame(results, columns=['ret', 'vol', 'sharpe', 'weights'])
        st.write("### Frontière Efficiente")
        st.scatter_chart(df_res, x='vol', y='ret', color='sharpe')
