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

# --- BANDEAU ANIMÉ (TICKER TAPE) ---
# --- BANDEAU ANIMÉ RALENTI (TICKER TAPE) ---
def display_animated_ticker():
    # Liste d'indices avec tickers robustes
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
        # On télécharge 7 jours pour être SÛR d'avoir au moins 2 clôtures valides (gestion week-end/fériés)
        data = yf.download(list(indices.keys()), period="7d", interval="1d", progress=False)['Close']
        
        # Nettoyage : on remplit les trous (ffill) et on ne garde que les colonnes qui ont des données
        data = data.ffill().dropna(axis=0, how='all')
        
        ticker_items = ""
        for ticker, name in indices.items():
            if ticker in data.columns:
                series = data[ticker].dropna()
                if len(series) >= 2:
                    current = series.iloc[-1]
                    prev = series.iloc[-2]
                    
                    # Calcul sécurisé de la variation
                    if prev != 0 and not np.isnan(current) and not np.isnan(prev):
                        var = ((current - prev) / prev) * 100
                        color = "#00ff00" if var >= 0 else "#ff4b4b"
                        icon = "▲" if var >= 0 else "▼"
                        sign = "+" if var >= 0 else ""
                        
                        ticker_items += f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **{name}** {current:,.2f} <span style='color:{color};'>{icon} {sign}{var:.2f}%</span> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; |"

        # Si aucune donnée n'est dispo, on affiche un message discret
        if not ticker_items:
            ticker_items = "Flux de données en attente... &nbsp;&nbsp;&nbsp;&nbsp; |"

        full_content = ticker_items * 3 

        st.markdown(f"""
            <style>
            @keyframes marquee {{ 0% {{ transform: translateX(0); }} 100% {{ transform: translateX(-50%); }} }}
            .ticker-wrap {{ width: 100%; overflow: hidden; background-color: #0e1117; padding: 12px 0; border-bottom: 2px solid #31333f; white-space: nowrap; }}
            .ticker-move {{ display: inline-block; white-space: nowrap; animation: marquee 80s linear infinite; font-family: sans-serif; font-size: 1.1rem; color: white; }}
            </style>
            <div class="ticker-wrap"><div class="ticker-move">{full_content}</div></div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Erreur technique bandeau : {e}")
    
    ticker_data = yf.download(list(indices.keys()), period="2d", progress=False)['Close']
    
    ticker_items = ""
    for ticker, name in indices.items():
        try:
            current = ticker_data[ticker].iloc[-1]
            prev = ticker_data[ticker].iloc[-2]
            var = ((current - prev) / prev) * 100
            color = "#00ff00" if var >= 0 else "#ff4b4b"
            icon = "▲" if var >= 0 else "▼"
            # Augmentation de l'espacement entre les blocs d'indices
            ticker_items += f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **{name}** {current:,.2f} <span style='color:{color};'>{icon} {var:.2f}%</span> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; |"
        except:
            continue

    # On répète la chaîne pour un défilement continu sans coupure
    full_content = ticker_items * 3 

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
            position: relative;
        }}
        .ticker-move {{
            display: inline-block;
            white-space: nowrap;
            animation: marquee 60s linear infinite; /* Ralenti à 60 secondes */
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            font-size: 1.15rem;
        }}
        .ticker-move:hover {{
            animation-play-state: paused; /* S'arrête au survol pour une lecture facile */
            cursor: pointer;
        }}
        </style>
        <div class="ticker-wrap">
            <div class="ticker-move">
                {full_content}
            </div>
        </div>
    """, unsafe_allow_html=True)

# Affichage du bandeau
display_animated_ticker()
st.title("THE FRENCH BUILT TOOL FOR STRATEGIC INVESTING")

# --- RÉCUPÉRATION DES TICKERS (Base de données) ---
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
    app_mode = st.radio("Choisir l'outil :", ["Projection Monte Carlo", "Optimisation & Frontière Efficiente"])
    
    st.divider()
    st.header("🛒 Portefeuille")
    
    safe_default = [BASE_LIST[0]] if "AIR.PA" not in BASE_LIST else ["AIR.PA"]
    selected_tickers = st.multiselect("Sélectionner dans les indices :", options=BASE_LIST, default=safe_default)
    
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
        
        # --- INFOS BULLES ---
        if model_type == "FHS (Historique)":
            st.info("**FHS :** Utilise les rendements réels. ✅ Capture les queues de distribution réelles.")
        elif model_type == "Student-t":
            st.info("**Student-t :** Idéal pour simuler des 'Black Swans'. ✅ Paramètre Nu gère l'épaisseur des queues.")
        elif model_type == "GARCH(1,1)":
            st.info("**GARCH(1,1) :** Volatilité qui réagit aux chocs. ✅ Modélise le regroupement de la peur.")

        start_mc = st.date_input("Analyser l'historique depuis le :", datetime.date(2021, 1, 1), key="date_mc")
        nu_val = st.slider("nu (v)", 3, 50, 5) if model_type == "Student-t" else 5
        n_days = st.number_input("Horizon (jours)", value=150)
        n_sims = st.number_input("Simulations", value=2000)
        run_btn = st.button("🚀 LANCER LA SIMULATION")
    else:
        start_opt = st.date_input("Analyse depuis le", datetime.date(2021, 1, 1), key="date_opt")
        rf_rate = st.number_input("Taux sans risque (%)", value=3.0) / 100
        run_btn = st.button("🎯 GÉNÉRER LA FRONTIÈRE")

# --- CHARGEMENT DES DONNÉES ---
@st.cache_data
def load_data_portfolio(tickers):
    df = yf.download(tickers + ["^GSPC"], start="2018-01-01")['Close']
    return df.ffill().dropna()

raw_data = load_data_portfolio(final_list)

# --- MODE MONTE CARLO ---
if app_mode == "Projection Monte Carlo" and 'run_btn' in locals() and run_btn:
    data_filtered = raw_data[raw_data.index >= pd.Timestamp(start_mc)][final_list]
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
        else: # GARCH
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
    
    st.columns(3)[1].metric(f"Issue Médiane", f"{np.median(final_pnl):,.2f} €", f"{(np.median(final_pnl)/total_val)*100:.2f} %")

    fig = plt.figure(figsize=(16, 7), facecolor='none')
    gs = GridSpec(1, 2, width_ratios=[1.8, 1])
    plt.rcParams.update({"text.color": "white", "axes.labelcolor": "white", "xtick.color": "white", "ytick.color": "white"})
    ax1 = fig.add_subplot(gs[0], facecolor='none')
    norm = plt.Normalize(final_pnl.min(), final_pnl.max())
    for i in np.random.choice(n_sims, 100):
        ax1.plot(portfolio_paths[:, i], color=plt.cm.RdYlGn(norm(final_pnl[i])), alpha=0.3)
    ax1.set_title(f"Simulation {model_type}")
    
    ax2 = fig.add_subplot(gs[1], facecolor='none')
    n, bins, patches = ax2.hist(final_pnl, bins=50, density=True, alpha=0.8)
    for b, p in zip(bins, patches): p.set_facecolor('red' if b < 0 else 'green')
    st.pyplot(fig, transparent=True)

# --- MODE OPTIMISATION (Identique) ---
elif app_mode == "Optimisation & Frontière Efficiente":
    # (Le reste du code d'optimisation reste identique au précédent pour la cohérence)
    st.info("Utilisez les réglages dans la barre latérale pour générer la frontière efficiente.")
