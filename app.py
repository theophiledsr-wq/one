import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec
import datetime
from scipy.linalg import cholesky

# --- CONFIGURATION DE LA PAGE & MASQUAGE MENU ---
st.set_page_config(page_title="European Portfolio Master Pro", layout="wide")

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            .stDeployButton {display:none;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# --- BANDEAU ANIMÉ RALENTI (100s) ---
def display_animated_ticker():
    indices = {
        "^FCHI": "CAC 40", "^GDAXI": "DAX 40", "^STOXX50E": "EURO 50",
        "^GSPC": "S&P 500", "^IXIC": "NASDAQ", "^N225": "NIKKEI",
        "BTC-USD": "BITCOIN", "GC=F": "OR"
    }
    try:
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
                ticker_items += f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <b>{name}</b> {current:,.2f} <span style='color:{color};'>{icon} {sign}{var:.2f}%</span> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; |"
        full_content = (ticker_items * 3) if ticker_items else "Chargement des marchés..."
        st.markdown(f"""
            <style>
            @keyframes marquee {{ 0% {{ transform: translateX(0); }} 100% {{ transform: translateX(-50%); }} }}
            .ticker-wrap {{ width: 100%; overflow: hidden; background-color: #0e1117; padding: 12px 0; border-bottom: 2px solid #31333f; white-space: nowrap; }}
            .ticker-move {{ display: inline-block; white-space: nowrap; animation: marquee 100s linear infinite; font-family: 'Segoe UI', sans-serif; font-size: 1.1rem; color: white; }}
            .ticker-move:hover {{ animation-play-state: paused; cursor: pointer; }}
            </style>
            <div class="ticker-wrap"><div class="ticker-move">{full_content}</div></div>
        """, unsafe_allow_html=True)
    except: st.error("Erreur de flux")

display_animated_ticker()
st.title("THE FRENCH BUILT TOOL FOR STRATEGIC INVESTING")

# --- FONCTION RECHERCHE TICKER ---
def get_full_ticker_info(symbol):
    try:
        t = yf.Ticker(symbol)
        # 1. On tente le longName dans .info
        name = t.info.get('longName')
        
        # 2. Si c'est vide, on tente le shortName
        if not name:
            name = t.info.get('shortName')
            
        # 3. Si c'est toujours vide, on utilise une autre méthode (history)
        # pour forcer la récupération des métadonnées
        if not name:
            _ = t.history(period="1d") # Force le chargement
            name = t.info.get('longName') or symbol
            
        return name
    except:
        return symbol # Retourne au moins le ticker si tout échoue

# --- SIDEBAR & RECHERCHE INTELLIGENTE ---
with st.sidebar:
    st.header("🧭 Navigation")
    app_mode = st.radio("Choisir l'outil :", ["Projection Monte Carlo", "Optimisation & Frontière Efficiente"])
    st.divider()
    
    st.header("🛒 Portefeuille")
    
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = {} # {Ticker: Nom_Complet}

    search_input = st.text_input("Ajouter un Ticker (ex: AAPL, MC.PA) :").upper().strip()
    
    if st.button("➕ Ajouter à l'analyse"):
        if search_input and search_input not in st.session_state.portfolio:
            with st.spinner('Récupération du nom...'):
                full_name = get_full_ticker_info(search_input)
                st.session_state.portfolio[search_input] = full_name
            st.rerun()

    # Affichage de la liste avec NOM COMPLET
    if st.session_state.portfolio:
        to_delete = []
        st.write("### Vos actifs :")
        for t, name in st.session_state.portfolio.items():
            c1, c2 = st.columns([4, 1])
            # Affiche le Nom Complet en gras et le Ticker en petit
            c1.markdown(f"**{name}** \n<small>{t}</small>", unsafe_allow_html=True)
            if c2.button("🗑️", key=f"del_{t}"):
                to_delete.append(t)
        
        for t in to_delete:
            del st.session_state.portfolio[t]
            st.rerun()
    
    final_list = list(st.session_state.portfolio.keys())
    
    if not final_list:
        st.info("Veuillez ajouter des actifs pour commencer.")
        st.stop()

    st.divider()
    # On utilise le nom complet dans l'étiquette du champ de saisie des quantités
    shares_dict = {t: st.number_input(f"Qté : {st.session_state.portfolio[t]}", value=10, min_value=1) for t in final_list}

    st.divider()
    if app_mode == "Projection Monte Carlo":
        model_type = st.radio("Modèle :", ["FHS (Historique)", "Student-t", "GARCH(1,1)"])
        start_date = st.date_input("Historique depuis :", datetime.date(2021, 1, 1))
        n_days = st.number_input("Horizon (jours)", value=150)
        n_sims = st.number_input("Simulations", value=2000)
        run_btn = st.button("🚀 LANCER LA SIMULATION")
    else:
        start_date = st.date_input("Analyse depuis :", datetime.date(2020, 1, 1))
        rf_rate = st.number_input("Taux sans risque (%)", value=3.0) / 100
        n_portfolios = st.number_input("Nombre de portefeuilles", value=5000)
        run_btn = st.button("🎯 GÉNÉRER LA FRONTIÈRE")

# --- CHARGEMENT DES DONNÉES ---
@st.cache_data
def load_data_portfolio(tickers):
    df = yf.download(tickers, start="2018-01-01", progress=False)['Close']
    if isinstance(df, pd.Series): # Cas où il n'y a qu'un seul ticker
        df = df.to_frame()
    return df.ffill().dropna()

raw_data = load_data_portfolio(final_list)

# --- MODE MONTE CARLO ---
if app_mode == "Projection Monte Carlo" and run_btn:
    data_filtered = raw_data[raw_data.index >= pd.Timestamp(start_date)]
    returns = np.log(data_filtered / data_filtered.shift(1)).dropna()
    last_prices = data_filtered.iloc[-1]
    total_val = sum(last_prices[t] * shares_dict[t] for t in final_list)
    
    price_paths = np.zeros((n_days, n_sims, len(final_list)))
    temp_prices = np.tile(last_prices.values, (n_sims, 1))
    decay = 0.94
    ewma_var = (returns**2).ewm(alpha=(1 - decay), adjust=False).mean()
    sim_vols = np.tile(np.sqrt(ewma_var.iloc[-1].values), (n_sims, 1))

    for t in range(n_days):
        shocks = np.random.normal(0, 1, size=(n_sims, len(final_list)))
        daily_ret = shocks * sim_vols
        temp_prices *= np.exp(daily_ret)
        price_paths[t] = temp_prices
        sim_vols = np.sqrt(decay * (sim_vols**2) + (1 - decay) * (daily_ret**2))

    portfolio_paths = np.sum(price_paths * [shares_dict[t] for t in final_list], axis=2)
    final_pnl = portfolio_paths[-1, :] - total_val
    
    st.columns(3)[1].metric(f"Issue Médiane", f"{np.median(final_pnl):,.2f} €", f"{(np.median(final_pnl)/total_val)*100:.2f} %")
    
    fig = plt.figure(figsize=(16, 7), facecolor='none')
    gs = GridSpec(1, 2, width_ratios=[1.8, 1])
    plt.rcParams.update({"text.color": "white", "axes.labelcolor": "white", "xtick.color": "white", "ytick.color": "white"})
    ax1 = fig.add_subplot(gs[0], facecolor='none'); norm = plt.Normalize(final_pnl.min(), final_pnl.max())
    for i in np.random.choice(n_sims, 100): ax1.plot(portfolio_paths[:, i], color=plt.cm.RdYlGn(norm(final_pnl[i])), alpha=0.3)
    ax2 = fig.add_subplot(gs[1], facecolor='none'); n, bins, patches = ax2.hist(final_pnl, bins=50, density=True, alpha=0.8)
    for b, p in zip(bins, patches): p.set_facecolor('red' if b < 0 else 'green')
    st.pyplot(fig, transparent=True)

# --- MODE OPTIMISATION ---
elif app_mode == "Optimisation & Frontière Efficiente" and run_btn:
    data_opt = raw_data[raw_data.index >= pd.Timestamp(start_date)]
    returns_daily = data_opt.pct_change().dropna()
    mean_returns = returns_daily.mean() * 252
    cov_matrix = returns_daily.cov() * 252
    
    last_p = data_opt.iloc[-1]
    vals = np.array([shares_dict[t] * last_p[t] for t in final_list])
    curr_w = vals / np.sum(vals)
    curr_ret = np.sum(mean_returns * curr_w)
    curr_vol = np.sqrt(np.dot(curr_w.T, np.dot(cov_matrix, curr_w)))
    curr_sharpe = (curr_ret - rf_rate) / curr_vol

    res = np.zeros((3, n_portfolios))
    w_rec = []
    for i in range(n_portfolios):
        w = np.random.random(len(final_list)); w /= np.sum(w)
        w_rec.append(w)
        r = np.sum(mean_returns * w)
        v = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
        res[0,i], res[1,i], res[2,i] = r, v, (r - rf_rate) / v

    best_idx = np.argmax(res[2])
    
    st.subheader("🎯 Comparaison : Actuel vs Optimisé")
    c1, c2 = st.columns([2, 1])
    with c1:
        fig_opt, ax_opt = plt.subplots(figsize=(10, 6), facecolor='none')
        ax_opt.set_facecolor('none')
        sc = ax_opt.scatter(res[1,:], res[0,:], c=res[2,:], cmap='viridis', s=10, alpha=0.3)
        ax_opt.scatter(res[1,best_idx], res[0,best_idx], marker='*', color='r', s=200, label='Optimal')
        ax_opt.scatter(curr_vol, curr_ret, marker='D', color='white', s=150, edgecolors='black', label='Actuel')
        ax_opt.legend(); st.pyplot(fig_opt, transparent=True)
    with c2:
        # Utilisation des noms complets pour l'index du tableau récapitulatif
        display_names = [st.session_state.portfolio[t] for t in final_list]
        comp_df = pd.DataFrame({
            'Actuel (%)': [round(x*100, 1) for x in curr_w],
            'Optimal (%)': [round(x*100, 1) for x in w_rec[best_idx]]
        }, index=display_names)
        st.table(comp_df)
        st.metric("Sharpe Optimal", f"{res[2, best_idx]:.2f}", f"{res[2, best_idx]-curr_sharpe:.2f}")
