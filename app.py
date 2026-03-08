import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec
import datetime

# --- CONFIGURATION ---
st.set_page_config(page_title="European Portfolio Master Pro", layout="wide")
hide_st_style = """<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;} .stDeployButton {display:none;}</style>"""
st.markdown(hide_st_style, unsafe_allow_html=True)

# --- BANDEAU MARCHÉ ---
def display_animated_ticker():
    indices = {"^FCHI": "CAC 40", "^GDAXI": "DAX 40", "^STOXX50E": "EURO 50", "^GSPC": "S&P 500", "BTC-USD": "BITCOIN", "GC=F": "OR"}
    try:
        # On télécharge les données pour le bandeau
        ticker_data = yf.download(list(indices.keys()), period="5d", progress=False)['Close'].ffill()
        ticker_items = ""
        for ticker, name in indices.items():
            series = ticker_data[ticker].dropna()
            if len(series) >= 2:
                current, prev = series.iloc[-1], series.iloc[-2]
                var = ((current - prev) / prev) * 100
                color, icon = ("#00ff00", "▲") if var >= 0 else ("#ff4b4b", "▼")
                ticker_items += f"&nbsp;&nbsp;&nbsp;&nbsp; <b>{name}</b> {current:,.2f} <span style='color:{color};'>{icon} {var:+.2f}%</span> &nbsp;&nbsp;&nbsp;&nbsp; |"
        st.markdown(f"""<style>@keyframes marquee {{ 0% {{ transform: translateX(0); }} 100% {{ transform: translateX(-50%); }} }} .ticker-wrap {{ width: 100%; overflow: hidden; background-color: #0e1117; padding: 12px 0; border-bottom: 2px solid #31333f; white-space: nowrap; }} .ticker-move {{ display: inline-block; white-space: nowrap; animation: marquee 100s linear infinite; font-family: sans-serif; font-size: 1.1rem; color: white; }}</style><div class="ticker-wrap"><div class="ticker-move">{(ticker_items * 3)}</div></div>""", unsafe_allow_html=True)
    except: pass

display_animated_ticker()
st.title("THE FRENCH BUILT TOOL FOR STRATEGIC INVESTING")

def get_full_ticker_info(symbol):
    try:
        search = yf.Search(symbol, max_results=1)
        if search.quotes: return search.quotes[0].get('longname') or search.quotes[0].get('shortname')
        return yf.Ticker(symbol).info.get('longName') or symbol
    except: return symbol

# --- SIDEBAR ---
with st.sidebar:
    st.header("🧭 Navigation")
    app_mode = st.radio("Choisir l'outil :", ["Projection Monte Carlo", "Optimisation & Frontière Efficiente", "Données historiques"])
    st.divider()
    
    st.header("🛒 Portefeuille")
    if 'portfolio' not in st.session_state: st.session_state.portfolio = {}
    if 'asset_fees' not in st.session_state: st.session_state.asset_fees = {}
    
    search_input = st.text_input("Rechercher un Ticker/ISIN :").upper()
    if st.button("➕ Ajouter à l'analyse"):
        if search_input:
            with st.spinner('Recherche...'): 
                name = get_full_ticker_info(search_input)
                st.session_state.portfolio[search_input] = name
                st.session_state.asset_fees[search_input] = {'entry': 0.0, 'mgmt': 0.0, 'perf': 0.0}
            st.rerun()

    if st.session_state.portfolio:
        to_delete = []
        for t, name in st.session_state.portfolio.items():
            c1, c2 = st.columns([4, 1])
            c1.caption(f"**{t}**\n{name}")
            if c2.button("**-**", key=f"del_{t}"): to_delete.append(t)
        for t in to_delete: 
            del st.session_state.portfolio[t]
            if t in st.session_state.asset_fees: del st.session_state.asset_fees[t]
            st.rerun()

    final_list = list(st.session_state.portfolio.keys())
    if not final_list: st.info("Ajoutez des actifs."); st.stop()
    
    st.divider()
    shares_dict = {t: st.number_input(f"Qte {t}", value=10, min_value=1, key=f"qty_{t}") for t in final_list}
    
    if app_mode == "Données historiques":
        st.subheader("⚙️ Configuration")
        selected_period = st.selectbox("Période d'analyse", ["1m", "6m", "1y", "3y", "5y", "10y", "all time"])
        for t in final_list:
            with st.expander(f"Frais : {t}"):
                st.session_state.asset_fees[t]['entry'] = st.number_input(f"Entrée %", value=0.0, step=0.1, key=f"ent_{t}") / 100
                st.session_state.asset_fees[t]['mgmt'] = st.number_input(f"Gestion Annuelle %", value=0.0, step=0.1, key=f"mgt_{t}") / 100
                st.session_state.asset_fees[t]['perf'] = st.number_input(f"Surperformance %", value=0.0, step=1.0, key=f"prf_{t}") / 100
        rf_hist = st.number_input("Taux sans risque %", value=3.0) / 100
        run_btn = st.button("📈 ANALYSER")
        
    elif app_mode == "Projection Monte Carlo":
        st.subheader("⚙️ Paramètres")
        start_date = st.date_input("Depuis (Volatilité historique) :", datetime.date(2021, 1, 1))
        n_days = st.number_input("Horizon de projection (jours)", value=150, min_value=1)
        st.caption("⚡ Simulations fixées à 5 000.")
        n_sims = 5000
        run_btn = st.button("🚀 LANCER SIMULATION")
        
    else:
        st.subheader("⚙️ Paramètres")
        start_date = st.date_input("Depuis :", datetime.date(2020, 1, 1))
        rf_rate = st.number_input("Taux sans risque %", value=3.0) / 100
        n_portfolios = st.number_input("Nombre de portefeuilles", value=5000)
        run_btn = st.button("🎯 GÉNÉRER FRONTIÈRE")

@st.cache_data
def load_data_full(tickers): 
    # Télécharge les actifs ET le S&P 500
    all_tickers = list(set(tickers + ["^GSPC"]))
    data = yf.download(all_tickers, start="2015-01-01", progress=False)['Close']
    return data.ffill().dropna()

# Chargement global
raw_data_all = load_data_full(final_list)

# --- ANALYSE HISTORIQUE ---
if app_mode == "Données historiques" and run_btn:
    p_map = {"1m":"1mo", "6m":"6mo", "1y":"1y", "3y":"3y", "5y":"5y", "10y":"10y", "all time":"max"}
    hist_subset = yf.download(list(set(final_list + ["^GSPC"])), period=p_map[selected_period], progress=False)['Close'].ffill().dropna()
    
    if not hist_subset.empty:
        # Calcul Portefeuille
        port_val = sum(hist_subset[t] * shares_dict[t] for t in final_list)
        init_invest = sum(hist_subset[t].iloc[0] * shares_dict[t] for t in final_list)
        
        # Rendements et métriques
        perf_brute = (port_val.iloc[-1] / port_val.iloc[0]) - 1
        vol = port_val.pct_change().dropna().std() * np.sqrt(252)
        
        # Benchmark S&P 500
        sp500 = hist_subset["^GSPC"]
        perf_sp = (sp500.iloc[-1] / sp500.iloc[0]) - 1

        st.subheader("📊 Comparaison Historique")
        c1, c2 = st.columns([3, 1])
        with c1:
            fig, ax = plt.subplots(figsize=(10, 5), facecolor='none'); ax.set_facecolor('none')
            ax.plot((port_val / port_val.iloc[0]) * 100, color='#00ff00', lw=2, label="Portefeuille")
            ax.plot((sp500 / sp500.iloc[0]) * 100, color='#A020F0', lw=1.5, ls='--', label="S&P 500")
            plt.rcParams.update({"text.color": "white", "axes.labelcolor": "white", "xtick.color": "white", "ytick.color": "white"})
            ax.legend(facecolor='#0e1117', edgecolor='white')
            ax.grid(True, alpha=0.1, color='white')
            st.pyplot(fig, transparent=True)
        with c2:
            st.metric("Performance Portefeuille", f"{perf_brute*100:.2f} %")
            st.metric("Performance S&P 500", f"{perf_sp*100:.2f} %")
            st.metric("Volatilité Ann.", f"{vol*100:.2f} %")

# --- PROJECTION MONTE CARLO ---
elif app_mode == "Projection Monte Carlo" and run_btn:
    # Filtrage des données selon la date choisie
    data_port = raw_data_all[final_list][raw_data_all.index >= pd.Timestamp(start_date)]
    data_sp = raw_data_all["^GSPC"][raw_data_all.index >= pd.Timestamp(start_date)]
    
    returns = np.log(data_port / data_port.shift(1)).dropna()
    last_prices = data_port.iloc[-1]
    total_val_init = sum(last_prices[t] * shares_dict[t] for t in final_list)
    
    # Simulation Portefeuille
    price_paths = np.zeros((n_days, n_sims, len(final_list)))
    temp_prices = np.tile(last_prices.values, (n_sims, 1))
    vols = returns.std().values
    
    for t in range(n_days):
        daily_ret = np.random.normal(0, 1, (n_sims, len(final_list))) * vols
        temp_prices *= np.exp(daily_ret)
        price_paths[t] = temp_prices
        
    portfolio_paths = np.sum(price_paths * [shares_dict[t] for t in final_list], axis=2)
    final_vals = portfolio_paths[-1, :]
    
    # Trajectoires clés
    p5 = portfolio_paths[:, np.argsort(final_vals)[int(n_sims * 0.05)]]
    p50 = portfolio_paths[:, np.argsort(final_vals)[int(n_sims * 0.50)]]
    p95 = portfolio_paths[:, np.argsort(final_vals)[int(n_sims * 0.95)]]

    # Simulation S&P 500 (pour comparaison au même montant)
    sp_ret = np.log(data_sp / data_sp.shift(1)).dropna()
    sp_last = data_sp.iloc[-1]
    sp_vol = sp_ret.std()
    sp_paths = np.zeros((n_days, n_sims))
    sp_temp = np.tile(sp_last, n_sims)
    for t in range(n_days):
        sp_temp *= np.exp(np.random.normal(0, 1, n_sims) * sp_vol)
        sp_paths[t] = sp_temp
    
    sp_scaled = (sp_paths / sp_last) * total_val_init
    sp_median = sp_scaled[:, np.argsort(sp_scaled[-1, :])[int(n_sims * 0.50)]]

    st.subheader(f"🚀 Simulation Monte Carlo (N={n_sims})")
    c1, c2 = st.columns([2, 1])
    with c1:
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='none'); ax.set_facecolor('none')
        ax.plot(p95, color='#00ff00', lw=1.5, label='Optimiste (95%)')
        ax.plot(p50, color='white', lw=2.5, label='Médian (Portefeuille)')
        ax.plot(p5, color='#ff4b4b', lw=1.5, label='Pessimiste (5%)')
        ax.plot(sp_median, color='#A020F0', lw=2, ls='--', label='Médian S&P 500')
        ax.fill_between(range(n_days), p5, p95, color='gray', alpha=0.1)
        plt.rcParams.update({"text.color": "white", "axes.labelcolor": "white", "xtick.color": "white", "ytick.color": "white"})
        ax.legend(facecolor='#0e1117', edgecolor='white')
        st.pyplot(fig, transparent=True)
    with c2:
        st.markdown("### 📊 Résultats à l'échéance")
        kpi = pd.DataFrame({
            "Indicateur": ["Valeur Initiale", "Médiane Finale", "Probabilité Profit", "VaR (95%)"],
            "Valeur": [f"{total_val_init:,.0f} €", f"{final_vals.mean():,.0f} €", f"{(final_vals > total_val_init).mean()*100:.1f} %", f"{total_val_init - np.percentile(final_vals, 5):,.0f} €"]
        })
        st.table(kpi.set_index("Indicateur"))

# --- OPTIMISATION ---
elif app_mode == "Optimisation & Frontière Efficiente" and run_btn:
    rets = raw_data_all[final_list][raw_data_all.index >= pd.Timestamp(start_date)].pct_change().dropna()
    mean_ret, cov = rets.mean() * 252, rets.cov() * 252
    
    # Portefeuille actuel
    curr_w = np.array([shares_dict[t] * raw_data_all[t].iloc[-1] for t in final_list]); curr_w /= np.sum(curr_w)
    curr_r, curr_v = np.sum(mean_ret * curr_w), np.sqrt(np.dot(curr_w.T, np.dot(cov, curr_w)))
    
    # Génération aléatoire
    res = np.zeros((3, n_portfolios)); w_store = []
    for i in range(n_portfolios):
        w = np.random.random(len(final_list)); w /= np.sum(w); w_store.append(w)
        r, v = np.sum(mean_ret * w), np.sqrt(np.dot(w.T, np.dot(cov, w)))
        res[0,i], res[1,i], res[2,i] = r, v, (r - rf_rate) / v
        
    best_idx = np.argmax(res[2])
    
    st.subheader("🎯 Optimisation Stratégique")
    c1, c2 = st.columns([2, 1])
    with c1:
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='none'); ax.set_facecolor('none')
        ax.scatter(res[1,:], res[0,:], c=res[2,:], cmap='viridis', s=10, alpha=0.3)
        ax.scatter(res[1,best_idx], res[0,best_idx], marker='*', color='r', s=200, label='Portefeuille Optimal')
        ax.scatter(curr_v, curr_r, marker='D', color='white', s=150, edgecolors='black', label='Actuel')
        plt.rcParams.update({"text.color": "white", "axes.labelcolor": "white", "xtick.color": "white", "ytick.color": "white"})
        ax.legend(); st.pyplot(fig, transparent=True)
    with c2:
        st.markdown("### ⚖️ Réallocation Suggérée")
        alloc = pd.DataFrame({'Actuel %': [round(x*100, 1) for x in curr_w], 'Optimal %': [round(x*100, 1) for x in w_store[best_idx]]}, index=final_list)
        st.table(alloc)
