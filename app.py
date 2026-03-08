import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime

# --- CONFIGURATION ---
st.set_page_config(page_title="European Portfolio Master Pro", layout="wide")
hide_st_style = """<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;} .stDeployButton {display:none;}</style>"""
st.markdown(hide_st_style, unsafe_allow_html=True)

# --- BANDEAU MARCHÉ ---
def display_animated_ticker():
    indices = {"^FCHI": "CAC 40", "^GDAXI": "DAX 40", "^STOXX50E": "EURO 50", "^GSPC": "S&P 500", "BTC-USD": "BITCOIN", "GC=F": "OR"}
    try:
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
    app_mode = st.radio("Choisir l'outil :", ["Analyse Stratégique Complète", "Données historiques"])
    st.divider()
    
    st.header("🛒 Portefeuille")
    if 'portfolio' not in st.session_state: st.session_state.portfolio = {}
    
    search_input = st.text_input("Ajouter Ticker (ex: MC.PA, ASML, BTC-USD) :").upper()
    if st.button("➕ Ajouter"):
        if search_input:
            st.session_state.portfolio[search_input] = get_full_ticker_info(search_input)
            st.rerun()

    if st.session_state.portfolio:
        to_delete = []
        for t, name in st.session_state.portfolio.items():
            c1, c2 = st.columns([4, 1])
            c1.caption(f"**{t}** : {name}")
            if c2.button("x", key=f"del_{t}"): to_delete.append(t)
        for t in to_delete: del st.session_state.portfolio[t]; st.rerun()

    final_list = list(st.session_state.portfolio.keys())
    if not final_list: st.info("Ajoutez des actifs pour commencer."); st.stop()
    
    st.divider()
    shares_dict = {t: st.number_input(f"Quantité {t}", value=10, min_value=1) for t in final_list}
    
    st.subheader("⚙️ Paramètres de l'Analyse")
    start_date = st.date_input("Historique de référence :", datetime.date(2021, 1, 1))
    horizon = st.number_input("Horizon de projection (jours)", value=252)
    rf_rate = st.number_input("Taux sans risque %", value=3.0) / 100
    run_btn = st.button("🚀 LANCER L'ANALYSE GLOBALE")

# --- CHARGEMENT DES DONNÉES ---
@st.cache_data
def load_data_all(tickers):
    all_t = list(set(tickers + ["^GSPC"]))
    return yf.download(all_t, start="2015-01-01", progress=False)['Close'].ffill().dropna()

raw_data = load_data_all(final_list)

if run_btn:
    # --- PRÉPARATION DES DONNÉES ---
    df = raw_data[raw_data.index >= pd.Timestamp(start_date)]
    df_port = df[final_list]
    df_sp = df["^GSPC"]
    
    rets_daily = df_port.pct_change().dropna()
    sp_rets_daily = df_sp.pct_change().dropna()
    
    # --- CALCUL ALPHA / BETA ---
    # Beta
    cov_matrix = np.cov(rets_daily.mean(axis=1), sp_rets_daily)[0, 1]
    market_var = np.var(sp_rets_daily)
    beta = cov_matrix / market_var
    
    # Alpha (Jensen)
    port_ann_ret = rets_daily.mean(axis=1).mean() * 252
    mkt_ann_ret = sp_rets_daily.mean() * 252
    alpha = port_ann_ret - (rf_rate + beta * (mkt_ann_ret - rf_rate))

    # --- SECTION 1 : MONTE CARLO ---
    st.header("1️⃣ Projection Monte Carlo")
    
    n_sims = 5000
    last_prices = df_port.iloc[-1]
    total_val_init = sum(last_prices[t] * shares_dict[t] for t in final_list)
    
    # Simulation
    log_rets = np.log(df_port / df_port.shift(1)).dropna()
    vols = log_rets.std().values
    
    price_paths = np.zeros((horizon, n_sims, len(final_list)))
    temp_prices = np.tile(last_prices.values, (n_sims, 1))
    
    for t in range(horizon):
        temp_prices *= np.exp(np.random.normal(0, 1, (n_sims, len(final_list))) * vols)
        price_paths[t] = temp_prices
        
    portfolio_paths = np.sum(price_paths * [shares_dict[t] for t in final_list], axis=2)
    final_vals = portfolio_paths[-1, :]
    
    # Stats Monte Carlo
    p50 = portfolio_paths[:, np.argsort(final_vals)[int(n_sims*0.5)]]
    p5 = portfolio_paths[:, np.argsort(final_vals)[int(n_sims*0.05)]]
    p95 = portfolio_paths[:, np.argsort(final_vals)[int(n_sims*0.95)]]
    
    col1, col2 = st.columns([2, 1])
    with col1:
        fig1, ax1 = plt.subplots(figsize=(10, 5), facecolor='none'); ax1.set_facecolor('none')
        ax1.plot(p95, color='#00ff00', label='Optimiste (95%)', alpha=0.6)
        ax1.plot(p50, color='white', lw=3, label='Médian')
        ax1.plot(p5, color='#ff4b4b', label='Pessimiste (5%)', alpha=0.6)
        ax1.fill_between(range(horizon), p5, p95, color='gray', alpha=0.1)
        plt.rcParams.update({"text.color": "white", "axes.labelcolor": "white", "xtick.color": "white", "ytick.color": "white"})
        ax1.legend(); st.pyplot(fig1, transparent=True)
    
    with col2:
        st.metric("Valeur Médiane Attendue", f"{p50[-1]:,.0f} €")
        st.metric("Gain Potentiel Médian", f"{p50[-1] - total_val_init:+,.0f} €")
        st.metric("Alpha (Performance Pure)", f"{alpha*100:+.2f} %", help="L'Alpha mesure votre talent : la performance au-dessus du marché ajustée du risque.")
        st.metric("Beta (Exposition Marché)", f"{beta:.2f}", help="Le Beta mesure votre sensibilité au S&P 500. > 1 = Agressif, < 1 = Défensif.")

    st.divider()

    # --- SECTION 2 : FRONTIÈRE EFFICIENTE ---
    st.header("2️⃣ Optimisation de la Frontière Efficiente")
    
    mean_ret = rets_daily.mean() * 252
    cov_matrix_opt = rets_daily.cov() * 252
    
    # Portefeuille actuel
    weights_curr = np.array([shares_dict[t] * last_prices[t] for t in final_list])
    weights_curr /= np.sum(weights_curr)
    curr_ret = np.sum(mean_ret * weights_curr)
    curr_vol = np.sqrt(np.dot(weights_curr.T, np.dot(cov_matrix_opt, weights_curr)))
    
    # Simulation Portefeuilles Aléatoires
    n_portfolios = 5000
    results = np.zeros((3, n_portfolios))
    w_store = []
    
    for i in range(n_portfolios):
        w = np.random.random(len(final_list))
        w /= np.sum(w)
        w_store.append(w)
        r = np.sum(mean_ret * w)
        v = np.sqrt(np.dot(w.T, np.dot(cov_matrix_opt, w)))
        results[0,i], results[1,i], results[2,i] = r, v, (r - rf_rate) / v
        
    best_idx = np.argmax(results[2])
    best_w = w_store[best_idx]
    
    col3, col4 = st.columns([2, 1])
    with col3:
        fig2, ax2 = plt.subplots(figsize=(10, 5), facecolor='none'); ax2.set_facecolor('none')
        ax2.scatter(results[1,:], results[0,:], c=results[2,:], cmap='viridis', s=10, alpha=0.3)
        ax2.scatter(curr_vol, curr_ret, marker='D', color='white', s=100, label='Votre Portefeuille')
        ax2.scatter(results[1,best_idx], results[0,best_idx], marker='*', color='red', s=200, label='Optimal (Sharpe)')
        ax2.set_xlabel("Volatilité (Risque)"); ax2.set_ylabel("Rendement Attendu")
        ax2.legend(); st.pyplot(fig2, transparent=True)
        
    with col4:
        st.subheader("⚖️ Réallocation Suggérée")
        comparison = pd.DataFrame({
            "Actuel %": weights_curr * 100,
            "Optimal %": best_w * 100
        }, index=final_list).round(1)
        st.table(comparison)
