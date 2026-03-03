import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec
import datetime

# --- CONFIGURATION DE LA PAGE & MASQUAGE MENU ---
st.set_page_config(page_title="European Portfolio Master Pro", layout="wide")

# Injection CSS pour cacher les éléments de développement Streamlit
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display:none;}
    </style>
""", unsafe_allow_html=True)

# --- BANDEAU ANIMÉ ---
def display_animated_ticker():
    indices = {
        "^FCHI": "CAC 40", "^GDAXI": "DAX 40", "^STOXX50E": "EURO 50",
        "^GSPC": "S&P 500", "^IXIC": "NASDAQ", "BTC-USD": "BITCOIN"
    }
    try:
        data = yf.download(list(indices.keys()), period="5d", progress=False)['Close'].ffill()
        items = ""
        for ticker, name in indices.items():
            val = data[ticker].dropna()
            if len(val) >= 2:
                change = ((val.iloc[-1] - val.iloc[-2]) / val.iloc[-2]) * 100
                color = "#00ff00" if change >= 0 else "#ff4b4b"
                items += f"&nbsp;&nbsp;&nbsp; <b>{name}</b> {val.iloc[-1]:,.2f} <span style='color:{color};'>{'▲' if change>=0 else '▼'} {change:.2f}%</span> &nbsp;&nbsp;&nbsp; |"
        st.markdown(f"""
            <div style="overflow:hidden; background:#0e1117; padding:10px 0; border-bottom:1px solid #31333f; white-space:nowrap;">
                <div style="display:inline-block; animation: marquee 100s linear infinite;">
                    <style>@keyframes marquee {{0% {{transform:translateX(0);}} 100% {{transform:translateX(-50%);}}}}</style>
                    {items * 3}
                </div>
            </div>
        """, unsafe_allow_html=True)
    except: pass

display_animated_ticker()
st.title("THE FRENCH BUILT TOOL FOR STRATEGIC INVESTING")

# --- FONCTION DE RECHERCHE ROBUSTE (NOM COMPLET) ---
@st.cache_data(ttl=86400) # Cache le nom pendant 24h
def get_verified_name(symbol):
    try:
        # Priorité 1 : La fonction Search (très fiable pour le nom)
        s = yf.Search(symbol, max_results=1)
        if s.quotes:
            return s.quotes[0].get('longname') or s.quotes[0].get('shortname') or symbol
        
        # Priorité 2 : Le dictionnaire info (si Search échoue)
        t = yf.Ticker(symbol)
        return t.info.get('longName') or symbol
    except:
        return symbol

# --- SIDEBAR ET GESTION PORTEFEUILLE ---
with st.sidebar:
    st.header("🧭 Navigation")
    app_mode = st.radio("Outil :", ["Projection Monte Carlo", "Optimisation & Frontière"])
    st.divider()

    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = {}

    search_q = st.text_input("Ajouter Ticker (ex: MC.PA, AAPL) :").upper().strip()
    if st.button("➕ Ajouter"):
        if search_q and search_q not in st.session_state.portfolio:
            with st.spinner('Identification...'):
                full_n = get_verified_name(search_q)
                st.session_state.portfolio[search_q] = full_n
            st.rerun()

    if st.session_state.portfolio:
        st.write("### Actifs :")
        to_del = []
        for t, n in st.session_state.portfolio.items():
            c1, c2 = st.columns([4, 1])
            c1.markdown(f"**{n}**\n<small>{t}</small>", unsafe_allow_html=True)
            if c2.button("🗑️", key=f"del_{t}"): to_del.append(t)
        for t in to_del:
            del st.session_state.portfolio[t]
            st.rerun()
    else:
        st.info("Ajoutez des actifs.")
        st.stop()

    st.divider()
    final_list = list(st.session_state.portfolio.keys())
    shares = {t: st.number_input(f"Quantité {t}", value=10, min_value=1) for t in final_list}
    
    run_btn = st.button("🚀 CALCULER")

# --- CHARGEMENT DONNÉES ---
@st.cache_data
def load_prices(tickers):
    return yf.download(tickers, start="2019-01-01", progress=False)['Close'].ffill().dropna()

df_prices = load_prices(final_list)

# --- LOGIQUE : MONTE CARLO ---
if app_mode == "Projection Monte Carlo" and run_btn:
    rets = np.log(df_prices / df_prices.shift(1)).dropna()
    last_p = df_prices.iloc[-1]
    total_val = sum(last_p[t] * shares[t] for t in final_list)
    
    # Simulation simplifiée EWMA
    n_days, n_sims = 150, 2000
    price_paths = np.zeros((n_days, n_sims, len(final_list)))
    curr_p = np.tile(last_p.values, (n_sims, 1))
    vols = np.tile(rets.std().values, (n_sims, 1))

    for t in range(n_days):
        shocks = np.random.normal(0, 1, (n_sims, len(final_list)))
        curr_p *= np.exp(shocks * vols)
        price_paths[t] = curr_p

    p_paths = np.sum(price_paths * [shares[t] for t in final_list], axis=2)
    pnl = p_paths[-1, :] - total_val
    
    st.metric("Résultat Médian", f"{np.median(pnl):,.2f} €")
    
    fig = plt.figure(figsize=(15, 6), facecolor='none')
    gs = GridSpec(1, 2, width_ratios=[1.5, 1])
    ax1 = fig.add_subplot(gs[0], facecolor='none')
    ax1.plot(p_paths[:, :100], alpha=0.3)
    ax1.set_title("Trajectoires du Portefeuille", color='white')
    
    ax2 = fig.add_subplot(gs[1], facecolor='none')
    ax2.hist(pnl, bins=50, color='skyblue', alpha=0.7)
    ax2.set_title("Distribution P&L", color='white')
    plt.tight_layout()
    st.pyplot(fig, transparent=True)

# --- LOGIQUE : OPTIMISATION ---
elif app_mode == "Optimisation & Frontière" and run_btn:
    rets_d = df_prices.pct_change().dropna()
    mean_r, cov_m = rets_d.mean() * 252, rets_d.cov() * 252
    
    # Portefeuille Actuel
    cur_v = np.array([shares[t] * df_prices.iloc[-1][t] for t in final_list])
    cur_w = cur_v / cur_v.sum()
    cur_r = np.sum(mean_r * cur_w)
    cur_v = np.sqrt(cur_w.T @ cov_m @ cur_w)

    # Simulation Frontière
    res = np.zeros((3, 5000))
    for i in range(5000):
        w = np.random.random(len(final_list)); w /= w.sum()
        res[0,i] = np.sum(mean_r * w)
        res[1,i] = np.sqrt(w.T @ cov_m @ w)
        res[2,i] = (res[0,i] - 0.03) / res[1,i]

    st.subheader("🎯 Analyse de la Performance")
    col1, col2 = st.columns([2, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='none')
        ax.set_facecolor('none')
        s = ax.scatter(res[1,:], res[0,:], c=res[2,:], cmap='viridis', alpha=0.4)
        ax.scatter(cur_v, cur_r, color='white', marker='D', s=100, label="Actuel")
        ax.set_xlabel("Volatilité", color='white'); ax.set_ylabel("Rendement", color='white')
        ax.legend(); st.pyplot(fig, transparent=True)
    with col2:
        st.write("**Poids du Portefeuille**")
        st.table(pd.DataFrame({'Actif': [st.session_state.portfolio[t] for t in final_list], 
                               'Poids (%)': [f"{x*100:.1f}%" for x in cur_w]}))
