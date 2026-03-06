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
    except: st.error("Erreur flux")

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
        st.subheader("⚙️ Configuration des frais")
        selected_period = st.selectbox("Période d'analyse", ["1m", "6m", "1y", "3y", "5y", "10y", "all time"])
        for t in final_list:
            with st.expander(f"Frais : {t}"):
                st.session_state.asset_fees[t]['entry'] = st.number_input(f"Entrée %", 0.0, step=0.1, key=f"ent_{t}") / 100
                st.session_state.asset_fees[t]['mgmt'] = st.number_input(f"Gestion Annuelle %", 0.0, step=0.1, key=f"mgt_{t}") / 100
                st.session_state.asset_fees[t]['perf'] = st.number_input(f"Surperformance %", 0.0, step=1.0, key=f"prf_{t}") / 100
        rf_hist = st.number_input("Taux sans risque (Sharpe) %", 3.0) / 100
        run_btn = st.button("📈 ANALYSER")
    elif app_mode == "Projection Monte Carlo":
        start_date = st.date_input("Depuis :", datetime.date(2021, 1, 1))
        n_days, n_sims = st.number_input("Horizon", 150), st.number_input("Sims", 2000)
        run_btn = st.button("🚀 LANCER SIMULATION")
    else:
        start_date = st.date_input("Depuis :", datetime.date(2020, 1, 1))
        rf_rate, n_portfolios = st.number_input("Taux sans risque %", 3.0)/100, st.number_input("Portefeuilles", 5000)
        run_btn = st.button("🎯 GÉNÉRER FRONTIÈRE")

@st.cache_data
def load_data_portfolio(tickers): 
    data = yf.download(tickers, start="2015-01-01", progress=False)['Close']
    return data.ffill().dropna()

raw_data = load_data_portfolio(final_list)

# --- LOGIQUE ANALYSE HISTORIQUE ---
if app_mode == "Données historiques" and run_btn:
    p_map = {"1m":"1mo", "6m":"6mo", "1y":"1y", "3y":"3y", "5y":"5y", "10y":"10y", "all time":"max"}
    
    # Récupération des données selon la période
    hist_df = yf.download(final_list, period=p_map[selected_period], progress=False)['Close'].ffill().dropna()
    
    if not hist_df.empty:
        # 1. Calcul de la performance brute du portefeuille (Valeur quotidienne)
        portfolio_value_series = sum(hist_df[t] * shares_dict[t] for t in final_list)
        
        # 2. Calcul des performances individuelles NETTES pour chaque actif
        total_initial_invested = sum(hist_df[t].iloc[0] * shares_dict[t] for t in final_list)
        total_final_net_value = 0
        
        years = (hist_df.index[-1] - hist_df.index[0]).days / 365.25
        
        for t in final_list:
            price_start = hist_df[t].iloc[0]
            price_end = hist_df[t].iloc[-1]
            qty = shares_dict[t]
            f = st.session_state.asset_fees[t]
            
            # Valeur initiale après frais d'entrée
            val_init_net = (price_start * qty) * (1 - f['entry'])
            # Croissance brute et frais de gestion
            val_apres_gestion = val_init_net * (price_end / price_start) * ((1 - f['mgmt']) ** years)
            # Frais de surperformance
            gain = val_apres_gestion - (price_start * qty)
            val_finale_asset = val_apres_gestion - (gain * f['perf']) if gain > 0 else val_apres_gestion
            
            total_final_net_value += val_finale_asset

        # Métriques globales
        perf_brute_globale = (portfolio_value_series.iloc[-1] / portfolio_value_series.iloc[0]) - 1
        perf_nette_globale = (total_final_net_value / total_initial_invested) - 1
        
        daily_ret = portfolio_value_series.pct_change().dropna()
        vol = daily_ret.std() * np.sqrt(252)
        sharpe = ((daily_ret.mean() * 252) - rf_hist) / vol if vol > 0 else 0
        max_dd = ((portfolio_value_series / portfolio_value_series.expanding().max()) - 1).min()

        # AFFICHAGE
        title_text = "Portefeuille Global" if len(final_list) > 1 else f"{st.session_state.portfolio[final_list[0]]}"
        st.subheader(f"📊 Analyse : {title_text}")
        
        c1, c2 = st.columns([3, 1])
        with c1:
            fig, ax = plt.subplots(figsize=(10, 5), facecolor='none'); ax.set_facecolor('none')
            # On normalise en base 100 pour la lisibilité si plusieurs actifs
            norm_series = (portfolio_value_series / portfolio_value_series.iloc[0]) * 100
            ax.plot(norm_series.index, norm_series.values, color='#00ff00', lw=2)
            ax.fill_between(norm_series.index, norm_series.values, norm_series.min()*0.95, color='#00ff00', alpha=0.1)
            plt.rcParams.update({"text.color": "white", "axes.labelcolor": "white", "xtick.color": "white", "ytick.color": "white"})
            ax.set_ylabel("Base 100")
            ax.grid(True, alpha=0.1, color='white'); st.pyplot(fig, transparent=True)
        
        with c2:
            st.metric("Performance Brute", f"{perf_brute_globale*100:.2f} %")
            st.metric("Performance Nette", f"{perf_nette_globale*100:.2f} %", f"Frais totaux: {((perf_brute_globale - perf_nette_globale)*100):+.2f}%", delta_color="inverse")
            st.metric("Volatilité Annuelle", f"{vol*100:.2f} %")
            st.metric("Max Drawdown", f"{max_dd*100:.2f} %")
            st.metric("Ratio de Sharpe", f"{sharpe:.2f}")
            if len(final_list) > 1:
                st.caption(f"Calcul basé sur {len(final_list)} actifs.")

# --- LES AUTRES MODES (MONTE CARLO / OPTI) ---
# (La logique reste identique à la version précédente pour assurer la cohérence)
elif app_mode == "Projection Monte Carlo" and run_btn:
    data_filtered = raw_data[raw_data.index >= pd.Timestamp(start_date)]
    returns = np.log(data_filtered / data_filtered.shift(1)).dropna()
    last_prices = data_filtered.iloc[-1]
    total_val = sum(last_prices[t] * shares_dict[t] for t in final_list)
    price_paths = np.zeros((n_days, n_sims, len(final_list)))
    temp_prices = np.tile(last_prices.values, (n_sims, 1))
    decay, ewma_var = 0.94, (returns**2).ewm(alpha=0.06, adjust=False).mean()
    sim_vols = np.tile(np.sqrt(ewma_var.iloc[-1].values), (n_sims, 1))
    for t in range(n_days):
        daily_ret = np.random.normal(0, 1, (n_sims, len(final_list))) * sim_vols
        temp_prices *= np.exp(daily_ret); price_paths[t] = temp_prices
        sim_vols = np.sqrt(decay * (sim_vols**2) + (1-decay) * (daily_ret**2))
    portfolio_paths = np.sum(price_paths * [shares_dict[t] for t in final_list], axis=2)
    final_pnl = portfolio_paths[-1, :] - total_val
    st.columns(3)[1].metric("Issue Médiane", f"{np.median(final_pnl):,.2f} €", f"{(np.median(final_pnl)/total_val)*100:.2f} %")
    fig = plt.figure(figsize=(16, 7), facecolor='none'); gs = GridSpec(1, 2, width_ratios=[1.8, 1])
    ax1 = fig.add_subplot(gs[0], facecolor='none'); norm = plt.Normalize(final_pnl.min(), final_pnl.max())
    for i in np.random.choice(n_sims, 100): ax1.plot(portfolio_paths[:, i], color=plt.cm.RdYlGn(norm(final_pnl[i])), alpha=0.3)
    ax2 = fig.add_subplot(gs[1], facecolor='none'); n, bins, patches = ax2.hist(final_pnl, 50, density=True, alpha=0.8)
    for b, p in zip(bins, patches): p.set_facecolor('red' if b < 0 else 'green')
    st.pyplot(fig, transparent=True)

elif app_mode == "Optimisation & Frontière Efficiente" and run_btn:
    returns_daily = raw_data[raw_data.index >= pd.Timestamp(start_date)].pct_change().dropna()
    mean_ret, cov = returns_daily.mean()*252, returns_daily.cov()*252
    curr_w = np.array([shares_dict[t]*raw_data[t].iloc[-1] for t in final_list]); curr_w /= np.sum(curr_w)
    curr_ret, curr_vol = np.sum(mean_ret*curr_w), np.sqrt(np.dot(curr_w.T, np.dot(cov, curr_w)))
    res = np.zeros((3, n_portfolios)); w_rec = []
    for i in range(n_portfolios):
        w = np.random.random(len(final_list)); w /= np.sum(w); w_rec.append(w)
        r, v = np.sum(mean_ret*w), np.sqrt(np.dot(w.T, np.dot(cov, w)))
        res[0,i], res[1,i], res[2,i] = r, v, (r-rf_rate)/v
    best_idx = np.argmax(res[2])
    c1, c2 = st.columns([2, 1])
    with c1:
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='none'); ax.set_facecolor('none')
        ax.scatter(res[1,:], res[0,:], c=res[2,:], cmap='viridis', s=10, alpha=0.3)
        ax.scatter(res[1,best_idx], res[0,best_idx], marker='*', color='r', s=200, label='Optimal')
        ax.scatter(curr_vol, curr_ret, marker='D', color='white', s=150, edgecolors='black', label='Actuel')
        ax.legend(); st.pyplot(fig, transparent=True)
    with c2: st.table(pd.DataFrame({'Actuel %': [round(x*100, 1) for x in curr_w], 'Optimal %': [round(x*100, 1) for x in w_rec[best_idx]]}, index=final_list))
