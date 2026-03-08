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
        start_date = st.date_input("Depuis (Historique de volatilité) :", datetime.date(2021, 1, 1))
        n_days = st.number_input("Horizon de projection (jours)", value=252, min_value=1)
        st.caption("⚡ Simulations fixées à 5 000.")
        n_sims = 5000
        run_btn = st.button("🚀 LANCER SIMULATION")
        
    else:
        st.subheader("⚙️ Paramètres")
        start_date = st.date_input("Depuis :", datetime.date(2020, 1, 1))
        rf_rate = st.number_input("Taux sans risque %", value=3.0) / 100
        n_portfolios = st.number_input("Portefeuilles", value=5000)
        run_btn = st.button("🎯 GÉNÉRER FRONTIÈRE")

@st.cache_data
def load_data_portfolio(tickers): 
    data = yf.download(tickers + ["^GSPC"], start="2015-01-01", progress=False)['Close']
    return data.ffill().dropna()

raw_data = load_data_portfolio(final_list)

# --- ANALYSE HISTORIQUE ---
if app_mode == "Données historiques" and run_btn:
    p_map = {"1m":"1mo", "6m":"6mo", "1y":"1y", "3y":"3y", "5y":"5y", "10y":"10y", "all time":"max"}
    hist_all = yf.download(final_list + ["^GSPC"], period=p_map[selected_period], progress=False)['Close'].ffill().dropna()
    
    if not hist_all.empty:
        # Analyse Portefeuille
        hist_assets = hist_all[final_list]
        portfolio_val = sum(hist_assets[t] * shares_dict[t] for t in final_list)
        init_invested = sum(hist_assets[t].iloc[0] * shares_dict[t] for t in final_list)
        final_net_val = 0
        years = max((hist_assets.index[-1] - hist_assets.index[0]).days / 365.25, 0.1)
        
        for t in final_list:
            p_start, p_end = hist_assets[t].iloc[0], hist_assets[t].iloc[-1]
            qty, f = shares_dict[t], st.session_state.asset_fees[t]
            val_init_net = (p_start * qty) * (1 - f['entry'])
            val_after_mgmt = val_init_net * (p_end / p_start) * ((1 - f['mgmt']) ** years)
            gain = val_after_mgmt - (p_start * qty)
            final_net_val += val_after_mgmt - (gain * f['perf']) if gain > 0 else val_after_mgmt

        perf_brute_p = (portfolio_val.iloc[-1] / portfolio_val.iloc[0]) - 1
        perf_nette_p = (final_net_val / init_invested) - 1
        daily_ret_p = portfolio_val.pct_change().dropna()
        vol_p = daily_ret_p.std() * np.sqrt(252)
        sharpe_p = ((daily_ret_p.mean() * 252) - rf_hist) / vol_p if vol_p > 0 else 0
        dd_p = ((portfolio_val / portfolio_val.expanding().max()) - 1).min()

        # Benchmark S&P 500
        sp500 = hist_all["^GSPC"]
        perf_brute_sp = (sp500.iloc[-1] / sp500.iloc[0]) - 1
        daily_ret_sp = sp500.pct_change().dropna()
        vol_sp = daily_ret_sp.std() * np.sqrt(252)
        sharpe_sp = ((daily_ret_sp.mean() * 252) - rf_hist) / vol_sp if vol_sp > 0 else 0
        dd_sp = ((sp500 / sp500.expanding().max()) - 1).min()

        st.subheader("📊 Comparaison Historique")
        c1, c2 = st.columns([3, 1])
        with c1:
            fig, ax = plt.subplots(figsize=(10, 5), facecolor='none'); ax.set_facecolor('none')
            ax.plot((portfolio_val / portfolio_val.iloc[0]) * 100, color='#00ff00', lw=2.5, label="Votre Portefeuille")
            ax.plot((sp500 / sp500.iloc[0]) * 100, color='#A020F0', lw=2, alpha=0.8, label="S&P 500")
            plt.rcParams.update({"text.color": "white", "axes.labelcolor": "white", "xtick.color": "white", "ytick.color": "white"})
            ax.legend(facecolor='#0e1117', edgecolor='white')
            ax.grid(True, alpha=0.1, color='white')
            st.pyplot(fig, transparent=True)
        
        with c2:
            st.markdown("### 🟢 Portefeuille")
            st.metric("Perf. Brute", f"{perf_brute_p*100:.2f} %")
            st.metric("Perf. Nette", f"{perf_nette_p*100:.2f} %")
            st.metric("Sharpe", f"{sharpe_p:.2f}")
            st.divider()
            st.markdown("<h3 style='color: #A020F0;'>🟣 S&P 500</h3>", unsafe_allow_html=True)
            st.metric("Performance", f"{perf_brute_sp*100:.2f} %")
            st.metric("Sharpe", f"{sharpe_sp:.2f}")

# --- MONTE CARLO ---
elif app_mode == "Projection Monte Carlo" and run_btn:
    data_filtered = raw_data[final_list][raw_data.index >= pd.Timestamp(start_date)]
    returns = np.log(data_filtered / data_filtered.shift(1)).dropna()
    last_prices = data_filtered.iloc[-1]
    total_val_init = sum(last_prices[t] * shares_dict[t] for t in final_list)
    
    price_paths = np.zeros((n_days, n_sims, len(final_list)))
    temp_prices = np.tile(last_prices.values, (n_sims, 1))
    decay, ewma_var = 0.94, (returns**2).ewm(alpha=0.06, adjust=False).mean()
    sim_vols = np.tile(np.sqrt(ewma_var.iloc[-1].values), (n_sims, 1))
    
    for t in range(n_days):
        daily_ret = np.random.normal(0, 1, (n_sims, len(final_list))) * sim_vols
        temp_prices *= np.exp(daily_ret); price_paths[t] = temp_prices
        sim_vols = np.sqrt(decay * (sim_vols**2) + (1-decay) * (daily_ret**2))
        
    portfolio_paths = np.sum(price_paths * [shares_dict[t] for t in final_list], axis=2)
    final_values = portfolio_paths[-1, :]
    
    indices_sorted = np.argsort(final_values)
    path_5pct = portfolio_paths[:, indices_sorted[int(n_sims * 0.05)]]
    path_median = portfolio_paths[:, indices_sorted[int(n_sims * 0.50)]]
    path_95pct = portfolio_paths[:, indices_sorted[int(n_sims * 0.95)]]

    # Benchmark S&P 500 Projection
    sp_data = raw_data["^GSPC"][raw_data.index >= pd.Timestamp(start_date)]
    sp_ret = np.log(sp_data / sp_data.shift(1)).dropna()
    sp_last = sp_data.iloc[-1]
    sp_paths = np.zeros((n_days, n_sims))
    sp_temp = np.tile(sp_last, n_sims)
    sp_vols = np.tile(sp_ret.std(), n_sims)
    
    for t in range(n_days):
        r = np.random.normal(0, 1, n_sims) * sp_vols
        sp_temp *= np.exp(r); sp_paths[t] = sp_temp
    
    sp_scaled = (sp_paths / sp_last) * total_val_init
    sp_median_path = sp_scaled[:, np.argsort(sp_scaled[-1, :])[int(n_sims * 0.50)]]

    st.subheader(f"🚀 Projection Monte Carlo ({n_sims} itérations)")
    c1, c2 = st.columns([2, 1])
    with c1:
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='none'); ax.set_facecolor('none')
        time_range = np.arange(n_days)
        ax.plot(time_range, path_95pct, color='#00ff00', lw=2, label='Optimiste (95%)')
        ax.plot(time_range, path_median, color='white', lw=3, label='Médian (50%)')
        ax.plot(time_range, path_5pct, color='#ff4b4b', lw=2, label='Pessimiste (5%)')
        ax.plot(time_range, sp_median_path, color='#A020F0', lw=2, ls='--', label='S&P 500 Médian')
        ax.fill_between(time_range, path_5pct, path_95pct, color='gray', alpha=0.1)
        plt.rcParams.update({"text.color": "white", "axes.labelcolor": "white", "xtick.color": "white", "ytick.color": "white"})
        ax.legend(facecolor='#0e1117', edgecolor='white')
        st.pyplot(fig, transparent=True)

    with c2:
        st.markdown("### 🛠️ Indicateurs")
        kpi_df = pd.DataFrame({
            "Indicateur": ["Initial", "Médiane Finale", "Probabilité Profit", "VaR 95%"],
            "Valeur": [f"{total_val_init:,.0f} €", f"{final_values.mean():,.0f} €", f"{(final_values > total_val_init).mean()*100:.1f} %", f"{total_val_init - np.percentile(final_values, 5):,.0f} €"]
        })
        st.table(kpi_df.set_index("Indicateur"))

# --- OPTIMISATION ---
elif app_mode == "Optimisation & Frontière Efficiente" and run_btn:
    returns_daily = raw_data[final_list][raw_data.index >= pd.Timestamp(start_date)].pct_change().dropna()
    mean_ret, cov = returns_daily.mean() * 252, returns_daily.cov() * 252
    
    curr_w = np.array([shares_dict[t] * raw_data[t].iloc[-1] for t in final_list]); curr_w /= np.sum(curr_w)
    curr_ret, curr_vol = np.sum(mean_ret * curr_w), np.sqrt(np.dot(curr_w.T, np.dot(cov, curr_w)))
    
    res = np.zeros((3, n_portfolios)); w_rec = []
    for i in range(n_portfolios):
        w = np.random.random(len(final_list)); w /= np.sum(w); w_rec.append(w)
        r, v = np.sum(mean_ret * w), np.sqrt(np.dot(w.T, np.dot(cov, w)))
        res[0,i], res[1,i], res[2,i] = r, v, (r - rf_rate) / v
        
    best_idx = np.argmax(res[2])
    
    st.subheader("🎯 Frontière Efficiente")
    c1, c2 = st.columns([2, 1])
    with c1:
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='none'); ax.set_facecolor('none')
        ax.scatter(res[1,:], res[0,:], c=res[2,:], cmap='viridis', s=10, alpha=0.3)
        ax.scatter(res[1,best_idx], res[0,best_idx], marker='*', color='r', s=200, label='Optimal')
        ax.scatter(curr_vol, curr_ret, marker='D', color='white', s=150, edgecolors='black', label='Actuel')
        plt.rcParams.update({"text.color": "white", "axes.labelcolor": "white", "xtick.color": "white", "ytick.color": "white"})
        ax.legend(); st.pyplot(fig, transparent=True)
    with c2:
        st.table(pd.DataFrame({'Actuel %': [round(x*100, 1) for x in curr_w], 'Optimal %': [round(x*100, 1) for x in w_rec[best_idx]]}, index=final_list))
