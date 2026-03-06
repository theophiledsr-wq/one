import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec
import datetime

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="European Portfolio Master Pro", layout="wide")
hide_st_style = """<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;} .stDeployButton {display:none;}</style>"""
st.markdown(hide_st_style, unsafe_allow_html=True)

# --- BANDEAU DÉFILANT DES MARCHÉS ---
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
    except: st.error("Erreur de flux de données")

display_animated_ticker()
st.title("THE FRENCH BUILT TOOL FOR STRATEGIC INVESTING")

def get_full_ticker_info(symbol):
    try:
        search = yf.Search(symbol, max_results=1)
        if search.quotes: return search.quotes[0].get('longname') or search.quotes[0].get('shortname')
        return yf.Ticker(symbol).info.get('longName') or symbol
    except: return symbol

# --- BARRE LATÉRALE (SIDEBAR) ---
with st.sidebar:
    st.header("🧭 Navigation")
    app_mode = st.radio("Choisir l'outil :", ["Projection Monte Carlo", "Optimisation & Frontière Efficiente", "Données historiques"])
    st.divider()
    
    st.header("🛒 Portefeuille")
    if 'portfolio' not in st.session_state: st.session_state.portfolio = {}
    if 'asset_fees' not in st.session_state: st.session_state.asset_fees = {}
    
    search_input = st.text_input("Ajouter un Ticker (ex: MC.PA, AAPL) :").upper()
    if st.button("➕ Ajouter"):
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
    if not final_list: st.info("Veuillez ajouter des actifs."); st.stop()
    
    st.divider()
    shares_dict = {t: st.number_input(f"Quantité {t}", value=10, min_value=1, key=f"qty_{t}") for t in final_list}
    
    if app_mode == "Données historiques":
        st.subheader("⚙️ Paramètres & Frais")
        selected_period = st.selectbox("Période d'analyse", ["1m", "6m", "1y", "3y", "5y", "10y", "all time"])
        for t in final_list:
            with st.expander(f"Frais pour {t}"):
                st.session_state.asset_fees[t]['entry'] = st.number_input(f"Entrée (%)", 0.0, step=0.1, key=f"ent_{t}") / 100
                st.session_state.asset_fees[t]['mgmt'] = st.number_input(f"Gestion Annuelle (%)", 0.0, step=0.1, key=f"mgt_{t}") / 100
                st.session_state.asset_fees[t]['perf'] = st.number_input(f"Surperformance (%)", 0.0, step=1.0, key=f"prf_{t}") / 100
        rf_hist = st.number_input("Taux sans risque (%)", 3.0) / 100
        run_btn = st.button("📈 LANCER L'ANALYSE")
    elif app_mode == "Projection Monte Carlo":
        start_date = st.date_input("Depuis :", datetime.date(2021, 1, 1))
        n_days, n_sims = st.number_input("Horizon (jours)", 150), st.number_input("Simulations", 2000)
        run_btn = st.button("🚀 LANCER SIMULATION")
    else:
        start_date = st.date_input("Depuis :", datetime.date(2020, 1, 1))
        rf_rate, n_portfolios = st.number_input("Taux sans risque (%)", 3.0)/100, st.number_input("Nombre Portefeuilles", 5000)
        run_btn = st.button("🎯 GÉNÉRER FRONTIÈRE")

# --- CHARGEMENT DES DONNÉES ---
@st.cache_data
def load_data(tickers, start="2015-01-01"): 
    data = yf.download(tickers, start=start, progress=False)['Close']
    return data.ffill().dropna()

raw_data = load_data(final_list + ["^GSPC"])

# --- MODE DONNÉES HISTORIQUES ---
if app_mode == "Données historiques" and run_btn:
    p_map = {"1m":"1mo", "6m":"6mo", "1y":"1y", "3y":"3y", "5y":"5y", "10y":"10y", "all time":"max"}
    hist_all = yf.download(final_list + ["^GSPC"], period=p_map[selected_period], progress=False)['Close'].ffill().dropna()
    
    if not hist_all.empty:
        # 1. Analyse du Portefeuille
        hist_assets = hist_all[final_list]
        portfolio_val = sum(hist_assets[t] * shares_dict[t] for t in final_list)
        init_invested = sum(hist_assets[t].iloc[0] * shares_dict[t] for t in final_list)
        final_net_val = 0
        years = (hist_assets.index[-1] - hist_assets.index[0]).days / 365.25
        
        for t in final_list:
            p_start, p_end = hist_assets[t].iloc[0], hist_assets[t].iloc[-1]
            qty, f = shares_dict[t], st.session_state.asset_fees[t]
            val_init_net = (p_start * qty) * (1 - f['entry'])
            val_after_mgmt = val_init_net * (p_end / p_start) * ((1 - f['mgmt']) ** (years if years > 0 else 1))
            gain = val_after_mgmt - (p_start * qty)
            final_net_val += val_after_mgmt - (gain * f['perf']) if gain > 0 else val_after_mgmt

        perf_brute_p = (portfolio_val.iloc[-1] / portfolio_val.iloc[0]) - 1
        perf_nette_p = (final_net_val / init_invested) - 1
        daily_ret_p = portfolio_val.pct_change().dropna()
        vol_p = daily_ret_p.std() * np.sqrt(252)
        sharpe_p = ((daily_ret_p.mean() * 252) - rf_hist) / vol_p if vol_p > 0 else 0
        dd_p = ((portfolio_val / portfolio_val.expanding().max()) - 1).min()

        # 2. Analyse du Benchmark (S&P 500)
        sp500 = hist_all["^GSPC"]
        perf_brute_sp = (sp500.iloc[-1] / sp500.iloc[0]) - 1
        daily_ret_sp = sp500.pct_change().dropna()
        vol_sp = daily_ret_sp.std() * np.sqrt(252)
        sharpe_sp = ((daily_ret_sp.mean() * 252) - rf_hist) / vol_sp if vol_sp > 0 else 0
        dd_sp = ((sp500 / sp500.expanding().max()) - 1).min()

        # AFFICHAGE
        st.subheader(f"📊 Analyse Comparative : Portefeuille vs S&P 500")
        c1, c2 = st.columns([3, 1])
        with c1:
            fig, ax = plt.subplots(figsize=(10, 5), facecolor='none'); ax.set_facecolor('none')
            ax.plot((portfolio_val / portfolio_val.iloc[0]) * 100, color='#00ff00', lw=2.5, label="Votre Portefeuille")
            ax.plot((sp500 / sp500.iloc[0]) * 100, color='#A020F0', lw=2, alpha=0.7, label="S&P 500")
            plt.rcParams.update({"text.color": "white", "axes.labelcolor": "white", "xtick.color": "white", "ytick.color": "white"})
            ax.legend(facecolor='#0e1117', edgecolor='white')
            ax.grid(True, alpha=0.1, color='white'); st.pyplot(fig, transparent=True)
        
        with c2:
            st.markdown("### 🟢 Portefeuille")
            st.metric("Perf. Brute", f"{perf_brute_p*100:.2f} %")
            st.metric("Perf. Nette", f"{perf_nette_p*100:.2f} %")
            st.metric("Sharpe", f"{sharpe_p:.2f}")
            st.metric("Max Drawdown", f"{dd_p*100:.2f} %")
            st.divider()
            st.markdown("<h3 style='color: #A020F0;'>🟣 S&P 500</h3>", unsafe_allow_html=True)
            st.markdown(f"<p style='color: #A020F0; font-size: 1.2rem;'>Perf. Brute: <b>{perf_brute_sp*100:.2f} %</b></p>", unsafe_allow_html=True)
            st.markdown(f"<p style='color: #A020F0; font-size: 1.2rem;'>Sharpe: <b>{sharpe_sp:.2f}</b></p>", unsafe_allow_html=True)
            st.markdown(f"<p style='color: #A020F0; font-size: 1.2rem;'>Max Drawdown: <b>{dd_sp*100:.2f} %</b></p>", unsafe_allow_html=True)

# --- MODE MONTE CARLO ---
elif app_mode == "Projection Monte Carlo" and run_btn:
    data_f = raw_data[final_list][raw_data.index >= pd.Timestamp(start_date)]
    rets = np.log(data_f / data_f.shift(1)).dropna()
    last_p = data_f.iloc[-1]
    total_v = sum(last_p[t] * shares_dict[t] for t in final_list)
    paths = np.zeros((n_days, n_sims, len(final_list)))
    tmp_p = np.tile(last_p.values, (n_sims, 1))
    dec, ewma = 0.94, (rets**2).ewm(alpha=0.06, adjust=False).mean()
    vols = np.tile(np.sqrt(ewma.iloc[-1].values), (n_sims, 1))
    for t in range(n_days):
        d_ret = np.random.normal(0, 1, (n_sims, len(final_list))) * vols
        tmp_p *= np.exp(d_ret); paths[t] = tmp_p
        vols = np.sqrt(dec * (vols**2) + (1-dec) * (d_ret**2))
    p_paths = np.sum(paths * [shares_dict[t] for t in final_list], axis=2)
    pnl = p_paths[-1, :] - total_v
    st.columns(3)[1].metric("Issue Médiane", f"{np.median(pnl):,.2f} €", f"{(np.median(pnl)/total_v)*100:.2f} %")
    fig = plt.figure(figsize=(16, 7), facecolor='none'); gs = GridSpec(1, 2, width_ratios=[1.8, 1])
    ax1 = fig.add_subplot(gs[0], facecolor='none'); nrm = plt.Normalize(pnl.min(), pnl.max())
    for i in np.random.choice(n_sims, 100): ax1.plot(p_paths[:, i], color=plt.cm.RdYlGn(nrm(pnl[i])), alpha=0.3)
    ax2 = fig.add_subplot(gs[1], facecolor='none'); _, b, ptch = ax2.hist(pnl, 50, density=True, alpha=0.8)
    for bi, pi in zip(b, ptch): pi.set_facecolor('red' if bi < 0 else 'green')
    st.pyplot(fig, transparent=True)

# --- MODE OPTIMISATION ---
elif app_mode == "Optimisation & Frontière Efficiente" and run_btn:
    d_opt = raw_data[final_list][raw_data.index >= pd.Timestamp(start_date)].pct_change().dropna()
    m_ret, cov = d_opt.mean()*252, d_opt.cov()*252
    c_w = np.array([shares_dict[t]*raw_data[t].iloc[-1] for t in final_list]); c_w /= np.sum(c_w)
    c_r, c_v = np.sum(m_ret*c_w), np.sqrt(np.dot(c_w.T, np.dot(cov, c_w)))
    res = np.zeros((3, n_portfolios)); w_r = []
    for i in range(n_portfolios):
        w = np.random.random(len(final_list)); w /= np.sum(w); w_r.append(w)
        r, v = np.sum(m_ret*w), np.sqrt(np.dot(w.T, np.dot(cov, w)))
        res[0,i], res[1,i], res[2,i] = r, v, (r-rf_rate)/v
    b_idx = np.argmax(res[2])
    c1, c2 = st.columns([2, 1])
    with c1:
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='none'); ax.set_facecolor('none')
        ax.scatter(res[1,:], res[0,:], c=res[2,:], cmap='viridis', s=10, alpha=0.3)
        ax.scatter(res[1,b_idx], res[0,b_idx], marker='*', color='r', s=200, label='Optimal')
        ax.scatter(c_v, c_r, marker='D', color='white', s=150, edgecolors='black', label='Actuel')
        ax.legend(); st.pyplot(fig, transparent=True)
    with c2: st.table(pd.DataFrame({'Actuel %': [round(x*100, 1) for x in c_w], 'Optimal %': [round(x*100, 1) for x in w_r[b_idx]]}, index=final_list))
