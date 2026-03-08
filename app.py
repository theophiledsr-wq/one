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
        st.subheader("⚙️ Paramètres")
        start_date = st.date_input("Depuis (Historique de volatilité) :", datetime.date(2021, 1, 1))
        n_days = st.number_input("Horizon de projection (jours)", 252, min_value=1)
        st.caption("⚡ Nombre de simulations fixé à 5 000 (précision optimale).")
        n_sims = 5000  # FIXÉ À 5000 SELON LA DEMANDE
        run_btn = st.button("🚀 LANCER SIMULATION")
        
    else:
        st.subheader("⚙️ Paramètres")
        start_date = st.date_input("Depuis :", datetime.date(2020, 1, 1))
        rf_rate, n_portfolios = st.number_input("Taux sans risque %", 3.0)/100, st.number_input("Portefeuilles", 5000)
        run_btn = st.button("🎯 GÉNÉRER FRONTIÈRE")

@st.cache_data
def load_data_portfolio(tickers): 
    # Ajout du S&P 500 d'office pour pouvoir s'en servir de benchmark
    data = yf.download(tickers + ["^GSPC"], start="2015-01-01", progress=False)['Close']
    return data.ffill().dropna()

raw_data = load_data_portfolio(final_list)

# --- MODE DONNÉES HISTORIQUES ---
if app_mode == "Données historiques" and run_btn:
    p_map = {"1m":"1mo", "6m":"6mo", "1y":"1y", "3y":"3y", "5y":"5y", "10y":"10y", "all time":"max"}
    hist_df = yf.download(final_list + ["^GSPC"], period=p_map[selected_period], progress=False)['Close'].ffill().dropna()
    
    if not hist_df.empty:
        # 1. Analyse du Portefeuille
        portfolio_value_series = sum(hist_df[t] * shares_dict[t] for t in final_list)
        total_initial_invested = sum(hist_df[t].iloc[0] * shares_dict[t] for t in final_list)
        total_final_net_value = 0
        years = (hist_df.index[-1] - hist_df.index[0]).days / 365.25
        
        for t in final_list:
            p_start, p_end = hist_df[t].iloc[0], hist_df[t].iloc[-1]
            qty, f = shares_dict[t], st.session_state.asset_fees[t]
            val_init_net = (p_start * qty) * (1 - f['entry'])
            val_apres_gestion = val_init_net * (p_end / p_start) * ((1 - f['mgmt']) ** (years if years > 0 else 1))
            gain = val_apres_gestion - (p_start * qty)
            total_final_net_value += val_apres_gestion - (gain * f['perf']) if gain > 0 else val_apres_gestion

        perf_brute_p = (portfolio_value_series.iloc[-1] / portfolio_value_series.iloc[0]) - 1
        perf_nette_p = (total_final_net_value / total_initial_invested) - 1
        daily_ret = portfolio_value_series.pct_change().dropna()
        vol = daily_ret.std() * np.sqrt(252)
        sharpe = ((daily_ret.mean() * 252) - rf_hist) / vol if vol > 0 else 0
        max_dd = ((portfolio_value_series / portfolio_value_series.expanding().max()) - 1).min()

        # 2. Benchmark S&P 500
        sp500 = hist_df["^GSPC"]
        perf_brute_sp = (sp500.iloc[-1] / sp500.iloc[0]) - 1
        daily_ret_sp = sp500.pct_change().dropna()
        vol_sp = daily_ret_sp.std() * np.sqrt(252)
        sharpe_sp = ((daily_ret_sp.mean() * 252) - rf_hist) / vol_sp if vol_sp > 0 else 0
        dd_sp = ((sp500 / sp500.expanding().max()) - 1).min()

        st.subheader(f"📊 Analyse Comparative : Portefeuille vs Benchmark")
        c1, c2 = st.columns([3, 1])
        with c1:
            fig, ax = plt.subplots(figsize=(10, 5), facecolor='none'); ax.set_facecolor('none')
            ax.plot((portfolio_value_series / portfolio_value_series.iloc[0]) * 100, color='#00ff00', lw=2.5, label="Votre Portefeuille")
            ax.plot((sp500 / sp500.iloc[0]) * 100, color='#A020F0', lw=2, alpha=0.8, label="S&P 500 (Base 100)")
            plt.rcParams.update({"text.color": "white", "axes.labelcolor": "white", "xtick.color": "white", "ytick.color": "white"})
            ax.legend(facecolor='#0e1117', edgecolor='white')
            ax.grid(True, alpha=0.1, color='white'); st.pyplot(fig, transparent=True)
        
        with c2:
            st.markdown("### 🟢 Portefeuille")
            st.metric("Performance Brute", f"{perf_brute_p*100:.2f} %")
            st.metric("Performance Nette", f"{perf_nette_p*100:.2f} %")
            st.metric("Sharpe / MaxDD", f"{sharpe:.2f} / {max_dd*100:.1f}%")
            st.divider()
            st.markdown("<h3 style='color: #A020F0;'>🟣 S&P 500</h3>", unsafe_allow_html=True)
            st.markdown(f"<p style='color: #A020F0; margin-bottom:0;'>Perf. Brute: <b>{perf_brute_sp*100:.2f} %</b></p>", unsafe_allow_html=True)
            st.markdown(f"<p style='color: #A020F0; margin-bottom:0;'>Sharpe: <b>{sharpe_sp:.2f}</b></p>", unsafe_allow_html=True)
            st.markdown(f"<p style='color: #A020F0; margin-bottom:0;'>Max Drawdown: <b>{dd_sp*100:.2f} %</b></p>", unsafe_allow_html=True)

# --- MODE PROJECTION MONTE CARLO ---
elif app_mode == "Projection Monte Carlo" and run_btn:
    data_filtered = raw_data[final_list][raw_data.index >= pd.Timestamp(start_date)]
    returns = np.log(data_filtered / data_filtered.shift(1)).dropna()
    last_prices = data_filtered.iloc[-1]
    total_val_init = sum(last_prices[t] * shares_dict[t] for t in final_list)
    
    # 1. Simulation du Portefeuille (5000 trajectoires)
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
    
    # Extraction des fractiles pour le portefeuille
    indices_sorted = np.argsort(final_values)
    path_5pct = portfolio_paths[:, indices_sorted[int(n_sims * 0.05)]]
    path_median = portfolio_paths[:, indices_sorted[int(n_sims * 0.50)]]
    path_95pct = portfolio_paths[:, indices_sorted[int(n_sims * 0.95)]]

    # 2. Simulation du Benchmark (S&P 500) pour comparaison
    sp500_data = raw_data["^GSPC"][raw_data.index >= pd.Timestamp(start_date)].to_frame()
    sp_ret = np.log(sp500_data / sp500_data.shift(1)).dropna()
    sp_last = sp500_data.iloc[-1].values[0]
    
    sp_paths = np.zeros((n_days, n_sims))
    sp_temp = np.tile(sp_last, n_sims)
    sp_ewma = (sp_ret**2).ewm(alpha=0.06, adjust=False).mean()
    sp_vols = np.tile(np.sqrt(sp_ewma.iloc[-1].values), n_sims)
    
    for t in range(n_days):
        sp_daily_ret = np.random.normal(0, 1, n_sims) * sp_vols
        sp_temp *= np.exp(sp_daily_ret)
        sp_paths[t] = sp_temp
        sp_vols = np.sqrt(decay * (sp_vols**2) + (1-decay) * (sp_daily_ret**2))
        
    # Calibrage du S&P 500 sur l'investissement initial du portefeuille
    sp_scaled_paths = (sp_paths / sp_last) * total_val_init
    sp_final_values = sp_scaled_paths[-1, :]
    sp_median_idx = np.argsort(sp_final_values)[int(n_sims * 0.50)]
    sp_path_median = sp_scaled_paths[:, sp_median_idx]

    # 3. Calcul des KPIs (Portefeuille)
    var_95 = total_val_init - np.percentile(final_values, 5)
    cvar_95 = total_val_init - final_values[final_values <= np.percentile(final_values, 5)].mean()
    prob_gain = (final_values > total_val_init).sum() / n_sims * 100
    std_error = final_values.std() / np.sqrt(n_sims)
    
    # Préparation des données de tracé (pour démarrer à l'instant T=0 proprement)
    time_range = np.arange(n_days + 1)
    plot_95 = np.insert(path_95pct, 0, total_val_init)
    plot_50 = np.insert(path_median, 0, total_val_init)
    plot_05 = np.insert(path_5pct, 0, total_val_init)
    plot_sp = np.insert(sp_path_median, 0, total_val_init)

    st.subheader(f"🚀 Projection Monte Carlo ({n_sims} itérations)")
    
    c1, c2 = st.columns([2, 1])
    with c1:
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='none'); ax.set_facecolor('none')
        
        # Tracé des 3 chemins du portefeuille
        ax.plot(time_range, plot_95, color='#00ff00', lw=2, label='Portefeuille: Optimiste (95%)')
        ax.plot(time_range, plot_50, color='white', lw=3, label='Portefeuille: Médian (50%)')
        ax.plot(time_range, plot_05, color='#ff4b4b', lw=2, label='Portefeuille: Pessimiste (5%)')
        
        # Tracé du Benchmark (Médian)
        ax.plot(time_range, plot_sp, color='#A020F0', lw=2, linestyle='--', label='S&P 500: Médian Projeté')
        
        # Remplissage du corridor de confiance (5% à 95%)
        ax.fill_between(time_range, plot_05, plot_95, color='gray', alpha=0.15)
        
        plt.rcParams.update({"text.color": "white", "axes.labelcolor": "white", "xtick.color": "white", "ytick.color": "white"})
        ax.set_ylabel("Valeur (€)")
        ax.set_xlabel("Jours boursiers futurs")
        ax.legend(facecolor='#0e1117', edgecolor='white')
        ax.grid(True, alpha=0.1, color='white'); st.pyplot(fig, transparent=True)

    with c2:
        st.markdown("### 🛠️ Indicateurs de Risque")
        kpi_df = pd.DataFrame({
            "Indicateur": [
                "Investissement Initial", 
                "Valeur Médiane Attendue", 
                "Probabilité de Profit", 
                "Value-at-Risk (VaR 95%)", 
                "Expected Shortfall (CVaR)", 
                "Erreur Standard Simulation"
            ],
            "Valeur": [
                f"{total_val_init:,.2f} €", 
                f"{final_values[indices_sorted[int(n_sims * 0.50)]]:,.2f} €", 
                f"{prob_gain:.1f} %", 
                f"{var_95:,.2f} €", 
                f"{cvar_95:,.2f} €", 
                f"± {std_error:,.2f} €"
            ]
        })
        st.table(kpi_df.set_index("Indicateur"))
        
        # Comparaison rapide
        perf_mediane_portefeuille = (final_values[indices_sorted[int(n_sims * 0.50)]] / total_val_init) - 1
        perf_mediane_sp500 = (sp_final_values[sp_median_idx] / total_val_init) - 1
        
        st.info(f"**Comparaison Médiane Projetée:**\n\n🟢 Portefeuille: **{perf_mediane_portefeuille*100:+.2f}%**\n🟣 S&P 500: **{perf_mediane_sp500*100:+.2f}%**")

# --- MODE OPTIMISATION & FRONTIÈRE EFFICIENTE ---
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
    
    st.subheader("🎯 Optimisation de Portefeuille (Markowitz)")
    c1, c2 = st.columns([2, 1])
    with c1:
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='none'); ax.set_facecolor('none')
        sc = ax.scatter(res[1,:], res[0,:], c=res[2,:], cmap='viridis', s=10, alpha=0.3)
        ax.scatter(res[1,best_idx], res[0,best_idx], marker='*', color='r', s=200, label='Portefeuille Optimal (Max Sharpe)')
        ax.scatter(curr_vol, curr_ret, marker='D', color='white', s=150, edgecolors='black', label='Portefeuille Actuel')
        
        plt.colorbar(sc, label='Ratio de Sharpe')
        plt.rcParams.update({"text.color": "white", "axes.labelcolor": "white", "xtick.color": "white", "ytick.color": "white"})
        ax.set_xlabel("Volatilité (Risque)")
        ax.set_ylabel("Rendement Espéré")
        ax.legend(facecolor='#0e1117', edgecolor='white')
        st.pyplot(fig, transparent=True)
        
    with c2:
        st.markdown("### ⚖️ Allocations")
        alloc_df = pd.DataFrame({
            'Actuel %': [round(x * 100, 1) for x in curr_w], 
            'Optimal %': [round(x * 100, 1) for x in w_rec[best_idx]]
        }, index=final_list)
        st.table(alloc_df)
        
        st.divider()
        st.metric("Ratio de Sharpe Actuel", f"{(curr_ret - rf_rate) / curr_vol:.2f}")
        st.metric("Ratio de Sharpe Optimal", f"{res[2,best_idx]:.2f}")
