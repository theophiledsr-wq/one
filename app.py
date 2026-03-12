import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from dateutil.relativedelta import relativedelta

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
st.title("PORTFOLIO MASTER PRO PREMIUM : L'outil de pilotage de portefeuille")

def get_full_ticker_info(symbol):
    try:
        search = yf.Search(symbol, max_results=1)
        if search.quotes: return search.quotes[0].get('longname') or search.quotes[0].get('shortname')
        return yf.Ticker(symbol).info.get('longName') or symbol
    except: return symbol

# --- SIDEBAR ---
with st.sidebar:
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
    
    st.subheader("⚙️ Paramètres")
    start_date = st.date_input("Historique de référence :", datetime.date(2021, 1, 1))
    horizon = st.number_input("Horizon de projection (jours)", value=252)
    rf_rate = st.number_input("Taux sans risque %", value=3.0) / 100
    n_portfolios = st.number_input("Simulations Frontière", value=5000, min_value=1000, step=1000)
    run_btn = st.button("🚀 LANCER L'ANALYSE GLOBALE")

# --- CHARGEMENT DES DONNÉES ---
@st.cache_data
def load_data_all(tickers):
    all_t = list(set(tickers + ["^GSPC"]))
    return yf.download(all_t, start="2015-01-01", progress=False)['Close'].ffill().dropna()

raw_data = load_data_all(final_list)

# --- FONCTIONS KPI ---
def calc_all_kpis(port_rets, bench_rets, rf_rate):
    ann_ret = port_rets.mean() * 252
    ann_vol = port_rets.std() * np.sqrt(252)
    sharpe = (ann_ret - rf_rate) / ann_vol if ann_vol > 0 else 0
    
    downside_rets = port_rets[port_rets < 0]
    down_vol = downside_rets.std() * np.sqrt(252)
    sortino = (ann_ret - rf_rate) / down_vol if down_vol > 0 else 0
    
    cum_rets = (1 + port_rets).cumprod()
    running_max = np.maximum.accumulate(cum_rets)
    dd = (cum_rets - running_max) / running_max
    max_dd = abs(dd.min())
    calmar = ann_ret / max_dd if max_dd > 0 else 0
    ulcer_index = np.sqrt(np.mean(dd**2)) * 100
    
    cov = np.cov(port_rets, bench_rets)[0, 1]
    var_bench = np.var(bench_rets)
    beta = cov / var_bench if var_bench > 0 else 1
    alpha = ann_ret - (rf_rate + beta * (bench_rets.mean() * 252 - rf_rate))
    
    return sharpe, sortino, calmar, ulcer_index, alpha, beta

# --- FONCTION GRAPHIQUE CAMEMBERT ---
def plot_pie_chart(weights, labels, title):
    fig, ax = plt.subplots(figsize=(3, 3), facecolor='none')
    ax.set_facecolor('none')
    # Filtrer les poids trop petits pour la lisibilité
    mask = weights > 0.01
    w_filtered = weights[mask]
    l_filtered = np.array(labels)[mask]
    
    ax.pie(w_filtered, labels=l_filtered, autopct='%1.1f%%', textprops={'color': "white", 'fontsize': 8}, 
           colors=plt.cm.Set3.colors[:len(w_filtered)])
    ax.set_title(title, color='white', fontsize=10, pad=10)
    return fig

if run_btn:
    # --- PRÉPARATION DES DONNÉES ---
    df = raw_data[raw_data.index >= pd.Timestamp(start_date)]
    df_port = df[final_list]
    df_sp = df["^GSPC"]
    
    last_prices = df_port.iloc[-1]
    total_val_init = sum(last_prices[t] * shares_dict[t] for t in final_list)

    port_hist_val = (df_port * [shares_dict[t] for t in final_list]).sum(axis=1)
    sp_hist_val = (df_sp / df_sp.iloc[0]) * port_hist_val.iloc[0]

    # --- SECTION : DONNÉES HISTORIQUES ---
    st.header("Analyse Historique & Performance")
    col_graph, col_controls = st.columns([3, 1])
    
    with col_controls:
        st.subheader("Période d'analyse")
        period_choice = st.radio("Sélectionnez l'horizon :", ["1 Mois", "3 Mois", "6 Mois", "1 An", "Depuis l'origine"], index=4)
        
        end_d = port_hist_val.index[-1]
        if period_choice == "1 Mois": start_d = end_d - relativedelta(months=1)
        elif period_choice == "3 Mois": start_d = end_d - relativedelta(months=3)
        elif period_choice == "6 Mois": start_d = end_d - relativedelta(months=6)
        elif period_choice == "1 An": start_d = end_d - relativedelta(years=1)
        else: start_d = port_hist_val.index[0]
        
        mask = (port_hist_val.index >= start_d)
        port_hist_filtered = port_hist_val[mask]
        sp_hist_filtered = (df_sp[mask] / df_sp[mask].iloc[0]) * port_hist_filtered.iloc[0]

        rets_p = port_hist_filtered.pct_change().dropna()
        rets_sp = sp_hist_filtered.pct_change().dropna()
        
        p_sharpe, p_sortino, p_calmar, p_ulcer, p_alpha, p_beta = calc_all_kpis(rets_p, rets_sp, rf_rate)

    with col_graph:
        fig_hist, ax_hist = plt.subplots(figsize=(10, 4), facecolor='none')
        ax_hist.set_facecolor('none')
        ax_hist.plot(port_hist_filtered.index, port_hist_filtered, color='#00ff00', label='Portefeuille')
        ax_hist.plot(sp_hist_filtered.index, sp_hist_filtered, color='orange', ls='--', label='S&P 500 (Base)')
        ax_hist.set_ylabel("Valeur (€)", color='white')
        ax_hist.legend(frameon=False, labelcolor='white')
        ax_hist.grid(alpha=0.2)
        ax_hist.tick_params(colors='white')
        st.pyplot(fig_hist, transparent=True)

    st.subheader("Comparatif des Risques & Surperformance")
    kpi_df = pd.DataFrame({
        "Alpha (Surperf.)": [f"{p_alpha*100:+.2f}%", "0.00%"],
        "Beta (Volatilité Rel.)": [f"{p_beta:.2f}", "1.00"],
        "Sharpe Ratio": [f"{p_sharpe:.2f}", f"{(rets_sp.mean()*252 - rf_rate)/(rets_sp.std()*np.sqrt(252)):.2f}"],
        "Sortino Ratio": [f"{p_sortino:.2f}", f"{(rets_sp.mean()*252 - rf_rate)/(rets_sp[rets_sp<0].std()*np.sqrt(252)):.2f}"],
        "Calmar Ratio": [f"{p_calmar:.2f}", "-"],
        "Ulcer Index": [f"{p_ulcer:.2f}%", "-"]
    }, index=["Portefeuille", "S&P 500"])
    st.table(kpi_df)

    st.divider()

    # --- SECTION : MONTE CARLO ---
    st.header("Projection Monte Carlo & Robustesse")
    
    n_sims_mc = 5000
    log_rets_port = np.log(df_port / df_port.shift(1)).dropna()
    vols_port = log_rets_port.std().values
    
    price_paths = np.zeros((horizon, n_sims_mc, len(final_list)))
    temp_prices = np.tile(last_prices.values, (n_sims_mc, 1))
    for t in range(horizon):
        temp_prices *= np.exp(np.random.normal(0, 1, (n_sims_mc, len(final_list))) * vols_port)
        price_paths[t] = temp_prices
        
    portfolio_paths = np.sum(price_paths * [shares_dict[t] for t in final_list], axis=2)
    final_vals = portfolio_paths[-1, :]
    
    sp_paths = np.zeros((horizon, n_sims_mc))
    sp_temp = np.full(n_sims_mc, total_val_init)
    vol_sp = np.log(df_sp / df_sp.shift(1)).std()
    for t in range(horizon):
        sp_temp *= np.exp(np.random.normal(0, 1, n_sims_mc) * vol_sp)
        sp_paths[t] = sp_temp
        
    p50 = portfolio_paths[:, np.argsort(final_vals)[int(n_sims_mc*0.5)]]
    p5 = portfolio_paths[:, np.argsort(final_vals)[int(n_sims_mc*0.05)]]
    p95 = portfolio_paths[:, np.argsort(final_vals)[int(n_sims_mc*0.95)]]
    sp_p50 = sp_paths[:, np.argsort(sp_paths[-1, :])[int(n_sims_mc*0.5)]]
    
    proba_gain = (final_vals > total_val_init).mean() * 100
    proba_loss_10 = (final_vals < total_val_init * 0.9).mean() * 100
    var_95 = total_val_init - np.percentile(final_vals, 5)
    cvar_95 = total_val_init - np.mean(final_vals[final_vals <= np.percentile(final_vals, 5)])
    
    col1, col2 = st.columns([2, 1])
    with col1:
        fig1, ax1 = plt.subplots(figsize=(10, 5), facecolor='none')
        ax1.set_facecolor('none')
        ax1.plot(p95, color='#00ff00', label='Optimiste (95%)', alpha=0.6)
        ax1.plot(p50, color='white', lw=3, label='Médian Portefeuille')
        ax1.plot(sp_p50, color='orange', lw=2, ls='--', label='Médian S&P 500')
        ax1.plot(p5, color='#ff4b4b', label='Pessimiste (5%)', alpha=0.6)
        ax1.fill_between(range(horizon), p5, p95, color='gray', alpha=0.1)
        ax1.legend(frameon=False, labelcolor='white')
        ax1.tick_params(colors='white')
        st.pyplot(fig1, transparent=True)
    
    with col2:
        st.metric("Valeur Médiane Attendue", f"{p50[-1]:,.0f} €")
        st.metric("Probabilité de Plus-Value", f"{proba_gain:.1f} %")
        st.metric("Proba. de perte > 10%", f"{proba_loss_10:.1f} %", help="Risque de perdre plus de 10% du capital.")
        st.metric("Value at Risk (95%)", f"- {var_95:,.0f} €", help="Perte au centile 5%.")
        st.metric("CVaR (Expected Shortfall 95%)", f"- {cvar_95:,.0f} €", help="Moyenne des pertes dans les 5% des pires scénarios. Très robuste.")

    st.divider()

    # --- SECTION : FRONTIÈRE EFFICIENTE VECTORISÉE ---
    st.header(f"Optimisation de la Frontière Efficiente ({n_portfolios} itérations)")
    
    rets_daily_assets = df_port.pct_change().dropna()
    
    # Génération de poids aléatoires uniformes (Dirichlet)
    np.random.seed(42)
    w_matrix = np.random.dirichlet(np.ones(len(final_list)), n_portfolios).T # Shape: (N_assets, N_portfolios)
    
    # Calcul matriciel des rendements quotidiens des N portefeuilles
    port_rets_matrix = rets_daily_assets.values @ w_matrix # Shape: (T_days, N_portfolios)
    
    # 1. Rendements et Volatilités annualisés
    ann_rets_arr = np.mean(port_rets_matrix, axis=0) * 252
    ann_vols_arr = np.std(port_rets_matrix, axis=0) * np.sqrt(252)
    sharpes_arr = (ann_rets_arr - rf_rate) / ann_vols_arr
    
    # 2. Sortino
    downside_rets = np.minimum(port_rets_matrix, 0)
    down_vols_arr = np.std(downside_rets, axis=0) * np.sqrt(252)
    sortinos_arr = np.divide((ann_rets_arr - rf_rate), down_vols_arr, out=np.zeros_like(ann_rets_arr), where=down_vols_arr!=0)
    
    # 3. Drawdowns, Calmar, Ulcer, CAGR
    cum_rets_matrix = np.cumprod(1 + port_rets_matrix, axis=0)
    running_max_matrix = np.maximum.accumulate(cum_rets_matrix, axis=0)
    dds_matrix = (cum_rets_matrix - running_max_matrix) / running_max_matrix
    max_dds_arr = np.abs(np.min(dds_matrix, axis=0))
    
    calmars_arr = np.divide(ann_rets_arr, max_dds_arr, out=np.zeros_like(ann_rets_arr), where=max_dds_arr!=0)
    ulcers_arr = np.sqrt(np.mean(dds_matrix**2, axis=0)) * 100
    cagrs_arr = (cum_rets_matrix[-1, :] ** (252 / len(port_rets_matrix))) - 1
    
    # --- Identification des meilleurs profils ---
    idx_sharpe = np.argmax(sharpes_arr)
    idx_sortino = np.argmax(sortinos_arr)
    idx_calmar = np.argmax(calmars_arr)
    idx_cagr = np.argmax(cagrs_arr)
    idx_ulcer = np.argmin(ulcers_arr)
    
    weights_curr = np.array([shares_dict[t] * last_prices[t] for t in final_list])
    weights_curr /= np.sum(weights_curr)
    curr_ret = np.sum(rets_daily_assets.mean() * 252 * weights_curr)
    curr_vol = np.sqrt(np.dot(weights_curr.T, np.dot(rets_daily_assets.cov() * 252, weights_curr)))

    # Graphique Frontière
    fig2, ax2 = plt.subplots(figsize=(10, 4), facecolor='none')
    ax2.set_facecolor('none')
    scatter = ax2.scatter(ann_vols_arr, ann_rets_arr, c=sharpes_arr, cmap='viridis', s=5, alpha=0.3)
    ax2.scatter(curr_vol, curr_ret, marker='D', color='white', s=100, label='Actuel', edgecolors='black')
    ax2.scatter(ann_vols_arr[idx_sharpe], ann_rets_arr[idx_sharpe], marker='*', color='red', s=150, label='Max Sharpe')
    ax2.scatter(ann_vols_arr[idx_sortino], ann_rets_arr[idx_sortino], marker='^', color='orange', s=100, label='Max Sortino')
    ax2.scatter(ann_vols_arr[idx_cagr], ann_rets_arr[idx_cagr], marker='P', color='cyan', s=100, label='Max CAGR')
    ax2.scatter(ann_vols_arr[idx_ulcer], ann_rets_arr[idx_ulcer], marker='v', color='magenta', s=100, label='Min Ulcer (Sécurité)')
    ax2.set_xlabel("Volatilité (Risque)", color='white')
    ax2.set_ylabel("Rendement Attendu", color='white')
    ax2.tick_params(colors='white')
    ax2.legend(frameon=False, labelcolor='white', bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig2, transparent=True)

    # --- Répartition Camemberts ---
    st.subheader("Répartitions Optimales Suggérées")
    c_pie1, c_pie2, c_pie3, c_pie4, c_pie5 = st.columns(5)
    
    with c_pie1: st.pyplot(plot_pie_chart(weights_curr, final_list, "Actuel"), transparent=True)
    with c_pie2: st.pyplot(plot_pie_chart(w_matrix[:, idx_sharpe], final_list, "Max Sharpe\n(Rendement/Risque)"), transparent=True)
    with c_pie3: st.pyplot(plot_pie_chart(w_matrix[:, idx_sortino], final_list, "Max Sortino\n(Risque Baisse)"), transparent=True)
    with c_pie4: st.pyplot(plot_pie_chart(w_matrix[:, idx_cagr], final_list, "Max CAGR\n(Croissance)"), transparent=True)
    with c_pie5: st.pyplot(plot_pie_chart(w_matrix[:, idx_ulcer], final_list, "Min Ulcer\n(Sommeil tranquille)"), transparent=True)

    # --- Tableau des Valeurs et Parts ---
    st.subheader("Action : Réallocation en Euros et Nombre de Parts")
    
    selected_opt = st.selectbox("Choisissez le profil à appliquer :", 
                                ["Max Sharpe", "Max Sortino", "Max CAGR", "Min Ulcer", "Max Calmar"])
    
    opt_map = {"Max Sharpe": idx_sharpe, "Max Sortino": idx_sortino, "Max CAGR": idx_cagr, 
               "Min Ulcer": idx_ulcer, "Max Calmar": idx_calmar}
    
    chosen_w = w_matrix[:, opt_map[selected_opt]]
    target_value_eur = chosen_w * total_val_init
    target_shares = np.round(target_value_eur / last_prices.values)
    
    comparison = pd.DataFrame({
        "Val Actuelle (€)": (weights_curr * total_val_init).round(0),
        "Parts Actuelles": [shares_dict[t] for t in final_list],
        "Cible Optimale (€)": target_value_eur.round(0),
        "Nouvelles Parts": target_shares.astype(int),
        "Différence (€)": (target_value_eur - (weights_curr * total_val_init)).round(0)
    }, index=final_list)
    
    st.table(comparison)
