import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec
import datetime

# --- CONFIGURATION & STYLE ---
st.set_page_config(page_title="European Portfolio Master Pro", layout="wide")
st.markdown("<style>#MainMenu, footer, header, .stDeployButton {visibility: hidden;}</style>", unsafe_allow_html=True)

# --- FONCTIONS UTILITAIRES ---
def get_full_ticker_info(symbol):
    try:
        search = yf.Search(symbol, max_results=1)
        if search.quotes:
            return search.quotes[0].get('longname') or search.quotes[0].get('shortname') or symbol
        return yf.Ticker(symbol).info.get('longName') or symbol
    except: return symbol

@st.cache_data
def load_data_portfolio(tickers):
    df = yf.download(tickers, start="2018-01-01", progress=False)['Close']
    return df.ffill().dropna()

# --- SIDEBAR ---
with st.sidebar:
    st.header("🧭 Navigation")
    app_mode = st.radio("Choisir l'outil :", ["Projection Monte Carlo", "Optimisation & Frontière Efficiente"])
    st.divider()
    
    if 'portfolio' not in st.session_state: st.session_state.portfolio = {}
    search_input = st.text_input("Ticker (ex: AAPL, MC.PA) :").upper()
    
    if st.button("➕ Ajouter"):
        if search_input:
            full_name = get_full_ticker_info(search_input)
            st.session_state.portfolio[search_input] = full_name
            st.rerun()

    if st.session_state.portfolio:
        to_delete = [t for t, name in st.session_state.portfolio.items() if st.button(f"🗑️ {t}", key=f"del_{t}")]
        for t in to_delete: 
            del st.session_state.portfolio[t]
            st.rerun()
    
    final_list = list(st.session_state.portfolio.keys())
    if not final_list: st.stop()

    st.divider()
    shares_dict = {t: st.number_input(f"Qte {t}", value=10, min_value=1) for t in final_list}
    
    if app_mode == "Projection Monte Carlo":
        n_days = st.number_input("Horizon (jours)", value=252)
        n_sims = st.number_input("Simulations", value=5000)
    else:
        rf_rate = st.number_input("Taux sans risque (%)", value=3.0) / 100
        n_portfolios = st.number_input("Nombre de simulations", value=5000)
    
    run_btn = st.button("🚀 LANCER L'ANALYSE")

raw_data = load_data_portfolio(final_list)

# --- MODE MONTE CARLO ---
if app_mode == "Projection Monte Carlo" and run_btn:
    returns = np.log(raw_data / raw_data.shift(1)).dropna()
    last_prices = raw_data.iloc[-1]
    total_val = sum(last_prices[t] * shares_dict[t] for t in final_list)
    
    # Simulation (Geometric Brownian Motion + EWMA Vol)
    price_paths = np.zeros((n_days, n_sims, len(final_list)))
    temp_prices = np.tile(last_prices.values, (n_sims, 1))
    vols = returns.std().values 

    for t in range(n_days):
        shocks = np.random.normal(0, 1, size=(n_sims, len(final_list)))
        temp_prices *= np.exp((returns.mean().values - 0.5 * vols**2) + shocks * vols)
        price_paths[t] = temp_prices

    portfolio_paths = np.sum(price_paths * [shares_dict[t] for t in final_list], axis=2)
    final_pnl = portfolio_paths[-1, :] - total_val
    
    # Statistiques Clés
    var_95 = np.percentile(final_pnl, 5) # Value at Risk 95%
    p_5 = np.percentile(final_pnl, 5)
    p_95 = np.percentile(final_pnl, 95)
    median_res = np.median(final_pnl)

    c1, c2, c3 = st.columns(3)
    c1.metric("Issue Médiane", f"{median_res:,.2f} €")
    c2.metric("VaR (95%)", f"{var_95:,.2f} €", delta_color="inverse")
    c3.metric("Intervalle (5% - 95%)", f"{p_95-p_5:,.0f} € span")

    fig = plt.figure(figsize=(16, 8), facecolor='none')
    gs = GridSpec(1, 2, width_ratios=[1.8, 1])
    plt.rcParams.update({"text.color": "white", "axes.labelcolor": "white", "xtick.color": "white", "ytick.color": "white"})
    
    ax1 = fig.add_subplot(gs[0], facecolor='none')
    ax1.plot(portfolio_paths[:, :100], alpha=0.2, color='gray')
    ax1.set_title("100 Scénarios de Projection")
    
    ax2 = fig.add_subplot(gs[1], facecolor='none')
    ax2.hist(final_pnl, bins=50, alpha=0.6, color='skyblue')
    ax2.axvline(var_95, color='red', linestyle='--', label=f'VaR 95%: {var_95:.0f}€')
    ax2.axvline(p_95, color='lime', linestyle='--', label=f'Top 5%: {p_95:.0f}€')
    ax2.legend()
    st.pyplot(fig, transparent=True)

# --- MODE OPTIMISATION ---
elif app_mode == "Optimisation & Frontière Efficiente" and run_btn:
    rets_d = raw_data.pct_change().dropna()
    mean_rets = rets_d.mean() * 252
    cov_mat = rets_d.cov() * 252

    # Portefeuille Actuel
    vals = np.array([shares_dict[t] * raw_data.iloc[-1][t] for t in final_list])
    curr_w = vals / np.sum(vals)
    curr_ret = np.sum(mean_rets * curr_w)
    curr_vol = np.sqrt(curr_w.T @ cov_m @ curr_w) if 'cov_m' in locals() else np.sqrt(np.dot(curr_w.T, np.dot(cov_mat, curr_w)))

    # Simulation de portefeuilles
    results = []
    for _ in range(n_portfolios):
        w = np.random.random(len(final_list)); w /= np.sum(w)
        r = np.sum(mean_rets * w)
        v = np.sqrt(w.T @ cov_mat @ w)
        
        # Sortino (Downside risk)
        downside_rets = rets_d[rets_d < 0].fillna(0)
        down_vol = np.sqrt(np.mean(downside_rets**2)) * np.sqrt(252)
        sortino = (r - rf_rate) / np.dot(w, down_vol)
        
        # Calmar (Max Drawdown - Approximation simplifiée sur hist)
        cum_rets = (1 + rets_d @ w).cumprod()
        peak = cum_rets.expanding(min_periods=1).max()
        dd = (cum_rets/peak - 1).min()
        calmar = r / abs(dd) if dd != 0 else 0
        
        results.append([r, v, (r-rf_rate)/v, sortino, calmar, w])

    df_res = pd.DataFrame(results, columns=['Ret', 'Vol', 'Sharpe', 'Sortino', 'Calmar', 'Weights'])
    
    # Points Optimaux
    best_sharpe = df_res.iloc[df_res['Sharpe'].idxmax()]
    best_sortino = df_res.iloc[df_res['Sortino'].idxmax()]
    best_calmar = df_res.iloc[df_res['Calmar'].idxmax()]

    fig_opt, ax_opt = plt.subplots(figsize=(12, 7), facecolor='none')
    ax_opt.set_facecolor('none')
    
    # Nuage de points
    plt.scatter(df_res.Vol, df_res.Ret, c=df_res.Sharpe, cmap='viridis', s=5, alpha=0.3)
    
    # Actifs Individuels
    for t in final_list:
        v_ind = np.sqrt(cov_mat.loc[t,t])
        r_ind = mean_rets[t]
        ax_opt.scatter(v_ind, r_ind, s=100, label=f'{t}', marker='o', edgecolors='white')

    # Portefeuilles Spéciaux
    ax_opt.scatter(best_sharpe.Vol, best_sharpe.Ret, color='red', marker='*', s=250, label='Max Sharpe')
    ax_opt.scatter(best_sortino.Vol, best_sortino.Ret, color='orange', marker='P', s=200, label='Max Sortino')
    ax_opt.scatter(best_calmar.Vol, best_calmar.Ret, color='cyan', marker='X', s=200, label='Max Calmar')
    ax_opt.scatter(curr_vol, curr_ret, color='white', marker='D', s=200, edgecolors='black', label='TON PORTFEUILLE')
    
    plt.colorbar(label='Ratio de Sharpe')
    ax_opt.legend(loc='best', fontsize='small')
    st.pyplot(fig_opt, transparent=True)

    # Tableau Recap
    st.write("### 🏆 Comparaison des Stratégies")
    comparison = pd.DataFrame({
        "Sharpe (Risque Total)": [best_sharpe.Ret, best_sharpe.Vol, best_sharpe.Sharpe],
        "Sortino (Risque de Baisse)": [best_sortino.Ret, best_sortino.Vol, best_sortino.Sortino],
        "Calmar (Max Drawdown)": [best_calmar.Ret, best_calmar.Vol, best_calmar.Calmar],
        "Ton Portefeuille": [curr_ret, curr_vol, (curr_ret-rf_rate)/curr_vol]
    }, index=["Rendement Annuel", "Volatilité Annuelle", "Ratio"])
    st.table(comparison.style.format("{:.2%}").highlight_max(axis=1, color="#1e4d2b"))
