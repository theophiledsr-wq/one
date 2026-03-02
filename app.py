import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec
import datetime
from scipy.linalg import cholesky

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="European Portfolio Master Pro", layout="wide")
st.title("THE FRENCH BUILT TOOL FOR STRATEGIC INVESTING")

# --- RÉCUPÉRATION DES TICKERS ---
@st.cache_data
def get_european_base_list():
    try:
        indices = {"CAC 40": "https://en.wikipedia.org/wiki/CAC_40", 
                   "DAX 40": "https://en.wikipedia.org/wiki/DAX", 
                   "IBEX 35": "https://en.wikipedia.org/wiki/IBEX_35"}
        tickers = []
        for url in indices.values():
            tables = pd.read_html(url)
            for t in tables:
                if 'Ticker' in t.columns:
                    suffix = ".PA" if "CAC" in url else ".DE" if "DAX" in url else ".MC"
                    tickers.extend([str(tk).split('.')[0] + suffix for tk in t['Ticker'].tolist()])
                    break
        return sorted(list(set(tickers))) if tickers else ["AIR.PA", "MC.PA", "OR.PA"]
    except:
        return ["AIR.PA", "MC.PA", "OR.PA", "SAP.DE", "ASML.AS"]

BASE_TICKERS = get_european_base_list()

# --- BARRE LATÉRALE ---
with st.sidebar:
    st.header("🧭 Navigation")
    app_mode = st.radio("Choisir l'outil :", ["Projection Monte Carlo", "Optimisation & Frontière Efficiente"])
    
    st.divider()
    st.header("🛒 Portefeuille")
    selected_tickers = st.multiselect("Actifs :", options=BASE_TICKERS, default=["AIR.PA", "MC.PA"])
    
    manual_t = st.text_input("Ajout manuel (ex: PUST.PA) :").upper()
    if 'manual_list' not in st.session_state: st.session_state.manual_list = []
    if st.button("➕ Ajouter"):
        if manual_t and manual_t not in st.session_state.manual_list:
            st.session_state.manual_list.append(manual_t)
            st.rerun()

    final_list = list(set(selected_tickers + st.session_state.manual_list))
    shares_dict = {t: st.number_input(f"Quantité {t}", value=10, min_value=1) for t in final_list}

    st.divider()
    if app_mode == "Projection Monte Carlo":
        model_type = st.radio("Modèle :", ["FHS (Historique)", "Student-t"])
        nu_val = st.slider("nu (v)", 3, 50, 5) if model_type == "Student-t" else 5
        n_days = st.number_input("Horizon (jours)", value=150)
        n_sims = st.number_input("Simulations", value=2000)
        run_btn = st.button("🚀 LANCER LA SIMULATION")
    else:
        start_opt = st.date_input("Analyse depuis le", datetime.date(2021, 1, 1))
        rf_rate = st.number_input("Taux sans risque (%)", value=3.0) / 100
        run_btn = st.button("🎯 GÉNÉRER LA FRONTIÈRE")

# --- LOGIQUE DE DONNÉES ---
if final_list:
    tickers_to_download = final_list + ["^GSPC"]
    raw_data = yf.download(tickers_to_download, start="2019-01-01")['Close']
    data = raw_data[final_list].ffill().dropna()
    sp500 = raw_data["^GSPC"].ffill().dropna()
    
    last_prices = data.iloc[-1]
    total_val = sum(last_prices[t] * shares_dict[t] for t in final_list)
    current_weights = np.array([(last_prices[t] * shares_dict[t]) / total_val for t in final_list])

# --- MODE 1 : MONTE CARLO ---
if app_mode == "Projection Monte Carlo" and 'run_btn' in locals() and run_btn:
    st.info(f"Période d'analyse : {data.index[0].date()} au {data.index[-1].date()}")
    
    returns = np.log(data / data.shift(1)).dropna()
    decay = 0.94
    ewma_var = (returns**2).ewm(alpha=(1 - decay), adjust=False).mean()
    curr_vol = np.sqrt(ewma_var.iloc[-1].values)
    
    price_paths = np.zeros((n_days, n_sims, len(final_list)))
    temp_prices = np.tile(last_prices.values, (n_sims, 1))
    sim_vols = np.tile(curr_vol, (n_sims, 1))
    
    if model_type == "Student-t": L = cholesky(returns.corr().values, lower=True)

    for t in range(n_days):
        if model_type == "FHS (Historique)":
            shocks = (returns.sample(n_sims, replace=True).values / np.sqrt(ewma_var.sample(n_sims).values))
        else:
            t_samples = np.random.standard_t(df=nu_val, size=(n_sims, len(final_list)))
            shocks = (t_samples @ L.T) * np.sqrt((nu_val - 2) / nu_val)
        
        daily_ret = shocks * sim_vols
        temp_prices *= np.exp(daily_ret)
        price_paths[t] = temp_prices
        sim_vols = np.sqrt(decay * (sim_vols**2) + (1 - decay) * (daily_ret**2))

    portfolio_paths = np.sum(price_paths * [shares_dict[t] for t in final_list], axis=2)
    final_pnl = portfolio_paths[-1, :] - total_val

    # GRAPHIQUE DOUBLE : TRAJECTOIRES + DISTRIBUTION
    plt.rcParams.update({"text.color": "white", "axes.labelcolor": "white", "xtick.color": "white", "ytick.color": "white"})
    fig = plt.figure(figsize=(16, 7), facecolor='none')
    gs = GridSpec(1, 2, width_ratios=[1.8, 1])

    ax1 = fig.add_subplot(gs[0], facecolor='none')
    norm = plt.Normalize(final_pnl.min(), final_pnl.max())
    cmap = plt.cm.RdYlGn
    for i in np.random.choice(n_sims, 100):
        ax1.plot(portfolio_paths[:, i], color=cmap(norm(final_pnl[i])), alpha=0.3)
    ax1.set_title(f"Trajectoires {model_type}")

    ax2 = fig.add_subplot(gs[1], facecolor='none')
    n, bins, patches = ax2.hist(final_pnl, bins=50, density=True, alpha=0.8)
    for b, p in zip(bins, patches):
        p.set_facecolor('red' if b < 0 else 'green')
    ax2.set_title("Distribution des P&L (Rouge = Perte / Vert = Gain)")
    ax2.axvline(0, color='white', lw=1, ls='--')
    
    st.pyplot(fig, transparent=True)

# --- MODE 2 : OPTIMISATION ---
elif app_mode == "Optimisation & Frontière Efficiente" and 'run_btn' in locals() and run_btn:
    st.info(f"Analyse calculée du {start_opt} au {datetime.date.today()}")
    
    ret_opt = data[data.index >= pd.Timestamp(start_opt)].pct_change().dropna()
    mean_ret = ret_opt.mean() * 252
    cov_mat = ret_opt.cov() * 252
    
    # S&P 500 Benchmark
    sp_ret_all = sp500[sp500.index >= pd.Timestamp(start_opt)].pct_change().dropna()
    sp_stats = [sp_ret_all.std() * np.sqrt(252), sp_ret_all.mean() * 252]

    # Simulation Portefeuilles
    n_p = 5000
    p_rets, p_vols, p_sha, p_sor, p_cal, p_weights = [], [], [], [], [], []
    for _ in range(n_p):
        w = np.random.random(len(final_list)); w /= np.sum(w)
        r = np.sum(mean_ret * w)
        v = np.sqrt(w.T @ cov_mat @ w)
        # Sortino
        p_ts = (ret_opt * w).sum(axis=1)
        downside = p_ts[p_ts < 0].std() * np.sqrt(252)
        # Calmar
        cum = (1 + p_ts).cumprod()
        mdd = abs(((cum / cum.expanding().max()) - 1).min())
        
        p_rets.append(r); p_vols.append(v)
        p_sha.append((r - rf_rate) / v)
        p_sor.append((r - rf_rate) / downside if downside != 0 else 0)
        p_cal.append(r / mdd if mdd != 0 else 0)
        p_weights.append(w)

    # Extraction des "Best"
    best_idx = {"Sharpe": np.argmax(p_sha), "Sortino": np.argmax(p_sor), "Calmar": np.argmax(p_cal)}
    
    # GRAPHE FRONTIÈRE
    
    fig_ef, ax_ef = plt.subplots(figsize=(12, 7), facecolor='none')
    plt.rcParams.update({"text.color": "white", "axes.labelcolor": "white"})
    sc = ax_ef.scatter(p_vols, p_rets, c=p_sha, cmap='RdYlGn', alpha=0.3)
    
    # 100% Actifs
    for i, t in enumerate(final_list):
        ax_ef.scatter(np.sqrt(cov_mat.iloc[i,i]), mean_ret[i], s=100, label=f"100% {t}", edgecolors='white')
    
    # Points Optimes
    markers = {"Sharpe": ("*", "gold", 300), "Sortino": ("v", "orange", 200), "Calmar": ("P", "magenta", 200)}
    for name, (m, c, s) in markers.items():
        idx = best_idx[name]
        ax_ef.scatter(p_vols[idx], p_rets[idx], color=c, marker=m, s=s, label=f"Max {name}", edgecolors='black')
    
    ax_ef.scatter(sp_stats[0], sp_stats[1], color='blue', marker='D', s=150, label='S&P 500')
    curr_v = np.sqrt(current_weights.T @ cov_mat @ current_weights)
    curr_r = np.sum(mean_ret * current_weights)
    ax_ef.scatter(curr_v, curr_r, color='white', marker='X', s=300, label='ACTUEL')

    ax_ef.set_xlabel("Risque (Volatilité)")
    ax_ef.set_ylabel("Rendement")
    ax_ef.legend(loc='upper left', bbox_to_anchor=(1, 1), facecolor='#262730')
    st.pyplot(fig_ef, transparent=True)

    # TABLEAU DES COMPOSITIONS PRÉCISES
    st.subheader("📋 Compositions des Stratégies Optimales")
    comp_data = {"Actif": final_list}
    for name, idx in best_idx.items():
        comp_data[f"Max {name} (%)"] = p_weights[idx] * 100
    
    st.table(pd.DataFrame(comp_data).style.format({c: "{:.2f}%" for c in comp_data if "%" in c}))
