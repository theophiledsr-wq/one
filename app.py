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
        fallback = ["AIR.PA", "MC.PA", "OR.PA", "RMS.PA", "SAP.DE", "ASML.AS", "SIE.DE"]
        indices = {"CAC 40": "https://en.wikipedia.org/wiki/CAC_40", 
                   "DAX 40": "https://en.wikipedia.org/wiki/DAX"}
        tickers = []
        for url in indices.values():
            tables = pd.read_html(url)
            for t in tables:
                if 'Ticker' in t.columns:
                    suffix = ".PA" if "CAC" in url else ".DE"
                    tickers.extend([str(tk).split('.')[0] + suffix for tk in t['Ticker'].tolist()])
                    break
        return sorted(list(set(tickers))) if tickers else fallback
    except:
        return ["AIR.PA", "MC.PA", "OR.PA", "SAP.DE", "ASML.AS"]

BASE_LIST = get_european_base_list()

# --- BARRE LATÉRALE ---
with st.sidebar:
    st.header("🧭 Navigation")
    app_mode = st.radio("Choisir l'outil :", ["Projection Monte Carlo", "Optimisation & Frontière Efficiente"])
    
    st.divider()
    st.header("🛒 Portefeuille")
    
    safe_default = [BASE_LIST[0]] if "AIR.PA" not in BASE_LIST else ["AIR.PA"]
    selected_tickers = st.multiselect("Sélectionner dans les indices :", options=BASE_LIST, default=safe_default)
    
    manual_t = st.text_input("Ajout manuel (ex: PUST.PA) :").upper()
    if 'manual_list' not in st.session_state: st.session_state.manual_list = []
    if st.button("➕ Ajouter"):
        if manual_t and manual_t not in st.session_state.manual_list:
            st.session_state.manual_list.append(manual_t)
            st.rerun()

    final_list = list(set(selected_tickers + st.session_state.manual_list))
    
    if not final_list:
        st.error("Veuillez sélectionner au moins un actif.")
        st.stop()

    shares_dict = {t: st.number_input(f"Quantité {t}", value=10, min_value=1) for t in final_list}

    st.divider()
    if app_mode == "Projection Monte Carlo":
        model_type = st.radio("Modèle :", ["FHS (Historique)", "Student-t"])
        nu_val = st.slider("nu (v)", 3, 50, 5, help="3=Risque extrême, 30+=Normal") if model_type == "Student-t" else 5
        n_days = st.number_input("Horizon (jours)", value=150)
        n_sims = st.number_input("Simulations", value=2000)
        run_btn = st.button("🚀 LANCER LA SIMULATION")
    else:
        start_opt = st.date_input("Analyse depuis le", datetime.date(2021, 1, 1))
        rf_rate = st.number_input("Taux sans risque (%)", value=3.0) / 100
        run_btn = st.button("🎯 GÉNÉRER LA FRONTIÈRE")

# --- CHARGEMENT DES DONNÉES ---
@st.cache_data
def load_data(tickers, start_date):
    download_list = tickers + ["^GSPC"]
    df = yf.download(download_list, start=start_date)['Close']
    if isinstance(df, pd.Series):
        df = df.to_frame()
    return df.ffill().dropna()

raw_data = load_data(final_list, "2019-01-01")

if raw_data.empty:
    st.error("Impossible de récupérer les données.")
    st.stop()

sp500 = raw_data["^GSPC"]
data = raw_data[final_list]
last_prices = data.iloc[-1]
total_val = sum(last_prices[t] * shares_dict[t] for t in final_list)
current_weights = np.array([(last_prices[t] * shares_dict[t]) / total_val for t in final_list])

# --- MODE 1 : MONTE CARLO ---
if app_mode == "Projection Monte Carlo" and 'run_btn' in locals() and run_btn:
    vol_date = data.index[-1].date()
    st.info(f"⚡ **Volatilité de départ (EWMA λ=0.94) fixée au : {vol_date}**")
    
    returns = np.log(data / data.shift(1)).dropna()
    decay = 0.94
    ewma_var = (returns**2).ewm(alpha=(1 - decay), adjust=False).mean()
    curr_vol = np.sqrt(ewma_var.iloc[-1].values)
    
    price_paths = np.zeros((n_days, n_sims, len(final_list)))
    temp_prices = np.tile(last_prices.values, (n_sims, 1))
    sim_vols = np.tile(curr_vol, (n_sims, 1))
    
    if model_type == "Student-t":
        corr = returns.corr().values
        L = cholesky(corr + np.eye(len(final_list)) * 1e-8, lower=True)

    # Pré-calcul des shocks standardisés pour FHS pour éviter l'erreur de sample
    if model_type == "FHS (Historique)":
        std_returns = (returns / np.sqrt(ewma_var.shift(1))).dropna()

    for t in range(n_days):
        if model_type == "FHS (Historique)":
            shocks = std_returns.sample(n_sims, replace=True).values
        else:
            t_samples = np.random.standard_t(df=nu_val, size=(n_sims, len(final_list)))
            shocks = (t_samples @ L.T) * np.sqrt((nu_val - 2) / nu_val)
        
        daily_ret = shocks * sim_vols
        temp_prices *= np.exp(daily_ret)
        price_paths[t] = temp_prices
        sim_vols = np.sqrt(decay * (sim_vols**2) + (1 - decay) * (daily_ret**2))

    portfolio_paths = np.sum(price_paths * [shares_dict[t] for t in final_list], axis=2)
    final_pnl = portfolio_paths[-1, :] - total_val
    
    # Affichage Médiane
    median_pnl = np.median(final_pnl)
    median_pct = (median_pnl / total_val) * 100
    st.columns(3)[1].metric(f"Issue Médiane ({model_type})", f"{median_pnl:,.2f} €", f"{median_pct:.2f} %")

    # Graphiques
    fig = plt.figure(figsize=(16, 7), facecolor='none')
    gs = GridSpec(1, 2, width_ratios=[1.8, 1])
    plt.rcParams.update({"text.color": "white", "axes.labelcolor": "white", "xtick.color": "white", "ytick.color": "white"})

    ax1 = fig.add_subplot(gs[0], facecolor='none')
    norm = plt.Normalize(final_pnl.min(), final_pnl.max())
    cmap = plt.cm.RdYlGn
    for i in np.random.choice(n_sims, 100):
        ax1.plot(portfolio_paths[:, i], color=cmap(norm(final_pnl[i])), alpha=0.3)
    ax1.set_title(f"Simulation {model_type}", fontweight='bold')

    ax2 = fig.add_subplot(gs[1], facecolor='none')
    n, bins, patches = ax2.hist(final_pnl, bins=50, density=True, alpha=0.8)
    for b, p in zip(bins, patches):
        p.set_facecolor('red' if b < 0 else 'green')
    ax2.axvline(0, color='white', lw=1, ls='--')
    st.pyplot(fig, transparent=True)

# --- MODE 2 : OPTIMISATION ---
elif app_mode == "Optimisation & Frontière Efficiente":
    st.subheader("📊 Composition Actuelle")
    c_pie, c_tab = st.columns([1, 1.5])
    with c_pie:
        fig_p, ax_p = plt.subplots(figsize=(5, 5), facecolor='none')
        ax_p.pie([last_prices[t]*shares_dict[t] for t in final_list], labels=final_list, autopct='%1.1f%%', textprops={'color':"w"})
        st.pyplot(fig_p, transparent=True)
    with c_tab:
        st.table(pd.DataFrame({"Actif": final_list, "Poids": [f"{w*100:.1f}%" for w in current_weights]}))

    if run_btn:
        ret_opt = data[data.index >= pd.Timestamp(start_opt)].pct_change().dropna()
        mean_ret = ret_opt.mean() * 252
        cov_mat = ret_opt.cov() * 252
        sp_ret = sp500[sp500.index >= pd.Timestamp(start_opt)].pct_change().dropna()
        sp_stats = [sp_ret.std() * np.sqrt(252), sp_ret.mean() * 252]

        n_p = 4000
        res = []
        for _ in range(n_p):
            w = np.random.random(len(final_list)); w /= np.sum(w)
            r = np.sum(mean_ret * w)
            v = np.sqrt(w.T @ cov_mat @ w)
            p_ts = (ret_opt * w).sum(axis=1)
            downside = p_ts[p_ts < 0].std() * np.sqrt(252)
            cum = (1 + p_ts).cumprod()
            mdd = abs(((cum / cum.expanding().max()) - 1).min())
            res.append([r, v, (r-rf_rate)/v, (r-rf_rate)/downside if downside!=0 else 0, r/mdd if mdd!=0 else 0, w])

        df_res = pd.DataFrame(res, columns=['ret', 'vol', 'sharpe', 'sortino', 'calmar', 'weights'])
        
        # Plot
        fig_ef, ax_ef = plt.subplots(figsize=(12, 7), facecolor='none')
        sc = ax_ef.scatter(df_res['vol'], df_res['ret'], c=df_res['sharpe'], cmap='RdYlGn', alpha=0.3)
        
        # Marqueurs spécifiques
        best_sharpe = df_res.iloc[df_res['sharpe'].idxmax()]
        best_sortino = df_res.iloc[df_res['sortino'].idxmax()]
        best_calmar = df_res.iloc[df_res['calmar'].idxmax()]
        
        ax_ef.scatter(best_sharpe['vol'], best_sharpe['ret'], color='gold', marker='*', s=300, label='MAX SHARPE', edgecolors='black')
        ax_ef.scatter(best_sortino['vol'], best_sortino['ret'], color='orange', marker='D', s=150, label='MAX SORTINO', edgecolors='black')
        ax_ef.scatter(best_calmar['vol'], best_calmar['ret'], color='magenta', marker='P', s=150, label='MAX CALMAR', edgecolors='black')
        
        for i, t in enumerate(final_list):
            ax_ef.scatter(np.sqrt(cov_mat.iloc[i,i]), mean_ret[i], s=80, label=f"100% {t}", alpha=0.6)
        
        ax_ef.scatter(sp_stats[0], sp_stats[1], color='blue', marker='s', s=100, label='S&P 500')
        ax_ef.legend(loc='upper left', bbox_to_anchor=(1.15, 1), facecolor='#262730')
        st.pyplot(fig_ef, transparent=True)
        
        # Tableau récapitulatif
        st.subheader("📋 Compositions des Portefeuilles Optimaux")
        df_weights = pd.DataFrame({"Actif": final_list})
        df_weights["Max Sharpe (%)"] = best_sharpe['weights'] * 100
        df_weights["Max Sortino (%)"] = best_sortino['weights'] * 100
        df_weights["Max Calmar (%)"] = best_calmar['weights'] * 100
        st.write(df_weights.style.format({c: "{:.2f}%" for c in df_weights.columns if "%" in c}))
