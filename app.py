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
        indices = {"CAC 40": "https://en.wikipedia.org/wiki/CAC_40", "DAX 40": "https://en.wikipedia.org/wiki/DAX"}
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
    if not final_list: st.stop()

    shares_dict = {t: st.number_input(f"Quantité {t}", value=10, min_value=1) for t in final_list}

    st.divider()
    if app_mode == "Projection Monte Carlo":
        model_type = st.radio("Modèle :", ["FHS (Historique)", "Student-t", "GARCH(1,1)"])
        
        # --- INFOS BULLES ---
        if model_type == "FHS (Historique)":
            st.info("**FHS :** Utilise les rendements réels du passé.")
        elif model_type == "Student-t":
            st.info("**Student-t :** Modèle à queues épaisses pour simuler les krachs.")
        elif model_type == "GARCH(1,1)":
            st.info("**GARCH(1,1) :** Volatilité dynamique auto-adaptative.")

        # NOUVEAU : Sélecteur de date pour Student-t et GARCH
        start_mc = st.date_input("Analyser l'historique depuis le :", datetime.date(2021, 1, 1), key="date_mc")
        
        nu_val = st.slider("nu (v)", 3, 50, 5) if model_type == "Student-t" else 5
        n_days = st.number_input("Horizon (jours)", value=150)
        n_sims = st.number_input("Simulations", value=2000)
        run_btn = st.button("🚀 LANCER LA SIMULATION")
    else:
        start_opt = st.date_input("Analyse depuis le", datetime.date(2021, 1, 1), key="date_opt")
        rf_rate = st.number_input("Taux sans risque (%)", value=3.0) / 100
        run_btn = st.button("🎯 GÉNÉRER LA FRONTIÈRE")

# --- CHARGEMENT ---
@st.cache_data
def load_data(tickers, start_date):
    # On télécharge depuis 2019 par défaut pour avoir assez de recul, 
    # mais on filtrera ensuite selon le sélecteur de l'utilisateur
    df = yf.download(tickers + ["^GSPC"], start="2018-01-01")['Close']
    return df.ffill().dropna()

raw_data = load_data(final_list, "2018-01-01")

# --- LOGIQUE MONTE CARLO ---
if app_mode == "Projection Monte Carlo" and 'run_btn' in locals() and run_btn:
    # Filtrage selon la date choisie par l'utilisateur
    data_filtered = raw_data[raw_data.index >= pd.Timestamp(start_mc)][final_list]
    
    if data_filtered.empty:
        st.error("Pas de données pour cette période.")
        st.stop()

    vol_date = data_filtered.index[-1].date()
    st.info(f"⚡ Calibration basée sur l'historique du {start_mc} au {vol_date}")
    
    returns = np.log(data_filtered / data_filtered.shift(1)).dropna()
    last_prices = data_filtered.iloc[-1]
    total_val = sum(last_prices[t] * shares_dict[t] for t in final_list)
    
    # Init sim
    price_paths = np.zeros((n_days, n_sims, len(final_list)))
    temp_prices = np.tile(last_prices.values, (n_sims, 1))
    
    decay = 0.94
    ewma_var = (returns**2).ewm(alpha=(1 - decay), adjust=False).mean()
    sim_vols = np.tile(np.sqrt(ewma_var.iloc[-1].values), (n_sims, 1))

    if model_type == "Student-t":
        L = cholesky(returns.corr().values + np.eye(len(final_list))*1e-8, lower=True)
    
    if model_type == "GARCH(1,1)":
        omega, alpha, beta = 1e-6, 0.05, 0.90

    for t in range(n_days):
        if model_type == "FHS (Historique)":
            std_rets = (returns / np.sqrt(ewma_var.shift(1))).dropna()
            shocks = std_rets.sample(n_sims, replace=True).values
        elif model_type == "Student-t":
            t_samples = np.random.standard_t(df=nu_val, size=(n_sims, len(final_list)))
            shocks = (t_samples @ L.T) * np.sqrt((nu_val - 2) / nu_val)
        else:
            shocks = np.random.normal(0, 1, size=(n_sims, len(final_list)))

        daily_ret = shocks * sim_vols
        temp_prices *= np.exp(daily_ret)
        price_paths[t] = temp_prices
        
        if model_type == "GARCH(1,1)":
            sim_vols = np.sqrt(omega + alpha * (daily_ret**2) + beta * (sim_vols**2))
        else:
            sim_vols = np.sqrt(decay * (sim_vols**2) + (1 - decay) * (daily_ret**2))

    portfolio_paths = np.sum(price_paths * [shares_dict[t] for t in final_list], axis=2)
    final_pnl = portfolio_paths[-1, :] - total_val
    
    st.columns(3)[1].metric(f"Issue Médiane", f"{np.median(final_pnl):,.2f} €", f"{(np.median(final_pnl)/total_val)*100:.2f} %")

    # Graphiques (Identiques à précédemment)
    fig = plt.figure(figsize=(16, 7), facecolor='none')
    gs = GridSpec(1, 2, width_ratios=[1.8, 1])
    plt.rcParams.update({"text.color": "white", "axes.labelcolor": "white", "xtick.color": "white", "ytick.color": "white"})
    ax1 = fig.add_subplot(gs[0], facecolor='none')
    norm = plt.Normalize(final_pnl.min(), final_pnl.max())
    for i in np.random.choice(n_sims, 100):
        ax1.plot(portfolio_paths[:, i], color=plt.cm.RdYlGn(norm(final_pnl[i])), alpha=0.3)
    ax2 = fig.add_subplot(gs[1], facecolor='none')
    n, bins, patches = ax2.hist(final_pnl, bins=50, density=True, alpha=0.8)
    for b, p in zip(bins, patches): p.set_facecolor('red' if b < 0 else 'green')
    st.pyplot(fig, transparent=True)

# --- MODE 2 : OPTIMISATION ---
elif app_mode == "Optimisation & Frontière Efficiente":
    data_opt = raw_data[final_list]
    last_prices = data_opt.iloc[-1]
    current_weights = np.array([(last_prices[t] * shares_dict[t]) / sum(last_prices[x]*shares_dict[x] for x in final_list) for t in final_list])
    
    st.subheader("📊 Composition Actuelle")
    c_pie, c_tab = st.columns([1, 1.5])
    with c_pie:
        fig_p, ax_p = plt.subplots(figsize=(5, 5), facecolor='none')
        ax_p.pie([last_prices[t]*shares_dict[t] for t in final_list], labels=final_list, autopct='%1.1f%%', textprops={'color':"w"})
        st.pyplot(fig_p, transparent=True)
    with c_tab:
        st.table(pd.DataFrame({"Actif": final_list, "Poids": [f"{w*100:.1f}%" for w in current_weights]}))

    if run_btn:
        ret_opt = data_opt[data_opt.index >= pd.Timestamp(start_opt)].pct_change().dropna()
        # ... (reste du code d'optimisation inchangé)
        mean_ret = ret_opt.mean() * 252
        cov_mat = ret_opt.cov() * 252
        sp_ret = raw_data["^GSPC"][raw_data.index >= pd.Timestamp(start_opt)].pct_change().dropna()
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
        fig_ef, ax_ef = plt.subplots(figsize=(12, 7), facecolor='none')
        sc = ax_ef.scatter(df_res['vol'], df_res['ret'], c=df_res['sharpe'], cmap='RdYlGn', alpha=0.3)
        pts = {"Sharpe": ("gold", "*", 300), "Sortino": ("orange", "D", 150), "Calmar": ("magenta", "P", 150)}
        for label, (color, marker, size) in pts.items():
            best = df_res.iloc[df_res[label.lower()].idxmax()]
            ax_ef.scatter(best['vol'], best['ret'], color=color, marker=marker, s=size, label=f'MAX {label}', edgecolors='black')
        ax_ef.scatter(sp_stats[0], sp_stats[1], color='blue', marker='s', s=100, label='S&P 500')
        ax_ef.legend(loc='upper left', bbox_to_anchor=(1.15, 1), facecolor='#262730')
        st.pyplot(fig_ef, transparent=True)
        
        df_weights = pd.DataFrame({"Actif": final_list})
        for label in pts.keys():
            df_weights[f"Max {label} (%)"] = df_res.iloc[df_res[label.lower()].idxmax()]['weights'] * 100
        st.write(df_weights.style.format({c: "{:.2f}%" for c in df_weights.columns if "%" in c}))
