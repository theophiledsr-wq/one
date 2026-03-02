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
    default_sel = ["AIR.PA", "MC.PA"] if "AIR.PA" in BASE_TICKERS else [BASE_TICKERS[0]]
    selected_tickers = st.multiselect("Actifs :", options=BASE_TICKERS, default=default_sel)
    
    manual_t = st.text_input("Ajout manuel (ex: PUST.PA) :").upper()
    if 'manual_list' not in st.session_state: st.session_state.manual_list = []
    if st.button("➕ Ajouter"):
        if manual_t and manual_t not in st.session_state.manual_list:
            st.session_state.manual_list.append(manual_t)
            st.rerun()

    final_list = list(set(selected_tickers + st.session_state.manual_list))
    
    shares_dict = {}
    if final_list:
        st.subheader("Unités détenues")
        for t in final_list:
            shares_dict[t] = st.number_input(f"Quantité {t}", value=10, min_value=1)

    st.divider()
    decay = 0.94 
    
    if app_mode == "Projection Monte Carlo":
        model_type = st.radio("Modèle :", ["FHS (Historique)", "Student-t"])
        nu = st.slider("nu (v)", 3, 50, 5, help="Épaisseur des queues (3=Crise, 30=Normal)") if model_type == "Student-t" else 5
        n_days = st.number_input("Horizon (jours)", value=150)
        n_sims = st.number_input("Simulations", value=2000)
        run_btn = st.button("🚀 LANCER LA SIMULATION")
    else:
        start_opt = st.date_input("Analyse depuis le", datetime.date(2021, 1, 1))
        rf_rate = st.number_input("Taux sans risque (%)", value=3.0) / 100
        n_portfolios = st.slider("Nombre de simulations", 1000, 10000, 5000)
        run_btn = st.button("🎯 GÉNÉRER LA FRONTIÈRE")

# --- CHARGEMENT DES DONNÉES ---
if final_list:
    try:
        data = yf.download(final_list, start="2020-01-01")['Close']
        if isinstance(data, pd.Series): data = data.to_frame(final_list[0])
        data = data.ffill().dropna()
        last_prices = data.iloc[-1]
        total_val = sum(last_prices[t] * shares_dict[t] for t in final_list)
        current_weights = np.array([(last_prices[t] * shares_dict[t]) / total_val for t in final_list])
    except Exception as e:
        st.error(f"Erreur Yahoo Finance : {e}")
        st.stop()

# --- MODE 1 : MONTE CARLO ---
if app_mode == "Projection Monte Carlo" and 'run_btn' in locals() and run_btn:
    returns = np.log(data / data.shift(1)).dropna()
    ewma_var = (returns**2).ewm(alpha=(1 - decay), adjust=False).mean()
    curr_vol = np.sqrt(ewma_var.iloc[-1].values)
    
    price_paths = np.zeros((n_days, n_sims, len(final_list)))
    temp_prices = np.tile(last_prices.values, (n_sims, 1))
    sim_vols = np.tile(curr_vol, (n_sims, 1))
    
    if model_type == "Student-t": L = cholesky(returns.corr().values, lower=True)

    for t in range(n_days):
        if model_type == "FHS (Historique)":
            idx = np.random.randint(0, len(returns), size=n_sims)
            shocks = (returns.values[idx] / np.sqrt(ewma_var.values[idx]))
        else:
            t_samples = np.random.standard_t(df=nu, size=(n_sims, len(final_list)))
            shocks = (t_samples @ L.T) * np.sqrt((nu - 2) / nu)
        
        daily_ret = shocks * sim_vols
        temp_prices *= np.exp(daily_ret)
        price_paths[t] = temp_prices
        sim_vols = np.sqrt(decay * (sim_vols**2) + (1 - decay) * (daily_ret**2))

    portfolio_paths = np.sum(price_paths * [shares_dict[t] for t in final_list], axis=2)
    final_pnl = portfolio_paths[-1, :] - total_val

    # Affichage
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Valeur Totale", f"{total_val:,.2f} €")
    c2.metric("Espérance Gain", f"{np.mean(final_pnl):+,.2f} €")
    c3.metric("Probabilité Profit", f"{np.mean(final_pnl > 0)*100:.1f} %")
    c4.metric("VaR 95%", f"{np.percentile(final_pnl, 5):,.2f} €")

    plt.rcParams.update({"text.color": "white", "axes.labelcolor": "white", "xtick.color": "white", "ytick.color": "white"})
    fig = plt.figure(figsize=(12, 6), facecolor='none')
    norm = plt.Normalize(final_pnl.min(), final_pnl.max())
    cmap = plt.cm.RdYlGn
    for i in np.random.choice(n_sims, 100):
        plt.plot(portfolio_paths[:, i], color=cmap(norm(final_pnl[i])), alpha=0.3)
    plt.title(f"Projection Monte Carlo ({model_type})", color="white")
    st.pyplot(fig, transparent=True)

# --- MODE 2 : OPTIMISATION & FRONTIÈRE ---
elif app_mode == "Optimisation & Frontière Efficiente":
    st.subheader("📊 Répartition Actuelle")
    col_pie, col_leg = st.columns([1, 1.2])
    with col_pie:
        fig_pie, ax_pie = plt.subplots(figsize=(5, 5), facecolor='none')
        ax_pie.pie([last_prices[t] * shares_dict[t] for t in final_list], labels=final_list, autopct='%1.1f%%', textprops={'color':"w"})
        st.pyplot(fig_pie, transparent=True)
    with col_leg:
        st.dataframe(pd.DataFrame({"Actif": final_list, "Poids Actuel": current_weights*100}).style.format({"Poids Actuel": "{:.2f}%"}), hide_index=True)

    if run_btn:
        st.divider()
        # Calculs financiers
        returns_opt = data[data.index >= pd.Timestamp(start_opt)].pct_change().dropna()
        mean_ret = returns_opt.mean() * 252
        cov_mat = returns_opt.cov() * 252
        
        # Monte Carlo des Portefeuilles
        port_rets, port_vols, port_sharpe = [], [], []
        for _ in range(n_portfolios):
            w = np.random.random(len(final_list))
            w /= np.sum(w)
            port_rets.append(np.sum(mean_ret * w))
            port_vols.append(np.sqrt(np.dot(w.T, np.dot(cov_mat, w))))
            port_sharpe.append((port_rets[-1] - rf_rate) / port_vols[-1])

        # Stratégies spécifiques
        idx_max_sharpe = np.argmax(port_sharpe)
        idx_min_vol = np.argmin(port_vols)
        
        curr_ret_ann = np.sum(mean_ret * current_weights)
        curr_vol_ann = np.sqrt(np.dot(current_weights.T, np.dot(cov_mat, current_weights)))

        # GRAPHE FRONTIÈRE
        
        fig_ef, ax_ef = plt.subplots(figsize=(12, 7), facecolor='none')
        sc = ax_ef.scatter(port_vols, port_rets, c=port_sharpe, cmap='RdYlGn', alpha=0.4)
        
        # Points stratégiques
        ax_ef.scatter(port_vols[idx_max_sharpe], port_rets[idx_max_sharpe], color='gold', marker='*', s=300, label='Max Sharpe (Optimal)')
        ax_ef.scatter(port_vols[idx_min_vol], port_rets[idx_min_vol], color='cyan', marker='o', s=200, label='Min Volatilité')
        ax_ef.scatter(curr_vol_ann, curr_ret_ann, color='white', marker='X', s=250, label='VOTRE PORTEFEUILLE')
        
        ax_ef.set_xlabel("Volatilité Annuelle (Risque)", color="white")
        ax_ef.set_ylabel("Rendement Annuel Espéré", color="white")
        plt.colorbar(sc, label='Ratio de Sharpe')
        ax_ef.legend(facecolor='#262730', edgecolor='white')
        ax_ef.grid(True, alpha=0.1)
        st.pyplot(fig_ef, transparent=True)

        # Ratios de votre portefeuille
        st.subheader("🏆 Vos Ratios Actuels")
        r1, r2, r3 = st.columns(3)
        r1.metric("Sharpe", f"{(curr_ret_ann - rf_rate)/curr_vol_ann:.2f}")
        # Sortino
        p_rets = (returns_opt * current_weights).sum(axis=1)
        downside = p_rets[p_rets < 0].std() * np.sqrt(252)
        r2.metric("Sortino", f"{(curr_ret_ann - rf_rate)/downside:.2f}")
        # Calmar
        cum_rets = (1 + p_rets).cumprod()
        mdd = abs(((cum_rets / cum_rets.expanding().max()) - 1).min())
        r3.metric("Calmar", f"{curr_ret_ann / mdd:.2f}")

        with st.expander("📚 Définitions des Stratégies"):
            st.write("**Max Sharpe (Étoile Or)** : Le point où vous gagnez le plus d'argent par 'gramme' de risque pris.")
            st.write("**Min Volatilité (Point Cyan)** : Le mélange d'actifs qui bouge le moins possible, idéal pour les prudents.")
            st.write("**Votre Portefeuille (X Blanc)** : Votre position actuelle. Si vous êtes loin en dessous de la courbe, vous prenez trop de risque pour pas assez de rendement.")
