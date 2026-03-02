import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec
import datetime
from scipy.linalg import cholesky

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="European Monte Carlo Pro", layout="wide")

st.title("THE FRENCH BUILT TOOL FOR MONTE CARLO PROJECTION FOR THE EUROPEAN INVESTOR")

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
                    tickers.extend([str(tk).split('.')[0] + (".PA" if "CAC" in url else ".DE" if "DAX" in url else ".MC") for tk in t['Ticker'].tolist()])
                    break
        return sorted(list(set(tickers)))
    except:
        return ["AIR.PA", "MC.PA", "SAP.DE", "ASML.AS"]

BASE_TICKERS = get_european_base_list()

# --- BARRE LATÉRALE ---
with st.sidebar:
    st.header("🛒 Configuration")
    selected_tickers = st.multiselect("Actifs :", options=BASE_TICKERS, default=["AIR.PA"])
    
    manual_t = st.text_input("Ajout manuel :").upper()
    if 'manual_list' not in st.session_state: st.session_state.manual_list = []
    if st.button("➕ Ajouter"):
        if manual_t and manual_t not in st.session_state.manual_list:
            st.session_state.manual_list.append(manual_t)

    final_list = list(set(selected_tickers + st.session_state.manual_list))
    
    # --- CHOIX DU MODÈLE ---
    st.divider()
    st.header("🔬 Modèle Mathématique")
    model_type = st.radio("Moteur de simulation :", 
                          ["FHS (Historique Bootstrappé)", "Student-t (Paramétrique Fat-Tails)"])
    
    nu = 5 # Valeur par défaut
    if model_type == "Student-t (Paramétrique Fat-Tails)":
        nu = st.slider("Degrés de liberté (nu)", 3, 50, 5, help="Plus nu est bas, plus le risque de krach est élevé dans la simulation.")

    with st.form("sim_form"):
        st.divider()
        shares_dict = {t: st.number_input(f"Unités {t}", value=10, min_value=1) for t in final_list}
        st.divider()
        start_date_hist = st.date_input("Historique", datetime.date(2019, 1, 1))
        sim_start_date = st.date_input("Date Pivot", datetime.date(2025, 1, 1))
        n_days = st.number_input("Horizon (jours)", value=150)
        n_sims = st.number_input("Nb Simulations", value=2000)
        decay = st.slider("Lambda (EWMA)", 0.80, 0.99, 0.94)
        run_sim = st.form_submit_button("🚀 LANCER LA SIMULATION")

# --- CALCULS ---
if run_sim and final_list:
    with st.spinner(f"Simulation via {model_type}..."):
        data = yf.download(final_list, start=start_date_hist.strftime("%Y-%m-%d"))['Close']
        if len(final_list) == 1: data = data.to_frame(final_list[0])
        data = data.ffill().dropna()
        last_prices = data.iloc[-1]
        
        returns = np.log(data / data.shift(1)).dropna()
        returns_calib = returns[returns.index < sim_start_date.strftime("%Y-%m-%d")].copy()
        
        start_prices_sim = data.iloc[-1].values
        fixed_shares = np.array([shares_dict[t] for t in final_list])
        initial_val = np.sum(start_prices_sim * fixed_shares)

        # Volatilité EWMA
        ewma_var = (returns_calib**2).ewm(alpha=(1 - decay), adjust=False).mean()
        hist_vol = np.sqrt(ewma_var)
        std_residuals = (returns_calib / hist_vol).values
        
        # Préparation Corrélation pour Student-t
        if model_type == "Student-t (Paramétrique Fat-Tails)":
            corr_matrix = returns_calib.corr().values
            # Cholesky pour corréler les variables aléatoires
            L = cholesky(corr_matrix, lower=True)

        # Simulation
        current_vol = hist_vol.iloc[-1].values
        sim_vols = np.tile(current_vol, (n_sims, 1))
        price_paths = np.zeros((n_days, n_sims, len(final_list)))
        temp_prices = np.tile(start_prices_sim, (n_sims, 1))

        for t in range(n_days):
            if model_type == "FHS (Historique Bootstrappé)":
                idx = np.random.randint(0, len(std_residuals), size=n_sims)
                shocks = std_residuals[idx]
            else:
                # Simulation Student-t corrélée
                # On génère des T-samples indépendants
                t_samples = np.random.standard_t(df=nu, size=(n_sims, len(final_list)))
                # On applique la corrélation via Cholesky
                shocks = t_samples @ L.T
                # On normalise les chocs Student pour qu'ils aient une variance de 1 
                # (car la variance d'une loi t est nu/(nu-2))
                shocks = shocks * np.sqrt((nu - 2) / nu)

            if len(final_list) == 1: shocks = shocks.reshape(-1, 1)
            
            daily_ret = shocks * sim_vols
            temp_prices *= np.exp(daily_ret)
            price_paths[t] = temp_prices
            sim_vols = np.sqrt(decay * (sim_vols**2) + (1 - decay) * (daily_ret**2))

        portfolio_sim = np.sum(price_paths * fixed_shares, axis=2) if len(final_list) > 1 else price_paths[:, :, 0] * fixed_shares[0]
        final_pnl = portfolio_sim[-1, :] - initial_val

        # --- AFFICHAGE KPIs ---
        st.subheader(f"📊 Résultats : Modèle {model_type}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Valeur Initiale", f"{initial_val:,.2f} €")
        c2.metric("Espérance", f"{np.mean(final_pnl):+,.2f} €", f"{(np.mean(final_pnl)/initial_val)*100:.2f} %")
        c3.metric("Probabilité Profit", f"{np.mean(final_pnl > 0)*100:.1f} %")
        c4.metric("VaR (95%)", f"{np.percentile(final_pnl, 5):,.2f} €", delta_color="inverse")

        # --- TABLEAU RÉCAP ---
        with st.expander("Voir le détail des positions actuelles"):
            df_summary = pd.DataFrame({
                "Ticker": final_list,
                "Prix (€)": [last_prices[t] for t in final_list],
                "Quantité": [shares_dict[t] for t in final_list],
                "Total (€)": [last_prices[t] * shares_dict[t] for t in final_list]
            })
            st.dataframe(df_summary, use_container_width=True, hide_index=True)

        # --- GRAPH ---
        plt.rcParams.update({"text.color": "#808495", "axes.labelcolor": "#808495", "axes.edgecolor": "#262730"})
        fig = plt.figure(figsize=(16, 7), facecolor='none')
        gs = GridSpec(1, 2, width_ratios=[1.8, 1], wspace=0.2)

        ax1 = fig.add_subplot(gs[0], facecolor='none')
        norm = plt.Normalize(final_pnl.min(), final_pnl.max())
        cmap = plt.cm.RdYlGn
        sample_idx = np.random.choice(n_sims, min(200, n_sims), replace=False)
        for i in sample_idx:
            ax1.plot(portfolio_sim[:, i], color=cmap(norm(final_pnl[i])), lw=0.8, alpha=0.3)
        ax1.plot(np.median(portfolio_sim, axis=1), color='white', lw=3, label='Médiane')
        ax1.set_title("PROJECTION DES TRAJECTOIRES", fontsize=12, fontweight='bold', color='white')
        ax1.grid(True, ls=':', alpha=0.3)

        ax2 = fig.add_subplot(gs[1], facecolor='none')
        n_bins, bins, patches = ax2.hist(final_pnl, bins=50, density=True, alpha=0.7)
        for b, p in zip(bins, patches): p.set_facecolor(cmap(norm(b)))
        ax2.axvline(np.percentile(final_pnl, 5), color='#FF4B4B', ls='--', lw=2, label="VaR")
        ax2.set_title("DISTRIBUTION DES P&L", fontsize=12, fontweight='bold', color='white')

        st.pyplot(fig, transparent=True)

st.divider()
st.caption("Note technique : La simulation Student-t utilise une décomposition de Cholesky pour maintenir les corrélations historiques entre les actifs.")
