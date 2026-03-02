import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec
import datetime

# Configuration de la page web
st.set_page_config(page_title="Simulateur Monte Carlo FHS", layout="wide")
st.title("📈 Simulateur Monte Carlo (Euronext Paris)")

# ==========================================
# 1. FORMULAIRE UTILISATEUR
# ==========================================
with st.sidebar.form("config_form"):
    st.header("Paramètres")
    tickers_input = st.text_input("Tickers (séparés par des virgules)", "PUST.PA, PTPXH.PA")
    shares_input = st.text_input("Quantités respectives (ex: 19, 17)", "19, 17")
    
    start_date_hist = st.date_input("Début de l'historique", datetime.date(2018, 1, 1))
    sim_start_date = st.date_input("Date de début de simulation", datetime.date(2026, 1, 19))
    compare_end_date = st.date_input("Date de fin (comparaison réelle)", datetime.date(2026, 2, 6))
    
    n_days_projection = st.number_input("Horizon de projection (jours)", value=150, step=10)
    n_simulations = st.number_input("Nombre de simulations", value=5000, step=500)
    decay_factor = st.slider("Facteur Lambda (EWMA)", 0.80, 0.99, 0.94, 0.01)
    
    submitted = st.form_submit_button("Lancer la simulation")

# ==========================================
# 2. EXECUTION DU MODELE
# ==========================================
if submitted:
    # Nettoyage des inputs
    tickers = [t.strip() for t in tickers_input.split(',')]
    try:
        fixed_shares = np.array([float(s.strip()) for s in shares_input.split(',')])
    except ValueError:
        st.error("Erreur dans les quantités. Assurez-vous d'utiliser des chiffres séparés par des virgules.")
        st.stop()

    if len(tickers) != len(fixed_shares):
        st.error("Le nombre de tickers doit correspondre au nombre de quantités.")
        st.stop()

    with st.spinner("Téléchargement des données et calculs en cours..."):
        # Conversion des dates pour YFinance
        start_hist_str = start_date_hist.strftime("%Y-%m-%d")
        sim_start_str = sim_start_date.strftime("%Y-%m-%d")
        comp_end_str = compare_end_date.strftime("%Y-%m-%d")

        # Acquisition
        data = yf.download(tickers, start=start_hist_str)['Close']
        # Si un seul ticker, YFinance renvoie une Series, on force en DataFrame
        if isinstance(data, pd.Series):
            data = data.to_frame(tickers[0])
        else:
            data = data[tickers]
            
        data = data.ffill()
        returns = np.log(data / data.shift(1)).dropna()

        # Separation Calibration / Realite
        returns_calib = returns[returns.index < sim_start_str].copy()
        data_real_window = data[(data.index >= sim_start_str) & (data.index <= comp_end_str)]
        
        if data_real_window.empty:
            st.warning("Attention : Aucune donnée réelle trouvée pour la période de comparaison. Affichage de la simulation seule.")
            start_prices = data.iloc[-1].values
            portfolio_real = None
        else:
            start_prices = data_real_window.iloc[0].values
            portfolio_real = np.sum(data_real_window.values * fixed_shares, axis=1)

        initial_portfolio_value = np.sum(start_prices * fixed_shares)

        # Modele EWMA pour la volatilite historique
        ewma_var = (returns_calib**2).ewm(alpha=(1 - decay_factor), adjust=False).mean()
        hist_vol = np.sqrt(ewma_var)
        std_residuals = returns_calib.values / hist_vol.values

        # MOTEUR DE SIMULATION FHS DYNAMIQUE
        current_vol = hist_vol.iloc[-1].values 
        sim_vols = np.tile(current_vol, (n_simulations, 1)) 
        price_paths = np.zeros((n_days_projection, n_simulations, len(tickers)))
        last_prices = np.tile(start_prices, (n_simulations, 1))

        for t in range(n_days_projection):
            idx = np.random.randint(0, len(std_residuals), size=n_simulations)
            shocks = std_residuals[idx]
            daily_ret = shocks * sim_vols
            last_prices = last_prices * np.exp(daily_ret)
            price_paths[t] = last_prices
            sim_vols = np.sqrt(decay_factor * (sim_vols**2) + (1 - decay_factor) * (daily_ret**2))

        # Calcul portefeuille
        portfolio_sim = np.sum(price_paths * fixed_shares, axis=2)

        # ANALYSE DES RESULTATS
        final_values = portfolio_sim[-1, :]
        final_pnl = final_values - initial_portfolio_value

        esperance_gain_eur = np.mean(final_pnl)
        rendement_espere_pct = (esperance_gain_eur / initial_portfolio_value) * 100
        rendement_ann_pct = ((1 + rendement_espere_pct/100)**(252/n_days_projection) - 1) * 100
        var_95 = np.percentile(final_pnl, 5)
        standard_error = np.std(final_values) / np.sqrt(n_simulations)

        # ==========================================
        # 3. DASHBOARD VISUEL (Matplotlib)
        # ==========================================
        fig = plt.figure(figsize=(20, 11))
        gs = GridSpec(2, 2, width_ratios=[2.5, 1], height_ratios=[2, 1], wspace=0.15, hspace=0.25)

        ax1 = fig.add_subplot(gs[0, 0])
        norm = plt.Normalize(final_pnl.min(), final_pnl.max())
        cmap = plt.cm.RdYlGn

        sample_idx = np.random.choice(n_simulations, min(250, n_simulations), replace=False)
        for i in sample_idx:
            ax1.plot(portfolio_sim[:, i], color=cmap(norm(final_pnl[i])), lw=0.5, alpha=0.2)

        ax1.plot(np.median(portfolio_sim, axis=1), color='black', lw=3, label='Médiane (p50)')
        
        if portfolio_real is not None:
            ax1.plot(portfolio_real, color='blue', lw=5, label='PORTEFEUILLE RÉEL', zorder=10)
            
        ax1.set_title(f"Projection FHS Dynamique ({n_simulations} sims)", fontsize=16, fontweight='bold')
        ax1.set_ylabel("Valeur du Portefeuille (€)")
        ax1.grid(True, alpha=0.2)
        ax1.legend(loc='upper left')

        ax2 = fig.add_subplot(gs[1, 0])
        n_bins, bins, patches = ax2.hist(final_pnl, bins=100, color='lightgray', alpha=0.5, density=True)
        for i in range(len(patches)):
            if bins[i] < var_95:
                patches[i].set_facecolor('red')
        ax2.axvline(var_95, color='darkred', ls='--', lw=2, label=f'VaR 95% : {var_95:.2f} EUR')
        ax2.axvline(esperance_gain_eur, color='green', ls='-', lw=2, label=f'Espérance : {esperance_gain_eur:.2f} EUR')
        ax2.axvline(0, color='black', lw=1)
        ax2.set_title("Distribution des Profits / Pertes", fontweight='bold')
        ax2.set_xlabel("Variation de valeur (€)")
        ax2.legend()

        ax_info = fig.add_subplot(gs[:, 1])
        ax_info.axis('off')

        info_text = (
            r"$\mathbf{MATHS\ &\ MODELE}$" + "\n\n"
            r"$\text{Rendement : } r_{t} = z_{t} \cdot \sigma_{t}$" + "\n"
            r"$\text{Vol EWMA : } \sigma_{t}^2 = \lambda \sigma_{t-1}^2 + (1-\lambda)r_{t-1}^2$" + "\n\n"
            "--------------------------------------\n"
            r"$\mathbf{PARAMETRES}$" + "\n"
            f"  Lambda (Decay) : {decay_factor}\n"
            f"  Simulations : {n_simulations}\n"
            f"  Horizon : {n_days_projection} jours\n\n"
            "--------------------------------------\n"
            r"$\mathbf{ESPERANCE\ ET\ RENDEMENT}$" + "\n"
            f"  Espérance Gain : {esperance_gain_eur:+.2f} EUR\n"
            f"  Rendement : {rendement_espere_pct:+.2f} %\n"
            f"  Rend. Ann. : {rendement_ann_pct:+.2f} %\n\n"
            "--------------------------------------\n"
            r"$\mathbf{GESTION\ DU\ RISQUE}$" + "\n"
            f"  VaR 95% : {var_95:.2f} EUR\n"
            f"  Erreur Std : +/- {standard_error:.2f} EUR\n"
            f"  Probabilité gain : {np.mean(final_pnl > 0)*100:.1f} %"
        )

        ax_info.text(0, 0.98, info_text, transform=ax_info.transAxes, fontsize=11, 
                     verticalalignment='top', family='monospace',
                     bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='navy', alpha=0.5))

        # Affichage du graphique dans Streamlit
        st.pyplot(fig)
