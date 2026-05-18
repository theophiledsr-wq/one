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

st.title("PORTFOLIO MASTER PRO PREMIUM : L'outil de pilotage de portefeuille")

def get_full_ticker_info(symbol):
    try:
        tk = yf.Ticker(symbol)
        info = tk.info
        name = info.get('longName') or info.get('shortName') or symbol
        website = info.get('website', '')
        domain = website.replace('https://www.', '').replace('http://www.', '').replace('https://', '').replace('http://', '').split('/')[0]
        logo_url = f"https://logo.clearbit.com/{domain}" if domain else ""
        return {"name": name, "logo": logo_url}
    except: 
        return {"name": symbol, "logo": ""}

# --- INITIALISATION DE LA MÉMOIRE (SESSION STATE) ---
if 'portfolio_main' not in st.session_state:
    st.session_state.portfolio_main = {
        "MC.PA": {"name": "LVMH", "logo": "https://logo.clearbit.com/lvmh.com"},
        "ASML": {"name": "ASML Holding", "logo": "https://logo.clearbit.com/asml.com"}
    }
if 'portfolio_bench' not in st.session_state:
    st.session_state.portfolio_bench = {
        "URTH": {"name": "iShares MSCI World", "logo": "https://logo.clearbit.com/ishares.com"}
    }

# --- FONCTION DE GESTION UI PORTFEUILLE ---
def render_portfolio_editor(session_dict_name, prefix):
    search_input = st.text_input(f"Ajouter Ticker (ex: AAPL, BTC-USD) :", key=f"add_{prefix}").upper()
    if st.button("➕ Ajouter", key=f"btn_{prefix}"):
        if search_input:
            st.session_state[session_dict_name][search_input] = get_full_ticker_info(search_input)
            st.rerun()

    to_delete = []
    for t, data in st.session_state[session_dict_name].items():
        name = data.get("name", t)
        logo = data.get("logo", "")
        
        c1, c2, c3 = st.columns([1, 4, 1])
        if logo:
            c1.markdown(f'<img src="{logo}" width="25" style="border-radius:50%;">', unsafe_allow_html=True)
        else:
            c1.write("📊")
        c2.caption(f"**{t}** : {name}")
        if c3.button("x", key=f"del_{prefix}_{t}"): to_delete.append(t)
        
    for t in to_delete: 
        del st.session_state[session_dict_name][t]
        st.rerun()

    final_list = list(st.session_state[session_dict_name].keys())
    shares_dict = {}
    if not final_list: 
        st.info("Ajoutez des actifs pour commencer.")
    else:
        st.divider()
        for t in final_list:
            shares_dict[t] = st.number_input(f"Quantité {t}", value=10, min_value=1, key=f"qty_{prefix}_{t}")
    return final_list, shares_dict

# --- SIDEBAR ---
with st.sidebar:
    st.header("🛒 Portefeuilles")
    
    tab_main, tab_bench = st.tabs(["💼 Principal", "⚖️ Benchmark"])
    
    with tab_main:
        st.subheader("Portfolio à tester")
        list_main, shares_main = render_portfolio_editor('portfolio_main', 'main')
        
    with tab_bench:
        st.subheader("Portfolio de référence")
        list_bench, shares_bench = render_portfolio_editor('portfolio_bench', 'bench')

    if not list_main or not list_bench:
        st.warning("Veuillez remplir les deux portefeuilles.")
        st.stop()

    st.divider()
    st.subheader("⚙️ Paramètres Globaux")
    start_date = st.date_input("Historique de référence :", datetime.date(2021, 1, 1))
    horizon = st.number_input("Horizon de projection (jours)", value=252)
    rf_rate = st.number_input("Taux sans risque %", value=3.0) / 100
    n_portfolios = st.number_input("Simulations Frontière", value=5000, min_value=1000, step=1000)

    if st.button("🚀 LANCER L'ANALYSE GLOBALE"):
        st.session_state.run_analysis = True

# --- CHARGEMENT DES DONNÉES ---
@st.cache_data
def load_data_all(tickers_main, tickers_bench):
    all_t = list(set(tickers_main + tickers_bench + ["^GSPC"])) # GSPC = S&P 500 obligatoire
    return yf.download(all_t, start="2015-01-01", progress=False)['Close'].ffill().dropna()

raw_data = load_data_all(list_main, list_bench)

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

def plot_pie_chart(weights, labels, title):
    fig, ax = plt.subplots(figsize=(3, 3), facecolor='none')
    ax.set_facecolor('none')
    mask = weights > 0.01
    w_filtered = weights[mask]
    l_filtered = np.array(labels)[mask]
    
    ax.pie(w_filtered, labels=l_filtered, autopct='%1.1f%%', textprops={'color': "white", 'fontsize': 8}, 
           colors=plt.cm.Set3.colors[:len(w_filtered)])
    ax.set_title(title, color='white', fontsize=10, pad=10)
    return fig

# --- DÉCLENCHEMENT DE L'ANALYSE ---
if st.session_state.get('run_analysis', False):
    
    # --- PRÉPARATION DES DONNÉES ---
    df = raw_data[raw_data.index >= pd.Timestamp(start_date)]
    df_main = df[list_main]
    df_bench_port = df[list_bench]
    df_sp500 = df["^GSPC"]
    
    last_prices_main = df_main.iloc[-1]
    total_val_init_main = sum(last_prices_main[t] * shares_main[t] for t in list_main)
    
    last_prices_bench = df_bench_port.iloc[-1]
    total_val_init_bench = sum(last_prices_bench[t] * shares_bench[t] for t in list_bench)

    # Valorisation historique des 3 entités
    port_main_hist_val = (df_main * [shares_main[t] for t in list_main]).sum(axis=1)
    port_bench_raw_val = (df_bench_port * [shares_bench[t] for t in list_bench]).sum(axis=1)
    
    # Base 100 réajustée sur le capital initial du portfeuille principal pour comparaison
    port_bench_hist_val = (port_bench_raw_val / port_bench_raw_val.iloc[0]) * total_val_init_main
    sp500_hist_val = (df_sp500 / df_sp500.iloc[0]) * total_val_init_main

    # --- SECTION : DONNÉES HISTORIQUES ---
    st.header("Analyse Historique & Superposition")
    col_graph, col_controls = st.columns([3, 1])
    
    with col_controls:
        st.subheader("Contrôles & Alpha")
        period_choice = st.radio("Sélectionnez l'horizon :", ["1 Mois", "3 Mois", "6 Mois", "1 An", "Depuis l'origine"], index=4)
        
        end_d = port_main_hist_val.index[-1]
        if period_choice == "1 Mois": start_d = end_d - relativedelta(months=1)
        elif period_choice == "3 Mois": start_d = end_d - relativedelta(months=3)
        elif period_choice == "6 Mois": start_d = end_d - relativedelta(months=6)
        elif period_choice == "1 An": start_d = end_d - relativedelta(years=1)
        else: start_d = port_main_hist_val.index[0]
        
        mask = (port_main_hist_val.index >= start_d)
        
        p_main_filtered = port_main_hist_val[mask]
        p_bench_filtered = (port_bench_hist_val[mask] / port_bench_hist_val[mask].iloc[0]) * p_main_filtered.iloc[0]
        sp500_filtered = (sp500_hist_val[mask] / sp500_hist_val[mask].iloc[0]) * p_main_filtered.iloc[0]

        rets_main = p_main_filtered.pct_change().dropna()
        rets_bench = p_bench_filtered.pct_change().dropna()
        rets_sp500 = sp500_filtered.pct_change().dropna()
        
        m_sharpe, m_sortino, m_calmar, m_ulcer, m_alpha, m_beta = calc_all_kpis(rets_main, rets_sp500, rf_rate)
        b_sharpe, b_sortino, b_calmar, b_ulcer, b_alpha, b_beta = calc_all_kpis(rets_bench, rets_sp500, rf_rate)
        sp_sharpe, sp_sortino, sp_calmar, sp_ulcer, _, _ = calc_all_kpis(rets_sp500, rets_sp500, rf_rate)
        
        st.divider()
        st.metric("Alpha Principal vs S&P500", f"{m_alpha*100:+.2f} %", help="Surperformance annualisée par rapport au S&P 500 pour le risque pris.")
        st.metric("Alpha Benchmark vs S&P500", f"{b_alpha*100:+.2f} %", help="Surperformance annualisée par rapport au S&P 500 pour le risque pris.")

    with col_graph:
        fig_hist, ax_hist = plt.subplots(figsize=(10, 5), facecolor='none')
        ax_hist.set_facecolor('none')
        ax_hist.plot(p_main_filtered.index, p_main_filtered, color='#00ff00', lw=2, label='Portfolio Principal')
        ax_hist.plot(p_bench_filtered.index, p_bench_filtered, color='#00bfff', lw=2, label='Portfolio Benchmark')
        ax_hist.plot(sp500_filtered.index, sp500_filtered, color='orange', ls='--', lw=1.5, label='S&P 500 (Base)')
        ax_hist.set_ylabel("Valeur (€)", color='white')
        ax_hist.legend(frameon=False, labelcolor='white')
        ax_hist.grid(alpha=0.2)
        ax_hist.tick_params(colors='white')
        st.pyplot(fig_hist, transparent=True)

    st.subheader("Comparatif des Risques & Performance (vs S&P 500)")
    kpi_df = pd.DataFrame({
        "Beta (Volatilité Rel.)": [f"{m_beta:.2f}", f"{b_beta:.2f}", "1.00"],
        "Sharpe Ratio": [f"{m_sharpe:.2f}", f"{b_sharpe:.2f}", f"{sp_sharpe:.2f}"],
        "Sortino Ratio": [f"{m_sortino:.2f}", f"{b_sortino:.2f}", f"{sp_sortino:.2f}"],
        "Calmar Ratio": [f"{m_calmar:.2f}", f"{b_calmar:.2f}", f"{sp_calmar:.2f}"],
        "Ulcer Index": [f"{m_ulcer:.2f}%", f"{b_ulcer:.2f}%", f"{sp_ulcer:.2f}%"]
    }, index=["Portfolio Principal", "Portfolio Benchmark", "S&P 500"])
    st.table(kpi_df)

    st.divider()

    # --- SECTION : MONTE CARLO (3 MODÈLES SUPERPOSÉS) ---
    st.header(f"Projection Monte Carlo des 3 Modèles (Portfolio Principal)")
    
    n_sims_mc = 5000
    log_rets_main = np.log(df_main / df_main.shift(1)).dropna()
    vols_main = log_rets_main.std().values
    means_main = log_rets_main.mean().values

    def run_simulation(mc_type, mc_df=4.0):
        temp_prices = np.tile(last_prices_main.values, (n_sims_mc, 1))
        price_paths = np.zeros((horizon, n_sims_mc, len(list_main)))
        scaling_t = np.sqrt((mc_df - 2) / mc_df) if mc_df > 2 else 1.0

        for t in range(horizon):
            if mc_type == "Normale":
                Z = np.random.normal(0, 1, (n_sims_mc, len(list_main)))
                temp_prices *= np.exp(means_main + Z * vols_main)
            elif mc_type == "Student":
                Z = np.random.standard_t(df=mc_df, size=(n_sims_mc, len(list_main)))
                temp_prices *= np.exp(Z * scaling_t * vols_main)
            elif mc_type == "Bootstrap":
                random_idx = np.random.randint(0, len(log_rets_main), size=n_sims_mc)
                drawn_rets = log_rets_main.iloc[random_idx].values
                temp_prices *= np.exp(drawn_rets)
            price_paths[t] = temp_prices
            
        return np.sum(price_paths * [shares_main[tk] for tk in list_main], axis=2)

    # Lancement des 3 simulations
    paths_norm = run_simulation("Normale")
    paths_stud = run_simulation("Student")
    paths_boot = run_simulation("Bootstrap")

    # Extractions des médianes et bornes de risque
    def get_median_path(paths):
        return paths[:, np.argsort(paths[-1, :])[int(n_sims_mc*0.5)]]
    
    p50_norm = get_median_path(paths_norm)
    p50_stud = get_median_path(paths_stud)
    p50_boot = get_median_path(paths_boot)
    
    p5_boot = paths_boot[:, np.argsort(paths_boot[-1, :])[int(n_sims_mc*0.05)]]
    p95_boot = paths_boot[:, np.argsort(paths_boot[-1, :])[int(n_sims_mc*0.95)]]
    
    final_vals_boot = paths_boot[-1, :]
    proba_gain = (final_vals_boot > total_val_init_main).mean() * 100
    var_95 = total_val_init_main - np.percentile(final_vals_boot, 5)
    cvar_95 = total_val_init_main - np.mean(final_vals_boot[final_vals_boot <= np.percentile(final_vals_boot, 5)])

    # Calcul des performances en %
    perf_norm_pct = (p50_norm[-1] / total_val_init_main - 1) * 100
    perf_stud_pct = (p50_stud[-1] / total_val_init_main - 1) * 100
    perf_boot_pct = (p50_boot[-1] / total_val_init_main - 1) * 100
    
    col1, col2 = st.columns([3, 1])
    with col1:
        fig1, ax1 = plt.subplots(figsize=(10, 5), facecolor='none')
        ax1.set_facecolor('none')
        ax1.plot(p50_norm, color='cyan', lw=2, label='Médiane Normale (GBM)')
        ax1.plot(p50_stud, color='magenta', lw=2, label='Médiane Student-T (Fat Tails)')
        ax1.plot(p50_boot, color='#00ff00', lw=3, label='Médiane Bootstrap Historique')
        
        # Bandes de risques uniquement pour le bootstrap pour garder le graphe lisible
        ax1.fill_between(range(horizon), p5_boot, p95_boot, color='gray', alpha=0.15, label="Zone de Risque 90% (Bootstrap)")
        
        ax1.legend(frameon=False, labelcolor='white')
        ax1.tick_params(colors='white')
        st.pyplot(fig1, transparent=True)
        
        # Affichage des perfs en % sous le graphique
        c_perf1, c_perf2, c_perf3 = st.columns(3)
        c_perf1.metric("Perf. Attendue (Normale)", f"{perf_norm_pct:+.2f} %")
        c_perf2.metric("Perf. Attendue (Student)", f"{perf_stud_pct:+.2f} %")
        c_perf3.metric("Perf. Attendue (Bootstrap)", f"{perf_boot_pct:+.2f} %")
    
    with col2:
        st.write("**Métriques Risque (Modèle Bootstrap) :**")
        st.metric("Valeur Médiane Attendue", f"{p50_boot[-1]:,.0f} €")
        st.metric("Probabilité de Plus-Value", f"{proba_gain:.1f} %")
        st.metric("Value at Risk (95%)", f"- {var_95:,.0f} €")
        st.metric("CVaR (Expected Shortfall 95%)", f"- {cvar_95:,.0f} €")

    st.divider()

    # --- SECTION : FRONTIÈRE EFFICIENTE MODULAIRE ---
    st.header(f"Optimisation de la Frontière Efficiente ({n_portfolios} itérations)")
    
    def generate_efficient_frontier(df_assets, list_assets, shares_assets, total_val, title):
        rets_daily = df_assets.pct_change().dropna()
        np.random.seed(42)
        w_matrix = np.random.dirichlet(np.ones(len(list_assets)), n_portfolios).T
        port_rets_matrix = rets_daily.values @ w_matrix
        
        ann_rets_arr = np.mean(port_rets_matrix, axis=0) * 252
        ann_vols_arr = np.std(port_rets_matrix, axis=0) * np.sqrt(252)
        sharpes_arr = (ann_rets_arr - rf_rate) / ann_vols_arr
        
        downside_rets = np.minimum(port_rets_matrix, 0)
        down_vols_arr = np.std(downside_rets, axis=0) * np.sqrt(252)
        sortinos_arr = np.divide((ann_rets_arr - rf_rate), down_vols_arr, out=np.zeros_like(ann_rets_arr), where=down_vols_arr!=0)
        
        cum_rets_matrix = np.cumprod(1 + port_rets_matrix, axis=0)
        running_max_matrix = np.maximum.accumulate(cum_rets_matrix, axis=0)
        dds_matrix = (cum_rets_matrix - running_max_matrix) / running_max_matrix
        max_dds_arr = np.abs(np.min(dds_matrix, axis=0))
        
        calmars_arr = np.divide(ann_rets_arr, max_dds_arr, out=np.zeros_like(ann_rets_arr), where=max_dds_arr!=0)
        ulcers_arr = np.sqrt(np.mean(dds_matrix**2, axis=0)) * 100
        cagrs_arr = (cum_rets_matrix[-1, :] ** (252 / len(port_rets_matrix))) - 1
        
        idx_sharpe = np.argmax(sharpes_arr)
        idx_sortino = np.argmax(sortinos_arr)
        idx_cagr = np.argmax(cagrs_arr)
        idx_ulcer = np.argmin(ulcers_arr)
        
        last_prices = df_assets.iloc[-1]
        weights_curr = np.array([shares_assets[t] * last_prices[t] for t in list_assets])
        weights_curr /= np.sum(weights_curr)
        curr_ret = np.sum(rets_daily.mean() * 252 * weights_curr)
        curr_vol = np.sqrt(np.dot(weights_curr.T, np.dot(rets_daily.cov() * 252, weights_curr)))

        fig, ax = plt.subplots(figsize=(10, 4), facecolor='none')
        ax.set_facecolor('none')
        scatter = ax.scatter(ann_vols_arr, ann_rets_arr, c=sharpes_arr, cmap='viridis', s=5, alpha=0.3)
        ax.scatter(curr_vol, curr_ret, marker='D', color='white', s=100, label='Actuel', edgecolors='black')
        ax.scatter(ann_vols_arr[idx_sharpe], ann_rets_arr[idx_sharpe], marker='*', color='red', s=150, label='Max Sharpe')
        ax.scatter(ann_vols_arr[idx_sortino], ann_rets_arr[idx_sortino], marker='^', color='orange', s=100, label='Max Sortino')
        ax.scatter(ann_vols_arr[idx_cagr], ann_rets_arr[idx_cagr], marker='P', color='cyan', s=100, label='Max CAGR')
        ax.scatter(ann_vols_arr[idx_ulcer], ann_rets_arr[idx_ulcer], marker='v', color='magenta', s=100, label='Min Ulcer (Sécurité)')
        ax.set_xlabel("Volatilité (Risque)", color='white')
        ax.set_ylabel("Rendement Attendu", color='white')
        ax.set_title(title, color='white')
        ax.tick_params(colors='white')
        ax.legend(frameon=False, labelcolor='white', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        profiles = {"Max Sharpe": idx_sharpe, "Max Sortino": idx_sortino, "Max CAGR": idx_cagr, "Min Ulcer": idx_ulcer}
        
        return fig, profiles, w_matrix, weights_curr, last_prices

    tab_ef_main, tab_ef_bench = st.tabs(["📊 Frontière Portefeuille Principal", "📊 Frontière Benchmark"])

    # --- Sous-section : Principal ---
    with tab_ef_main:
        fig_main, profs_main, w_mat_main, w_curr_main, lp_main = generate_efficient_frontier(
            df_main, list_main, shares_main, total_val_init_main, "Frontière Efficiente - Principal"
        )
        st.pyplot(fig_main, transparent=True)
        
        st.markdown("#### Répartitions Optimales (Principal)")
        c_pie1, c_pie2, c_pie3, c_pie4, c_pie5 = st.columns(5)
        with c_pie1: st.pyplot(plot_pie_chart(w_curr_main, list_main, "Actuel"), transparent=True)
        with c_pie2: st.pyplot(plot_pie_chart(w_mat_main[:, profs_main["Max Sharpe"]], list_main, "Max Sharpe"), transparent=True)
        with c_pie3: st.pyplot(plot_pie_chart(w_mat_main[:, profs_main["Max Sortino"]], list_main, "Max Sortino"), transparent=True)
        with c_pie4: st.pyplot(plot_pie_chart(w_mat_main[:, profs_main["Max CAGR"]], list_main, "Max CAGR"), transparent=True)
        with c_pie5: st.pyplot(plot_pie_chart(w_mat_main[:, profs_main["Min Ulcer"]], list_main, "Min Ulcer"), transparent=True)

        c_tab1, c_tab2 = st.columns(2)
        with c_tab1:
            st.markdown("**Allocation en Nombre de Parts (Principal)**")
            shares_df = pd.DataFrame(index=list_main)
            shares_df["Parts Actuelles"] = [shares_main[t] for t in list_main]
            for p_name, p_idx in profs_main.items():
                target_val = w_mat_main[:, p_idx] * total_val_init_main
                shares_df[f"{p_name}"] = np.round(target_val / lp_main.values).astype(int)
            st.dataframe(shares_df, use_container_width=True)

        with c_tab2:
            st.markdown("**Pondération du Portefeuille (%) (Principal)**")
            weights_df = pd.DataFrame(index=list_main)
            weights_df["Actuel (%)"] = (w_curr_main * 100).round(1)
            for p_name, p_idx in profs_main.items():
                weights_df[f"{p_name} (%)"] = (w_mat_main[:, p_idx] * 100).round(1)
            st.dataframe(weights_df, use_container_width=True)

    # --- Sous-section : Benchmark ---
    with tab_ef_bench:
        if len(list_bench) > 1:
            fig_bench, profs_bench, w_mat_bench, w_curr_bench, lp_bench = generate_efficient_frontier(
                df_bench_port, list_bench, shares_bench, total_val_init_bench, "Frontière Efficiente - Benchmark"
            )
            st.pyplot(fig_bench, transparent=True)
            
            st.markdown("#### Répartitions Optimales (Benchmark)")
            c_pie1_b, c_pie2_b, c_pie3_b, c_pie4_b, c_pie5_b = st.columns(5)
            with c_pie1_b: st.pyplot(plot_pie_chart(w_curr_bench, list_bench, "Actuel"), transparent=True)
            with c_pie2_b: st.pyplot(plot_pie_chart(w_mat_bench[:, profs_bench["Max Sharpe"]], list_bench, "Max Sharpe"), transparent=True)
            with c_pie3_b: st.pyplot(plot_pie_chart(w_mat_bench[:, profs_bench["Max Sortino"]], list_bench, "Max Sortino"), transparent=True)
            with c_pie4_b: st.pyplot(plot_pie_chart(w_mat_bench[:, profs_bench["Max CAGR"]], list_bench, "Max CAGR"), transparent=True)
            with c_pie5_b: st.pyplot(plot_pie_chart(w_mat_bench[:, profs_bench["Min Ulcer"]], list_bench, "Min Ulcer"), transparent=True)

            c_tab1_b, c_tab2_b = st.columns(2)
            with c_tab1_b:
                st.markdown("**Allocation en Nombre de Parts (Benchmark)**")
                shares_df_b = pd.DataFrame(index=list_bench)
                shares_df_b["Parts Actuelles"] = [shares_bench[t] for t in list_bench]
                for p_name, p_idx in profs_bench.items():
                    target_val = w_mat_bench[:, p_idx] * total_val_init_bench
                    shares_df_b[f"{p_name}"] = np.round(target_val / lp_bench.values).astype(int)
                st.dataframe(shares_df_b, use_container_width=True)

            with c_tab2_b:
                st.markdown("**Pondération du Portefeuille (%) (Benchmark)**")
                weights_df_b = pd.DataFrame(index=list_bench)
                weights_df_b["Actuel (%)"] = (w_curr_bench * 100).round(1)
                for p_name, p_idx in profs_bench.items():
                    weights_df_b[f"{p_name} (%)"] = (w_mat_bench[:, p_idx] * 100).round(1)
                st.dataframe(weights_df_b, use_container_width=True)
        else:
            st.info("Le portefeuille Benchmark doit contenir au moins 2 actifs pour générer une frontière efficiente.")
