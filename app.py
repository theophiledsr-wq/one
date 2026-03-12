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
st.title("PORTFOLIO MASTER PRO PREMIUM")

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

# --- SIDEBAR ---
with st.sidebar:
    st.header("🛒 Portefeuille")
    if 'portfolio' not in st.session_state: st.session_state.portfolio = {}
    
    search_input = st.text_input("Ajouter Ticker (ex: MC.PA, ASML) :").upper()
    if st.button("➕ Ajouter"):
        if search_input:
            st.session_state.portfolio[search_input] = get_full_ticker_info(search_input)
            st.rerun()

    if st.session_state.portfolio:
        to_delete = []
        for t, data in st.session_state.portfolio.items():
            name, logo = data.get("name", t), data.get("logo", "")
            c1, c2, c3 = st.columns([1.5, 4, 1])
            if logo:
                c1.markdown(f'<img src="{logo}" width="45" style="border-radius:8px; margin-top:5px;">', unsafe_allow_html=True)
            else: c1.write("📊")
            c2.caption(f"**{t}**\n{name}")
            if c3.button("x", key=f"del_{t}"): to_delete.append(t)
        for t in to_delete: del st.session_state.portfolio[t]; st.rerun()

    final_list = list(st.session_state.portfolio.keys())
    if not final_list: st.info("Ajoutez des actifs."); st.stop()
    
    st.divider()
    shares_dict = {t: st.number_input(f"Quantité {t}", value=10, min_value=1) for t in final_list}
    
    st.subheader("⚙️ Paramètres")
    start_date = st.date_input("Historique :", datetime.date(2021, 1, 1))
    horizon = st.number_input("Projection (jours)", value=252)
    rf_rate = st.number_input("Taux sans risque %", value=3.0) / 100
    n_portfolios = st.number_input("Simulations", value=5000, step=1000)
    
    if st.button("🚀 LANCER L'ANALYSE"):
        st.session_state.run_analysis = True

# --- LOGIQUE D'ANALYSE ---
def calc_all_kpis(port_rets, bench_rets, rf_rate):
    ann_ret = port_rets.mean() * 252
    ann_vol = port_rets.std() * np.sqrt(252)
    sharpe = (ann_ret - rf_rate) / ann_vol if ann_vol > 0 else 0
    down_vol = port_rets[port_rets < 0].std() * np.sqrt(252)
    sortino = (ann_ret - rf_rate) / down_vol if down_vol > 0 else 0
    cum_rets = (1 + port_rets).cumprod()
    dd = (cum_rets - np.maximum.accumulate(cum_rets)) / np.maximum.accumulate(cum_rets)
    max_dd = abs(dd.min())
    calmar = ann_ret / max_dd if max_dd > 0 else 0
    ulcer = np.sqrt(np.mean(dd**2)) * 100
    beta = np.cov(port_rets, bench_rets)[0, 1] / np.var(bench_rets) if np.var(bench_rets) > 0 else 1
    alpha = ann_ret - (rf_rate + beta * (bench_rets.mean() * 252 - rf_rate))
    return sharpe, sortino, calmar, ulcer, alpha, beta

def plot_donut_chart(weights, labels, title):
    fig, ax = plt.subplots(figsize=(3, 3), facecolor='none')
    mask = weights > 0.01
    # Couleurs "Pro" mates (Bleu, Orange, Vert, Rouge, Violet, Gris)
    prof_colors = ['#4A79A7', '#F28E2B', '#59A14F', '#E15759', '#B07AA1', '#9C755F', '#BAB0AC']
    ax.pie(weights[mask], labels=np.array(labels)[mask], autopct='%1.1f%%', pctdistance=0.8,
           textprops={'color': "white", 'fontsize': 7}, colors=prof_colors,
           wedgeprops=dict(width=0.35, edgecolor='#1E1E1E'))
    ax.set_title(title, color='white', fontsize=9, weight='bold')
    return fig

if st.session_state.get('run_analysis', False):
    raw_data = load_data_all(final_list)
    df = raw_data[raw_data.index >= pd.Timestamp(start_date)]
    df_port, df_sp = df[final_list], df["^GSPC"]
    last_prices = df_port.iloc[-1]
    
    total_val_init = sum(last_prices[t] * shares_dict[t] for t in final_list)
    port_hist_val = (df_port * [shares_dict[t] for t in final_list]).sum(axis=1)

    st.header("Analyse Historique")
    c_graph, c_ctrl = st.columns([3, 1])
    with c_ctrl:
        # On utilise une clé unique pour éviter le rafraîchissement parasite
        period = st.radio("Horizon :", ["1 Mois", "3 Mois", "6 Mois", "1 An", "Max"], index=4, key="period_selector")
        end_d = port_hist_val.index[-1]
        deltas = {"1 Mois": relativedelta(months=1), "3 Mois": relativedelta(months=3), 
                  "6 Mois": relativedelta(months=6), "1 An": relativedelta(years=1)}
        start_d = end_d - deltas.get(period, relativedelta(years=10)) if period != "Max" else port_hist_val.index[0]
        
        p_filtered = port_hist_val[port_hist_val.index >= start_d]
        sp_filtered = (df_sp[df_sp.index >= start_d] / df_sp[df_sp.index >= start_d].iloc[0]) * p_filtered.iloc[0]
        
        rp, rsp = p_filtered.pct_change().dropna(), sp_filtered.pct_change().dropna()
        kp = calc_all_kpis(rp, rsp, rf_rate)
        ksp = calc_all_kpis(rsp, rsp, rf_rate)

    with c_graph:
        fig_h, ax_h = plt.subplots(figsize=(10, 3.5), facecolor='none')
        ax_h.set_facecolor('none')
        ax_h.plot(p_filtered.index, p_filtered, color='#4A79A7', lw=2, label='Portefeuille')
        ax_h.plot(sp_filtered.index, sp_filtered, color='#BAB0AC', ls='--', label='S&P 500')
        ax_h.legend(labelcolor='white', frameon=False); ax_h.tick_params(colors='white')
        st.pyplot(fig_h, transparent=True)

    st.table(pd.DataFrame({
        "Sharpe": [f"{kp[0]:.2f}", f"{ksp[0]:.2f}"], "Sortino": [f"{kp[1]:.2f}", f"{ksp[1]:.2f}"],
        "Calmar": [f"{kp[2]:.2f}", f"{ksp[2]:.2f}"], "Ulcer": [f"{kp[3]:.1f}%", f"{ksp[3]:.1f}%"]
    }, index=["Votre Portefeuille", "Benchmark S&P 500"]))

    st.divider()
    
    # --- OPTIMISATION ---
    st.header("Optimisation & Réallocation")
    rets_assets = df_port.pct_change().dropna()
    w_matrix = np.random.dirichlet(np.ones(len(final_list)), n_portfolios).T
    p_rets = rets_assets.values @ w_matrix
    
    ann_r = np.mean(p_rets, axis=0) * 252
    ann_v = np.std(p_rets, axis=0) * np.sqrt(252)
    sharpes = (ann_r - rf_rate) / ann_v
    
    idx_s, idx_u = np.argmax(sharpes), np.argmin(np.sqrt(np.mean((np.cumprod(1+p_rets, axis=0)-np.maximum.accumulate(np.cumprod(1+p_rets, axis=0)))**2, axis=0)))

    weights_curr = np.array([shares_dict[t] * last_prices[t] for t in final_list])
    weights_curr /= weights_curr.sum()

    cols_pie = st.columns(3)
    cols_pie[0].pyplot(plot_donut_chart(weights_curr, final_list, "Actuel"), transparent=True)
    cols_pie[1].pyplot(plot_donut_chart(w_matrix[:, idx_s], final_list, "Optimale (Sharpe)"), transparent=True)
    cols_pie[2].pyplot(plot_donut_chart(w_matrix[:, idx_u], final_list, "Sécuritaire (Ulcer)"), transparent=True)

    st.markdown("**Comparatif des parts**")
    res_df = pd.DataFrame(index=final_list)
    res_df["Parts Actuelles"] = [shares_dict[t] for t in final_list]
    res_df["Cible Sharpe"] = np.round((w_matrix[:, idx_s] * total_val_init) / last_prices.values).astype(int)
    res_df["Cible Sécurité"] = np.round((w_matrix[:, idx_u] * total_val_init) / last_prices.values).astype(int)
    st.dataframe(res_df, use_container_width=True)
