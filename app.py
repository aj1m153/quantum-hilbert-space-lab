import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QuantLab Terminal",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:       #080c10;
    --surface:  #0d1117;
    --surface2: #161b22;
    --border:   #21262d;
    --accent:   #00d4aa;
    --accent2:  #f7c948;
    --accent3:  #ff6b6b;
    --text:     #e6edf3;
    --muted:    #7d8590;
    --bull:     #26a641;
    --bear:     #da3633;
    --neutral:  #6e7681;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem; }

/* Sidebar nav buttons */
div[data-testid="stSidebar"] .stButton button {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    color: var(--muted) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.72rem !important;
    text-align: left !important;
    width: 100% !important;
    padding: 0.6rem 0.9rem !important;
    border-radius: 4px !important;
    transition: all 0.2s !important;
    margin-bottom: 4px;
}
div[data-testid="stSidebar"] .stButton button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    background: rgba(0,212,170,0.06) !important;
}

/* Main run buttons */
[data-testid="stMainBlockContainer"] .stButton button {
    background: var(--accent) !important;
    color: #000 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 3px !important;
    padding: 0.55rem 1.5rem !important;
    transition: opacity 0.2s !important;
}
[data-testid="stMainBlockContainer"] .stButton button:hover { opacity: 0.85 !important; }

/* Inputs */
input, .stTextInput input, .stNumberInput input, .stSelectbox select {
    background-color: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
    border-radius: 3px !important;
}
.stSelectbox > div > div {
    background-color: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    padding: 1rem !important;
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 0.72rem !important; font-family: 'Space Mono', monospace !important; text-transform: uppercase; letter-spacing: 0.08em; }
[data-testid="stMetricValue"] { color: var(--text) !important; font-family: 'Space Mono', monospace !important; }
[data-testid="stMetricDelta"] { font-family: 'Space Mono', monospace !important; }

/* Headers */
h1, h2, h3 { font-family: 'Space Mono', monospace !important; }
h1 { font-size: 1.4rem !important; letter-spacing: -0.02em; }

/* Info/warning boxes */
.stAlert { background: var(--surface2) !important; border-radius: 4px !important; border-left: 3px solid var(--accent) !important; }

/* Section pill */
.section-pill {
    display: inline-block;
    background: rgba(0,212,170,0.12);
    border: 1px solid rgba(0,212,170,0.3);
    color: var(--accent);
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    padding: 2px 10px;
    border-radius: 20px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.35rem;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 4px;
}
.section-desc {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.88rem;
    color: var(--muted);
    margin-bottom: 1.5rem;
    line-height: 1.6;
}
.divider { border: none; border-top: 1px solid var(--border); margin: 1.5rem 0; }

/* Regime badge */
.badge-bull   { color: #26a641; font-family: 'Space Mono', monospace; font-size: 0.75rem; }
.badge-bear   { color: #da3633; font-family: 'Space Mono', monospace; font-size: 0.75rem; }
.badge-sideways { color: #f7c948; font-family: 'Space Mono', monospace; font-size: 0.75rem; }

/* Stats table */
.stats-table { width: 100%; border-collapse: collapse; font-family: 'Space Mono', monospace; font-size: 0.78rem; }
.stats-table th { color: var(--muted); border-bottom: 1px solid var(--border); padding: 6px 10px; text-align: left; font-weight: 400; text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.68rem; }
.stats-table td { color: var(--text); padding: 6px 10px; border-bottom: 1px solid rgba(33,38,45,0.5); }
.stats-table tr:last-child td { border-bottom: none; }
.pos { color: #26a641 !important; } .neg { color: #da3633 !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-family:'Space Mono',monospace; font-size:1.1rem; font-weight:700;
         color:#00d4aa; letter-spacing:0.05em; padding:0.5rem 0 0.2rem;">
    ⬡ QUANTLAB
    </div>
    <div style="font-family:'DM Sans',sans-serif; font-size:0.72rem; color:#7d8590;
         margin-bottom:1.5rem; letter-spacing:0.02em;">
    Quantitative Research Terminal
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='font-family:Space Mono,monospace;font-size:0.65rem;color:#7d8590;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:8px;'>Modules</div>", unsafe_allow_html=True)

    pages = {
        "01 · Market Regime Clustering":   "regime",
        "02 · Pairs Trading":              "pairs",
        "03 · Volatility Surface":         "vol_surface",
        "04 · CVaR Optimisation":          "cvar",
    }

    if "page" not in st.session_state:
        st.session_state.page = "regime"

    for label, key in pages.items():
        if st.button(label, key=f"nav_{key}"):
            st.session_state.page = key

    st.markdown("<hr style='border-color:#21262d;margin:1.5rem 0 1rem;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-family:Space Mono,monospace;font-size:0.6rem;color:#7d8590;line-height:1.8;'>Data via yfinance<br>Models: hmmlearn · statsmodels<br>Solver: CVXPY · SCS<br>Vis: Plotly</div>", unsafe_allow_html=True)

page = st.session_state.page

# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 1 — MARKET REGIME CLUSTERING
# ═══════════════════════════════════════════════════════════════════════════════
if page == "regime":
    st.markdown('<div class="section-pill">Module 01</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Market Regime Clustering</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Uses a Gaussian Hidden Markov Model trained on rolling returns and realised volatility to detect latent market states — Bull, Bear, or Sideways — for each trading day.</div>', unsafe_allow_html=True)

    col_cfg, col_out = st.columns([1, 2.8], gap="large")

    with col_cfg:
        st.markdown("**Configuration**")
        ticker_hmm   = st.text_input("Ticker", value="SPY", key="hmm_tick").upper()
        period_hmm   = st.selectbox("History", ["2y","3y","5y","10y"], index=2, key="hmm_per")
        n_states_hmm = st.selectbox("HMM States", [2, 3, 4], index=1, key="hmm_st")
        roll_win     = st.slider("Rolling Window (days)", 5, 30, 10, key="hmm_roll")
        n_iter_hmm   = st.slider("EM Iterations", 50, 300, 150, key="hmm_iter")
        run_hmm      = st.button("Run HMM", key="run_hmm")

    with col_out:
        if run_hmm:
            with st.spinner("Fetching data & fitting HMM…"):
                try:
                    from hmmlearn.hmm import GaussianHMM
                    from sklearn.preprocessing import StandardScaler

                    raw = yf.download(ticker_hmm, period=period_hmm, auto_adjust=True, progress=False)
                    if raw.empty:
                        st.error("No data returned — check ticker symbol.")
                        st.stop()

                    close = raw["Close"].squeeze().dropna()
                    log_ret  = np.log(close / close.shift(1)).dropna()
                    roll_ret = log_ret.rolling(roll_win).mean()
                    roll_vol = log_ret.rolling(roll_win).std()
                    feat_df  = pd.DataFrame({"ret": roll_ret, "vol": roll_vol}).dropna()

                    X = feat_df.values
                    scaler = StandardScaler()
                    X_sc = scaler.fit_transform(X)

                    model = GaussianHMM(n_components=n_states_hmm, covariance_type="full",
                                        n_iter=n_iter_hmm, random_state=42)
                    model.fit(X_sc)
                    hidden = model.predict(X_sc)
                    feat_df["state"] = hidden

                    # Map states to regime labels by mean return
                    state_ret = feat_df.groupby("state")["ret"].mean().sort_values()
                    if n_states_hmm == 2:
                        label_map = {state_ret.index[0]: "Bear", state_ret.index[-1]: "Bull"}
                    elif n_states_hmm == 3:
                        label_map = {state_ret.index[0]: "Bear", state_ret.index[1]: "Sideways", state_ret.index[2]: "Bull"}
                    else:
                        keys = list(state_ret.index)
                        label_map = {keys[0]: "Bear", keys[-1]: "Bull"}
                        for k in keys[1:-1]: label_map[k] = "Sideways"
                    feat_df["regime"] = feat_df["state"].map(label_map)

                    price_aligned = close.loc[feat_df.index]

                    COLORS = {"Bull": "#26a641", "Bear": "#da3633", "Sideways": "#f7c948"}

                    # ── Chart ──────────────────────────────────────────
                    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                        row_heights=[0.55, 0.25, 0.2],
                                        vertical_spacing=0.03)

                    # Price line
                    fig.add_trace(go.Scatter(x=price_aligned.index, y=price_aligned.values,
                                             mode="lines", line=dict(color="#7d8590", width=1),
                                             name="Price", showlegend=False), row=1, col=1)

                    # Shaded regimes on price
                    regime_changes = feat_df["regime"].ne(feat_df["regime"].shift()).cumsum()
                    for grp_id in regime_changes.unique():
                        seg = feat_df[regime_changes == grp_id]
                        regime_name = seg["regime"].iloc[0]
                        color = COLORS.get(regime_name, "#555")
                        x0, x1 = seg.index[0], seg.index[-1]
                        fig.add_vrect(x0=x0, x1=x1,
                                      fillcolor=color, opacity=0.18,
                                      layer="below", line_width=0, row=1, col=1)

                    # Regime dots
                    for regime_name, color in COLORS.items():
                        mask = feat_df["regime"] == regime_name
                        if mask.any():
                            fig.add_trace(go.Scatter(
                                x=feat_df.index[mask],
                                y=price_aligned[mask],
                                mode="markers",
                                marker=dict(color=color, size=3, opacity=0.6),
                                name=regime_name,
                            ), row=1, col=1)

                    # Rolling Vol
                    fig.add_trace(go.Scatter(x=feat_df.index, y=feat_df["vol"] * np.sqrt(252),
                                             mode="lines", line=dict(color="#f7c948", width=1.2),
                                             name="Ann. Vol", showlegend=False), row=2, col=1)

                    # Regime state bar
                    state_colors = [COLORS.get(r, "#555") for r in feat_df["regime"]]
                    fig.add_trace(go.Bar(x=feat_df.index, y=[1]*len(feat_df),
                                         marker_color=state_colors, showlegend=False,
                                         name="Regime"), row=3, col=1)

                    fig.update_layout(
                        paper_bgcolor="#080c10", plot_bgcolor="#080c10",
                        font=dict(family="Space Mono", color="#7d8590", size=10),
                        legend=dict(orientation="h", y=1.02, x=0,
                                    bgcolor="rgba(0,0,0,0)",
                                    font=dict(color="#e6edf3", size=10)),
                        height=560, margin=dict(l=10, r=10, t=30, b=10),
                        hovermode="x unified",
                    )
                    for row in [1,2,3]:
                        fig.update_xaxes(gridcolor="#161b22", showgrid=True,
                                         zeroline=False, row=row, col=1)
                        fig.update_yaxes(gridcolor="#161b22", showgrid=True,
                                         zeroline=False, row=row, col=1)
                    fig.update_yaxes(title_text="Price", row=1, col=1)
                    fig.update_yaxes(title_text="Ann. Vol", row=2, col=1)
                    fig.update_yaxes(showticklabels=False, row=3, col=1)

                    st.plotly_chart(fig, use_container_width=True)

                    # ── Stats ──────────────────────────────────────────
                    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
                    cols_s = st.columns(len(COLORS))
                    for i, (rname, color) in enumerate(COLORS.items()):
                        mask = feat_df["regime"] == rname
                        if not mask.any():
                            continue
                        days = mask.sum()
                        avg_ret = feat_df.loc[mask, "ret"].mean() * 252 * 100
                        avg_vol = feat_df.loc[mask, "vol"].mean() * np.sqrt(252) * 100
                        pct     = days / len(feat_df) * 100
                        with cols_s[i]:
                            st.markdown(f"""
                            <div style="background:#0d1117;border:1px solid #21262d;border-top:2px solid {color};
                                 border-radius:6px;padding:1rem;font-family:'Space Mono',monospace;">
                                <div style="color:{color};font-size:0.7rem;text-transform:uppercase;
                                     letter-spacing:0.1em;margin-bottom:0.5rem;">{rname}</div>
                                <div style="color:#e6edf3;font-size:1.4rem;font-weight:700;">{pct:.0f}%</div>
                                <div style="color:#7d8590;font-size:0.65rem;margin-top:4px;">of trading days</div>
                                <hr style="border-color:#21262d;margin:0.6rem 0;">
                                <div style="display:flex;justify-content:space-between;font-size:0.68rem;">
                                    <span style="color:#7d8590;">Ann. Ret</span>
                                    <span style="color:{'#26a641' if avg_ret>0 else '#da3633'};">{avg_ret:+.1f}%</span>
                                </div>
                                <div style="display:flex;justify-content:space-between;font-size:0.68rem;margin-top:4px;">
                                    <span style="color:#7d8590;">Ann. Vol</span>
                                    <span style="color:#e6edf3;">{avg_vol:.1f}%</span>
                                </div>
                            </div>""", unsafe_allow_html=True)

                    # Convergence note
                    st.markdown(f"""<div style="margin-top:1rem;font-family:'Space Mono',monospace;
                        font-size:0.68rem;color:#7d8590;">
                        Model log-likelihood: <span style="color:#00d4aa;">{model.monitor_.history[-1]:.2f}</span>
                        &nbsp;·&nbsp; Converged: <span style="color:{'#26a641' if model.monitor_.converged else '#da3633'};">
                        {'Yes' if model.monitor_.converged else 'No'}</span>
                        &nbsp;·&nbsp; States: {n_states_hmm} &nbsp;·&nbsp; Window: {roll_win}d
                    </div>""", unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.markdown("""
            <div style="border:1px dashed #21262d;border-radius:8px;padding:3rem;text-align:center;margin-top:1rem;">
                <div style="font-family:'Space Mono',monospace;font-size:0.75rem;color:#7d8590;">
                    Configure parameters and click <strong style="color:#00d4aa;">Run HMM</strong> to detect market regimes
                </div>
            </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 2 — PAIRS TRADING
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "pairs":
    st.markdown('<div class="section-pill">Module 02</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Statistical Pairs Trading</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Engle-Granger cointegration test identifies structurally linked assets. Deviations in the spread beyond a z-score threshold generate long/short signals that profit as the spread mean-reverts.</div>', unsafe_allow_html=True)

    col_cfg, col_out = st.columns([1, 2.8], gap="large")

    with col_cfg:
        st.markdown("**Configuration**")
        tick1    = st.text_input("Asset A", value="KO", key="p_t1").upper()
        tick2    = st.text_input("Asset B", value="PEP", key="p_t2").upper()
        period_p = st.selectbox("History", ["2y","3y","5y"], index=1, key="p_per")
        zscore_entry = st.slider("Entry Z-Score", 1.0, 3.0, 2.0, 0.25, key="p_ze")
        zscore_exit  = st.slider("Exit Z-Score",  0.0, 1.5, 0.5, 0.25, key="p_zx")
        roll_z   = st.slider("Z-Score Window", 20, 120, 60, key="p_rz")
        run_pairs = st.button("Run Backtest", key="run_pairs")

    with col_out:
        if run_pairs:
            with st.spinner("Running pairs analysis…"):
                try:
                    from statsmodels.tsa.stattools import coint, adfuller
                    from statsmodels.regression.linear_model import OLS
                    import statsmodels.api as sm

                    raw = yf.download([tick1, tick2], period=period_p, auto_adjust=True, progress=False)["Close"]
                    raw = raw.dropna()
                    # yfinance returns columns in alphabetical order — remap explicitly
                    col_map = {c: c for c in raw.columns}
                    raw = raw.rename(columns=col_map)
                    if tick1 not in raw.columns or tick2 not in raw.columns:
                        st.error(f"Could not find both tickers in returned data. Got: {list(raw.columns)}")
                        st.stop()

                    # Cointegration test
                    score, pvalue, _ = coint(raw[tick1], raw[tick2])

                    # Hedge ratio via OLS
                    X = sm.add_constant(raw[tick2])
                    res = OLS(raw[tick1], X).fit()
                    hedge_ratio = res.params[tick2]
                    spread = raw[tick1] - hedge_ratio * raw[tick2]

                    # Rolling z-score
                    roll_mean = spread.rolling(roll_z).mean()
                    roll_std  = spread.rolling(roll_z).std()
                    zscore    = (spread - roll_mean) / roll_std

                    # Stateful signal: enter on threshold cross, exit when spread reverts
                    # Uses NaN-then-ffill pattern so held positions carry forward correctly
                    raw_signal = pd.Series(np.nan, index=raw.index)
                    raw_signal[zscore >  zscore_entry] = -1   # short spread
                    raw_signal[zscore < -zscore_entry] =  1   # long spread
                    raw_signal[abs(zscore) <= zscore_exit] = 0  # explicit flat at reversion
                    signal = raw_signal.ffill().fillna(0)        # carry position between signals

                    # Returns — computed at leg level to avoid divide-by-zero
                    # when spread crosses zero, pct_change() produces nonsense
                    ret_a      = raw[tick1].pct_change().fillna(0)
                    ret_b      = raw[tick2].pct_change().fillna(0)
                    spread_ret = ret_a - hedge_ratio * ret_b
                    strat_ret  = signal.shift(1) * spread_ret
                    cum_strat  = (1 + strat_ret).cumprod()
                    cum_bh_a   = (1 + ret_a).cumprod()
                    cum_bh_b   = (1 + ret_b).cumprod()

                    sharpe = (strat_ret.mean() / strat_ret.std() * np.sqrt(252)) if strat_ret.std() > 0 else 0
                    total_ret = (cum_strat.iloc[-1] - 1) * 100
                    max_dd = ((cum_strat / cum_strat.cummax()) - 1).min() * 100
                    n_trades = (signal.diff() != 0).sum()

                    # ── Charts ─────────────────────────────────────────
                    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                                        row_heights=[0.3, 0.22, 0.28, 0.2],
                                        vertical_spacing=0.025,
                                        subplot_titles=["Normalised Prices", "Spread", "Z-Score + Signals", "Cumulative P&L"])

                    # Normalised prices
                    for t, col in [(tick1, "#00d4aa"), (tick2, "#f7c948")]:
                        n = raw[t] / raw[t].iloc[0]
                        fig.add_trace(go.Scatter(x=raw.index, y=n, mode="lines",
                                                 line=dict(color=col, width=1.4), name=t), row=1, col=1)

                    # Spread
                    fig.add_trace(go.Scatter(x=spread.index, y=spread, mode="lines",
                                             line=dict(color="#7d8590", width=1), name="Spread",
                                             showlegend=False), row=2, col=1)
                    fig.add_trace(go.Scatter(x=roll_mean.index, y=roll_mean, mode="lines",
                                             line=dict(color="#00d4aa", width=1, dash="dash"),
                                             name="Mean", showlegend=False), row=2, col=1)

                    # Z-score
                    fig.add_trace(go.Scatter(x=zscore.index, y=zscore, mode="lines",
                                             line=dict(color="#e6edf3", width=1), name="Z-Score",
                                             showlegend=False), row=3, col=1)
                    for lvl, col, dash in [(zscore_entry,"#26a641","dash"),
                                           (-zscore_entry,"#da3633","dash"),
                                           (zscore_exit,"#7d8590","dot"),
                                           (-zscore_exit,"#7d8590","dot"),
                                           (0,"#7d8590","solid")]:
                        fig.add_hline(y=lvl, line_color=col, line_dash=dash,
                                      line_width=0.8, row=3, col=1)

                    # Entry markers
                    long_entry  = zscore[zscore < -zscore_entry]
                    short_entry = zscore[zscore >  zscore_entry]
                    fig.add_trace(go.Scatter(x=long_entry.index,  y=long_entry,  mode="markers",
                                             marker=dict(color="#26a641", size=5, symbol="triangle-up"),
                                             name="Long", showlegend=False), row=3, col=1)
                    fig.add_trace(go.Scatter(x=short_entry.index, y=short_entry, mode="markers",
                                             marker=dict(color="#da3633", size=5, symbol="triangle-down"),
                                             name="Short", showlegend=False), row=3, col=1)

                    # P&L
                    fig.add_trace(go.Scatter(x=cum_strat.index, y=cum_strat, mode="lines",
                                             line=dict(color="#00d4aa", width=2), name="Strategy",
                                             showlegend=False), row=4, col=1)
                    fig.add_trace(go.Scatter(x=cum_bh_a.index, y=cum_bh_a, mode="lines",
                                             line=dict(color="#7d8590", width=1, dash="dot"),
                                             name=tick1, showlegend=False), row=4, col=1)
                    fig.add_trace(go.Scatter(x=cum_bh_b.index, y=cum_bh_b, mode="lines",
                                             line=dict(color="#444", width=1, dash="dot"),
                                             name=tick2, showlegend=False), row=4, col=1)
                    fig.add_hline(y=1, line_color="#21262d", line_width=0.8, row=4, col=1)

                    fig.update_layout(
                        paper_bgcolor="#080c10", plot_bgcolor="#080c10",
                        font=dict(family="Space Mono", color="#7d8590", size=10),
                        legend=dict(orientation="h", y=1.01, bgcolor="rgba(0,0,0,0)",
                                    font=dict(color="#e6edf3", size=10)),
                        height=640, margin=dict(l=10, r=10, t=30, b=10),
                    )
                    for i in range(1,5):
                        fig.update_xaxes(gridcolor="#161b22", showgrid=True, zeroline=False, row=i, col=1)
                        fig.update_yaxes(gridcolor="#161b22", showgrid=True, zeroline=False, row=i, col=1)
                    fig.update_annotations(font=dict(family="Space Mono", color="#7d8590", size=9))

                    st.plotly_chart(fig, use_container_width=True)

                    # ── Stats ──────────────────────────────────────────
                    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
                    coint_color = "#26a641" if pvalue < 0.05 else "#da3633"
                    coint_label = "Cointegrated ✓" if pvalue < 0.05 else "Not Cointegrated ✗"

                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Coint. p-value", f"{pvalue:.4f}", delta=coint_label)
                    m2.metric("Hedge Ratio", f"{hedge_ratio:.4f}")
                    m3.metric("Total Return", f"{total_ret:+.1f}%")
                    m4.metric("Sharpe Ratio", f"{sharpe:.2f}")
                    m5.metric("Max Drawdown", f"{max_dd:.1f}%")
                    st.caption(f"Trades executed: {n_trades}  ·  Entry |Z| > {zscore_entry}  ·  Exit |Z| < {zscore_exit}  ·  Window: {roll_z}d")

                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.markdown("""
            <div style="border:1px dashed #21262d;border-radius:8px;padding:3rem;text-align:center;margin-top:1rem;">
                <div style="font-family:'Space Mono',monospace;font-size:0.75rem;color:#7d8590;">
                    Set your pair and parameters, then click <strong style="color:#00d4aa;">Run Backtest</strong>
                </div>
            </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 3 — VOLATILITY SURFACE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "vol_surface":
    st.markdown('<div class="section-pill">Module 03</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Implied Volatility Surface</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Fetches live options chain data, inverts Black-Scholes analytically for each strike & expiry, and renders the full IV surface in 3D — revealing the volatility smile and term structure.</div>', unsafe_allow_html=True)

    col_cfg, col_out = st.columns([1, 2.8], gap="large")

    with col_cfg:
        st.markdown("**Configuration**")
        ticker_vs   = st.text_input("Ticker (options-liquid)", value="SPY", key="vs_tick").upper()
        rf_rate     = st.number_input("Risk-Free Rate (%)", value=5.0, step=0.25, key="vs_rf") / 100
        min_oi      = st.number_input("Min Open Interest", value=100, step=50, key="vs_oi")
        colorscale  = st.selectbox("Colorscale", ["Viridis","Plasma","Turbo","RdYlGn"], index=1, key="vs_cs")
        option_type = st.selectbox("Option Type", ["calls", "puts", "both"], index=0, key="vs_ot")
        run_vs      = st.button("Build Surface", key="run_vs")

    with col_out:
        if run_vs:
            with st.spinner("Fetching options chain & computing IV…"):
                try:
                    from scipy.optimize import brentq
                    from scipy.stats import norm
                    import datetime as dt

                    def bs_price(S, K, T, r, sigma, flag="c"):
                        if T <= 0 or sigma <= 0:
                            return max(S - K, 0) if flag == "c" else max(K - S, 0)
                        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
                        d2 = d1 - sigma*np.sqrt(T)
                        if flag == "c":
                            return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
                        else:
                            return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

                    def implied_vol(price, S, K, T, r, flag="c"):
                        if T <= 1e-6:
                            return np.nan
                        intrinsic = max(S - K, 0) if flag == "c" else max(K - S, 0)
                        if price <= intrinsic + 1e-6:
                            return np.nan
                        try:
                            return brentq(lambda s: bs_price(S, K, T, r, s, flag) - price,
                                          1e-6, 10.0, maxiter=200, xtol=1e-6)
                        except Exception:
                            return np.nan

                    tkr  = yf.Ticker(ticker_vs)
                    spot = None
                    try:
                        fi = tkr.fast_info
                        spot = fi["lastPrice"] or fi["regularMarketPrice"]
                    except Exception:
                        pass
                    if not spot:
                        try:
                            hist = tkr.history(period="1d")
                            spot = float(hist["Close"].iloc[-1]) if not hist.empty else None
                        except Exception:
                            pass
                    if not spot:
                        st.error("Could not fetch spot price — check ticker."); st.stop()

                    exps = tkr.options
                    if not exps:
                        st.error("No options data available for this ticker."); st.stop()

                    today = dt.date.today()
                    records = []
                    flags_to_use = ["calls"] if option_type == "calls" else \
                                   ["puts"]  if option_type == "puts"  else ["calls","puts"]

                    progress = st.progress(0, text="Loading expiries…")
                    for idx, exp in enumerate(exps):
                        progress.progress((idx+1)/len(exps), text=f"Expiry {exp}…")
                        try:
                            chain = tkr.option_chain(exp)
                        except Exception:
                            continue
                        exp_date = dt.datetime.strptime(exp, "%Y-%m-%d").date()
                        T = (exp_date - today).days / 365.0
                        if T <= 0:
                            continue
                        for flag in flags_to_use:
                            df_opt = getattr(chain, flag)
                            df_opt["openInterest"] = pd.to_numeric(df_opt["openInterest"], errors="coerce").fillna(0)
                            df_opt = df_opt[df_opt["openInterest"] >= min_oi].copy()
                            for _, row in df_opt.iterrows():
                                bid = row.get("bid") or 0
                                ask = row.get("ask") or 0
                                mid = (bid + ask) / 2.0
                                if mid <= 0:
                                    mid = float(row.get("lastPrice") or 0)
                                if mid <= 0:
                                    continue
                                K = float(row["strike"])
                                if K <= 0 or K < spot * 0.5 or K > spot * 2.0:
                                    continue
                                iv = implied_vol(mid, spot, K, T, rf_rate, "c" if flag=="calls" else "p")
                                if iv and 0.01 < iv < 5.0:
                                    records.append({"strike": K, "T": T, "iv": iv*100,
                                                    "expiry": exp, "type": flag,
                                                    "moneyness": K/spot})
                    progress.empty()

                    if not records:
                        st.error("No valid IV data computed. Try different filters or a more liquid ticker.")
                        st.stop()

                    iv_df = pd.DataFrame(records)
                    iv_df = iv_df[(iv_df["iv"] > 1) & (iv_df["iv"] < 150)]

                    # ── 3D Surface ─────────────────────────────────────
                    # Pivot to grid for surface
                    from scipy.interpolate import griddata

                    xi = np.linspace(iv_df["moneyness"].min(), iv_df["moneyness"].max(), 60)
                    yi = np.linspace(iv_df["T"].min(), iv_df["T"].max(), 40)
                    xi_g, yi_g = np.meshgrid(xi, yi)
                    zi_g = griddata((iv_df["moneyness"], iv_df["T"]), iv_df["iv"],
                                    (xi_g, yi_g), method="linear")

                    surf = go.Surface(
                        x=xi_g, y=yi_g*365, z=zi_g,
                        colorscale=colorscale,
                        showscale=True,
                        opacity=0.92,
                        contours=dict(
                            z=dict(show=True, usecolormap=True, highlightcolor="white", project_z=True)
                        ),
                        colorbar=dict(
                            title=dict(text="IV (%)", font=dict(family="Space Mono", size=10)),
                            tickfont=dict(family="Space Mono", size=9),
                            len=0.6, thickness=12)
                    )

                    scatter = go.Scatter3d(
                        x=iv_df["moneyness"], y=iv_df["T"]*365, z=iv_df["iv"],
                        mode="markers",
                        marker=dict(size=2, color=iv_df["iv"], colorscale=colorscale,
                                    opacity=0.4),
                        showlegend=False, name="Data pts"
                    )

                    fig = go.Figure(data=[surf, scatter])
                    fig.update_layout(
                        paper_bgcolor="#080c10",
                        scene=dict(
                            bgcolor="#0d1117",
                            xaxis=dict(title=dict(text="Moneyness (K/S)", font=dict(family="Space Mono",size=10)),
                                       gridcolor="#21262d", backgroundcolor="#0d1117"),
                            yaxis=dict(title=dict(text="Days to Expiry", font=dict(family="Space Mono",size=10)),
                                       gridcolor="#21262d", backgroundcolor="#0d1117"),
                            zaxis=dict(title=dict(text="Implied Vol (%)", font=dict(family="Space Mono",size=10)),
                                       gridcolor="#21262d", backgroundcolor="#0d1117"),
                        ),
                        font=dict(family="Space Mono", color="#7d8590", size=9),
                        height=600, margin=dict(l=0, r=0, t=30, b=0),
                        title=dict(text=f"IV Surface · {ticker_vs} · Spot ${spot:.2f}",
                                   font=dict(family="Space Mono", color="#e6edf3", size=12)),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # ── Smile cross-sections ───────────────────────────
                    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
                    st.markdown("<div style='font-family:Space Mono,monospace;font-size:0.75rem;color:#7d8590;margin-bottom:0.5rem;text-transform:uppercase;letter-spacing:0.1em;'>Volatility Smile — Nearest 3 Expiries</div>", unsafe_allow_html=True)

                    smile_fig = go.Figure()
                    colors_sm = ["#00d4aa","#f7c948","#ff6b6b","#a78bfa","#60a5fa"]
                    near_exps = sorted(iv_df["expiry"].unique())[:5]
                    for i, exp in enumerate(near_exps):
                        sub = iv_df[iv_df["expiry"] == exp].sort_values("moneyness")
                        smile_fig.add_trace(go.Scatter(
                            x=sub["moneyness"], y=sub["iv"], mode="lines+markers",
                            line=dict(color=colors_sm[i%len(colors_sm)], width=1.5),
                            marker=dict(size=4), name=exp
                        ))
                    smile_fig.add_vline(x=1.0, line_color="#7d8590", line_dash="dot", line_width=0.8)
                    smile_fig.update_layout(
                        paper_bgcolor="#080c10", plot_bgcolor="#080c10",
                        font=dict(family="Space Mono", color="#7d8590", size=10),
                        xaxis=dict(title="Moneyness", gridcolor="#161b22"),
                        yaxis=dict(title="IV (%)", gridcolor="#161b22"),
                        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#e6edf3",size=10)),
                        height=300, margin=dict(l=10, r=10, t=10, b=10)
                    )
                    st.plotly_chart(smile_fig, use_container_width=True)

                    # Stats
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("ATM IV (nearest)", f"{iv_df.loc[(iv_df['moneyness']-1).abs().idxmin(),'iv']:.1f}%")
                    m2.metric("Data Points", len(iv_df))
                    m3.metric("Expiries", iv_df['expiry'].nunique())
                    m4.metric("Spot Price", f"${spot:.2f}")

                except Exception as e:
                    st.error(f"Error: {e}")
                    import traceback; st.code(traceback.format_exc())
        else:
            st.markdown("""
            <div style="border:1px dashed #21262d;border-radius:8px;padding:3rem;text-align:center;margin-top:1rem;">
                <div style="font-family:'Space Mono',monospace;font-size:0.75rem;color:#7d8590;">
                    Click <strong style="color:#00d4aa;">Build Surface</strong> to compute the implied volatility surface.<br>
                    <span style="font-size:0.65rem;">Best with liquid underlyings: SPY, QQQ, AAPL, TSLA, AMZN</span>
                </div>
            </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 4 — CVaR PORTFOLIO OPTIMISATION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "cvar":
    st.markdown('<div class="section-pill">Module 04</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">CVaR Portfolio Optimisation</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Minimises Conditional Value-at-Risk (Expected Shortfall) — the average loss in the worst α% of scenarios — using convex optimisation via CVXPY. Compared against equal-weight and max-Sharpe portfolios.</div>', unsafe_allow_html=True)

    col_cfg, col_out = st.columns([1, 2.8], gap="large")

    with col_cfg:
        st.markdown("**Configuration**")
        tickers_cvar = st.text_area("Tickers (one per line)", value="AAPL\nMSFT\nGOOGL\nAMZN\nJPM\nGS\nXOM\nJNJ", height=200, key="cv_ticks")
        period_cv    = st.selectbox("History", ["2y","3y","5y"], index=1, key="cv_per")
        alpha_cv     = st.slider("CVaR Confidence Level α", 0.90, 0.99, 0.95, 0.01, key="cv_a",
                                  help="Minimise expected loss in worst (1-α)% of days")
        max_weight   = st.slider("Max Weight per Asset (%)", 10, 60, 40, 5, key="cv_mw") / 100
        run_cvar     = st.button("Optimise Portfolio", key="run_cvar")

    with col_out:
        if run_cvar:
            with st.spinner("Fetching data & solving CVaR optimisation…"):
                try:
                    import cvxpy as cp

                    tlist = [t.strip().upper() for t in tickers_cvar.strip().split("\n") if t.strip()]
                    if len(tlist) < 2:
                        st.error("Enter at least 2 tickers."); st.stop()

                    raw = yf.download(tlist, period=period_cv, auto_adjust=True, progress=False)["Close"]
                    if isinstance(raw, pd.Series):
                        raw = raw.to_frame()
                    raw = raw.dropna(axis=1, how="all").dropna()
                    tlist = list(raw.columns)
                    n = len(tlist)
                    if n < 2:
                        st.error("Insufficient data for selected tickers."); st.stop()

                    rets = raw.pct_change().dropna().values  # (T, n)
                    T, n = rets.shape

                    # ── CVaR Optimisation ──────────────────────────────
                    w    = cp.Variable(n)
                    aux  = cp.Variable()          # VaR threshold
                    z    = cp.Variable(T)          # CVaR aux vars
                    port_ret = rets @ w
                    k    = int(np.ceil((1 - alpha_cv) * T))

                    constraints = [
                        cp.sum(w) == 1,
                        w >= 0,
                        w <= max_weight,
                        z >= 0,
                        z >= -port_ret - aux,
                    ]
                    cvar_obj = aux + (1 / (k)) * cp.sum(z)
                    prob = cp.Problem(cp.Minimize(cvar_obj), constraints)
                    prob.solve(solver=cp.SCS, verbose=False)

                    if prob.status not in ["optimal","optimal_inaccurate"]:
                        st.error(f"Solver status: {prob.status}"); st.stop()

                    w_cvar = np.array(w.value).flatten()
                    w_cvar = np.maximum(w_cvar, 0)
                    w_cvar /= w_cvar.sum()

                    # Equal weight
                    w_eq = np.ones(n) / n

                    # Max Sharpe (simple numerical approach)
                    mu  = rets.mean(axis=0) * 252
                    cov = np.cov(rets.T) * 252

                    def neg_sharpe(w, mu, cov, rf=0.05):
                        pr = w @ mu
                        pv = np.sqrt(w @ cov @ w)
                        return -(pr - rf) / pv if pv > 0 else 1e9

                    from scipy.optimize import minimize
                    bounds = [(0, max_weight)] * n
                    cons_ms = [{"type":"eq","fun": lambda w: w.sum()-1}]
                    res_ms = minimize(neg_sharpe, w_eq, args=(mu, cov),
                                      method="SLSQP", bounds=bounds, constraints=cons_ms,
                                      options={"maxiter":500})
                    w_ms = np.maximum(res_ms.x, 0); w_ms /= w_ms.sum()

                    def port_stats(w, rets):
                        pr = (rets @ w)
                        ann_ret = (1+pr).prod() ** (252/len(pr)) - 1
                        ann_vol = pr.std() * np.sqrt(252)
                        sharpe  = (ann_ret - 0.05) / ann_vol if ann_vol > 0 else 0
                        sorted_pr = np.sort(pr)
                        k2 = max(1, int(np.floor((1 - alpha_cv) * len(pr))))
                        cvar_val = -sorted_pr[:k2].mean()
                        cum = (1+pr).cumprod()
                        dd  = ((cum / cum.cummax()) - 1).min()
                        return dict(ann_ret=ann_ret*100, ann_vol=ann_vol*100,
                                    sharpe=sharpe, cvar=cvar_val*100, max_dd=dd*100)

                    stats_cvar = port_stats(w_cvar, rets)
                    stats_eq   = port_stats(w_eq,   rets)
                    stats_ms   = port_stats(w_ms,   rets)

                    cum_cvar = (1 + rets@w_cvar).cumprod()
                    cum_eq   = (1 + rets@w_eq).cumprod()
                    cum_ms   = (1 + rets@w_ms).cumprod()
                    dates    = raw.index[1:]

                    # ── Charts ─────────────────────────────────────────
                    fig = make_subplots(rows=1, cols=2,
                                        column_widths=[0.62, 0.38],
                                        subplot_titles=["Cumulative Return", "CVaR-Optimal Weights"])

                    for cum, name, col, dash in [
                        (cum_cvar, "CVaR-Optimal", "#00d4aa", "solid"),
                        (cum_eq,   "Equal Weight", "#7d8590", "dot"),
                        (cum_ms,   "Max Sharpe",   "#f7c948", "dash"),
                    ]:
                        fig.add_trace(go.Scatter(x=dates, y=cum, mode="lines",
                                                 line=dict(color=col, width=2, dash=dash),
                                                 name=name), row=1, col=1)
                    fig.add_hline(y=1, line_color="#21262d", line_width=0.8, row=1, col=1)

                    # Weight bar chart
                    sorted_idx = np.argsort(w_cvar)[::-1]
                    bar_colors = ["#00d4aa" if w_cvar[i] > 0.001 else "#21262d" for i in sorted_idx]
                    fig.add_trace(go.Bar(
                        x=[tlist[i] for i in sorted_idx],
                        y=[w_cvar[i]*100 for i in sorted_idx],
                        marker_color=bar_colors, showlegend=False, name="Weights"
                    ), row=1, col=2)

                    fig.update_layout(
                        paper_bgcolor="#080c10", plot_bgcolor="#080c10",
                        font=dict(family="Space Mono", color="#7d8590", size=10),
                        legend=dict(orientation="h", y=1.05, bgcolor="rgba(0,0,0,0)",
                                    font=dict(color="#e6edf3", size=10)),
                        height=420, margin=dict(l=10, r=10, t=40, b=10),
                    )
                    for c in [1,2]:
                        fig.update_xaxes(gridcolor="#161b22", showgrid=True, zeroline=False, row=1, col=c)
                        fig.update_yaxes(gridcolor="#161b22", showgrid=True, zeroline=False, row=1, col=c)
                    fig.update_yaxes(title_text="Growth of $1", row=1, col=1)
                    fig.update_yaxes(title_text="Weight (%)", row=1, col=2)
                    fig.update_annotations(font=dict(family="Space Mono", color="#7d8590", size=9))

                    st.plotly_chart(fig, use_container_width=True)

                    # ── Comparison table ───────────────────────────────
                    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
                    st.markdown("<div style='font-family:Space Mono,monospace;font-size:0.72rem;color:#7d8590;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.75rem;'>Performance Comparison</div>", unsafe_allow_html=True)

                    def fmt(v, pct=True, pos_good=True):
                        s = f"{v:+.2f}%" if pct else f"{v:.3f}"
                        color = "#26a641" if (v > 0) == pos_good else "#da3633"
                        return f'<span style="color:{color};font-family:Space Mono,monospace;font-size:0.78rem;">{s}</span>'

                    rows = [
                        ("Ann. Return",  [stats_cvar["ann_ret"], stats_eq["ann_ret"], stats_ms["ann_ret"]], True, True),
                        ("Ann. Volatility", [stats_cvar["ann_vol"], stats_eq["ann_vol"], stats_ms["ann_vol"]], True, False),
                        ("Sharpe Ratio", [stats_cvar["sharpe"], stats_eq["sharpe"], stats_ms["sharpe"]], False, True),
                        (f"CVaR {int(alpha_cv*100)}%", [stats_cvar["cvar"], stats_eq["cvar"], stats_ms["cvar"]], True, False),
                        ("Max Drawdown", [stats_cvar["max_dd"], stats_eq["max_dd"], stats_ms["max_dd"]], True, False),
                    ]

                    table_html = """<table class='stats-table'>
                    <tr><th>Metric</th><th style="color:#00d4aa;">CVaR-Optimal</th><th>Equal Weight</th><th style="color:#f7c948;">Max Sharpe</th></tr>"""
                    for label, vals, pct, pos_good in rows:
                        table_html += f"<tr><td style='color:#7d8590;'>{label}</td>"
                        for v in vals:
                            table_html += f"<td>{fmt(v, pct, pos_good)}</td>"
                        table_html += "</tr>"
                    table_html += "</table>"
                    st.markdown(table_html, unsafe_allow_html=True)

                    # Weight allocation detail
                    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
                    st.markdown("<div style='font-family:Space Mono,monospace;font-size:0.72rem;color:#7d8590;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.75rem;'>CVaR-Optimal Allocation Detail</div>", unsafe_allow_html=True)

                    alloc_html = "<table class='stats-table'><tr><th>Ticker</th><th>CVaR Weight</th><th>Equal Weight</th><th>Max Sharpe</th><th>Ann. Return</th><th>Ann. Vol</th></tr>"
                    for i in sorted_idx:
                        ticker_i = tlist[i]
                        col_rets = rets[:, i]
                        a_ret_i  = ((1+col_rets).prod()**(252/len(col_rets))-1)*100
                        a_vol_i  = col_rets.std()*np.sqrt(252)*100
                        alloc_html += f"""<tr>
                            <td style='color:#e6edf3;font-weight:500;'>{ticker_i}</td>
                            <td style='color:#00d4aa;'>{w_cvar[i]*100:.1f}%</td>
                            <td style='color:#7d8590;'>{w_eq[i]*100:.1f}%</td>
                            <td style='color:#f7c948;'>{w_ms[i]*100:.1f}%</td>
                            <td class='{"pos" if a_ret_i>0 else "neg"}'>{a_ret_i:+.1f}%</td>
                            <td style='color:#7d8590;'>{a_vol_i:.1f}%</td>
                        </tr>"""
                    alloc_html += "</table>"
                    st.markdown(alloc_html, unsafe_allow_html=True)
                    st.caption(f"α = {alpha_cv:.0%}  ·  Max single weight = {max_weight:.0%}  ·  Solver: CVXPY/SCS  ·  Assets: {n}")

                except Exception as e:
                    st.error(f"Error: {e}")
                    import traceback; st.code(traceback.format_exc())
        else:
            st.markdown("""
            <div style="border:1px dashed #21262d;border-radius:8px;padding:3rem;text-align:center;margin-top:1rem;">
                <div style="font-family:'Space Mono',monospace;font-size:0.75rem;color:#7d8590;">
                    Enter tickers and click <strong style="color:#00d4aa;">Optimise Portfolio</strong><br>
                    <span style="font-size:0.65rem;">CVaR minimisation via convex optimisation (CVXPY)</span>
                </div>
            </div>""", unsafe_allow_html=True)
