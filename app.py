"""
╔══════════════════════════════════════════════════════════════╗
║         QUANTUM HILBERT SPACE LAB  —  qhsl v1.0             ║
║   8D state-space projection · quantum observables ·          ║
║   manifold analysis · anomaly detection as state deviation   ║
╚══════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
import warnings, time, requests
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Quantum Hilbert Space Lab",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:ital,wght@0,300;0,400;0,700;1,400&family=Orbitron:wght@400;600;700;900&display=swap');

:root {
    --bg:       #04060d;
    --surface:  #080c18;
    --surface2: #0c1120;
    --border:   #141d35;
    --accent:   #00e8ff;
    --accent2:  #8b3dff;
    --accent3:  #ff2070;
    --warn:     #ffb300;
    --ok:       #00e8b5;
    --text:     #bfd0e8;
    --muted:    #3a4d6a;
    --glow:     rgba(0,232,255,0.12);
}

html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace;
}

/* ── header ── */
.qlab-header {
    text-align: center;
    padding: 2.5rem 0 2rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}
.qlab-title {
    font-family: 'Orbitron', monospace;
    font-size: 2.6rem;
    font-weight: 900;
    background: linear-gradient(135deg,
        var(--accent) 0%, var(--accent2) 48%, var(--accent3) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: 4px;
    text-transform: uppercase;
    margin: 0;
    text-shadow: none;
}
.qlab-sub {
    font-size: 0.65rem;
    color: var(--muted);
    letter-spacing: 5px;
    text-transform: uppercase;
    margin-top: 0.6rem;
}

/* ── KPI cards ── */
.qkpi {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.1rem 1.3rem;
    position: relative;
    overflow: hidden;
}
.qkpi::after {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent 0%, var(--accent) 50%, transparent 100%);
}
.qkpi-label {
    font-size: 0.58rem; color: var(--muted);
    letter-spacing: 2.5px; text-transform: uppercase;
    margin-bottom: 0.35rem;
}
.qkpi-value {
    font-family: 'Orbitron', monospace;
    font-size: 1.45rem; font-weight: 700;
    color: var(--accent); line-height: 1;
}
.qkpi-value.danger { color: var(--accent3); }
.qkpi-value.warn   { color: var(--warn); }
.qkpi-value.ok     { color: var(--ok); }
.qkpi-sub {
    font-size: 0.62rem; color: var(--muted); margin-top: 0.25rem;
}

/* ── section title ── */
.q-section {
    font-family: 'Orbitron', monospace;
    font-size: 0.68rem; letter-spacing: 3px;
    color: var(--accent); text-transform: uppercase;
    border-left: 2px solid var(--accent);
    padding-left: 0.9rem;
    margin: 2rem 0 1rem;
}

/* ── sidebar ── */
[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label {
    color: var(--muted) !important;
    font-size: 0.68rem !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
}
[data-testid="stSidebar"] h3 {
    color: var(--accent) !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
}

/* ── tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background-color: var(--surface) !important;
    border-radius: 8px; padding: 4px; gap: 3px;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Orbitron', monospace !important;
    font-size: 0.6rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
    border-radius: 6px !important;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    background: var(--bg) !important;
}

/* ── info box ── */
.q-info {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem 1.4rem;
    font-size: 0.76rem;
    line-height: 1.8;
    color: var(--muted);
}
.q-info strong { color: var(--text); }
.q-info code {
    color: var(--accent); background: var(--surface);
    padding: 1px 5px; border-radius: 3px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.73rem;
}
</style>
""", unsafe_allow_html=True)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
SYSTEMS = {
    "Apple (AAPL)":      "AAPL",
    "NVIDIA (NVDA)":     "NVDA",
    "Tesla (TSLA)":      "TSLA",
    "Microsoft (MSFT)":  "MSFT",
    "Amazon (AMZN)":     "AMZN",
    "Meta (META)":       "META",
    "S&P 500 ETF (SPY)": "SPY",
    "Custom":            "CUSTOM",
}

DIM_NAMES = [
    "Ψ₁ Return", "Ψ₂ Volatility", "Ψ₃ Momentum", "Ψ₄ Volume",
    "Ψ₅ RSI", "Ψ₆ Autocorr", "Ψ₇ Skewness", "Ψ₈ MeanRev",
]
DIM_SHORT = ["Return", "Vol", "Mom", "Volume", "RSI", "Autocorr", "Skew", "MeanRev"]

THEME = dict(
    paper_bgcolor="#04060d", plot_bgcolor="#04060d",
    font=dict(family="JetBrains Mono, monospace", color="#bfd0e8", size=10),
    xaxis=dict(gridcolor="#141d35", linecolor="#141d35", showgrid=True),
    yaxis=dict(gridcolor="#141d35", linecolor="#141d35", showgrid=True),
    margin=dict(l=44, r=16, t=48, b=40),
    legend=dict(bgcolor="#080c18", bordercolor="#141d35", borderwidth=1),
)

# ─── DATA HELPERS ─────────────────────────────────────────────────────────────

def _session():
    s = requests.Session()
    s.headers["User-Agent"] = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122 Safari/537.36"
    )
    return s

@st.cache_data(ttl=3600)
def fetch_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    try:
        time.sleep(0.4)
        df = yf.download(ticker, period=period, auto_adjust=True,
                         progress=False, session=_session())
        if df is not None and not df.empty:
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            return df.dropna()
    except Exception:
        pass
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise ValueError(f"No data returned for {ticker}.")
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df.dropna()

# ─── HILBERT SPACE ENGINE ─────────────────────────────────────────────────────

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Project market observables onto 8 orthogonal field dimensions.
    Each dimension is a continuous-valued observable of the system.
    """
    close  = df["Close"].squeeze()
    volume = df["Volume"].squeeze() if "Volume" in df.columns else pd.Series(1.0, index=df.index)

    feat = pd.DataFrame(index=close.index)

    # Ψ₁  Log-return field
    feat["psi_return"] = np.log(close / close.shift(1))

    # Ψ₂  Volatility field  (20-period rolling σ)
    feat["psi_vol"] = feat["psi_return"].rolling(20).std()

    # Ψ₃  Momentum field  (10-period price change)
    feat["psi_momentum"] = close.pct_change(10)

    # Ψ₄  Volume flux  (deviation from 20-day mean, normalised)
    vmean = volume.rolling(20).mean()
    feat["psi_volume"] = (volume - vmean) / (vmean + 1e-9)

    # Ψ₅  RSI oscillator  (14-period, normalised 0-1)
    d = feat["psi_return"]
    gain = d.clip(lower=0).rolling(14).mean()
    loss = (-d.clip(upper=0)).rolling(14).mean()
    feat["psi_rsi"] = 1 - 1 / (1 + gain / (loss + 1e-9))

    # Ψ₆  Autocorrelation  (lag-1 rolling 20)
    feat["psi_autocorr"] = feat["psi_return"].rolling(20).apply(
        lambda x: float(pd.Series(x).autocorr(lag=1)) if len(x) > 3 else 0.0,
        raw=False,
    )

    # Ψ₇  Skewness field  (20-period)
    feat["psi_skew"] = feat["psi_return"].rolling(20).skew()

    # Ψ₈  Mean-reversion field  (price / SMA-50 - 1)
    sma50 = close.rolling(50).mean()
    feat["psi_meanrev"] = (close - sma50) / (sma50 + 1e-9)

    return feat.dropna()


def build_hilbert_states(features: pd.DataFrame):
    """Normalise feature matrix into unit-norm state vectors |ψₜ⟩."""
    scaler = StandardScaler()
    X = scaler.fit_transform(features.values)              # (T, 8)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X_unit = X / (norms + 1e-12)                           # unit sphere
    return X_unit, scaler


def compute_hamiltonian(X: np.ndarray):
    """
    H  = Cov(X)   — encodes the 'energy landscape' of the system.
    Eigendecomposition → energy levels Eₙ and eigenstates |n⟩.
    """
    H = np.cov(X.T)                                        # (8, 8)
    eigvals, eigvecs = np.linalg.eigh(H)
    order = np.argsort(eigvals)[::-1]
    return H, eigvals[order], eigvecs[:, order]


def compute_observables(X: np.ndarray, H: np.ndarray,
                        eigvals: np.ndarray, eigvecs: np.ndarray) -> dict:
    """
    Compute time-series of quantum expectation values for each state vector.
    """
    T = len(X)

    # ⟨Ĥ⟩  System energy — quadratic form
    energy = np.einsum("ti,ij,tj->t", X, H, X)

    # ⟨P̂⟩  State momentum — Euclidean speed in Hilbert space
    momentum = np.zeros(T)
    momentum[1:] = np.linalg.norm(np.diff(X, axis=0), axis=1)

    # Eigenstate projections: cₙ(t) = ⟨n|ψ(t)⟩
    projections = X @ eigvecs                              # (T, 8)

    # |⟨ψ|ψ₀⟩|²  Coherence with ground (mean) state
    psi0 = X.mean(axis=0)
    psi0 /= np.linalg.norm(psi0) + 1e-12
    coherence = (X @ psi0) ** 2

    # S(ρ)  Von-Neumann-like entropy
    probs = projections ** 2
    probs /= probs.sum(axis=1, keepdims=True) + 1e-12
    entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1) / np.log(8)

    # ΔE  Spread across eigenstates (uncertainty)
    uncertainty = np.abs(projections).std(axis=1)

    return {
        "energy":       energy,
        "momentum":     momentum,
        "coherence":    coherence,
        "entropy":      entropy,
        "uncertainty":  uncertainty,
        "projections":  projections,
    }


def manifold_distance(X: np.ndarray, threshold_pct: int = 95):
    """Mahalanobis distance from manifold centroid.  Anomaly = above threshold."""
    mu   = X.mean(axis=0)
    cov  = np.cov(X.T)
    cov_inv = np.linalg.pinv(cov)
    diff = X - mu
    dists = np.sqrt(np.einsum("ti,ij,tj->t", diff, cov_inv, diff))
    threshold  = np.percentile(dists, threshold_pct)
    anomalies  = dists > threshold
    return dists, anomalies, threshold


def project_3d(X: np.ndarray):
    """PCA: 8D → 3D for geometric visualisation."""
    pca = PCA(n_components=3)
    return pca.fit_transform(X), pca.explained_variance_ratio_, pca


def project_full(X: np.ndarray):
    pca = PCA(n_components=8)
    pca.fit(X)
    return pca

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚛️ Lab Controls")
    st.markdown("---")

    sys_choice = st.selectbox("System / Ticker", list(SYSTEMS.keys()))
    if sys_choice == "Custom":
        ticker = st.text_input("Enter Ticker", "AAPL").upper().strip()
    else:
        ticker = SYSTEMS[sys_choice]

    period = st.select_slider(
        "Observation Window",
        options=["6mo", "1y", "2y", "3y", "5y"],
        value="2y",
    )

    st.markdown("---")
    st.markdown("### Manifold Parameters")

    anomaly_pct = st.slider(
        "Deviation Threshold (percentile)", 80, 99, 95,
        help="States above this Mahalanobis percentile are flagged as manifold deviations."
    )

    st.markdown("---")
    run_btn = st.button("⚛  INITIALISE LAB", use_container_width=True, type="primary")

# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="qlab-header">
  <p class="qlab-title">Quantum Hilbert Space Lab</p>
  <p class="qlab-sub">8-Dimensional State Space  ·  Quantum Observables  ·  Manifold Geometry  ·  State Deviation</p>
</div>
""", unsafe_allow_html=True)

if not run_btn:
    st.markdown("""
    <div class="q-info" style="text-align:center; padding:3.5rem 2rem; border-radius:12px;">
      <p style="font-family:Orbitron,monospace; color:#00e8ff; font-size:0.9rem; letter-spacing:4px;">
        SYSTEM DORMANT — AWAITING INITIALISATION
      </p>
      <p style="max-width:520px; margin: 1.5rem auto; line-height:2;">
        Most dashboards show you what <strong>already happened</strong>.<br>
        This one shows how your system <strong>behaves in latent space</strong>.<br><br>
        Gateway performance isn't just plotted — it's projected into an
        <strong>8-dimensional Hilbert space</strong>, then observed through
        measurable operators.<br><br>
        <code>Temporal trajectories → geometric structures</code><br>
        <code>Stability → a region, not a metric</code><br>
        <code>Anomalies → state deviations, not thresholds</code><br><br>
        Select a system on the left and click <strong>INITIALISE LAB</strong>.
      </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─── COMPUTE ──────────────────────────────────────────────────────────────────
with st.spinner("Fetching system data…"):
    try:
        df = fetch_data(ticker, period)
    except Exception as e:
        st.error(f"Data error: {e}")
        st.stop()

with st.spinner("Projecting into 8D Hilbert space…"):
    features          = extract_features(df)
    X, scaler         = build_hilbert_states(features)
    H, eigvals, eigvecs = compute_hamiltonian(X)
    obs               = compute_observables(X, H, eigvals, eigvecs)
    dists, anomalies, dist_threshold = manifold_distance(X, anomaly_pct)
    X3, var_ratio, _  = project_3d(X)
    pca8              = project_full(X)
    dates             = features.index
    close_aligned     = df["Close"].squeeze().reindex(dates)

# ─── KPI ROW ──────────────────────────────────────────────────────────────────
last_price   = float(close_aligned.iloc[-1])
n_anomalies  = int(anomalies.sum())
state_status = "DEVIATION" if anomalies[-1] else "WITHIN MANIFOLD"
status_cls   = "danger" if anomalies[-1] else "ok"

c1, c2, c3, c4, c5, c6 = st.columns(6)

def kpi(label, value, sub, cls=""):
    return f"""<div class="qkpi">
        <div class="qkpi-label">{label}</div>
        <div class="qkpi-value {cls}">{value}</div>
        <div class="qkpi-sub">{sub}</div>
    </div>"""

with c1: st.markdown(kpi("System",         ticker,                          f"${last_price:.2f}"),             unsafe_allow_html=True)
with c2: st.markdown(kpi("Current State",  state_status,                    "manifold position", status_cls),  unsafe_allow_html=True)
with c3: st.markdown(kpi("⟨Ĥ⟩ Energy",    f"{obs['energy'][-1]:.4f}",       "ψᵀ·H·ψ"),                        unsafe_allow_html=True)
with c4: st.markdown(kpi("Coherence",      f"{obs['coherence'][-1]:.3f}",   "|⟨ψ|ψ₀⟩|²"),                    unsafe_allow_html=True)
with c5: st.markdown(kpi("S(ρ) Entropy",   f"{obs['entropy'][-1]:.3f}",     "normalised von Neumann"),         unsafe_allow_html=True)
with c6: st.markdown(kpi("Deviations",     str(n_anomalies),                f"{n_anomalies/len(dates)*100:.1f}% of states", "warn"), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── TABS ─────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "⚛  Hilbert Trajectory",
    "📡  Quantum Operators",
    "🌀  Manifold Geometry",
    "⚡  Deviation Events",
    "🔬  Eigenspectrum",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1  —  HILBERT TRAJECTORY
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="q-section">Temporal Trajectory in 8D Hilbert Space → 3D Projection</div>', unsafe_allow_html=True)

    t_norm = np.linspace(0, 1, len(X3))

    fig_3d = go.Figure()

    # Trajectory line (coloured by time via marker colorscale)
    fig_3d.add_trace(go.Scatter3d(
        x=X3[:, 0], y=X3[:, 1], z=X3[:, 2],
        mode="lines+markers",
        line=dict(width=2, color=t_norm, colorscale="Plasma"),
        marker=dict(
            size=2.5,
            color=t_norm, colorscale="Plasma", opacity=0.55,
            colorbar=dict(title="Time →", thickness=8, len=0.6, x=1.01,
                          tickfont=dict(size=9), titlefont=dict(size=9)),
        ),
        name="State Trajectory",
        hovertext=[str(d.date()) for d in dates],
        hoverinfo="text",
    ))

    # Anomaly points
    if anomalies.any():
        fig_3d.add_trace(go.Scatter3d(
            x=X3[anomalies, 0], y=X3[anomalies, 1], z=X3[anomalies, 2],
            mode="markers",
            marker=dict(size=6, color="#ff2070", opacity=0.9,
                        line=dict(color="white", width=0.5), symbol="cross"),
            name=f"Manifold Deviation  ({n_anomalies})",
            hovertext=[str(d.date()) for d in dates[anomalies]],
            hoverinfo="text",
        ))

    # Start / end
    fig_3d.add_trace(go.Scatter3d(
        x=[X3[0, 0]], y=[X3[0, 1]], z=[X3[0, 2]],
        mode="markers+text", text=["  |ψ₀⟩ INIT"],
        marker=dict(size=9, color="#00e8ff", symbol="diamond"),
        name="Initial State",
    ))
    fig_3d.add_trace(go.Scatter3d(
        x=[X3[-1, 0]], y=[X3[-1, 1]], z=[X3[-1, 2]],
        mode="markers+text", text=["  |ψₜ⟩ NOW"],
        marker=dict(size=9, color="#ff2070", symbol="circle"),
        name="Current State",
    ))

    fig_3d.update_layout(
        paper_bgcolor="#04060d",
        scene=dict(
            bgcolor="#04060d",
            xaxis=dict(gridcolor="#141d35", showbackground=False,
                       title=dict(text=f"PC₁  ({var_ratio[0]*100:.1f}%)", font=dict(size=9))),
            yaxis=dict(gridcolor="#141d35", showbackground=False,
                       title=dict(text=f"PC₂  ({var_ratio[1]*100:.1f}%)", font=dict(size=9))),
            zaxis=dict(gridcolor="#141d35", showbackground=False,
                       title=dict(text=f"PC₃  ({var_ratio[2]*100:.1f}%)", font=dict(size=9))),
            camera=dict(eye=dict(x=1.4, y=1.4, z=0.7)),
        ),
        font=dict(family="JetBrains Mono", color="#bfd0e8"),
        title=dict(
            text=f"8D Hilbert Space Trajectory — 3D Projection  "
                 f"({sum(var_ratio)*100:.1f}% total variance explained)",
            font=dict(size=11, family="Orbitron, monospace"),
        ),
        height=580,
        legend=dict(bgcolor="#080c18", bordercolor="#141d35", font=dict(size=9)),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    st.plotly_chart(fig_3d, use_container_width=True)

    # ── 2D panels ──
    col_a, col_b = st.columns(2)

    with col_a:
        fig_pc12 = go.Figure()
        # Density-coloured scatter
        fig_pc12.add_trace(go.Scatter(
            x=X3[:, 0], y=X3[:, 1],
            mode="markers",
            marker=dict(size=3, color=t_norm, colorscale="Plasma",
                        opacity=0.6, showscale=False),
            name="State Cloud",
        ))
        if anomalies.any():
            fig_pc12.add_trace(go.Scatter(
                x=X3[anomalies, 0], y=X3[anomalies, 1],
                mode="markers",
                marker=dict(size=7, color="#ff2070", symbol="x", opacity=0.9),
                name="Deviation",
            ))
        # Trajectory path
        fig_pc12.add_trace(go.Scatter(
            x=X3[:, 0], y=X3[:, 1], mode="lines",
            line=dict(color="rgba(0,232,255,0.15)", width=1),
            showlegend=False,
        ))
        fig_pc12.update_layout(
            **THEME,
            title="PC₁ – PC₂  Phase Plane", height=330,
            xaxis_title="PC₁", yaxis_title="PC₂",
        )
        st.plotly_chart(fig_pc12, use_container_width=True)

    with col_b:
        fig_price_anom = go.Figure()
        fig_price_anom.add_trace(go.Scatter(
            x=dates, y=close_aligned,
            line=dict(color="#00e8ff", width=1.3),
            fill="tozeroy", fillcolor="rgba(0,232,255,0.04)",
            name="Price",
        ))
        if anomalies.any():
            fig_price_anom.add_trace(go.Scatter(
                x=dates[anomalies], y=close_aligned[anomalies],
                mode="markers",
                marker=dict(color="#ff2070", size=6, symbol="x-open", line=dict(width=1.5)),
                name="Manifold Deviation",
            ))
        fig_price_anom.update_layout(
            **THEME,
            title="Price  ×  Manifold Deviations", height=330,
            xaxis_title="Date", yaxis_title="Price ($)",
        )
        st.plotly_chart(fig_price_anom, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2  —  QUANTUM OPERATORS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="q-section">Observable Operators   ⟨O⟩ = ⟨ψ|Ô|ψ⟩</div>', unsafe_allow_html=True)

    fig_obs = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.055,
        subplot_titles=[
            "⟨Ĥ⟩  System Energy (quadratic form  ψᵀHψ)",
            "⟨P̂⟩  State Momentum (rate of state change  ‖Δψ‖)",
            "|⟨ψ|ψ₀⟩|²  Coherence with Ground State",
            "S(ρ)  Normalised von Neumann Entropy",
        ],
    )

    traces_cfg = [
        (obs["energy"],     "#00e8ff",  "rgba(0,232,255,0.05)",   "Energy",    1),
        (obs["momentum"],   "#8b3dff",  "rgba(139,61,255,0.05)",  "Momentum",  2),
        (obs["coherence"],  "#ff2070",  "rgba(255,32,112,0.0)",   "Coherence", 3),
        (obs["entropy"],    "#ffb300",  "rgba(255,179,0,0.05)",   "Entropy",   4),
    ]

    for y, color, fill, name, row in traces_cfg:
        fig_obs.add_trace(go.Scatter(
            x=dates, y=y,
            line=dict(color=color, width=1.4),
            fill="tozeroy", fillcolor=fill,
            name=name,
        ), row=row, col=1)

    # Coherence: mark anomaly windows
    if anomalies.any():
        for i in range(len(dates)):
            if anomalies[i]:
                fig_obs.add_vrect(
                    x0=dates[i], x1=dates[min(i + 1, len(dates) - 1)],
                    fillcolor="rgba(255,32,112,0.08)", line_width=0,
                    row=3, col=1,
                )

    fig_obs.add_hline(y=0.5, line_dash="dot", line_color="#ffb300",
                      annotation_text="Coherence ½", row=3, col=1,
                      annotation_font=dict(size=8))

    fig_obs.update_layout(
        **THEME, height=720, showlegend=False,
        title=dict(text="Quantum Observables Over Time",
                   font=dict(size=11, family="Orbitron, monospace")),
    )
    for i in range(1, 5):
        fig_obs.update_xaxes(gridcolor="#141d35", row=i, col=1)
        fig_obs.update_yaxes(gridcolor="#141d35", row=i, col=1)

    st.plotly_chart(fig_obs, use_container_width=True)

    # ── Operator cross-correlation ──
    st.markdown('<div class="q-section">Observable Cross-Correlation Matrix</div>', unsafe_allow_html=True)

    op_names = ["Energy", "Momentum", "Coherence", "Entropy", "Uncertainty"]
    op_mat   = np.column_stack([
        obs["energy"], obs["momentum"], obs["coherence"],
        obs["entropy"], obs["uncertainty"],
    ])
    corr = np.corrcoef(op_mat.T)

    fig_corr = go.Figure(go.Heatmap(
        z=corr, x=op_names, y=op_names,
        colorscale=[[0, "#ff2070"], [0.5, "#04060d"], [1, "#00e8ff"]],
        zmid=0,
        text=np.round(corr, 2),
        texttemplate="%{text}",
        textfont=dict(size=10),
        colorbar=dict(title="ρ", thickness=10),
    ))
    fig_corr.update_layout(
        **THEME,
        title="[Ôᵢ, Ôⱼ] — Do the Operators Commute?", height=340,
    )
    st.plotly_chart(fig_corr, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3  —  MANIFOLD GEOMETRY
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="q-section">Natural Manifold — Stability Is a Region, Not a Metric</div>', unsafe_allow_html=True)

    # ── Mahalanobis timeline ──
    roll_mean = pd.Series(dists).rolling(20).mean().values
    roll_std  = pd.Series(dists).rolling(20).std().values

    fig_mah = go.Figure()

    # Shaded 1σ band
    fig_mah.add_trace(go.Scatter(
        x=np.concatenate([dates, dates[::-1]]),
        y=np.concatenate([roll_mean + roll_std, (roll_mean - roll_std)[::-1]]),
        fill="toself", fillcolor="rgba(139,61,255,0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Rolling ±1σ",
    ))

    # Normal states
    fig_mah.add_trace(go.Scatter(
        x=dates[~anomalies], y=dists[~anomalies],
        mode="markers",
        marker=dict(color="#00e8ff", size=3, opacity=0.45),
        name="Within Manifold",
    ))

    # Anomalous states
    if anomalies.any():
        fig_mah.add_trace(go.Scatter(
            x=dates[anomalies], y=dists[anomalies],
            mode="markers",
            marker=dict(color="#ff2070", size=5, opacity=0.9),
            name="Outside Manifold",
        ))

    fig_mah.add_trace(go.Scatter(
        x=dates, y=roll_mean,
        line=dict(color="#8b3dff", width=2),
        name="20-day Rolling Mean",
    ))
    fig_mah.add_hline(
        y=dist_threshold, line_dash="dash", line_color="#ffb300",
        annotation_text=f"Manifold Boundary ({anomaly_pct}th pct.)",
        annotation_font=dict(size=8),
    )

    fig_mah.update_layout(
        **THEME,
        title="Mahalanobis Distance from Manifold Centroid",
        xaxis_title="Date", yaxis_title="Distance  d(ψ, μ)", height=380,
    )
    st.plotly_chart(fig_mah, use_container_width=True)

    # ── Density landscape ──
    st.markdown('<div class="q-section">State-Space Density  —  High Density = Stable Region</div>', unsafe_allow_html=True)

    col_d1, col_d2 = st.columns([3, 2])

    with col_d1:
        fig_dens = go.Figure()
        fig_dens.add_trace(go.Histogram2dContour(
            x=X3[:, 0], y=X3[:, 1],
            colorscale=[
                [0,   "rgba(0,0,0,0)"],
                [0.2, "rgba(0,232,255,0.07)"],
                [0.6, "rgba(139,61,255,0.25)"],
                [1.0, "rgba(0,232,255,0.75)"],
            ],
            ncontours=18,
            showscale=True,
            colorbar=dict(title="Density", thickness=8, titlefont=dict(size=9)),
            name="State Density",
        ))
        if anomalies.any():
            fig_dens.add_trace(go.Scatter(
                x=X3[anomalies, 0], y=X3[anomalies, 1],
                mode="markers",
                marker=dict(color="#ff2070", size=5, symbol="x"),
                name="Manifold Exit",
            ))
        fig_dens.update_layout(
            **THEME,
            xaxis_title=f"PC₁  ({var_ratio[0]*100:.1f}%)",
            yaxis_title=f"PC₂  ({var_ratio[1]*100:.1f}%)",
            title="Phase-Space Density Landscape", height=400,
        )
        st.plotly_chart(fig_dens, use_container_width=True)

    with col_d2:
        # Distance distribution histogram
        fig_dist_h = go.Figure()
        fig_dist_h.add_trace(go.Histogram(
            x=dists[~anomalies], nbinsx=40,
            marker_color="#00e8ff", opacity=0.6, name="Within",
        ))
        if anomalies.any():
            fig_dist_h.add_trace(go.Histogram(
                x=dists[anomalies], nbinsx=20,
                marker_color="#ff2070", opacity=0.7, name="Deviations",
            ))
        fig_dist_h.add_vline(
            x=dist_threshold, line_dash="dash", line_color="#ffb300",
            annotation_text="Boundary",
        )
        fig_dist_h.update_layout(
            **THEME,
            barmode="overlay",
            title="Distance Distribution", height=400,
            xaxis_title="Mahalanobis d", yaxis_title="Count",
        )
        st.plotly_chart(fig_dist_h, use_container_width=True)

    # ── Feature loadings ──
    st.markdown('<div class="q-section">Hilbert Basis Vectors — How Each Ψ Contributes to Each Axis</div>', unsafe_allow_html=True)

    loadings = pca8.components_[:4]

    fig_load = go.Figure()
    pal = ["#00e8ff", "#8b3dff", "#ff2070", "#ffb300"]
    evr = pca8.explained_variance_ratio_

    for i, (pc, col) in enumerate(zip(loadings, pal)):
        fig_load.add_trace(go.Bar(
            name=f"PC{i+1}  ({evr[i]*100:.1f}%)",
            x=DIM_NAMES, y=pc,
            marker_color=col, opacity=0.82,
        ))

    fig_load.add_hline(y=0, line_color="#3a4d6a")
    fig_load.update_layout(
        **THEME,
        title="Basis Vector Loadings  — PC₁…PC₄",
        barmode="group", height=360,
        yaxis_title="Loading Coefficient",
        xaxis_tickangle=-25,
    )
    st.plotly_chart(fig_load, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4  —  DEVIATION EVENTS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="q-section">Manifold Deviation Events — Anomalies as State Evolution Breaks</div>', unsafe_allow_html=True)

    if not anomalies.any():
        st.markdown("""<div class="q-info">
            No deviation events detected above the chosen threshold.
            The system has remained within its natural manifold for the
            entire observation window. Try reducing the threshold.
        </div>""", unsafe_allow_html=True)
    else:
        # Event table
        event_df = pd.DataFrame({
            "Date":              dates[anomalies].strftime("%Y-%m-%d"),
            "d(ψ, μ)":          np.round(dists[anomalies], 3),
            "⟨Ĥ⟩ Energy":       np.round(obs["energy"][anomalies], 4),
            "⟨P̂⟩ Momentum":     np.round(obs["momentum"][anomalies], 4),
            "Coherence":         np.round(obs["coherence"][anomalies], 4),
            "S(ρ) Entropy":      np.round(obs["entropy"][anomalies], 4),
            "Price":             np.round(close_aligned[anomalies].values, 2),
        })
        event_df = event_df.sort_values("d(ψ, μ)", ascending=False).reset_index(drop=True)
        st.dataframe(event_df, use_container_width=True, height=280)

        # ── Deviation timeline (highlighted) ──
        fig_dev = go.Figure()

        # Add anomaly background bands
        for i, (d, is_anom) in enumerate(zip(dates, anomalies)):
            if is_anom:
                fig_dev.add_vrect(
                    x0=d, x1=dates[min(i + 1, len(dates) - 1)],
                    fillcolor="rgba(255,32,112,0.13)", line_width=0,
                )

        fig_dev.add_trace(go.Scatter(
            x=dates, y=dists,
            fill="tozeroy", fillcolor="rgba(0,232,255,0.04)",
            line=dict(color="#00e8ff", width=1.2),
            name="Mahalanobis d",
        ))
        # rolling
        fig_dev.add_trace(go.Scatter(
            x=dates, y=pd.Series(dists).rolling(20).mean(),
            line=dict(color="#8b3dff", width=1.8, dash="dot"),
            name="20d Rolling Mean",
        ))
        fig_dev.add_hline(
            y=dist_threshold, line_dash="dash", line_color="#ffb300",
            annotation_text=f"Boundary ({anomaly_pct}th pct.)",
        )
        fig_dev.update_layout(
            **THEME,
            title="State Deviation Timeline  —  Pink Bands = System Outside Natural Manifold",
            xaxis_title="Date", yaxis_title="d(ψ, μ)", height=360,
        )
        st.plotly_chart(fig_dev, use_container_width=True)

        # ── Radar: anomaly vs normal feature profile ──
        st.markdown('<div class="q-section">Feature-Space Profile: Deviation vs Normal States</div>', unsafe_allow_html=True)

        anom_mean = features.values[anomalies].mean(axis=0)
        norm_mean = features.values[~anomalies].mean(axis=0)

        # Robust normalise per feature (0..1 across both groups)
        combined = np.vstack([anom_mean, norm_mean])
        lo, hi   = combined.min(axis=0), combined.max(axis=0)
        rng      = hi - lo + 1e-9
        a_scaled = (anom_mean - lo) / rng
        n_scaled = (norm_mean - lo) / rng

        theta = DIM_SHORT + [DIM_SHORT[0]]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=list(a_scaled) + [a_scaled[0]], theta=theta,
            fill="toself", fillcolor="rgba(255,32,112,0.10)",
            line=dict(color="#ff2070", width=2),
            name="Deviation States",
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=list(n_scaled) + [n_scaled[0]], theta=theta,
            fill="toself", fillcolor="rgba(0,232,255,0.07)",
            line=dict(color="#00e8ff", width=2),
            name="Normal States",
        ))
        fig_radar.update_layout(
            paper_bgcolor="#04060d",
            polar=dict(
                bgcolor="#080c18",
                radialaxis=dict(gridcolor="#141d35", linecolor="#141d35",
                                range=[0, 1.05], tickfont=dict(size=8)),
                angularaxis=dict(gridcolor="#141d35", linecolor="#141d35"),
            ),
            font=dict(family="JetBrains Mono", color="#bfd0e8"),
            title=dict(text="Hilbert Space Feature Profile: Deviation vs Normal",
                       font=dict(size=11, family="Orbitron, monospace")),
            height=460,
            legend=dict(bgcolor="#080c18", bordercolor="#141d35"),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # ── Paired energy–momentum scatter coloured by anomaly ──
        st.markdown('<div class="q-section">Energy–Momentum Phase Portrait</div>', unsafe_allow_html=True)

        fig_phase = go.Figure()
        fig_phase.add_trace(go.Scatter(
            x=obs["energy"][~anomalies], y=obs["momentum"][~anomalies],
            mode="markers",
            marker=dict(color="#00e8ff", size=3, opacity=0.4),
            name="Normal",
        ))
        if anomalies.any():
            fig_phase.add_trace(go.Scatter(
                x=obs["energy"][anomalies], y=obs["momentum"][anomalies],
                mode="markers",
                marker=dict(color="#ff2070", size=6, opacity=0.85, symbol="x"),
                name="Deviation",
            ))
        fig_phase.update_layout(
            **THEME,
            title="⟨Ĥ⟩ vs ⟨P̂⟩  — Energy–Momentum Phase Portrait",
            xaxis_title="⟨Ĥ⟩ Energy", yaxis_title="⟨P̂⟩ Momentum", height=360,
        )
        st.plotly_chart(fig_phase, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5  —  EIGENSPECTRUM
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="q-section">Hamiltonian Eigenspectrum — Energy Levels of the System</div>', unsafe_allow_html=True)

    # Energy levels
    fig_eig = go.Figure(go.Bar(
        x=[f"|E{i+1}⟩" for i in range(8)],
        y=eigvals,
        marker=dict(
            color=eigvals,
            colorscale=[[0, "#8b3dff"], [0.5, "#00e8ff"], [1, "#ff2070"]],
            line=dict(color="#141d35", width=1),
        ),
        text=[f"{v:.3f}" for v in eigvals],
        textposition="auto",
        textfont=dict(size=9),
    ))
    fig_eig.update_layout(
        **THEME,
        title="Hamiltonian Eigenvalues  Eₙ  (Covariance Spectrum)",
        xaxis_title="Eigenstate |n⟩", yaxis_title="Energy  Eₙ", height=340,
    )
    st.plotly_chart(fig_eig, use_container_width=True)

    # Cumulative energy capture
    cum_var = np.cumsum(eigvals) / eigvals.sum() * 100

    fig_cum = go.Figure()
    fig_cum.add_trace(go.Bar(
        x=[f"|E{i+1}⟩" for i in range(8)],
        y=eigvals / eigvals.sum() * 100,
        marker_color="#8b3dff", opacity=0.6, name="Contribution %",
    ))
    fig_cum.add_trace(go.Scatter(
        x=[f"|E{i+1}⟩" for i in range(8)],
        y=cum_var,
        mode="lines+markers",
        line=dict(color="#00e8ff", width=2),
        marker=dict(size=8, color="#ff2070"),
        name="Cumulative %",
        yaxis="y2",
    ))
    fig_cum.add_hline(y=90, line_dash="dot", line_color="#ffb300",
                      annotation_text="90 % threshold", yref="y2",
                      annotation_font=dict(size=8))
    fig_cum.update_layout(
        **THEME,
        title="Energy Distribution & Cumulative Capture",
        xaxis_title="Eigenstate |n⟩",
        yaxis=dict(title="Energy %", gridcolor="#141d35"),
        yaxis2=dict(title="Cumulative %", overlaying="y", side="right",
                    gridcolor="#141d35", range=[0, 108]),
        height=320,
    )
    st.plotly_chart(fig_cum, use_container_width=True)

    # Eigenstate occupations over time
    st.markdown('<div class="q-section">Eigenstate Occupation Amplitudes  cₙ(t) = ⟨n|ψ(t)⟩</div>', unsafe_allow_html=True)

    proj = obs["projections"]
    pal4 = ["#00e8ff", "#8b3dff", "#ff2070", "#ffb300"]

    fig_proj = go.Figure()
    for i, c in enumerate(pal4):
        fig_proj.add_trace(go.Scatter(
            x=dates, y=proj[:, i],
            line=dict(color=c, width=1.1),
            name=f"|E{i+1}⟩  (E={eigvals[i]:.3f})",
        ))

    if anomalies.any():
        for i in range(len(dates)):
            if anomalies[i]:
                fig_proj.add_vrect(
                    x0=dates[i], x1=dates[min(i + 1, len(dates) - 1)],
                    fillcolor="rgba(255,32,112,0.07)", line_width=0,
                )

    fig_proj.update_layout(
        **THEME,
        title="Eigenstate Amplitudes — Dominant Mode Shifts Signal Regime Change",
        xaxis_title="Date", yaxis_title="Amplitude  cₙ(t)", height=360,
    )
    st.plotly_chart(fig_proj, use_container_width=True)

    # ── Hilbert basis reference table ──
    st.markdown('<div class="q-section">Hilbert Space Basis — 8 Field Dimensions</div>', unsafe_allow_html=True)

    basis_rows = [
        ("Ψ₁", "Return Field",        "Log daily price change",                      "Ψ = 0 at equilibrium, position in return space"),
        ("Ψ₂", "Volatility Field",     "20-period rolling σ",                         "Uncertainty operator — spread of the wavepacket"),
        ("Ψ₃", "Momentum Field",       "10-period price momentum",                    "Kinetic energy of price motion"),
        ("Ψ₄", "Volume Flux",          "Volume deviation from 20d mean",              "Particle flux — external force on the system"),
        ("Ψ₅", "RSI Oscillator",       "14-period relative-strength, normalised 0-1", "Oscillatory potential; 0.5 = ground state"),
        ("Ψ₆", "Autocorrelation",      "Lag-1 return autocorr, rolling 20",           "Time-correlation — memory of past states"),
        ("Ψ₇", "Skewness Field",       "20-period return skewness",                   "Asymmetric potential well; sign = tilt direction"),
        ("Ψ₈", "Mean-Reversion Field", "Price / SMA-50 − 1",                          "Restoring force — distance from equilibrium"),
    ]

    basis_df = pd.DataFrame(
        basis_rows,
        columns=["Dim", "Field Name", "Definition", "Physical Analogy"],
    )
    st.dataframe(basis_df, use_container_width=True, hide_index=True, height=330)
