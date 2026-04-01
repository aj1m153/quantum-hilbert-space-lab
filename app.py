"""
⚛ Quantum Hilbert Space Lab — Stock Edition
Stock market analysis through 8D state space geometry.

Instead of asking "Is this value too high?"
We ask: "Did the stock leave its natural market manifold?"
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
from scipy.optimize import minimize
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Quantum Hilbert Space Lab",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "⚛ Quantum Hilbert Space Lab — Stock market analysis through 8D state space geometry.",
    },
)

# Inject Streamlit theme config programmatically (mirrors .streamlit/config.toml)
import os, pathlib, textwrap
_cfg_dir  = pathlib.Path(".streamlit")
_cfg_file = _cfg_dir / "config.toml"
_cfg_dir.mkdir(exist_ok=True)
if not _cfg_file.exists():
    _cfg_file.write_text(textwrap.dedent("""
        [theme]
        primaryColor = CB_BLUE
        backgroundColor = "#F8FAFC"
        secondaryBackgroundColor = "#E5E7EB"
        textColor = "#111827"
        font = "sans serif"
    """).strip())

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
:root {
  --bg:#F8FAFC;
  --surface:#ffffff;
  --surface2:#E5E7EB;
  --border:#E5E7EB;
  --border2:#d1d5db;
  --accent:#0077BB;
  --accent-light:#E8F4FD;
  --accent-dark:#005B8E;
  --positive:#009988;
  --positive-light:#E0F5F3;
  --negative:#CC3311;
  --negative-light:#FAEAE6;
  --warning:#EE7733;
  --warning-light:#FDF2E9;
  --text:#111827;
  --text2:#374151;
  --muted:#6B7280;
  --muted2:#9ca3af;
}
*,*::before,*::after{box-sizing:border-box;}
html,body,[class*="css"]{
  background-color:var(--bg)!important;
  color:var(--text)!important;
  font-family:'Inter',sans-serif;
}
.qh-header{
  padding:2rem 0 1.4rem;
  border-bottom:1px solid var(--border);
  margin-bottom:1.8rem;
}
.qh-title{
  font-family:'Inter',sans-serif;
  font-size:1.9rem;
  font-weight:700;
  letter-spacing:-0.5px;
  color:var(--text);
  margin:0;
  display:flex;
  align-items:center;
  gap:10px;
}
.qh-title .atom{
  font-size:1.7rem;
}
.qh-title .highlight{
  color:var(--accent);
}
.qh-sub{
  font-family:'JetBrains Mono',monospace;
  font-size:0.68rem;
  color:var(--muted);
  letter-spacing:0.5px;
  margin-top:6px;
}
.badge{
  display:inline-block;
  background:var(--accent-light);
  color:var(--accent-dark);
  border-radius:6px;
  font-size:0.65rem;
  font-family:'JetBrains Mono',monospace;
  padding:3px 10px;
  letter-spacing:0.3px;
  margin-right:6px;
  margin-top:8px;
  font-weight:500;
}
.m-grid{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:1.5rem;}
.m-card{
  background:var(--surface);
  border:1px solid var(--border);
  border-radius:12px;
  padding:1rem 1.1rem;
  flex:1;
  min-width:130px;
  box-shadow:0 1px 2px rgba(0,0,0,0.04);
  transition:box-shadow .15s;
}
.m-card:hover{box-shadow:0 4px 12px rgba(0,0,0,0.08);}
.m-label{
  font-family:'JetBrains Mono',monospace;
  font-size:0.6rem;
  color:var(--muted);
  text-transform:uppercase;
  letter-spacing:0.8px;
  margin-bottom:6px;
  font-weight:500;
}
.m-val{
  font-family:'Inter',sans-serif;
  font-size:1.45rem;
  font-weight:700;
  line-height:1;
  color:var(--text);
}
.m-val.c-a { color:var(--positive); }
.m-val.c-a2{ color:var(--accent); }
.m-val.c-a3{ color:var(--negative); }
.m-val.c-g { color:var(--warning); }
.m-delta{font-family:'JetBrains Mono',monospace;font-size:.62rem;color:var(--muted2);margin-top:5px;}
.sec{
  font-size:.72rem;
  font-weight:600;
  color:var(--text2);
  text-transform:uppercase;
  letter-spacing:1px;
  border-left:3px solid var(--accent);
  padding-left:.75rem;
  margin:1.8rem 0 .75rem;
  font-family:'JetBrains Mono',monospace;
}
.ibox{
  background:var(--surface);
  border:1px solid var(--border);
  border-left:3px solid var(--accent);
  border-radius:8px;
  padding:.9rem 1.1rem;
  font-size:.83rem;
  line-height:1.8;
  color:var(--text2);
  margin-bottom:1rem;
}
.ibox strong{color:var(--text);}
.hl {color:var(--positive);font-weight:600;}
.hlw{color:var(--negative);font-weight:600;}
.hlg{color:var(--warning);font-weight:600;}
[data-testid="stSidebar"]{
  background-color:var(--surface)!important;
  border-right:1px solid var(--border)!important;
}
[data-testid="stSidebar"] label{
  color:var(--muted)!important;
  font-family:'JetBrains Mono',monospace!important;
  font-size:.67rem!important;
  text-transform:uppercase!important;
  letter-spacing:0.6px!important;
}
[data-testid="stSidebar"] h2{
  color:var(--text)!important;
  font-family:'Inter',sans-serif!important;
  font-size:.8rem!important;
  font-weight:700!important;
  letter-spacing:0!important;
  text-transform:none!important;
}
.stTabs [data-baseweb="tab-list"]{
  background-color:var(--surface2)!important;
  border-radius:10px;
  gap:2px;
  padding:3px;
  border:1px solid var(--border)!important;
}
.stTabs [data-baseweb="tab"]{
  font-family:'Inter',sans-serif!important;
  font-size:.75rem!important;
  font-weight:500!important;
  color:var(--muted)!important;
  background:transparent!important;
  border-radius:8px!important;
  padding:6px 12px!important;
}
.stTabs [aria-selected="true"]{
  background:var(--surface)!important;
  color:var(--accent)!important;
  font-weight:600!important;
  box-shadow:0 1px 3px rgba(0,0,0,0.08)!important;
}
.idle-screen{
  text-align:center;
  padding:4rem 2rem;
  background:var(--surface);
  border:1px solid var(--border);
  border-radius:16px;
  margin-top:1rem;
}
.idle-screen .big{font-size:2.8rem;margin-bottom:1rem;}
.idle-screen p{font-size:.88rem;line-height:2.1;color:var(--muted);}
.idle-screen strong{color:var(--text);}
[data-testid="stDataFrame"]{border:1px solid var(--border);border-radius:8px;}
#MainMenu,footer{visibility:hidden;}
header[data-testid="stHeader"]{background:transparent;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
# ── Comprehensive stock universe ──────────────────────────────────────────────
# Grouped by sector/category for easy navigation in the searchable dropdown.
STOCK_UNIVERSE = {
    # ── US Large Cap Tech ──────────────────────────────────────────────────────
    "Apple (AAPL)":              "AAPL",
    "Microsoft (MSFT)":          "MSFT",
    "NVIDIA (NVDA)":             "NVDA",
    "Alphabet A (GOOGL)":        "GOOGL",
    "Alphabet C (GOOG)":         "GOOG",
    "Meta Platforms (META)":     "META",
    "Amazon (AMZN)":             "AMZN",
    "Tesla (TSLA)":              "TSLA",
    "Broadcom (AVGO)":           "AVGO",
    "AMD (AMD)":                 "AMD",
    "Intel (INTC)":              "INTC",
    "Qualcomm (QCOM)":           "QCOM",
    "Texas Instruments (TXN)":   "TXN",
    "Salesforce (CRM)":          "CRM",
    "Oracle (ORCL)":             "ORCL",
    "Adobe (ADBE)":              "ADBE",
    "ServiceNow (NOW)":          "NOW",
    "Palo Alto Networks (PANW)": "PANW",
    "Snowflake (SNOW)":          "SNOW",
    "Palantir (PLTR)":           "PLTR",
    "Cloudflare (NET)":          "NET",
    "Datadog (DDOG)":            "DDOG",
    "CrowdStrike (CRWD)":        "CRWD",
    "Workday (WDAY)":            "WDAY",
    "Autodesk (ADSK)":           "ADSK",
    "Intuit (INTU)":             "INTU",
    "Fortinet (FTNT)":           "FTNT",
    "Marvell Tech (MRVL)":       "MRVL",
    "Micron (MU)":               "MU",
    "Applied Materials (AMAT)":  "AMAT",
    "ASML (ASML)":               "ASML",
    "Taiwan Semi (TSM)":         "TSM",
    "Samsung (SSNLF)":           "SSNLF",
    "Shopify (SHOP)":            "SHOP",
    "Uber (UBER)":               "UBER",
    "Airbnb (ABNB)":             "ABNB",
    "DoorDash (DASH)":           "DASH",
    "Spotify (SPOT)":            "SPOT",
    "Netflix (NFLX)":            "NFLX",
    # ── US Financials ─────────────────────────────────────────────────────────
    "JPMorgan Chase (JPM)":      "JPM",
    "Goldman Sachs (GS)":        "GS",
    "Morgan Stanley (MS)":       "MS",
    "Bank of America (BAC)":     "BAC",
    "Wells Fargo (WFC)":         "WFC",
    "Citigroup (C)":             "C",
    "BlackRock (BLK)":           "BLK",
    "Visa (V)":                  "V",
    "Mastercard (MA)":           "MA",
    "American Express (AXP)":    "AXP",
    "PayPal (PYPL)":             "PYPL",
    "Block (SQ)":                "SQ",
    "Charles Schwab (SCHW)":     "SCHW",
    "Berkshire Hathaway B (BRK-B)": "BRK-B",
    # ── US Healthcare ─────────────────────────────────────────────────────────
    "Johnson & Johnson (JNJ)":   "JNJ",
    "UnitedHealth (UNH)":        "UNH",
    "Eli Lilly (LLY)":           "LLY",
    "AbbVie (ABBV)":             "ABBV",
    "Pfizer (PFE)":              "PFE",
    "Merck (MRK)":               "MRK",
    "Bristol-Myers (BMY)":       "BMY",
    "Moderna (MRNA)":            "MRNA",
    "Gilead Sciences (GILD)":    "GILD",
    "Amgen (AMGN)":              "AMGN",
    "Intuitive Surgical (ISRG)": "ISRG",
    # ── US Consumer / Retail ──────────────────────────────────────────────────
    "Walmart (WMT)":             "WMT",
    "Costco (COST)":             "COST",
    "Home Depot (HD)":           "HD",
    "Nike (NKE)":                "NKE",
    "McDonald's (MCD)":          "MCD",
    "Starbucks (SBUX)":          "SBUX",
    "Procter & Gamble (PG)":     "PG",
    "Coca-Cola (KO)":            "KO",
    "PepsiCo (PEP)":             "PEP",
    "Chipotle (CMG)":            "CMG",
    "Lululemon (LULU)":          "LULU",
    "Target (TGT)":              "TGT",
    # ── US Energy ─────────────────────────────────────────────────────────────
    "ExxonMobil (XOM)":          "XOM",
    "Chevron (CVX)":             "CVX",
    "ConocoPhillips (COP)":      "COP",
    "NextEra Energy (NEE)":      "NEE",
    # ── US Industrials / Defence ──────────────────────────────────────────────
    "Boeing (BA)":               "BA",
    "Caterpillar (CAT)":         "CAT",
    "Lockheed Martin (LMT)":     "LMT",
    "RTX Corp (RTX)":            "RTX",
    "Deere & Co (DE)":           "DE",
    "Honeywell (HON)":           "HON",
    "GE Aerospace (GE)":         "GE",
    # ── US Real Estate / REITs ────────────────────────────────────────────────
    "American Tower (AMT)":      "AMT",
    "Prologis (PLD)":            "PLD",
    "Equinix (EQIX)":            "EQIX",
    # ── Crypto-adjacent ───────────────────────────────────────────────────────
    "Coinbase (COIN)":           "COIN",
    "MicroStrategy (MSTR)":      "MSTR",
    "Marathon Digital (MARA)":   "MARA",
    "Riot Platforms (RIOT)":     "RIOT",
    # ── ETFs ──────────────────────────────────────────────────────────────────
    "S&P 500 ETF (SPY)":         "SPY",
    "Nasdaq 100 ETF (QQQ)":      "QQQ",
    "Total Market ETF (VTI)":    "VTI",
    "Dow Jones ETF (DIA)":       "DIA",
    "Small Cap ETF (IWM)":       "IWM",
    "Growth ETF (VUG)":          "VUG",
    "Value ETF (VTV)":           "VTV",
    "Tech Sector ETF (XLK)":     "XLK",
    "Financial Sector ETF (XLF)":"XLF",
    "Healthcare ETF (XLV)":      "XLV",
    "Energy Sector ETF (XLE)":   "XLE",
    "Bitcoin ETF (IBIT)":        "IBIT",
    "Gold ETF (GLD)":            "GLD",
    "Silver ETF (SLV)":          "SLV",
    "Bond ETF 20Y (TLT)":        "TLT",
    "Inverse S&P (SH)":          "SH",
    "VIX ETF (VIXY)":            "VIXY",
    # ── International / ADRs ──────────────────────────────────────────────────
    "Alibaba (BABA)":            "BABA",
    "Baidu (BIDU)":              "BIDU",
    "JD.com (JD)":               "JD",
    "NIO (NIO)":                 "NIO",
    "Toyota (TM)":               "TM",
    "Sony (SONY)":               "SONY",
    "LVMH (LVMUY)":              "LVMUY",
    "Novo Nordisk (NVO)":        "NVO",
    "SAP (SAP)":                 "SAP",
    "Shell (SHEL)":              "SHEL",
    "BP (BP)":                   "BP",
    "HSBC (HSBC)":               "HSBC",
    # ── Custom ────────────────────────────────────────────────────────────────
    "✏ Custom ticker…":          "CUSTOM",
}

# Flat list of all tickers for the portfolio multiselect
ALL_TICKERS = [v for v in STOCK_UNIVERSE.values() if v != "CUSTOM"]

# 8 Hilbert dimensions — financial features
FEATURES = [
    "return_1d", "volatility_20d", "rsi_14",
    "macd_signal", "momentum_10d", "volume_ratio",
    "bb_position", "drawdown",
]
LABELS = [
    "Return 1D", "Volatility 20D", "RSI 14",
    "MACD Signal", "Momentum 10D", "Volume Ratio",
    "BB Position", "Drawdown",
]

PTHEME = dict(
    paper_bgcolor="#F8FAFC", plot_bgcolor="#ffffff",
    font=dict(family="Inter, sans-serif", color="#111827", size=12),
    xaxis=dict(gridcolor="#E5E7EB", linecolor="#D1D5DB", showgrid=True, zeroline=False),
    yaxis=dict(gridcolor="#E5E7EB", linecolor="#D1D5DB", showgrid=True, zeroline=False),
    margin=dict(l=55, r=25, t=50, b=45),
    legend=dict(bgcolor="#ffffff", bordercolor="#E5E7EB", borderwidth=1,
                font=dict(color="#111827", size=11)),
)

# ── Paul Tol colorblind-safe palette ──────────────────────────────────────────
# Works for deuteranopia, protanopia, and tritanopia.
CB_BLUE    = "#0077BB"   # primary / nominal
CB_CYAN    = "#33BBEE"   # secondary / MA50 / PC2
CB_TEAL    = "#009988"   # positive / gain
CB_ORANGE  = "#EE7733"   # warning / caution
CB_RED     = "#CC3311"   # negative / alert / anomaly
CB_MAGENTA = "#EE3377"   # highlight / special point
CB_GREY    = "#BBBBBB"   # neutral / background traces
CB_NAVY    = "#003366"   # dark accent

# 8-color sequence safe for all common colour-vision deficiencies
CB_SEQ = [CB_BLUE, CB_CYAN, CB_TEAL, CB_ORANGE, CB_RED, CB_MAGENTA, CB_GREY, CB_NAVY]

# Colorblind-safe diverging scale (blue → white → red/orange)
CB_DIVERGE = [[0.0, CB_RED], [0.5, "#F8FAFC"], [1.0, CB_BLUE]]

# Sequential (low→high) scale
CB_SEQ_SCALE = [[0.0, "#F8FAFC"], [0.35, CB_CYAN], [0.7, CB_BLUE], [1.0, CB_NAVY]]

# Sharpe colorscale (low=grey → high=teal/navy)
CB_SHARPE_SCALE = [
    [0.0,  CB_GREY],
    [0.35, CB_CYAN],
    [0.65, CB_BLUE],
    [1.0,  CB_NAVY],
]

# ─────────────────────────────────────────────────────────────────────────────
# DATA: FETCH + FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_stock(ticker: str, period: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    close  = df["Close"].squeeze()
    volume = df["Volume"].squeeze()

    feat = pd.DataFrame(index=close.index)

    # ψ₁ — 1-day return
    feat["return_1d"] = close.pct_change()

    # ψ₂ — 20-day rolling volatility (annualised)
    feat["volatility_20d"] = close.pct_change().rolling(20).std() * np.sqrt(252)

    # ψ₃ — RSI 14
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-9)
    feat["rsi_14"] = 100 - (100 / (1 + rs))

    # ψ₄ — MACD signal line diff (12/26/9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    feat["macd_signal"] = macd - signal

    # ψ₅ — 10-day price momentum
    feat["momentum_10d"] = close.pct_change(10)

    # ψ₆ — Volume ratio vs 20-day avg
    feat["volume_ratio"] = volume / (volume.rolling(20).mean() + 1e-9)

    # ψ₇ — Bollinger band position [-1, +1]
    sma20  = close.rolling(20).mean()
    std20  = close.rolling(20).std()
    feat["bb_position"] = (close - sma20) / (2 * std20 + 1e-9)

    # ψ₈ — Rolling max drawdown (60-day window)
    roll_max = close.rolling(60, min_periods=1).max()
    feat["drawdown"] = (close - roll_max) / (roll_max + 1e-9)

    return feat.dropna()

# ─────────────────────────────────────────────────────────────────────────────
# MONTE CARLO — GBM in Hilbert Space
# ─────────────────────────────────────────────────────────────────────────────
def run_monte_carlo(close: pd.Series, n_paths: int, horizon: int, seed: int):
    rng    = np.random.default_rng(seed)
    ret    = close.pct_change().dropna()
    mu     = float(ret.mean())
    sigma  = float(ret.std())
    S0     = float(close.iloc[-1])
    dt     = 1 / 252

    paths  = np.zeros((n_paths, horizon + 1))
    paths[:, 0] = S0
    for t in range(1, horizon + 1):
        z = rng.standard_normal(n_paths)
        paths[:, t] = paths[:, t-1] * np.exp(
            (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
        )
    return paths, mu, sigma

# ─────────────────────────────────────────────────────────────────────────────
# QUANTUM MATH
# ─────────────────────────────────────────────────────────────────────────────
def build_states(feat_df: pd.DataFrame):
    X      = feat_df[FEATURES].values.astype(float)
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    norms  = np.linalg.norm(X_sc, axis=1, keepdims=True)
    norms  = np.where(norms < 1e-10, 1.0, norms)
    return X_sc, X_sc / norms, scaler

def density_matrix(states):
    N = len(states)
    return np.einsum("ni,nj->ij", states, states) / N

def vn_entropy(rho):
    eigs = np.linalg.eigvalsh(rho)
    eigs = eigs[eigs > 1e-12]
    return float(-np.dot(eigs, np.log(eigs)))

def rolling_observables(states, window=30):
    n, d  = states.shape
    coherence = np.zeros(n)
    entropy   = np.zeros(n)
    stability = np.zeros(n)
    purity    = np.zeros(n)
    mask = ~np.eye(d, dtype=bool)
    for i in range(n):
        w = states[max(0, i - window): i + 1]
        if len(w) < 2:
            continue
        rho          = density_matrix(w)
        coherence[i] = float(np.mean(np.abs(rho[mask])))
        entropy[i]   = vn_entropy(rho)
        stability[i] = 1.0 / (1.0 + float(np.mean(np.var(w, axis=0))))
        purity[i]    = float(np.trace(rho @ rho))
    return coherence, entropy, stability, purity

def manifold_fit(states_sc, train_frac):
    n_tr    = max(30, int(len(states_sc) * train_frac))
    tr      = states_sc[:n_tr]
    mu      = tr.mean(axis=0)
    cov     = np.cov(tr.T) + 1e-6 * np.eye(tr.shape[1])
    try:    cov_inv = np.linalg.inv(cov)
    except: cov_inv = np.eye(tr.shape[1])
    diffs   = states_sc - mu
    dists   = np.sqrt(np.maximum(0., np.einsum("ni,ij,nj->n", diffs, cov_inv, diffs)))
    return dists, mu, cov

def pca_proj(states, k):
    pca = PCA(n_components=k)
    return pca.fit_transform(states), pca

def shade_anomalies(fig, pred, ts, row=None, col=None):
    kwargs = {}
    if row is not None: kwargs["row"] = row
    if col is not None: kwargs["col"] = col
    in_b = False; t0 = None
    for flag, t in zip(pred, ts):
        if flag and not in_b:
            in_b = True; t0 = t
        elif not flag and in_b:
            in_b = False
            fig.add_vrect(x0=t0, x1=t, fillcolor="rgba(204,51,17,0.07)",
                          line_width=0, **kwargs)
    if in_b:
        fig.add_vrect(x0=t0, x1=ts.iloc[-1],
                      fillcolor="rgba(204,51,17,0.07)", line_width=0, **kwargs)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚛ HILBERT LAB")
    st.markdown("---")

    # ── Stock selector — full universe, searchable ─────────────────────────
    stock_choice = st.selectbox(
        "Stock",
        options=list(STOCK_UNIVERSE.keys()),
        index=0,
        help="Type to search across 100+ stocks, ETFs & ADRs",
    )
    if STOCK_UNIVERSE[stock_choice] == "CUSTOM":
        ticker = st.text_input("Ticker Symbol", value="AAPL",
                               max_chars=12).upper().strip()
    else:
        ticker = STOCK_UNIVERSE[stock_choice]

    period = st.select_slider("Historical Period",
                               options=["1y", "2y", "3y", "5y"], value="3y")

    st.markdown("---")
    st.markdown("## 🌐 Manifold Config")
    train_frac   = st.slider("Training Regime Fraction", 0.15, 0.75, 0.50, 0.05,
                              help="First X% used to define the natural market manifold")
    sigma_thresh = st.slider("Manifold Boundary σ", 1.0, 6.0, 3.0, 0.1)
    obs_win      = st.slider("Observable Window", 10, 60, 20, 5)

    st.markdown("---")
    st.markdown("## 💹 Monte Carlo")
    n_paths  = st.slider("Simulation Paths", 100, 3000, 1000, 100)
    horizon  = st.slider("Forecast Horizon (days)", 30, 365, 252, 30)
    mc_seed  = st.number_input("Random Seed", 0, 9999, 42, step=1)

    st.markdown("---")
    st.markdown("## 🌌 Portfolio Optimizer")

    # Default selection: active ticker + a sensible starter basket
    _starter = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "JPM", "V", "UNH"]
    _pre_select = list(dict.fromkeys([ticker] + _starter))[:6]

    port_tickers_sel = st.multiselect(
        "Portfolio Tickers",
        options=ALL_TICKERS,
        default=[t for t in _pre_select if t in ALL_TICKERS],
        help="Search & select 2–10 tickers from the full universe. Active stock is always included.",
    )

    # Custom ticker entry for anything not in the universe
    _custom_extra = st.text_input(
        "Add tickers not in the list",
        value="",
        placeholder="e.g. BRK-B, ARM, TSMC",
    )
    if _custom_extra.strip():
        _extras = [t.strip().upper() for t in _custom_extra.split(",") if t.strip()]
        port_tickers_sel = list(dict.fromkeys(port_tickers_sel + _extras))

    # Always ensure the active ticker is included
    if ticker not in port_tickers_sel:
        port_tickers_sel = [ticker] + port_tickers_sel

    port_tickers_raw = ", ".join(port_tickers_sel)

    port_n_sim  = st.slider("Frontier Simulations", 1000, 15000, 6000, 1000)
    risk_free   = st.number_input("Risk-Free Rate (%)", 0.0, 10.0, 5.0, 0.25) / 100

    st.markdown("---")
    run_btn = st.button("⚛  RUN ANALYSIS", use_container_width=True, type="primary")

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="qh-header">
  <p class="qh-title">⚛ Quantum Hilbert <span>Space Lab</span></p>
  <p class="qh-sub">8D Market State Space · Manifold Monitoring · Monte Carlo · Density Matrix · Portfolio Optimizer</p>
  <div style="margin-top:10px">
    <span class="badge">8D Feature Space</span>
    <span class="badge">Manifold Detection</span>
    <span class="badge">Monte Carlo GBM</span>
    <span class="badge">Density Matrix</span>
  </div>
</div>
""", unsafe_allow_html=True)

if not run_btn:
    st.markdown("""
    <div class="idle-screen">
      <div class="big">⚛</div>
      <p>
        Most dashboards show you <strong>what already happened.</strong><br>
        This one shows you <strong>how a stock behaves in latent space.</strong><br><br>
        Eight market signals — price, volatility, RSI, MACD, momentum,<br>
        volume, Bollinger bands, drawdown — become a single<br>
        <strong>8-dimensional state vector |ψ_t⟩.</strong><br><br>
        Instead of asking <em>"Is RSI too high?"</em><br>
        You ask: <strong>"Did the stock leave its natural manifold?"</strong><br><br>
        👈 Select a stock in the sidebar → click <strong>⚛ Run Analysis</strong>
      </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# LOAD + COMPUTE
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner(f"Fetching {ticker} and building quantum state vectors…"):
    try:
        df_raw = fetch_stock(ticker, period)
        if df_raw.empty:
            st.error(f"No data found for **{ticker}**. Check the symbol.")
            st.stop()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        st.stop()

    feat_df = build_features(df_raw)
    close   = df_raw["Close"].squeeze().loc[feat_df.index]
    ts      = feat_df.index

    states_sc, states_norm, scaler = build_states(feat_df)
    proj3, pca3 = pca_proj(states_sc, 3)
    proj2, pca2 = pca_proj(states_sc, 2)
    proj8, pca8 = pca_proj(states_sc, min(8, states_sc.shape[1]))

    d_arr, man_mu, man_cov = manifold_fit(states_sc, train_frac)
    coherence, entropy, stability, purity = rolling_observables(states_norm, obs_win)

    pred_anom  = d_arr > sigma_thresh
    n_detected = int(pred_anom.sum())
    recent_d   = float(d_arr[-30:].mean())
    sys_state  = "ANOMALOUS" if recent_d > sigma_thresh else "NOMINAL"
    ts_str     = pd.Series(ts).dt.strftime("%Y-%m-%d").values
    ev_sum     = float(pca3.explained_variance_ratio_.sum() * 100)

    # Price stats
    last_price = float(close.iloc[-1])
    prev_price = float(close.iloc[-2])
    pct_chg    = (last_price - prev_price) / prev_price * 100
    ann_ret    = float(close.pct_change().mean() * 252 * 100)
    ann_vol    = float(close.pct_change().std() * np.sqrt(252) * 100)

    # Monte Carlo
    mc_paths, mc_mu, mc_sigma = run_monte_carlo(close, n_paths, horizon, int(mc_seed))
    end_prices  = mc_paths[:, -1]
    prob_above  = float((end_prices > last_price).mean() * 100)
    mc_expected = float(end_prices.mean())
    mc_median   = float(np.percentile(end_prices, 50))
    var_95      = float(np.percentile(end_prices, 5))

    # Fetch info
    try:
        info      = yf.Ticker(ticker).info
        comp_name = info.get("longName", ticker)
        sector    = info.get("sector", "—")
    except Exception:
        comp_name = ticker
        sector    = "—"

# ─────────────────────────────────────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────────────────────────────────────
vc = "c-a3" if sys_state == "ANOMALOUS" else "c-a"
dc = "c-a3" if n_detected > 0 else "c-a"
rc = "c-a3" if recent_d > sigma_thresh else "c-g"
pc = "c-a3" if pct_chg < 0 else "c-a"

st.markdown(f"### {comp_name} &nbsp;<code style='font-size:.8rem;background:#f0ede8;padding:3px 8px;border-radius:4px'>{ticker}</code>", unsafe_allow_html=True)
st.markdown(f"<span style='font-family:JetBrains Mono;font-size:.72rem;color:#374151'>{sector}</span>", unsafe_allow_html=True)

st.markdown(f"""
<div class="m-grid">
  <div class="m-card"><div class="m-label">Last Price</div>
    <div class="m-val">${last_price:.2f}</div>
    <div class="m-delta">USD</div></div>
  <div class="m-card"><div class="m-label">Day Change</div>
    <div class="m-val {pc}">{pct_chg:+.2f}%</div>
    <div class="m-delta">vs prev close</div></div>
  <div class="m-card"><div class="m-label">Manifold State</div>
    <div class="m-val {vc}">{sys_state}</div>
    <div class="m-delta">8D geometry</div></div>
  <div class="m-card"><div class="m-label">Deviations Detected</div>
    <div class="m-val {dc}">{n_detected}</div>
    <div class="m-delta">σ &gt; {sigma_thresh:.1f}</div></div>
  <div class="m-card"><div class="m-label">Recent Distance</div>
    <div class="m-val {rc}">{recent_d:.2f}σ</div>
    <div class="m-delta">Mahalanobis (30d)</div></div>
  <div class="m-card"><div class="m-label">Ann. Return</div>
    <div class="m-val {'c-a' if ann_ret >= 0 else 'c-a3'}">{ann_ret:.1f}%</div>
    <div class="m-delta">Historical</div></div>
  <div class="m-card"><div class="m-label">Ann. Volatility</div>
    <div class="m-val c-g">{ann_vol:.1f}%</div>
    <div class="m-delta">1σ annual</div></div>
  <div class="m-card"><div class="m-label">vN Entropy S(ρ)</div>
    <div class="m-val c-a2">{entropy[-1]:.3f}</div>
    <div class="m-delta">State disorder</div></div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📈 Price & Monte Carlo",
    "⚛ Hilbert Space",
    "📡 Observables",
    "🌀 State Evolution",
    "🗺 Manifold Map",
    "🧬 Density Matrix",
    "🌌 Portfolio Optimizer",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PRICE + MONTE CARLO
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="sec">Historical Price with Manifold Anomalies</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="ibox">
    Standard price chart — but <span class="hlw">pink shading</span> marks periods where
    the stock's 8D state vector deviated beyond the natural market manifold.
    These are not just price spikes — they are <strong>full-regime departures</strong>
    across all 8 financial dimensions simultaneously.
    </div>""", unsafe_allow_html=True)

    close_plot = df_raw["Close"].squeeze()
    ma50  = close_plot.rolling(50).mean()
    ma200 = close_plot.rolling(200).mean()

    fig_price = go.Figure()
    shade_anomalies(fig_price, pred_anom, pd.Series(ts))
    fig_price.add_trace(go.Scatter(
        x=close_plot.index, y=close_plot,
        line=dict(color=CB_BLUE, width=1.8),
        fill="tozeroy", fillcolor="rgba(0,119,187,0.04)",
        name="Close",
    ))
    fig_price.add_trace(go.Scatter(x=ma50.index, y=ma50,
        line=dict(color=CB_CYAN, width=1, dash="dot"), name="MA 50"))
    fig_price.add_trace(go.Scatter(x=ma200.index, y=ma200,
        line=dict(color=CB_RED, width=1, dash="dot"), name="MA 200"))
    fig_price.update_layout(**PTHEME, height=380,
        title=dict(text=f"{ticker} — Adjusted Close  |  Pink = Manifold Deviation",
                   font=dict(color="#374151", size=12, family="Inter")),
        xaxis_title="Date", yaxis_title="Price ($)")
    st.plotly_chart(fig_price, use_container_width=True)

    # Volume
    vol_colors = [CB_BLUE if r >= 0 else CB_RED
                  for r in df_raw["Close"].squeeze().pct_change().fillna(0)]
    fig_vol = go.Figure(go.Bar(
        x=df_raw.index, y=df_raw["Volume"].squeeze(),
        marker_color=vol_colors, name="Volume",
    ))
    fig_vol.update_layout(**PTHEME, height=180,
        title=dict(text="Volume", font=dict(color="#374151", size=12, family="Inter")))
    st.plotly_chart(fig_vol, use_container_width=True)

    # ── MONTE CARLO ──
    st.markdown('<div class="sec">Monte Carlo Simulation (GBM) — Hilbert-Informed</div>',
                unsafe_allow_html=True)
    st.markdown(f"""
    <div class="ibox">
    Standard GBM simulation seeded with the stock's historical drift (μ={mc_mu*252*100:.1f}%/yr)
    and volatility (σ={mc_sigma*np.sqrt(252)*100:.1f}%/yr).
    The <span class="hlw">manifold state</span> contextualises which simulated paths are
    consistent with the stock's current regime — <strong>{sys_state}</strong>.
    </div>""", unsafe_allow_html=True)

    t_axis  = np.arange(horizon + 1)
    pct5    = np.percentile(mc_paths, 5,  axis=0)
    pct25   = np.percentile(mc_paths, 25, axis=0)
    pct50   = np.percentile(mc_paths, 50, axis=0)
    pct75   = np.percentile(mc_paths, 75, axis=0)
    pct95   = np.percentile(mc_paths, 95, axis=0)

    fig_mc = go.Figure()
    rng2   = np.random.default_rng(int(mc_seed) + 1)
    sample = rng2.choice(n_paths, min(80, n_paths), replace=False)
    for idx in sample:
        fig_mc.add_trace(go.Scatter(
            x=t_axis, y=mc_paths[idx],
            line=dict(color="rgba(0,119,187,0.06)", width=0.8),
            showlegend=False, hoverinfo="skip",
        ))
    fig_mc.add_trace(go.Scatter(
        x=np.concatenate([t_axis, t_axis[::-1]]),
        y=np.concatenate([pct95, pct5[::-1]]),
        fill="toself", fillcolor="rgba(0,119,187,0.05)",
        line=dict(color="rgba(0,0,0,0)"), name="5–95%"))
    fig_mc.add_trace(go.Scatter(
        x=np.concatenate([t_axis, t_axis[::-1]]),
        y=np.concatenate([pct75, pct25[::-1]]),
        fill="toself", fillcolor="rgba(0,119,187,0.06)",
        line=dict(color="rgba(0,0,0,0)"), name="25–75%"))
    fig_mc.add_trace(go.Scatter(x=t_axis, y=pct50,
        line=dict(color=CB_BLUE, width=2), name="Median"))
    fig_mc.add_trace(go.Scatter(x=t_axis, y=pct5,
        line=dict(color=CB_RED, width=1, dash="dash"), name="5th pct"))
    fig_mc.add_trace(go.Scatter(x=t_axis, y=pct95,
        line=dict(color=CB_CYAN, width=1, dash="dash"), name="95th pct"))
    fig_mc.update_layout(**PTHEME, height=420,
        title=dict(text=f"Monte Carlo GBM — {n_paths} paths, {horizon}d horizon",
                   font=dict(color="#374151", size=12, family="Inter")),
        xaxis_title="Trading Days", yaxis_title="Price ($)")
    st.plotly_chart(fig_mc, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Expected Price", f"${mc_expected:.2f}",
              f"{(mc_expected/last_price-1)*100:+.1f}%")
    c2.metric("Median Price", f"${mc_median:.2f}",
              f"{(mc_median/last_price-1)*100:+.1f}%")
    c3.metric("Prob. Above Current", f"{prob_above:.1f}%")
    c4.metric("VaR 95%", f"${var_95:.2f}",
              f"{(var_95/last_price-1)*100:+.1f}%")

    # Return distribution
    st.markdown('<div class="sec">Return Distribution</div>', unsafe_allow_html=True)
    ret_dist = (end_prices / last_price - 1) * 100
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(x=ret_dist, nbinsx=80,
        marker_color=CB_BLUE, opacity=0.8, name="Returns"))
    fig_dist.add_vline(x=0, line_dash="dash", line_color=CB_RED,
                       annotation_text="Break-even",
                       annotation_font=dict(color=CB_RED, size=10))
    fig_dist.add_vline(x=ret_dist.mean(), line_dash="dot", line_color=CB_BLUE,
                       annotation_text=f"Mean {ret_dist.mean():.1f}%",
                       annotation_font=dict(color=CB_BLUE, size=10))
    fig_dist.update_layout(**PTHEME, height=280,
        title=dict(text="1-Year Return Distribution (Monte Carlo)",
                   font=dict(color="#374151", size=12, family="Inter")),
        xaxis_title="Return (%)", yaxis_title="Frequency")
    st.plotly_chart(fig_dist, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — HILBERT SPACE 3D
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="sec">Market State Trajectory in 8D Hilbert Space</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="ibox">
    The 8 financial features (return, volatility, RSI, MACD, momentum, volume, BB, drawdown)
    are combined into a single state vector <strong>|ψ_t⟩</strong> and projected into 3D.
    Each point is one trading day. The <span class="hl">path is the stock's market history as geometry.</span>
    Tightly coiled = stable regime. Wide excursions = regime departure.
    <span class="hlw">Pink diamonds</span> = manifold deviations.
    </div>""", unsafe_allow_html=True)

    ni = np.where(~pred_anom)[0]
    ai = np.where( pred_anom)[0]

    fig3d = go.Figure()
    fig3d.add_trace(go.Scatter3d(
        x=proj3[ni,0], y=proj3[ni,1], z=proj3[ni,2], mode="markers",
        marker=dict(
            size=2.5, color=d_arr[ni],
            colorscale=[[0,CB_BLUE],[0.4,CB_CYAN],[0.75,CB_ORANGE],[1,CB_RED]],
            cmin=0, cmax=sigma_thresh*1.8,
            colorbar=dict(title="Mahal σ", thickness=10, x=1.02, titlefont=dict(color="#374151")),
            opacity=0.65,
        ),
        name="Normal States",
        text=[f"{ts_str[i]}<br>σ={d_arr[i]:.2f}" for i in ni],
        hovertemplate="%{text}<extra></extra>",
    ))
    if len(ai) > 0:
        fig3d.add_trace(go.Scatter3d(
            x=proj3[ai,0], y=proj3[ai,1], z=proj3[ai,2], mode="markers",
            marker=dict(size=7, color=CB_RED, symbol="diamond",
                        line=dict(color="#ffffff", width=0.8), opacity=0.95),
            name="Manifold Deviations",
            text=[f"{ts_str[i]}<br>σ={d_arr[i]:.2f}" for i in ai],
            hovertemplate="%{text}<extra></extra>",
        ))
    rn = min(250, len(proj3))
    fig3d.add_trace(go.Scatter3d(
        x=proj3[-rn:,0], y=proj3[-rn:,1], z=proj3[-rn:,2], mode="lines",
        line=dict(color="rgba(0,119,187,0.20)", width=1),
        name="Recent Path", hoverinfo="skip",
    ))
    pc_ev = pca3.explained_variance_ratio_ * 100
    fig3d.update_layout(
        paper_bgcolor="#ffffff",
        scene=dict(
            bgcolor="#ffffff",
            xaxis=dict(backgroundcolor="#ffffff", gridcolor="#E5E7EB",
                       title=f"PC₁ {pc_ev[0]:.1f}%"),
            yaxis=dict(backgroundcolor="#ffffff", gridcolor="#E5E7EB",
                       title=f"PC₂ {pc_ev[1]:.1f}%"),
            zaxis=dict(backgroundcolor="#ffffff", gridcolor="#E5E7EB",
                       title=f"PC₃ {pc_ev[2]:.1f}%"),
        ),
        font=dict(family="Inter", color="#374151", size=10),
        legend=dict(bgcolor="#ffffff", bordercolor="#E5E7EB", borderwidth=1, x=0, y=1),
        margin=dict(l=0, r=0, t=30, b=0), height=540,
        title=dict(text=f"{ticker} 8D Market State Space — 3D Projection",
                   font=dict(color="#374151", size=12, family="Inter")),
    )
    st.plotly_chart(fig3d, use_container_width=True)

    # Variance spectrum
    st.markdown('<div class="sec">Explained Variance Spectrum — 8 Financial Dimensions</div>',
                unsafe_allow_html=True)
    ev_r   = pca8.explained_variance_ratio_ * 100
    cum_ev = np.cumsum(ev_r)
    pc_lbl = [f"PC{i+1}" for i in range(len(ev_r))]
    fig_ev = make_subplots(specs=[[{"secondary_y": True}]])
    fig_ev.add_trace(go.Bar(
        x=pc_lbl, y=ev_r,
        marker_color=[CB_BLUE if i<3 else "#E5E7EB" for i in range(len(ev_r))],
        marker_line_color=CB_NAVY,
        marker_line_width=[2 if i<3 else 0.5 for i in range(len(ev_r))],
        name="Variance %",
    ), secondary_y=False)
    fig_ev.add_trace(go.Scatter(
        x=pc_lbl, y=cum_ev, mode="lines+markers",
        line=dict(color=CB_BLUE, width=2, dash="dot"),
        marker=dict(size=6, color=CB_BLUE), name="Cumulative %",
    ), secondary_y=True)
    fig_ev.update_layout(**PTHEME, height=260,
        title=dict(text="PCA Variance — How much each PC captures of the 8D market state",
                   font=dict(color="#374151", size=12, family="Inter")))
    fig_ev.update_yaxes(title_text="Variance (%)", secondary_y=False,
                         gridcolor="#E5E7EB", color="#6b7280")
    fig_ev.update_yaxes(title_text="Cumulative (%)", secondary_y=True,
                         showgrid=False, color=CB_CYAN)
    st.plotly_chart(fig_ev, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — OBSERVABLES
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="sec">Quantum Observable Measurements</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="ibox">
    Four <strong>operator expectation values</strong> measured on rolling windows of state vectors:<br>
    <span class="hl">⟨Stability⟩</span> — how consistent the 8D market state is &nbsp;|&nbsp;
    <span class="hl">⟨Coherence⟩</span> — feature cross-correlations &nbsp;|&nbsp;
    <span class="hlg">⟨Entropy⟩</span> — S(ρ) market uncertainty &nbsp;|&nbsp;
    <span class="hlw">⟨Purity⟩</span> — Tr(ρ²) regime concentration.<br>
    <span class="hlw">Pink shading</span> = manifold deviations (anomalous market regime).
    </div>""", unsafe_allow_html=True)

    fig_obs = make_subplots(rows=2, cols=2,
        subplot_titles=["⟨Stability⟩","⟨Coherence⟩","⟨Entropy⟩ S(ρ)","⟨Purity⟩ Tr(ρ²)"],
        vertical_spacing=0.16, horizontal_spacing=0.1)
    for (ser, color, fill, r, c) in [
        (stability,CB_BLUE,"rgba(0,153,136,0.08)",  1,1),
        (coherence,CB_CYAN,"rgba(51,187,238,0.07)", 1,2),
        (entropy,  CB_ORANGE,"rgba(238,119,51,0.08)",2,1),
        (purity,   CB_RED,"rgba(238,51,119,0.08)",2,2),
    ]:
        fig_obs.add_trace(go.Scatter(
            x=ts, y=ser, mode="lines",
            line=dict(color=color, width=1.4),
            fill="tozeroy", fillcolor=fill, showlegend=False,
        ), row=r, col=c)
        shade_anomalies(fig_obs, pred_anom, pd.Series(ts), row=r, col=c)
    fig_obs.update_layout(**PTHEME, height=520,
        title=dict(text="Market Observables — Rolling Window Analysis",
                   font=dict(color="#374151", size=12, family="Inter")))
    for r in range(1,3):
        for c in range(1,3):
            fig_obs.update_xaxes(gridcolor="#E5E7EB", linecolor="#E5E7EB", row=r, col=c)
            fig_obs.update_yaxes(gridcolor="#E5E7EB", linecolor="#E5E7EB", row=r, col=c)
    st.plotly_chart(fig_obs, use_container_width=True)

    # Correlation
    st.markdown('<div class="sec">Observable Correlation Matrix</div>',
                unsafe_allow_html=True)
    obs_df = pd.DataFrame({"Stability":stability,"Coherence":coherence,
                           "Entropy":entropy,"Purity":purity,"Mahal dist":d_arr})
    corr   = obs_df.corr().values
    clbls  = obs_df.columns.tolist()
    fig_corr = go.Figure(go.Heatmap(
        z=corr, x=clbls, y=clbls,
        colorscale=[[0,CB_RED],[0.5,"#f8f7f4"],[1,CB_BLUE]],
        zmin=-1, zmax=1,
        text=np.round(corr,2), texttemplate="%{text}",
        textfont=dict(size=11, color="#111827", family="Inter"),
        colorbar=dict(title="ρ", thickness=12),
    ))
    _ptheme_no_axes = {k:v for k,v in PTHEME.items() if k not in ("xaxis","yaxis")}
    fig_corr.update_layout(**_ptheme_no_axes, height=300,
        title=dict(text="Observable × Manifold Distance Correlation",
                   font=dict(color="#374151", size=12, family="Inter")),
        xaxis=dict(gridcolor="#E5E7EB", linecolor="#E5E7EB", showgrid=False),
        yaxis=dict(gridcolor="#E5E7EB", linecolor="#E5E7EB", showgrid=False))
    st.plotly_chart(fig_corr, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — STATE EVOLUTION
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="sec">8D Market State Heatmap</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="ibox">
    Each row is one financial dimension. Each column is one trading day.
    Watch for <span class="hlw">amplitude shifts propagating across multiple rows</span>
    simultaneously — those are regime changes the manifold detector catches
    before any single indicator would.
    </div>""", unsafe_allow_html=True)

    ds    = max(1, len(states_sc) // 500)
    sc_ds = states_sc[::ds].T
    ts_ds = pd.Series(ts[::ds]).dt.strftime("%Y-%m-%d").values

    # Clamp z to ±4σ so moderate signals have visible colour, not just outliers
    z_clamped = np.clip(sc_ds, -4, 4)

    fig_heat = go.Figure(go.Heatmap(
        z=z_clamped, x=ts_ds, y=LABELS,
        # High-contrast diverging scale: strong red → off-white → strong blue
        # No wide flat band — every σ step gets a distinct colour
        colorscale=[
            [0.00, "#7B0000"],   # deep red  (−4σ)
            [0.20, CB_RED],      # CB red    (−2.4σ)
            [0.40, "#FFCCBB"],   # pale red  (−0.8σ)
            [0.50, "#F0F0F0"],   # light grey midpoint (0)
            [0.60, "#BBDDFF"],   # pale blue (+0.8σ)
            [0.80, CB_BLUE],     # CB blue   (+2.4σ)
            [1.00, "#003366"],   # deep navy (+4σ)
        ],
        colorbar=dict(
            title="σ units",
            thickness=14,
            tickvals=[-4, -2, 0, 2, 4],
            ticktext=["-4σ", "-2σ", "0", "+2σ", "+4σ"],
            titlefont=dict(color="#374151", size=11),
            tickfont=dict(color="#111827", size=11),
            outlinecolor="#E5E7EB",
            outlinewidth=1,
        ),
        zmid=0, zmin=-4, zmax=4,
        hovertemplate="<b>%{y}</b><br>Date: %{x}<br>Value: %{z:.2f}σ<extra></extra>",
    ))
    _heat_theme = {k: v for k, v in PTHEME.items()
                   if k not in ("xaxis", "yaxis", "paper_bgcolor", "plot_bgcolor")}
    fig_heat.update_layout(
        **_heat_theme,
        paper_bgcolor="#F8FAFC",
        plot_bgcolor="#1E2433",
        height=360,
        title=dict(text="Market State Heatmap — 8 Financial Dimensions Over Time",
                   font=dict(color="#374151", size=12, family="Inter")),
        xaxis=dict(showgrid=False, linecolor="#374151", tickfont=dict(color="#374151"),
                   nticks=12, tickangle=-45),
        yaxis=dict(showgrid=False, linecolor="#374151", tickfont=dict(color="#374151"),
                   autorange="reversed"),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # Feature selector
    st.markdown('<div class="sec">Per-Feature Trajectories</div>', unsafe_allow_html=True)
    sel_dims = st.multiselect("Select features", options=LABELS, default=LABELS[:4])
    sel_idx  = [LABELS.index(d) for d in sel_dims]
    palette  = CB_SEQ
    if sel_idx:
        fig_traj = go.Figure()
        shade_anomalies(fig_traj, pred_anom, pd.Series(ts))
        for i, si in enumerate(sel_idx):
            fig_traj.add_trace(go.Scatter(
                x=ts, y=states_sc[:,si], mode="lines",
                line=dict(color=palette[i%len(palette)], width=1.3),
                name=LABELS[si],
            ))
        fig_traj.update_layout(**PTHEME, height=340,
            title=dict(text="Standardised Feature Trajectories (σ units)",
                       font=dict(color="#374151", size=12, family="Inter")),
            xaxis_title="Date", yaxis_title="Amplitude (σ)")
        st.plotly_chart(fig_traj, use_container_width=True)

    # Phase portrait
    st.markdown('<div class="sec">Phase Portrait — PC₁ × PC₂ (Time → Color)</div>',
                unsafe_allow_html=True)
    n_show  = min(600, len(proj2))
    t_color = np.linspace(0, 1, n_show)
    fig_ph  = go.Figure()
    fig_ph.add_trace(go.Scatter(
        x=proj2[-n_show:,0], y=proj2[-n_show:,1], mode="markers+lines",
        marker=dict(size=3.5, color=t_color,
                    colorscale=[[0,"#E5E7EB"],[0.45,CB_CYAN],[1,CB_NAVY]],
                    colorbar=dict(title="Time →", thickness=10),
                    showscale=True),
        line=dict(color="rgba(51,187,238,0.18)", width=0.7),
        showlegend=False,
    ))
    aw = pred_anom[-n_show:]
    if aw.any():
        fig_ph.add_trace(go.Scatter(
            x=proj2[-n_show:][aw,0], y=proj2[-n_show:][aw,1], mode="markers",
            marker=dict(size=9, color=CB_RED, symbol="x-thin",
                        line=dict(color=CB_RED, width=2)),
            name="Anomaly",
        ))
    pc2_ev = pca2.explained_variance_ratio_ * 100
    fig_ph.update_layout(**PTHEME, height=380,
        title=dict(text="Phase Portrait — Market State Trajectory",
                   font=dict(color="#374151", size=12, family="Inter")),
        xaxis_title=f"PC₁ ({pc2_ev[0]:.1f}%)",
        yaxis_title=f"PC₂ ({pc2_ev[1]:.1f}%)")
    st.plotly_chart(fig_ph, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — MANIFOLD MAP
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="sec">Mahalanobis Distance From Natural Market Manifold</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="ibox">
    The <span class="hl">Mahalanobis distance</span> measures how far each day's 8D state
    sits from the center of the normal market operating region, accounting for all
    cross-feature covariances. <br>
    Distance &gt; <span class="hlw">σ threshold</span> = the stock has entered
    unexplored state space. This is <strong>regime detection, not threshold alerting.</strong>
    </div>""", unsafe_allow_html=True)

    fig_mah = go.Figure()
    fig_mah.add_trace(go.Scatter(x=ts, y=d_arr, mode="lines",
        fill="tozeroy", fillcolor="rgba(0,119,187,0.04)",
        line=dict(color=CB_BLUE, width=1.5), name="Distance σ"))
    shade_anomalies(fig_mah, pred_anom, pd.Series(ts))
    fig_mah.add_hline(y=sigma_thresh, line_dash="dash", line_color=CB_RED,
        line_width=1.5, annotation_text=f"  Manifold boundary σ={sigma_thresh:.1f}",
        annotation_font=dict(color=CB_RED, size=10))
    fig_mah.add_hline(y=sigma_thresh*0.6, line_dash="dot", line_color=CB_ORANGE,
        line_width=1, annotation_text="  Warning zone",
        annotation_font=dict(color=CB_ORANGE, size=9))
    fig_mah.update_layout(**PTHEME, height=330,
        title=dict(text="Manifold Distance Over Time — Did the stock leave its natural regime?",
                   font=dict(color="#374151", size=12, family="Inter")),
        xaxis_title="Date", yaxis_title="Distance (σ)")
    st.plotly_chart(fig_mah, use_container_width=True)

    # Manifold boundary
    st.markdown('<div class="sec">Manifold Boundary — 2D Projection with σ Contours</div>',
                unsafe_allow_html=True)
    n_tr   = int(len(states_sc) * train_frac)
    tr2d   = proj2[:n_tr]
    mu2d   = tr2d.mean(axis=0)
    cov2d  = np.cov(tr2d.T) + 1e-8*np.eye(2)
    vals_e, vecs_e = np.linalg.eigh(cov2d)
    theta  = np.linspace(0, 2*np.pi, 360)
    unit   = np.column_stack([np.cos(theta), np.sin(theta)])

    fig_mb = go.Figure()
    for (sig, col_s, ds_s, nm) in [
        (sigma_thresh,       CB_RED,"solid", f"σ={sigma_thresh:.1f} (boundary)"),
        (sigma_thresh * 0.7, CB_ORANGE,"dot",   f"σ={sigma_thresh*0.7:.1f} (warning)"),
        (sigma_thresh * 0.4, CB_BLUE,"dot",   f"σ={sigma_thresh*0.4:.1f} (nominal)"),
    ]:
        ell = mu2d + sig*(unit @ np.diag(np.sqrt(vals_e)) @ vecs_e.T)
        fig_mb.add_trace(go.Scatter(x=ell[:,0], y=ell[:,1], mode="lines",
            line=dict(color=col_s, width=1.5, dash=ds_s), name=nm))
    fig_mb.add_trace(go.Scatter(
        x=proj2[~pred_anom,0], y=proj2[~pred_anom,1], mode="markers",
        marker=dict(size=2.5, color="rgba(0,119,187,0.40)"), name="Normal"))
    if pred_anom.any():
        fig_mb.add_trace(go.Scatter(
            x=proj2[pred_anom,0], y=proj2[pred_anom,1], mode="markers",
            marker=dict(size=7, color=CB_RED, symbol="diamond",
                        line=dict(color="#ffffff", width=0.6)),
            name="Outside Manifold"))
    fig_mb.add_trace(go.Scatter(x=[mu2d[0]], y=[mu2d[1]], mode="markers",
        marker=dict(size=14, color=CB_BLUE, symbol="cross",
                    line=dict(color="#ffffff", width=2)),
        name="Manifold Center"))
    fig_mb.update_layout(**PTHEME, height=440,
        title=dict(text="Natural Market Manifold — PC₁ × PC₂",
                   font=dict(color="#374151", size=12, family="Inter")),
        xaxis_title="PC₁", yaxis_title="PC₂")
    st.plotly_chart(fig_mb, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig_dh = go.Figure()
        fig_dh.add_trace(go.Histogram(x=d_arr[~pred_anom], nbinsx=60,
            marker_color=CB_BLUE, opacity=0.85, name="Normal"))
        if pred_anom.any():
            fig_dh.add_trace(go.Histogram(x=d_arr[pred_anom], nbinsx=40,
                marker_color=CB_RED, opacity=0.85, name="Anomalous"))
        fig_dh.add_vline(x=sigma_thresh, line_dash="dash", line_color=CB_ORANGE,
            annotation_text=f" σ={sigma_thresh:.1f}",
            annotation_font=dict(color=CB_ORANGE, size=10))
        fig_dh.update_layout(**PTHEME, height=280, barmode="overlay",
            title=dict(text="Distance Distribution",
                       font=dict(color="#374151", size=12, family="Inter")),
            xaxis_title="Mahalanobis σ", yaxis_title="Days")
        st.plotly_chart(fig_dh, use_container_width=True)
    with c2:
        sd    = np.sort(d_arr)
        pctls = np.linspace(0, 100, len(sd))
        fig_cdf = go.Figure()
        fig_cdf.add_trace(go.Scatter(x=sd, y=pctls, mode="lines",
            line=dict(color=CB_BLUE, width=2),
            fill="tozeroy", fillcolor="rgba(0,119,187,0.05)"))
        fig_cdf.add_vline(x=sigma_thresh, line_dash="dash", line_color=CB_RED,
            annotation_text=" threshold",
            annotation_font=dict(color=CB_RED, size=10))
        fig_cdf.update_layout(**PTHEME, height=280,
            title=dict(text="Empirical CDF",
                       font=dict(color="#374151", size=12, family="Inter")),
            xaxis_title="Mahalanobis σ", yaxis_title="Percentile (%)")
        st.plotly_chart(fig_cdf, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — DENSITY MATRIX
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown('<div class="sec">Quantum Density Matrix ρ</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="ibox">
    <strong>ρ = (1/N) Σ |ψ_t⟩⟨ψ_t|</strong> — the full statistical state of the market regime.<br>
    <span class="hl">Diagonal</span> = how much each financial feature dominates the regime.<br>
    <span class="hl">Off-diagonal</span> = hidden correlations between features —
    what threshold monitoring never sees.<br>
    The <span class="hlw">shift matrix</span> shows exactly which features
    changed relationship during anomalous periods.
    </div>""", unsafe_allow_html=True)

    n_tr  = int(len(states_norm) * train_frac)
    rho_n = density_matrix(states_norm[:n_tr])
    rho_a = density_matrix(states_norm[pred_anom]) if pred_anom.any() \
            else density_matrix(states_norm)

    c1, c2 = st.columns(2)
    for (rho, title_s, col_hi, col_obj) in [
        (rho_n, "ρ — Normal Market Regime",   CB_BLUE, c1),
        (rho_a, "ρ — Anomalous Market State", CB_RED, c2),
    ]:
        ar = np.abs(rho)
        fig_rho = go.Figure(go.Heatmap(
            z=ar, x=LABELS, y=LABELS,
            colorscale=[[0,"#E5E7EB"],[0.4,"#E5E7EB"],[0.7,CB_CYAN],[1,col_hi]],
            colorbar=dict(title="|ρᵢⱼ|", thickness=10),
            text=np.round(ar,3), texttemplate="%{text}",
            textfont=dict(size=10, color="#111827", family="Inter"),
        ))
        fig_rho.update_layout(**{k:v for k,v in PTHEME.items() if k not in ("xaxis","yaxis")}, height=370,
            title=dict(text=title_s, font=dict(color="#374151", size=12, family="Inter")),
            xaxis=dict(gridcolor="#E5E7EB", linecolor="#E5E7EB", showgrid=False, tickangle=35),
            yaxis=dict(gridcolor="#E5E7EB", linecolor="#E5E7EB", showgrid=False))
        col_obj.plotly_chart(fig_rho, use_container_width=True)

    st.markdown('<div class="sec">Eigenvalue Spectrum — Market Mode Analysis</div>',
                unsafe_allow_html=True)
    eigs_n = np.sort(np.linalg.eigvalsh(rho_n))[::-1]
    eigs_a = np.sort(np.linalg.eigvalsh(rho_a))[::-1]
    lbl_e  = [f"λ{i+1}" for i in range(len(eigs_n))]
    fig_eig = go.Figure()
    fig_eig.add_trace(go.Bar(x=lbl_e, y=eigs_n,
        marker_color=CB_BLUE, opacity=0.85, name="Normal ρ"))
    fig_eig.add_trace(go.Bar(x=lbl_e, y=eigs_a,
        marker_color=CB_RED, opacity=0.85, name="Anomalous ρ"))
    fig_eig.update_layout(**PTHEME, height=280, barmode="group",
        title=dict(text="Eigenvalue Spectrum λᵢ — Which market modes dominate each regime",
                   font=dict(color="#374151", size=12, family="Inter")),
        yaxis_title="Eigenvalue λ")
    st.plotly_chart(fig_eig, use_container_width=True)

    st.markdown('<div class="sec">ρ Shift Matrix: |ρ_anomalous − ρ_normal|</div>',
                unsafe_allow_html=True)
    rho_diff = np.abs(rho_a - rho_n)
    fig_diff = go.Figure(go.Heatmap(
        z=rho_diff, x=LABELS, y=LABELS,
        colorscale=[[0,"#E5E7EB"],[0.5,CB_CYAN],[1,CB_ORANGE]],
        colorbar=dict(title="Δ|ρ|", thickness=12),
        text=np.round(rho_diff,3), texttemplate="%{text}",
        textfont=dict(size=10, color="#111827", family="Inter"),
    ))
    fig_diff.update_layout(**{k:v for k,v in PTHEME.items() if k not in ("xaxis","yaxis")}, height=360,
        title=dict(text="Which feature correlations break down during anomalous regimes",
                   font=dict(color="#374151", size=12, family="Inter")),
        xaxis=dict(gridcolor="#E5E7EB", linecolor="#E5E7EB", showgrid=False, tickangle=35),
        yaxis=dict(gridcolor="#E5E7EB", linecolor="#E5E7EB", showgrid=False))
    st.plotly_chart(fig_diff, use_container_width=True)

    S_n  = vn_entropy(rho_n); S_a = vn_entropy(rho_a)
    P_n  = float(np.trace(rho_n @ rho_n)); P_a = float(np.trace(rho_a @ rho_a))
    frob = float(np.linalg.norm(rho_diff))
    sc   = "c-a3" if S_a > S_n else "c-a"
    st.markdown(f"""
    <div class="m-grid">
      <div class="m-card"><div class="m-label">Normal S(ρ)</div>
        <div class="m-val c-a">{S_n:.4f}</div><div class="m-delta">Baseline entropy</div></div>
      <div class="m-card"><div class="m-label">Anomalous S(ρ)</div>
        <div class="m-val c-a3">{S_a:.4f}</div><div class="m-delta">Anomaly entropy</div></div>
      <div class="m-card"><div class="m-label">Entropy ΔS</div>
        <div class="m-val {sc}">{S_a-S_n:+.4f}</div><div class="m-delta">Regime shift</div></div>
      <div class="m-card"><div class="m-label">Normal Purity</div>
        <div class="m-val c-a2">{P_n:.4f}</div><div class="m-delta">Tr(ρ²)</div></div>
      <div class="m-card"><div class="m-label">Anomalous Purity</div>
        <div class="m-val c-g">{P_a:.4f}</div><div class="m-delta">Tr(ρ²)</div></div>
      <div class="m-card"><div class="m-label">Frobenius Shift</div>
        <div class="m-val c-a3">{frob:.4f}</div><div class="m-delta">‖ρ_anom − ρ_norm‖_F</div></div>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — QUANTUM PORTFOLIO OPTIMIZER + EFFICIENT FRONTIER
# ══════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.markdown('<div class="sec">Quantum-Inspired Portfolio Optimizer</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="ibox">
    The <strong>Efficient Frontier</strong> shows every possible portfolio as a point in
    risk-return space — from thousands of random weight combinations.<br><br>
    <strong>Quantum framing:</strong> portfolio weights form a probability amplitude vector
    <strong>|w⟩</strong> normalised to 1. The covariance matrix is the portfolio's
    <strong>Hamiltonian Ĥ</strong> — minimising variance is finding the
    <strong>ground state of Ĥ</strong>.<br><br>
    <span class="hl">⭐ Max Sharpe</span> — highest risk-adjusted return &nbsp;|&nbsp;
    <span class="hlw">◆ Min Volatility</span> — lowest possible risk &nbsp;|&nbsp;
    <span class="hlg">● Equal Weight</span> — naive baseline
    </div>""", unsafe_allow_html=True)

    # ── parse tickers ──────────────────────────────────────────────────────────
    port_tickers = [t.strip().upper() for t in port_tickers_raw.split(",") if t.strip()]
    port_tickers = list(dict.fromkeys(port_tickers))[:10]

    if len(port_tickers) < 2:
        st.warning("Enter at least 2 tickers in the sidebar Portfolio Optimizer section.")
    else:
        # ── fetch all assets ──────────────────────────────────────────────────
        with st.spinner(f"Fetching data for {', '.join(port_tickers)}…"):
            port_closes = {}
            failed_tickers = []
            for t in port_tickers:
                try:
                    raw = yf.download(t, period=period, auto_adjust=True, progress=False)
                    raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
                    if not raw.empty:
                        port_closes[t] = raw["Close"].squeeze()
                    else:
                        failed_tickers.append(t)
                except Exception:
                    failed_tickers.append(t)

            if failed_tickers:
                st.warning(f"Could not fetch: {', '.join(failed_tickers)}. Skipping.")

            valid_tickers = list(port_closes.keys())
            if len(valid_tickers) < 2:
                st.error("Need at least 2 valid tickers. Check symbols and retry.")
                st.stop()

            prices_df  = pd.DataFrame(port_closes).dropna()
            returns_df = prices_df.pct_change().dropna()

        n_assets    = len(valid_tickers)
        mean_ret    = returns_df.mean().values * 252
        cov_matrix  = returns_df.cov().values  * 252
        corr_matrix = returns_df.corr().values

        # ── Manifold anomaly score per asset ──────────────────────────────────
        asset_anomaly = {}
        for t in valid_tickers:
            try:
                sub_df = prices_df[[t]].rename(columns={t: "Close"})
                sub_df["Volume"] = 1e6  # dummy volume if not available
                try:
                    raw_v = yf.download(t, period=period, auto_adjust=True, progress=False)
                    raw_v.columns = [c[0] if isinstance(c, tuple) else c for c in raw_v.columns]
                    sub_df["Volume"] = raw_v["Volume"].squeeze().reindex(sub_df.index).fillna(1e6)
                except Exception:
                    pass
                f_sub = build_features(sub_df)
                sc_sub, _, _ = build_states(f_sub)
                d_sub, _, _  = manifold_fit(sc_sub, train_frac)
                asset_anomaly[t] = float((d_sub > sigma_thresh).mean() * 100)
            except Exception:
                asset_anomaly[t] = 0.0

        # ── Monte Carlo simulation of portfolios ──────────────────────────────
        rng_pf   = np.random.default_rng(42)
        sim_rets = np.zeros(port_n_sim)
        sim_vols = np.zeros(port_n_sim)
        sim_wgts = np.zeros((port_n_sim, n_assets))

        for i in range(port_n_sim):
            w = rng_pf.random(n_assets)
            w = w / w.sum()
            sim_wgts[i]  = w
            sim_rets[i]  = float(w @ mean_ret)
            sim_vols[i]  = float(np.sqrt(w @ cov_matrix @ w))

        sim_sharpe = (sim_rets - risk_free) / (sim_vols + 1e-9)

        # ── Optimisation: Max Sharpe ──────────────────────────────────────────
        def neg_sharpe(w):
            r = float(w @ mean_ret)
            v = float(np.sqrt(w @ cov_matrix @ w))
            return -(r - risk_free) / (v + 1e-9)

        def port_vol(w):
            return float(np.sqrt(w @ cov_matrix @ w))

        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
        bounds      = [(0.01, 0.60)] * n_assets
        w0          = np.ones(n_assets) / n_assets

        res_sharpe = minimize(neg_sharpe, w0, method="SLSQP",
                              bounds=bounds, constraints=constraints,
                              options={"maxiter": 1000, "ftol": 1e-9})
        w_sharpe   = res_sharpe.x
        r_sharpe   = float(w_sharpe @ mean_ret)
        v_sharpe   = float(np.sqrt(w_sharpe @ cov_matrix @ w_sharpe))
        s_sharpe   = (r_sharpe - risk_free) / v_sharpe

        # ── Optimisation: Min Volatility ──────────────────────────────────────
        res_minvol = minimize(port_vol, w0, method="SLSQP",
                              bounds=bounds, constraints=constraints,
                              options={"maxiter": 1000, "ftol": 1e-9})
        w_minvol   = res_minvol.x
        r_minvol   = float(w_minvol @ mean_ret)
        v_minvol   = float(np.sqrt(w_minvol @ cov_matrix @ w_minvol))
        s_minvol   = (r_minvol - risk_free) / v_minvol

        # ── Equal weight baseline ─────────────────────────────────────────────
        w_eq  = np.ones(n_assets) / n_assets
        r_eq  = float(w_eq @ mean_ret)
        v_eq  = float(np.sqrt(w_eq @ cov_matrix @ w_eq))
        s_eq  = (r_eq - risk_free) / v_eq

        # ── Individual asset stats ────────────────────────────────────────────
        ind_vols = np.sqrt(np.diag(cov_matrix))
        ind_rets = mean_ret

        # ── KPI row ───────────────────────────────────────────────────────────
        st.markdown(f"""
        <div class="m-grid">
          <div class="m-card">
            <div class="m-label">Max Sharpe Ratio</div>
            <div class="m-val c-a">{s_sharpe:.3f}</div>
            <div class="m-delta">Optimal risk-adjusted</div>
          </div>
          <div class="m-card">
            <div class="m-label">Max Sharpe Return</div>
            <div class="m-val c-a">{r_sharpe*100:.1f}%</div>
            <div class="m-delta">Annualised</div>
          </div>
          <div class="m-card">
            <div class="m-label">Max Sharpe Volatility</div>
            <div class="m-val c-a2">{v_sharpe*100:.1f}%</div>
            <div class="m-delta">Annualised σ</div>
          </div>
          <div class="m-card">
            <div class="m-label">Min Vol Return</div>
            <div class="m-val c-g">{r_minvol*100:.1f}%</div>
            <div class="m-delta">Ground state</div>
          </div>
          <div class="m-card">
            <div class="m-label">Min Volatility</div>
            <div class="m-val c-a2">{v_minvol*100:.1f}%</div>
            <div class="m-delta">Lowest risk frontier</div>
          </div>
          <div class="m-card">
            <div class="m-label">Equal Weight Sharpe</div>
            <div class="m-val">{s_eq:.3f}</div>
            <div class="m-delta">Naive baseline</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── EFFICIENT FRONTIER CHART ──────────────────────────────────────────
        st.markdown('<div class="sec">Efficient Frontier — Risk vs Return Space</div>',
                    unsafe_allow_html=True)

        fig_ef = go.Figure()

        # Simulated portfolios coloured by Sharpe
        fig_ef.add_trace(go.Scatter(
            x=sim_vols * 100, y=sim_rets * 100,
            mode="markers",
            marker=dict(
                size=3,
                color=sim_sharpe,
                colorscale=[
                    [0.0, "#E5E7EB"],
                    [0.3, CB_CYAN],
                    [0.6, CB_CYAN],
                    [1.0, CB_NAVY],
                ],
                colorbar=dict(
                    title="Sharpe Ratio",
                    thickness=12,
                    titlefont=dict(color="#374151", size=10),
                    tickfont=dict(color="#111827", size=10),
                ),
                opacity=0.65,
                cmin=float(np.percentile(sim_sharpe, 5)),
                cmax=float(np.percentile(sim_sharpe, 95)),
            ),
            text=[
                f"Return: {r*100:.1f}%<br>Vol: {v*100:.1f}%<br>Sharpe: {s:.3f}"
                for r, v, s in zip(sim_rets, sim_vols, sim_sharpe)
            ],
            hovertemplate="%{text}<extra>Simulated Portfolio</extra>",
            name=f"{port_n_sim:,} Random Portfolios",
        ))

        # Individual assets — bold ALL labels; active ticker gets navy + bigger
        _asset_sizes  = [18 if t == ticker else 11 for t in valid_tickers]
        _asset_colors = [CB_NAVY if t == ticker else CB_RED for t in valid_tickers]
        _asset_labels = [f"<b>{t} ★</b>" if t == ticker else f"<b>{t}</b>" for t in valid_tickers]
        fig_ef.add_trace(go.Scatter(
            x=ind_vols * 100, y=ind_rets * 100,
            mode="markers+text",
            marker=dict(size=_asset_sizes, color=_asset_colors,
                        line=dict(color="#ffffff", width=2)),
            text=_asset_labels,
            textposition="top center",
            textfont=dict(size=12, color="#111827", family="Inter"),
            hovertemplate="<b>%{text}</b><br>Return: %{y:.1f}%<br>Vol: %{x:.1f}%<extra></extra>",
            name="Individual Assets",
        ))

        # Equal weight
        fig_ef.add_trace(go.Scatter(
            x=[v_eq * 100], y=[r_eq * 100],
            mode="markers+text",
            marker=dict(size=14, color=CB_ORANGE, symbol="circle",
                        line=dict(color="#ffffff", width=2)),
            text=["<b>Equal Weight</b>"],
            textposition="top right",
            textfont=dict(size=12, color="#111827", family="Inter"),
            hovertemplate=f"<b>Equal Weight</b><br>Return: {r_eq*100:.1f}%<br>Vol: {v_eq*100:.1f}%<br>Sharpe: {s_eq:.3f}<extra></extra>",
            name=f"Equal Weight (Sharpe {s_eq:.2f})",
        ))

        # Min Vol portfolio
        fig_ef.add_trace(go.Scatter(
            x=[v_minvol * 100], y=[r_minvol * 100],
            mode="markers+text",
            marker=dict(size=18, color=CB_CYAN, symbol="diamond",
                        line=dict(color="#ffffff", width=2)),
            text=["<b>Min Vol</b>"],
            textposition="top right",
            textfont=dict(size=12, color="#111827", family="Inter"),
            hovertemplate=f"<b>Min Volatility</b><br>Return: {r_minvol*100:.1f}%<br>Vol: {v_minvol*100:.1f}%<br>Sharpe: {s_minvol:.3f}<extra></extra>",
            name=f"Min Volatility (Sharpe {s_minvol:.2f})",
        ))

        # Max Sharpe portfolio
        fig_ef.add_trace(go.Scatter(
            x=[v_sharpe * 100], y=[r_sharpe * 100],
            mode="markers+text",
            marker=dict(size=22, color=CB_NAVY, symbol="star",
                        line=dict(color="#ffffff", width=2)),
            text=["<b>⭐ Max Sharpe</b>"],
            textposition="top right",
            textfont=dict(size=13, color=CB_NAVY, family="Inter"),
            hovertemplate=f"<b>Max Sharpe Portfolio</b><br>Return: {r_sharpe*100:.1f}%<br>Vol: {v_sharpe*100:.1f}%<br>Sharpe: {s_sharpe:.3f}<extra></extra>",
            name=f"Max Sharpe (Sharpe {s_sharpe:.2f})",
        ))

        # Capital Market Line
        cml_vols = np.linspace(0, max(sim_vols) * 1.1, 100)
        cml_rets = risk_free + s_sharpe * cml_vols
        fig_ef.add_trace(go.Scatter(
            x=cml_vols * 100, y=cml_rets * 100,
            mode="lines",
            line=dict(color=CB_NAVY, width=1.5, dash="dash"),
            hoverinfo="skip",
            name="Capital Market Line",
        ))

        layout = PTHEME.copy()
        
        layout["height"] = 520
        layout["title"] = dict(
            text="Efficient Frontier — Each dot is a portfolio. Color = Sharpe ratio.",
            font=dict(color="#374151", size=12, family="Inter")
        )
        
        # ✅ FIX: update existing axis dicts instead of using xaxis_title
        layout["xaxis"].update(title="Annualised Volatility (%)")
        layout["yaxis"].update(title="Annualised Return (%)")
        
        layout["legend"] = dict(
            bgcolor="#ffffff",
            bordercolor="#E5E7EB",
            borderwidth=1,
            font=dict(size=11, color="#111827"),
            orientation="v",
            x=1.12,
            y=1,
        )
        
        fig_ef.update_layout(**layout)
        st.plotly_chart(fig_ef, use_container_width=True)

        # ── Optimal weight allocations ────────────────────────────────────────
        st.markdown('<div class="sec">Optimal Portfolio Weight Allocations</div>',
                    unsafe_allow_html=True)

        col_ms, col_mv = st.columns(2)

        for (col, w_opt, label, color, sharpe_v) in [
            (col_ms, w_sharpe, "⭐ Max Sharpe Portfolio", CB_BLUE, s_sharpe),
            (col_mv, w_minvol, "◆ Min Volatility Portfolio", CB_TEAL, s_minvol),
        ]:
            with col:
                st.markdown(f"""
                <div style="background:#fafaf8;border:1.5px solid #E5E7EB;border-radius:12px;
                            padding:1rem 1.2rem;margin-bottom:1rem;">
                  <div style="font-weight:700;font-size:.9rem;color:{color};margin-bottom:.2rem">
                    {label}
                  </div>
                  <div style="font-family:'JetBrains Mono',monospace;font-size:.7rem;color:#374151">
                    Sharpe: {sharpe_v:.3f}
                  </div>
                </div>
                """, unsafe_allow_html=True)

                # Active ticker gets navy + thicker border; all labels bold
                _bar_colors  = [CB_NAVY if t == ticker else color for t in valid_tickers]
                _bar_borders = [CB_NAVY if t == ticker else "#ffffff" for t in valid_tickers]
                _bar_widths  = [3.0 if t == ticker else 1.5 for t in valid_tickers]
                _bar_text    = [f"<b>{w*100:.1f}%</b>" for w in w_opt]
                _x_labels    = [f"<b>{t} ★</b>" if t == ticker else f"<b>{t}</b>"
                                 for t in valid_tickers]
                fig_wgt = go.Figure(go.Bar(
                    x=_x_labels,
                    y=w_opt * 100,
                    marker_color=_bar_colors,
                    marker_line_color=_bar_borders,
                    marker_line_width=_bar_widths,
                    text=_bar_text,
                    textposition="outside",
                    textfont=dict(size=12, color="#111827", family="Inter"),
                ))
            
                _wgt_theme = {k: v for k, v in PTHEME.items() if k not in ("xaxis", "yaxis")}
                fig_wgt.update_layout(
                    **_wgt_theme,
                    height=320,
                    title=dict(text=f"{label} — Weights", font=dict(color="#111827", size=13, family="Inter")),
                    xaxis=dict(gridcolor="#E5E7EB", linecolor="#E5E7EB", showgrid=False, zeroline=False,
                               tickfont=dict(size=12, color="#111827", family="Inter")),
                    yaxis=dict(
                        title="Weight (%)",
                        gridcolor="#E5E7EB",
                        linecolor="#E5E7EB",
                        zeroline=False,
                        tickfont=dict(size=11, color="#111827"),
                        range=[0, max(w_opt)*1.35]
                    ),
                    showlegend=False,
                )
                st.plotly_chart(fig_wgt, use_container_width=True)

        # Pie charts side by side
        # Active ticker slice gets a contrasting pull-out effect via offset
        _pie_pull = [0.08 if t == ticker else 0 for t in valid_tickers]
        col_p1, col_p2 = st.columns(2)
        for (col, w_opt, label, colors_pie) in [
            (col_p1, w_sharpe, "Max Sharpe",      CB_SEQ),
            (col_p2, w_minvol, "Min Volatility",  [CB_CYAN,CB_BLUE,CB_TEAL,CB_NAVY,
                                                    CB_ORANGE,CB_RED,CB_MAGENTA,CB_GREY,
                                                    "#34d399","#fb923c"]),
        ]:
            with col:
                _pie_labels = [f"<b>{t}</b>" if t == ticker else t for t in valid_tickers]
                fig_pie = go.Figure(go.Pie(
                    labels=_pie_labels, values=w_opt * 100,
                    hole=0.45,
                    pull=_pie_pull,
                    marker=dict(colors=colors_pie[:n_assets],
                                line=dict(color="#ffffff", width=2)),
                    textfont=dict(size=12, color="#111827", family="Inter"),
                    textinfo="label+percent",
                    hovertemplate="<b>%{label}</b><br>Weight: %{value:.1f}%<extra></extra>",
                ))
                fig_pie.update_layout(
                    **PTHEME, height=320,
                    title=dict(text=f"{label} — Allocation",
                               font=dict(color="#374151", size=12, family="Inter")),
                    showlegend=False,
                )
                st.plotly_chart(fig_pie, use_container_width=True)

        # ── Correlation matrix ────────────────────────────────────────────────
        st.markdown('<div class="sec">Asset Correlation Matrix</div>', unsafe_allow_html=True)

        # Bold the active ticker in correlation axis labels
        _corr_labels = [f"<b>{t} ★</b>" if t == ticker else f"<b>{t}</b>"
                        for t in valid_tickers]
        fig_corr = go.Figure(go.Heatmap(
            z=corr_matrix,
            x=_corr_labels, y=_corr_labels,
            colorscale=[
                [0.0, CB_RED],
                [0.5, "#F0F0F0"],
                [1.0, CB_BLUE],
            ],
            zmin=-1, zmax=1,
            text=np.round(corr_matrix, 2),
            texttemplate="%{text}",
            textfont=dict(size=12, color="#111827", family="Inter"),
            colorbar=dict(title="ρ", thickness=12,
                          titlefont=dict(color="#111827", size=12),
                          tickfont=dict(color="#111827", size=11)),
        ))
        _pt2 = {k:v for k,v in PTHEME.items() if k not in ("xaxis","yaxis")}
        fig_corr.update_layout(
            **_pt2, height=420,
            title=dict(
                text="Return Correlation — Lower correlation = better diversification",
                font=dict(color="#111827", size=13, family="Inter")
            ),
            xaxis=dict(gridcolor="#E5E7EB", linecolor="#E5E7EB", showgrid=False,
                       tickfont=dict(size=12, color="#111827", family="Inter")),
            yaxis=dict(gridcolor="#E5E7EB", linecolor="#E5E7EB", showgrid=False,
                       tickfont=dict(size=12, color="#111827", family="Inter")),
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        # ── Manifold anomaly score per asset ──────────────────────────────────
        st.markdown('<div class="sec">Manifold Anomaly Score per Asset</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div class="ibox">
        What percentage of each asset's history was spent <strong>outside its natural
        8D manifold</strong>? High anomaly % = historically erratic behaviour —
        useful context when choosing portfolio weights.
        </div>""", unsafe_allow_html=True)

        anom_vals   = [asset_anomaly[t] for t in valid_tickers]
        anom_colors = [CB_RED if v > 20 else CB_ORANGE if v > 10 else CB_TEAL
                       for v in anom_vals]
        # Bold ALL labels; active ticker gets navy border + star
        _anom_borders = [CB_NAVY if t == ticker else "#ffffff" for t in valid_tickers]
        _anom_widths  = [3.0 if t == ticker else 1.5 for t in valid_tickers]
        _anom_text    = [f"<b>{v:.1f}%</b>" for v in anom_vals]
        _anom_xlabels = [f"<b>{t} ★</b>" if t == ticker else f"<b>{t}</b>"
                         for t in valid_tickers]

        fig_anom = go.Figure(go.Bar(
            x=_anom_xlabels, y=anom_vals,
            marker_color=anom_colors,
            marker_line_color=_anom_borders,
            marker_line_width=_anom_widths,
            text=_anom_text,
            textposition="outside",
            textfont=dict(size=12, color="#111827", family="Inter"),
        ))
        fig_anom.add_hline(y=20, line_dash="dash", line_color=CB_RED,
                           annotation_text=" High anomaly threshold (20%)",
                           annotation_font=dict(color=CB_RED, size=10))
        fig_anom.add_hline(y=10, line_dash="dot", line_color=CB_ORANGE,
                           annotation_text=" Moderate threshold (10%)",
                           annotation_font=dict(color=CB_ORANGE, size=10))
        _anom_theme = {k: v for k, v in PTHEME.items() if k not in ("xaxis", "yaxis")}
        fig_anom.update_layout(
            **_anom_theme, height=340,
            title=dict(
                text="% of Days Outside Natural 8D Manifold — quantum regime instability",
                font=dict(color="#111827", size=13, family="Inter")
            ),
            xaxis=dict(gridcolor="#E5E7EB", linecolor="#E5E7EB", showgrid=False, zeroline=False,
                       tickfont=dict(size=12, color="#111827", family="Inter")),
            yaxis=dict(title="Anomaly Rate (%)", gridcolor="#E5E7EB", linecolor="#E5E7EB",
                       zeroline=False, tickfont=dict(size=11, color="#111827"),
                       range=[0, max(anom_vals or [10]) * 1.4]),
            showlegend=False,
        )
        st.plotly_chart(fig_anom, use_container_width=True)

        # ── Summary table ─────────────────────────────────────────────────────
        st.markdown('<div class="sec">Portfolio Comparison Summary</div>',
                    unsafe_allow_html=True)

        summary_data = {
            "Portfolio":   ["Max Sharpe", "Min Volatility", "Equal Weight"],
            "Return (%)":  [f"{r_sharpe*100:.2f}", f"{r_minvol*100:.2f}", f"{r_eq*100:.2f}"],
            "Volatility (%)": [f"{v_sharpe*100:.2f}", f"{v_minvol*100:.2f}", f"{v_eq*100:.2f}"],
            "Sharpe Ratio":   [f"{s_sharpe:.3f}", f"{s_minvol:.3f}", f"{s_eq:.3f}"],
        }
        # Add individual weights
        for t, ws, wm, we in zip(valid_tickers, w_sharpe, w_minvol, w_eq):
            summary_data[f"{t} weight"] = [
                f"{ws*100:.1f}%", f"{wm*100:.1f}%", f"{we*100:.1f}%"
            ]

        st.dataframe(
            pd.DataFrame(summary_data),
            hide_index=True,
            use_container_width=True,
        )

        st.markdown("""
        <div class="ibox" style="margin-top:1rem">
        <strong>⚠️ Disclaimer:</strong> This is for educational purposes only. Not financial advice.
        Past returns do not guarantee future performance. Portfolio optimisation assumes
        normally distributed returns and stable covariance — real markets violate both.
        </div>""", unsafe_allow_html=True)
