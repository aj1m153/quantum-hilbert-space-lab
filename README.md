# ⚛ Quantum Hilbert Space Lab

> *Most dashboards show you what already happened.  
> This one shows you how your system behaves in latent space.*

---

## What it is

A Streamlit financial-analytics dashboard that doesn't plot raw metrics — it **projects system behaviour into an 8-dimensional Hilbert space** and then observes it through quantum-inspired operators.

Instead of asking **"Is this value too high?"** you ask **"Did the system leave its natural manifold?"**

---

## Architecture

### The 8 Field Dimensions (Ψ₁ … Ψ₈)

| Dim | Field | Definition |
|-----|-------|------------|
| Ψ₁ | Return Field | Log daily price change |
| Ψ₂ | Volatility Field | 20-period rolling σ |
| Ψ₃ | Momentum Field | 10-period price momentum |
| Ψ₄ | Volume Flux | Volume deviation from 20d mean |
| Ψ₅ | RSI Oscillator | 14-period RSI, normalised 0–1 |
| Ψ₆ | Autocorrelation | Lag-1 rolling autocorr |
| Ψ₇ | Skewness Field | 20-period return skewness |
| Ψ₈ | Mean-Reversion | Price / SMA-50 − 1 |

Each row of data becomes a **unit-norm state vector** `|ψₜ⟩` in this 8D space.

### Quantum Engine

| Concept | Implementation |
|---------|---------------|
| Hamiltonian H | Covariance matrix of all state vectors |
| Energy `⟨Ĥ⟩` | Quadratic form `ψᵀ H ψ` |
| Momentum `⟨P̂⟩` | `‖ψₜ − ψₜ₋₁‖` (state-space velocity) |
| Coherence | `\|⟨ψ\|ψ₀⟩\|²` — overlap with ground state |
| Entropy S(ρ) | Normalised von Neumann entropy across eigenstates |
| Anomaly | Mahalanobis distance above threshold percentile |
| Eigenstates | Eigen-decomposition of H → energy levels |

---

## Five Tabs

1. **Hilbert Trajectory** — 3D PCA projection of the 8D path through time. Trajectory coloured by time. Anomalies marked as red crosses.
2. **Quantum Operators** — Time-series of all four observables with deviation windows highlighted.
3. **Manifold Geometry** — Density landscape, Mahalanobis timeline, feature loadings per principal axis.
4. **Deviation Events** — Ranked table of anomaly dates, radar profile (anomaly vs normal), energy–momentum phase portrait.
5. **Eigenspectrum** — Energy levels of H, eigenstate occupation amplitudes over time, basis reference table.

---

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Streamlit Cloud

1. Push to a public GitHub repo.
2. Connect at [share.streamlit.io](https://share.streamlit.io).
3. Set `app.py` as the entry point.

---

## Key Design Choices

- **No statsmodels dependency** — OLS is hand-rolled with NumPy (`lstsq` + manual t-stats).
- **No pandas-datareader** — only `yfinance` + a session spoof to reduce rate-limit hits.
- **Caching** — all heavy computations are wrapped in `@st.cache_data(ttl=3600)`.
- **Graceful fallback** — if yfinance with a spoofed session fails, retries with the plain API.

---

## Extending the Lab

- **Add a 5-factor model**: include Ψ₉ (IV spread) and Ψ₁₀ (earnings distance) for options data.
- **Real-time mode**: swap `yfinance` for a WebSocket feed and call `st.rerun()` every N seconds.
- **Multi-asset entanglement**: project two tickers into the same Hilbert space and measure `|⟨ψ_A|ψ_B⟩|²` as correlation.
- **Regime labelling**: cluster eigenstates with k-means → label epochs as "ground state", "excited", "tunnel event".
