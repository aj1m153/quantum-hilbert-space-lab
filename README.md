# ⬡ QuantLab Terminal

A professional quantitative research Streamlit app with four integrated modules.

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Modules

### 01 · Market Regime Clustering
- **Model**: Gaussian Hidden Markov Model (hmmlearn)
- **Features**: Rolling log-returns + realised volatility
- **Output**: Bull / Bear / Sideways regime labels per trading day
- **Config**: Ticker, history, HMM states (2–4), rolling window, EM iterations

### 02 · Statistical Pairs Trading
- **Test**: Engle-Granger cointegration (statsmodels)
- **Signal**: Rolling z-score of OLS-hedged spread
- **Output**: Entry/exit signals, cumulative P&L vs buy-and-hold
- **Config**: Two tickers, entry/exit z-score thresholds, z-score window

### 03 · Volatility Surface Builder
- **Data**: Live options chain via yfinance
- **Model**: Black-Scholes inversion (scipy.optimize.brentq)
- **Output**: 3D IV surface (Plotly) + volatility smile cross-sections
- **Config**: Ticker, risk-free rate, min open interest, option type

### 04 · CVaR Portfolio Optimisation
- **Solver**: CVXPY with SCS backend
- **Objective**: Minimise Conditional Value-at-Risk (Expected Shortfall)
- **Benchmarks**: Equal-weight, Max-Sharpe (scipy)
- **Output**: Optimal weights, cumulative return chart, stats table

## Notes
- Options data best with liquid US underlyings: SPY, QQQ, AAPL, TSLA
- CVaR solver uses linear programming reformulation (Rockafellar & Uryasev 2000)
- HMM states are mapped to regimes by mean rolling return (lowest = Bear, highest = Bull)
