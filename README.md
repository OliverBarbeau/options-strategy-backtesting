# Options Backtesting Engine

A solo-built, AI-assisted engine for **backtesting and simulating options
strategies** (primarily put-credit spreads and other multi-leg spreads) on
historical data. It hand-rolls Black-Scholes pricing and the full Greeks, exposes
an extensible `PricingProvider` abstraction, and adds a quant-validation layer
(walk-forward, bootstrap, A/B testing, explicit bias controls). Python / FastAPI
backend, Next.js frontend. (The package is named `tradelab`.)

> **Disclaimer.** This is a research and educational project. It prices, backtests,
> and simulates strategies on historical data — it does **not** place trades,
> connect to a broker, or run live, and nothing here is financial advice.
> Backtested results are hypothetical and do not represent real returns.

## What's inside

- **Pricing & Greeks** — Black-Scholes from scratch (delta, gamma, theta, vega, rho),
  historical and EWM volatility, and multi-leg spread pricing (put credit spreads,
  iron condors, calendars).
- **Strategies** — several put-credit-spread variants (pullback entry, regime-adaptive,
  conservative, and more) under `tradelab/strategies/`.
- **Quant validation** — walk-forward out-of-sample testing, bootstrap resampling,
  A/B testing with Fisher's exact test, and explicit look-ahead-bias and
  survivorship-bias controls.
- **Extensible `PricingProvider`** so pricing models and data sources can be swapped
  without touching strategy code.

## Quick start

```bash
pip install -e .

# Run the test suite
PYTHONPATH=. python -m pytest tests/ -q

# Scan the market for candidates, then backtest the top ones
PYTHONPATH=. python run_sim.py scan --pullback --backtest

# Simulation accounts
PYTHONPATH=. python run_sim.py list
PYTHONPATH=. python run_sim.py advance
```

### Web app

```bash
# Terminal 1 — API
PYTHONPATH=. uvicorn api.main:app --reload --port 8000

# Terminal 2 — frontend (Node 22+)
cd web && npm install && npm run dev
```

Then open http://localhost:3000.

## Backtested strategies

Put-credit-spread strategies, all 30 DTE open / 14 DTE close, evaluated on a
10-year, 25-ticker historical backtest:

| Strategy | Notes |
|----------|-------|
| Pullback Entry | Enter on a 3%+ pullback from the 20-day high; best risk-adjusted in backtest |
| Regime Adaptive | Scale the buffer to the SPY volatility regime; highest raw backtest returns |
| Conservative | Fixed 10% buffer every cycle; simplest |
| Aggressive Pullback | Tighter buffer + stacking; risk-tolerant |

Headline backtest returns look high, but they are **gross** — realistic frictions
(commissions + slippage) erase a large share of gross P/L (see Key findings #8).

## Key findings

1. **A 97% win rate is not enough** — breakeven sits at 90-98% with tight spreads.
2. **Higher vol is more profitable** — premiums outweigh breach risk in the 25-45% HV range.
3. **~14-day DTE is the sweet spot** — balances premium against risk exposure.
4. **Pullback entry is the best filter** — a 3%+ dip captures elevated vol.
5. **Iron condors underperform here** — the call side gets breached too often.
6. **Simple beats complex** — always-close-at-14-DTE beats conditional logic.
7. **Earnings gaps destroy buffers** — filter trades with earnings in the hold period.
8. **Friction is real** — $0.65/leg + 2% slippage eats ~40% of gross P/L.

## Project structure

```
tradelab/            Core library
  pricing/           PricingProvider abstraction (Black-Scholes, mock, ThetaData)
  strategies/        Strategy implementations
api/                 FastAPI backend (backtests, accounts, scanner)
web/                 Next.js frontend
tests/               pytest suite
experiments/         Research scripts
```

Built solo and AI-assisted.
