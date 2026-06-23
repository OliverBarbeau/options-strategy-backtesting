# Findings

A plain-language summary of what this project set out to test, how the results
were validated, and what held up — for readers who would rather not dig through
the code. Everything here is **backtested research on historical data**, not live
trading and not financial advice.

## The question

Can a systematic **put-credit-spread** program — selling out-of-the-money put
spreads on liquid large-cap names and managing them by simple rules — be made to
work, and where does it break? The engine prices options from scratch
(Black-Scholes plus the full Greeks), simulates multi-leg spreads day by day, and
scores strategies across a 10+ year, ~25-ticker universe.

## How the results were validated

The hard part of backtesting isn't producing a good-looking number; it's trusting
it. This project leans on several guardrails:

- **Real historical options data.** Absolute P/L is measured against actual
  historical bid/ask quotes, not just Black-Scholes estimates. Model and market
  prices diverge (volatility skew, implied vs. realized vol), so the model is
  trusted only where that gap is understood.
- **Walk-forward, out-of-sample testing.** Strategies are judged on data they
  were not tuned on, so results aren't just curve-fit to the past.
- **Bootstrap resampling** for confidence intervals around performance, rather
  than trusting a single historical path.
- **A/B testing with Fisher's exact test** to decide whether one strategy truly
  beats another or just got lucky.
- **Explicit look-ahead and survivorship-bias controls** — nothing uses data that
  wouldn't have been available at decision time, and the universe is deliberately
  diversified rather than cherry-picked survivors.

## What held up

1. **A high win rate is not the same as an edge.** Put credit spreads win the
   large majority of the time, but with tight spreads the breakeven win rate sits
   around **90-98%** — a couple of bad losses erase many small wins. The win rate
   is almost a distraction; the loss distribution is what matters.
2. **Higher volatility was more profitable, within a band.** In roughly the
   **25-45% historical-vol** range, the extra premium collected outweighed the
   higher breach risk. Too calm and there's no premium; too wild and breaches
   dominate.
3. **Closing early beat holding to expiration.** Opening ~30 days out and closing
   around 14 days consistently beat both holding longer and conditional exit
   logic. Simple, fixed-time exits won.
4. **Entry timing mattered more than clever exits.** Entering on a 3%+ pullback
   from a recent high — selling into elevated volatility — was the single most
   useful filter. Most "safety feature" exit rules *hurt* performance in calm
   markets and only helped in sustained bear markets.
5. **Iron condors underperformed.** Adding a call spread to collect more premium
   backfired: the call side got breached too often in up markets. Put spreads
   alone did better.
6. **Earnings are a landmine.** A single earnings gap inside the holding period
   can blow through the buffer. Filtering out trades with earnings before expiry
   was necessary, not optional.
7. **Friction is the silent killer.** Realistic costs — about **$0.65 per leg plus
   ~2% slippage** — ate on the order of **40% of gross P/L**. Any backtest that
   ignores friction tells a much better story than reality.
8. **Drawdown is the binding constraint.** The strategies could compound through
   calm and rising regimes, but a sustained bear market (2022-style) drove deep
   drawdowns. The risk isn't the win rate — it's the depth of the bad years, and
   position sizing and loss limits matter more than squeezing out extra edge.

## On the numbers

This write-up is intentionally light on headline return figures. Black-Scholes-
priced backtests systematically misprice options (real puts cost more than the
model implies, especially out-of-the-money), so absolute P/L from model-priced
runs is treated as **directional only** — useful for ranking strategies against
each other, not for quoting returns. The real-data validation runs are the
reference for absolute claims, and even those are **gross** figures before the
frictions above. The honest summary is qualitative: the approach looks viable in
calm and rising markets, vulnerable in sustained selloffs, and considerably less
lucrative net of costs than gross backtests suggest.

## Scope & limitations

- A ~25-ticker large-cap universe over 10+ years; results won't necessarily
  generalize to illiquid names or different market regimes.
- Backtests assume fills at modeled or observed prices; real-world fills,
  assignment, and liquidity are not fully modeled.
- This is a research and educational tool. It does not place trades, connect to a
  broker, or run live, and nothing here is financial advice.

For setup see the [README](README.md); for the implementation behind every claim
here, see `tradelab/` and the 219 passing tests under `tests/`.
