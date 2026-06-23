# Experiments

Research scripts used to develop and stress-test the strategies, kept for
provenance. Most can be re-run, though some depend on cached data or local
account state.

Run with `PYTHONPATH` set:

```bash
PYTHONPATH=. python experiments/<script>.py
```

The work progressed in rough phases:

- **Strategy discovery:** buffer / DTE / spread-width sweeps, and the core
  "always close at 14 DTE" finding.
- **Portfolio construction:** selection and position-sizing across a multi-ticker
  portfolio.
- **Strategy variants:** pullback, regime-adaptive, and safety-feature experiments.
- **Real-data validation:** re-running on historical bid/ask quotes rather than
  Black-Scholes estimates. The early Black-Scholes-priced runs are useful for
  *relative* comparisons between strategies; the real-data phase is the reference
  for absolute P/L.

See [`FINDINGS.md`](../FINDINGS.md) for a directional summary of what these
experiments showed.
