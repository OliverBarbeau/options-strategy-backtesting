"""Clean validation: run core configs against real Theta Data across core universe.

This is the honest follow-up to the split-adjustment correction. Every number
here is on real EOD options data with correctly matched raw underlying prices.

What we're testing:
  1. Our 10%/3% baseline
  2. 7%/2% — the config that parameter_challenge_findings.md dismissed
  3. 5%/3% — tight buffer
  4. 60/14 DTE — longer theta capture
  5. 1% pullback — trade everything qualifying

Across NVDA, AVGO, MSFT, META, GOOG, CAT for 2024.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd

from tradelab.pipeline import DataPipeline
from tradelab.pricing.thetadata import ThetaDataProvider
from tradelab.strategies.pullback_entry import PullbackEntryStrategy

pipe = DataPipeline()
theta = ThetaDataProvider()

if not theta.check_connection():
    print("Theta Terminal not reachable. Aborting.")
    exit(1)

theta.reset_stats()

TICKERS = ["NVDA", "AVGO", "MSFT", "GOOG", "CAT", "AAPL"]
CONFIGS = [
    ("10%/3% (baseline)",    {"buffer": 0.10, "pullback_threshold": 0.03}),
    ("7%/2% (aggressive)",   {"buffer": 0.07, "pullback_threshold": 0.02}),
    ("5%/3% (tight buf)",    {"buffer": 0.05, "pullback_threshold": 0.03}),
    ("60/14 DTE",            {"buffer": 0.10, "pullback_threshold": 0.03, "dte_open": 60}),
    ("1% pullback",          {"buffer": 0.10, "pullback_threshold": 0.01}),
]


def run_all_tickers(params: dict) -> dict:
    """Run a config across all tickers, aggregate results."""
    totals = {"trades": 0, "winners": 0, "pnl": 0.0}
    per_ticker = {}
    for ticker in TICKERS:
        try:
            df = pipe.fetch_stock(ticker, start="2024-01-01", end="2024-12-31")
        except Exception:
            continue

        strat = PullbackEntryStrategy(**params)
        result = strat.run(df, max_contracts=10, ticker=ticker, provider=theta)

        per_ticker[ticker] = {
            "trades": result.total_trades,
            "winners": result.winners,
            "pnl": result.total_pnl,
            "wr": result.win_rate,
            "dd": result.max_drawdown_pct,
        }
        totals["trades"] += result.total_trades
        totals["winners"] += result.winners
        totals["pnl"] += result.total_pnl

    totals["wr"] = totals["winners"] / totals["trades"] if totals["trades"] else 0
    totals["avg_pnl"] = totals["pnl"] / totals["trades"] if totals["trades"] else 0
    return {"totals": totals, "per_ticker": per_ticker}


W = 100

print("=" * W)
print(f"{'CLEAN VALIDATION: REAL THETA DATA, 2024, CORE UNIVERSE':^{W}}")
print(f"{'(post split-adjustment fix)':^{W}}")
print("=" * W)
print()

results = {}
for name, params in CONFIGS:
    print(f"Running: {name}...")
    results[name] = run_all_tickers(params)

print()
print("=" * W)
print(f"{'AGGREGATE RESULTS ACROSS 6 TICKERS':^{W}}")
print("=" * W)
print(f"{'Config':<28} {'Trades':>7} {'WR':>6} {'Total P/L':>12} {'$/Trade':>9}")
print("-" * W)

# Sort by P/L
ranked = sorted(results.items(), key=lambda x: x[1]["totals"]["pnl"], reverse=True)
baseline_pnl = results["10%/3% (baseline)"]["totals"]["pnl"]

for name, r in ranked:
    t = r["totals"]
    delta = t["pnl"] - baseline_pnl
    delta_str = f" ({'+' if delta >= 0 else ''}{delta / baseline_pnl * 100:.0f}%)" if baseline_pnl else ""
    marker = "  <--" if "baseline" in name else ""
    print(f"{name:<28} {t['trades']:>7} {t['wr']:>5.1%} "
          f"${t['pnl']:>+11,.0f}{delta_str:<8} ${t['avg_pnl']:>+8.2f}{marker}")

# Per-ticker breakdown for top 3 configs
print()
print("=" * W)
print(f"{'PER-TICKER BREAKDOWN (TOP 3 CONFIGS)':^{W}}")
print("=" * W)

top3_names = [name for name, _ in ranked[:3]]
print(f"{'Ticker':<8}", end="")
for name in top3_names:
    print(f"  {name[:20]:>20}", end="")
print()
print("-" * (8 + 22 * 3))

for ticker in TICKERS:
    print(f"{ticker:<8}", end="")
    for name in top3_names:
        r = results[name]["per_ticker"].get(ticker, {})
        if r:
            print(f"  ${r['pnl']:>+9,.0f} ({r['wr']:>4.0%})   ", end="")
        else:
            print(f"  {'--':>20}", end="")
    print()

print()
print("=" * W)
theta.print_stats()
