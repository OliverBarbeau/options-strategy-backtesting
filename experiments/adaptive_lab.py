"""Strategy Lab: backtest each AdaptivePullback feature in isolation.

Runs the same ticker universe and date range through:
  1. Baseline (no features -- should match pullback_entry)
  2. +vol_pause alone
  3. +cooldown alone
  4. +adaptive_threshold alone
  5. +stop_loss_breach alone
  6. +fast_profit alone
  7. Combined (all features)

Each run reports trades, win rate, P/L, drawdown. The diagnostic counters
show how often each feature actually triggered -- if a feature never fires
on real data, it's noise. If it fires often but hurts P/L, it's worse than
useless.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd

from tradelab.pipeline import DataPipeline
from tradelab.options import historical_volatility
from tradelab.strategies.adaptive_pullback import AdaptivePullbackStrategy

pipe = DataPipeline()

TICKERS = ["NVDA", "AVGO", "MSFT", "META", "GOOG", "CAT"]
START = "2020-01-01"
END = "2025-12-31"

print("=" * 110)
print(f"{'ADAPTIVE PULLBACK LAB: 5 YEARS, 6 TICKERS':^110}")
print(f"{'Each feature tested in isolation vs combined':^110}")
print("=" * 110)

# Pre-load data and market vol
print("\nLoading data...")
DATA = {}
for t in TICKERS + ["SPY"]:
    DATA[t] = pipe.fetch_stock(t, start=START, end=END)
SPY_VOL = historical_volatility(DATA["SPY"]["close"], window=30)
print(f"Loaded {len(TICKERS)} tickers + SPY\n")


def run_config(name, **kwargs):
    """Run a config across all tickers, return aggregated metrics."""
    strat = AdaptivePullbackStrategy(
        buffer=0.10,
        spread_pct=0.02,
        pullback_threshold=0.03,
        **kwargs,
    )

    total_trades = 0
    total_winners = 0
    total_pnl = 0.0
    max_dds = []
    skip_vol = 0
    skip_cool = 0
    skip_adapt = 0
    close_stop = 0
    close_fast = 0

    for ticker in TICKERS:
        df = DATA[ticker]
        result = strat.run(df, market_vol_series=SPY_VOL, max_contracts=10)
        total_trades += result.total_trades
        total_winners += result.winners
        total_pnl += result.total_pnl
        max_dds.append(result.max_drawdown_pct)
        skip_vol += result.skipped_vol_pause
        skip_cool += result.skipped_cooldown
        skip_adapt += result.skipped_adaptive_threshold
        close_stop += result.closed_early_stop_loss
        close_fast += result.closed_fast_profit

    wr = total_winners / total_trades if total_trades else 0
    avg_pnl = total_pnl / total_trades if total_trades else 0
    avg_dd = sum(max_dds) / len(max_dds)

    return {
        "name": name,
        "trades": total_trades,
        "win_rate": wr,
        "total_pnl": total_pnl,
        "avg_pnl": avg_pnl,
        "avg_dd": avg_dd,
        "skip_vol": skip_vol,
        "skip_cool": skip_cool,
        "skip_adapt": skip_adapt,
        "close_stop": close_stop,
        "close_fast": close_fast,
    }


# ---- Configurations ----
configs = [
    ("BASELINE",               {}),
    ("vol_pause 25%",          {"vol_pause_threshold": 0.25}),
    ("vol_pause 30%",          {"vol_pause_threshold": 0.30}),
    ("cooldown 10d",           {"cooldown_days": 10}),
    ("cooldown 20d",           {"cooldown_days": 20}),
    ("adaptive_pullback",      {"adaptive_pullback": True}),
    ("stop_loss_breach",       {"stop_loss_breach": True}),
    ("fast_profit 75%/5d",     {"fast_profit_target": 0.75, "fast_profit_window": 5}),
    ("fast_profit 50%/5d",     {"fast_profit_target": 0.50, "fast_profit_window": 5}),
    ("fast_profit 50%/10d",    {"fast_profit_target": 0.50, "fast_profit_window": 10}),
    # Promising combos
    ("cooldown + fast_profit", {"cooldown_days": 10, "fast_profit_target": 0.50, "fast_profit_window": 10}),
    ("vol_pause + cooldown",   {"vol_pause_threshold": 0.25, "cooldown_days": 10}),
    ("ALL FEATURES",           {
        "vol_pause_threshold": 0.25,
        "cooldown_days": 10,
        "adaptive_pullback": True,
        "stop_loss_breach": True,
        "fast_profit_target": 0.50,
        "fast_profit_window": 10,
    }),
]

print(f"{'Config':<26} {'Trades':>7} {'WR':>6} {'Total P/L':>12} {'$/Trade':>9} {'AvgDD':>8} {'Triggers':>24}")
print("-" * 110)

baseline_pnl = None
results = []

for name, kwargs in configs:
    r = run_config(name, **kwargs)
    results.append(r)

    if name == "BASELINE":
        baseline_pnl = r["total_pnl"]
        delta_str = ""
    else:
        delta = r["total_pnl"] - baseline_pnl
        delta_pct = (delta / baseline_pnl * 100) if baseline_pnl != 0 else 0
        delta_str = f" ({'+' if delta >= 0 else ''}{delta_pct:.0f}%)"

    triggers = []
    if r["skip_vol"]:
        triggers.append(f"vol:{r['skip_vol']}")
    if r["skip_cool"]:
        triggers.append(f"cd:{r['skip_cool']}")
    if r["skip_adapt"]:
        triggers.append(f"ad:{r['skip_adapt']}")
    if r["close_stop"]:
        triggers.append(f"sl:{r['close_stop']}")
    if r["close_fast"]:
        triggers.append(f"fp:{r['close_fast']}")
    trig_str = ", ".join(triggers) if triggers else "-"

    print(
        f"{name:<26} {r['trades']:>7} {r['win_rate']:>5.1%} ${r['total_pnl']:>+11,.0f}{delta_str:<9} "
        f"${r['avg_pnl']:>+8.2f} {r['avg_dd']:>7.1%}  {trig_str:<22}"
    )

# Rankings
print()
print("=" * 110)
print(f"{'RANKED BY TOTAL P/L':^110}")
print("=" * 110)
ranked = sorted(results, key=lambda r: r["total_pnl"], reverse=True)
for i, r in enumerate(ranked, 1):
    delta = r["total_pnl"] - baseline_pnl
    print(f"  {i:>2}. {r['name']:<28} ${r['total_pnl']:>+11,.0f}  "
          f"({'+' if delta >= 0 else ''}${delta:,.0f} vs baseline)  "
          f"WR={r['win_rate']:.1%}  $/trade=${r['avg_pnl']:+.2f}")

# Sharpe proxy: P/L / |max drawdown|
print()
print("=" * 110)
print(f"{'RANKED BY RISK-ADJUSTED RETURN (|P/L| / |AvgDD|)':^110}")
print("=" * 110)

def risk_adj(r):
    if r["avg_dd"] == 0:
        return r["total_pnl"]
    return abs(r["total_pnl"]) / abs(r["avg_dd"] * 10000)  # scale

risk_ranked = sorted(results, key=risk_adj, reverse=True)
for i, r in enumerate(risk_ranked, 1):
    print(f"  {i:>2}. {r['name']:<28} P/L=${r['total_pnl']:>+10,.0f}  "
          f"DD={r['avg_dd']:>6.1%}  score={risk_adj(r):>7.1f}")

# Feature contribution analysis
print()
print("=" * 110)
print(f"{'FEATURE CONTRIBUTION (delta vs baseline, single-feature configs)':^110}")
print("=" * 110)

single_features = [r for r in results if r["name"] != "BASELINE" and "ALL" not in r["name"] and " + " not in r["name"]]

print(f"\n{'Feature':<28} {'Delta P/L':>12} {'Delta %':>10} {'Trades':>8} {'Impact':<30}")
print("-" * 95)
for r in sorted(single_features, key=lambda x: x["total_pnl"] - baseline_pnl, reverse=True):
    delta = r["total_pnl"] - baseline_pnl
    delta_pct = (delta / baseline_pnl * 100) if baseline_pnl != 0 else 0
    trade_delta = r["trades"] - next(x["trades"] for x in results if x["name"] == "BASELINE")
    impact = (
        "strongly positive" if delta_pct > 10
        else "moderately positive" if delta_pct > 2
        else "neutral" if abs(delta_pct) < 2
        else "moderately negative" if delta_pct > -10
        else "strongly negative"
    )
    print(f"{r['name']:<28} ${delta:>+11,.0f} {delta_pct:>+9.1f}% {trade_delta:>+7} trades  {impact}")
