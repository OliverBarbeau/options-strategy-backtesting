"""Parameter Challenge: empirically test our core assumptions.

This experiment challenges five core beliefs of the pullback strategy:

1. Is 3% the right pullback threshold, or did we pick it arbitrarily?
2. Is 10% the right buffer? We never tested 8%, 11%, 12%.
3. Is 30/14 the right DTE combo? We tested 30/14 vs hold-to-expiry,
   but never 21/14, 45/14, 30/7, 30/21.
4. Do results hold across different market regimes? (2020-2022 vs 2023-2025)
5. Does the strategy work on a broader ticker universe?

Each sweep tests a single parameter axis while holding others constant.
If the strategy is robust, we should see smooth gradients (not cliffs).
Cliffs indicate overfitting or curve-fitting to a specific value.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd

from tradelab.pipeline import DataPipeline
from tradelab.strategies.pullback_entry import PullbackEntryStrategy

pipe = DataPipeline()

# Our research-validated universe
TIER1 = ["NVDA", "AVGO", "MSFT"]
TIER2 = ["META", "GOOG", "CAT"]
CORE = TIER1 + TIER2
EXPANDED = CORE + ["AAPL", "AMD", "QQQ", "SPY"]


def run_single(ticker: str, df, **kwargs) -> dict:
    """Run pullback strategy with specified params."""
    strat = PullbackEntryStrategy(**kwargs)
    result = strat.run(df, max_contracts=10)
    return {
        "ticker": ticker,
        "trades": result.total_trades,
        "winners": result.winners,
        "losers": result.losers,
        "pnl": result.total_pnl,
        "wr": result.win_rate,
        "dd": result.max_drawdown_pct,
    }


def run_universe(tickers: list[str], data: dict, **kwargs) -> dict:
    """Run over a universe, return aggregated metrics."""
    total_trades = 0
    total_winners = 0
    total_pnl = 0.0
    dds = []
    for t in tickers:
        if t not in data:
            continue
        r = run_single(t, data[t], **kwargs)
        total_trades += r["trades"]
        total_winners += r["winners"]
        total_pnl += r["pnl"]
        dds.append(r["dd"])

    wr = total_winners / total_trades if total_trades else 0
    return {
        "trades": total_trades,
        "wr": wr,
        "pnl": total_pnl,
        "avg_pnl": total_pnl / total_trades if total_trades else 0,
        "avg_dd": sum(dds) / len(dds) if dds else 0,
    }


# ---- Load data once ----
print("Loading data...")
DATA = {}
for t in EXPANDED:
    try:
        DATA[t] = pipe.fetch_stock(t, start="2020-01-01", end="2025-12-31")
    except Exception as e:
        print(f"  Skip {t}: {e}")
print(f"Loaded {len(DATA)} tickers\n")


W = 100

# =============================================================
# CHALLENGE 1: Pullback threshold (was 3% picked arbitrarily?)
# =============================================================
print("=" * W)
print(f"{'CHALLENGE 1: Pullback threshold fine sweep (core universe, 2020-2025)':^{W}}")
print("=" * W)
print(f"\n{'Threshold':>10} {'Trades':>7} {'WR':>6} {'Total P/L':>11} {'$/Trade':>9} {'AvgDD':>8}")
print("-" * 60)

thresholds = [0.01, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05, 0.06, 0.08, 0.10]
pullback_results = []
for pt in thresholds:
    r = run_universe(CORE, DATA, pullback_threshold=pt)
    pullback_results.append((pt, r))
    marker = "  <--" if pt == 0.03 else ""
    print(f"  {pt*100:>6.1f}% {r['trades']:>7} {r['wr']:>5.1%} ${r['pnl']:>+10,.0f} ${r['avg_pnl']:>+8.2f} {r['avg_dd']:>7.1%}{marker}")

# Best threshold by total P/L
best_t = max(pullback_results, key=lambda x: x[1]["pnl"])
print(f"\n  Best by P/L: {best_t[0]*100:.1f}% (${best_t[1]['pnl']:+,.0f})")
print(f"  Our pick:    3.0% (${pullback_results[3][1]['pnl']:+,.0f})")

# =============================================================
# CHALLENGE 2: Buffer size
# =============================================================
print()
print("=" * W)
print(f"{'CHALLENGE 2: Buffer fine sweep':^{W}}")
print("=" * W)
print(f"\n{'Buffer':>8} {'Trades':>7} {'WR':>6} {'Total P/L':>11} {'$/Trade':>9} {'AvgDD':>8}")
print("-" * 60)

buffers = [0.05, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.15]
buffer_results = []
for b in buffers:
    r = run_universe(CORE, DATA, buffer=b)
    buffer_results.append((b, r))
    marker = "  <--" if b == 0.10 else ""
    print(f"  {b*100:>5.1f}% {r['trades']:>7} {r['wr']:>5.1%} ${r['pnl']:>+10,.0f} ${r['avg_pnl']:>+8.2f} {r['avg_dd']:>7.1%}{marker}")

best_b = max(buffer_results, key=lambda x: x[1]["pnl"])
print(f"\n  Best by P/L: {best_b[0]*100:.1f}% (${best_b[1]['pnl']:+,.0f})")
print(f"  Our pick:    10.0% (${buffer_results[4][1]['pnl']:+,.0f})")

# =============================================================
# CHALLENGE 3: DTE combinations
# =============================================================
print()
print("=" * W)
print(f"{'CHALLENGE 3: DTE open/close combinations':^{W}}")
print("=" * W)
print(f"\n{'DTE Open':>9} {'DTE Close':>9} {'Hold':>6} {'Trades':>7} {'WR':>6} {'Total P/L':>11} {'$/Trade':>9}")
print("-" * 75)

dte_combos = [
    (21, 7),   # short-short
    (21, 14),  # short-medium
    (30, 7),   # standard-short close
    (30, 14),  # our standard
    (30, 21),  # standard-long close (less theta captured)
    (45, 14),  # long-standard
    (45, 21),  # long-long
    (60, 14),  # very long
    (60, 21),
]
dte_results = []
for dto, dtc in dte_combos:
    hold = dto - dtc
    r = run_universe(CORE, DATA, dte_open=dto, dte_close=dtc)
    dte_results.append(((dto, dtc), r))
    marker = "  <--" if (dto, dtc) == (30, 14) else ""
    print(f"  {dto:>7} {dtc:>9} {hold:>6}d {r['trades']:>7} {r['wr']:>5.1%} ${r['pnl']:>+10,.0f} ${r['avg_pnl']:>+8.2f}{marker}")

best_dte = max(dte_results, key=lambda x: x[1]["pnl"])
print(f"\n  Best by P/L: {best_dte[0][0]}/{best_dte[0][1]} (${best_dte[1]['pnl']:+,.0f})")
print(f"  Our pick:    30/14 (${dte_results[3][1]['pnl']:+,.0f})")

# =============================================================
# CHALLENGE 4: Time period sensitivity
# =============================================================
print()
print("=" * W)
print(f"{'CHALLENGE 4: Time period sensitivity (baseline strategy, 3% / 10% / 30/14)':^{W}}")
print("=" * W)

periods = [
    ("2020-2021 (COVID+Bull)", "2020-01-01", "2021-12-31"),
    ("2022-2022 (Bear)",       "2022-01-01", "2022-12-31"),
    ("2023-2023 (Recovery)",   "2023-01-01", "2023-12-31"),
    ("2024-2024 (Bull)",       "2024-01-01", "2024-12-31"),
    ("2025-2025 (Current)",    "2025-01-01", "2025-12-31"),
]

print(f"\n{'Period':<30} {'Trades':>7} {'WR':>6} {'Total P/L':>11} {'$/Trade':>9}")
print("-" * 70)

for name, start, end in periods:
    # Filter each ticker's data to this period
    period_data = {}
    start_ts = int(pd.Timestamp(start).timestamp())
    end_ts = int(pd.Timestamp(end).timestamp())
    for t, df in DATA.items():
        period_data[t] = df[(df.index >= start_ts) & (df.index <= end_ts)]

    r = run_universe(CORE, period_data)
    print(f"  {name:<28} {r['trades']:>7} {r['wr']:>5.1%} ${r['pnl']:>+10,.0f} ${r['avg_pnl']:>+8.2f}")

# =============================================================
# CHALLENGE 5: Tier 1 vs Tier 2 vs Expanded
# =============================================================
print()
print("=" * W)
print(f"{'CHALLENGE 5: Universe sensitivity (baseline strategy)':^{W}}")
print("=" * W)

universes = [
    ("Tier 1 (NVDA, AVGO, MSFT)", TIER1),
    ("Tier 2 (META, GOOG, CAT)",  TIER2),
    ("Core (6 tickers)",          CORE),
    ("Expanded (10 tickers)",     EXPANDED),
]

print(f"\n{'Universe':<30} {'Trades':>7} {'WR':>6} {'Total P/L':>11} {'$/Trade':>9} {'AvgDD':>8}")
print("-" * 80)

for name, tickers in universes:
    r = run_universe(tickers, DATA)
    print(f"  {name:<28} {r['trades']:>7} {r['wr']:>5.1%} ${r['pnl']:>+10,.0f} ${r['avg_pnl']:>+8.2f} {r['avg_dd']:>7.1%}")

# Per-ticker breakdown on expanded universe
print(f"\nPer-ticker results (expanded universe):")
print(f"  {'Ticker':<8} {'Trades':>7} {'WR':>6} {'Total P/L':>11} {'$/Trade':>9}")
print(f"  {'-'*55}")
for t in EXPANDED:
    if t not in DATA:
        continue
    r = run_single(t, DATA[t])
    print(f"  {t:<8} {r['trades']:>7} {r['wr']:>5.1%} ${r['pnl']:>+10,.0f} ${r['pnl']/r['trades']:>+8.2f}" if r['trades'] else f"  {t:<8} {'0':>7}")

# =============================================================
# SENSITIVITY HEATMAP: pullback x buffer
# =============================================================
print()
print("=" * W)
print(f"{'CHALLENGE 6: Joint sensitivity (pullback x buffer)':^{W}}")
print("=" * W)
print(f"\nTotal P/L across (pullback_threshold, buffer) grid on core universe:\n")

buffers_grid = [0.07, 0.09, 0.10, 0.11, 0.13]
thresholds_grid = [0.02, 0.03, 0.04, 0.05]

# Header
print(f"  {'PB':>6}", end="")
for b in buffers_grid:
    print(f"  buf={b*100:>4.1f}%", end="")
print()
print(f"  {'----':>6}" + "  ---------" * len(buffers_grid))

for pt in thresholds_grid:
    print(f"  {pt*100:>4.1f}%", end="")
    for b in buffers_grid:
        r = run_universe(CORE, DATA, pullback_threshold=pt, buffer=b)
        print(f"  ${r['pnl']:>+8,.0f}", end="")
    print()
