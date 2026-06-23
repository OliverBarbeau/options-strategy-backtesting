"""Tier1Max experiment: does stacking position size on rare RSI<25 signals
beat both the baseline pullback and a plain RSI<25 strategy?

Tests three sizing multipliers: 1.0x (unstacked), 2.0x, and 3.0x.
Also runs PullbackEntryStrategy (baseline) and RSIPullbackStrategy(25) for
reference. Aggregates across 7 tickers over 2014-2024 using B-S pricing.

Outputs:
- Aggregate comparison table.
- Year-by-year P/L breakdown for the best Tier1Max multiplier, with
  explicit focus on 2018, 2020, 2022 (bear-market years where the
  original research showed RSI<25 dominated).
"""

from __future__ import annotations

from collections import defaultdict

import pandas as pd

from tradelab.pipeline import DataPipeline
from tradelab.strategies.pullback_entry import PullbackEntryStrategy
from tradelab.strategies.rsi_pullback import RSIPullbackStrategy
from tradelab.strategies.tier1_max import Tier1MaxStrategy


TICKERS = ["NVDA", "AAPL", "GOOG", "MSFT", "CAT", "AVGO", "META"]
START = "2014-01-01"
END = "2024-12-31"
MAX_CONTRACTS = 10


def load_data(tickers, start, end):
    pipe = DataPipeline()
    data = {}
    print("Loading data...")
    for t in tickers:
        try:
            df = pipe.fetch_stock(t, start=start, end=end)
        except Exception as e:
            print(f"  {t}: ERROR {e}")
            continue
        if df is not None and len(df) > 60:
            data[t] = df
            print(f"  {t}: {len(df)} bars")
        else:
            print(f"  {t}: SKIPPED (insufficient data)")
    return data


def run_across_tickers(strategy, data, max_contracts=MAX_CONTRACTS):
    """Run a strategy across all tickers, aggregate results."""
    all_trades = []
    total_pnl = 0.0
    cum_pnl = 0.0
    peak = 0.0
    max_dd = 0.0

    # Per-ticker run, then merge trades for global drawdown curve
    for ticker, df in data.items():
        res = strategy.run(df, max_contracts=max_contracts)
        for t in res.trade_log:
            t2 = dict(t)
            t2["ticker"] = ticker
            all_trades.append(t2)
        total_pnl += res.total_pnl

    # Sort by entry date and recompute an aggregate max drawdown
    all_trades.sort(key=lambda t: t["date"])
    for t in all_trades:
        cum_pnl += t["pnl"]
        peak = max(peak, cum_pnl)
        max_dd = min(max_dd, cum_pnl - peak)

    winners = sum(1 for t in all_trades if t["winner"])
    losers = len(all_trades) - winners
    wr = winners / len(all_trades) if all_trades else 0.0
    ppt = total_pnl / len(all_trades) if all_trades else 0.0

    return {
        "total_trades": len(all_trades),
        "winners": winners,
        "losers": losers,
        "total_pnl": total_pnl,
        "win_rate": wr,
        "pnl_per_trade": ppt,
        "max_dd_dollars": max_dd,
        "max_dd_pct": max_dd / peak if peak > 0 else 0.0,
        "trades": all_trades,
    }


def year_breakdown(trades):
    """Group trades by entry year, return {year: {trades, pnl, wins, losses}}."""
    by_year = defaultdict(lambda: {"trades": 0, "pnl": 0.0, "wins": 0, "losses": 0})
    for t in trades:
        y = pd.Timestamp(t["date"]).year
        by_year[y]["trades"] += 1
        by_year[y]["pnl"] += t["pnl"]
        if t["winner"]:
            by_year[y]["wins"] += 1
        else:
            by_year[y]["losses"] += 1
    return dict(sorted(by_year.items()))


def print_aggregate_table(rows):
    print()
    print("=" * 96)
    print("AGGREGATE COMPARISON (2014-2024, 7 tickers, B-S pricing)")
    print("=" * 96)
    header = f"{'Strategy':<28} {'Trades':>7} {'WR':>7} {'Total P/L':>14} {'$/Trade':>12} {'MaxDD $':>14} {'MaxDD%':>8}"
    print(header)
    print("-" * 96)
    for name, r in rows:
        print(
            f"{name:<28} "
            f"{r['total_trades']:>7} "
            f"{r['win_rate']*100:>6.1f}% "
            f"{'$'+format(r['total_pnl'], ',.0f'):>14} "
            f"{'$'+format(r['pnl_per_trade'], ',.2f'):>12} "
            f"{'$'+format(r['max_dd_dollars'], ',.0f'):>14} "
            f"{r['max_dd_pct']*100:>7.1f}%"
        )
    print("=" * 96)


def print_year_breakdown(name, trades):
    by_year = year_breakdown(trades)
    print()
    print("=" * 72)
    print(f"YEAR-BY-YEAR BREAKDOWN: {name}")
    print("=" * 72)
    print(f"{'Year':>6} {'Trades':>8} {'Wins':>6} {'Losses':>8} {'P/L':>14} {'$/Trade':>12}")
    print("-" * 72)
    for y, d in by_year.items():
        ppt = d["pnl"] / d["trades"] if d["trades"] else 0.0
        marker = "  <-- bear-market year" if y in (2018, 2020, 2022) else ""
        print(
            f"{y:>6} {d['trades']:>8} {d['wins']:>6} {d['losses']:>8} "
            f"{'$'+format(d['pnl'], ',.0f'):>14} "
            f"{'$'+format(ppt, ',.2f'):>12}{marker}"
        )
    print("=" * 72)

    # Highlight the three crucial bear years
    print()
    print("BEAR-MARKET YEAR FOCUS:")
    for y in (2018, 2020, 2022):
        if y in by_year:
            d = by_year[y]
            print(f"  {y}: {d['trades']} trades, {d['wins']}W/{d['losses']}L, ${d['pnl']:+,.2f}")
        else:
            print(f"  {y}: NO TRADES (signal never fired)")


def main():
    data = load_data(TICKERS, START, END)
    if not data:
        print("No data loaded; aborting.")
        return

    variants = [
        ("Pullback (baseline)", PullbackEntryStrategy()),
        ("RSIPullback (<25)", RSIPullbackStrategy(rsi_oversold=25.0)),
        ("Tier1Max (1.0x)", Tier1MaxStrategy(size_multiplier=1.0)),
        ("Tier1Max (2.0x)", Tier1MaxStrategy(size_multiplier=2.0)),
        ("Tier1Max (3.0x)", Tier1MaxStrategy(size_multiplier=3.0)),
    ]

    rows = []
    for name, strat in variants:
        print(f"\nRunning: {name}")
        r = run_across_tickers(strat, data)
        print(
            f"  {r['total_trades']} trades, "
            f"WR {r['win_rate']*100:.1f}%, "
            f"P/L ${r['total_pnl']:+,.2f}, "
            f"MaxDD ${r['max_dd_dollars']:+,.0f} ({r['max_dd_pct']*100:.1f}%)"
        )
        rows.append((name, r))

    print_aggregate_table(rows)

    # Pick best Tier1Max by total P/L and show year-by-year
    tier_rows = [(n, r) for n, r in rows if n.startswith("Tier1Max")]
    best_name, best_r = max(tier_rows, key=lambda x: x[1]["total_pnl"])
    print(f"\nBest Tier1Max by total P/L: {best_name}")
    print_year_breakdown(best_name, best_r["trades"])

    # Also print year breakdowns for 1x and 2x side-by-side (helps comparison)
    print()
    print("=" * 72)
    print("ALL-TIER1MAX YEAR-BY-YEAR P/L (for comparison)")
    print("=" * 72)
    years = set()
    by_variant = {}
    for name, r in tier_rows:
        by_y = year_breakdown(r["trades"])
        by_variant[name] = by_y
        years.update(by_y.keys())
    years = sorted(years)
    header = f"{'Year':>6}"
    for name, _ in tier_rows:
        header += f" {name:>18}"
    print(header)
    print("-" * len(header))
    for y in years:
        line = f"{y:>6}"
        for name, _ in tier_rows:
            d = by_variant[name].get(y)
            if d:
                line += f" {'$'+format(d['pnl'], ',.0f'):>18}"
            else:
                line += f" {'--':>18}"
        print(line)
    print("=" * 72)


if __name__ == "__main__":
    main()
