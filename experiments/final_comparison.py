"""Final comparison: composite strategy vs baseline and best single-filter variants.

Focused test of the tiered composite strategy against:
1. Baseline pullback (3% threshold)
2. RSI < 25 (best single filter)
3. RSI < 30 (near-breakeven, more trades)
4. Composite (tiered RSI + deep pullback)
5. Composite aggressive (lower tier thresholds for more trades)

Also includes per-tier P/L breakdown for the composite to validate the
tiered sizing hypothesis.
"""

from __future__ import annotations

import time

import pandas as pd
import numpy as np

from tradelab.pipeline import DataPipeline
from tradelab.options import historical_volatility
from tradelab.strategies.pullback_entry import PullbackEntryStrategy
from tradelab.strategies.rsi_pullback import RSIPullbackStrategy
from tradelab.strategies.composite_pullback import CompositePullbackStrategy


TICKERS = ["NVDA", "AAPL", "GOOG", "MSFT", "CAT", "AVGO"]
START = "2014-01-01"
END = "2024-12-31"
MAX_CONTRACTS = 10


def load_data():
    pipe = DataPipeline()
    data = {}
    for t in TICKERS:
        df = pipe.fetch_stock(t, start=START, end=END)
        if df is not None and len(df) > 60:
            data[t] = df
    return data


def run_aggregate(strategy, data, max_contracts=MAX_CONTRACTS):
    total_trades = 0
    total_winners = 0
    total_pnl = 0.0
    worst_dd = 0.0
    all_trades = []

    for ticker, df in data.items():
        try:
            result = strategy.run(df, max_contracts=max_contracts)
        except Exception as e:
            print(f"  ERROR on {ticker}: {e}")
            continue
        total_trades += result.total_trades
        total_winners += result.winners
        total_pnl += result.total_pnl
        worst_dd = min(worst_dd, result.max_drawdown_pct)
        all_trades.extend(result.trade_log)

    wr = total_winners / total_trades if total_trades > 0 else 0
    ppt = total_pnl / total_trades if total_trades > 0 else 0
    return {
        "trades": total_trades,
        "win_rate": wr,
        "total_pnl": total_pnl,
        "pnl_per_trade": ppt,
        "max_dd": worst_dd,
        "trade_log": all_trades,
    }


def main():
    t0 = time.time()
    print("=" * 80)
    print("FINAL STRATEGY COMPARISON")
    print("=" * 80)
    print()

    data = load_data()
    print(f"Loaded {len(data)} tickers\n")

    strategies = [
        ("Baseline Pullback", PullbackEntryStrategy()),
        ("RSI < 25 (best filter)", RSIPullbackStrategy(rsi_oversold=25.0)),
        ("RSI < 30", RSIPullbackStrategy(rsi_oversold=30.0)),
        ("RSI < 35", RSIPullbackStrategy(rsi_oversold=35.0)),
        ("Composite (default)", CompositePullbackStrategy()),
        ("Composite (relaxed)", CompositePullbackStrategy(
            tier1_rsi=30, tier2_rsi=40, tier3_pullback=0.04,
        )),
        ("Composite (no vol pause)", CompositePullbackStrategy(vol_pause=1.0)),
        ("Composite (tight tiers)", CompositePullbackStrategy(
            tier1_rsi=20, tier2_rsi=30, tier3_pullback=0.06,
            tier2_scale=0.70, tier3_scale=0.50,
        )),
    ]

    results = []
    for name, strategy in strategies:
        r = run_aggregate(strategy, data)
        results.append((name, r))

    # ---- Main results table ----
    print("=" * 80)
    print("RESULTS (sorted by $/trade)")
    print("=" * 80)
    print()

    header = f"{'Strategy':<30} {'Trades':>6} {'WR':>7} {'Total P/L':>12} {'$/Trade':>10}"
    print(header)
    print("-" * len(header))

    for name, m in sorted(results, key=lambda x: x[1]["pnl_per_trade"], reverse=True):
        print(
            f"{name:<30} "
            f"{m['trades']:>6} "
            f"{m['win_rate']:>6.1%} "
            f"{m['total_pnl']:>+11,.2f} "
            f"{m['pnl_per_trade']:>+9,.2f}"
        )

    # ---- Composite tier breakdown ----
    print()
    print("=" * 80)
    print("COMPOSITE TIER BREAKDOWN")
    print("=" * 80)

    for name, m in results:
        if "Composite" not in name:
            continue

        trades_df = pd.DataFrame(m["trade_log"])
        if trades_df.empty or "tier" not in trades_df.columns:
            continue

        print(f"\n  {name}:")
        print(f"  {'Tier':<12} {'Trades':>6} {'WR':>7} {'P/L':>12} {'$/Trade':>10}")
        print(f"  {'-'*52}")

        for tier in ["tier1", "tier2", "tier3"]:
            tier_df = trades_df[trades_df["tier"] == tier]
            if tier_df.empty:
                print(f"  {tier:<12} {'0':>6} {'N/A':>7} {'$0.00':>12} {'$0.00':>10}")
                continue
            wr = tier_df["winner"].sum() / len(tier_df)
            pnl = tier_df["pnl"].sum()
            ppt = pnl / len(tier_df)
            print(
                f"  {tier:<12} "
                f"{len(tier_df):>6} "
                f"{wr:>6.1%} "
                f"{pnl:>+11,.2f} "
                f"{ppt:>+9,.2f}"
            )

    # ---- Year-by-year: baseline vs best composite ----
    print()
    print("=" * 80)
    print("YEAR-BY-YEAR: Baseline vs Composite variants")
    print("=" * 80)

    for name, m in results:
        if name not in ["Baseline Pullback", "Composite (default)", "Composite (relaxed)"]:
            continue

        trades_df = pd.DataFrame(m["trade_log"])
        if trades_df.empty:
            continue
        trades_df["year"] = trades_df["date"].dt.year

        print(f"\n  {name}:")
        print(f"  {'Year':<6} {'Trades':>6} {'WR':>7} {'P/L':>12}")
        print(f"  {'-'*35}")

        for year in sorted(trades_df["year"].unique()):
            yr = trades_df[trades_df["year"] == year]
            wr = yr["winner"].sum() / len(yr) if len(yr) > 0 else 0
            print(f"  {year:<6} {len(yr):>6} {wr:>6.1%} {yr['pnl'].sum():>+11,.2f}")

    # ---- Per-ticker for best composite ----
    print()
    print("=" * 80)
    print("PER-TICKER: Composite (default)")
    print("=" * 80)

    composite_strat = CompositePullbackStrategy()
    print(f"\n  {'Ticker':<8} {'Trades':>6} {'WR':>7} {'P/L':>12} {'T1':>4} {'T2':>4} {'T3':>4}")
    print(f"  {'-'*52}")

    for ticker, df in data.items():
        result = composite_strat.run(df, max_contracts=MAX_CONTRACTS)
        wr = result.win_rate
        print(
            f"  {ticker:<8} "
            f"{result.total_trades:>6} "
            f"{wr:>6.1%} "
            f"{result.total_pnl:>+11,.2f} "
            f"{result.tier1_trades:>4} "
            f"{result.tier2_trades:>4} "
            f"{result.tier3_trades:>4}"
        )

    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
