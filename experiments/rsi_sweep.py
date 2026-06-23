"""RSI parameter sweep: find optimal RSI thresholds and combinations.

The shootout showed RSI filtering dramatically improves P/L per trade.
Now sweep across:
1. RSI oversold thresholds: 25, 30, 35, 40, 45
2. RSI extreme floors: 0 (off), 10, 15, 20
3. RSI + Momentum combo: RSI < 35 + 50 SMA trend filter
4. RSI + Pullback depth: RSI < 35 + 5% pullback (deeper dip)
"""

from __future__ import annotations

import time

import pandas as pd

from tradelab.pipeline import DataPipeline
from tradelab.options import historical_volatility
from tradelab.strategies.pullback_entry import PullbackEntryStrategy
from tradelab.strategies.rsi_pullback import RSIPullbackStrategy
from tradelab.strategies.momentum_pullback import MomentumPullbackStrategy


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


def run_on_all(strategy, data, max_contracts=MAX_CONTRACTS):
    """Run a strategy on all tickers and return aggregate metrics."""
    total_trades = 0
    total_winners = 0
    total_pnl = 0.0
    worst_dd = 0.0
    all_trades = []

    for ticker, df in data.items():
        try:
            result = strategy.run(df, max_contracts=max_contracts)
        except Exception:
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
    print("RSI PARAMETER SWEEP")
    print("=" * 80)
    print()

    data = load_data()
    print(f"Loaded {len(data)} tickers\n")

    results = []

    # Baseline
    r = run_on_all(PullbackEntryStrategy(), data)
    results.append(("Baseline (no RSI)", r))

    # ---- Sweep 1: RSI oversold thresholds ----
    print("Sweep 1: RSI oversold thresholds (floor=0)")
    for rsi_thresh in [25, 30, 35, 40, 45]:
        s = RSIPullbackStrategy(rsi_oversold=float(rsi_thresh))
        r = run_on_all(s, data)
        results.append((f"RSI < {rsi_thresh}", r))

    # ---- Sweep 2: RSI with extreme floor ----
    print("Sweep 2: RSI < 35 with extreme floors")
    for floor in [10, 15, 20]:
        s = RSIPullbackStrategy(rsi_oversold=35.0, rsi_extreme_floor=float(floor))
        r = run_on_all(s, data)
        results.append((f"RSI < 35, floor {floor}", r))

    # ---- Sweep 3: RSI + deeper pullbacks ----
    print("Sweep 3: RSI < 35 with deeper pullback thresholds")
    for pb in [0.03, 0.04, 0.05, 0.06]:
        s = RSIPullbackStrategy(rsi_oversold=35.0, pullback_threshold=pb)
        r = run_on_all(s, data)
        results.append((f"RSI<35 + {pb:.0%} pullback", r))

    # ---- Sweep 4: RSI + wider buffers ----
    print("Sweep 4: RSI < 35 with wider buffers")
    for buf in [0.10, 0.12, 0.15]:
        s = RSIPullbackStrategy(rsi_oversold=35.0, buffer=buf)
        r = run_on_all(s, data)
        results.append((f"RSI<35 + {buf:.0%} buffer", r))

    # ---- Combination: RSI + Momentum SMA ----
    # Build a combined strategy class inline
    print("Sweep 5: Combined RSI + Momentum filters")

    class RSIMomentumPullback:
        """Combine RSI oversold + SMA trend filter + pullback."""
        def __init__(self, rsi_oversold=35.0, trend_sma=50, pullback_threshold=0.03):
            self.rsi_strat = RSIPullbackStrategy(rsi_oversold=rsi_oversold,
                                                  pullback_threshold=pullback_threshold)
            self.trend_sma = trend_sma
            self.pullback_threshold = pullback_threshold

        def run(self, df, max_contracts=10, close_col="close"):
            from tradelab.strategies.rsi_pullback import compute_rsi
            import numpy as np

            # Pre-filter: only keep entries where price > SMA
            sma = df[close_col].rolling(self.trend_sma).mean()

            # Run RSI strategy but post-filter trades that were below SMA
            result = self.rsi_strat.run(df, max_contracts=max_contracts)

            # Re-check each trade against SMA at entry
            filtered = []
            for t in result.trade_log:
                entry_date = t["date"]
                # Find index in df
                idx = df.index.get_indexer(
                    [int(entry_date.timestamp())], method="nearest"
                )[0]
                if idx >= self.trend_sma and not np.isnan(sma.iloc[idx]):
                    if df[close_col].iloc[idx] >= sma.iloc[idx]:
                        filtered.append(t)

            from tradelab.strategies.rsi_pullback import RSIPullbackResult
            winners = sum(1 for t in filtered if t["winner"])
            total_pnl = sum(t["pnl"] for t in filtered)
            return RSIPullbackResult(
                total_trades=len(filtered),
                winners=winners,
                losers=len(filtered) - winners,
                total_pnl=total_pnl,
                max_drawdown_pct=result.max_drawdown_pct,
                trade_log=filtered,
            )

    for rsi_thresh in [30, 35, 40]:
        for sma in [50, 100]:
            s = RSIMomentumPullback(rsi_oversold=float(rsi_thresh), trend_sma=sma)
            r = run_on_all(s, data)
            results.append((f"RSI<{rsi_thresh} + {sma}SMA", r))

    # ---- Print all results ----
    print()
    print("=" * 80)
    print("ALL RESULTS (sorted by P/L per trade)")
    print("=" * 80)
    print()

    header = f"{'Config':<30} {'Trades':>6} {'WR':>7} {'Total P/L':>12} {'$/Trade':>10} {'Max DD':>8}"
    print(header)
    print("-" * len(header))

    for name, m in sorted(results, key=lambda x: x[1]["pnl_per_trade"], reverse=True):
        print(
            f"{name:<30} "
            f"{m['trades']:>6} "
            f"{m['win_rate']:>6.1%} "
            f"{m['total_pnl']:>+11,.2f} "
            f"{m['pnl_per_trade']:>+9,.2f} "
            f"{m['max_dd']:>7.1%}"
        )

    # ---- Year-by-year for top 3 new strategies ----
    top_new = [(n, m) for n, m in sorted(results, key=lambda x: x[1]["pnl_per_trade"], reverse=True)
               if n != "Baseline (no RSI)"][:3]

    print()
    print("=" * 80)
    print("TOP 3 NEW STRATEGIES — YEAR-BY-YEAR")
    print("=" * 80)

    for name, m in top_new:
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

    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
