"""Per-ticker hybrid strategy test.

Compares PerTickerHybridStrategy (which dispatches to the empirically-best
exit rule per ticker, from Experiment 6) against the baseline
PullbackEntryStrategy (3% pullback, always close at 14 DTE) on the same
universe, same period, same pricing (inline Black-Scholes).

Period: 2014-01-01 to 2024-12-31.

The tickers in the per-ticker config are:
    QQQ, AVGO, CAT, MSFT, NVDA, AAPL, SPY, META, AMD, GOOG
We attempt all of them; any ticker with insufficient data is skipped.
"""

from __future__ import annotations

import os
import sys
import time

import pandas as pd

from tradelab.pipeline import DataPipeline
from tradelab.strategies.pullback_entry import PullbackEntryStrategy
from tradelab.strategies.per_ticker_hybrid import (
    PerTickerHybridStrategy,
    _TICKER_CONFIGS,
)


TICKERS = ["NVDA", "AAPL", "GOOG", "MSFT", "CAT", "AVGO", "META", "QQQ", "SPY", "AMD"]
START = "2014-01-01"
END = "2024-12-31"
MAX_CONTRACTS = 10

# Resolve data cache path: if running from a git worktree that has no
# local cache, fall back to the main repo's cache. Also honor an explicit
# override via TRADELAB_CACHE env var.
_DEFAULT_CACHE = os.environ.get("TRADELAB_CACHE")
if _DEFAULT_CACHE is None:
    here = os.path.dirname(os.path.abspath(__file__))
    local_cache = os.path.normpath(os.path.join(here, "..", "data", "cache"))
    fallback_cache = r"D:\code\backtesting-26\data\cache"
    if os.path.isdir(os.path.join(local_cache, "yfinance")) and os.listdir(
        os.path.join(local_cache, "yfinance")
    ):
        _DEFAULT_CACHE = local_cache
    elif os.path.isdir(fallback_cache):
        _DEFAULT_CACHE = fallback_cache
    else:
        _DEFAULT_CACHE = local_cache


def load_data(tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    pipe = DataPipeline(cache_dir=_DEFAULT_CACHE)
    data = {}
    for t in tickers:
        try:
            df = pipe.fetch_stock(t, start=start, end=end)
        except Exception as e:
            print(f"  {t}: ERROR ({e})")
            continue
        if df is not None and len(df) > 60:
            data[t] = df
            print(f"  {t}: {len(df)} bars")
        else:
            print(f"  {t}: SKIPPED (insufficient data)")
    return data


def describe_config(ticker: str) -> str:
    cfg = _TICKER_CONFIGS.get(ticker)
    if cfg is None:
        return "default (always_close 14 DTE)"
    mode = cfg["mode"]
    if mode == "always_close":
        return f"always_close {cfg['dte_close']}d"
    if mode == "hold":
        return "hold to expiry"
    if mode == "hybrid":
        cp = cfg["checkpoint_drop_pct"]
        stop = cfg.get("emergency_stop_pct")
        stop_str = f"{int(stop*100)}% stop" if stop is not None else "no stop"
        return f"hybrid {int(cp*100)}% chkpt / {stop_str}"
    return mode


def main():
    t0 = time.time()
    print("=" * 80)
    print("PER-TICKER HYBRID STRATEGY TEST")
    print(f"Period: {START} to {END}")
    print("=" * 80)
    print()

    print("Loading data...")
    data = load_data(TICKERS, START, END)
    if not data:
        print("No data loaded. Exiting.")
        sys.exit(1)
    print(f"  Loaded {len(data)}/{len(TICKERS)} tickers")
    print()

    # --- Run strategies ---
    baseline = PullbackEntryStrategy()
    per_ticker_results: dict[str, object] = {}
    baseline_results: dict[str, object] = {}

    for ticker, df in data.items():
        strat = PerTickerHybridStrategy(ticker=ticker)
        per_ticker_results[ticker] = strat.run(df, max_contracts=MAX_CONTRACTS)
        baseline_results[ticker] = baseline.run(df, max_contracts=MAX_CONTRACTS)

    # --- Per-ticker breakdown ---
    print("=" * 100)
    print("PER-TICKER BREAKDOWN")
    print("=" * 100)
    header = (
        f"{'Ticker':<7} {'ExitPolicy':<28} "
        f"{'PT Trd':>6} {'PT WR':>6} {'PT P/L':>11} {'PT $/t':>8}  "
        f"{'BL Trd':>6} {'BL WR':>6} {'BL P/L':>11} {'BL $/t':>8}"
    )
    print(header)
    print("-" * len(header))

    pt_total_trades = 0
    pt_total_winners = 0
    pt_total_pnl = 0.0
    bl_total_trades = 0
    bl_total_winners = 0
    bl_total_pnl = 0.0

    ordered = [t for t in _TICKER_CONFIGS.keys() if t in data] + [
        t for t in data.keys() if t not in _TICKER_CONFIGS
    ]

    for ticker in ordered:
        pt = per_ticker_results[ticker]
        bl = baseline_results[ticker]

        pt_ppt = pt.total_pnl / pt.total_trades if pt.total_trades > 0 else 0.0
        bl_ppt = bl.total_pnl / bl.total_trades if bl.total_trades > 0 else 0.0

        print(
            f"{ticker:<7} {describe_config(ticker):<28} "
            f"{pt.total_trades:>6} {pt.win_rate:>5.1%} "
            f"{pt.total_pnl:>+11,.0f} {pt_ppt:>+8,.0f}  "
            f"{bl.total_trades:>6} {bl.win_rate:>5.1%} "
            f"{bl.total_pnl:>+11,.0f} {bl_ppt:>+8,.0f}"
        )

        pt_total_trades += pt.total_trades
        pt_total_winners += pt.winners
        pt_total_pnl += pt.total_pnl
        bl_total_trades += bl.total_trades
        bl_total_winners += bl.winners
        bl_total_pnl += bl.total_pnl

    print("-" * len(header))

    pt_agg_wr = pt_total_winners / pt_total_trades if pt_total_trades > 0 else 0.0
    bl_agg_wr = bl_total_winners / bl_total_trades if bl_total_trades > 0 else 0.0
    pt_agg_ppt = pt_total_pnl / pt_total_trades if pt_total_trades > 0 else 0.0
    bl_agg_ppt = bl_total_pnl / bl_total_trades if bl_total_trades > 0 else 0.0

    print(
        f"{'TOTAL':<7} {'(mixed)':<28} "
        f"{pt_total_trades:>6} {pt_agg_wr:>5.1%} "
        f"{pt_total_pnl:>+11,.0f} {pt_agg_ppt:>+8,.0f}  "
        f"{bl_total_trades:>6} {bl_agg_wr:>5.1%} "
        f"{bl_total_pnl:>+11,.0f} {bl_agg_ppt:>+8,.0f}"
    )

    # --- Aggregate summary ---
    print()
    print("=" * 80)
    print("AGGREGATE SUMMARY")
    print("=" * 80)
    print(
        f"{'Strategy':<28} {'Trades':>7} {'Win Rate':>10} "
        f"{'Total P/L':>14} {'$/Trade':>12}"
    )
    print("-" * 80)
    print(
        f"{'PerTickerHybrid':<28} {pt_total_trades:>7} "
        f"{pt_agg_wr:>9.1%}  {pt_total_pnl:>+13,.2f} {pt_agg_ppt:>+11,.2f}"
    )
    print(
        f"{'Baseline Pullback':<28} {bl_total_trades:>7} "
        f"{bl_agg_wr:>9.1%}  {bl_total_pnl:>+13,.2f} {bl_agg_ppt:>+11,.2f}"
    )

    delta = pt_total_pnl - bl_total_pnl
    pct_improvement = (delta / abs(bl_total_pnl) * 100) if bl_total_pnl != 0 else 0.0
    print("-" * 80)
    print(
        f"{'DELTA (PT - BL)':<28} {pt_total_trades - bl_total_trades:>+7} "
        f"{'':>10}  {delta:>+13,.2f} {'':>12}"
    )
    print(f"  P/L improvement: {pct_improvement:+.1f}% vs baseline")
    print()

    # --- Exit-reason breakdown ---
    print("=" * 80)
    print("EXIT REASON BREAKDOWN (PerTickerHybrid)")
    print("=" * 80)
    for ticker in ordered:
        pt = per_ticker_results[ticker]
        if pt.exit_reason_counts:
            reasons = ", ".join(
                f"{k}={v}" for k, v in sorted(pt.exit_reason_counts.items())
            )
            print(f"  {ticker:<7} [{pt.exit_mode:<13}] {reasons}")

    # --- Win/loss list ---
    print()
    print("=" * 80)
    print("PER-TICKER WIN/LOSS vs BASELINE")
    print("=" * 80)
    wins = []
    losses = []
    for ticker in ordered:
        pt = per_ticker_results[ticker]
        bl = baseline_results[ticker]
        diff = pt.total_pnl - bl.total_pnl
        if diff > 0:
            wins.append((ticker, diff))
        elif diff < 0:
            losses.append((ticker, diff))
    print(f"  PT beat baseline on {len(wins)}/{len(ordered)} tickers")
    for ticker, d in sorted(wins, key=lambda x: -x[1]):
        print(f"    +{d:>10,.0f}   {ticker}")
    print(f"  PT trailed baseline on {len(losses)}/{len(ordered)} tickers")
    for ticker, d in sorted(losses, key=lambda x: x[1]):
        print(f"    {d:>+11,.0f}   {ticker}")

    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
