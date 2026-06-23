"""Strategy shootout: head-to-head comparison of all pullback variants.

Runs each strategy on the same tickers over the same 10-year period
(2014-2024) using inline Black-Scholes pricing. This is a relative
comparison — absolute P/L values are B-S approximate (10-30% optimistic),
but the *ranking* between strategies is reliable since they all use the
same pricing model.

Strategies compared:
1. Baseline Pullback (3% threshold, no filters)
2. Momentum Pullback (pullback + 50 SMA trend filter)
3. Dual Momentum (pullback + 50 SMA + 200 SMA)
4. RSI Pullback (pullback + RSI < 35)
5. RSI Pullback strict (pullback + RSI < 30, floor at 15)
6. Vol-Scaled Pullback (tiered position sizing)
7. Vol-Targeted Pullback (continuous vol-targeting at 25%)
8. Gap Recovery (2%+ single-day drops)
9. Gap Recovery + Volume (gap-down with above-avg volume)
10. Adaptive Pullback (all 5 safety features on)

Output: side-by-side table of trades, win rate, P/L, max DD, P/L per trade.
"""

from __future__ import annotations

import sys
import time

import pandas as pd

from tradelab.pipeline import DataPipeline
from tradelab.options import historical_volatility

# Strategies
from tradelab.strategies.pullback_entry import PullbackEntryStrategy
from tradelab.strategies.momentum_pullback import MomentumPullbackStrategy
from tradelab.strategies.rsi_pullback import RSIPullbackStrategy
from tradelab.strategies.volscaled_pullback import VolScaledPullbackStrategy
from tradelab.strategies.gap_recovery import GapRecoveryStrategy
from tradelab.strategies.adaptive_pullback import AdaptivePullbackStrategy


# ---- Configuration ----
TICKERS = ["NVDA", "AAPL", "GOOG", "MSFT", "CAT", "AVGO"]
START = "2014-01-01"
END = "2024-12-31"
MAX_CONTRACTS = 10


def load_data(tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    """Load OHLCV data for all tickers."""
    pipe = DataPipeline()
    data = {}
    for t in tickers:
        df = pipe.fetch_stock(t, start=start, end=end)
        if df is not None and len(df) > 60:
            data[t] = df
            print(f"  {t}: {len(df)} bars")
        else:
            print(f"  {t}: SKIPPED (insufficient data)")
    return data


def build_strategies() -> list[tuple[str, object, dict]]:
    """Return (name, strategy_instance, extra_run_kwargs) tuples."""
    return [
        # 1. Baseline
        (
            "Pullback (baseline)",
            PullbackEntryStrategy(),
            {},
        ),
        # 2. Momentum: pullback + 50 SMA
        (
            "Momentum (50 SMA)",
            MomentumPullbackStrategy(trend_sma=50),
            {},
        ),
        # 3. Dual momentum: pullback + 50 SMA + 200 SMA
        (
            "Dual Momentum (50+200)",
            MomentumPullbackStrategy(trend_sma=50, long_sma=200),
            {},
        ),
        # 4. RSI: pullback + RSI < 35
        (
            "RSI Pullback (<35)",
            RSIPullbackStrategy(rsi_oversold=35.0),
            {},
        ),
        # 5. RSI strict: pullback + RSI < 30, crash floor at 15
        (
            "RSI Strict (<30, floor 15)",
            RSIPullbackStrategy(rsi_oversold=30.0, rsi_extreme_floor=15.0),
            {},
        ),
        # 6. Vol-scaled: tiered sizing
        (
            "Vol-Scaled (tiered)",
            VolScaledPullbackStrategy(),
            {},
        ),
        # 7. Vol-targeted: continuous sizing targeting 25% vol
        (
            "Vol-Targeted (25%)",
            VolScaledPullbackStrategy(vol_target=0.25),
            {},
        ),
        # 8. Gap recovery: 2%+ single-day drops
        (
            "Gap Recovery (2%+)",
            GapRecoveryStrategy(gap_threshold=0.02),
            {},
        ),
        # 9. Gap recovery + volume confirmation
        (
            "Gap + Volume",
            GapRecoveryStrategy(gap_threshold=0.02, require_volume=True),
            {},
        ),
        # 10. Adaptive: all safety features on
        (
            "Adaptive (all on)",
            AdaptivePullbackStrategy(
                vol_pause_threshold=0.30,
                cooldown_days=10,
                adaptive_pullback=True,
                stop_loss_breach=True,
                fast_profit_target=0.75,
            ),
            {"needs_market_vol": True},
        ),
    ]


def run_strategy(name, strategy, df, **kwargs):
    """Run a strategy and return its result, handling different interfaces."""
    if isinstance(strategy, (PullbackEntryStrategy,)):
        return strategy.run(df, max_contracts=MAX_CONTRACTS)
    elif isinstance(strategy, (MomentumPullbackStrategy, RSIPullbackStrategy)):
        return strategy.run(df, max_contracts=MAX_CONTRACTS)
    elif isinstance(strategy, VolScaledPullbackStrategy):
        return strategy.run(df)
    elif isinstance(strategy, GapRecoveryStrategy):
        return strategy.run(df)
    elif isinstance(strategy, AdaptivePullbackStrategy):
        market_vol = kwargs.get("market_vol")
        return strategy.run(df, market_vol_series=market_vol, max_contracts=MAX_CONTRACTS)
    else:
        return strategy.run(df, max_contracts=MAX_CONTRACTS)


def main():
    t0 = time.time()
    print("=" * 80)
    print("STRATEGY SHOOTOUT: 10 variants x 6 tickers x 10 years")
    print("=" * 80)
    print()

    # ---- Load data ----
    print("Loading data...")
    data = load_data(TICKERS, START, END)
    if not data:
        print("No data loaded. Exiting.")
        sys.exit(1)
    print()

    # Load SPY for market vol (needed by adaptive strategy)
    pipe = DataPipeline()
    spy_df = pipe.fetch_stock("SPY", start=START, end=END)
    spy_vol = historical_volatility(spy_df["close"], window=30) if spy_df is not None else None

    # ---- Build strategies ----
    strategies = build_strategies()

    # ---- Run all strategies on all tickers ----
    # Accumulate results: {strategy_name: {metric: value}}
    agg = {}

    for sname, strategy, extra in strategies:
        total_trades = 0
        total_winners = 0
        total_losers = 0
        total_pnl = 0.0
        worst_dd = 0.0
        all_trades = []

        for ticker, df in data.items():
            kwargs = {}
            if extra.get("needs_market_vol") and spy_vol is not None:
                kwargs["market_vol"] = spy_vol

            try:
                result = run_strategy(sname, strategy, df, **kwargs)
            except Exception as e:
                print(f"  ERROR: {sname} on {ticker}: {e}")
                continue

            total_trades += result.total_trades
            total_winners += result.winners
            total_losers += result.losers
            total_pnl += result.total_pnl
            worst_dd = min(worst_dd, result.max_drawdown_pct)
            all_trades.extend(result.trade_log)

        win_rate = total_winners / total_trades if total_trades > 0 else 0
        pnl_per_trade = total_pnl / total_trades if total_trades > 0 else 0

        agg[sname] = {
            "trades": total_trades,
            "winners": total_winners,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "pnl_per_trade": pnl_per_trade,
            "max_dd": worst_dd,
        }

    # ---- Print results table ----
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()

    header = f"{'Strategy':<28} {'Trades':>6} {'WR':>7} {'Total P/L':>12} {'$/Trade':>10} {'Max DD':>8}"
    print(header)
    print("-" * len(header))

    # Sort by total P/L descending
    for sname, m in sorted(agg.items(), key=lambda x: x[1]["total_pnl"], reverse=True):
        print(
            f"{sname:<28} "
            f"{m['trades']:>6} "
            f"{m['win_rate']:>6.1%} "
            f"{m['total_pnl']:>+11,.2f} "
            f"{m['pnl_per_trade']:>+9,.2f} "
            f"{m['max_dd']:>7.1%}"
        )

    print()

    # ---- Per-ticker breakdown for top 3 ----
    top3 = sorted(agg.items(), key=lambda x: x[1]["total_pnl"], reverse=True)[:3]
    print("=" * 80)
    print("TOP 3 — PER-TICKER BREAKDOWN")
    print("=" * 80)

    for sname, _ in top3:
        # Re-run to get per-ticker detail (fast with B-S)
        strategy = None
        extra = {}
        for n, s, e in strategies:
            if n == sname:
                strategy = s
                extra = e
                break

        print(f"\n  {sname}:")
        print(f"  {'Ticker':<8} {'Trades':>6} {'WR':>7} {'P/L':>12} {'$/Trade':>10}")
        print(f"  {'-'*48}")

        for ticker, df in data.items():
            kwargs = {}
            if extra.get("needs_market_vol") and spy_vol is not None:
                kwargs["market_vol"] = spy_vol

            try:
                result = run_strategy(sname, strategy, df, **kwargs)
            except Exception:
                continue

            wr = result.win_rate
            ppt = result.total_pnl / result.total_trades if result.total_trades > 0 else 0
            print(
                f"  {ticker:<8} "
                f"{result.total_trades:>6} "
                f"{wr:>6.1%} "
                f"{result.total_pnl:>+11,.2f} "
                f"{ppt:>+9,.2f}"
            )

    # ---- Year-by-year for baseline vs best new ----
    best_new_name = None
    for sname, m in sorted(agg.items(), key=lambda x: x[1]["total_pnl"], reverse=True):
        if sname != "Pullback (baseline)":
            best_new_name = sname
            break

    if best_new_name:
        print()
        print("=" * 80)
        print(f"YEAR-BY-YEAR: Baseline vs {best_new_name}")
        print("=" * 80)

        for sname in ["Pullback (baseline)", best_new_name]:
            strategy = None
            extra = {}
            for n, s, e in strategies:
                if n == sname:
                    strategy = s
                    extra = e
                    break

            all_trades = []
            for ticker, df in data.items():
                kwargs = {}
                if extra.get("needs_market_vol") and spy_vol is not None:
                    kwargs["market_vol"] = spy_vol
                try:
                    result = run_strategy(sname, strategy, df, **kwargs)
                    all_trades.extend(result.trade_log)
                except Exception:
                    continue

            print(f"\n  {sname}:")
            print(f"  {'Year':<6} {'Trades':>6} {'WR':>7} {'P/L':>12}")
            print(f"  {'-'*35}")

            trades_df = pd.DataFrame(all_trades)
            if not trades_df.empty:
                trades_df["year"] = trades_df["date"].dt.year
                for year in sorted(trades_df["year"].unique()):
                    yr = trades_df[trades_df["year"] == year]
                    wr = yr["winner"].sum() / len(yr) if len(yr) > 0 else 0
                    print(f"  {year:<6} {len(yr):>6} {wr:>6.1%} {yr['pnl'].sum():>+11,.2f}")

    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
