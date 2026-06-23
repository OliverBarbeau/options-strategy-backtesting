"""Bear detection: which signals actually improve outcomes over 5 years?

The key test isn't "did the signal fire before the bear" — it's
"does pausing trading on this signal produce a better 5-year result
than not pausing, after accounting for false positives?"

Approach: for each signal, replay all 398 real trades from 2018-2024.
On each trade's entry date, check if the signal is active. If active,
skip the trade (assume we paused). Compute the resulting equity curve.

This gives us the TRUE impact: bear trades avoided MINUS bull trades missed.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from tradelab.pipeline import DataPipeline
from tradelab.account import SimulatedAccount


def load_spy_indicators(start="2017-06-01", end="2025-01-01"):
    """Build SPY indicator DataFrame."""
    pipe = DataPipeline()
    spy = pipe.fetch_stock("SPY", start=start, end=end)
    spy.index = pd.Index(
        [pd.Timestamp(ts, unit="s").strftime("%Y-%m-%d") for ts in spy.index],
        name="date",
    )

    spy["sma50"] = spy["close"].rolling(50).mean()
    spy["sma200"] = spy["close"].rolling(200).mean()
    spy["high_52w"] = spy["close"].rolling(252).max()
    spy["dd_from_high"] = (spy["close"] - spy["high_52w"]) / spy["high_52w"]
    spy["roc_10"] = spy["close"].pct_change(10)
    spy["roc_20"] = spy["close"].pct_change(20)
    spy["hv30"] = (
        np.log(spy["close"] / spy["close"].shift(1)).rolling(30).std() * np.sqrt(252)
    )

    # Trailing strategy P/L (computed from our actual trades, added later)
    return spy


def load_all_trades():
    """Load all trades from 5 years, sorted by entry date."""
    all_trades = []
    for year in ["2018", "2019", "2022", "2023", "2024"]:
        try:
            acct = SimulatedAccount.load(f"accounts/portfolio_theta_{year}.json")
            for t in acct.trades:
                all_trades.append({
                    "entry_date": t.entry_date[:10],
                    "exit_date": t.exit_date[:10],
                    "ticker": t.ticker,
                    "pnl": t.pnl,
                    "winner": t.winner,
                    "collateral": t.collateral,
                })
        except Exception as e:
            print(f"Warning: {year}: {e}")

    return sorted(all_trades, key=lambda t: t["entry_date"])


def simulate_with_signal(trades, spy, signal_fn, starting_capital=25000):
    """Replay trades, skipping those where signal says 'pause'.

    Returns dict with equity curve and metrics.
    """
    equity = starting_capital
    peak = equity
    max_dd = 0.0
    taken = []
    skipped = 0

    for t in trades:
        date = t["entry_date"]
        if date not in spy.index:
            # Can't check signal, take the trade
            equity += t["pnl"]
            taken.append(t)
            continue

        row = spy.loc[date]
        if signal_fn(row):
            # Signal says pause — skip this trade
            skipped += 1
            continue

        equity += t["pnl"]
        taken.append(t)
        peak = max(peak, equity)
        dd = (equity - peak) / peak if peak > 0 else 0
        max_dd = min(max_dd, dd)

    winners = sum(1 for t in taken if t["winner"])
    total = len(taken)

    return {
        "final_equity": equity,
        "total_return": (equity / starting_capital - 1) * 100,
        "trades": total,
        "skipped": skipped,
        "wr": winners / total if total else 0,
        "max_dd": max_dd,
        "winners": winners,
        "losers": total - winners,
    }


def main():
    W = 110

    print("Loading data...")
    spy = load_spy_indicators()
    trades = load_all_trades()
    print(f"  {len(trades)} trades from 5 years")
    print(f"  SPY indicators: {len(spy)} days")

    # Add trailing strategy performance to SPY for strategy-internal signals
    # Compute trailing 5-trade and 10-trade win rate at each date
    trailing_wr = {}
    recent_trades = []
    for t in trades:
        recent_trades.append(t)
        if len(recent_trades) >= 5:
            last5 = recent_trades[-5:]
            trailing_wr[t["entry_date"]] = {
                "wr5": sum(1 for x in last5 if x["winner"]) / 5,
                "pnl5": sum(x["pnl"] for x in last5),
            }
        if len(recent_trades) >= 10:
            last10 = recent_trades[-10:]
            trailing_wr[t["entry_date"]]["wr10"] = sum(1 for x in last10 if x["winner"]) / 10
            trailing_wr[t["entry_date"]]["pnl10"] = sum(x["pnl"] for x in last10)

    # Define signals to test
    signals = {
        # Market technicals
        "SPY < 50-SMA":            lambda r: r["close"] < r.get("sma50", r["close"]),
        "SPY < 200-SMA":           lambda r: r["close"] < r.get("sma200", r["close"]),
        "50-SMA < 200-SMA":        lambda r: r.get("sma50", 1) < r.get("sma200", 0),
        "DD > 5%":                 lambda r: r.get("dd_from_high", 0) < -0.05,
        "DD > 7%":                 lambda r: r.get("dd_from_high", 0) < -0.07,
        "DD > 10%":                lambda r: r.get("dd_from_high", 0) < -0.10,
        "10d mom < -3%":           lambda r: r.get("roc_10", 0) < -0.03,
        "20d mom < -5%":           lambda r: r.get("roc_20", 0) < -0.05,
        "HV30 > 20%":             lambda r: r.get("hv30", 0) > 0.20,
        "HV30 > 25%":             lambda r: r.get("hv30", 0) > 0.25,

        # Combo signals
        "SPY<50 AND DD>5%":        lambda r: r["close"] < r.get("sma50", r["close"]) and r.get("dd_from_high", 0) < -0.05,
        "SPY<200 AND DD>7%":       lambda r: r["close"] < r.get("sma200", r["close"]) and r.get("dd_from_high", 0) < -0.07,
        "SPY<50 AND HV>20%":      lambda r: r["close"] < r.get("sma50", r["close"]) and r.get("hv30", 0) > 0.20,
        "DD>5% AND 20dmom<-3%":   lambda r: r.get("dd_from_high", 0) < -0.05 and r.get("roc_20", 0) < -0.03,
        "DD>7% AND HV>20%":       lambda r: r.get("dd_from_high", 0) < -0.07 and r.get("hv30", 0) > 0.20,
    }

    # Also test strategy-internal signals
    # These need a wrapper that checks trailing_wr dict
    def make_internal_signal(key, threshold, direction="below"):
        def signal_fn(row):
            date = row.name if hasattr(row, "name") else ""
            tw = trailing_wr.get(date, {})
            val = tw.get(key, 0.5 if "wr" in key else 0)
            if direction == "below":
                return val < threshold
            return val > threshold
        return signal_fn

    internal_signals = {
        "Trail 5-WR < 40%":   make_internal_signal("wr5", 0.40),
        "Trail 5-WR < 50%":   make_internal_signal("wr5", 0.50),
        "Trail 5-WR < 60%":   make_internal_signal("wr5", 0.60),
        "Trail 10-WR < 50%":  make_internal_signal("wr10", 0.50),
        "Trail 5-PnL < 0":    make_internal_signal("pnl5", 0, "below"),
        "Trail 10-PnL < 0":   make_internal_signal("pnl10", 0, "below"),
    }

    all_signals = {**signals, **internal_signals}

    # Baseline: no signal (take all trades)
    baseline = simulate_with_signal(trades, spy, lambda r: False)

    print()
    print("=" * W)
    print(f"{'BEAR DETECTION SIGNALS: WHICH ONES IMPROVE 5-YEAR OUTCOME?':^{W}}")
    print(f"{'(398 real trades from Theta Data, 2018-2024)':^{W}}")
    print("=" * W)
    print(f"\n  Baseline: {baseline['trades']} trades, {baseline['wr']:.0%} WR, "
          f"${baseline['final_equity']:,.0f} ({baseline['total_return']:+.1f}%)")
    print()

    print(f"{'Signal':<28} {'Trades':>7} {'Skip':>5} {'WR':>6} {'Final $':>10} {'Return':>8} {'MaxDD':>7} {'vs Base':>9}")
    print("-" * W)

    results = []
    for name, sig_fn in all_signals.items():
        r = simulate_with_signal(trades, spy, sig_fn)
        delta = r["final_equity"] - baseline["final_equity"]
        results.append((name, r, delta))

    # Sort by final equity (best first)
    results.sort(key=lambda x: x[2], reverse=True)

    for name, r, delta in results:
        marker = " +" if delta > 500 else " -" if delta < -500 else ""
        print(f"{name:<28} {r['trades']:>7} {r['skipped']:>5} {r['wr']:>5.0%} "
              f"${r['final_equity']:>9,.0f} {r['total_return']:>+7.1f}% "
              f"{r['max_dd']*100:>6.1f}% ${delta:>+8,.0f}{marker}")

    # Top 5 analysis
    print()
    print("=" * W)
    print(f"{'TOP 5 SIGNALS: DETAILED YEAR-BY-YEAR IMPACT':^{W}}")
    print("=" * W)

    top5 = results[:5]
    for name, r, delta in top5:
        print(f"\n  {name} (${delta:+,.0f} vs baseline):")
        # Replay per-year
        for year in ["2018", "2019", "2022", "2023", "2024"]:
            year_trades = [t for t in trades if t["entry_date"].startswith(year)]
            yr_base = simulate_with_signal(year_trades, spy, lambda r: False)
            yr_sig = simulate_with_signal(year_trades, spy, all_signals[name])
            yr_delta = yr_sig["final_equity"] - yr_base["final_equity"]
            saved = yr_sig["skipped"]
            print(f"    {year}: base ${yr_base['total_return']:>+6.1f}% -> signal ${yr_sig['total_return']:>+6.1f}%  "
                  f"({saved} trades skipped, ${yr_delta:>+7,.0f})")

    # Summary
    best = results[0]
    print()
    print("=" * W)
    print(f"{'BEST SIGNAL: ' + best[0]:^{W}}")
    print("=" * W)
    print(f"  5-year improvement: ${best[2]:+,.0f}")
    print(f"  Trades taken: {best[1]['trades']} (skipped {best[1]['skipped']})")
    print(f"  Win rate: {best[1]['wr']:.0%} (baseline: {baseline['wr']:.0%})")
    print(f"  Max drawdown: {best[1]['max_dd']*100:.1f}% (baseline: {baseline['max_dd']*100:.1f}%)")

    return 0


if __name__ == "__main__":
    main()
