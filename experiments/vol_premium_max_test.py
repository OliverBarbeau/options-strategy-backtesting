"""Backtest VolPremiumMaxStrategy vs baseline vs VolScaledPullback.

Runs three strategies on a 7-ticker, ~11-year universe and reports per-ticker
and aggregate P&L, win rate, and max drawdown. Also reports the VolPremiumMax
band fire counts so we can verify the sweet/ok/skip policy is triggering.

Usage:
    python experiments/vol_premium_max_test.py
"""

from __future__ import annotations

import pandas as pd

from tradelab.pipeline import DataPipeline
from tradelab.strategies.pullback_entry import PullbackEntryStrategy
from tradelab.strategies.volscaled_pullback import VolScaledPullbackStrategy
from tradelab.strategies.vol_premium_max import VolPremiumMaxStrategy


TICKERS = ["NVDA", "AAPL", "GOOG", "MSFT", "CAT", "AVGO", "META"]
START = "2014-01-01"
END = "2024-12-31"
MAX_CONTRACTS = 10


def fmt_money(x: float) -> str:
    return f"${x:+,.0f}"


def run_baseline(df: pd.DataFrame):
    strat = PullbackEntryStrategy()
    return strat.run(df, max_contracts=MAX_CONTRACTS)


def run_volscaled(df: pd.DataFrame):
    # VolScaledPullbackStrategy takes max_contracts in its constructor, not run()
    strat = VolScaledPullbackStrategy(max_contracts=MAX_CONTRACTS)
    return strat.run(df)


def run_volpremium(df: pd.DataFrame):
    strat = VolPremiumMaxStrategy()
    return strat.run(df, max_contracts=MAX_CONTRACTS)


def main() -> None:
    pipe = DataPipeline()

    rows_base: list[dict] = []
    rows_vs: list[dict] = []
    rows_vp: list[dict] = []

    print(f"Backtesting {len(TICKERS)} tickers {START} to {END}")
    print(f"  Strategies: Baseline / VolScaled (loser) / VolPremiumMax (new)")
    print(f"  max_contracts = {MAX_CONTRACTS}")
    print()

    for ticker in TICKERS:
        try:
            df = pipe.fetch_stock(ticker, start=START, end=END)
        except Exception as exc:
            print(f"  [skip] {ticker}: {exc}")
            continue

        r_base = run_baseline(df)
        r_vs = run_volscaled(df)
        r_vp = run_volpremium(df)

        rows_base.append({
            "ticker": ticker,
            "trades": r_base.total_trades,
            "win_rate": r_base.win_rate,
            "total_pnl": r_base.total_pnl,
            "pnl_per_trade": r_base.total_pnl / r_base.total_trades if r_base.total_trades else 0,
            "max_dd_pct": r_base.max_drawdown_pct,
        })
        rows_vs.append({
            "ticker": ticker,
            "trades": r_vs.total_trades,
            "win_rate": r_vs.win_rate,
            "total_pnl": r_vs.total_pnl,
            "pnl_per_trade": r_vs.total_pnl / r_vs.total_trades if r_vs.total_trades else 0,
            "max_dd_pct": r_vs.max_drawdown_pct,
            "avg_contracts": r_vs.avg_contracts,
        })
        rows_vp.append({
            "ticker": ticker,
            "trades": r_vp.total_trades,
            "win_rate": r_vp.win_rate,
            "total_pnl": r_vp.total_pnl,
            "pnl_per_trade": r_vp.total_pnl / r_vp.total_trades if r_vp.total_trades else 0,
            "max_dd_pct": r_vp.max_drawdown_pct,
            "sweet": r_vp.sweet_trades,
            "ok": r_vp.ok_trades,
            "skip": r_vp.skipped_vol_band,
        })

    # ---------- Per-ticker tables ----------
    print("=" * 100)
    print("PER-TICKER P/L")
    print("=" * 100)
    header = f"{'Ticker':<7} {'Baseline':>14} {'VolScaled':>14} {'VolPremiumMax':>16}  {'Base/Tr':>10} {'VS/Tr':>10} {'VP/Tr':>10}"
    print(header)
    print("-" * len(header))
    for b, v, p in zip(rows_base, rows_vs, rows_vp):
        print(
            f"{b['ticker']:<7} "
            f"{fmt_money(b['total_pnl']):>14} "
            f"{fmt_money(v['total_pnl']):>14} "
            f"{fmt_money(p['total_pnl']):>16}  "
            f"{fmt_money(b['pnl_per_trade']):>10} "
            f"{fmt_money(v['pnl_per_trade']):>10} "
            f"{fmt_money(p['pnl_per_trade']):>10}"
        )

    # ---------- Per-ticker trade counts / DD ----------
    print()
    print("=" * 100)
    print("TRADE COUNTS & MAX DRAWDOWN")
    print("=" * 100)
    header2 = f"{'Ticker':<7} {'B#':>5} {'VS#':>5} {'VP#':>5} {'B DD':>8} {'VS DD':>8} {'VP DD':>8} {'B WR':>7} {'VS WR':>7} {'VP WR':>7}"
    print(header2)
    print("-" * len(header2))
    for b, v, p in zip(rows_base, rows_vs, rows_vp):
        print(
            f"{b['ticker']:<7} "
            f"{b['trades']:>5} {v['trades']:>5} {p['trades']:>5} "
            f"{b['max_dd_pct']:>7.1%} {v['max_dd_pct']:>7.1%} {p['max_dd_pct']:>7.1%} "
            f"{b['win_rate']:>6.1%} {v['win_rate']:>6.1%} {p['win_rate']:>6.1%}"
        )

    # ---------- VolPremiumMax band counts ----------
    print()
    print("=" * 100)
    print("VOL-PREMIUM-MAX BAND FIRE COUNTS (per ticker)")
    print("=" * 100)
    header3 = f"{'Ticker':<7} {'Sweet':>8} {'OK':>8} {'Skipped':>10} {'Sweet%':>8}"
    print(header3)
    print("-" * len(header3))
    total_sweet = total_ok = total_skip = 0
    for p in rows_vp:
        taken = p["sweet"] + p["ok"]
        sweet_pct = p["sweet"] / taken if taken else 0
        print(
            f"{p['ticker']:<7} "
            f"{p['sweet']:>8} {p['ok']:>8} {p['skip']:>10} "
            f"{sweet_pct:>7.1%}"
        )
        total_sweet += p["sweet"]
        total_ok += p["ok"]
        total_skip += p["skip"]

    # ---------- Aggregates ----------
    def agg(rows, key):
        return sum(r[key] for r in rows)

    def agg_pnl_per_trade(rows):
        t = agg(rows, "trades")
        return agg(rows, "total_pnl") / t if t else 0

    def worst_dd(rows):
        return min(r["max_dd_pct"] for r in rows) if rows else 0

    print()
    print("=" * 100)
    print("AGGREGATE RESULTS")
    print("=" * 100)
    print(f"{'Strategy':<22} {'Trades':>8} {'Total P/L':>14} {'P/L per trade':>16} {'Worst Max DD':>15}")
    print("-" * 80)
    print(
        f"{'Baseline':<22} "
        f"{agg(rows_base, 'trades'):>8} "
        f"{fmt_money(agg(rows_base, 'total_pnl')):>14} "
        f"{fmt_money(agg_pnl_per_trade(rows_base)):>16} "
        f"{worst_dd(rows_base):>14.1%}"
    )
    print(
        f"{'VolScaled (loser)':<22} "
        f"{agg(rows_vs, 'trades'):>8} "
        f"{fmt_money(agg(rows_vs, 'total_pnl')):>14} "
        f"{fmt_money(agg_pnl_per_trade(rows_vs)):>16} "
        f"{worst_dd(rows_vs):>14.1%}"
    )
    print(
        f"{'VolPremiumMax (new)':<22} "
        f"{agg(rows_vp, 'trades'):>8} "
        f"{fmt_money(agg(rows_vp, 'total_pnl')):>14} "
        f"{fmt_money(agg_pnl_per_trade(rows_vp)):>16} "
        f"{worst_dd(rows_vp):>14.1%}"
    )

    print()
    print(
        f"VolPremiumMax aggregate bands: "
        f"sweet={total_sweet}  ok={total_ok}  skipped={total_skip}  "
        f"(sweet% of taken = {total_sweet / (total_sweet + total_ok):.1%})"
    )


if __name__ == "__main__":
    main()
