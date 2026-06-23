"""Regime-switching RSI strategy evaluation.

Compares three put credit spread strategies over 2014-2024 on 7 high-beta
names, with a focus on how each handles the 2018 Q4 and 2022 bear regimes
versus calm years.

Strategies:
    A. PullbackEntryStrategy           -- baseline 3% pullback
    B. RSIPullbackStrategy(25)         -- RSI<25 + pullback (rare, bear-only)
    C. RegimeRSIStrategy               -- SPY HV30 switches between A and B

Market regime signal: SPY HV30 (annualized log-return stdev, window=30).
"""

from __future__ import annotations

import pandas as pd

from tradelab.pipeline import DataPipeline
from tradelab.options import historical_volatility
from tradelab.strategies.pullback_entry import PullbackEntryStrategy
from tradelab.strategies.rsi_pullback import RSIPullbackStrategy
from tradelab.strategies.regime_rsi import RegimeRSIStrategy


TICKERS = ["NVDA", "AAPL", "GOOG", "MSFT", "CAT", "AVGO", "META"]
START = "2014-01-01"
END = "2024-12-31"
FOCUS_YEARS = [2018, 2020, 2022, 2023, 2024]


def build_spy_hv30(pipe: DataPipeline) -> pd.Series:
    """Fetch SPY and return HV30 indexed by tz-naive DatetimeIndex."""
    spy = pipe.fetch_stock("SPY", start=START, end=END)
    hv = historical_volatility(spy["close"], window=30)
    # Convert unix-timestamp int index to tz-naive DatetimeIndex
    dt_idx = pd.to_datetime(spy.index.values, unit="s", utc=True).tz_convert(None)
    hv.index = dt_idx
    hv = hv.dropna()
    return hv


def year_of(trade: dict) -> int:
    return pd.Timestamp(trade["date"]).year


def yearly_pnl(trade_log: list[dict]) -> dict[int, float]:
    out: dict[int, float] = {}
    for t in trade_log:
        y = year_of(t)
        out[y] = out.get(y, 0.0) + t["pnl"]
    return out


def yearly_trades(trade_log: list[dict]) -> dict[int, int]:
    out: dict[int, int] = {}
    for t in trade_log:
        y = year_of(t)
        out[y] = out.get(y, 0) + 1
    return out


def main():
    print("=" * 78)
    print("Regime-Switching RSI Put Credit Spread Backtest")
    print("=" * 78)
    print(f"Tickers: {', '.join(TICKERS)}")
    print(f"Period:  {START} to {END}")

    pipe = DataPipeline()
    print("\nFetching SPY for regime signal...")
    spy_hv30 = build_spy_hv30(pipe)
    print(
        f"  SPY HV30: {len(spy_hv30)} days, "
        f"range [{spy_hv30.min():.2%}, {spy_hv30.max():.2%}], "
        f"median {spy_hv30.median():.2%}"
    )
    pct_bear = (spy_hv30 >= 0.25).mean()
    print(f"  Days with HV30 >= 25% (bear regime): {pct_bear:.1%}")

    baseline = PullbackEntryStrategy()
    rsi_only = RSIPullbackStrategy(rsi_oversold=25.0)
    regime = RegimeRSIStrategy()

    # Aggregate accumulators
    agg = {
        "baseline": {"pnl": 0.0, "trades": 0, "wins": 0, "losses": 0, "logs": []},
        "rsi25":    {"pnl": 0.0, "trades": 0, "wins": 0, "losses": 0, "logs": []},
        "regime":   {"pnl": 0.0, "trades": 0, "wins": 0, "losses": 0, "logs": [],
                     "calm": 0, "bear": 0},
    }

    per_ticker = []

    for ticker in TICKERS:
        print(f"\n--- {ticker} ---")
        try:
            df = pipe.fetch_stock(ticker, start=START, end=END)
        except Exception as e:
            print(f"  SKIP: fetch error: {e}")
            continue

        print(f"  rows={len(df)}")

        r_base = baseline.run(df, max_contracts=10)
        r_rsi = rsi_only.run(df, max_contracts=10)
        r_reg = regime.run(df, market_vol_series=spy_hv30, max_contracts=10)

        print(f"  Baseline   : trades={r_base.total_trades:3d}  "
              f"wr={r_base.win_rate:5.1%}  pnl=${r_base.total_pnl:+10,.0f}")
        print(f"  RSI<25     : trades={r_rsi.total_trades:3d}  "
              f"wr={r_rsi.win_rate:5.1%}  pnl=${r_rsi.total_pnl:+10,.0f}")
        print(f"  RegimeRSI  : trades={r_reg.total_trades:3d}  "
              f"wr={r_reg.win_rate:5.1%}  pnl=${r_reg.total_pnl:+10,.0f}  "
              f"(calm={r_reg.calm_trades}, bear={r_reg.bear_trades})")

        per_ticker.append({
            "ticker": ticker,
            "base_pnl": r_base.total_pnl,
            "base_trades": r_base.total_trades,
            "rsi_pnl": r_rsi.total_pnl,
            "rsi_trades": r_rsi.total_trades,
            "reg_pnl": r_reg.total_pnl,
            "reg_trades": r_reg.total_trades,
            "reg_calm": r_reg.calm_trades,
            "reg_bear": r_reg.bear_trades,
        })

        agg["baseline"]["pnl"] += r_base.total_pnl
        agg["baseline"]["trades"] += r_base.total_trades
        agg["baseline"]["wins"] += r_base.winners
        agg["baseline"]["losses"] += r_base.losers
        agg["baseline"]["logs"].extend(r_base.trade_log)

        agg["rsi25"]["pnl"] += r_rsi.total_pnl
        agg["rsi25"]["trades"] += r_rsi.total_trades
        agg["rsi25"]["wins"] += r_rsi.winners
        agg["rsi25"]["losses"] += r_rsi.losers
        agg["rsi25"]["logs"].extend(r_rsi.trade_log)

        agg["regime"]["pnl"] += r_reg.total_pnl
        agg["regime"]["trades"] += r_reg.total_trades
        agg["regime"]["wins"] += r_reg.winners
        agg["regime"]["losses"] += r_reg.losers
        agg["regime"]["logs"].extend(r_reg.trade_log)
        agg["regime"]["calm"] += r_reg.calm_trades
        agg["regime"]["bear"] += r_reg.bear_trades

    # --- Per-ticker summary table ---
    print("\n" + "=" * 78)
    print("PER-TICKER SUMMARY")
    print("=" * 78)
    print(f"{'Ticker':<7} {'Base P/L':>12} {'Base T':>7} "
          f"{'RSI P/L':>12} {'RSI T':>7} "
          f"{'Regime P/L':>13} {'Reg T':>7} {'Calm':>5} {'Bear':>5}")
    print("-" * 78)
    for row in per_ticker:
        print(f"{row['ticker']:<7} "
              f"${row['base_pnl']:>10,.0f} {row['base_trades']:>7d} "
              f"${row['rsi_pnl']:>10,.0f} {row['rsi_trades']:>7d} "
              f"${row['reg_pnl']:>11,.0f} {row['reg_trades']:>7d} "
              f"{row['reg_calm']:>5d} {row['reg_bear']:>5d}")

    # --- Aggregate summary ---
    print("\n" + "=" * 78)
    print("AGGREGATE SUMMARY")
    print("=" * 78)

    def wr(a):
        return a["wins"] / a["trades"] if a["trades"] else 0.0

    print(f"{'Strategy':<14} {'Trades':>8} {'Win Rate':>10} "
          f"{'Total P/L':>14} {'Avg/Trade':>12}")
    print("-" * 70)
    for name, label in [("baseline", "Baseline"),
                        ("rsi25", "RSI<25 only"),
                        ("regime", "RegimeRSI")]:
        a = agg[name]
        avg = a["pnl"] / a["trades"] if a["trades"] else 0.0
        print(f"{label:<14} {a['trades']:>8d} {wr(a):>9.1%} "
              f"${a['pnl']:>+12,.0f} ${avg:>+10,.2f}")

    print(f"\nRegimeRSI mode split: calm={agg['regime']['calm']} "
          f"({agg['regime']['calm']/max(1,agg['regime']['trades']):.1%}) "
          f"| bear={agg['regime']['bear']} "
          f"({agg['regime']['bear']/max(1,agg['regime']['trades']):.1%})")

    # --- Year-by-year breakdown ---
    print("\n" + "=" * 78)
    print("YEAR-BY-YEAR P/L")
    print("=" * 78)
    base_yr = yearly_pnl(agg["baseline"]["logs"])
    rsi_yr = yearly_pnl(agg["rsi25"]["logs"])
    reg_yr = yearly_pnl(agg["regime"]["logs"])
    base_t = yearly_trades(agg["baseline"]["logs"])
    rsi_t = yearly_trades(agg["rsi25"]["logs"])
    reg_t = yearly_trades(agg["regime"]["logs"])

    years = sorted(set(base_yr) | set(rsi_yr) | set(reg_yr))
    print(f"{'Year':<6} "
          f"{'Baseline':>14} {'Bt':>5} "
          f"{'RSI<25':>14} {'Rt':>5} "
          f"{'RegimeRSI':>14} {'Gt':>5}")
    print("-" * 70)
    for y in years:
        marker = "  *" if y in FOCUS_YEARS else "   "
        print(f"{y}{marker:<3} "
              f"${base_yr.get(y, 0):>+12,.0f} {base_t.get(y, 0):>5d} "
              f"${rsi_yr.get(y, 0):>+12,.0f} {rsi_t.get(y, 0):>5d} "
              f"${reg_yr.get(y, 0):>+12,.0f} {reg_t.get(y, 0):>5d}")

    # --- Focus year verdicts ---
    print("\n" + "=" * 78)
    print("FOCUS YEARS (bears: 2018 2022, normals: 2020 2023 2024)")
    print("=" * 78)
    for y in FOCUS_YEARS:
        b = base_yr.get(y, 0.0)
        r = rsi_yr.get(y, 0.0)
        g = reg_yr.get(y, 0.0)
        print(f"  {y}: base=${b:+,.0f}  rsi25=${r:+,.0f}  regime=${g:+,.0f}  "
              f"(regime-base=${g-b:+,.0f}, regime-rsi=${g-r:+,.0f})")

    # --- Verdict ---
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    bp = agg["baseline"]["pnl"]
    rp = agg["rsi25"]["pnl"]
    gp = agg["regime"]["pnl"]
    print(f"  vs baseline : ${gp - bp:+,.0f}")
    print(f"  vs RSI<25   : ${gp - rp:+,.0f}")
    if gp > bp and gp > rp:
        print("  >> RegimeRSI BEATS BOTH components (positive-sum regime switching)")
    elif gp > bp:
        print("  >> RegimeRSI beats baseline but not RSI<25")
    elif gp > rp:
        print("  >> RegimeRSI beats RSI<25 but not baseline")
    else:
        print("  >> RegimeRSI loses to both components")


if __name__ == "__main__":
    main()
