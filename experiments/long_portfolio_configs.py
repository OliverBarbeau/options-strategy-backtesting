"""Long portfolio simulations: 7 years of real data, multiple configs.

Runs each config compounded through 2018-2024 (7 calendar years, 5 with
option data: 2018, 2019, 2022, 2023, 2024). Reports per-year returns and
the compounded final capital.

Configs tested:
  1. Baseline (no protection)
  2. 20% loss limit (best from bear_protection_deep.py)
  3. 7% buffer + 20% loss limit (best overall from same experiment)
  4. Trail 5 PnL filter (best from bear_detection.py, but needs validation)
  5. 20% loss limit + drawdown scaling
  6. 13% buffer (widest we tested — most conservative)
  7. 7% buffer only (no loss limit — risk-on)
  8. 20% loss limit + 7% buffer + drawdown scaling
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

from tradelab.account import SimulatedAccount
from tradelab.portfolio_simulator import PortfolioSimulator, PortfolioConfig
from tradelab.pricing.thetadata import ThetaDataProvider


TICKERS = ["NVDA", "AVGO", "MSFT", "GOOG", "CAT", "AAPL"]
YEARS = ["2018", "2019", "2022", "2023", "2024"]
START_CAPITAL = 25000.0


def run_compounded(name: str, provider: ThetaDataProvider, **overrides) -> dict:
    capital = START_CAPITAL
    per_year = []

    for year in YEARS:
        path = f"accounts/_long_{name}_{year}.json"
        if os.path.exists(path):
            os.remove(path)

        acct = SimulatedAccount.load_or_create(
            path, starting_capital=capital, name=f"{name}_{year}", strategy="test",
        )

        config = PortfolioConfig(
            tickers=TICKERS,
            start_date=f"{year}-01-02",
            end_date=f"{year}-12-31",
            starting_capital=capital,
            **overrides,
        )

        sim = PortfolioSimulator(acct, provider, config, verbose=False)
        sim.run()

        ending = acct.equity
        pnl = ending - capital
        ret = pnl / capital * 100 if capital > 0 else 0

        equities = [s.equity for s in acct.equity_curve]
        max_dd = 0.0
        if equities:
            peak = equities[0]
            for e in equities:
                peak = max(peak, e)
                dd = (e - peak) / peak
                max_dd = min(max_dd, dd)

        per_year.append({
            "year": year, "start": capital, "end": ending,
            "ret": ret, "trades": len(acct.trades),
            "wr": acct.win_rate, "dd": max_dd,
        })
        capital = ending

        try:
            os.remove(path)
        except:
            pass

    return {
        "name": name, "final": capital,
        "total_ret": (capital / START_CAPITAL - 1) * 100,
        "per_year": per_year,
    }


def main():
    provider = ThetaDataProvider()
    if not provider.check_connection():
        print("Theta Terminal not reachable.")
        return 1

    configs = [
        ("BASELINE",                {}),
        ("buf10_loss20",            {"max_portfolio_loss_pct": 0.20}),
        ("buf7_loss20",             {"buffer": 0.07, "max_portfolio_loss_pct": 0.20}),
        ("buf7_only",               {"buffer": 0.07}),
        ("buf13_only",              {"buffer": 0.13}),
        ("buf10_loss20_dd",         {"max_portfolio_loss_pct": 0.20, "drawdown_scaling": True}),
        ("buf7_loss20_dd",          {"buffer": 0.07, "max_portfolio_loss_pct": 0.20, "drawdown_scaling": True}),
        ("buf10_loss15",            {"max_portfolio_loss_pct": 0.15}),
        ("buf10_sma200",            {"trend_sma_days": 200}),
        ("buf7_loss20_sma200",      {"buffer": 0.07, "max_portfolio_loss_pct": 0.20, "trend_sma_days": 200}),
    ]

    W = 120
    print("=" * W)
    print(f"{'LONG PORTFOLIO SIMULATIONS: 5 YEARS COMPOUNDED ON REAL THETA DATA':^{W}}")
    print(f"{'$25K starting, 6 tickers, pullback strategy, real bid/ask':^{W}}")
    print("=" * W)

    results = []
    for name, overrides in configs:
        print(f"  Running: {name}...", end="", flush=True)
        r = run_compounded(name, provider, **overrides)
        results.append(r)
        print(f" ${r['final']:,.0f} ({r['total_ret']:+.1f}%)")

    # Sort by final capital
    results.sort(key=lambda x: x["final"], reverse=True)
    baseline = next(r for r in results if r["name"] == "BASELINE")

    print()
    print("=" * W)
    print(f"{'RESULTS (ranked by final capital)':^{W}}")
    print("=" * W)
    print(f"{'Rank':<5} {'Config':<24} {'2018':>8} {'2019':>8} {'2022':>8} {'2023':>8} {'2024':>8} {'Final':>11} {'5yr':>8} {'vs Base':>10}")
    print("-" * W)

    for i, r in enumerate(results, 1):
        y = {yr["year"]: yr for yr in r["per_year"]}
        delta = r["final"] - baseline["final"]
        marker = " <--" if r["name"] == "BASELINE" else ""
        print(
            f"{i:<5} {r['name']:<24}"
            f" {y['2018']['ret']:>+7.1f}%"
            f" {y['2019']['ret']:>+7.1f}%"
            f" {y['2022']['ret']:>+7.1f}%"
            f" {y['2023']['ret']:>+7.1f}%"
            f" {y['2024']['ret']:>+7.1f}%"
            f" ${r['final']:>10,.0f}"
            f" {r['total_ret']:>+7.1f}%"
            f" ${delta:>+9,.0f}{marker}"
        )

    # Detailed per-year for top 3
    print()
    print("=" * W)
    print(f"{'TOP 3 DETAILED':^{W}}")
    print("=" * W)

    for r in results[:3]:
        print(f"\n  {r['name']}:")
        print(f"  {'Year':<6} {'Start':>11} {'End':>11} {'Return':>8} {'Trades':>7} {'WR':>6} {'MaxDD':>7}")
        for yr in r["per_year"]:
            print(f"  {yr['year']:<6} ${yr['start']:>10,.0f} ${yr['end']:>10,.0f} "
                  f"{yr['ret']:>+7.1f}% {yr['trades']:>7} {yr['wr']:>5.0%} {yr['dd']*100:>6.1f}%")
        print(f"  {'TOTAL':<6} ${START_CAPITAL:>10,.0f} ${r['final']:>10,.0f} {r['total_ret']:>+7.1f}%")

    # Risk metrics
    print()
    print("=" * W)
    print(f"{'RISK METRICS':^{W}}")
    print("=" * W)
    print(f"{'Config':<24} {'Worst Year':>10} {'Best Year':>10} {'Years +':>8} {'Years -':>8} {'Worst DD':>9}")
    print("-" * 75)

    for r in results:
        rets = [yr["ret"] for yr in r["per_year"]]
        dds = [yr["dd"] for yr in r["per_year"]]
        pos = sum(1 for x in rets if x > 0)
        neg = len(rets) - pos
        print(f"{r['name']:<24} {min(rets):>+9.1f}% {max(rets):>+9.1f}% {pos:>8} {neg:>8} {min(dds)*100:>8.1f}%")

    # Final verdict
    best = results[0]
    print()
    print("=" * W)
    print(f"  BEST: {best['name']}")
    print(f"  $25,000 -> ${best['final']:,.0f} ({best['total_ret']:+.1f}%) over 5 years")
    print(f"  vs BASELINE: ${baseline['final']:,.0f} ({baseline['total_ret']:+.1f}%)")
    print(f"  Improvement: ${best['final'] - baseline['final']:+,.0f}")
    print("=" * W)

    return 0


if __name__ == "__main__":
    sys.exit(main())
