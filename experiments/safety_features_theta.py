"""Revisit safety features: test on real Theta Data across 2022-2024.

Previous test (adaptive_lab.py) concluded all features hurt P/L.
That was on B-S data dominated by bull years. Now we have real options data
across a bear (2022), recovery (2023), and bull (2024) market. The question:
do any features improve the 3-year compounded result?

Configs tested:
  0. BASELINE (no features)
  1. max_portfolio_loss 20%
  2. max_portfolio_loss 30%
  3. max_heat 50%
  4. drawdown_scaling
  5. trend_sma 50-day
  6. trend_sma 200-day
  7. loss_limit 20% + drawdown_scaling
  8. loss_limit 25% + trend_sma 200
  9. ALL: loss 25% + heat 50% + drawdown + sma200
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

from tradelab.account import SimulatedAccount
from tradelab.portfolio_simulator import PortfolioSimulator, PortfolioConfig
from tradelab.pricing.thetadata import ThetaDataProvider


def run_config(name: str, years: list[str], provider: ThetaDataProvider, **overrides) -> dict:
    """Run a config across multiple years, return per-year and compounded results."""
    results = []
    compounded_capital = 25000.0

    for year in years:
        account_file = f"accounts/_feature_test_{name}_{year}.json"
        if os.path.exists(account_file):
            os.remove(account_file)

        account = SimulatedAccount.load_or_create(
            account_file,
            starting_capital=compounded_capital,
            name=f"{name}_{year}",
            strategy="pullback_theta",
        )

        config = PortfolioConfig(
            tickers=["NVDA", "AVGO", "MSFT", "GOOG", "CAT", "AAPL"],
            start_date=f"{year}-01-02",
            end_date=f"{year}-12-31",
            starting_capital=compounded_capital,
            **overrides,
        )

        sim = PortfolioSimulator(account, provider, config, verbose=False)
        sim.run()

        ending = account.equity
        pnl = ending - compounded_capital
        ret = pnl / compounded_capital * 100

        # Max drawdown from equity curve
        equities = [s.equity for s in account.equity_curve]
        max_dd = 0.0
        if equities:
            peak = equities[0]
            for e in equities:
                peak = max(peak, e)
                dd = (e - peak) / peak
                max_dd = min(max_dd, dd)

        results.append({
            "year": year,
            "starting": compounded_capital,
            "ending": ending,
            "pnl": pnl,
            "return_pct": ret,
            "trades": len(account.trades),
            "wr": account.win_rate,
            "max_dd": max_dd,
        })

        compounded_capital = ending

        # Clean up temp account file
        try:
            os.remove(account_file)
        except:
            pass

    total_return = (compounded_capital / 25000 - 1) * 100
    return {
        "name": name,
        "per_year": results,
        "final_capital": compounded_capital,
        "total_return": total_return,
    }


def main():
    provider = ThetaDataProvider()
    if not provider.check_connection():
        print("Theta Terminal not reachable.")
        return 1

    years = ["2022", "2023", "2024"]

    configs = [
        ("BASELINE",              {}),
        ("loss_limit_20pct",      {"max_portfolio_loss_pct": 0.20}),
        ("loss_limit_25pct",      {"max_portfolio_loss_pct": 0.25}),
        ("loss_limit_30pct",      {"max_portfolio_loss_pct": 0.30}),
        ("heat_50pct",            {"max_heat": 0.50}),
        ("dd_scaling",            {"drawdown_scaling": True}),
        ("sma_50",                {"trend_sma_days": 50}),
        ("sma_200",               {"trend_sma_days": 200}),
        ("loss20_dd",             {"max_portfolio_loss_pct": 0.20, "drawdown_scaling": True}),
        ("loss25_sma200",         {"max_portfolio_loss_pct": 0.25, "trend_sma_days": 200}),
        ("ALL_safe",              {"max_portfolio_loss_pct": 0.25, "max_heat": 0.50,
                                   "drawdown_scaling": True, "trend_sma_days": 200}),
    ]

    W = 115
    print("=" * W)
    print(f"{'SAFETY FEATURES REVISITED: REAL THETA DATA, 2022-2024 COMPOUNDED':^{W}}")
    print("=" * W)

    all_results = []
    for name, overrides in configs:
        print(f"  Running: {name}...", end="", flush=True)
        r = run_config(name, years, provider, **overrides)
        all_results.append(r)
        print(f" ${r['final_capital']:,.0f} ({r['total_return']:+.1f}%)")

    # Summary table
    print()
    print("=" * W)
    print(f"{'RESULTS (sorted by final capital after 3 compounded years)':^{W}}")
    print("=" * W)
    print(f"{'Config':<22} {'2022':>9} {'2023':>9} {'2024':>9} {'Final $':>12} {'Total':>9} {'2022 DD':>8}")
    print("-" * W)

    all_results.sort(key=lambda r: r["final_capital"], reverse=True)
    baseline = next(r for r in all_results if r["name"] == "BASELINE")

    for r in all_results:
        y = {yr["year"]: yr for yr in r["per_year"]}
        marker = "  <--" if r["name"] == "BASELINE" else ""
        delta = r["final_capital"] - baseline["final_capital"]
        delta_str = f" (+${delta:,.0f})" if delta > 0 else ""

        print(
            f"{r['name']:<22} "
            f"{y['2022']['return_pct']:>+8.1f}% "
            f"{y['2023']['return_pct']:>+8.1f}% "
            f"{y['2024']['return_pct']:>+8.1f}% "
            f"${r['final_capital']:>11,.0f} "
            f"{r['total_return']:>+8.1f}% "
            f"{y['2022']['max_dd']:>7.1%}"
            f"{delta_str}{marker}"
        )

    # Detail: per-year trades and WR for top 5
    print()
    print("=" * W)
    print(f"{'DETAIL: TOP 5 vs BASELINE':^{W}}")
    print("=" * W)

    top5 = all_results[:5]
    if baseline not in top5:
        top5.append(baseline)

    for r in top5:
        print(f"\n  {r['name']}:")
        for yr in r["per_year"]:
            print(f"    {yr['year']}: ${yr['starting']:>10,.0f} -> ${yr['ending']:>10,.0f}  "
                  f"{yr['return_pct']:>+6.1f}%  {yr['trades']:>3} trades  "
                  f"{yr['wr']:.0%} WR  DD={yr['max_dd']:.1%}")

    # Improvement analysis
    print()
    print("=" * W)
    print(f"{'IMPROVEMENT vs BASELINE':^{W}}")
    print("=" * W)
    print(f"{'Config':<22} {'Baseline $':>12} {'Config $':>12} {'Delta $':>12} {'2022 saved':>12}")
    print("-" * 65)

    for r in all_results:
        if r["name"] == "BASELINE":
            continue
        delta = r["final_capital"] - baseline["final_capital"]
        b22 = next(yr for yr in baseline["per_year"] if yr["year"] == "2022")
        c22 = next(yr for yr in r["per_year"] if yr["year"] == "2022")
        saved_22 = c22["ending"] - b22["ending"]
        print(f"{r['name']:<22} ${baseline['final_capital']:>11,.0f} ${r['final_capital']:>11,.0f} "
              f"${delta:>+11,.0f} ${saved_22:>+11,.0f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
