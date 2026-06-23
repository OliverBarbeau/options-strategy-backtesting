"""T-bill allocation backtests: maintain 50% of capital in T-bills.

Strategy: target half the portfolio in T-bills (safe yield), half in
put credit spread trading. Rebalance monthly by adjusting the
max_heat to limit deployment to ~50% of total capital.

This tests whether the T-bill ballast improves risk-adjusted returns
by reducing max drawdown while still earning idle yield.

Configs:
  1. Full deployment (baseline, ~80% heat limit)
  2. 50% T-bill allocation (40% heat limit + idle yield)
  3. 70% T-bill / 30% trading (aggressive safety)
  4. 50% T-bill + 20% loss limit

Saves results as backtest-type accounts for the web dashboard.
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
TBILL = {"2018": 0.019, "2019": 0.022, "2022": 0.015, "2023": 0.052, "2024": 0.053}


def run_compounded(name: str, provider: ThetaDataProvider, save_final: bool = False, **overrides) -> dict:
    capital = 25000.0
    per_year = []

    for year in YEARS:
        path = f"accounts/_bt_{name}_{year}.json"
        if os.path.exists(path):
            os.remove(path)

        acct = SimulatedAccount.load_or_create(
            path, starting_capital=capital, name=f"{name}_{year}",
            strategy="pullback", account_type="backtest",
        )

        # Determine idle yield: use override if 0.0 explicitly, otherwise T-bill rate
        clean_overrides = {k: v for k, v in overrides.items()}
        if "idle_yield_annual" not in clean_overrides:
            clean_overrides["idle_yield_annual"] = TBILL[year]

        config = PortfolioConfig(
            tickers=TICKERS,
            start_date=f"{year}-01-02",
            end_date=f"{year}-12-31",
            starting_capital=capital,
            **clean_overrides,
        )

        sim = PortfolioSimulator(acct, provider, config, verbose=False)
        sim.run()

        ending = acct.equity
        per_year.append({
            "year": year, "start": capital, "end": ending,
            "ret": (ending / capital - 1) * 100,
            "trades": len(acct.trades), "wr": acct.win_rate,
        })

        # Save the final year's account if requested
        if save_final and year == YEARS[-1]:
            final_path = f"accounts/bt_{name}.json"
            if os.path.exists(final_path):
                os.remove(final_path)
            # Create a summary account with the full history
            summary = SimulatedAccount.load_or_create(
                final_path, starting_capital=25000, name=f"bt_{name}",
                strategy="pullback", account_type="backtest",
            )
            summary.balance = ending
            summary.last_advanced_date = f"{year}-12-31"
            # Copy trades from all years
            all_trades_list = []
            for y in YEARS:
                ypath = f"accounts/_bt_{name}_{y}.json"
                if os.path.exists(ypath):
                    ya = SimulatedAccount.load(ypath)
                    all_trades_list.extend(ya.trades)
                    for snap in ya.equity_curve:
                        summary.equity_curve.append(snap)
            summary.trades = all_trades_list
            summary.save()

        capital = ending
        try:
            os.remove(path)
        except:
            pass

    return {
        "name": name, "final": capital,
        "total_ret": (capital / 25000 - 1) * 100,
        "per_year": per_year,
    }


def main():
    provider = ThetaDataProvider()
    if not provider.check_connection():
        print("Theta Terminal not reachable.")
        return 1

    configs = [
        # (name, description, overrides)
        ("full_deploy",
         "Full deployment, no T-bill allocation",
         {"max_heat": 0.80, "idle_yield_annual": 0.0}),

        ("full_deploy_tbills",
         "Full deployment + T-bills on idle",
         {"max_heat": 0.80}),

        ("half_tbills",
         "50% T-bill allocation (40% heat limit)",
         {"max_heat": 0.40}),

        ("half_tbills_loss20",
         "50% T-bills + 20% loss limit",
         {"max_heat": 0.40, "max_portfolio_loss_pct": 0.20}),

        ("seventy_tbills",
         "70% T-bills / 30% trading (25% heat)",
         {"max_heat": 0.25}),

        ("best_config",
         "7% buf + 20% loss + T-bills (research winner)",
         {"buffer": 0.07, "max_portfolio_loss_pct": 0.20, "max_heat": 0.80}),

        ("best_half_tbills",
         "7% buf + 20% loss + 50% T-bill allocation",
         {"buffer": 0.07, "max_portfolio_loss_pct": 0.20, "max_heat": 0.40}),
    ]

    W = 115
    print("=" * W)
    print(f"{'T-BILL ALLOCATION BACKTESTS: 5 YEARS COMPOUNDED ON REAL THETA DATA':^{W}}")
    print("=" * W)

    results = []
    for name, desc, overrides in configs:
        print(f"  Running: {name}...", end="", flush=True)
        r = run_compounded(name, provider, save_final=True, **overrides)
        results.append({**r, "desc": desc})
        print(f" ${r['final']:,.0f} ({r['total_ret']:+.1f}%)")

    results.sort(key=lambda x: x["final"], reverse=True)

    print()
    print(f"{'Config':<24} {'Desc':<42} {'Final':>11} {'5yr':>8}")
    print("-" * W)
    for r in results:
        print(f"{r['name']:<24} {r['desc']:<42} ${r['final']:>10,.0f} {r['total_ret']:>+7.1f}%")

    print()
    print(f"{'Config':<24} {'2018':>8} {'2019':>8} {'2022':>8} {'2023':>8} {'2024':>8}")
    print("-" * 70)
    for r in results:
        y = {yr["year"]: yr for yr in r["per_year"]}
        print(f"{r['name']:<24}" + "".join(f" {y[yr]['ret']:>+7.1f}%" for yr in YEARS))

    # Show backtest accounts created
    print()
    import glob
    bt_files = sorted(glob.glob("accounts/bt_*.json"))
    print(f"Created {len(bt_files)} backtest accounts for web dashboard:")
    for f in bt_files:
        a = SimulatedAccount.load(f)
        print(f"  {a.name:<24} ${a.equity:>10,.0f}  {len(a.trades)} trades  {a.account_type}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
