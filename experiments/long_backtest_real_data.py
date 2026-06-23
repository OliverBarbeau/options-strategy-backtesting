"""Long-duration backtests on real Theta Data historical options.

Runs our primary strategy configurations across 8 continuous years
(2018-2025) using cached real option chain data. Previous experiments
skipped 2020-2021; this includes them for complete market-cycle coverage
(bull, COVID crash, recovery, 2022 bear, 2023-2024 bull, 2025 volatility).

Configs tested:
  1. Baseline        — 10% buffer, no safety features
  2. Best config     — 7% buffer + 20% loss limit (research winner)
  3. Conservative    — 13% buffer + 20% loss limit
  4. Aggressive      — 7% buffer, no safety (risk-on)

All use T-bill idle yield at historical rates and save final accounts
for the web dashboard.
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

from tradelab.account import SimulatedAccount
from tradelab.portfolio_simulator import PortfolioSimulator, PortfolioConfig
from tradelab.pricing.thetadata import ThetaDataProvider

TICKERS = ["NVDA", "AVGO", "MSFT", "GOOG", "CAT", "AAPL"]
YEARS = ["2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025"]

# Historical 1-year T-bill rates (approximate annual averages)
TBILL = {
    "2018": 0.019,
    "2019": 0.022,
    "2020": 0.004,  # COVID near-zero rates
    "2021": 0.001,  # Still near-zero
    "2022": 0.015,  # Rising rates mid-year
    "2023": 0.052,
    "2024": 0.053,
    "2025": 0.045,  # Modestly lower
}

CONFIGS = [
    ("baseline",
     "10% buffer, no safety features",
     {"max_heat": 0.80, "buffer": 0.10}),

    ("best_config_8yr",
     "7% buffer + 20% loss limit (research winner)",
     {"max_heat": 0.80, "buffer": 0.07, "max_portfolio_loss_pct": 0.20}),

    ("conservative_8yr",
     "13% buffer + 20% loss limit",
     {"max_heat": 0.80, "buffer": 0.13, "max_portfolio_loss_pct": 0.20}),

    ("aggressive_8yr",
     "7% buffer, no loss limit (risk-on)",
     {"max_heat": 0.80, "buffer": 0.07}),
]


def run_compounded(name: str, provider: ThetaDataProvider, save_final: bool = False, **overrides) -> dict:
    """Run year-by-year compounding simulation, return results dict."""
    capital = 25000.0
    per_year = []
    all_trades = []
    all_equity_curve = []

    for year in YEARS:
        path = f"accounts/_bt_{name}_{year}.json"
        if os.path.exists(path):
            os.remove(path)

        acct = SimulatedAccount.load_or_create(
            path, starting_capital=capital, name=f"{name}_{year}",
            strategy="pullback", account_type="backtest",
        )

        # Use T-bill rate for idle yield unless explicitly set to 0
        clean_overrides = dict(overrides)
        if "idle_yield_annual" not in clean_overrides:
            clean_overrides["idle_yield_annual"] = TBILL[year]

        # 2025: end at March 31 (latest reliable data)
        end_date = f"{year}-12-31" if year != "2025" else "2025-03-31"

        config = PortfolioConfig(
            tickers=TICKERS,
            start_date=f"{year}-01-02",
            end_date=end_date,
            starting_capital=capital,
            **clean_overrides,
        )

        sim = PortfolioSimulator(acct, provider, config, verbose=False)
        print(f" {year}", end="", flush=True)
        sim.run()

        ending = acct.equity
        trades_count = len(acct.trades)
        wr = acct.win_rate

        per_year.append({
            "year": year, "start": capital, "end": ending,
            "ret": (ending / capital - 1) * 100,
            "trades": trades_count, "wr": wr,
        })

        # Accumulate for final account
        all_trades.extend(acct.trades)
        all_equity_curve.extend(acct.equity_curve)

        capital = ending

        # Clean up temp file
        try:
            os.remove(path)
        except Exception:
            pass

    # Save final aggregated account for web dashboard
    if save_final:
        final_path = f"accounts/bt_{name}.json"
        if os.path.exists(final_path):
            os.remove(final_path)

        summary = SimulatedAccount.load_or_create(
            final_path, starting_capital=25000, name=f"bt_{name}",
            strategy="pullback", account_type="backtest",
        )
        summary.balance = capital
        summary.last_advanced_date = f"{YEARS[-1]}-12-31"
        summary.trades = all_trades
        summary.equity_curve = all_equity_curve
        summary.save()

    return {
        "name": name, "final": capital,
        "total_ret": (capital / 25000 - 1) * 100,
        "per_year": per_year,
    }


def main():
    provider = ThetaDataProvider()

    W = 120
    print("=" * W)
    print(f"{'LONG-DURATION BACKTESTS: 8 YEARS (2018-2025) ON REAL THETA DATA':^{W}}")
    print(f"{'Tickers: ' + ', '.join(TICKERS):^{W}}")
    print(f"{'Starting capital: $25,000 | Compounded year-over-year':^{W}}")
    print("=" * W)
    print()

    results = []
    for name, desc, overrides in CONFIGS:
        print(f"  Running: {name} ({desc})...", end="", flush=True)
        r = run_compounded(name, provider, save_final=True, **overrides)
        results.append({**r, "desc": desc})
        print(f" ${r['final']:,.0f} ({r['total_ret']:+.1f}%)")

    results.sort(key=lambda x: x["final"], reverse=True)

    # Summary table
    print()
    print(f"{'Config':<22} {'Description':<44} {'Final':>11} {'8yr':>8}")
    print("-" * W)
    for r in results:
        print(f"{r['name']:<22} {r['desc']:<44} ${r['final']:>10,.0f} {r['total_ret']:>+7.1f}%")

    # Year-by-year breakdown
    print()
    header = f"{'Config':<22}" + "".join(f" {y:>8}" for y in YEARS)
    print(header)
    print("-" * len(header))
    for r in results:
        y = {yr["year"]: yr for yr in r["per_year"]}
        line = f"{r['name']:<22}" + "".join(f" {y[yr]['ret']:>+7.1f}%" for yr in YEARS)
        print(line)

    # Trade counts per year
    print()
    header2 = f"{'Config':<22}" + "".join(f" {y:>8}" for y in YEARS) + "   Total"
    print(header2)
    print("-" * len(header2))
    for r in results:
        y = {yr["year"]: yr for yr in r["per_year"]}
        total = sum(y[yr]["trades"] for yr in YEARS)
        line = f"{r['name']:<22}" + "".join(f" {y[yr]['trades']:>8}" for yr in YEARS) + f"   {total:>5}"
        print(line)

    # Show saved accounts
    print()
    import glob
    bt_files = sorted(glob.glob("accounts/bt_*8yr*.json") + glob.glob("accounts/bt_baseline.json"))
    print(f"Saved {len(bt_files)} backtest accounts for web dashboard:")
    for f in bt_files:
        a = SimulatedAccount.load(f)
        print(f"  {a.name:<28} ${a.equity:>10,.0f}  {len(a.trades)} trades  {a.account_type}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
