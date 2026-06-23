"""Portfolio simulation: pullback strategy on real Theta Data options.

Runs the pullback strategy across 6 core tickers simultaneously with a
shared $25K account from Jan 2 to Dec 31, 2024.

Usage:
    python experiments/portfolio_theta_pullback.py [--reset] [--month 2024-06]

Flags:
    --reset      Delete the existing account JSON and start fresh
    --month YYYY-MM  Run only that month (smoke test)
    --verbose    Print day-by-day trade actions
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

from tradelab.account import SimulatedAccount
from tradelab.portfolio_simulator import PortfolioSimulator, PortfolioConfig
from tradelab.pricing.thetadata import ThetaDataProvider


def main():
    args = sys.argv[1:]
    reset = "--reset" in args
    verbose = "--verbose" in args

    # Date range
    start_date = "2024-01-02"
    end_date = "2024-12-31"
    account_file = "accounts/portfolio_theta_2024.json"

    # --year shortcut for annual runs
    for i, a in enumerate(args):
        if a == "--year" and i + 1 < len(args):
            yr = args[i + 1]
            start_date = f"{yr}-01-02"
            end_date = f"{yr}-12-31"
            account_file = f"accounts/portfolio_theta_{yr}.json"

    # --month shortcut for smoke testing
    for i, a in enumerate(args):
        if a == "--month" and i + 1 < len(args):
            ym = args[i + 1]
            start_date = f"{ym}-01"
            import calendar as _cal
            year, month = [int(x) for x in ym.split("-")]
            last_day = _cal.monthrange(year, month)[1]
            end_date = f"{ym}-{last_day:02d}"
            account_file = f"accounts/portfolio_theta_{ym}.json"

    # Check Theta connection
    provider = ThetaDataProvider(verbose=False)
    if not provider.check_connection():
        print("ERROR: Theta Terminal not reachable. Start it first:")
        print("  cd theta-data && java -jar ThetaTerminalv3.jar")
        return 1

    # Handle account reset
    if reset and os.path.exists(account_file):
        os.remove(account_file)
        print(f"Reset: deleted {account_file}")

    # Create account
    starting_capital = 25000.0
    account = SimulatedAccount.load_or_create(
        account_file,
        starting_capital=starting_capital,
        name=f"Portfolio Theta {start_date[:7]}",
        strategy="portfolio_pullback_theta",
    )

    if account.total_trades_count > 0:
        print(f"Warning: account already has {account.total_trades_count} trades.")
        print(f"  Use --reset to start fresh, or delete {account_file}")
        print(f"  Current equity: ${account.equity:,.2f}")
        return 1

    # Configure
    config = PortfolioConfig(
        tickers=["NVDA", "AVGO", "MSFT", "GOOG", "CAT", "AAPL"],
        start_date=start_date,
        end_date=end_date,
        starting_capital=starting_capital,
        max_positions=6,
        max_pct_per_position=0.15,
        max_contracts=10,
        buffer=0.10,
        spread_pct=0.02,
        pullback_threshold=0.03,
        pullback_lookback=20,
        dte_open=30,
        dte_close=14,
    )

    print("=" * 70)
    print(f"Portfolio simulation: {start_date} to {end_date}")
    print(f"Tickers: {', '.join(config.tickers)}")
    print(f"Starting capital: ${starting_capital:,.0f}")
    print(f"Max positions: {config.max_positions}, max {config.max_pct_per_position:.0%} per position")
    print("=" * 70)
    print()

    # Run
    sim = PortfolioSimulator(account, provider, config, verbose=verbose)
    sim.run()

    # Report
    report = sim.report()
    print()
    print(report.summary())

    print()
    provider.print_stats()

    return 0


if __name__ == "__main__":
    sys.exit(main())
