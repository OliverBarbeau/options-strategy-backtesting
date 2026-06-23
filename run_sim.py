"""CLI for managing simulation accounts and running backtests.

Usage:
    python run_sim.py create <name> <strategy> [--capital 25000]
    python run_sim.py advance [<name>] [--to 2026-04-04]
    python run_sim.py status [<name>]
    python run_sim.py mtm [<name>]
    python run_sim.py list
    python run_sim.py compare

    python run_sim.py scan                       # Scan market for candidates
    python run_sim.py scan --pullback            # Only show tickers in pullback
    python run_sim.py scan --backtest            # Scan + backtest top candidates
    python run_sim.py calibrate status           # Show calibration data summary
    python run_sim.py calibrate theta <ticker>   # Calibrate B-S against Theta Data
    python run_sim.py calibrate check            # Check if Theta Terminal is reachable

    python run_sim.py validate <ticker>          # Run pullback strategy with B-S AND Theta Data
    python run_sim.py validate <ticker> --theta  # Theta Data only

Strategies: pullback, regime_adaptive, conservative, aggressive_pullback
"""

import sys
import os
import glob

from tradelab.account import SimulatedAccount
from tradelab.simulator import Simulator

ACCOUNTS_DIR = "accounts"


def get_account_files() -> list[str]:
    os.makedirs(ACCOUNTS_DIR, exist_ok=True)
    return sorted(glob.glob(os.path.join(ACCOUNTS_DIR, "*.json")))


def get_account(name: str) -> SimulatedAccount:
    filepath = os.path.join(ACCOUNTS_DIR, f"{name}.json")
    if not os.path.exists(filepath):
        print(f"Account '{name}' not found at {filepath}")
        sys.exit(1)
    return SimulatedAccount.load(filepath)


def cmd_create(args):
    if len(args) < 2:
        print("Usage: create <name> <strategy> [--capital N] [--type live|backtest]")
        return

    name = args[0]
    strategy = args[1]
    capital = 25000
    account_type = "live"

    for i, a in enumerate(args):
        if a == "--capital" and i + 1 < len(args):
            capital = float(args[i + 1])
        if a == "--type" and i + 1 < len(args):
            account_type = args[i + 1]

    if account_type not in SimulatedAccount.ACCOUNT_TYPES:
        print(f"Unknown type '{account_type}'. Options: {SimulatedAccount.ACCOUNT_TYPES}")
        return

    if strategy not in Simulator.STRATEGIES:
        print(f"Unknown strategy '{strategy}'. Options: {Simulator.STRATEGIES}")
        return

    filepath = os.path.join(ACCOUNTS_DIR, f"{name}.json")
    if os.path.exists(filepath):
        print(f"Account '{name}' already exists.")
        return

    account = SimulatedAccount.load_or_create(
        filepath, starting_capital=capital, name=name, strategy=strategy,
        account_type=account_type,
    )
    print(f"Created {account_type} account '{name}' with ${capital:,.0f} using {strategy} strategy")
    print(f"Saved to {filepath}")


def cmd_advance(args):
    end_date = None
    for i, a in enumerate(args):
        if a == "--to" and i + 1 < len(args):
            end_date = args[i + 1]
            args = args[:i] + args[i + 2:]
            break

    if args:
        accounts = [get_account(args[0])]
    else:
        files = get_account_files()
        if not files:
            print("No accounts found. Create one first.")
            return
        accounts = [SimulatedAccount.load(f) for f in files]

    for account in accounts:
        strategy = account.strategy or "pullback"
        print(f"\n--- Advancing: {account.name} ({strategy}) ---")
        sim = Simulator(account, strategy=strategy)
        sim.catch_up(end_date)


def cmd_status(args):
    if args:
        account = get_account(args[0])
        print(account.status())
    else:
        files = get_account_files()
        if not files:
            print("No accounts found.")
            return
        for f in files:
            account = SimulatedAccount.load(f)
            print(account.status())
            print()


def cmd_mtm(args):
    if not args:
        print("Usage: mtm <account_name>")
        return

    account = get_account(args[0])
    sim = Simulator(account, strategy=account.strategy or "pullback")
    sim._load_data()

    positions = sim.mark_to_market()
    if not positions:
        print(f"{account.name}: No open positions")
        return

    print(f"Mark-to-Market: {account.name}")
    print(f"{'Ticker':<6} {'Entry':>8} {'Now':>8} {'Chg':>7} {'Strike':>8} {'Buffer':>7} "
          f"{'Credit':>8} {'Close$':>8} {'Unreal':>9} {'DTE':>4}")
    print("-" * 90)

    for p in positions:
        print(f"{p['ticker']:<6} ${p['entry_price']:>7.2f} ${p['current_price']:>7.2f} "
              f"{p['price_change']:>+6.1f}% ${p['short_strike']:>7.2f} {p['buffer_remaining']:>6.1f}% "
              f"${p['credit']:>7.0f} ${p['close_cost']:>7.0f} ${p['unrealized_pnl']:>+8.0f} "
              f"{p['dte_remaining']:>4}")

    total_unreal = sum(p["unrealized_pnl"] for p in positions)
    print(f"\nTotal unrealized: ${total_unreal:+,.2f}")


def cmd_list(args):
    files = get_account_files()
    if not files:
        print("No accounts found. Create one with: python run_sim.py create <name> <strategy>")
        return

    print(f"{'Name':<20} {'Strategy':<18} {'Capital':>10} {'Equity':>12} {'P/L':>10} {'Trades':>6} {'WR':>6} {'Last Updated':<12}")
    print("-" * 100)
    for f in files:
        acct = SimulatedAccount.load(f)
        print(f"{acct.name:<20} {acct.strategy:<18} ${acct.starting_capital:>9,.0f} "
              f"${acct.equity:>11,.2f} ${acct.total_pnl:>+9,.2f} "
              f"{acct.total_trades_count:>6} {acct.win_rate:>5.1%} {acct.last_advanced_date[:10]:<12}")


def cmd_compare(args):
    files = get_account_files()
    if len(files) < 2:
        print("Need at least 2 accounts to compare.")
        return

    accounts = [SimulatedAccount.load(f) for f in files]

    print(f"{'Account':<20} {'Strategy':<15} {'Start $':>10} {'Equity':>12} {'Return':>9} {'Trades':>6} {'WR':>6} {'Max DD':>8}")
    print("-" * 90)

    for acct in accounts:
        ret = acct.total_pnl / acct.starting_capital * 100
        # Compute max DD from equity curve
        if acct.equity_curve:
            equities = [e.equity for e in acct.equity_curve]
            peak = equities[0]
            max_dd = 0
            for eq in equities:
                peak = max(peak, eq)
                dd = (eq - peak) / peak
                max_dd = min(max_dd, dd)
        else:
            max_dd = 0

        print(f"{acct.name:<20} {acct.strategy:<15} ${acct.starting_capital:>9,.0f} "
              f"${acct.equity:>11,.2f} {ret:>+8.1f}% "
              f"{acct.total_trades_count:>6} {acct.win_rate:>5.1%} {max_dd:>7.1%}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1]
    args = sys.argv[2:]

    commands = {
        "create": cmd_create,
        "advance": cmd_advance,
        "status": cmd_status,
        "mtm": cmd_mtm,
        "list": cmd_list,
        "compare": cmd_compare,
        "scan": cmd_scan,
        "calibrate": cmd_calibrate,
        "validate": cmd_validate,
        "theta": cmd_theta,
    }

    if cmd in commands:
        commands[cmd](args)
    else:
        print(f"Unknown command: {cmd}")
        print(f"Available: {', '.join(commands)}")


def cmd_scan(args):
    import warnings
    warnings.filterwarnings("ignore")
    from tradelab.scanner import StrategyScanner

    pullback_only = "--pullback" in args
    do_backtest = "--backtest" in args

    print("Scanning market..." + (" (pullback only)" if pullback_only else ""))
    scanner = StrategyScanner()

    if do_backtest:
        results = scanner.scan_and_backtest(n=15, pullback_only=pullback_only)
        print(f"\n{'Ticker':<6} {'Score':>5} {'Sector':<10} {'HV30':>6} {'Pull':>6} {'CrPot':>6} {'Breach':>6}  {'Trades':>6} {'WR':>6} {'P/L':>10} {'$/Trd':>8}")
        print("-" * 95)
        for r in results:
            earn = " [EARN]" if r.get("has_earnings_soon") else ""
            print(f"{r['ticker']:<6} {r['score']:>5.0f} {r['sector']:<10} {r['hv30']:>5.0%} {r['pullback_pct']:>+5.1%} {r['credit_potential']:>5.1%} {r['breach_rate']:>5.1%}"
                  f"  {r['trades']:>6} {r['win_rate']:>5.1%} ${r['total_pnl']:>+9,.0f} ${r['avg_pnl']:>+7.2f}{earn}")
    else:
        results = scanner.scan(pullback_only=pullback_only)
        qualified = [r for r in results if r.qualifies]
        print(f"\nFound {len(qualified)} qualified candidates out of {len(results)} scanned\n")
        print(f"{'Ticker':<6} {'Score':>5} {'Sector':<10} {'Price':>8} {'HV30':>6} {'Pull':>7} {'CrPot':>6} {'Breach':>6}  Flags")
        print("-" * 90)
        for r in results[:25]:
            flag_str = ", ".join(r.flags) if r.flags else ""
            marker = "*" if r.qualifies else " "
            print(f"{marker}{r.ticker:<5} {r.score:>5.0f} {r.sector:<10} ${r.price:>7.0f} {r.hv30:>5.0%} {r.pullback_pct:>+6.1%} {r.credit_potential:>5.1%} {r.breach_rate:>5.1%}  {flag_str}")
        print(f"\n* = qualifies (score >= 60, no earnings)")


def cmd_theta(args):
    """Theta Data cache operations."""
    import warnings
    warnings.filterwarnings("ignore")

    if not args:
        print("Usage: theta <stats|backfill|clear>")
        print("  stats              Show cache statistics")
        print("  backfill <tickers> Proactively fetch chains for a universe")
        print("                     Example: theta backfill NVDA,AVGO,MSFT --start 2024-01-01 --end 2024-12-31")
        return

    subcmd = args[0]
    sub_args = args[1:]

    from tradelab.pricing.thetadata import ThetaDataProvider
    theta = ThetaDataProvider()

    if subcmd == "stats":
        from pathlib import Path
        cache_root = Path(theta.cache.cache_dir)
        if not cache_root.exists():
            print("Cache is empty")
            return

        print(f"Theta Data Cache: {cache_root}")
        print("=" * 60)
        total_files = 0
        total_size = 0
        for subdir in sorted(cache_root.iterdir()):
            if subdir.is_dir():
                files = list(subdir.glob("*.parquet"))
                size = sum(f.stat().st_size for f in files)
                total_files += len(files)
                total_size += size
                print(f"  {subdir.name:<30} {len(files):>6} files  {size / 1024 / 1024:>7.1f} MB")
        print("-" * 60)
        print(f"  {'TOTAL':<30} {total_files:>6} files  {total_size / 1024 / 1024:>7.1f} MB")

    elif subcmd == "backfill":
        if not sub_args:
            print("Usage: theta backfill <tickers> [--start YYYY-MM-DD] [--end YYYY-MM-DD] [--every N]")
            print("  tickers: comma-separated, e.g. NVDA,AVGO,MSFT")
            return

        tickers = [t.strip().upper() for t in sub_args[0].split(",")]
        start = "2024-01-01"
        end = "2024-12-31"
        every = 5

        for i, a in enumerate(sub_args):
            if a == "--start" and i + 1 < len(sub_args):
                start = sub_args[i + 1]
            if a == "--end" and i + 1 < len(sub_args):
                end = sub_args[i + 1]
            if a == "--every" and i + 1 < len(sub_args):
                every = int(sub_args[i + 1])

        from tradelab.pricing.backfill import backfill_tickers
        summary = backfill_tickers(
            theta, tickers=tickers, start=start, end=end,
            sample_every_n_days=every, verbose=True,
        )
        print(f"\nBackfill complete: {summary['successful_fetches']} successful, "
              f"{summary['failed_fetches']} failed")

    else:
        print(f"Unknown theta subcommand: {subcmd}")


def cmd_validate(args):
    """Run a pullback backtest with both B-S and Theta Data for comparison."""
    import warnings
    warnings.filterwarnings("ignore")

    if not args:
        print("Usage: validate <ticker> [--start YYYY-MM-DD] [--end YYYY-MM-DD] [--theta-only]")
        return

    ticker = args[0].upper()
    start = "2024-01-01"
    end = "2024-12-31"
    theta_only = "--theta-only" in args

    for i, a in enumerate(args):
        if a == "--start" and i + 1 < len(args):
            start = args[i + 1]
        if a == "--end" and i + 1 < len(args):
            end = args[i + 1]

    from tradelab.pipeline import DataPipeline
    from tradelab.strategies.pullback_entry import PullbackEntryStrategy

    pipe = DataPipeline()
    try:
        df = pipe.fetch_stock(ticker, start=start, end=end)
    except Exception as e:
        print(f"Failed to fetch {ticker}: {e}")
        return

    if len(df) < 100:
        print(f"Not enough data for {ticker}")
        return

    print(f"Pullback Strategy Validation: {ticker}  ({start} to {end})")
    print("=" * 70)

    strat = PullbackEntryStrategy()

    if not theta_only:
        result_bs = strat.run(df, max_contracts=10)
        avg = result_bs.total_pnl / result_bs.total_trades if result_bs.total_trades else 0
        print(f"\n  B-S (inline): {result_bs.total_trades} trades, "
              f"{result_bs.win_rate:.1%} WR, ${result_bs.total_pnl:+,.0f} total, ${avg:+.2f}/trade")

    print(f"\n  Theta Data (loading chains... this takes a minute)")
    try:
        from tradelab.pricing.thetadata import ThetaDataProvider
        theta = ThetaDataProvider()
        if not theta.check_connection():
            print("  Theta Terminal not reachable. Run 'calibrate check' for setup.")
            return

        result_theta = strat.run(df, max_contracts=10, ticker=ticker, provider=theta)
        avg = result_theta.total_pnl / result_theta.total_trades if result_theta.total_trades else 0
        print(f"  Theta Data:   {result_theta.total_trades} trades, "
              f"{result_theta.win_rate:.1%} WR, ${result_theta.total_pnl:+,.0f} total, ${avg:+.2f}/trade")

        if hasattr(result_theta, "skipped_no_entry"):
            print(f"    (skipped {result_theta.skipped_no_entry} entries, "
                  f"{result_theta.skipped_no_exit} exits due to missing data)")

        if not theta_only:
            diff = result_theta.total_pnl - result_bs.total_pnl
            diff_pct = (diff / result_bs.total_pnl * 100) if result_bs.total_pnl != 0 else 0
            print(f"\n  Delta: ${diff:+,.0f} ({diff_pct:+.1f}%)")

            if abs(diff_pct) < 15:
                verdict = "B-S is trustworthy for this ticker"
            elif abs(diff_pct) < 40:
                verdict = "B-S has a moderate bias, use Theta for precision"
            else:
                verdict = "B-S is unreliable for this ticker, use Theta"
            print(f"  Verdict: {verdict}")

        print()
        theta.print_stats()

    except Exception as e:
        print(f"  Theta Data failed: {e}")
        import traceback
        traceback.print_exc()


def cmd_calibrate(args):
    """Calibration commands for comparing pricing providers."""
    import warnings
    warnings.filterwarnings("ignore")

    if not args:
        print("Usage: calibrate <status|check|theta>")
        print("  status             Show calibration log summary")
        print("  check              Verify Theta Terminal is reachable")
        print("  theta <ticker>     Run calibration: B-S vs Theta Data for a ticker")
        return

    subcmd = args[0]
    sub_args = args[1:]

    if subcmd == "status":
        from tradelab.pricing import BlackScholesProvider, MockProvider
        from tradelab.pricing.calibration import Calibrator
        cal = Calibrator(baseline=BlackScholesProvider(), reference=MockProvider())
        report = cal.load_log()
        print(report.summary())

    elif subcmd == "check":
        from tradelab.pricing import ThetaConfig
        from tradelab.pricing.thetadata import ThetaDataProvider
        config = ThetaConfig()
        provider = ThetaDataProvider(config=config)
        print(f"Checking Theta Terminal at {provider.base_url}...")
        if provider.check_connection():
            print("  Connection OK")
            print("  Terminal is running and responsive.")
        else:
            print("  Connection FAILED")
            print("  Is the Java Terminal running?")
            print("  Download: https://thetadata.net/downloads")
            print("  Launch:   java -jar ThetaTerminalv3.jar")
            print("  Verify:   curl http://127.0.0.1:25510/v2/list/roots")

    elif subcmd == "theta":
        if not sub_args:
            print("Usage: calibrate theta <ticker> [--days 30]")
            return

        ticker = sub_args[0]
        days = 30
        for i, a in enumerate(sub_args):
            if a == "--days" and i + 1 < len(sub_args):
                days = int(sub_args[i + 1])

        from datetime import datetime, timedelta
        from tradelab.pricing import BlackScholesProvider
        from tradelab.pricing.thetadata import ThetaDataProvider
        from tradelab.pricing.calibration import Calibrator

        theta = ThetaDataProvider()
        if not theta.check_connection():
            print("Theta Terminal not reachable. Run 'calibrate check' for details.")
            return

        bs = BlackScholesProvider()
        cal = Calibrator(baseline=bs, reference=theta)

        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        print(f"Calibrating {ticker}: B-S vs Theta Data ({start} to {end})")
        print("This may take a few minutes due to rate limiting...")

        report = cal.calibrate_ticker(
            ticker=ticker, start=start, end=end, sample_every_n_days=5,
        )
        print()
        print(report.summary())

    else:
        print(f"Unknown calibrate subcommand: {subcmd}")



if __name__ == "__main__":
    main()
