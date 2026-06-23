"""Validate pullback strategy: B-S (inline) vs Theta Data (real).

Runs the same strategy with the same parameters on the same ticker/date
range with both pricing engines, and compares results side by side.

Expected findings from calibration:
- High-vol tickers (NVDA, AVGO): results should be close
- Low-vol tickers (AAPL, MSFT): B-S should be wildly off
"""

import warnings
warnings.filterwarnings("ignore")

from tradelab.pipeline import DataPipeline
from tradelab.pricing import BlackScholesProvider
from tradelab.pricing.thetadata import ThetaDataProvider
from tradelab.strategies.pullback_entry import PullbackEntryStrategy

pipe = DataPipeline()
theta = ThetaDataProvider()
bs_provider = BlackScholesProvider()

TICKERS = ["NVDA", "AVGO", "AAPL", "MSFT"]
START = "2024-01-01"
END = "2024-12-31"

print("=" * 95)
print(f"{'PULLBACK STRATEGY VALIDATION: B-S vs REAL HISTORICAL OPTIONS DATA':^95}")
print(f"{'Theta Data Standard (v3 Terminal)   |   2024 full year':^95}")
print("=" * 95)
print()
print(f"{'Ticker':<8} {'Path':<18} {'Trades':>7} {'WR':>6} {'Total P/L':>12} {'$/Trade':>10} {'Max DD':>8}")
print("-" * 95)

summary = []

for ticker in TICKERS:
    try:
        df = pipe.fetch_stock(ticker, start=START, end=END)
    except Exception as e:
        print(f"{ticker}: failed to load data: {e}")
        continue

    if len(df) < 100:
        continue

    # Run 1: Inline B-S (legacy fast path)
    strat = PullbackEntryStrategy()
    result_bs = strat.run(df, max_contracts=10)
    bs_avg = result_bs.total_pnl / result_bs.total_trades if result_bs.total_trades else 0
    print(f"{ticker:<8} {'B-S (inline)':<18} {result_bs.total_trades:>7} {result_bs.win_rate:>5.1%} "
          f"${result_bs.total_pnl:>+11,.0f} ${bs_avg:>+9.2f} {result_bs.max_drawdown_pct:>7.1%}")

    # Run 2: Theta Data (real)
    try:
        result_theta = strat.run(df, max_contracts=10, ticker=ticker, provider=theta)
        theta_avg = result_theta.total_pnl / result_theta.total_trades if result_theta.total_trades else 0
        print(f"{'':<8} {'Theta Data':<18} {result_theta.total_trades:>7} {result_theta.win_rate:>5.1%} "
              f"${result_theta.total_pnl:>+11,.0f} ${theta_avg:>+9.2f} {result_theta.max_drawdown_pct:>7.1%}")

        # Difference
        pnl_diff = result_theta.total_pnl - result_bs.total_pnl
        trade_diff = result_theta.total_trades - result_bs.total_trades
        print(f"{'':<8} {'diff':<18} {trade_diff:>+7} {'':>6} ${pnl_diff:>+11,.0f} {'':>10} {'':>8}")

        summary.append({
            "ticker": ticker,
            "bs_trades": result_bs.total_trades,
            "bs_pnl": result_bs.total_pnl,
            "bs_wr": result_bs.win_rate,
            "theta_trades": result_theta.total_trades,
            "theta_pnl": result_theta.total_pnl,
            "theta_wr": result_theta.win_rate,
            "pnl_diff": pnl_diff,
            "pnl_diff_pct": (pnl_diff / result_bs.total_pnl * 100) if result_bs.total_pnl != 0 else 0,
        })
    except Exception as e:
        print(f"{'':<8} {'Theta Data':<18} FAILED: {e}")

    print()

# Summary
if summary:
    print("=" * 95)
    print(f"{'SUMMARY':^95}")
    print("=" * 95)
    print(f"{'Ticker':<8} {'B-S P/L':>12} {'Theta P/L':>12} {'Delta $':>12} {'Delta %':>10}  Verdict")
    print("-" * 95)
    for s in summary:
        verdict = (
            "B-S close enough" if abs(s["pnl_diff_pct"]) < 15
            else "B-S moderately off" if abs(s["pnl_diff_pct"]) < 40
            else "B-S WILDLY off"
        )
        print(f"{s['ticker']:<8} ${s['bs_pnl']:>+11,.0f} ${s['theta_pnl']:>+11,.0f} "
              f"${s['pnl_diff']:>+11,.0f} {s['pnl_diff_pct']:>+9.1f}%  {verdict}")
    print()
    print("Interpretation:")
    print("  'B-S close enough'   -- use B-S for this ticker, results are trustworthy")
    print("  'B-S moderately off' -- known bias, apply correction or use Theta for validation")
    print("  'B-S WILDLY off'     -- cannot use B-S for absolute P/L on this ticker")
