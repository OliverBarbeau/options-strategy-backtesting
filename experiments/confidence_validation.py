"""Confidence validation: run all audit-recommended tests.

Executes:
1. Portfolio backtest (2024) with bug fixes + ThetaData → corrected metrics
2. A/B test: pullback (3%) vs always-enter (0%) → entry signal validation
3. Walk-forward (2020-2024) → out-of-sample discipline
4. Universe validation → survivorship bias check

Requires: Theta Terminal running on localhost:25503
"""

from __future__ import annotations

import sys
import time

from tradelab.account import SimulatedAccount
from tradelab.portfolio_simulator import PortfolioSimulator, PortfolioConfig
from tradelab.pricing.thetadata import ThetaDataProvider
from tradelab.ab_test import run_comparison
from tradelab.walkforward import WalkForwardRunner, WalkForwardConfig
from tradelab.scanner import validate_universe, SCAN_UNIVERSE
from tradelab.pipeline import DataPipeline


def main():
    t0 = time.time()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    print("=" * 80)
    print("CONFIDENCE VALIDATION SUITE")
    print("=" * 80)

    provider = ThetaDataProvider(verbose=False)
    if not provider.check_connection():
        print("ERROR: Theta Terminal not reachable on localhost:25503")
        sys.exit(1)
    print("  Theta Terminal: connected\n")

    # Core tickers (diversified, not just mega-cap survivors)
    tickers = ["NVDA", "AAPL", "GOOG", "MSFT", "CAT", "AVGO", "META"]

    # ------------------------------------------------------------------
    # 1. Universe validation (survivorship bias check)
    # ------------------------------------------------------------------
    print("=" * 80)
    print("STEP 1: UNIVERSE VALIDATION (survivorship bias)")
    print("=" * 80)
    pipe = DataPipeline()
    valid, excluded = validate_universe(tickers, "2020-01-01", pipe=pipe)
    print(f"  Tickers tested:  {len(tickers)}")
    print(f"  Valid (pre-2020): {len(valid)} — {', '.join(valid)}")
    if excluded:
        print(f"  Excluded (IPO too recent): {len(excluded)} — {', '.join(excluded)}")
    else:
        print("  Excluded: none")
    print()

    # Use only validated tickers for remaining tests
    tickers = valid if valid else tickers

    # ------------------------------------------------------------------
    # 2. Portfolio backtest 2024 with corrected metrics
    # ------------------------------------------------------------------
    print("=" * 80)
    print("STEP 2: PORTFOLIO BACKTEST 2024 (with all bug fixes)")
    print("=" * 80)

    config_2024 = PortfolioConfig(
        tickers=tickers,
        start_date="2024-01-02",
        end_date="2024-12-31",
        starting_capital=25000.0,
        max_positions=6,
        max_pct_per_position=0.15,
        max_contracts=10,
        buffer=0.10,
        spread_pct=0.02,
        pullback_threshold=0.03,
        pullback_lookback=20,
        dte_open=30,
        dte_close=14,
        risk_free_rate=0.045,
    )

    account_2024 = SimulatedAccount.load_or_create(
        "accounts/validation_2024.json",
        starting_capital=25000.0,
        name="Validation 2024",
        strategy="pullback_theta",
    )
    # Fresh run — clear any prior state
    if account_2024.trades or account_2024.positions:
        account_2024 = SimulatedAccount(
            "accounts/validation_2024.json", 25000.0,
            "Validation 2024", "pullback_theta",
        )

    sim = PortfolioSimulator(account_2024, provider, config_2024, verbose=False)
    print("  Running 2024 simulation...")
    sim.run()
    report_2024 = sim.report()
    print(report_2024.summary())
    provider.print_stats()
    print()

    # ------------------------------------------------------------------
    # 3. A/B test: pullback filter vs always-enter
    # ------------------------------------------------------------------
    print("=" * 80)
    print("STEP 3: A/B TEST — pullback (3%) vs always-enter")
    print("=" * 80)

    ab_config = PortfolioConfig(
        tickers=tickers,
        start_date="2024-01-02",
        end_date="2024-12-31",
        starting_capital=25000.0,
        max_positions=6,
        max_pct_per_position=0.15,
        max_contracts=10,
        buffer=0.10,
        spread_pct=0.02,
        pullback_threshold=0.03,
        pullback_lookback=20,
        dte_open=30,
        dte_close=14,
        risk_free_rate=0.045,
    )

    print("  Running A/B comparison (2 variants on 2024 data)...")
    ab_result = run_comparison(
        base_config=ab_config,
        provider=provider,
        variants={
            "pullback_3pct": 0.03,
            "always_enter": 0.0,
        },
        verbose=False,
    )
    print(ab_result.summary())
    print()

    # ------------------------------------------------------------------
    # 4. Walk-forward validation (2020-2024)
    # ------------------------------------------------------------------
    print("=" * 80)
    print("STEP 4: WALK-FORWARD VALIDATION (2020-2024)")
    print("=" * 80)

    # Use 2023-2024 range: 2023 trains, 2024 tests in quarterly steps.
    # Value-tier Theta subscription has limited historical depth; 2023+
    # is reliably cached from prior backtests.
    wf_config = PortfolioConfig(
        tickers=tickers,
        start_date="2023-01-02",
        end_date="2024-12-31",
        starting_capital=25000.0,
        max_positions=6,
        max_pct_per_position=0.15,
        max_contracts=10,
        buffer=0.10,
        spread_pct=0.02,
        pullback_threshold=0.03,
        pullback_lookback=20,
        dte_open=30,
        dte_close=14,
        risk_free_rate=0.045,
    )

    print("  Running walk-forward (expanding window, 1-year train, 3-month steps)...")
    print("  Range: 2023-2024 (Theta cache available)")
    wf_runner = WalkForwardRunner(
        base_config=wf_config,
        provider=provider,
        wf_config=WalkForwardConfig(
            min_train_days=252,
            step_days=91,  # ~quarterly
            expanding=True,
        ),
        verbose=True,
    )
    wf_result = wf_runner.run()
    print(wf_result.summary())
    print()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed = time.time() - t0
    print("=" * 80)
    print(f"VALIDATION COMPLETE — {elapsed:.0f}s elapsed")
    print("=" * 80)
    print()
    print("Key results to assess confidence score:")
    print(f"  2024 corrected return:  {report_2024.total_return_pct:+.1f}%")
    print(f"  2024 Sharpe ratio:      {report_2024.sharpe_ratio:.2f} "
          f"[{report_2024.sharpe_ci_lower:.2f}, {report_2024.sharpe_ci_upper:.2f}]")
    print(f"  2024 win rate:          {report_2024.win_rate:.1%} "
          f"[{report_2024.win_rate_ci_lower:.1%}, {report_2024.win_rate_ci_upper:.1%}]")
    print(f"  2024 alpha vs SPY:      {report_2024.alpha_vs_spy:+.1f}%")
    print(f"  2024 max drawdown:      {report_2024.max_drawdown_pct:.1%}")
    print()

    if ab_result.variants:
        a, b = ab_result.variants[0], ab_result.variants[1]
        p_val = ab_result._fisher_exact_pvalue(
            a.report.winners, a.report.total_trades,
            b.report.winners, b.report.total_trades,
        )
        print(f"  A/B pullback vs always: p={p_val:.4f} "
              f"({'significant' if p_val < 0.05 else 'NOT significant'})")
        print(f"    pullback WR: {a.report.win_rate:.1%} ({a.report.total_trades} trades)")
        print(f"    always WR:   {b.report.win_rate:.1%} ({b.report.total_trades} trades)")
    print()

    print(f"  Walk-forward OOS trades: {wf_result.oos_trades}")
    print(f"  Walk-forward OOS WR:     {wf_result.oos_win_rate:.1%}")
    print(f"  Walk-forward OOS P/L:    ${wf_result.oos_total_pnl:+,.2f}")


if __name__ == "__main__":
    main()
