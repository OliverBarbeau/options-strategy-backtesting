"""Walk-forward validation framework.

Partitions the backtest period into train/test windows and runs the
strategy on each test window with parameters frozen from the training
period. This provides true out-of-sample performance metrics.

Usage::

    from tradelab.walkforward import WalkForwardRunner, WalkForwardConfig

    wf = WalkForwardRunner(
        base_config=config,
        provider=provider,
        wf_config=WalkForwardConfig(min_train_days=252, step_days=63),
    )
    result = wf.run()
    print(result.summary())
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from tradelab.account import SimulatedAccount
from tradelab.portfolio_simulator import (
    PortfolioConfig,
    PortfolioReport,
    PortfolioSimulator,
    _wilson_ci,
)
from tradelab.pricing.thetadata import ThetaDataProvider


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward validation."""
    # Minimum trading days for training window before first test.
    min_train_days: int = 252
    # Number of calendar days per test window (step forward).
    step_days: int = 63
    # If True, training window expands from start. If False, slides (fixed width).
    expanding: bool = True


@dataclass
class WindowResult:
    """Result for a single train/test window."""
    window_idx: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_report: PortfolioReport | None
    test_report: PortfolioReport | None


@dataclass
class WalkForwardResult:
    """Aggregated walk-forward validation results."""
    windows: list[WindowResult]
    config: WalkForwardConfig

    @property
    def oos_trades(self) -> int:
        return sum(w.test_report.total_trades for w in self.windows if w.test_report)

    @property
    def oos_winners(self) -> int:
        return sum(w.test_report.winners for w in self.windows if w.test_report)

    @property
    def oos_win_rate(self) -> float:
        total = self.oos_trades
        return self.oos_winners / total if total > 0 else 0.0

    @property
    def oos_total_pnl(self) -> float:
        return sum(w.test_report.total_pnl for w in self.windows if w.test_report)

    def summary(self) -> str:
        wr_ci_lo, wr_ci_hi = _wilson_ci(self.oos_winners, self.oos_trades)
        lines = [
            "=" * 80,
            "WALK-FORWARD VALIDATION RESULTS",
            "=" * 80,
            f"  Windows:         {len(self.windows)}",
            f"  Mode:            {'Expanding' if self.config.expanding else 'Sliding'} window",
            f"  Step size:       {self.config.step_days} calendar days",
            "",
            "  Out-of-sample aggregate:",
            f"    Total trades:  {self.oos_trades}",
            f"    Win rate:      {self.oos_win_rate:.1%}  [{wr_ci_lo:.1%}, {wr_ci_hi:.1%}] 95% CI",
            f"    Total P/L:     ${self.oos_total_pnl:+,.2f}",
            "",
            f"  {'Window':<8} {'Train':>20} {'Test':>20} {'Trades':>8} {'WR':>8} {'P/L':>12}",
            "-" * 80,
        ]
        for w in self.windows:
            tr = w.test_report
            if tr is None:
                continue
            lines.append(
                f"  {w.window_idx:<8} "
                f"{w.train_start}->{w.train_end} "
                f"{w.test_start}->{w.test_end} "
                f"{tr.total_trades:>5} "
                f"{tr.win_rate:>7.0%} "
                f"${tr.total_pnl:>+10,.2f}"
            )

        # In-sample vs OOS comparison
        is_trades = sum(
            w.train_report.total_trades for w in self.windows if w.train_report
        )
        is_winners = sum(
            w.train_report.winners for w in self.windows if w.train_report
        )
        is_wr = is_winners / is_trades if is_trades > 0 else 0
        lines.extend([
            "",
            "  In-sample vs Out-of-sample:",
            f"    In-sample WR:  {is_wr:.1%} ({is_trades} trades)",
            f"    OOS WR:        {self.oos_win_rate:.1%} ({self.oos_trades} trades)",
            f"    Delta:         {self.oos_win_rate - is_wr:+.1%}",
        ])
        lines.append("=" * 80)
        return "\n".join(lines)


class WalkForwardRunner:
    """Runs walk-forward validation on the portfolio strategy."""

    def __init__(
        self,
        base_config: PortfolioConfig,
        provider: ThetaDataProvider,
        wf_config: WalkForwardConfig | None = None,
        verbose: bool = False,
    ):
        self.base_config = base_config
        self.provider = provider
        self.wf_config = wf_config or WalkForwardConfig()
        self.verbose = verbose

    def _build_windows(self) -> list[tuple[str, str, str, str]]:
        """Build (train_start, train_end, test_start, test_end) tuples."""
        start = datetime.fromisoformat(self.base_config.start_date)
        end = datetime.fromisoformat(self.base_config.end_date)
        step = timedelta(days=self.wf_config.step_days)
        # Approximate trading days as calendar days * 252/365
        min_train_cal = int(self.wf_config.min_train_days * 365 / 252)

        windows = []
        train_start = start
        test_start = start + timedelta(days=min_train_cal)

        while test_start < end:
            test_end = min(test_start + step, end)
            train_end = test_start - timedelta(days=1)

            if not self.wf_config.expanding:
                # Sliding window: train window has fixed width
                train_start = train_end - timedelta(days=min_train_cal)

            windows.append((
                train_start.strftime("%Y-%m-%d"),
                train_end.strftime("%Y-%m-%d"),
                test_start.strftime("%Y-%m-%d"),
                test_end.strftime("%Y-%m-%d"),
            ))

            test_start = test_end

        return windows

    def _run_window(
        self, start: str, end: str, name: str
    ) -> PortfolioReport | None:
        """Run a single backtest window."""
        import tempfile

        cfg = deepcopy(self.base_config)
        cfg.start_date = start
        cfg.end_date = end

        import os
        tmp = tempfile.NamedTemporaryFile(
            suffix=".json", prefix=f"wf_{name}_", delete=False
        )
        tmp_path = tmp.name
        tmp.close()
        os.unlink(tmp_path)  # remove so load_or_create creates fresh
        account = SimulatedAccount.load_or_create(
            tmp_path,
            starting_capital=cfg.starting_capital,
            name=f"wf_{name}",
        )

        sim = PortfolioSimulator(account, self.provider, cfg, verbose=False)
        try:
            sim.run()
            return sim.report()
        except Exception as e:
            if self.verbose:
                print(f"  Window {name} failed: {e}")
            return None

    def run(self) -> WalkForwardResult:
        """Execute all walk-forward windows and aggregate results."""
        windows = self._build_windows()
        if not windows:
            return WalkForwardResult(windows=[], config=self.wf_config)

        if self.verbose:
            print(f"Walk-forward: {len(windows)} windows, "
                  f"{'expanding' if self.wf_config.expanding else 'sliding'} mode")

        results = []
        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            if self.verbose:
                print(f"\n  Window {i}: train {train_start}->{train_end}, "
                      f"test {test_start}->{test_end}")

            train_report = self._run_window(
                train_start, train_end, f"train_{i}"
            )
            test_report = self._run_window(
                test_start, test_end, f"test_{i}"
            )

            results.append(WindowResult(
                window_idx=i,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_report=train_report,
                test_report=test_report,
            ))

        return WalkForwardResult(windows=results, config=self.wf_config)
