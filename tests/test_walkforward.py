"""Tests for walk-forward validation and A/B test frameworks."""

from __future__ import annotations

from datetime import datetime

import pytest

from tradelab.walkforward import WalkForwardConfig, WalkForwardRunner
from tradelab.ab_test import ABTestResult, _norm_cdf
from tradelab.portfolio_simulator import PortfolioConfig


class TestWalkForwardWindows:
    """Verify walk-forward window construction."""

    def _make_config(self, start: str, end: str) -> PortfolioConfig:
        return PortfolioConfig(
            tickers=["AAPL"],
            start_date=start,
            end_date=end,
            starting_capital=25000,
        )

    def test_windows_dont_overlap(self):
        """Train and test periods must not overlap within a window."""
        cfg = self._make_config("2020-01-01", "2024-12-31")
        runner = WalkForwardRunner(
            base_config=cfg,
            provider=None,  # type: ignore — not running, just building windows
            wf_config=WalkForwardConfig(min_train_days=252, step_days=63),
        )
        windows = runner._build_windows()
        assert len(windows) > 0

        for train_start, train_end, test_start, test_end in windows:
            # Train end must be before test start
            assert train_end < test_start, (
                f"Overlap: train ends {train_end}, test starts {test_start}"
            )
            # Test start before test end
            assert test_start <= test_end

    def test_expanding_window_grows(self):
        """In expanding mode, train_start should be constant."""
        cfg = self._make_config("2020-01-01", "2024-12-31")
        runner = WalkForwardRunner(
            base_config=cfg,
            provider=None,  # type: ignore
            wf_config=WalkForwardConfig(
                min_train_days=252, step_days=63, expanding=True
            ),
        )
        windows = runner._build_windows()
        assert len(windows) >= 2

        # All windows should have the same train start
        starts = [w[0] for w in windows]
        assert len(set(starts)) == 1, "Expanding window: train_start should be constant"

    def test_sliding_window_moves(self):
        """In sliding mode, train_start should advance."""
        cfg = self._make_config("2020-01-01", "2024-12-31")
        runner = WalkForwardRunner(
            base_config=cfg,
            provider=None,  # type: ignore
            wf_config=WalkForwardConfig(
                min_train_days=252, step_days=63, expanding=False
            ),
        )
        windows = runner._build_windows()
        assert len(windows) >= 2

        starts = [w[0] for w in windows]
        # At least some train starts should differ
        assert len(set(starts)) > 1, "Sliding window: train_start should move"

    def test_test_periods_cover_full_range(self):
        """Test periods should tile from first test start to end date."""
        cfg = self._make_config("2020-01-01", "2023-12-31")
        runner = WalkForwardRunner(
            base_config=cfg,
            provider=None,  # type: ignore
            wf_config=WalkForwardConfig(min_train_days=252, step_days=90),
        )
        windows = runner._build_windows()
        assert len(windows) >= 2

        # Last test_end should reach or approach the end date
        last_test_end = windows[-1][3]
        assert last_test_end == "2023-12-31"

    def test_short_period_no_windows(self):
        """Period shorter than min_train should produce no windows."""
        cfg = self._make_config("2024-01-01", "2024-06-01")
        runner = WalkForwardRunner(
            base_config=cfg,
            provider=None,  # type: ignore
            wf_config=WalkForwardConfig(min_train_days=252, step_days=63),
        )
        windows = runner._build_windows()
        assert len(windows) == 0


class TestNormCDF:
    """Basic test for the normal CDF approximation."""

    def test_zero(self):
        assert _norm_cdf(0) == pytest.approx(0.5, abs=0.001)

    def test_large_positive(self):
        assert _norm_cdf(3.0) > 0.998

    def test_symmetry(self):
        assert _norm_cdf(1.0) + _norm_cdf(-1.0) == pytest.approx(1.0, abs=0.001)
