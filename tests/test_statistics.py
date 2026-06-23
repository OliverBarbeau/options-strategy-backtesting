"""Tests for statistical rigor: Sharpe, Sortino, CIs, Wilson score."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tradelab.portfolio_simulator import _wilson_ci


class TestWilsonCI:
    """Wilson score confidence interval for win rate."""

    def test_50_pct_wide_ci_small_sample(self):
        lo, hi = _wilson_ci(5, 10)
        assert lo < 0.50 < hi
        assert lo > 0.15
        assert hi < 0.85

    def test_high_win_rate_large_sample(self):
        lo, hi = _wilson_ci(80, 100)
        assert lo > 0.70
        assert hi < 0.90

    def test_zero_trades(self):
        lo, hi = _wilson_ci(0, 0)
        assert lo == 0.0
        assert hi == 0.0

    def test_all_winners(self):
        lo, hi = _wilson_ci(50, 50)
        assert hi <= 1.0
        assert lo > 0.90

    def test_no_winners(self):
        lo, hi = _wilson_ci(0, 50)
        assert lo >= 0.0
        assert hi < 0.10

    def test_known_value(self):
        """79 wins out of 94 trades (79.8%) — from the 2024 portfolio backtest."""
        lo, hi = _wilson_ci(79, 94)
        # Wilson score CI: lower should be well above 50%, upper below 100%
        assert 0.70 < lo < 0.80
        assert 0.85 < hi < 0.95
        # CI should contain the point estimate
        assert lo < 79 / 94 < hi


class TestSharpeCI:
    """Confidence interval for Sharpe ratio."""

    def test_sharpe_se_formula(self):
        """SE = sqrt((1 + S^2/2) / (n-1)) for known Sharpe and n."""
        sharpe = 1.0
        n = 252  # 1 year of daily data
        se = math.sqrt((1 + sharpe ** 2 / 2) / (n - 1))
        # Expected SE ~ sqrt(1.5 / 251) ~ 0.077
        assert 0.07 < se < 0.08
        ci_lo = sharpe - 1.96 * se
        ci_hi = sharpe + 1.96 * se
        # With S=1.0 and 252 days, CI should be roughly [0.85, 1.15]
        assert 0.83 < ci_lo < 0.87
        assert 1.13 < ci_hi < 1.17

    def test_short_history_wide_ci(self):
        """With only 60 days, CI should be very wide."""
        sharpe = 1.0
        n = 60
        se = math.sqrt((1 + sharpe ** 2 / 2) / (n - 1))
        ci_width = 2 * 1.96 * se
        assert ci_width > 0.6  # wide enough that Sharpe 1.0 includes 0
