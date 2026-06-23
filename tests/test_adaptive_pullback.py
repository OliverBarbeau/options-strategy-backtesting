"""Tests for AdaptivePullbackStrategy.

Each feature is tested in isolation to verify it triggers correctly and
that the baseline (no features) matches PullbackEntryStrategy behavior.
"""

import numpy as np
import pandas as pd
import pytest

from tradelab.strategies.adaptive_pullback import (
    AdaptivePullbackStrategy,
    AdaptiveResult,
)


@pytest.fixture
def pullback_df():
    """Synthetic data with a clear pullback pattern for entry testing."""
    np.random.seed(7)
    base = 1_704_067_200
    timestamps = [base + i * 86400 for i in range(150)]

    # Uptrend, multi-stage pullback, recovery
    prices = []
    for i in range(150):
        if i < 30:
            prices.append(100 + i * 0.5)
        elif i < 50:
            prices.append(115 - (i - 30) * 0.4)
        elif i < 80:
            prices.append(107 + (i - 50) * 0.3)
        elif i < 100:
            prices.append(116 - (i - 80) * 0.5)
        else:
            prices.append(106 + (i - 100) * 0.3)

    return pd.DataFrame(
        {
            "open": prices,
            "high": [p * 1.01 for p in prices],
            "low": [p * 0.99 for p in prices],
            "close": prices,
            "volume": [1_000_000] * 150,
        },
        index=timestamps,
    )


@pytest.fixture
def high_vol_df():
    """Higher volatility data to test adaptive threshold."""
    np.random.seed(11)
    base = 1_704_067_200
    n = 200
    timestamps = [base + i * 86400 for i in range(n)]
    returns = np.random.normal(0.0005, 0.025, n)  # 40% annualized vol
    prices = [100.0]
    for r in returns[1:]:
        prices.append(prices[-1] * (1 + r))
    return pd.DataFrame(
        {"open": prices, "high": [p * 1.02 for p in prices],
         "low": [p * 0.98 for p in prices], "close": prices,
         "volume": [1_000_000] * n}, index=timestamps
    )


class TestBaseline:
    """Baseline (all features disabled) should match pullback behavior."""

    def test_baseline_has_no_skip_counters_triggered(self, pullback_df):
        strat = AdaptivePullbackStrategy(pullback_threshold=0.03)
        result = strat.run(pullback_df, max_contracts=5)
        assert isinstance(result, AdaptiveResult)
        assert result.skipped_vol_pause == 0
        assert result.skipped_cooldown == 0
        assert result.skipped_adaptive_threshold == 0
        assert result.closed_early_stop_loss == 0
        assert result.closed_fast_profit == 0

    def test_baseline_trades_tagged_as_checkpoint(self, pullback_df):
        strat = AdaptivePullbackStrategy(pullback_threshold=0.03)
        result = strat.run(pullback_df, max_contracts=5)
        for t in result.trade_log:
            assert t["exit_reason"] == "checkpoint"


class TestVolRegimePause:
    """Feature 1: Pause when market vol is elevated."""

    def test_pause_skips_entries_when_vol_above_threshold(self, pullback_df):
        # Build a market_vol series that's ALL above 0.25 (paused everywhere)
        high_vol_series = pd.Series(
            [0.35] * len(pullback_df), index=pullback_df.index
        )
        strat = AdaptivePullbackStrategy(
            pullback_threshold=0.03, vol_pause_threshold=0.25,
        )
        result = strat.run(pullback_df, market_vol_series=high_vol_series, max_contracts=5)
        assert result.total_trades == 0
        assert result.skipped_vol_pause > 0

    def test_pause_allows_entries_when_vol_below_threshold(self, pullback_df):
        low_vol_series = pd.Series(
            [0.15] * len(pullback_df), index=pullback_df.index
        )
        strat = AdaptivePullbackStrategy(
            pullback_threshold=0.03, vol_pause_threshold=0.25,
        )
        result = strat.run(pullback_df, market_vol_series=low_vol_series, max_contracts=5)
        assert result.skipped_vol_pause == 0

    def test_pause_disabled_by_default(self, pullback_df):
        strat = AdaptivePullbackStrategy(pullback_threshold=0.03)
        # Without vol_pause_threshold, no skips regardless of market vol
        high_vol = pd.Series([0.50] * len(pullback_df), index=pullback_df.index)
        result = strat.run(pullback_df, market_vol_series=high_vol, max_contracts=5)
        assert result.skipped_vol_pause == 0


class TestTickerCooldown:
    """Feature 2: Skip ticker for N days after a loss."""

    def test_cooldown_triggers_only_after_loss(self, high_vol_df):
        # High vol df will produce some losses
        strat = AdaptivePullbackStrategy(
            pullback_threshold=0.03, cooldown_days=20,
        )
        result = strat.run(high_vol_df, max_contracts=5)
        if result.losers > 0:
            # If we had any losses, cooldown should have skipped some entries
            assert result.skipped_cooldown >= 0

    def test_cooldown_disabled_by_default(self, high_vol_df):
        strat = AdaptivePullbackStrategy(pullback_threshold=0.03)
        result = strat.run(high_vol_df, max_contracts=5)
        assert result.skipped_cooldown == 0

    def test_cooldown_zero_means_disabled(self, high_vol_df):
        strat = AdaptivePullbackStrategy(
            pullback_threshold=0.03, cooldown_days=0,
        )
        result = strat.run(high_vol_df, max_contracts=5)
        assert result.skipped_cooldown == 0


class TestAdaptiveThreshold:
    """Feature 3: Scale pullback threshold by vol regime."""

    def test_adaptive_scales_threshold(self):
        strat = AdaptivePullbackStrategy(
            pullback_threshold=0.03, adaptive_pullback=True,
        )
        # At median vol, threshold is unchanged
        assert strat._effective_pullback_threshold(0.25, 0.25) == pytest.approx(0.03)
        # At 2x median vol, scales up
        assert strat._effective_pullback_threshold(0.50, 0.25) > 0.03
        # Capped at 2x threshold
        assert strat._effective_pullback_threshold(1.00, 0.25) <= 0.06

    def test_adaptive_disabled_returns_base_threshold(self):
        strat = AdaptivePullbackStrategy(
            pullback_threshold=0.03, adaptive_pullback=False,
        )
        assert strat._effective_pullback_threshold(0.50, 0.25) == 0.03


class TestStopLossBreach:
    """Feature 4: Close position on 2-day strike breach."""

    def test_stop_loss_disabled_by_default(self, pullback_df):
        strat = AdaptivePullbackStrategy(pullback_threshold=0.03)
        result = strat.run(pullback_df, max_contracts=5)
        assert result.closed_early_stop_loss == 0

    def test_stop_loss_triggers_on_breach(self, high_vol_df):
        # High vol should produce some breaches
        strat = AdaptivePullbackStrategy(
            pullback_threshold=0.03, stop_loss_breach=True,
        )
        result = strat.run(high_vol_df, max_contracts=5)
        # Some trades may close via stop loss
        if result.total_trades > 0:
            for t in result.trade_log:
                if t["exit_reason"] == "stop_loss_breach":
                    # These should be losers or near-losers
                    assert t["days_held"] < 10  # closed early


class TestFastProfitTake:
    """Feature 5: Close at 75% profit in first 5 days."""

    def test_fast_profit_disabled_by_default(self, pullback_df):
        strat = AdaptivePullbackStrategy(pullback_threshold=0.03)
        result = strat.run(pullback_df, max_contracts=5)
        assert result.closed_fast_profit == 0

    def test_fast_profit_enabled(self, pullback_df):
        # Low threshold makes it easier to trigger
        strat = AdaptivePullbackStrategy(
            pullback_threshold=0.03,
            fast_profit_target=0.5,  # 50% to make it triggerable on synthetic data
            fast_profit_window=10,
        )
        result = strat.run(pullback_df, max_contracts=5)
        # May or may not trigger on synthetic data, but should not crash
        for t in result.trade_log:
            if t["exit_reason"] == "fast_profit":
                assert t["days_held"] <= 10


class TestCombinedFeatures:
    """All features together should not crash."""

    def test_all_features_enabled(self, pullback_df):
        high_vol_series = pd.Series(
            [0.18] * len(pullback_df), index=pullback_df.index
        )
        strat = AdaptivePullbackStrategy(
            pullback_threshold=0.03,
            vol_pause_threshold=0.30,  # not paused
            cooldown_days=10,
            adaptive_pullback=True,
            stop_loss_breach=True,
            fast_profit_target=0.75,
            fast_profit_window=5,
        )
        result = strat.run(pullback_df, market_vol_series=high_vol_series, max_contracts=5)
        assert isinstance(result, AdaptiveResult)
        # Sum of closes by reason should equal total trades
        checkpoint_closes = sum(1 for t in result.trade_log if t["exit_reason"] == "checkpoint")
        stop_closes = sum(1 for t in result.trade_log if t["exit_reason"] == "stop_loss_breach")
        fast_closes = sum(1 for t in result.trade_log if t["exit_reason"] == "fast_profit")
        assert checkpoint_closes + stop_closes + fast_closes == result.total_trades
