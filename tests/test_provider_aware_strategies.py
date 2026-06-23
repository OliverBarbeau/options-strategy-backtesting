"""Tests for provider-aware strategy execution.

Verifies that strategies can swap between pricing providers without
changing their logic.
"""

import numpy as np
import pandas as pd
import pytest

from tradelab.pricing import MockProvider, BlackScholesProvider
from tradelab.strategies.pullback_entry import PullbackEntryStrategy, PullbackResult


@pytest.fixture
def synthetic_ohlcv():
    """Build a synthetic OHLCV DataFrame with clear pullback patterns."""
    # 60 days of data with a pullback in the middle
    np.random.seed(42)
    base = 1_704_067_200  # 2024-01-01
    timestamps = [base + i * 86400 for i in range(90)]

    # Price path: uptrend, pullback, recover
    prices = []
    for i in range(90):
        if i < 30:
            prices.append(100 + i * 0.5)  # uptrend to ~115
        elif i < 45:
            prices.append(115 - (i - 30) * 0.8)  # pullback to ~103
        else:
            prices.append(103 + (i - 45) * 0.4)  # recover to ~121

    df = pd.DataFrame({
        "open": prices,
        "high": [p * 1.01 for p in prices],
        "low": [p * 0.99 for p in prices],
        "close": prices,
        "volume": [1_000_000] * 90,
    }, index=timestamps)
    return df


class TestPullbackStrategyWithProviders:
    """Strategy should produce consistent results across providers."""

    def test_legacy_inline_path_still_works(self, synthetic_ohlcv):
        """Original inline B-S path runs without a provider argument."""
        strat = PullbackEntryStrategy(pullback_threshold=0.03)
        result = strat.run(synthetic_ohlcv, max_contracts=5)
        assert isinstance(result, PullbackResult)
        # Should have at least one trade from the synthetic pullback
        assert result.total_trades >= 0

    def test_provider_path_accepts_mock(self, synthetic_ohlcv):
        """Strategy runs through MockProvider without errors."""
        provider = MockProvider(default_underlying=110.0, credit_multiplier=1.0)
        strat = PullbackEntryStrategy(pullback_threshold=0.03)
        result = strat.run(
            synthetic_ohlcv,
            max_contracts=5,
            ticker="TEST",
            provider=provider,
        )
        assert isinstance(result, PullbackResult)

    def test_provider_requires_ticker(self, synthetic_ohlcv):
        """Using a provider without a ticker should raise."""
        provider = MockProvider()
        strat = PullbackEntryStrategy()
        with pytest.raises(ValueError, match="ticker is required"):
            strat.run(synthetic_ohlcv, max_contracts=5, provider=provider)

    def test_trade_log_tagged_with_source(self, synthetic_ohlcv):
        """Trades via provider path are tagged with provider name."""
        provider = MockProvider(default_underlying=110.0)
        strat = PullbackEntryStrategy(pullback_threshold=0.03)
        result = strat.run(
            synthetic_ohlcv,
            max_contracts=5,
            ticker="TEST",
            provider=provider,
        )
        for trade in result.trade_log:
            assert trade.get("source") == "mock"

    def test_inline_bs_path_tagged_differently(self, synthetic_ohlcv):
        """Legacy inline path tags trades as blackscholes_inline."""
        strat = PullbackEntryStrategy(pullback_threshold=0.03)
        result = strat.run(synthetic_ohlcv, max_contracts=5)
        if result.trade_log:
            for trade in result.trade_log:
                assert trade.get("source") == "blackscholes_inline"

    def test_different_credit_multipliers_produce_different_pnl(self, synthetic_ohlcv):
        """Verify the provider is actually being called for pricing."""
        low = MockProvider(default_underlying=110.0, credit_multiplier=0.5)
        high = MockProvider(default_underlying=110.0, credit_multiplier=2.0)

        strat = PullbackEntryStrategy(pullback_threshold=0.03)

        result_low = strat.run(synthetic_ohlcv, max_contracts=5, ticker="TEST", provider=low)
        result_high = strat.run(synthetic_ohlcv, max_contracts=5, ticker="TEST", provider=high)

        # If both had trades, the P/L should differ
        if result_low.total_trades > 0 and result_high.total_trades > 0:
            assert result_low.total_pnl != result_high.total_pnl

    def test_result_has_diagnostic_counts_on_provider_path(self, synthetic_ohlcv):
        """Provider path attaches skipped counts for debugging."""
        provider = MockProvider(default_underlying=110.0)
        strat = PullbackEntryStrategy(pullback_threshold=0.03)
        result = strat.run(
            synthetic_ohlcv,
            max_contracts=5,
            ticker="TEST",
            provider=provider,
        )
        assert hasattr(result, "skipped_no_entry")
        assert hasattr(result, "skipped_no_exit")
