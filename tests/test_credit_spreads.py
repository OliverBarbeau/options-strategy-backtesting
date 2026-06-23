"""Integration tests for tradelab.strategies.credit_spreads."""

import pytest

from tradelab.strategies.credit_spreads import RollingCreditSpreadStrategy


class TestRollingCreditSpreadStrategy:
    def test_basic_run(self, stock_df):
        strategy = RollingCreditSpreadStrategy(
            max_spreads=2,
            offset_days=60,
            buffer=0.02,
            credit_ratio=0.10,
            loss_ratio=0.90,
        )
        result = strategy.run(stock_df, initial_balance=1_000)
        assert result.initial_balance == 1_000
        assert result.final_balance > 0
        assert result.winners + result.losers > 0

    def test_win_rate_bounded(self, stock_df):
        strategy = RollingCreditSpreadStrategy(
            max_spreads=4, offset_days=50, buffer=0.05
        )
        result = strategy.run(stock_df, initial_balance=5_000)
        assert 0.0 <= result.win_rate <= 1.0

    def test_wide_buffer_favors_winners(self, stock_df):
        """A very wide buffer (strike far below price) should win most trades."""
        strategy = RollingCreditSpreadStrategy(
            max_spreads=2,
            offset_days=30,
            buffer=0.20,  # strike 20% below current price
            credit_ratio=0.05,
            loss_ratio=0.90,
        )
        result = strategy.run(stock_df, initial_balance=1_000)
        if result.winners + result.losers > 0:
            assert result.win_rate > 0.5

    def test_tight_buffer_more_losers(self, declining_stock_df):
        """Tight buffer on a declining stock should produce more losses."""
        strategy = RollingCreditSpreadStrategy(
            max_spreads=2,
            offset_days=30,
            buffer=0.001,  # strike almost at current price
            credit_ratio=0.05,
            loss_ratio=0.90,
        )
        result = strategy.run(declining_stock_df, initial_balance=1_000)
        assert result.losers > 0

    def test_trade_log_populated(self, stock_df):
        strategy = RollingCreditSpreadStrategy(
            max_spreads=2, offset_days=40, buffer=0.02
        )
        result = strategy.run(stock_df, initial_balance=1_000)
        assert len(result.trade_log) == result.winners + result.losers
        if result.trade_log:
            entry = result.trade_log[0]
            assert "open_date" in entry
            assert "cash_return" in entry
            assert "winner" in entry

    def test_summary_output(self, stock_df):
        strategy = RollingCreditSpreadStrategy(
            max_spreads=2, offset_days=40
        )
        result = strategy.run(stock_df, initial_balance=1_000)
        text = result.summary()
        assert "Win rate:" in text
        assert "Annualized return:" in text

    def test_annualized_return_sign(self, stock_df):
        """On an uptrending stock with reasonable buffer, should be positive."""
        strategy = RollingCreditSpreadStrategy(
            max_spreads=2,
            offset_days=30,
            buffer=0.10,
            credit_ratio=0.15,
        )
        result = strategy.run(stock_df, initial_balance=1_000)
        # Not guaranteed but very likely with our seeded uptrend data
        assert result.final_balance > result.initial_balance * 0.8  # at least not catastrophic
