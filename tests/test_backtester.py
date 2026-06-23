"""Integration tests for tradelab.backtester."""

import pytest
import pandas as pd

from tradelab.models import Position
from tradelab.backtester import Backtester, BacktestResult


class TestBacktesterBuyAndHold:
    """Default signal (always long) = buy and hold."""

    def test_returns_match_market(self, stock_df):
        bt = Backtester(capital=10_000, leverage=1, base_fee_rate=0.0)
        result = bt.run(stock_df, price_col="c")
        # With no fees and 1x leverage, strategy should track market
        assert result.strategy_return_pct == pytest.approx(
            result.market_return_pct, abs=0.5
        )

    def test_single_trade_with_fees(self, stock_df):
        bt = Backtester(capital=10_000, leverage=1)
        result = bt.run(stock_df, price_col="c")
        assert result.total_fees > 0
        assert result.trade_count >= 1  # at least the final close

    def test_result_has_equity_curve(self, stock_df):
        bt = Backtester(capital=10_000)
        result = bt.run(stock_df, price_col="c")
        assert isinstance(result.trades, pd.DataFrame)
        assert "market" in result.trades.columns
        assert "strategy" in result.trades.columns
        assert len(result.trades) == len(stock_df)


class TestBacktesterSignals:
    def test_always_flat_earns_nothing(self, stock_df):
        bt = Backtester(capital=10_000, base_fee_rate=0.0)
        result = bt.run(
            stock_df, price_col="c", signal_fn=lambda i, r, p: None
        )
        assert result.strategy_return_pct == pytest.approx(0.0)
        assert result.trade_count == 0

    def test_alternating_signal_generates_trades(self, stock_df):
        call_count = [0]

        def alternating(index, row, pos):
            call_count[0] += 1
            return Position.LONG if call_count[0] % 20 < 10 else None

        bt = Backtester(capital=10_000)
        result = bt.run(stock_df, price_col="c", signal_fn=alternating)
        assert result.trade_count > 2

    def test_short_disabled(self, stock_df):
        def short_signal(index, row, pos):
            return Position.SHORT

        bt = Backtester(capital=10_000, allow_short=False)
        result = bt.run(stock_df, price_col="c", signal_fn=short_signal)
        # Should never open a position
        assert result.trade_count == 0
        assert result.strategy_return_pct == pytest.approx(0.0)


class TestBacktesterEdgeCases:
    def test_flat_market_fee_drag(self, flat_stock_df):
        """On a flat market with fees, strategy should lose money from fees."""
        call_count = [0]

        def churn(index, row, pos):
            call_count[0] += 1
            return Position.LONG if call_count[0] % 10 < 5 else None

        bt = Backtester(capital=10_000)
        result = bt.run(flat_stock_df, price_col="c", signal_fn=churn)
        assert result.total_fees > 0
        assert result.final_equity < 10_000  # fees erode capital

    def test_leverage_amplifies_returns(self, stock_df):
        bt_1x = Backtester(capital=10_000, leverage=1, base_fee_rate=0.0)
        bt_3x = Backtester(capital=10_000, leverage=3, base_fee_rate=0.0)
        r1 = bt_1x.run(stock_df, price_col="c")
        r3 = bt_3x.run(stock_df, price_col="c")
        # 3x leverage should amplify magnitude of returns (positive or negative)
        assert abs(r3.strategy_return_pct) > abs(r1.strategy_return_pct)

    def test_summary_string(self, stock_df):
        bt = Backtester(capital=10_000)
        result = bt.run(stock_df, price_col="c")
        text = result.summary()
        assert "Market:" in text
        assert "Alpha:" in text
        assert "$" in text
