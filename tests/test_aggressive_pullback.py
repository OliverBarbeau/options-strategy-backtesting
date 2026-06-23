"""Tests for AggressivePullbackStrategy."""

import numpy as np
import pandas as pd
import pytest

from tradelab.strategies.aggressive_pullback import (
    AggressivePullbackStrategy,
    AggressiveResult,
)


class TestAggressiveResult:
    def test_win_rate_no_trades(self):
        result = AggressiveResult()
        assert result.win_rate == 0.0

    def test_win_rate_calculation(self):
        result = AggressiveResult(total_trades=10, winners=7, losers=3)
        assert result.win_rate == pytest.approx(0.7)

    def test_summary_no_trades(self):
        result = AggressiveResult()
        assert result.summary() == "No trades"

    def test_summary_with_trades(self):
        result = AggressiveResult(
            total_trades=20,
            winners=14,
            losers=6,
            total_pnl=500.0,
            max_drawdown_pct=-0.05,
            stacked_trades=4,
        )
        text = result.summary()
        assert "20" in text
        assert "70.0%" in text
        assert "+500" in text
        assert "Stacked" in text


class TestQualifies:
    def _make_df(self, prices):
        n = len(prices)
        ts = np.arange(n) * 86400 + 1_600_000_000
        return pd.DataFrame(
            {"close": prices, "volume": 1_000_000},
            index=ts,
        )

    def test_no_pullback(self):
        prices = np.linspace(100, 110, 50)
        df = self._make_df(prices)
        strategy = AggressivePullbackStrategy()
        qualifies, drawdown = strategy.qualifies(df, 49)
        assert not qualifies
        assert drawdown > -0.03

    def test_pullback_triggers(self):
        prices = np.concatenate([
            np.linspace(100, 120, 30),
            np.linspace(120, 114, 10),
        ])
        df = self._make_df(prices)
        strategy = AggressivePullbackStrategy()
        qualifies, drawdown = strategy.qualifies(df, 39)
        assert qualifies
        assert drawdown < -0.03

    def test_deep_pullback(self):
        prices = np.concatenate([
            np.linspace(100, 120, 30),
            np.linspace(120, 112, 10),
        ])
        df = self._make_df(prices)
        strategy = AggressivePullbackStrategy()
        qualifies, drawdown = strategy.qualifies(df, 39)
        assert qualifies
        assert drawdown <= -0.05

    def test_idx_too_early(self):
        prices = np.linspace(100, 90, 50)
        df = self._make_df(prices)
        strategy = AggressivePullbackStrategy()
        qualifies, _ = strategy.qualifies(df, 5)
        assert not qualifies


class TestAggressivePullbackStrategy:
    def test_basic_run(self, stock_df):
        strategy = AggressivePullbackStrategy()
        result = strategy.run(stock_df, close_col="c")
        assert isinstance(result, AggressiveResult)
        assert result.total_trades >= 0
        assert 0.0 <= result.win_rate <= 1.0

    def test_run_on_declining_stock(self, declining_stock_df):
        strategy = AggressivePullbackStrategy()
        result = strategy.run(declining_stock_df, close_col="c")
        assert isinstance(result, AggressiveResult)
        assert result.total_trades > 0
        assert result.losers > 0

    def test_stacking_on_deep_pullback(self, declining_stock_df):
        strategy = AggressivePullbackStrategy(
            pullback_threshold=0.02,
            deep_pullback=0.04,
        )
        result = strategy.run(declining_stock_df, close_col="c")
        assert result.total_trades > 0
        stacked = [t for t in result.trade_log if t["stacked"]]
        assert result.stacked_trades == len(stacked)

    def test_streak_bonus_uses_tighter_buffer(self, stock_df):
        strategy = AggressivePullbackStrategy(
            pullback_threshold=0.005,
            streak_bonus_threshold=1,
            buffer=0.07,
            streak_buffer=0.05,
        )
        result = strategy.run(stock_df, close_col="c")
        streak_trades = [t for t in result.trade_log if t["streak_entry"]]
        for t in streak_trades:
            assert t["buffer_used"] == pytest.approx(0.05)

    def test_custom_parameters(self, stock_df):
        strategy = AggressivePullbackStrategy(
            buffer=0.08,
            spread_pct=0.025,
            pullback_threshold=0.02,
            dte_open=21,
            dte_close=7,
        )
        result = strategy.run(stock_df, close_col="c")
        assert isinstance(result, AggressiveResult)

    def test_trade_log_fields(self, declining_stock_df):
        strategy = AggressivePullbackStrategy(pullback_threshold=0.02)
        result = strategy.run(declining_stock_df, close_col="c")
        if result.total_trades > 0:
            trade = result.trade_log[0]
            expected_keys = {
                "date", "exit_date", "entry_price", "exit_price",
                "pullback_pct", "sigma", "contracts", "credit",
                "pnl", "winner", "buffer_used", "short_strike",
                "long_strike", "stacked", "streak_entry",
            }
            assert set(trade.keys()) == expected_keys

    def test_max_contracts_respected(self, declining_stock_df):
        strategy = AggressivePullbackStrategy(pullback_threshold=0.02)
        result = strategy.run(declining_stock_df, close_col="c", max_contracts=5)
        for t in result.trade_log:
            assert t["contracts"] == 5

    def test_flat_market_few_trades(self, flat_stock_df):
        strategy = AggressivePullbackStrategy()
        result = strategy.run(flat_stock_df, close_col="c")
        assert result.total_trades == 0

    def test_winners_plus_losers_equals_total(self, declining_stock_df):
        strategy = AggressivePullbackStrategy(pullback_threshold=0.02)
        result = strategy.run(declining_stock_df, close_col="c")
        assert result.winners + result.losers == result.total_trades
