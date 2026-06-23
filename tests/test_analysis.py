"""Tests for tradelab.analysis.ProbabilityEngine."""

import pytest
import pandas as pd

from tradelab.analysis import ProbabilityEngine


class TestProbabilityEngine:
    def test_time_period_loss_basic(self, stock_df):
        engine = ProbabilityEngine(stock_df[["c"]])
        hits, total = engine.time_period_loss(offset=10, price_adjust=0.95)
        assert total == len(stock_df) - 10
        assert 0 <= hits <= total

    def test_adjust_1_means_any_drop(self, stock_df):
        engine = ProbabilityEngine(stock_df[["c"]])
        hits, total = engine.time_period_loss(offset=5, price_adjust=1.0)
        # price_adjust=1.0 means "how often was price lower at all"
        assert hits > 0  # with random walk, some drops are expected

    def test_extreme_adjust_zero_hits(self, stock_df):
        engine = ProbabilityEngine(stock_df[["c"]])
        hits, _ = engine.time_period_loss(offset=5, price_adjust=0.01)
        # 99% drop is effectively impossible in our synthetic data
        assert hits == 0

    def test_offset_larger_than_data(self, stock_df):
        engine = ProbabilityEngine(stock_df[["c"]])
        hits, total = engine.time_period_loss(offset=9999)
        assert total == 0
        assert hits == 0

    def test_no_data_raises(self):
        engine = ProbabilityEngine()
        with pytest.raises(ValueError, match="No data loaded"):
            engine.time_period_loss()

    def test_strike_probability_table(self, stock_df):
        engine = ProbabilityEngine(stock_df[["c"]])
        table = engine.strike_probability_table(
            offset=20, adjust_start=0.90, adjust_end=1.00, step=0.05
        )
        assert isinstance(table, pd.DataFrame)
        assert list(table.columns) == ["adjust", "adj_price", "probability", "prob_delta"]
        assert len(table) == 3  # 0.90, 0.95, 1.00

    def test_probability_increases_with_adjust(self, stock_df):
        engine = ProbabilityEngine(stock_df[["c"]])
        table = engine.strike_probability_table(
            offset=20, adjust_start=0.80, adjust_end=1.05, step=0.05
        )
        # Higher adjust -> more likely the price is below threshold
        probs = table["probability"].tolist()
        assert probs == sorted(probs)

    def test_declining_stock_high_loss_probability(self, declining_stock_df):
        engine = ProbabilityEngine(declining_stock_df)
        hits, total = engine.time_period_loss(offset=20, price_adjust=1.0)
        # In a downtrend, most of the time the future price is lower
        assert hits / total > 0.5

    def test_summary_output(self, stock_df):
        engine = ProbabilityEngine(stock_df[["c"]])
        text = engine.summary("TEST", offset=20)
        assert "TEST" in text
        assert "$" in text
        assert "days" in text
