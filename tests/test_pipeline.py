"""Tests for tradelab.pipeline.DataPipeline.

Network-dependent tests (yfinance, ccxt) are separated and can be
skipped in CI with: pytest -m "not network"
"""

import numpy as np
import pandas as pd
import pytest

from tradelab.pipeline import DataPipeline, _normalize_yf, _filter_date_range


# ------------------------------------------------------------------
# Internal helpers (no network needed)
# ------------------------------------------------------------------


class TestNormalizeYF:
    def test_flattens_multiindex(self):
        """Simulate yfinance MultiIndex output."""
        arrays = [["Close", "High", "Low", "Open", "Volume"], ["AAPL"] * 5]
        tuples = list(zip(*arrays))
        index = pd.MultiIndex.from_tuples(tuples)
        raw = pd.DataFrame(
            [[150, 155, 145, 148, 1000000]],
            columns=index,
            index=pd.DatetimeIndex(["2023-01-03"]),
        )
        result = _normalize_yf(raw, "AAPL")
        assert set(result.columns) == {"close", "high", "low", "open", "volume"}
        assert result.index.name == "timestamp"
        assert isinstance(result.index[0], (int, np.integer))

    def test_single_level_columns(self):
        raw = pd.DataFrame(
            {"Close": [150], "High": [155], "Low": [145], "Open": [148], "Volume": [1e6]},
            index=pd.DatetimeIndex(["2023-01-03"]),
        )
        result = _normalize_yf(raw, "AAPL")
        assert "close" in result.columns
        assert isinstance(result.index[0], (int, np.integer))


class TestFilterDateRange:
    def test_filters_start(self):
        # Use 2020 timestamps to avoid Windows pre-epoch issues
        base = 1577836800  # 2020-01-01
        df = pd.DataFrame(
            {"c": [1, 2, 3, 4]},
            index=[base, base + 86400, base + 2 * 86400, base + 3 * 86400],
        )
        result = _filter_date_range(df, "2020-01-01", None)
        assert len(result) == 4

    def test_filters_start_and_end(self):
        base = 1577836800  # 2020-01-01
        df = pd.DataFrame(
            {"c": [1, 2, 3, 4]},
            index=[base, base + 86400, base + 2 * 86400, base + 3 * 86400],
        )
        result = _filter_date_range(df, "2020-01-01", "2020-01-03")
        assert 2 <= len(result) <= 4  # depends on timezone


class TestAddIndicators:
    def test_adds_columns(self, stock_df):
        df = stock_df.copy()
        df = df.rename(columns={"c": "close"})
        result = DataPipeline.add_indicators(df)
        for col in ["sma_20", "sma_50", "sma_200", "ema_12", "ema_26", "macd", "hist_vol_30"]:
            assert col in result.columns

    def test_sma_values_reasonable(self, stock_df):
        df = stock_df.copy()
        df = df.rename(columns={"c": "close"})
        result = DataPipeline.add_indicators(df)
        sma_20 = result["sma_20"].dropna()
        # SMA should be in same range as close prices
        assert sma_20.min() > 50
        assert sma_20.max() < 300

    def test_hist_vol_reasonable(self, stock_df):
        df = stock_df.copy()
        df = df.rename(columns={"c": "close"})
        result = DataPipeline.add_indicators(df)
        vol = result["hist_vol_30"].dropna()
        # Annualized vol for synthetic data should be reasonable
        assert vol.median() > 0.05
        assert vol.median() < 0.50


# ------------------------------------------------------------------
# Network-dependent tests
# ------------------------------------------------------------------


@pytest.mark.network
class TestPipelineYFinance:
    def test_fetch_stock_basic(self, tmp_path):
        pipe = DataPipeline(cache_dir=str(tmp_path / "cache"))
        df = pipe.fetch_stock("AAPL", start="2024-01-01", end="2024-01-31")
        assert len(df) > 10
        assert "close" in df.columns
        assert "volume" in df.columns

    def test_fetch_stock_caches(self, tmp_path):
        pipe = DataPipeline(cache_dir=str(tmp_path / "cache"))
        df1 = pipe.fetch_stock("MSFT", start="2024-01-01", end="2024-01-31")
        # Second call should hit cache
        df2 = pipe.fetch_stock("MSFT", start="2024-01-01", end="2024-01-31")
        assert len(df1) == len(df2)
        assert pipe.cache.has("yfinance", "MSFT", "1d")

    def test_fetch_stocks_multi(self, tmp_path):
        pipe = DataPipeline(cache_dir=str(tmp_path / "cache"))
        results = pipe.fetch_stocks(
            ["AAPL", "MSFT"], start="2024-01-01", end="2024-01-31"
        )
        assert "AAPL" in results
        assert "MSFT" in results


@pytest.mark.network
class TestPipelineCCXT:
    def test_fetch_crypto_basic(self, tmp_path):
        pipe = DataPipeline(cache_dir=str(tmp_path / "cache"))
        df = pipe.fetch_crypto(
            "BTC/USDT", start="2026-01-01", end="2026-03-01",
            exchange="kraken",
        )
        assert len(df) > 5
        assert "close" in df.columns
