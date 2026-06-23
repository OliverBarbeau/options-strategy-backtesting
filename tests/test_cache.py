"""Tests for tradelab.cache.DataCache."""

import pandas as pd
import pytest

from tradelab.cache import DataCache


@pytest.fixture
def cache(tmp_path) -> DataCache:
    return DataCache(str(tmp_path / "cache"))


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {"close": [100, 101, 102, 103, 104]},
        index=[1000, 2000, 3000, 4000, 5000],
    )


class TestDataCache:
    def test_put_and_get(self, cache, sample_df):
        cache.put("test_source", "AAPL", "D", sample_df)
        result = cache.get("test_source", "AAPL", "D")
        assert result is not None
        assert len(result) == 5
        assert list(result.columns) == ["close"]

    def test_get_missing_returns_none(self, cache):
        assert cache.get("test_source", "MISSING", "D") is None

    def test_has(self, cache, sample_df):
        assert cache.has("test_source", "AAPL", "D") is False
        cache.put("test_source", "AAPL", "D", sample_df)
        assert cache.has("test_source", "AAPL", "D") is True

    def test_update_appends(self, cache):
        df1 = pd.DataFrame({"close": [100, 101]}, index=[1000, 2000])
        df2 = pd.DataFrame({"close": [102, 103]}, index=[3000, 4000])

        cache.put("src", "SPY", "D", df1)
        result = cache.update("src", "SPY", "D", df2)
        assert len(result) == 4

    def test_update_deduplicates(self, cache):
        df1 = pd.DataFrame({"close": [100, 101]}, index=[1000, 2000])
        df2 = pd.DataFrame({"close": [999, 103]}, index=[2000, 3000])

        cache.put("src", "SPY", "D", df1)
        result = cache.update("src", "SPY", "D", df2)
        assert len(result) == 3
        # The overlapping row (2000) should use new data
        assert result.loc[2000, "close"] == 999

    def test_list_cached(self, cache, sample_df):
        cache.put("yfinance", "AAPL", "D", sample_df)
        cache.put("yfinance", "MSFT", "D", sample_df)
        cache.put("ccxt", "BTC-USDT", "1d", sample_df)

        yf_files = cache.list_cached("yfinance")
        assert len(yf_files) == 2

        all_files = cache.list_cached()
        assert len(all_files) == 3

    def test_clear_source(self, cache, sample_df):
        cache.put("yfinance", "AAPL", "D", sample_df)
        cache.put("ccxt", "BTC", "1d", sample_df)
        removed = cache.clear("yfinance")
        assert removed == 1
        assert cache.get("yfinance", "AAPL", "D") is None
        assert cache.get("ccxt", "BTC", "1d") is not None

    def test_clear_all(self, cache, sample_df):
        cache.put("yfinance", "AAPL", "D", sample_df)
        cache.put("ccxt", "BTC", "1d", sample_df)
        removed = cache.clear()
        assert removed == 2

    def test_special_chars_in_symbol(self, cache, sample_df):
        cache.put("ccxt", "BTC/USDT", "1h", sample_df)
        result = cache.get("ccxt", "BTC/USDT", "1h")
        assert result is not None
