"""Tests for ticker universe validation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from tradelab.scanner import validate_universe


class TestValidateUniverse:
    """Verify IPO date filtering logic."""

    def _mock_pipe(self, data_start: str, data_end: str):
        """Create a mock DataPipeline that returns data for a date range."""
        pipe = MagicMock()
        dates = pd.bdate_range(data_start, data_end)
        ts_index = np.array([int(d.timestamp()) for d in dates])
        df = pd.DataFrame(
            {"close": np.linspace(100, 110, len(dates))},
            index=ts_index,
        )
        pipe.fetch_stock.return_value = df
        return pipe

    def test_valid_ticker_passes(self):
        """Ticker with data well before start_date should pass."""
        pipe = self._mock_pipe("2015-01-01", "2024-01-01")
        valid, excluded = validate_universe(["AAPL"], "2020-01-01", pipe=pipe)
        assert "AAPL" in valid
        assert len(excluded) == 0

    def test_recent_ipo_excluded(self):
        """Ticker with data starting after required date should be excluded."""
        pipe = self._mock_pipe("2020-06-01", "2024-01-01")
        valid, excluded = validate_universe(
            ["NEWIPO"], "2020-01-01", pipe=pipe, min_history_days=60
        )
        assert "NEWIPO" in excluded
        assert len(valid) == 0

    def test_empty_data_excluded(self):
        """Ticker returning no data should be excluded."""
        pipe = MagicMock()
        pipe.fetch_stock.return_value = pd.DataFrame()
        valid, excluded = validate_universe(["DELIST"], "2020-01-01", pipe=pipe)
        assert "DELIST" in excluded

    def test_mixed_universe(self):
        """Mix of valid and invalid tickers."""
        pipe = MagicMock()

        def fetch(ticker, **kwargs):
            if ticker == "OLD":
                dates = pd.bdate_range("2010-01-01", "2024-01-01")
            else:
                dates = pd.bdate_range("2023-01-01", "2024-01-01")
            ts_index = np.array([int(d.timestamp()) for d in dates])
            return pd.DataFrame({"close": np.ones(len(dates))}, index=ts_index)

        pipe.fetch_stock.side_effect = fetch
        valid, excluded = validate_universe(
            ["OLD", "NEW"], "2020-01-01", pipe=pipe
        )
        assert "OLD" in valid
        assert "NEW" in excluded
