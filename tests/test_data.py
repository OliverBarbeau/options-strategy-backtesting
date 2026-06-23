"""Tests for tradelab.data (CSV loading only -- API tests need network)."""

import pandas as pd

from tradelab.data import load_csv


class TestLoadCsv:
    def test_loads_and_indexes(self, sample_csv_path):
        df = load_csv(sample_csv_path, index_col="time")
        assert isinstance(df, pd.DataFrame)
        assert df.index.name == "time"
        assert "c" in df.columns
        assert len(df) > 0

    def test_missing_index_col_keeps_default(self, sample_csv_path):
        df = load_csv(sample_csv_path, index_col="nonexistent")
        # Should still load, just without setting the index
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
