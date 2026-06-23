"""Local disk cache for market data.

Stores DataFrames as parquet files, keyed by (source, symbol, resolution).
Fetch once, reuse forever. Supports incremental updates.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


class DataCache:
    """Parquet-backed local cache for OHLCV data.

    Directory structure:
        cache_dir/
            yfinance/
                AAPL_D.parquet
                SPY_D.parquet
            ccxt/
                kraken_BTC-USDT_1d.parquet
            finnhub/
                AAPL_D.parquet

    Usage::

        cache = DataCache("data/cache")
        df = cache.get("yfinance", "AAPL", "D")
        if df is None:
            df = fetch_from_api(...)
            cache.put("yfinance", "AAPL", "D", df)
    """

    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, source: str, symbol: str, resolution: str) -> Path:
        safe_symbol = symbol.replace("/", "-").replace(":", "-")
        source_dir = self.cache_dir / source
        source_dir.mkdir(parents=True, exist_ok=True)
        return source_dir / f"{safe_symbol}_{resolution}.parquet"

    def get(self, source: str, symbol: str, resolution: str) -> pd.DataFrame | None:
        """Load cached data. Returns None if not cached."""
        path = self._path(source, symbol, resolution)
        if not path.exists():
            return None
        return pd.read_parquet(path)

    def put(
        self,
        source: str,
        symbol: str,
        resolution: str,
        df: pd.DataFrame,
    ) -> Path:
        """Store data to cache. Returns the file path."""
        path = self._path(source, symbol, resolution)
        df.to_parquet(path)
        return path

    def update(
        self,
        source: str,
        symbol: str,
        resolution: str,
        new_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Append new data to existing cache, deduplicating by index."""
        existing = self.get(source, symbol, resolution)
        if existing is not None:
            combined = pd.concat([existing, new_data])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined = combined.sort_index()
        else:
            combined = new_data
        self.put(source, symbol, resolution, combined)
        return combined

    def has(self, source: str, symbol: str, resolution: str) -> bool:
        return self._path(source, symbol, resolution).exists()

    def list_cached(self, source: str | None = None) -> list[str]:
        """List all cached files, optionally filtered by source."""
        if source:
            source_dir = self.cache_dir / source
            if not source_dir.exists():
                return []
            return [f.name for f in source_dir.glob("*.parquet")]
        return [
            f"{d.name}/{f.name}"
            for d in self.cache_dir.iterdir()
            if d.is_dir()
            for f in d.glob("*.parquet")
        ]

    def clear(self, source: str | None = None) -> int:
        """Delete cached files. Returns count of files removed."""
        count = 0
        if source:
            source_dir = self.cache_dir / source
            if source_dir.exists():
                for f in source_dir.glob("*.parquet"):
                    f.unlink()
                    count += 1
        else:
            for d in self.cache_dir.iterdir():
                if d.is_dir():
                    for f in d.glob("*.parquet"):
                        f.unlink()
                        count += 1
        return count
