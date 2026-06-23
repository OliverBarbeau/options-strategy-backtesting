"""Shared test fixtures for tradelab tests.

All fixtures generate deterministic synthetic data so tests never
depend on external APIs.
"""

import os

import numpy as np
import pandas as pd
import pytest


def pytest_collection_modifyitems(config, items):
    """Skip @pytest.mark.network tests unless --network flag is passed."""
    if not config.getoption("--network", default=False):
        skip_network = pytest.mark.skip(reason="needs --network flag to run")
        for item in items:
            if "network" in item.keywords:
                item.add_marker(skip_network)


def pytest_addoption(parser):
    parser.addoption(
        "--network", action="store_true", default=False,
        help="run tests that require network access",
    )

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


@pytest.fixture
def stock_df() -> pd.DataFrame:
    """Synthetic daily stock data (~2 years, 504 rows).

    Simulates a stock starting at $100, trending upward with noise.
    Indexed by unix timestamp (daily spacing, 86400s apart).
    """
    np.random.seed(42)
    n = 504
    base_ts = 1_420_070_400  # 2015-01-01
    timestamps = [base_ts + i * 86400 for i in range(n)]

    # Random walk with slight upward drift
    returns = np.random.normal(0.0003, 0.012, n)
    prices = 100 * np.cumprod(1 + returns)

    df = pd.DataFrame(
        {
            "o": prices * (1 - np.random.uniform(0, 0.005, n)),
            "h": prices * (1 + np.random.uniform(0, 0.01, n)),
            "l": prices * (1 - np.random.uniform(0, 0.01, n)),
            "c": prices,
            "v": np.random.randint(1_000_000, 10_000_000, n),
        },
        index=timestamps,
    )
    df.index.name = "timestamp"
    return df


@pytest.fixture
def crypto_df() -> pd.DataFrame:
    """Synthetic hourly crypto data (~30 days, 720 rows).

    Simulates BTC-like prices starting at $30,000 with higher volatility.
    """
    np.random.seed(123)
    n = 720
    base_ts = 1_640_995_200  # 2022-01-01
    timestamps = [base_ts + i * 3600 for i in range(n)]

    returns = np.random.normal(0.0001, 0.008, n)
    prices = 30_000 * np.cumprod(1 + returns)

    df = pd.DataFrame(
        {
            "o": prices * (1 - np.random.uniform(0, 0.003, n)),
            "h": prices * (1 + np.random.uniform(0, 0.005, n)),
            "l": prices * (1 - np.random.uniform(0, 0.005, n)),
            "c": prices,
            "v": np.random.uniform(10, 500, n),
        },
        index=timestamps,
    )
    df.index.name = "timestamp"
    return df


@pytest.fixture
def declining_stock_df() -> pd.DataFrame:
    """Stock data with a clear downtrend -- useful for testing loss scenarios."""
    np.random.seed(99)
    n = 300
    base_ts = 1_420_070_400
    timestamps = [base_ts + i * 86400 for i in range(n)]

    returns = np.random.normal(-0.002, 0.01, n)
    prices = 200 * np.cumprod(1 + returns)

    df = pd.DataFrame({"c": prices}, index=timestamps)
    df.index.name = "timestamp"
    return df


@pytest.fixture
def flat_stock_df() -> pd.DataFrame:
    """Stock data that stays flat -- useful for testing fee drag."""
    n = 200
    base_ts = 1_420_070_400
    timestamps = [base_ts + i * 86400 for i in range(n)]
    prices = [100.0] * n

    df = pd.DataFrame({"c": prices}, index=timestamps)
    df.index.name = "timestamp"
    return df


@pytest.fixture
def sample_csv_path(stock_df: pd.DataFrame, tmp_path) -> str:
    """Write the stock_df fixture to a temp CSV and return the path."""
    path = tmp_path / "sample_stock.csv"
    out = stock_df.copy()
    out.index.name = "time"
    out.to_csv(path)
    return str(path)
