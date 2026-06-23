"""Pricing providers for options data.

Abstract interface that lets strategies price options without caring about
the data source. Providers include:

- BlackScholesProvider: analytical pricing from underlying OHLCV (fast, free, approximate)
- ThetaDataProvider: real historical option chains from Theta Data REST API (accurate, paid)
- PolygonProvider: alternative real data source via Polygon.io (API key, no Java)
- MockProvider: deterministic synthetic data for testing

All providers implement the same `PricingProvider` interface, so strategies
can swap between them without changes.

Usage::

    from tradelab.pricing import BlackScholesProvider, ThetaDataProvider

    # Default: fast analytical pricing
    provider = BlackScholesProvider()

    # When Theta Terminal is running locally
    provider = ThetaDataProvider(cache_dir="data/theta_cache")

    # In strategy code:
    quote = provider.get_spread_quote(
        ticker="AAPL",
        date="2024-06-14",
        short_strike=170,
        long_strike=165,
        expiry="2024-07-19",
        underlying_price=195.50,
    )
    print(quote.net_credit, quote.max_loss)
"""

from tradelab.pricing.base import (
    PricingProvider,
    SpreadQuote,
    OptionQuote,
    PricingError,
)
from tradelab.pricing.blackscholes import BlackScholesProvider
from tradelab.pricing.mock import MockProvider
from tradelab.pricing.yfinance_provider import YFinanceProvider


def _lazy_theta():
    """Lazy import to avoid requiring httpx if not used."""
    from tradelab.pricing.thetadata import ThetaDataProvider, ThetaConfig
    return ThetaDataProvider, ThetaConfig


__all__ = [
    "PricingProvider",
    "SpreadQuote",
    "OptionQuote",
    "PricingError",
    "BlackScholesProvider",
    "MockProvider",
    "YFinanceProvider",
    "ThetaDataProvider",
    "ThetaConfig",
]


def __getattr__(name: str):
    """Lazy-import ThetaDataProvider only when accessed."""
    if name in ("ThetaDataProvider", "ThetaConfig"):
        from tradelab.pricing.thetadata import ThetaDataProvider, ThetaConfig
        return {"ThetaDataProvider": ThetaDataProvider, "ThetaConfig": ThetaConfig}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
