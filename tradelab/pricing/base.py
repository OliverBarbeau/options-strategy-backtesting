"""Abstract base class for options pricing providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime


class PricingError(Exception):
    """Raised when a provider cannot price a given request."""
    pass


@dataclass
class OptionQuote:
    """A single option contract quote.

    Depending on the provider, may include live bid/ask (real data) or only
    theoretical mid (B-S).
    """
    ticker: str
    strike: float
    expiry: str              # YYYY-MM-DD
    put_call: str            # "P" or "C"

    # Pricing (all in dollars per share)
    bid: float = 0.0
    ask: float = 0.0
    mid: float = 0.0

    # Greeks (if available)
    delta: float | None = None
    gamma: float | None = None
    theta: float | None = None
    vega: float | None = None
    implied_vol: float | None = None

    # Market data (if available)
    volume: int | None = None
    open_interest: int | None = None

    # Metadata
    underlying_price: float = 0.0
    quote_date: str = ""
    source: str = "unknown"   # "blackscholes", "thetadata", "polygon", "mock"

    @property
    def spread_pct(self) -> float:
        """Bid-ask spread as percentage of mid."""
        if self.mid <= 0:
            return 0.0
        return (self.ask - self.bid) / self.mid


@dataclass
class SpreadQuote:
    """A put credit spread quote from any provider.

    All dollar amounts are per-contract (already multiplied by 100).
    """
    ticker: str
    short_strike: float
    long_strike: float
    expiry: str
    quote_date: str

    # Credit received (short_bid - long_ask for conservative estimate)
    net_credit: float        # per contract (already * 100)
    net_credit_mid: float    # using mid prices per contract

    # Risk
    max_loss: float          # per contract
    spread_width: float      # (short_strike - long_strike) * 100
    credit_potential: float  # net_credit / max_loss

    # Underlying
    underlying_price: float
    dte: int

    # Individual leg quotes
    short_quote: OptionQuote
    long_quote: OptionQuote

    # Metadata
    source: str = "unknown"
    notes: str = ""

    @property
    def implied_vol_avg(self) -> float | None:
        """Average IV of the two legs, if available."""
        if self.short_quote.implied_vol is None or self.long_quote.implied_vol is None:
            return None
        return (self.short_quote.implied_vol + self.long_quote.implied_vol) / 2


class PricingProvider(ABC):
    """Abstract base for options pricing data providers.

    Subclasses must implement the core methods. Strategy code calls this
    interface without knowing whether data comes from B-S, Theta Data, or
    a mock.
    """

    name: str = "base"

    @abstractmethod
    def get_option_quote(
        self,
        ticker: str,
        strike: float,
        expiry: str,
        put_call: str,
        date: str,
        underlying_price: float | None = None,
    ) -> OptionQuote:
        """Get a quote for a single option contract on a specific date.

        Args:
            ticker: Underlying symbol.
            strike: Strike price.
            expiry: Expiration date (YYYY-MM-DD).
            put_call: "P" for put, "C" for call.
            date: The date of the quote (YYYY-MM-DD).
            underlying_price: Override the underlying price (provider may
                use this or look it up).

        Returns:
            OptionQuote with as much data as the provider has available.

        Raises:
            PricingError: If the provider cannot price this contract.
        """
        raise NotImplementedError

    @abstractmethod
    def get_spread_quote(
        self,
        ticker: str,
        short_strike: float,
        long_strike: float,
        expiry: str,
        date: str,
        underlying_price: float | None = None,
        put_call: str = "P",
    ) -> SpreadQuote:
        """Get a quote for a vertical spread on a specific date.

        Returns:
            SpreadQuote with combined credit, max loss, and both leg quotes.
        """
        raise NotImplementedError

    def find_spread_strikes(
        self,
        ticker: str,
        date: str,
        buffer: float = 0.10,
        spread_pct: float = 0.02,
        dte_target: int = 30,
        dte_tolerance: int = 5,
        underlying_price: float | None = None,
    ) -> SpreadQuote | None:
        """Find the best put credit spread matching the given parameters.

        Default implementation calls `get_spread_quote` with strikes
        computed from the underlying price. Providers with a real option
        chain can override to search the actual chain for available strikes.
        """
        if underlying_price is None:
            raise PricingError(
                "Default find_spread_strikes requires underlying_price. "
                "Override this method in providers with live chain data."
            )

        short_strike = underlying_price * (1 - buffer)
        long_strike = short_strike - underlying_price * spread_pct

        return self.get_spread_quote(
            ticker=ticker,
            short_strike=short_strike,
            long_strike=long_strike,
            expiry=self._compute_expiry(date, dte_target),
            date=date,
            underlying_price=underlying_price,
        )

    def _compute_expiry(self, from_date: str, dte: int) -> str:
        """Compute an expiration date dte calendar days from from_date."""
        dt = datetime.fromisoformat(from_date)
        from datetime import timedelta
        return (dt + timedelta(days=dte)).strftime("%Y-%m-%d")

    def supports_greeks(self) -> bool:
        """Return True if the provider supplies real Greeks (not B-S computed)."""
        return False

    def supports_historical(self) -> bool:
        """Return True if the provider has historical data."""
        return True

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
