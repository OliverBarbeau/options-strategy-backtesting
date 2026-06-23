"""Strike increment utilities for realistic option chain modeling.

Real option chains have strikes at specific increments, not arbitrary
fractional values. This module provides utilities to:

- Determine the likely strike increment for a given underlying
- Snap theoretical strikes to realistic chain values
- Generate a chain of realistic strikes around a center price

The increment conventions below are based on observed OCC chains for
liquid US equity options (2024-2026). High-volume tickers tend to have
denser chains ($1 increments) while lower-volume or higher-priced names
may use $2.50 or $5 increments.
"""

from __future__ import annotations

# Tickers with known $1 increments regardless of price (dense weekly chains)
DENSE_STRIKE_TICKERS = {
    "SPY", "QQQ", "IWM", "DIA",  # major ETFs
    "AAPL", "MSFT", "NVDA", "META", "GOOG", "GOOGL", "AMZN", "TSLA",  # mega-cap tech
    "AMD", "AVGO", "NFLX",
    "JPM", "BAC", "V", "MA",  # major financials
}

# Tickers/ETFs that typically have $0.50 increments when under $25
HALF_DOLLAR_TICKERS = {
    "XLF", "XLE", "KRE", "HYG",
}


def strike_increment(ticker: str, price: float) -> float:
    """Return the typical strike increment for a ticker at a given price.

    Rules:
    - Dense tickers (SPY, AAPL, etc.): $1 always
    - Under $25: $0.50
    - $25-$200: $1
    - $200-$500: $2.50
    - $500+: $5
    """
    if ticker.upper() in DENSE_STRIKE_TICKERS:
        return 1.0

    if price < 25:
        return 0.50
    elif price < 200:
        return 1.0
    elif price < 500:
        return 2.50
    else:
        return 5.0


def snap_to_increment(strike: float, increment: float, mode: str = "nearest") -> float:
    """Snap a theoretical strike to the nearest real increment.

    Args:
        strike: Theoretical strike price.
        increment: The strike increment ($0.50, $1, $2.50, $5).
        mode: "nearest" (round), "down" (floor), or "up" (ceil).

    Returns:
        Snapped strike price.
    """
    if increment <= 0:
        return strike
    if mode == "down":
        import math
        return math.floor(strike / increment) * increment
    elif mode == "up":
        import math
        return math.ceil(strike / increment) * increment
    else:  # nearest
        return round(strike / increment) * increment


def snap_put_credit_spread(
    ticker: str,
    underlying_price: float,
    target_buffer: float = 0.10,
    target_spread_dollars: float | None = None,
    target_spread_pct: float = 0.02,
) -> tuple[float, float]:
    """Find realistic short/long strikes for a put credit spread.

    The short strike is snapped DOWN (further OTM, more conservative) to
    ensure the actual buffer is at least the target. The long strike is
    snapped DOWN from there to maintain the spread width.

    Args:
        ticker: Underlying symbol.
        underlying_price: Current price.
        target_buffer: Target distance below price (e.g., 0.10 = 10% OTM).
        target_spread_dollars: Explicit spread width in dollars. If None,
            computed from target_spread_pct * underlying_price.
        target_spread_pct: Spread width as fraction of underlying (if dollars not given).

    Returns:
        (short_strike, long_strike) at realistic chain increments.
    """
    increment = strike_increment(ticker, underlying_price)

    # Short strike: target is price * (1 - buffer), snap down for safety
    target_short = underlying_price * (1 - target_buffer)
    short_strike = snap_to_increment(target_short, increment, mode="down")

    # Long strike: short - spread_width, snap down
    if target_spread_dollars is None:
        target_spread_dollars = underlying_price * target_spread_pct

    # Round spread width to nearest increment too (minimum one increment)
    spread_width = max(increment, round(target_spread_dollars / increment) * increment)
    long_strike = short_strike - spread_width

    return short_strike, long_strike


def effective_buffer(underlying_price: float, short_strike: float) -> float:
    """Compute the actual buffer percentage from a real strike.

    When we snap to real strikes, the effective buffer differs from target.
    """
    if underlying_price <= 0:
        return 0.0
    return (underlying_price - short_strike) / underlying_price
