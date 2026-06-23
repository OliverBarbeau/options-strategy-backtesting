"""Black-Scholes options pricing from underlying OHLCV data.

Approximates option premiums and Greeks using:
- Underlying price from OHLCV
- Historical volatility as IV proxy
- Standard Black-Scholes model (European, no dividends)

This lets us backtest options strategies (credit spreads, iron condors, etc.)
using only the underlying's price history.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


# ------------------------------------------------------------------
# Core Black-Scholes
# ------------------------------------------------------------------


def bs_put_price(
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float,
    sigma: float | np.ndarray,
) -> float | np.ndarray:
    """Black-Scholes European put price.

    Args:
        S: Underlying price.
        K: Strike price.
        T: Time to expiry in years.
        r: Risk-free interest rate (annualized, e.g. 0.05 = 5%).
        sigma: Volatility (annualized, e.g. 0.20 = 20%).
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_call_price(
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float,
    sigma: float | np.ndarray,
) -> float | np.ndarray:
    """Black-Scholes European call price."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


# ------------------------------------------------------------------
# Greeks
# ------------------------------------------------------------------


def bs_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "put",
) -> dict[str, float]:
    """Compute Black-Scholes Greeks.

    Returns dict with: delta, gamma, theta, vega, rho.
    """
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    gamma = norm.pdf(d1) / (S * sigma * sqrt_T)
    vega = S * norm.pdf(d1) * sqrt_T / 100  # per 1% vol change

    if option_type == "call":
        delta = norm.cdf(d1)
        theta = (
            -S * norm.pdf(d1) * sigma / (2 * sqrt_T)
            - r * K * np.exp(-r * T) * norm.cdf(d2)
        ) / 365
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        delta = norm.cdf(d1) - 1
        theta = (
            -S * norm.pdf(d1) * sigma / (2 * sqrt_T)
            + r * K * np.exp(-r * T) * norm.cdf(-d2)
        ) / 365
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

    return {
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "vega": vega,
        "rho": rho,
    }


# ------------------------------------------------------------------
# Volatility estimation
# ------------------------------------------------------------------


def historical_volatility(
    prices: pd.Series, window: int = 30, trading_days: int = 252
) -> pd.Series:
    """Compute annualized historical volatility from a price series.

    Uses log returns with a rolling window.
    """
    log_returns = np.log(prices / prices.shift(1))
    # Shift returns by 1 so vol[i] uses only returns up to day i-1,
    # avoiding look-ahead bias (day i's own price move is excluded).
    return log_returns.shift(1).rolling(window).std() * np.sqrt(trading_days)


def ewm_volatility(
    prices: pd.Series, span: int = 30, trading_days: int = 252
) -> pd.Series:
    """Exponentially-weighted historical volatility.

    Gives more weight to recent price action.
    """
    log_returns = np.log(prices / prices.shift(1))
    return log_returns.shift(1).ewm(span=span).std() * np.sqrt(trading_days)


# ------------------------------------------------------------------
# Spread pricing
# ------------------------------------------------------------------


def put_credit_spread_price(
    S: float,
    K_short: float,
    K_long: float,
    T: float,
    r: float,
    sigma: float,
) -> dict[str, float]:
    """Price a put credit spread (sell higher strike, buy lower strike).

    Args:
        S: Underlying price.
        K_short: Short put strike (higher, the one you sell).
        K_long: Long put strike (lower, the one you buy).
        T: Time to expiry in years.
        r: Risk-free rate.
        sigma: Volatility.

    Returns dict with:
        short_premium: Premium received for selling the short put.
        long_premium: Premium paid for buying the long put.
        net_credit: Net credit received (short - long).
        max_loss: Maximum loss (spread width - net credit).
        spread_width: Distance between strikes.
        credit_potential: net_credit / max_loss as a percentage.
    """
    short_premium = bs_put_price(S, K_short, T, r, sigma)
    long_premium = bs_put_price(S, K_long, T, r, sigma)
    net_credit = short_premium - long_premium
    spread_width = (K_short - K_long) * 100  # per contract (100 shares)
    net_credit_dollar = net_credit * 100
    max_loss = spread_width - net_credit_dollar

    return {
        "short_premium": float(short_premium),
        "long_premium": float(long_premium),
        "net_credit": float(net_credit),
        "net_credit_dollar": float(net_credit_dollar),
        "max_loss": float(max_loss),
        "spread_width": float(spread_width),
        "credit_potential": float(net_credit_dollar / max_loss) if max_loss > 0 else 0.0,
    }


def iron_condor_price(
    S: float,
    put_K_short: float,
    put_K_long: float,
    call_K_short: float,
    call_K_long: float,
    T: float,
    r: float,
    sigma: float,
) -> dict[str, float]:
    """Price an iron condor (sell OTM put spread + sell OTM call spread).

    Args:
        S: Underlying price.
        put_K_short: Short put strike (below S).
        put_K_long: Long put strike (below put_K_short).
        call_K_short: Short call strike (above S).
        call_K_long: Long call strike (above call_K_short).
        T: Time to expiry in years.
        r: Risk-free rate.
        sigma: Volatility.

    Returns dict with:
        put_credit, call_credit, total_credit, total_credit_dollar,
        max_loss (wider leg), spread_width, credit_potential.
    """
    put_credit = bs_put_price(S, put_K_short, T, r, sigma) - bs_put_price(S, put_K_long, T, r, sigma)
    call_credit = bs_call_price(S, call_K_short, T, r, sigma) - bs_call_price(S, call_K_long, T, r, sigma)
    total_credit = put_credit + call_credit
    total_credit_dollar = total_credit * 100

    put_width = (put_K_short - put_K_long) * 100
    call_width = (call_K_long - call_K_short) * 100
    max_width = max(put_width, call_width)
    max_loss = max_width - total_credit_dollar

    return {
        "put_credit": float(put_credit),
        "call_credit": float(call_credit),
        "total_credit": float(total_credit),
        "total_credit_dollar": float(total_credit_dollar),
        "max_loss": float(max_loss),
        "spread_width": float(max_width),
        "credit_potential": float(total_credit_dollar / max_loss) if max_loss > 0 else 0.0,
    }


def calendar_spread_price(
    S: float,
    K: float,
    T_near: float,
    T_far: float,
    r: float,
    sigma: float,
    option_type: str = "put",
) -> dict[str, float]:
    """Price a calendar spread (sell near-term, buy far-term, same strike).

    Args:
        S: Underlying price.
        K: Strike price (same for both legs).
        T_near: Near-term time to expiry in years (sell this).
        T_far: Far-term time to expiry in years (buy this).
        r: Risk-free rate.
        sigma: Volatility.
        option_type: "put" or "call".

    Returns dict with:
        near_premium, far_premium, net_debit, max_profit_estimate.
    """
    if option_type == "put":
        near = bs_put_price(S, K, T_near, r, sigma)
        far = bs_put_price(S, K, T_far, r, sigma)
    else:
        near = bs_call_price(S, K, T_near, r, sigma)
        far = bs_call_price(S, K, T_far, r, sigma)

    net_debit = (far - near) * 100  # cost to enter
    # Max profit is hard to compute exactly (depends on vol at near expiry)
    # but roughly = far_value_at_near_expiry - net_debit
    # Estimate: assume stock stays at S, reprice far leg at T_far - T_near
    far_at_near_expiry = (
        bs_put_price(S, K, T_far - T_near, r, sigma)
        if option_type == "put"
        else bs_call_price(S, K, T_far - T_near, r, sigma)
    )
    max_profit_estimate = (far_at_near_expiry - near) * 100 - net_debit + near * 100

    return {
        "near_premium": float(near),
        "far_premium": float(far),
        "net_debit": float(net_debit),
        "max_profit_estimate": float(max_profit_estimate),
        "near_premium_dollar": float(near * 100),
        "far_premium_dollar": float(far * 100),
    }


def price_spread_series(
    df: pd.DataFrame,
    strike_buffer: float = 0.05,
    spread_width: float = 5.0,
    days_to_expiry: int = 30,
    r: float = 0.05,
    close_col: str = "close",
    vol_col: str | None = None,
    vol_window: int = 30,
) -> pd.DataFrame:
    """Price a put credit spread at each row of an OHLCV DataFrame.

    For each row, sets the short strike at (close * (1 - strike_buffer))
    and the long strike at (short_strike - spread_width).

    Args:
        df: OHLCV DataFrame.
        strike_buffer: How far OTM to place the short strike (fraction).
        spread_width: Dollar distance between strikes.
        days_to_expiry: Assumed DTE for each row.
        r: Risk-free rate.
        close_col: Close price column name.
        vol_col: Pre-computed volatility column. If None, computes
                 historical vol with vol_window.
        vol_window: Window for historical vol calculation.

    Returns:
        DataFrame with columns: short_strike, long_strike, net_credit,
        max_loss, credit_potential, sigma.
    """
    prices = df[close_col]

    if vol_col and vol_col in df.columns:
        sigma = df[vol_col]
    else:
        sigma = historical_volatility(prices, window=vol_window)

    T = days_to_expiry / 365.0

    short_strikes = prices * (1 - strike_buffer)
    long_strikes = short_strikes - spread_width

    short_premiums = bs_put_price(prices.values, short_strikes.values, T, r, sigma.values)
    long_premiums = bs_put_price(prices.values, long_strikes.values, T, r, sigma.values)

    net_credit = short_premiums - long_premiums
    net_credit_dollar = net_credit * 100
    spread_width_dollar = spread_width * 100
    max_loss = spread_width_dollar - net_credit_dollar

    result = pd.DataFrame(
        {
            "short_strike": short_strikes,
            "long_strike": long_strikes,
            "net_credit": net_credit,
            "net_credit_dollar": net_credit_dollar,
            "max_loss": max_loss,
            "credit_potential": net_credit_dollar / np.clip(max_loss, a_min=0.01, a_max=None),
            "sigma": sigma,
        },
        index=df.index,
    )
    return result.dropna()
