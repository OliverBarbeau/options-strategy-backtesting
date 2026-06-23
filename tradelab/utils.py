"""Financial math utilities, fee schedules, and leverage helpers."""

import math

import pandas as pd


# ---------------------------------------------------------------------------
# Financial math (from time_period_gains_calc.ipynb)
# ---------------------------------------------------------------------------

def reverse_interest(years: float, principal: float, final: float) -> float:
    """Compute the annualized rate of return.

    Solves: final = principal * (1 + r)^years  =>  r = (final/principal)^(1/years) - 1
    """
    if years <= 0 or principal <= 0:
        return 0.0
    return (final / principal) ** (1 / years) - 1


def compound_interest(balance: float, rate: float, periods: int) -> float:
    """Apply *rate* compounding over *periods*."""
    for _ in range(periods):
        balance *= 1 + rate
    return balance


def days_to_trading_days(calendar_days: int, trading_days_per_year: int = 252) -> int:
    """Convert calendar days to approximate trading days."""
    return math.floor(calendar_days * (trading_days_per_year / 365))


# ---------------------------------------------------------------------------
# Fee schedule (from money_printer.ipynb -- Kraken-style tiered fees)
# ---------------------------------------------------------------------------

# (volume_threshold, rate_reduction) -- cumulative reductions
_FEE_TIERS: list[tuple[int, float]] = [
    (50_000, 0.0002),
    (100_000, 0.0002),
    (250_000, 0.0002),
    (500_000, 0.0002),
    (1_000_000, 0.0002),
    (2_500_000, 0.0002),
    (5_000_000, 0.0002),
    (10_000_000, 0.0002),
]


def calc_fee_rate(trade_volume: float, base_rate: float = 0.0026) -> float:
    """Return the taker fee rate for a given 30-day trade volume.

    Default tiers mirror Kraken's schedule.
    """
    rate = base_rate
    for threshold, reduction in _FEE_TIERS:
        if trade_volume > threshold:
            rate -= reduction
    return max(rate, 0.0)


# ---------------------------------------------------------------------------
# Dynamic leverage (from money_printer.ipynb)
# ---------------------------------------------------------------------------

def get_leverage(
    prediction: list[float],
    confidence: float = 0.50,
    base_leverage: int = 2,
    max_leverage: int = 5,
) -> int:
    """Scale leverage based on model prediction confidence.

    The further the prediction probabilities diverge, the higher the leverage
    (up to *max_leverage*).

    Args:
        prediction: Two-element list [prob_sell, prob_buy].
        confidence: Minimum confidence threshold.
        base_leverage: Starting leverage.
        max_leverage: Maximum allowed leverage.
    """
    diff = abs(prediction[0] - prediction[1])
    lev = base_leverage
    margin = 1 - confidence
    if margin <= 0:
        return base_leverage

    section = margin / (max_leverage - base_leverage)
    while lev < max_leverage and diff > section * (lev - base_leverage + 1):
        lev += 1

    return lev


# ---------------------------------------------------------------------------
# CSV loader (moved from data.py)
# ---------------------------------------------------------------------------

def load_csv(filepath: str, index_col: str = "time") -> pd.DataFrame:
    """Load price data from a local CSV file."""
    df = pd.read_csv(filepath)
    if index_col in df.columns:
        df.set_index(index_col, inplace=True)
    return df
