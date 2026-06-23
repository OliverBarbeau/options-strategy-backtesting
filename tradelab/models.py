"""Core trading models: positions and option spreads."""

from __future__ import annotations

import datetime


class Position:
    """A leveraged long or short position.

    Ported from money_printer.ipynb's position class.
    """

    LONG = 1
    SHORT = 0

    def __init__(self, ratio: float, margin: float, leverage: float, side: int):
        """
        Args:
            ratio: Entry price.
            margin: Available margin used for the position.
            leverage: Leverage multiplier.
            side: Position.LONG (1) or Position.SHORT (0).
        """
        self.side = side
        self.ratio = ratio
        self.cost = margin * leverage
        self.volume = self.cost / self.ratio
        self.leverage = leverage

    def profit_loss(self, current_price: float) -> float:
        """Unrealized P&L at *current_price*."""
        if self.side == self.LONG:
            return (current_price - self.ratio) * self.volume
        return (self.ratio - current_price) * self.volume

    def __repr__(self) -> str:
        side_label = "LONG" if self.side == self.LONG else "SHORT"
        return (
            f"Position({side_label}, entry={self.ratio:.2f}, "
            f"cost={self.cost:.2f}, vol={self.volume:.4f}, lev={self.leverage}x)"
        )


class PutCreditSpread:
    """Models a single put credit spread contract.

    Ported from time_period_gains_calc.ipynb.
    """

    def __init__(
        self,
        underlying_price: float,
        strike_price: float,
        collateral: float,
        open_date: int,
        expiry: int,
    ):
        """
        Args:
            underlying_price: Price of the underlying at open.
            strike_price: Short put strike price.
            collateral: Capital allocated to this spread.
            open_date: Unix timestamp of open.
            expiry: Unix timestamp of expiry.
        """
        self.underlying_price = underlying_price
        self.strike_price = int(strike_price)
        self.collateral = collateral
        self.open_date = open_date
        self.expiry = expiry

    @property
    def open_datetime(self) -> datetime.datetime:
        return datetime.datetime.fromtimestamp(self.open_date)

    @property
    def expiry_datetime(self) -> datetime.datetime:
        return datetime.datetime.fromtimestamp(self.expiry)

    def is_expired(self, current_timestamp: int) -> bool:
        return current_timestamp >= self.expiry

    def evaluate(
        self,
        current_price: float,
        credit_ratio: float = 0.20,
        loss_ratio: float = 0.95,
    ) -> tuple[float, bool]:
        """Evaluate the spread at expiry/close.

        Args:
            current_price: Current underlying price.
            credit_ratio: Fraction of collateral received as premium on win.
            loss_ratio: Fraction of collateral lost on assignment.

        Returns:
            (cash_return, is_winner) tuple.
        """
        if current_price <= self.strike_price:
            # Assigned -- loss
            return self.collateral * (1 - loss_ratio), False
        # Expired worthless -- profit
        return self.collateral * (1 + credit_ratio), True

    def __repr__(self) -> str:
        return (
            f"PutCreditSpread(strike={self.strike_price}, "
            f"collateral={self.collateral:.2f}, "
            f"open={self.open_datetime:%Y-%m-%d}, "
            f"expiry={self.expiry_datetime:%Y-%m-%d})"
        )
