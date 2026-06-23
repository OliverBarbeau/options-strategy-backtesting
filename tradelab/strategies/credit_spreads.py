"""Rolling put credit spread strategy.

Ported from time_period_gains_calc.ipynb cells 6-9.

This strategy maintains a portfolio of overlapping put credit spreads,
opening new ones on a schedule and closing them at expiry.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field

import pandas as pd

from tradelab.models import PutCreditSpread
from tradelab.utils import reverse_interest


@dataclass
class SpreadResult:
    """Results from a rolling spread backtest."""

    initial_balance: float = 0.0
    final_balance: float = 0.0
    winners: int = 0
    losers: int = 0
    annualized_return: float = 0.0
    years: float = 0.0
    trade_log: list[dict] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        total = self.winners + self.losers
        return self.winners / total if total > 0 else 0.0

    def summary(self) -> str:
        return (
            f"Period:            {self.years:.1f} years\n"
            f"Initial balance:   ${self.initial_balance:,.2f}\n"
            f"Final balance:     ${self.final_balance:,.2f}\n"
            f"Win rate:          {self.win_rate:.1%} ({self.winners}W / {self.losers}L)\n"
            f"Annualized return: {self.annualized_return:.2%}"
        )


class RollingCreditSpreadStrategy:
    """Simulate rolling put credit spreads over historical price data.

    Opens *max_spreads* overlapping spreads, staggered evenly across the
    expiry window, and closes each at expiry based on whether the underlying
    breached the strike.

    Args:
        max_spreads: Number of concurrent spreads to maintain.
        offset_days: Expiry window in trading days.
        buffer: How far below current price to set the strike (fraction).
        credit_ratio: Premium received as fraction of collateral on a win.
        loss_ratio: Fraction of collateral lost on assignment.
    """

    def __init__(
        self,
        max_spreads: int = 4,
        offset_days: int = 274,
        buffer: float = 0.019,
        credit_ratio: float = 0.20,
        loss_ratio: float = 0.95,
    ):
        self.max_spreads = max_spreads
        self.offset_days = offset_days
        self.buffer = buffer
        self.credit_ratio = credit_ratio
        self.loss_ratio = loss_ratio

    def run(
        self,
        df: pd.DataFrame,
        initial_balance: float = 1_000,
        close_col: str = "c",
    ) -> SpreadResult:
        """Run the strategy over a price DataFrame.

        Args:
            df: DataFrame indexed by unix timestamp with a close price column.
            initial_balance: Starting cash.
            close_col: Column name for close prices.
        """
        cash = initial_balance
        spreads: list[PutCreditSpread] = []
        trading_days_between = max(1, self.offset_days // self.max_spreads)
        days_since_last_trade = trading_days_between  # open one immediately

        winners = 0
        losers = 0
        trade_log: list[dict] = []

        n = df.shape[0]
        for i in range(n - self.offset_days):
            price: float = df[close_col].iloc[i]
            date: int = df.index[i]
            portfolio_val = sum(s.collateral for s in spreads) + cash

            # --- open new spread if schedule allows ---
            if (
                days_since_last_trade >= trading_days_between
                and len(spreads) < self.max_spreads
            ):
                allocation = portfolio_val / self.max_spreads
                allocation = min(allocation, cash)
                if allocation > 0:
                    strike = price * (1 - self.buffer)
                    expiry = date + self.offset_days * 86400
                    spread = PutCreditSpread(
                        underlying_price=price,
                        strike_price=strike,
                        collateral=allocation,
                        open_date=date,
                        expiry=expiry,
                    )
                    spreads.append(spread)
                    cash -= allocation
                    days_since_last_trade = 0
            else:
                days_since_last_trade += 1

            # --- close oldest spread if expired ---
            if len(spreads) == self.max_spreads and spreads[0].is_expired(date):
                oldest = spreads.pop(0)
                cash_return, is_winner = oldest.evaluate(
                    price, self.credit_ratio, self.loss_ratio
                )
                cash += cash_return

                if is_winner:
                    winners += 1
                else:
                    losers += 1

                trade_log.append(
                    {
                        "open_date": datetime.datetime.fromtimestamp(oldest.open_date),
                        "close_date": datetime.datetime.fromtimestamp(date),
                        "collateral": oldest.collateral,
                        "strike": oldest.strike_price,
                        "entry_price": oldest.underlying_price,
                        "exit_price": price,
                        "cash_return": cash_return,
                        "winner": is_winner,
                    }
                )

        # Final portfolio value
        final_balance = sum(s.collateral for s in spreads) + cash

        # Annualized return
        seconds = int(df.index[-1] - df.index[0])
        years = datetime.timedelta(seconds=seconds).days / 365
        annual_return = reverse_interest(years, initial_balance, final_balance)

        return SpreadResult(
            initial_balance=initial_balance,
            final_balance=final_balance,
            winners=winners,
            losers=losers,
            annualized_return=annual_return,
            years=years,
            trade_log=trade_log,
        )
