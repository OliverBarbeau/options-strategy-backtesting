"""Generic backtesting engine for leveraged positions.

Ported from money_printer.ipynb's backtest() function, generalized to work
with any signal source (not just the LSTM model).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from tradelab.models import Position
from tradelab.utils import calc_fee_rate, get_leverage


@dataclass
class BacktestResult:
    """Container for backtest output metrics."""

    start_date: object = None
    end_date: object = None
    start_price: float = 0.0
    end_price: float = 0.0
    trade_count: int = 0
    market_return_pct: float = 0.0
    strategy_return_pct: float = 0.0
    strategy_return_nofees_pct: float = 0.0
    alpha_pct: float = 0.0
    alpha_nofees_pct: float = 0.0
    final_equity: float = 0.0
    final_equity_nofees: float = 0.0
    total_fees: float = 0.0
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)

    def summary(self) -> str:
        return (
            f"Period:       {self.start_date} -> {self.end_date}\n"
            f"Trades:       {self.trade_count}\n"
            f"Market:       {self.market_return_pct:+.2f}%\n"
            f"Strategy:     {self.strategy_return_pct:+.2f}%\n"
            f"Strategy(nf): {self.strategy_return_nofees_pct:+.2f}%\n"
            f"Alpha:        {self.alpha_pct:+.2f}%\n"
            f"Alpha(nf):    {self.alpha_nofees_pct:+.2f}%\n"
            f"Final equity: ${self.final_equity:,.2f}\n"
            f"Fees paid:    ${self.total_fees:,.2f}"
        )


class Backtester:
    """Event-driven backtester that steps through price data and delegates
    trade decisions to a signal function.

    The signal function receives (index, row, current_position) and must
    return one of:
        Position.LONG  (1) -- go/stay long
        Position.SHORT (0) -- go/stay short
        None               -- no position / close existing

    Example usage::

        def my_signal(index, row, current_pos):
            if row["close"] > row["sma_200"]:
                return Position.LONG
            return None

        bt = Backtester(capital=10_000, leverage=2)
        result = bt.run(df, price_col="close", signal_fn=my_signal)
        print(result.summary())
    """

    def __init__(
        self,
        capital: float = 10_000,
        leverage: int = 1,
        dynamic_leverage: bool = False,
        confidence: float = 0.50,
        base_fee_rate: float = 0.0026,
        allow_short: bool = True,
    ):
        self.capital = capital
        self.leverage = leverage
        self.dynamic_leverage = dynamic_leverage
        self.confidence = confidence
        self.base_fee_rate = base_fee_rate
        self.allow_short = allow_short

    def run(
        self,
        df: pd.DataFrame,
        price_col: str = "c",
        signal_fn=None,
    ) -> BacktestResult:
        """Run the backtest over *df*.

        Args:
            df: DataFrame with at minimum a price column.
            price_col: Name of the close price column.
            signal_fn: Callable(index, row, current_position) -> int | None.
                       If None, a buy-and-hold baseline is run.
        """
        if signal_fn is None:
            signal_fn = lambda idx, row, pos: Position.LONG

        equity = self.capital
        equity_nf = self.capital  # no-fees tracker
        pos: Position | None = None
        pos_nf: Position | None = None
        fees = 0.0
        trade_count = 0
        trade_volume = 0.0

        start_price = None
        start_date = None
        trade_log: list[dict] = []

        for index, row in df.iterrows():
            price = row[price_col] if isinstance(row, pd.Series) else row
            if start_price is None:
                start_price = price
                start_date = index

            fee_rate = calc_fee_rate(trade_volume, self.base_fee_rate)
            desired_side = signal_fn(index, row, pos)

            # --- close existing position if signal changed ---
            if pos is not None and desired_side != pos.side:
                equity += pos.profit_loss(price)
                equity_nf += pos_nf.profit_loss(price)
                fee = pos.volume * price * fee_rate
                equity -= fee
                fees += fee
                trade_volume += pos.volume * price
                trade_count += 1
                pos = None
                pos_nf = None

            # --- open new position if we have a signal and no position ---
            if pos is None and desired_side is not None:
                if desired_side == Position.SHORT and not self.allow_short:
                    pass  # skip short if disabled
                else:
                    lev = self.leverage
                    if self.dynamic_leverage:
                        # Caller can attach prediction to row for dynamic lev
                        pred = getattr(row, "prediction", None)
                        if pred is not None:
                            lev = get_leverage(pred, self.confidence)
                    pos = Position(price, equity, lev, desired_side)
                    pos_nf = Position(price, equity_nf, lev, desired_side)
                    fee = pos.volume * price * fee_rate
                    equity -= fee
                    fees += fee
                    trade_volume += pos.cost

            # --- log equity curve ---
            market_value = price * self.capital / start_price
            trade_log.append(
                {
                    "timestamp": index,
                    "market": market_value,
                    "strategy": equity + (pos.profit_loss(price) if pos else 0),
                    "strategy_nf": equity_nf + (pos_nf.profit_loss(price) if pos_nf else 0),
                }
            )

        # --- final close if still holding ---
        end_price = df[price_col].iloc[-1]
        if pos is not None:
            equity += pos.profit_loss(end_price)
            equity_nf += pos_nf.profit_loss(end_price)
            fee = pos.volume * end_price * calc_fee_rate(trade_volume, self.base_fee_rate)
            equity -= fee
            fees += fee
            trade_count += 1

        trades_df = pd.DataFrame(trade_log)
        if not trades_df.empty:
            trades_df = trades_df.set_index("timestamp")

        market_ret = (end_price - start_price) / start_price * 100
        strat_ret = (equity - self.capital) / self.capital * 100
        strat_ret_nf = (equity_nf - self.capital) / self.capital * 100

        return BacktestResult(
            start_date=start_date,
            end_date=df.index[-1],
            start_price=start_price,
            end_price=end_price,
            trade_count=trade_count,
            market_return_pct=market_ret,
            strategy_return_pct=strat_ret,
            strategy_return_nofees_pct=strat_ret_nf,
            alpha_pct=strat_ret - market_ret,
            alpha_nofees_pct=strat_ret_nf - market_ret,
            final_equity=equity,
            final_equity_nofees=equity_nf,
            total_fees=fees,
            trades=trades_df,
        )
