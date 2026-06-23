"""Hybrid put credit spread strategy: 30 DTE open, conditional close.

Opens spreads at 30 DTE to capture richer premium, then manages the exit:
- Hold to expiry when the trade is working (stock flat or up)
- Close early at the 14 DTE checkpoint if the stock has dropped past a threshold
- Emergency close at any point if the stock breaches a hard stop

This combines:
- Strategy B's advantage: 30-day premium (richer credit)
- Strategy C's advantage: capped losses (early close when threatened)
- Strategy A's advantage: full premium capture when the trade is working
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tradelab.options import (
    bs_put_price,
    put_credit_spread_price,
    historical_volatility,
)


@dataclass
class HybridResult:
    """Results from the hybrid spread backtest."""

    initial_balance: float = 0.0
    final_balance: float = 0.0
    total_trades: int = 0
    held_to_expiry: int = 0
    closed_at_checkpoint: int = 0
    emergency_closed: int = 0
    winners: int = 0
    losers: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    trade_log: list[dict] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        return self.winners / self.total_trades if self.total_trades > 0 else 0.0

    def summary(self) -> str:
        lines = [
            f"Trades:            {self.total_trades}",
            f"Win rate:          {self.win_rate:.1%} ({self.winners}W / {self.losers}L)",
            f"Total P/L:         ${self.total_pnl:+,.2f}",
            f"Avg P/L per trade: ${self.total_pnl / self.total_trades:+,.2f}" if self.total_trades else "",
            f"Max drawdown:      ${self.max_drawdown:+,.2f}",
            f"",
            f"Exit breakdown:",
            f"  Held to expiry:     {self.held_to_expiry}",
            f"  Closed at 14 DTE:   {self.closed_at_checkpoint}",
            f"  Emergency closed:   {self.emergency_closed}",
        ]
        return "\n".join(lines)


class HybridSpreadStrategy:
    """Open 30 DTE put credit spreads with conditional exit management.

    Args:
        buffer: Short strike distance below price (e.g. 0.10 = 10% OTM).
        spread_pct: Spread width as fraction of underlying (e.g. 0.02 = 2%).
        dte_open: Days to expiry at open (default 30).
        dte_checkpoint: Days to expiry for the management checkpoint (default 14).
        close_threshold: Underlying drop % that triggers early close at checkpoint
            (e.g. 0.03 = close if stock dropped 3%+ since open).
        emergency_stop: Underlying drop % that triggers immediate close at any
            point during the trade (e.g. 0.07 = close if stock dropped 7%+).
        risk_free_rate: Annualized risk-free rate for B-S pricing.
        vol_window: Rolling window for historical volatility estimation.
    """

    def __init__(
        self,
        buffer: float = 0.10,
        spread_pct: float = 0.02,
        dte_open: int = 30,
        dte_checkpoint: int = 14,
        close_threshold: float = 0.03,
        emergency_stop: float = 0.07,
        risk_free_rate: float = 0.05,
        vol_window: int = 30,
    ):
        self.buffer = buffer
        self.spread_pct = spread_pct
        self.dte_open = dte_open
        self.dte_checkpoint = dte_checkpoint
        self.close_threshold = close_threshold
        self.emergency_stop = emergency_stop
        self.r = risk_free_rate
        self.vol_window = vol_window

    def run(
        self,
        df: pd.DataFrame,
        close_col: str = "close",
    ) -> HybridResult:
        """Run the hybrid strategy over an OHLCV DataFrame.

        Args:
            df: DataFrame indexed by unix timestamp with OHLCV columns.
            close_col: Name of the close price column.

        Returns:
            HybridResult with trade log and aggregate metrics.
        """
        close = df[close_col].values
        timestamps = df.index.values
        vol = historical_volatility(df[close_col], window=self.vol_window)

        # Trading day offsets
        offset_open = max(1, int(self.dte_open * 21 / 30))
        offset_checkpoint = max(1, int((self.dte_open - self.dte_checkpoint) * 21 / 30))

        trade_log = []
        cumulative_pnl = 0.0
        peak_pnl = 0.0
        max_drawdown = 0.0

        i = self.vol_window
        while i < len(df) - offset_open:
            entry_price = close[i]
            entry_vol = vol.iloc[i]

            if np.isnan(entry_vol) or entry_vol <= 0:
                i += 1
                continue

            # Size the spread
            spread_width = entry_price * self.spread_pct
            short_strike = entry_price * (1 - self.buffer)
            long_strike = short_strike - spread_width

            if long_strike <= 0:
                i += offset_open
                continue

            # Price the spread at entry (30 DTE)
            T_open = self.dte_open / 365.0
            sp = put_credit_spread_price(
                entry_price, short_strike, long_strike, T_open, self.r, entry_vol
            )
            entry_credit = sp["net_credit_dollar"]
            max_loss = sp["max_loss"]

            if max_loss <= 0 or entry_credit <= 0:
                i += offset_open
                continue

            # --- PHASE 1: Monitor daily for emergency stop ---
            exit_type = "expiry"
            exit_day = offset_open
            pnl = 0.0

            emergency_triggered = False
            for day in range(1, offset_open + 1):
                if i + day >= len(close):
                    break
                current_price = close[i + day]
                price_change = (current_price - entry_price) / entry_price

                # Emergency stop check
                if price_change <= -self.emergency_stop:
                    # Close immediately -- price the spread at current mark
                    dte_remaining = max(1, self.dte_open - int(day * 30 / 21))
                    T_remaining = dte_remaining / 365.0
                    current_vol = vol.iloc[i + day] if (i + day) < len(vol) else entry_vol
                    if np.isnan(current_vol) or current_vol <= 0:
                        current_vol = entry_vol

                    close_cost = (
                        bs_put_price(current_price, short_strike, T_remaining, self.r, current_vol)
                        - bs_put_price(current_price, long_strike, T_remaining, self.r, current_vol)
                    ) * 100

                    pnl = entry_credit - close_cost
                    exit_type = "emergency"
                    exit_day = day
                    emergency_triggered = True
                    break

                # Checkpoint check (at the ~14 DTE mark)
                if day == offset_checkpoint and not emergency_triggered:
                    if price_change <= -self.close_threshold:
                        # Close at checkpoint
                        T_remaining = self.dte_checkpoint / 365.0
                        current_vol = vol.iloc[i + day] if (i + day) < len(vol) else entry_vol
                        if np.isnan(current_vol) or current_vol <= 0:
                            current_vol = entry_vol

                        close_cost = (
                            bs_put_price(current_price, short_strike, T_remaining, self.r, current_vol)
                            - bs_put_price(current_price, long_strike, T_remaining, self.r, current_vol)
                        ) * 100

                        pnl = entry_credit - close_cost
                        exit_type = "checkpoint"
                        exit_day = day
                        break

            # If we reached expiry without early close
            if exit_type == "expiry":
                expiry_idx = min(i + offset_open, len(close) - 1)
                expiry_price = close[expiry_idx]
                if expiry_price > short_strike:
                    pnl = entry_credit
                else:
                    pnl = -max_loss

            winner = pnl > 0
            exit_idx = min(i + exit_day, len(close) - 1)

            trade_log.append({
                "date": pd.Timestamp(timestamps[i], unit="s"),
                "exit_date": pd.Timestamp(timestamps[exit_idx], unit="s"),
                "entry_price": entry_price,
                "exit_price": close[exit_idx],
                "short_strike": short_strike,
                "long_strike": long_strike,
                "entry_credit": entry_credit,
                "max_loss": max_loss,
                "pnl": pnl,
                "winner": winner,
                "exit_type": exit_type,
                "exit_day": exit_day,
                "sigma": entry_vol,
                "price_change_pct": (close[exit_idx] - entry_price) / entry_price * 100,
            })

            cumulative_pnl += pnl
            peak_pnl = max(peak_pnl, cumulative_pnl)
            drawdown = cumulative_pnl - peak_pnl
            max_drawdown = min(max_drawdown, drawdown)

            # Advance past this trade
            i += exit_day if exit_day > 0 else offset_open

        # Build result
        trades_df = pd.DataFrame(trade_log)
        if trades_df.empty:
            return HybridResult()

        return HybridResult(
            total_trades=len(trades_df),
            held_to_expiry=int((trades_df["exit_type"] == "expiry").sum()),
            closed_at_checkpoint=int((trades_df["exit_type"] == "checkpoint").sum()),
            emergency_closed=int((trades_df["exit_type"] == "emergency").sum()),
            winners=int(trades_df["winner"].sum()),
            losers=int((~trades_df["winner"]).sum()),
            total_pnl=float(trades_df["pnl"].sum()),
            max_drawdown=float(max_drawdown),
            trade_log=trade_log,
        )
