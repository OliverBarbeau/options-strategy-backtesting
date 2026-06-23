"""Aggressive pullback strategy for risk-tolerant individual accounts.

Builds on the core theta capture finding (30 DTE open, 14 DTE close) with
aggressive risk levers:
- Tighter buffer (7% vs 10%) for richer premiums
- Wider spreads (3% vs 2%) for more credit per trade
- Position stacking on deep pullbacks (5%+): 2 positions per ticker
- Streak bonus: tighten buffer to 6% after 2+ consecutive winners
- Higher max concurrent positions (8 vs 6)

Designed for accounts that can tolerate -15%+ drawdowns in exchange for
higher expected CAGR.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tradelab.options import (
    bs_put_price,
    put_credit_spread_price,
    historical_volatility,
)


@dataclass
class AggressiveResult:
    total_trades: int = 0
    winners: int = 0
    losers: int = 0
    total_pnl: float = 0.0
    max_drawdown_pct: float = 0.0
    stacked_trades: int = 0
    streak_trades: int = 0
    trade_log: list[dict] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        return self.winners / self.total_trades if self.total_trades > 0 else 0.0

    def summary(self) -> str:
        return (
            f"Trades:     {self.total_trades}\n"
            f"Win rate:   {self.win_rate:.1%} ({self.winners}W / {self.losers}L)\n"
            f"Total P/L:  ${self.total_pnl:+,.2f}\n"
            f"Max DD:     {self.max_drawdown_pct:.1%}\n"
            f"Stacked:    {self.stacked_trades} trades ({self.stacked_trades/self.total_trades:.0%} of total)"
            if self.total_trades > 0 else "No trades"
        )


class AggressivePullbackStrategy:
    """Enter put credit spreads on pullbacks with aggressive sizing.

    Args:
        buffer: Short strike OTM distance (default 0.07 = 7%).
        spread_pct: Spread width as % of underlying (default 0.03 = 3%).
        pullback_threshold: Min drawdown from recent high (default 0.03 = 3%).
        deep_pullback: Threshold for stacking a 2nd position (default 0.05 = 5%).
        lookback: Days to look back for the recent high (default 20).
        dte_open: Days to expiry at entry (default 30).
        dte_close: Remaining DTE when we close (default 14).
        streak_bonus_threshold: Consecutive wins to trigger tighter buffer (default 2).
        streak_buffer: Buffer used during a win streak (default 0.06 = 6%).
        risk_free_rate: Annualized risk-free rate (default 0.05).
        vol_window: Rolling window for historical vol (default 30).
        commission_per_contract: Per-leg commission (default 0.65).
        slippage_pct: Credit lost to bid-ask (default 0.02).
    """

    def __init__(
        self,
        buffer: float = 0.07,
        spread_pct: float = 0.03,
        pullback_threshold: float = 0.03,
        deep_pullback: float = 0.05,
        lookback: int = 20,
        dte_open: int = 30,
        dte_close: int = 14,
        streak_bonus_threshold: int = 2,
        streak_buffer: float = 0.06,
        risk_free_rate: float = 0.05,
        vol_window: int = 30,
        commission_per_contract: float = 0.65,
        slippage_pct: float = 0.02,
    ):
        self.buffer = buffer
        self.spread_pct = spread_pct
        self.pullback_threshold = pullback_threshold
        self.deep_pullback = deep_pullback
        self.lookback = lookback
        self.dte_open = dte_open
        self.dte_close = dte_close
        self.streak_bonus_threshold = streak_bonus_threshold
        self.streak_buffer = streak_buffer
        self.r = risk_free_rate
        self.vol_window = vol_window
        self.commission = commission_per_contract
        self.slippage_pct = slippage_pct

    def qualifies(self, df: pd.DataFrame, idx: int, close_col: str = "close") -> tuple[bool, float]:
        """Check if the ticker qualifies for entry at this index.

        Returns (qualifies, drawdown_pct) where drawdown_pct is negative.
        """
        if idx < self.lookback:
            return False, 0.0
        prices = df[close_col].iloc[idx - self.lookback : idx + 1]
        recent_high = prices.max()
        current = prices.iloc[-1]
        drawdown = (current - recent_high) / recent_high
        return drawdown <= -self.pullback_threshold, drawdown

    def _execute_trade(
        self,
        close: np.ndarray,
        vol: pd.Series,
        i: int,
        buffer: float,
        max_contracts: int,
        offset_close: int,
    ) -> dict | None:
        """Price and execute a single trade, returning trade log entry or None."""
        price = close[i]
        vol_val = vol.iloc[i]
        if np.isnan(vol_val) or vol_val <= 0:
            return None

        sw = price * self.spread_pct
        sk = price * (1 - buffer)
        lk = sk - sw
        if lk <= 0:
            return None

        sp = put_credit_spread_price(price, sk, lk, self.dte_open / 365, self.r, vol_val)
        credit = sp["net_credit_dollar"]
        max_loss = sp["max_loss"]
        if credit <= 0 or max_loss <= 0:
            return None

        open_comm = self.commission * 2 * max_contracts
        slippage = credit * max_contracts * self.slippage_pct
        net_credit = credit * max_contracts - slippage

        close_idx = min(i + offset_close, len(close) - 1)
        exit_price = close[close_idx]
        exit_vol = vol.iloc[close_idx] if close_idx < len(vol) else vol_val
        if np.isnan(exit_vol) or exit_vol <= 0:
            exit_vol = vol_val

        close_cost = (
            bs_put_price(exit_price, sk, self.dte_close / 365, self.r, exit_vol)
            - bs_put_price(exit_price, lk, self.dte_close / 365, self.r, exit_vol)
        ) * 100 * max_contracts
        close_comm = self.commission * 2 * max_contracts

        pnl = net_credit - close_cost - open_comm - close_comm
        winner = pnl > 0

        return {
            "entry_price": price,
            "exit_price": exit_price,
            "close_idx": close_idx,
            "buffer_used": buffer,
            "short_strike": sk,
            "long_strike": lk,
            "sigma": vol_val,
            "contracts": max_contracts,
            "credit": net_credit,
            "pnl": pnl,
            "winner": winner,
        }

    def run(
        self,
        df: pd.DataFrame,
        close_col: str = "close",
        max_contracts: int = 10,
    ) -> AggressiveResult:
        """Run the strategy on a single ticker's OHLCV DataFrame."""
        close = df[close_col].values
        timestamps = df.index.values
        vol = historical_volatility(df[close_col], window=self.vol_window)

        offset_open = max(1, int(self.dte_open * 21 / 30))
        offset_close = max(1, int((self.dte_open - self.dte_close) * 21 / 30))

        trade_log = []
        cum_pnl = 0.0
        peak = 0.0
        max_dd = 0.0
        consecutive_wins = 0
        stacked_count = 0
        streak_count = 0

        i = max(self.vol_window, self.lookback)
        while i < len(df) - offset_open:
            qualifies, drawdown = self.qualifies(df, i, close_col)

            if not qualifies:
                i += 1
                continue

            # Determine buffer: use streak bonus if on a hot streak
            is_streak = consecutive_wins >= self.streak_bonus_threshold
            buffer = self.streak_buffer if is_streak else self.buffer

            # Execute primary trade
            trade = self._execute_trade(close, vol, i, buffer, max_contracts, offset_close)
            if trade is None:
                i += offset_open
                continue

            is_deep = drawdown <= -self.deep_pullback
            trade["date"] = pd.Timestamp(timestamps[i], unit="s")
            trade["exit_date"] = pd.Timestamp(timestamps[trade["close_idx"]], unit="s")
            trade["pullback_pct"] = drawdown * 100
            trade["stacked"] = False
            trade["streak_entry"] = is_streak
            del trade["close_idx"]
            trade_log.append(trade)

            if is_streak:
                streak_count += 1

            # Track streak
            if trade["winner"]:
                consecutive_wins += 1
            else:
                consecutive_wins = 0

            cum_pnl += trade["pnl"]
            peak = max(peak, cum_pnl)
            max_dd = min(max_dd, cum_pnl - peak)

            # Stack a 2nd position on deep pullbacks
            if is_deep:
                stack_trade = self._execute_trade(
                    close, vol, i, self.buffer, max_contracts, offset_close
                )
                if stack_trade is not None:
                    stack_trade["date"] = pd.Timestamp(timestamps[i], unit="s")
                    stack_trade["exit_date"] = pd.Timestamp(
                        timestamps[stack_trade["close_idx"]], unit="s"
                    )
                    stack_trade["pullback_pct"] = drawdown * 100
                    stack_trade["stacked"] = True
                    stack_trade["streak_entry"] = False
                    del stack_trade["close_idx"]
                    trade_log.append(stack_trade)

                    if stack_trade["winner"]:
                        consecutive_wins += 1
                    else:
                        consecutive_wins = 0

                    cum_pnl += stack_trade["pnl"]
                    peak = max(peak, cum_pnl)
                    max_dd = min(max_dd, cum_pnl - peak)
                    stacked_count += 1

            i += offset_close

        return AggressiveResult(
            total_trades=len(trade_log),
            winners=sum(1 for t in trade_log if t["winner"]),
            losers=sum(1 for t in trade_log if not t["winner"]),
            total_pnl=sum(t["pnl"] for t in trade_log),
            max_drawdown_pct=max_dd / peak if peak > 0 else 0,
            stacked_trades=stacked_count,
            streak_trades=streak_count,
            trade_log=trade_log,
        )
