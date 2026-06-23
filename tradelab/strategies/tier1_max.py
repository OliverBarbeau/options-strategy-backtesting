"""Tier1Max: high-conviction RSI<25 only, with stacked position sizing.

Hypothesis from prior research:
- RSI<25 + 3% pullback (composite "Tier 1") produced the highest per-trade
  edge ever measured (~+$75/trade, 66.7% WR) but only ~6 trades over
  10 years across 6 tickers.
- Composite Tier 3 (5% pullback, no RSI) was -$6.55/trade, dragging the
  overall composite down.

Individualized confident claim being exploited here: when RSI<25 fires
alongside a qualifying pullback, the per-trade edge is so large that the
correct response is to STACK the position rather than diversify into
weaker signals. Sparse signal frequency (~1-2 trades/year per ticker) is
acceptable if each trade is sized to compensate.

Entry: requires BOTH RSI <= rsi_threshold AND drawdown from 20-day high
>= pullback_threshold. No fallback tier.

Sizing: effective_contracts = max(1, round(max_contracts * size_multiplier)).

Exit: standard 30 DTE -> 14 DTE checkpoint close.

Risk warning: stacking on rare signals concentrates tail risk. Losses
on a stacked trade hit the equity curve hard — max drawdown is the
critical metric to watch alongside total P/L.
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
from tradelab.strategies.rsi_pullback import compute_rsi


@dataclass
class Tier1MaxResult:
    total_trades: int = 0
    winners: int = 0
    losers: int = 0
    total_pnl: float = 0.0
    max_drawdown_pct: float = 0.0
    trade_log: list[dict] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        return self.winners / self.total_trades if self.total_trades > 0 else 0.0

    def summary(self) -> str:
        return (
            f"Trades:     {self.total_trades}\n"
            f"Win rate:   {self.win_rate:.1%} ({self.winners}W / {self.losers}L)\n"
            f"Total P/L:  ${self.total_pnl:+,.2f}\n"
            f"Max DD:     {self.max_drawdown_pct:.1%}"
        )


class Tier1MaxStrategy:
    """High-conviction RSI<25 + pullback entries with stacked sizing.

    Args:
        buffer: Short strike OTM distance (default 0.10).
        spread_pct: Spread width as % of underlying (default 0.02).
        pullback_threshold: Min drawdown from recent high (default 0.03).
        lookback: Days to look back for recent high (default 20).
        dte_open: DTE at entry (default 30).
        dte_close: Remaining DTE at close (default 14).
        rsi_period: RSI calculation period (default 14).
        rsi_threshold: RSI must be <= this to enter (default 25.0).
        size_multiplier: Position stack factor — effective_contracts =
            max(1, round(max_contracts * size_multiplier)). Default 2.0.
        vol_window: HV rolling window (default 30).
        risk_free_rate: Annualized risk-free rate (default 0.05).
        commission_per_contract: Per-leg commission (default 0.65).
        slippage_pct: Bid-ask slippage (default 0.02).
    """

    def __init__(
        self,
        buffer: float = 0.10,
        spread_pct: float = 0.02,
        pullback_threshold: float = 0.03,
        lookback: int = 20,
        dte_open: int = 30,
        dte_close: int = 14,
        rsi_period: int = 14,
        rsi_threshold: float = 25.0,
        size_multiplier: float = 2.0,
        vol_window: int = 30,
        risk_free_rate: float = 0.05,
        commission_per_contract: float = 0.65,
        slippage_pct: float = 0.02,
    ):
        self.buffer = buffer
        self.spread_pct = spread_pct
        self.pullback_threshold = pullback_threshold
        self.lookback = lookback
        self.dte_open = dte_open
        self.dte_close = dte_close
        self.rsi_period = rsi_period
        self.rsi_threshold = rsi_threshold
        self.size_multiplier = size_multiplier
        self.vol_window = vol_window
        self.r = risk_free_rate
        self.commission = commission_per_contract
        self.slippage_pct = slippage_pct

    def run(
        self,
        df: pd.DataFrame,
        close_col: str = "close",
        max_contracts: int = 10,
    ) -> Tier1MaxResult:
        close = df[close_col].values
        timestamps = df.index.values
        vol = historical_volatility(df[close_col], window=self.vol_window)
        rsi = compute_rsi(df[close_col], period=self.rsi_period)

        offset_open = max(1, int(self.dte_open * 21 / 30))
        offset_close = max(1, int((self.dte_open - self.dte_close) * 21 / 30))

        # Stacked sizing: this is the key knob of the strategy
        effective_contracts = max(1, round(max_contracts * self.size_multiplier))

        trade_log = []
        cum_pnl = 0.0
        peak = 0.0
        max_dd = 0.0

        start = max(self.vol_window, self.lookback, self.rsi_period + 1)
        i = start
        while i < len(df) - offset_open:
            current = close[i]

            # --- Pullback condition ---
            recent_high = df[close_col].iloc[i - self.lookback : i + 1].max()
            drawdown = (current - recent_high) / recent_high
            if drawdown > -self.pullback_threshold:
                i += 1
                continue

            # --- HV sanity ---
            vol_val = vol.iloc[i]
            if np.isnan(vol_val) or vol_val <= 0:
                i += 1
                continue

            # --- RSI condition (strict: no fallback) ---
            rsi_val = rsi.iloc[i]
            if np.isnan(rsi_val):
                i += 1
                continue
            if rsi_val > self.rsi_threshold:
                i += 1
                continue

            # --- Price the spread ---
            sw = current * self.spread_pct
            sk = current * (1 - self.buffer)
            lk = sk - sw
            if lk <= 0:
                i += offset_open
                continue

            sp = put_credit_spread_price(
                current, sk, lk, self.dte_open / 365, self.r, vol_val
            )
            credit = sp["net_credit_dollar"]
            max_loss = sp["max_loss"]
            if credit <= 0 or max_loss <= 0:
                i += offset_open
                continue

            # Friction proportional to stacked size
            open_comm = self.commission * 2 * effective_contracts
            slippage = credit * effective_contracts * self.slippage_pct
            net_credit = credit * effective_contracts - slippage

            # Close at checkpoint
            close_idx = min(i + offset_close, len(close) - 1)
            exit_price = close[close_idx]
            exit_vol = vol.iloc[close_idx] if close_idx < len(vol) else vol_val
            if np.isnan(exit_vol) or exit_vol <= 0:
                exit_vol = vol_val

            close_cost = (
                bs_put_price(exit_price, sk, self.dte_close / 365, self.r, exit_vol)
                - bs_put_price(exit_price, lk, self.dte_close / 365, self.r, exit_vol)
            ) * 100 * effective_contracts
            close_comm = self.commission * 2 * effective_contracts

            pnl = net_credit - close_cost - open_comm - close_comm
            winner = pnl > 0

            trade_log.append({
                "date": pd.Timestamp(timestamps[i], unit="s"),
                "exit_date": pd.Timestamp(timestamps[close_idx], unit="s"),
                "entry_price": current,
                "exit_price": exit_price,
                "pullback_pct": drawdown * 100,
                "rsi": rsi_val,
                "sigma": vol_val,
                "contracts": effective_contracts,
                "size_multiplier": self.size_multiplier,
                "credit": net_credit,
                "pnl": pnl,
                "winner": winner,
            })

            cum_pnl += pnl
            peak = max(peak, cum_pnl)
            max_dd = min(max_dd, cum_pnl - peak)

            i += offset_close

        return Tier1MaxResult(
            total_trades=len(trade_log),
            winners=sum(1 for t in trade_log if t["winner"]),
            losers=sum(1 for t in trade_log if not t["winner"]),
            total_pnl=sum(t["pnl"] for t in trade_log),
            max_drawdown_pct=max_dd / peak if peak > 0 else 0.0,
            trade_log=trade_log,
        )
