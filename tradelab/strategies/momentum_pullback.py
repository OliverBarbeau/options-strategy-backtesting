"""Momentum-filtered pullback: only enter pullbacks within confirmed uptrends.

Hypothesis: A 3%+ pullback in a stock that's above its 50-day SMA is a
mean-reversion dip-buy opportunity. The same pullback in a stock *below*
its 50-day SMA is more likely a trend continuation — selling premium into
a falling knife.

This filter should:
- Reduce losses during extended drawdowns (2022 H1, Oct 2018)
- Slightly reduce trade count (skip bear-market entries)
- Improve win rate and P/L per trade

All other mechanics (30 DTE open, 14 DTE close, friction) identical to
the validated pullback strategy for clean comparison.
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
class MomentumPullbackResult:
    total_trades: int = 0
    winners: int = 0
    losers: int = 0
    total_pnl: float = 0.0
    max_drawdown_pct: float = 0.0
    trade_log: list[dict] = field(default_factory=list)

    # Diagnostic
    skipped_below_sma: int = 0
    skipped_below_long_sma: int = 0

    @property
    def win_rate(self) -> float:
        return self.winners / self.total_trades if self.total_trades > 0 else 0.0

    def summary(self) -> str:
        lines = [
            f"Trades:     {self.total_trades}",
            f"Win rate:   {self.win_rate:.1%} ({self.winners}W / {self.losers}L)",
            f"Total P/L:  ${self.total_pnl:+,.2f}",
            f"Max DD:     {self.max_drawdown_pct:.1%}",
        ]
        if self.skipped_below_sma:
            lines.append(f"Skipped (below trend SMA):  {self.skipped_below_sma}")
        if self.skipped_below_long_sma:
            lines.append(f"Skipped (below long SMA):   {self.skipped_below_long_sma}")
        return "\n".join(lines)


class MomentumPullbackStrategy:
    """Enter put credit spreads on pullbacks, but only when trend is intact.

    Two trend filters (independently toggleable):
    1. **Trend SMA** (default 50-day): Price must be above this SMA to enter.
       Filters out pullbacks in downtrends.
    2. **Long SMA** (optional, e.g. 200-day): Price must also be above this
       longer-term trend. Adds a secular trend filter on top of the
       intermediate trend.

    When both are enabled, this implements a "dual momentum" screen:
    the stock must be in both intermediate and long-term uptrends before
    we sell premium into a short-term dip.

    Args:
        buffer: Short strike OTM distance (default 0.10).
        spread_pct: Spread width as % of underlying (default 0.02).
        pullback_threshold: Min drawdown from recent high (default 0.03).
        lookback: Days to look back for recent high (default 20).
        dte_open: DTE at entry (default 30).
        dte_close: Remaining DTE at close (default 14).
        trend_sma: Short/intermediate trend SMA period (default 50).
        long_sma: Long-term trend SMA period (0 = disabled, default 0).
        risk_free_rate: Annualized risk-free rate.
        vol_window: HV rolling window (default 30).
        commission_per_contract: Per-leg commission.
        slippage_pct: Bid-ask slippage.
    """

    def __init__(
        self,
        buffer: float = 0.10,
        spread_pct: float = 0.02,
        pullback_threshold: float = 0.03,
        lookback: int = 20,
        dte_open: int = 30,
        dte_close: int = 14,
        trend_sma: int = 50,
        long_sma: int = 0,
        risk_free_rate: float = 0.05,
        vol_window: int = 30,
        commission_per_contract: float = 0.65,
        slippage_pct: float = 0.02,
    ):
        self.buffer = buffer
        self.spread_pct = spread_pct
        self.pullback_threshold = pullback_threshold
        self.lookback = lookback
        self.dte_open = dte_open
        self.dte_close = dte_close
        self.trend_sma = trend_sma
        self.long_sma = long_sma
        self.r = risk_free_rate
        self.vol_window = vol_window
        self.commission = commission_per_contract
        self.slippage_pct = slippage_pct

    def run(
        self,
        df: pd.DataFrame,
        close_col: str = "close",
        max_contracts: int = 10,
    ) -> MomentumPullbackResult:
        close = df[close_col].values
        timestamps = df.index.values
        vol = historical_volatility(df[close_col], window=self.vol_window)

        # Compute SMAs
        sma_short = df[close_col].rolling(self.trend_sma).mean()
        sma_long = (
            df[close_col].rolling(self.long_sma).mean()
            if self.long_sma > 0 else None
        )

        offset_open = max(1, int(self.dte_open * 21 / 30))
        offset_close = max(1, int((self.dte_open - self.dte_close) * 21 / 30))

        trade_log = []
        cum_pnl = 0.0
        peak = 0.0
        max_dd = 0.0
        skipped_below_sma = 0
        skipped_below_long_sma = 0

        start = max(self.vol_window, self.lookback, self.trend_sma, self.long_sma)
        i = start
        while i < len(df) - offset_open:
            current = close[i]
            # --- Pullback condition ---
            recent_high = df[close_col].iloc[i - self.lookback : i + 1].max()
            drawdown = (current - recent_high) / recent_high
            if drawdown > -self.pullback_threshold:
                i += 1
                continue

            vol_val = vol.iloc[i]
            if np.isnan(vol_val) or vol_val <= 0:
                i += 1
                continue

            # --- Trend filter: intermediate SMA ---
            sma_val = sma_short.iloc[i]
            if np.isnan(sma_val) or current < sma_val:
                skipped_below_sma += 1
                i += 1
                continue

            # --- Trend filter: long-term SMA ---
            if sma_long is not None:
                long_val = sma_long.iloc[i]
                if np.isnan(long_val) or current < long_val:
                    skipped_below_long_sma += 1
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

            # Friction
            open_comm = self.commission * 2 * max_contracts
            slippage = credit * max_contracts * self.slippage_pct
            net_credit = credit * max_contracts - slippage

            # Close at checkpoint
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

            trade_log.append({
                "date": pd.Timestamp(timestamps[i], unit="s"),
                "exit_date": pd.Timestamp(timestamps[close_idx], unit="s"),
                "entry_price": current,
                "exit_price": exit_price,
                "pullback_pct": drawdown * 100,
                "sma_short": sma_val,
                "sma_long": sma_long.iloc[i] if sma_long is not None else None,
                "sigma": vol_val,
                "contracts": max_contracts,
                "credit": net_credit,
                "pnl": pnl,
                "winner": winner,
            })

            cum_pnl += pnl
            peak = max(peak, cum_pnl)
            max_dd = min(max_dd, cum_pnl - peak)

            i += offset_close

        return MomentumPullbackResult(
            total_trades=len(trade_log),
            winners=sum(1 for t in trade_log if t["winner"]),
            losers=sum(1 for t in trade_log if not t["winner"]),
            total_pnl=sum(t["pnl"] for t in trade_log),
            max_drawdown_pct=max_dd / peak if peak > 0 else 0,
            trade_log=trade_log,
            skipped_below_sma=skipped_below_sma,
            skipped_below_long_sma=skipped_below_long_sma,
        )
