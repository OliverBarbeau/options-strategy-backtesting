"""RSI-exhaustion pullback: combine price pullback with momentum exhaustion.

Hypothesis: The base pullback strategy uses a price-level filter (3% from
20-day high). Adding an RSI oversold condition captures a *different*
dimension — momentum exhaustion. When both trigger simultaneously, the
selling pressure is likely near a local bottom, making the mean-reversion
thesis stronger.

Expected behavior:
- Fewer trades (dual filter is more selective)
- Higher win rate per trade (better-timed entries)
- Fewer catastrophic losses (RSI won't be oversold during orderly declines)

RSI is computed using the standard Wilder smoothing (exponential moving
average of gains/losses over N periods).
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


def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI: standard 14-period with EMA smoothing."""
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


@dataclass
class RSIPullbackResult:
    total_trades: int = 0
    winners: int = 0
    losers: int = 0
    total_pnl: float = 0.0
    max_drawdown_pct: float = 0.0
    trade_log: list[dict] = field(default_factory=list)

    # Diagnostics
    skipped_rsi_not_oversold: int = 0
    skipped_rsi_extreme: int = 0

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
        if self.skipped_rsi_not_oversold:
            lines.append(f"Skipped (RSI not oversold): {self.skipped_rsi_not_oversold}")
        if self.skipped_rsi_extreme:
            lines.append(f"Skipped (RSI too extreme):  {self.skipped_rsi_extreme}")
        return "\n".join(lines)


class RSIPullbackStrategy:
    """Pullback + RSI oversold confirmation for higher-precision entries.

    Entry requires BOTH conditions simultaneously:
    1. Price is 3%+ below 20-day high (standard pullback)
    2. RSI is below rsi_oversold threshold (momentum exhaustion)

    Optional: rsi_extreme_floor rejects entries where RSI is *too* low
    (e.g., < 15), which often signals a crash/capitulation where selling
    premium is dangerous rather than a normal dip.

    Args:
        buffer: Short strike OTM distance (default 0.10).
        spread_pct: Spread width as % of underlying (default 0.02).
        pullback_threshold: Min drawdown from recent high (default 0.03).
        lookback: Days to look back for recent high (default 20).
        dte_open: DTE at entry (default 30).
        dte_close: Remaining DTE at close (default 14).
        rsi_period: RSI calculation period (default 14).
        rsi_oversold: RSI threshold for entry (default 35).
        rsi_extreme_floor: RSI below this = too dangerous (default 0, disabled).
        risk_free_rate: Annualized risk-free rate.
        vol_window: HV rolling window.
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
        rsi_period: int = 14,
        rsi_oversold: float = 35.0,
        rsi_extreme_floor: float = 0.0,
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
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_extreme_floor = rsi_extreme_floor
        self.r = risk_free_rate
        self.vol_window = vol_window
        self.commission = commission_per_contract
        self.slippage_pct = slippage_pct

    def run(
        self,
        df: pd.DataFrame,
        close_col: str = "close",
        max_contracts: int = 10,
    ) -> RSIPullbackResult:
        close = df[close_col].values
        timestamps = df.index.values
        vol = historical_volatility(df[close_col], window=self.vol_window)
        rsi = compute_rsi(df[close_col], period=self.rsi_period)

        offset_open = max(1, int(self.dte_open * 21 / 30))
        offset_close = max(1, int((self.dte_open - self.dte_close) * 21 / 30))

        trade_log = []
        cum_pnl = 0.0
        peak = 0.0
        max_dd = 0.0
        skipped_rsi_not_oversold = 0
        skipped_rsi_extreme = 0

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

            vol_val = vol.iloc[i]
            if np.isnan(vol_val) or vol_val <= 0:
                i += 1
                continue

            # --- RSI condition ---
            rsi_val = rsi.iloc[i]
            if np.isnan(rsi_val):
                i += 1
                continue

            if rsi_val > self.rsi_oversold:
                skipped_rsi_not_oversold += 1
                i += 1
                continue

            if self.rsi_extreme_floor > 0 and rsi_val < self.rsi_extreme_floor:
                skipped_rsi_extreme += 1
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
                "rsi": rsi_val,
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

        return RSIPullbackResult(
            total_trades=len(trade_log),
            winners=sum(1 for t in trade_log if t["winner"]),
            losers=sum(1 for t in trade_log if not t["winner"]),
            total_pnl=sum(t["pnl"] for t in trade_log),
            max_drawdown_pct=max_dd / peak if peak > 0 else 0,
            trade_log=trade_log,
            skipped_rsi_not_oversold=skipped_rsi_not_oversold,
            skipped_rsi_extreme=skipped_rsi_extreme,
        )
