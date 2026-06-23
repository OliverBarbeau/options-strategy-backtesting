"""Gap recovery strategy: sell premium after sharp single-day drops.

Hypothesis: The base pullback strategy measures drawdown from a 20-day high,
which captures *rolling* declines. But *gap-downs* — sharp single-session
drops — are a different signal. They often represent:
1. Overreaction to news (earnings excluded via filter)
2. Sector rotation or macro shock that mean-reverts
3. Elevated intraday vol that pumps option premiums

By entering specifically after gap-down days (>2% single-day drop), we
capture a distinct entry signal from the rolling pullback. The two signals
may be complementary — a gap-down within a rolling pullback is especially
interesting.

Safety mechanisms:
- Earnings filter: Skip if stock has earnings within hold period (caller's
  responsibility, same as other strategies).
- Gap floor: Reject gaps larger than 10% (crash / capitulation territory).
- Volume confirmation (optional): Require above-average volume on the gap
  day, confirming real selling pressure (vs. low-liquidity drift).
- Consecutive gap filter: Don't re-enter if there was a gap within the
  last N days (avoids stacking into an accelerating decline).
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
class GapRecoveryResult:
    total_trades: int = 0
    winners: int = 0
    losers: int = 0
    total_pnl: float = 0.0
    max_drawdown_pct: float = 0.0
    trade_log: list[dict] = field(default_factory=list)

    # Diagnostics
    skipped_gap_too_large: int = 0
    skipped_low_volume: int = 0
    skipped_recent_gap: int = 0

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
        if self.skipped_gap_too_large:
            lines.append(f"Skipped (gap too large): {self.skipped_gap_too_large}")
        if self.skipped_low_volume:
            lines.append(f"Skipped (low volume):    {self.skipped_low_volume}")
        if self.skipped_recent_gap:
            lines.append(f"Skipped (recent gap):    {self.skipped_recent_gap}")
        return "\n".join(lines)


class GapRecoveryStrategy:
    """Sell put credit spreads after sharp single-day drops.

    Args:
        buffer: Short strike OTM distance (default 0.10).
        spread_pct: Spread width as % of underlying (default 0.02).
        gap_threshold: Minimum single-day drop to trigger (default 0.02 = 2%).
        gap_ceiling: Maximum gap size — larger gaps are crashes (default 0.10).
        dte_open: DTE at entry (default 30).
        dte_close: Remaining DTE at close (default 14).
        require_volume: If True, gap-day volume must exceed 20-day average.
        gap_cooldown: Trading days to wait before another gap entry (default 5).
        risk_free_rate: Annualized risk-free rate.
        vol_window: HV rolling window.
        commission_per_contract: Per-leg commission.
        slippage_pct: Bid-ask slippage.
    """

    def __init__(
        self,
        buffer: float = 0.10,
        spread_pct: float = 0.02,
        gap_threshold: float = 0.02,
        gap_ceiling: float = 0.10,
        dte_open: int = 30,
        dte_close: int = 14,
        require_volume: bool = False,
        gap_cooldown: int = 5,
        max_contracts: int = 10,
        risk_free_rate: float = 0.05,
        vol_window: int = 30,
        commission_per_contract: float = 0.65,
        slippage_pct: float = 0.02,
    ):
        self.buffer = buffer
        self.spread_pct = spread_pct
        self.gap_threshold = gap_threshold
        self.gap_ceiling = gap_ceiling
        self.dte_open = dte_open
        self.dte_close = dte_close
        self.require_volume = require_volume
        self.gap_cooldown = gap_cooldown
        self.max_contracts = max_contracts
        self.r = risk_free_rate
        self.vol_window = vol_window
        self.commission = commission_per_contract
        self.slippage_pct = slippage_pct

    def run(
        self,
        df: pd.DataFrame,
        close_col: str = "close",
    ) -> GapRecoveryResult:
        close = df[close_col].values
        timestamps = df.index.values
        vol = historical_volatility(df[close_col], window=self.vol_window)

        # Compute daily returns
        daily_return = df[close_col].pct_change()

        # Volume average (20-day) for volume confirmation
        has_volume = "volume" in df.columns
        vol_avg = df["volume"].rolling(20).mean() if has_volume else None

        offset_open = max(1, int(self.dte_open * 21 / 30))
        offset_close = max(1, int((self.dte_open - self.dte_close) * 21 / 30))

        trade_log = []
        cum_pnl = 0.0
        peak = 0.0
        max_dd = 0.0
        skipped_gap_too_large = 0
        skipped_low_volume = 0
        skipped_recent_gap = 0
        last_gap_idx = -self.gap_cooldown - 1  # allow first entry

        start = max(self.vol_window, 21)  # need 20-day vol avg
        i = start
        while i < len(df) - offset_open:
            current = close[i]
            ret = daily_return.iloc[i]

            if np.isnan(ret):
                i += 1
                continue

            # --- Gap-down condition ---
            if ret > -self.gap_threshold:
                i += 1
                continue

            gap_size = abs(ret)

            # --- Gap too large (crash territory) ---
            if gap_size > self.gap_ceiling:
                skipped_gap_too_large += 1
                i += 1
                continue

            # --- Cooldown: no recent gap entries ---
            if i - last_gap_idx < self.gap_cooldown:
                skipped_recent_gap += 1
                i += 1
                continue

            # --- Volume confirmation ---
            if self.require_volume and has_volume and vol_avg is not None:
                today_vol = df["volume"].iloc[i]
                avg = vol_avg.iloc[i]
                if not np.isnan(avg) and avg > 0 and today_vol < avg:
                    skipped_low_volume += 1
                    i += 1
                    continue

            vol_val = vol.iloc[i]
            if np.isnan(vol_val) or vol_val <= 0:
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

            contracts = self.max_contracts

            # Friction
            open_comm = self.commission * 2 * contracts
            slippage = credit * contracts * self.slippage_pct
            net_credit = credit * contracts - slippage

            # Close at checkpoint
            close_idx = min(i + offset_close, len(close) - 1)
            exit_price = close[close_idx]
            exit_vol = vol.iloc[close_idx] if close_idx < len(vol) else vol_val
            if np.isnan(exit_vol) or exit_vol <= 0:
                exit_vol = vol_val

            close_cost = (
                bs_put_price(exit_price, sk, self.dte_close / 365, self.r, exit_vol)
                - bs_put_price(exit_price, lk, self.dte_close / 365, self.r, exit_vol)
            ) * 100 * contracts
            close_comm = self.commission * 2 * contracts

            pnl = net_credit - close_cost - open_comm - close_comm
            winner = pnl > 0

            trade_log.append({
                "date": pd.Timestamp(timestamps[i], unit="s"),
                "exit_date": pd.Timestamp(timestamps[close_idx], unit="s"),
                "entry_price": current,
                "exit_price": exit_price,
                "gap_pct": ret * 100,
                "sigma": vol_val,
                "contracts": contracts,
                "credit": net_credit,
                "pnl": pnl,
                "winner": winner,
            })

            last_gap_idx = i
            cum_pnl += pnl
            peak = max(peak, cum_pnl)
            max_dd = min(max_dd, cum_pnl - peak)

            i += offset_close

        return GapRecoveryResult(
            total_trades=len(trade_log),
            winners=sum(1 for t in trade_log if t["winner"]),
            losers=sum(1 for t in trade_log if not t["winner"]),
            total_pnl=sum(t["pnl"] for t in trade_log),
            max_drawdown_pct=max_dd / peak if peak > 0 else 0,
            trade_log=trade_log,
            skipped_gap_too_large=skipped_gap_too_large,
            skipped_low_volume=skipped_low_volume,
            skipped_recent_gap=skipped_recent_gap,
        )
