"""VolPremiumMax: upsize into the empirical HV30 sweet spot (25-45%).

This is the INVERSE of VolScaledPullbackStrategy. The research finding
motivating this: vol-scaling (shrinking size as vol rises) failed because
it scaled the wrong direction. The richest credit-spread premiums sit in
the 25-45% HV30 band (Experiment 11, scanner validation); below 25% the
premium is too thin to beat friction, above 45-50% breach risk dominates.

Sizing policy at entry:
- HV30 in [sweet_low, sweet_high]  -> UPSIZE (sweet_multiplier x max)
- HV30 in [ok_low, sweet_low) or (sweet_high, ok_high]  -> baseline
- Outside [ok_low, ok_high]        -> SKIP (counted as skipped_vol_band)

Entry/exit are otherwise identical to PullbackEntryStrategy (3% pullback
from 20d high, 30 DTE open -> 14 DTE close). Friction scales with the
actual contracts traded, not max_contracts.
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
class VolPremiumMaxResult:
    total_trades: int = 0
    winners: int = 0
    losers: int = 0
    total_pnl: float = 0.0
    max_drawdown_pct: float = 0.0
    trade_log: list[dict] = field(default_factory=list)

    # Diagnostics
    sweet_trades: int = 0
    ok_trades: int = 0
    skipped_vol_band: int = 0

    @property
    def win_rate(self) -> float:
        return self.winners / self.total_trades if self.total_trades > 0 else 0.0

    def summary(self) -> str:
        lines = [
            f"Trades:     {self.total_trades}",
            f"Win rate:   {self.win_rate:.1%} ({self.winners}W / {self.losers}L)",
            f"Total P/L:  ${self.total_pnl:+,.2f}",
            f"Max DD:     {self.max_drawdown_pct:.1%}",
            f"Sweet band trades:  {self.sweet_trades}",
            f"OK band trades:     {self.ok_trades}",
            f"Skipped (vol band): {self.skipped_vol_band}",
        ]
        return "\n".join(lines)


class VolPremiumMaxStrategy:
    """Pullback entry with INVERSE vol-scaling: upsize into the HV sweet spot.

    Args:
        buffer: Short strike OTM distance (default 0.10 = 10%).
        spread_pct: Spread width as % of underlying (default 0.02).
        pullback_threshold: Min pullback from 20d high (default 0.03).
        lookback: Pullback lookback (default 20).
        dte_open: DTE at entry (default 30).
        dte_close: Remaining DTE at close (default 14).
        vol_window: HV rolling window (default 30 -> HV30).
        risk_free_rate: Risk-free rate (default 0.05).
        commission_per_contract: Per-leg commission (default 0.65).
        slippage_pct: Slippage as fraction of credit (default 0.02).
        sweet_low / sweet_high: The HV30 sweet spot band (default 0.25-0.45).
        ok_low / ok_high: Baseline-sized band outside the sweet spot (0.15-0.55).
        sweet_multiplier: Size multiplier inside the sweet spot (default 1.5).
        ok_multiplier: Size multiplier in the OK band (default 1.0).
    """

    def __init__(
        self,
        buffer: float = 0.10,
        spread_pct: float = 0.02,
        pullback_threshold: float = 0.03,
        lookback: int = 20,
        dte_open: int = 30,
        dte_close: int = 14,
        vol_window: int = 30,
        risk_free_rate: float = 0.05,
        commission_per_contract: float = 0.65,
        slippage_pct: float = 0.02,
        sweet_low: float = 0.25,
        sweet_high: float = 0.45,
        ok_low: float = 0.15,
        ok_high: float = 0.55,
        sweet_multiplier: float = 1.5,
        ok_multiplier: float = 1.0,
    ):
        self.buffer = buffer
        self.spread_pct = spread_pct
        self.pullback_threshold = pullback_threshold
        self.lookback = lookback
        self.dte_open = dte_open
        self.dte_close = dte_close
        self.vol_window = vol_window
        self.r = risk_free_rate
        self.commission = commission_per_contract
        self.slippage_pct = slippage_pct
        self.sweet_low = sweet_low
        self.sweet_high = sweet_high
        self.ok_low = ok_low
        self.ok_high = ok_high
        self.sweet_multiplier = sweet_multiplier
        self.ok_multiplier = ok_multiplier

    def _classify_band(self, vol_val: float) -> str:
        """Return 'sweet', 'ok', or 'skip' based on HV30 bands."""
        if self.sweet_low <= vol_val <= self.sweet_high:
            return "sweet"
        if (self.ok_low <= vol_val < self.sweet_low) or (
            self.sweet_high < vol_val <= self.ok_high
        ):
            return "ok"
        return "skip"

    def _contracts_for_band(self, band: str, max_contracts: int) -> int:
        if band == "sweet":
            return max(1, round(max_contracts * self.sweet_multiplier))
        if band == "ok":
            return max(1, round(max_contracts * self.ok_multiplier))
        return 0

    def run(
        self,
        df: pd.DataFrame,
        close_col: str = "close",
        max_contracts: int = 10,
    ) -> VolPremiumMaxResult:
        close = df[close_col].values
        timestamps = df.index.values
        vol = historical_volatility(df[close_col], window=self.vol_window)

        offset_open = max(1, int(self.dte_open * 21 / 30))
        offset_close = max(1, int((self.dte_open - self.dte_close) * 21 / 30))

        trade_log: list[dict] = []
        cum_pnl = 0.0
        peak = 0.0
        max_dd = 0.0
        sweet_trades = 0
        ok_trades = 0
        skipped_vol_band = 0

        i = max(self.vol_window, self.lookback)
        while i < len(df) - offset_open:
            current = close[i]

            # Pullback gate
            recent_high = df[close_col].iloc[i - self.lookback : i + 1].max()
            drawdown = (current - recent_high) / recent_high
            if drawdown > -self.pullback_threshold:
                i += 1
                continue

            vol_val = vol.iloc[i]
            if np.isnan(vol_val) or vol_val <= 0:
                i += 1
                continue

            # Vol band classification
            band = self._classify_band(vol_val)
            if band == "skip":
                skipped_vol_band += 1
                i += 1
                continue

            contracts = self._contracts_for_band(band, max_contracts)
            if contracts <= 0:
                i += 1
                continue

            # Price the spread
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

            # Friction scaled by actual contracts
            open_comm = self.commission * 2 * contracts
            slippage = credit * contracts * self.slippage_pct
            net_credit = credit * contracts - slippage

            # Exit at dte_close remaining
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
                "pullback_pct": drawdown * 100,
                "sigma": vol_val,
                "band": band,
                "contracts": contracts,
                "credit": net_credit,
                "pnl": pnl,
                "winner": winner,
            })

            if band == "sweet":
                sweet_trades += 1
            else:
                ok_trades += 1

            cum_pnl += pnl
            peak = max(peak, cum_pnl)
            max_dd = min(max_dd, cum_pnl - peak)

            i += offset_close

        return VolPremiumMaxResult(
            total_trades=len(trade_log),
            winners=sum(1 for t in trade_log if t["winner"]),
            losers=sum(1 for t in trade_log if not t["winner"]),
            total_pnl=sum(t["pnl"] for t in trade_log),
            max_drawdown_pct=max_dd / peak if peak > 0 else 0,
            trade_log=trade_log,
            sweet_trades=sweet_trades,
            ok_trades=ok_trades,
            skipped_vol_band=skipped_vol_band,
        )
