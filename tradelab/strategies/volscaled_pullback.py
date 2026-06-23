"""Volatility-scaled pullback: dynamic position sizing based on realized vol.

Hypothesis: The base pullback strategy uses fixed max_contracts regardless of
market conditions. But the *risk per contract* varies enormously with vol —
a 10-contract position on NVDA at 50% HV is far riskier than at 20% HV.

This strategy keeps the same entry/exit logic but scales position size
inversely with current HV:
- Low vol (HV < 20%): full size (max_contracts)
- Normal vol (20-35%): 70% of max
- High vol (35-50%): 40% of max
- Extreme vol (>50%): 20% of max (or skip entirely)

The effect is a crude risk-parity approach: risk-per-trade stays more
constant across regimes, which should smooth the equity curve and reduce
max drawdown without sacrificing much total P/L.

Also implements an optional **vol-targeting** mode: instead of discrete
tiers, compute contracts = floor(target_vol / current_vol * base_contracts).
This produces smoother sizing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
import pandas as pd

from tradelab.options import (
    bs_put_price,
    put_credit_spread_price,
    historical_volatility,
)


@dataclass
class VolScaledResult:
    total_trades: int = 0
    winners: int = 0
    losers: int = 0
    total_pnl: float = 0.0
    max_drawdown_pct: float = 0.0
    trade_log: list[dict] = field(default_factory=list)

    # Diagnostics
    avg_contracts: float = 0.0
    skipped_extreme_vol: int = 0

    @property
    def win_rate(self) -> float:
        return self.winners / self.total_trades if self.total_trades > 0 else 0.0

    def summary(self) -> str:
        lines = [
            f"Trades:     {self.total_trades}",
            f"Win rate:   {self.win_rate:.1%} ({self.winners}W / {self.losers}L)",
            f"Total P/L:  ${self.total_pnl:+,.2f}",
            f"Max DD:     {self.max_drawdown_pct:.1%}",
            f"Avg size:   {self.avg_contracts:.1f} contracts",
        ]
        if self.skipped_extreme_vol:
            lines.append(f"Skipped (extreme vol):  {self.skipped_extreme_vol}")
        return "\n".join(lines)


class VolScaledPullbackStrategy:
    """Pullback strategy with volatility-scaled position sizing.

    Two sizing modes:
    1. **Tiered** (default): Discrete vol buckets with fixed scale factors.
    2. **Vol-targeting** (vol_target > 0): Continuous scaling to target a
       specific annualized vol exposure per trade.

    Args:
        buffer: Short strike OTM distance (default 0.10).
        spread_pct: Spread width as % of underlying (default 0.02).
        pullback_threshold: Min drawdown from recent high (default 0.03).
        lookback: Days to look back for recent high (default 20).
        dte_open: DTE at entry (default 30).
        dte_close: Remaining DTE at close (default 14).
        max_contracts: Ceiling for position size (default 10).
        vol_target: Target annualized vol for continuous scaling.
            0 = use tiered mode (default).
        skip_extreme_vol: If True, skip entries when HV > 55%.
        risk_free_rate: Annualized risk-free rate.
        vol_window: HV rolling window.
        commission_per_contract: Per-leg commission.
        slippage_pct: Bid-ask slippage.
    """

    # Tiered sizing: (vol_upper_bound, scale_factor)
    VOL_TIERS = [
        (0.20, 1.00),   # Low vol: full size
        (0.35, 0.70),   # Normal: 70%
        (0.50, 0.40),   # High: 40%
        (0.55, 0.20),   # Very high: 20%
    ]

    def __init__(
        self,
        buffer: float = 0.10,
        spread_pct: float = 0.02,
        pullback_threshold: float = 0.03,
        lookback: int = 20,
        dte_open: int = 30,
        dte_close: int = 14,
        max_contracts: int = 10,
        vol_target: float = 0.0,
        skip_extreme_vol: bool = True,
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
        self.max_contracts = max_contracts
        self.vol_target = vol_target
        self.skip_extreme_vol = skip_extreme_vol
        self.r = risk_free_rate
        self.vol_window = vol_window
        self.commission = commission_per_contract
        self.slippage_pct = slippage_pct

    def _compute_contracts(self, vol_val: float) -> int:
        """Determine position size based on current vol."""
        if self.vol_target > 0:
            # Continuous vol targeting
            raw = self.vol_target / vol_val * self.max_contracts
            return max(1, min(self.max_contracts, math.floor(raw)))

        # Tiered sizing
        for upper, scale in self.VOL_TIERS:
            if vol_val < upper:
                return max(1, round(self.max_contracts * scale))
        # Above all tiers
        return max(1, round(self.max_contracts * 0.20))

    def run(
        self,
        df: pd.DataFrame,
        close_col: str = "close",
    ) -> VolScaledResult:
        close = df[close_col].values
        timestamps = df.index.values
        vol = historical_volatility(df[close_col], window=self.vol_window)

        offset_open = max(1, int(self.dte_open * 21 / 30))
        offset_close = max(1, int((self.dte_open - self.dte_close) * 21 / 30))

        trade_log = []
        cum_pnl = 0.0
        peak = 0.0
        max_dd = 0.0
        skipped_extreme_vol = 0
        total_contracts_used = 0

        start = max(self.vol_window, self.lookback)
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

            # --- Extreme vol gate ---
            if self.skip_extreme_vol and vol_val > 0.55:
                skipped_extreme_vol += 1
                i += 1
                continue

            # --- Dynamic position sizing ---
            contracts = self._compute_contracts(vol_val)

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

            # Friction (scales with actual contracts)
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
                "pullback_pct": drawdown * 100,
                "sigma": vol_val,
                "contracts": contracts,
                "credit": net_credit,
                "pnl": pnl,
                "winner": winner,
            })

            total_contracts_used += contracts
            cum_pnl += pnl
            peak = max(peak, cum_pnl)
            max_dd = min(max_dd, cum_pnl - peak)

            i += offset_close

        avg_contracts = total_contracts_used / len(trade_log) if trade_log else 0

        return VolScaledResult(
            total_trades=len(trade_log),
            winners=sum(1 for t in trade_log if t["winner"]),
            losers=sum(1 for t in trade_log if not t["winner"]),
            total_pnl=sum(t["pnl"] for t in trade_log),
            max_drawdown_pct=max_dd / peak if peak > 0 else 0,
            trade_log=trade_log,
            avg_contracts=avg_contracts,
            skipped_extreme_vol=skipped_extreme_vol,
        )
