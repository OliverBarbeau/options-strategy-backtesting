"""Regime-switching RSI+pullback strategy.

Hypothesis (derived from prior research):
- Standard 3% pullback put credit spreads (no RSI filter) earn well in
  calm/trending markets but bleed in bear regimes (2018 Q4, 2022).
- RSI<25 + pullback has ~100% win rate in bear markets but fires very
  rarely in calm markets, leaving money on the table.
- The two modes are COMPLEMENTARY. Use SPY HV30 as a regime detector:
  when the market is calm (HV30 < 25%), enter on any 3% pullback;
  when the market is stressed (HV30 >= 25%), demand confirmation via
  RSI<25 oversold before entering.

This is a pure switching wrapper — no new strike/close logic. Standard
30->14 DTE hold with the same friction model as rsi_pullback.
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

try:
    # Reuse the canonical helper when rsi_pullback is available.
    from tradelab.strategies.rsi_pullback import compute_rsi
except ImportError:  # pragma: no cover - fallback for stripped checkouts
    def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Wilder's RSI: standard N-period with EMA smoothing.

        Fallback copy of tradelab.strategies.rsi_pullback.compute_rsi so this
        module is self-contained if rsi_pullback isn't present.
        """
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))


@dataclass
class RegimeRSIResult:
    total_trades: int = 0
    winners: int = 0
    losers: int = 0
    total_pnl: float = 0.0
    max_drawdown_pct: float = 0.0
    trade_log: list[dict] = field(default_factory=list)

    # Mode breakdown
    calm_trades: int = 0
    bear_trades: int = 0

    # Diagnostics
    skipped_no_regime: int = 0
    skipped_bear_rsi: int = 0

    @property
    def win_rate(self) -> float:
        return self.winners / self.total_trades if self.total_trades > 0 else 0.0

    def summary(self) -> str:
        lines = [
            f"Trades:     {self.total_trades} (calm={self.calm_trades}, bear={self.bear_trades})",
            f"Win rate:   {self.win_rate:.1%} ({self.winners}W / {self.losers}L)",
            f"Total P/L:  ${self.total_pnl:+,.2f}",
            f"Max DD:     {self.max_drawdown_pct:.1%}",
        ]
        if self.skipped_no_regime:
            lines.append(f"Skipped (no regime data):  {self.skipped_no_regime}")
        if self.skipped_bear_rsi:
            lines.append(f"Skipped (bear: RSI>25):    {self.skipped_bear_rsi}")
        return "\n".join(lines)


class RegimeRSIStrategy:
    """Regime-switching put credit spread entry.

    Uses SPY HV30 to decide between two entry modes on each candidate day:
    - calm (HV30 < regime_threshold): only require a pullback
    - bear (HV30 >= regime_threshold): require pullback AND RSI < bear_rsi
    - missing SPY data: skip day

    Standard 30->14 DTE hold. Inline Black-Scholes pricing.

    Args:
        buffer: Short strike OTM distance (default 0.10).
        spread_pct: Spread width as % of underlying (default 0.02).
        dte_open: DTE at entry (default 30).
        dte_close: Remaining DTE at close (default 14).
        lookback: Days to look back for pullback high (default 20).
        vol_window: HV rolling window (default 30).
        rsi_period: RSI calculation period (default 14).
        risk_free_rate: Annualized risk-free rate (default 0.05).
        commission_per_contract: Per-leg commission (default 0.65).
        slippage_pct: Bid-ask slippage (default 0.02).
        regime_threshold: SPY HV30 level that defines the bear regime
            (default 0.25 = 25% annualized).
        calm_pullback: Required pullback in calm regime (default 0.03).
        bear_rsi: RSI oversold threshold in bear regime (default 25.0).
        bear_pullback: Required pullback in bear regime (default 0.03).
    """

    def __init__(
        self,
        buffer: float = 0.10,
        spread_pct: float = 0.02,
        dte_open: int = 30,
        dte_close: int = 14,
        lookback: int = 20,
        vol_window: int = 30,
        rsi_period: int = 14,
        risk_free_rate: float = 0.05,
        commission_per_contract: float = 0.65,
        slippage_pct: float = 0.02,
        regime_threshold: float = 0.25,
        calm_pullback: float = 0.03,
        bear_rsi: float = 25.0,
        bear_pullback: float = 0.03,
    ):
        self.buffer = buffer
        self.spread_pct = spread_pct
        self.dte_open = dte_open
        self.dte_close = dte_close
        self.lookback = lookback
        self.vol_window = vol_window
        self.rsi_period = rsi_period
        self.r = risk_free_rate
        self.commission = commission_per_contract
        self.slippage_pct = slippage_pct
        self.regime_threshold = regime_threshold
        self.calm_pullback = calm_pullback
        self.bear_rsi = bear_rsi
        self.bear_pullback = bear_pullback

    def _align_market_vol(
        self, market_vol_series: pd.Series, timestamps: np.ndarray
    ) -> np.ndarray:
        """Resolve SPY HV30 for each day in `timestamps`.

        The ticker df is indexed by unix-timestamp ints (see pipeline.py
        `fetch_stock`), but callers may pass a market_vol_series indexed
        either the same way or by a pandas DatetimeIndex (tz-naive or
        tz-aware). We normalize everything to UTC tz-naive DatetimeIndex,
        then reindex to the target timestamps with forward-fill so each
        day gets the most recent available SPY HV30 reading.

        Returns a numpy array of len(timestamps); NaN means "no data".
        """
        if market_vol_series is None or len(market_vol_series) == 0:
            return np.full(len(timestamps), np.nan, dtype=float)

        mv = market_vol_series.copy()

        # Normalize mv index to tz-naive DatetimeIndex
        if isinstance(mv.index, pd.DatetimeIndex):
            if mv.index.tz is not None:
                mv.index = mv.index.tz_convert(None)
        else:
            # Assume numeric unix seconds
            try:
                mv.index = pd.to_datetime(mv.index, unit="s", utc=True).tz_convert(None)
            except Exception:
                mv.index = pd.to_datetime(mv.index)
                if getattr(mv.index, "tz", None) is not None:
                    mv.index = mv.index.tz_convert(None)

        mv = mv[~mv.index.duplicated(keep="last")].sort_index()

        # Convert ticker timestamps (unix seconds) to DatetimeIndex
        try:
            target_idx = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(None)
        except Exception:
            target_idx = pd.to_datetime(timestamps)
            if getattr(target_idx, "tz", None) is not None:
                target_idx = target_idx.tz_convert(None)

        # Normalize both sides to date-level to avoid HH:MM:SS mismatches
        mv_daily = mv.copy()
        mv_daily.index = mv_daily.index.normalize()
        mv_daily = mv_daily[~mv_daily.index.duplicated(keep="last")].sort_index()

        target_daily = pd.DatetimeIndex(target_idx).normalize()

        aligned = mv_daily.reindex(target_daily, method="ffill")
        return aligned.values.astype(float)

    def run(
        self,
        df: pd.DataFrame,
        market_vol_series: pd.Series | None = None,
        close_col: str = "close",
        max_contracts: int = 10,
    ) -> RegimeRSIResult:
        close = df[close_col].values
        timestamps = df.index.values
        vol = historical_volatility(df[close_col], window=self.vol_window)
        rsi = compute_rsi(df[close_col], period=self.rsi_period)
        market_vol_arr = self._align_market_vol(market_vol_series, timestamps)

        offset_open = max(1, int(self.dte_open * 21 / 30))
        offset_close = max(1, int((self.dte_open - self.dte_close) * 21 / 30))

        trade_log = []
        cum_pnl = 0.0
        peak = 0.0
        max_dd = 0.0
        calm_trades = 0
        bear_trades = 0
        skipped_no_regime = 0
        skipped_bear_rsi = 0

        start = max(self.vol_window, self.lookback, self.rsi_period + 1)
        i = start
        while i < len(df) - offset_open:
            current = close[i]

            # --- Resolve regime ---
            mv = market_vol_arr[i]
            if np.isnan(mv):
                skipped_no_regime += 1
                i += 1
                continue

            is_bear = mv >= self.regime_threshold

            # --- Pullback condition (depends on regime) ---
            required_pullback = self.bear_pullback if is_bear else self.calm_pullback
            recent_high = df[close_col].iloc[i - self.lookback : i + 1].max()
            drawdown = (current - recent_high) / recent_high
            if drawdown > -required_pullback:
                i += 1
                continue

            vol_val = vol.iloc[i]
            if np.isnan(vol_val) or vol_val <= 0:
                i += 1
                continue

            # --- RSI condition (bear only) ---
            rsi_val = rsi.iloc[i]
            if np.isnan(rsi_val):
                i += 1
                continue

            if is_bear and rsi_val > self.bear_rsi:
                skipped_bear_rsi += 1
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
            mode = "bear" if is_bear else "calm"
            if is_bear:
                bear_trades += 1
            else:
                calm_trades += 1

            trade_log.append({
                "date": pd.Timestamp(timestamps[i], unit="s"),
                "exit_date": pd.Timestamp(timestamps[close_idx], unit="s"),
                "entry_price": current,
                "exit_price": exit_price,
                "pullback_pct": drawdown * 100,
                "rsi": rsi_val,
                "spy_hv30": mv,
                "mode": mode,
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

        return RegimeRSIResult(
            total_trades=len(trade_log),
            winners=sum(1 for t in trade_log if t["winner"]),
            losers=sum(1 for t in trade_log if not t["winner"]),
            total_pnl=sum(t["pnl"] for t in trade_log),
            max_drawdown_pct=max_dd / peak if peak > 0 else 0,
            trade_log=trade_log,
            calm_trades=calm_trades,
            bear_trades=bear_trades,
            skipped_no_regime=skipped_no_regime,
            skipped_bear_rsi=skipped_bear_rsi,
        )
