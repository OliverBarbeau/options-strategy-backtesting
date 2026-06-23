"""Simulation engine: advances accounts day-by-day to current date.

Simulates locally using the B-S pricing engine and yfinance data.

The engine replays each trading day, checking for entries/exits,
pricing positions, and recording snapshots.

Usage::

    from tradelab.simulator import Simulator
    from tradelab.account import SimulatedAccount

    account = SimulatedAccount.load_or_create("accounts/my_account.json", 25000)
    sim = Simulator(account, strategy="pullback")
    sim.catch_up()  # advance from last_advanced_date to today
    print(account.status())
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from tradelab.account import SimulatedAccount
from tradelab.pipeline import DataPipeline
from tradelab.options import (
    bs_put_price,
    put_credit_spread_price,
    historical_volatility,
)
from tradelab.pricing.strikes import snap_put_credit_spread, effective_buffer


class Simulator:
    """Advances a SimulatedAccount through time using a named strategy.

    Args:
        account: The account to simulate on.
        strategy: Strategy name ("pullback", "regime_adaptive", "conservative").
        tickers: Tickers to trade. Defaults to our research winners.
        pipe: DataPipeline instance (shared across simulators).
        max_contracts: Max contracts per position.
        max_positions: Max concurrent open positions.
        max_pct_per_position: Max % of buying power per position.
    """

    STRATEGIES = ["pullback", "regime_adaptive", "conservative", "aggressive_pullback"]

    # Research winners and common large-cap underlyings
    # Ordered by typical collateral: cheapest first (important for $2K accounts)
    # Expanded default universe: diversified across sectors for reduced
    # survivorship bias. Use validate_universe() to filter by data availability.
    DEFAULT_TICKERS = [
        # Tech
        "AAPL", "MSFT", "GOOG", "META", "NVDA", "AVGO", "CRM", "ORCL",
        # Industrial
        "CAT", "DE", "HON", "UNP",
        # Consumer
        "COST", "HD", "WMT", "MCD",
        # Healthcare
        "JNJ", "UNH", "ABT", "MRK",
        # ETFs
        "QQQ", "IWM",
    ]

    def __init__(
        self,
        account: SimulatedAccount,
        strategy: str = "pullback",
        tickers: list[str] | None = None,
        pipe: DataPipeline | None = None,
        max_contracts: int = 10,
        max_positions: int = 6,
        max_pct_per_position: float = 0.25,
        # Tier 1 enhancements
        enable_earnings_filter: bool = True,
        enable_profit_target: bool = True,
        profit_target_pct: float = 0.50,
        enable_heat_control: bool = True,
        max_portfolio_heat: float = 0.80,
        # Pricing provider for spread quotes (optional -- None = inline B-S for speed)
        pricing_provider=None,
    ):
        self.account = account
        self.strategy_name = strategy
        self.tickers = tickers or self.DEFAULT_TICKERS
        self.pipe = pipe or DataPipeline()
        self.max_contracts = max_contracts
        self.max_positions = max_positions
        self.max_pct = max_pct_per_position
        self.pricing_provider = pricing_provider

        # Tier 1 enhancements
        self.enable_earnings_filter = enable_earnings_filter
        self.enable_profit_target = enable_profit_target
        self.profit_target_pct = profit_target_pct
        self.enable_heat_control = enable_heat_control
        self.max_portfolio_heat = max_portfolio_heat

        # Strategy params — research-validated defaults (7% buffer + loss limit)
        # See experiments/long_portfolio_configs.py for validation on real data
        self.buffer = 0.07
        self.spread_pct = 0.02
        self.dte_open = 30
        self.dte_close = 14
        self.r = 0.05
        self.pullback_threshold = 0.03
        self.pullback_lookback = 20
        self.vol_window = 30

        # Earnings cache
        self._earnings_cache: dict[str, bool] = {}

        # Cache
        self._data: dict[str, pd.DataFrame] = {}
        self._vol: dict[str, pd.Series] = {}

    def _load_data(self, start: str = "2016-01-01", end: str | None = None):
        """Load/refresh price data for all tickers."""
        if end is None:
            end = datetime.now().strftime("%Y-%m-%d")
        for ticker in self.tickers + ["SPY"]:
            if ticker not in self._data:
                self._data[ticker] = self.pipe.fetch_stock(
                    ticker, start=start, end=end
                )
                self._vol[ticker] = historical_volatility(
                    self._data[ticker]["close"], window=self.vol_window
                )

    def _get_price(self, ticker: str, date_ts: int) -> float | None:
        df = self._data.get(ticker)
        if df is None or len(df) == 0:
            return None
        # Use side='right' - 1 to get the last data point <= date_ts,
        # avoiding forward bias on weekends/holidays.
        idx = np.searchsorted(df.index.values, date_ts, side="right") - 1
        if idx < 0:
            return None
        # Only return if the data point is within 3 days
        if abs(df.index[idx] - date_ts) > 3 * 86400:
            return None
        return float(df["close"].iloc[idx])

    def _get_vol(self, ticker: str, date_ts: int) -> float | None:
        vol = self._vol.get(ticker)
        df = self._data.get(ticker)
        if vol is None or df is None:
            return None
        idx = np.searchsorted(df.index.values, date_ts, side="right") - 1
        if idx < 0:
            return None
        idx = min(idx, len(vol) - 1)
        v = vol.iloc[idx]
        return float(v) if not np.isnan(v) and v > 0 else None

    def _get_drawdown_from_high(self, ticker: str, date_ts: int) -> float:
        df = self._data.get(ticker)
        if df is None:
            return 0
        idx = np.searchsorted(df.index.values, date_ts, side="right") - 1
        if idx < 0:
            return 0
        idx = min(idx, len(df) - 1)
        start = max(0, idx - self.pullback_lookback)
        if start >= idx:
            return 0
        window = df["close"].iloc[start : idx + 1]
        high = window.max()
        current = window.iloc[-1]
        return (current - high) / high

    def _get_regime_buffer(self, date_ts: int) -> float:
        spy_vol = self._get_vol("SPY", date_ts)
        if spy_vol is None:
            return self.buffer
        if spy_vol < 0.15:
            return 0.07
        elif spy_vol < 0.25:
            return 0.10
        return 0.13

    def _has_upcoming_earnings(self, ticker: str) -> bool:
        """Check if ticker has earnings within the hold period."""
        if not self.enable_earnings_filter:
            return False
        if ticker in self._earnings_cache:
            return self._earnings_cache[ticker]
        try:
            import yfinance as yf
            t = yf.Ticker(ticker)
            dates = t.earnings_dates
            if dates is None or dates.empty:
                self._earnings_cache[ticker] = False
                return False
            now = pd.Timestamp.now(tz="America/New_York")
            end = now + pd.Timedelta(days=self.dte_open)
            upcoming = dates.index[(dates.index >= now) & (dates.index <= end)]
            result = len(upcoming) > 0
            self._earnings_cache[ticker] = result
            return result
        except Exception:
            self._earnings_cache[ticker] = False
            return False

    def _portfolio_heat(self) -> float:
        """Calculate total portfolio risk as fraction of equity."""
        if not self.account.positions:
            return 0.0
        total_risk = sum(
            (p.short_strike - p.long_strike) * 100 * p.contracts - p.credit_received
            for p in self.account.positions
        )
        return total_risk / self.account.equity if self.account.equity > 0 else 1.0

    def _check_profit_target(self, pos, current_price: float, current_vol: float) -> float | None:
        """Check if a position has hit the profit target.
        Returns close cost if target hit, None otherwise."""
        if not self.enable_profit_target:
            return None

        entry_dt = datetime.fromisoformat(pos.entry_date)
        days_held = (datetime.now() - entry_dt).days
        dte_remaining = max(1, self.dte_open - days_held)

        close_cost = (
            bs_put_price(current_price, pos.short_strike, dte_remaining / 365, self.r, current_vol)
            - bs_put_price(current_price, pos.long_strike, dte_remaining / 365, self.r, current_vol)
        ) * 100 * pos.contracts

        # Profit target: close if we've captured X% of the credit
        unrealized_pnl = pos.credit_received - close_cost
        if unrealized_pnl >= pos.credit_received * self.profit_target_pct:
            return close_cost
        return None

    def _price_spread_close(
        self,
        ticker: str,
        short_strike: float,
        long_strike: float,
        price: float,
        vol: float,
        dte_remaining: int,
        date_str: str,
        expiry: str,
        contracts: int,
    ) -> float | None:
        """Price a spread for closing using provider or inline B-S.

        Returns total close cost (dollar amount for all contracts),
        or None if pricing fails.
        """
        if self.pricing_provider is not None:
            try:
                quote = self.pricing_provider.get_spread_quote(
                    ticker=ticker,
                    short_strike=short_strike,
                    long_strike=long_strike,
                    expiry=expiry,
                    date=date_str,
                    underlying_price=price,
                )
                return quote.net_credit_mid * contracts
            except Exception:
                pass  # fall through to B-S

        return (
            bs_put_price(price, short_strike, dte_remaining / 365, self.r, vol)
            - bs_put_price(price, long_strike, dte_remaining / 365, self.r, vol)
        ) * 100 * contracts

    def _effective_max_positions(self) -> int:
        """Adjust max positions based on portfolio heat and drawdown."""
        if not self.enable_heat_control:
            return self.max_positions

        # Reduce positions during drawdowns
        if self.account.equity_curve:
            peak = max(e.equity for e in self.account.equity_curve)
            dd = (self.account.equity - peak) / peak if peak > 0 else 0
            if dd < -0.10:
                return max(1, self.max_positions // 3)
            elif dd < -0.05:
                return max(2, self.max_positions // 2)

        # Reduce if portfolio heat is high
        heat = self._portfolio_heat()
        if heat >= self.max_portfolio_heat:
            return len(self.account.positions)  # no new positions

        return self.max_positions

    def _get_live_spread_quote(
        self, ticker: str, underlying_price: float, buffer: float, date_str: str,
    ) -> dict | None:
        """Try to get real option chain pricing from yfinance.

        Only works for current/recent dates (yfinance has no historical chains).
        Returns dict with short_strike, long_strike, credit, collateral, expiry
        or None if unavailable.
        """
        from datetime import datetime as _dt, timedelta as _td

        # Only attempt for dates within 3 days of now
        try:
            trade_date = _dt.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            return None
        if abs((_dt.now() - trade_date).days) > 3:
            return None

        try:
            import yfinance as yf
            t = yf.Ticker(ticker)
            expirations = t.options
            if not expirations:
                return None

            # Find expiration nearest to dte_open
            target = trade_date + _td(days=self.dte_open)
            best_exp = min(expirations, key=lambda e: abs(
                (_dt.strptime(e, "%Y-%m-%d") - target).days
            ))

            chain = t.option_chain(best_exp)
            puts = chain.puts
            if puts.empty:
                return None

            # Find short strike near target buffer
            target_short = underlying_price * (1 - buffer)
            puts_sorted = puts.sort_values("strike")
            valid = puts_sorted[puts_sorted["strike"] <= target_short + 0.5]
            if valid.empty:
                return None
            short_row = valid.iloc[-1]  # nearest strike <= target
            sk = float(short_row["strike"])

            # Find long strike: nearest below short by ~spread_pct
            target_long = sk - underlying_price * self.spread_pct
            long_candidates = puts_sorted[puts_sorted["strike"] < sk - 0.5]
            if long_candidates.empty:
                return None
            long_row = long_candidates.iloc[
                (long_candidates["strike"] - target_long).abs().argmin()
            ]
            lk = float(long_row["strike"])

            if lk >= sk:
                return None

            short_bid = float(short_row.get("bid", 0))
            long_ask = float(long_row.get("ask", 0))

            if short_bid <= 0 or long_ask < 0:
                return None

            net_credit = (short_bid - long_ask) * 100  # per contract
            collateral = (sk - lk) * 100

            if net_credit <= 0:
                return None

            return {
                "short_strike": sk,
                "long_strike": lk,
                "credit": net_credit,
                "collateral": collateral,
                "expiry": best_exp,
                "short_bid": short_bid,
                "long_ask": long_ask,
                "source": "yfinance_live",
            }

        except Exception:
            return None

    def _should_enter(self, ticker: str, date_ts: int) -> tuple[bool, float]:
        """Check if we should enter a trade. Returns (should_enter, buffer)."""
        # Earnings filter
        if self._has_upcoming_earnings(ticker):
            return False, self.buffer

        if self.strategy_name == "pullback":
            dd = self._get_drawdown_from_high(ticker, date_ts)
            if dd > -self.pullback_threshold:
                return False, self.buffer
            return True, self.buffer

        elif self.strategy_name == "regime_adaptive":
            buffer = self._get_regime_buffer(date_ts)
            return True, buffer

        else:  # conservative
            return True, self.buffer

    def _days_between(self, date1: str, date2: str) -> int:
        d1 = datetime.fromisoformat(date1)
        d2 = datetime.fromisoformat(date2)
        return (d2 - d1).days

    def _add_calendar_days(self, date_str: str, days: int) -> str:
        dt = datetime.fromisoformat(date_str)
        return (dt + timedelta(days=days)).isoformat()

    def catch_up(self, end_date: str | None = None):
        """Advance the account from last_advanced_date to end_date (or today).

        This is the main entry point. Call it to bring an account up to date.
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        start = self.account.last_advanced_date
        if not start:
            start = "2024-01-01"  # default start

        self._load_data("2016-01-01", end_date)

        # Get all trading days from SPY
        spy = self._data.get("SPY")
        if spy is None or len(spy) == 0:
            print("No SPY data available")
            return

        start_ts = int(pd.Timestamp(start).timestamp())
        end_ts = int(pd.Timestamp(end_date).timestamp())

        trading_days = spy.index[(spy.index >= start_ts) & (spy.index <= end_ts)]

        if len(trading_days) == 0:
            print(f"No trading days between {start} and {end_date}")
            return

        trades_opened = 0
        trades_closed = 0

        for ts in trading_days:
            date_str = pd.Timestamp(ts, unit="s").strftime("%Y-%m-%d")

            # --- CHECK: profit target on open positions ---
            if self.enable_profit_target:
                for pos in list(self.account.positions):
                    price = self._get_price(pos.ticker, ts)
                    vol = self._get_vol(pos.ticker, ts)
                    if price is None or vol is None:
                        continue
                    entry_ts = int(pd.Timestamp(pos.entry_date).timestamp())
                    days_held = max(1, (ts - entry_ts) // 86400)
                    dte_remaining = max(1, self.dte_open - days_held)

                    close_cost = self._price_spread_close(
                        pos.ticker, pos.short_strike, pos.long_strike,
                        price, vol, dte_remaining, date_str,
                        pos.notes, pos.contracts,
                    )
                    if close_cost is None:
                        continue

                    unrealized = pos.credit_received - close_cost
                    if unrealized >= pos.credit_received * self.profit_target_pct:
                        self.account.close_position(
                            pos.id, date_str, price, close_cost, "profit_target"
                        )
                        trades_closed += 1

            # --- CHECK: close positions at their target date ---
            for pos in list(self.account.positions):
                target_ts = int(pd.Timestamp(pos.close_target_date).timestamp())
                if ts >= target_ts:
                    exit_price = self._get_price(pos.ticker, ts)
                    exit_vol = self._get_vol(pos.ticker, ts)
                    if exit_price is None or exit_vol is None:
                        continue

                    close_cost = self._price_spread_close(
                        pos.ticker, pos.short_strike, pos.long_strike,
                        exit_price, exit_vol, self.dte_close, date_str,
                        pos.notes, pos.contracts,
                    )
                    if close_cost is None:
                        continue

                    self.account.close_position(
                        pos.id, date_str, exit_price, close_cost, "checkpoint"
                    )
                    trades_closed += 1

            # --- CHECK: open new positions (with heat control) ---
            effective_max = self._effective_max_positions()
            if len(self.account.positions) < effective_max:
                open_tickers = self.account.open_tickers
                for ticker in self.tickers:
                    if ticker in open_tickers:
                        continue
                    if len(self.account.positions) >= effective_max:
                        break

                    should_enter, buffer = self._should_enter(ticker, ts)
                    if not should_enter:
                        continue

                    price = self._get_price(ticker, ts)
                    vol = self._get_vol(ticker, ts)
                    if price is None or vol is None:
                        continue

                    # Try pricing provider, then live yfinance, then B-S
                    sk = lk = credit = col = 0.0
                    contract_expiry = ""
                    if self.pricing_provider is not None:
                        try:
                            sq = self.pricing_provider.find_spread_strikes(
                                ticker=ticker,
                                date=date_str,
                                buffer=buffer,
                                spread_pct=self.spread_pct,
                                dte_target=self.dte_open,
                                underlying_price=price,
                            )
                            if sq is not None and sq.net_credit > 0:
                                sk = sq.short_strike
                                lk = sq.long_strike
                                credit = sq.net_credit
                                col = sq.spread_width
                                contract_expiry = sq.expiry
                        except Exception:
                            pass  # fall through

                    if credit <= 0:
                        live_quote = self._get_live_spread_quote(ticker, price, buffer, date_str)
                        if live_quote is not None:
                            sk = live_quote["short_strike"]
                            lk = live_quote["long_strike"]
                            credit = live_quote["credit"]
                            col = live_quote["collateral"]
                            contract_expiry = live_quote.get("expiry", "")

                    if credit <= 0:
                        # Fall back to B-S with snapped strikes
                        sk, lk = snap_put_credit_spread(
                            ticker=ticker,
                            underlying_price=price,
                            target_buffer=buffer,
                            target_spread_pct=self.spread_pct,
                        )
                        if lk <= 0:
                            continue
                        sw = sk - lk
                        sp = put_credit_spread_price(
                            price, sk, lk, self.dte_open / 365, self.r, vol
                        )
                        credit = sp["net_credit_dollar"]
                        col = sw * 100

                    if credit <= 0 or col <= 0:
                        continue

                    # Heat-aware sizing
                    max_loss = col - credit  # max loss = collateral - credit received
                    max_alloc = self.account.buying_power * self.max_pct
                    if self.enable_heat_control and max_loss > 0:
                        max_risk = self.account.equity * 0.15
                        max_by_risk = max(1, int(max_risk / max_loss))
                    else:
                        max_by_risk = self.max_contracts
                    contracts = min(
                        self.max_contracts, max_by_risk, max(1, int(max_alloc / col))
                    )

                    close_target = self._add_calendar_days(date_str, self.dte_open - self.dte_close)

                    self.account.open_position(
                        ticker=ticker,
                        date=date_str,
                        entry_price=price,
                        short_strike=sk,
                        long_strike=lk,
                        contracts=contracts,
                        credit_per_contract=credit,
                        collateral_per_contract=col,
                        close_target_date=close_target,
                        buffer=buffer,
                        entry_vol=vol,
                        entry_regime=self.strategy_name,
                        notes=contract_expiry,  # actual option expiry from chain
                    )
                    trades_opened += 1

            # Weekly snapshot
            day_of_week = pd.Timestamp(ts, unit="s").dayofweek
            if day_of_week == 4:  # Friday
                self.account.snapshot(date_str)

        # Final snapshot
        self.account.snapshot(date_str)

        print(
            f"Advanced {self.account.name} to {date_str}: "
            f"{trades_opened} opened, {trades_closed} closed, "
            f"equity ${self.account.equity:,.2f}"
        )

    def mark_to_market(self) -> list[dict]:
        """Price all open positions at latest available market data.

        Returns list of position dicts with current P/L estimates.
        """
        self._load_data()
        mtm = []
        for pos in self.account.positions:
            # Use the latest available data point instead of datetime.now()
            df = self._data.get(pos.ticker)
            if df is None or len(df) == 0:
                continue
            latest_ts = df.index[-1]
            price = float(df['close'].iloc[-1])
            vol = self._get_vol(pos.ticker, latest_ts)
            if price is None or vol is None:
                continue

            # Estimate remaining DTE
            entry_dt = datetime.fromisoformat(pos.entry_date)
            days_held = (datetime.now() - entry_dt).days
            dte_remaining = max(1, self.dte_open - days_held)

            close_cost = (
                bs_put_price(price, pos.short_strike, dte_remaining / 365, self.r, vol)
                - bs_put_price(price, pos.long_strike, dte_remaining / 365, self.r, vol)
            ) * 100 * pos.contracts

            unrealized_pnl = pos.credit_received - close_cost
            price_change = (price - pos.entry_price) / pos.entry_price * 100

            mtm.append({
                "id": pos.id,
                "ticker": pos.ticker,
                "entry_date": pos.entry_date,
                "entry_price": pos.entry_price,
                "current_price": price,
                "price_change": price_change,
                "short_strike": pos.short_strike,
                "buffer_remaining": (price - pos.short_strike) / price * 100,
                "contracts": pos.contracts,
                "credit": pos.credit_received,
                "close_cost": close_cost,
                "unrealized_pnl": unrealized_pnl,
                "dte_remaining": dte_remaining,
                "close_target": pos.close_target_date,
            })
        return mtm
