"""Service layer for simulation accounts."""

from __future__ import annotations

import glob
import os

from tradelab.account import SimulatedAccount
from tradelab.simulator import Simulator

from api.config import settings
from api.schemas.account import (
    AccountSummary,
    AccountDetail,
    PositionDetail,
    TradeDetail,
    MtmPosition,
)


def _account_path(name: str) -> str:
    return os.path.join(settings.accounts_dir, f"{name}.json")


def list_accounts() -> list[AccountSummary]:
    """List all simulation accounts (excludes underscore-prefixed archives)."""
    files = sorted(glob.glob(os.path.join(settings.accounts_dir, "*.json")))
    results = []
    for f in files:
        basename = os.path.basename(f)
        # Skip archived/experimental accounts (underscore prefix)
        if basename.startswith("_"):
            continue
        acct = SimulatedAccount.load(f)
        file_id = os.path.splitext(basename)[0]
        results.append(_to_summary(acct, file_id))
    return results


def get_account(name: str) -> AccountDetail | None:
    """Get full account detail."""
    path = _account_path(name)
    if not os.path.exists(path):
        return None
    acct = SimulatedAccount.load(path)

    positions = [
        PositionDetail(
            id=p.id, ticker=p.ticker, entry_date=p.entry_date,
            entry_price=p.entry_price, short_strike=p.short_strike,
            long_strike=p.long_strike, contracts=p.contracts,
            collateral=p.collateral, credit_received=p.credit_received,
            buffer=p.buffer, close_target_date=p.close_target_date,
            notes=p.notes,
        )
        for p in acct.positions
    ]

    # Return all trades for backtest accounts, last 50 for live
    trade_limit = len(acct.trades) if acct.account_type == "backtest" else 50
    recent = acct.trades[-trade_limit:] if acct.trades else []
    trades = [
        TradeDetail(
            id=t.id, ticker=t.ticker, entry_date=t.entry_date,
            exit_date=t.exit_date, entry_price=t.entry_price,
            exit_price=t.exit_price, contracts=t.contracts,
            pnl=t.pnl, winner=t.winner, exit_reason=t.exit_reason,
            buffer=t.buffer,
        )
        for t in recent
    ]

    equity_curve = [
        {"date": e.date, "equity": e.equity, "positions": e.open_positions}
        for e in acct.equity_curve
    ]

    # Build config summary
    config_summary = _build_config_summary(acct)

    return AccountDetail(
        summary=_to_summary(acct),
        positions=positions,
        recent_trades=trades,
        equity_curve=equity_curve,
        config_summary=config_summary,
    )


def mark_to_market(name: str) -> list[MtmPosition]:
    """Mark all open positions to current market."""
    path = _account_path(name)
    if not os.path.exists(path):
        return []
    acct = SimulatedAccount.load(path)
    sim = Simulator(acct, strategy=acct.strategy or "pullback")
    mtm_data = sim.mark_to_market()
    return [MtmPosition(**m) for m in mtm_data]


def advance_account(name: str, end_date: str | None = None) -> AccountSummary | None:
    """Advance an account to a date (or today)."""
    path = _account_path(name)
    if not os.path.exists(path):
        return None
    acct = SimulatedAccount.load(path)
    sim = Simulator(acct, strategy=acct.strategy or "pullback")
    sim.catch_up(end_date)
    return _to_summary(acct)


def _build_config_summary(acct: SimulatedAccount) -> str:
    """Build a human-readable config description for export."""
    trades = acct.trades
    winners = sum(1 for t in trades if t.winner)
    losers = len(trades) - winners
    avg_winner = sum(t.pnl for t in trades if t.winner) / winners if winners else 0
    avg_loser = sum(t.pnl for t in trades if not t.winner) / losers if losers else 0

    # Determine tickers traded
    tickers = sorted(set(t.ticker for t in trades)) if trades else []

    # Date range
    if trades:
        first_date = min(t.entry_date[:10] for t in trades)
        last_date = max(t.exit_date[:10] for t in trades)
    elif acct.equity_curve:
        first_date = acct.equity_curve[0].date[:10]
        last_date = acct.equity_curve[-1].date[:10]
    else:
        first_date = last_date = acct.last_advanced_date[:10] if acct.last_advanced_date else "N/A"

    # Infer buffer from trades
    buffers = set()
    for t in trades:
        if t.entry_price > 0 and t.short_strike > 0:
            buf = (t.entry_price - t.short_strike) / t.entry_price
            buffers.add(round(buf * 100))
    buffer_str = "/".join(f"{b}%" for b in sorted(buffers)) if buffers else "N/A"

    total_return_pct = acct.total_pnl / acct.starting_capital * 100 if acct.starting_capital > 0 else 0
    pricing_label = "live yfinance option chains" if acct.account_type == "live" else "Theta Data EOD option chains"

    # Build narrative paragraph
    ticker_count = len(tickers)
    ticker_list_str = ", ".join(tickers[:5])
    if ticker_count > 5:
        ticker_list_str += f", and {ticker_count - 5} more"
    elif ticker_count == 0:
        ticker_list_str = "no tickers"

    profit_loss_word = "profit" if acct.total_pnl >= 0 else "loss"
    narrative = (
        f"This {acct.account_type} account runs the {acct.strategy} strategy, "
        f"selling put credit spreads on {ticker_list_str}. "
        f"Entries trigger when a stock pulls back 3% from its 20-day high, "
        f"targeting {buffer_str} out-of-the-money buffer between the underlying "
        f"price and the short strike. "
        f"Spreads open around 30 DTE and close at 14 DTE to capture the steepest "
        f"part of theta decay while limiting gamma exposure. "
        f"Over the period {first_date} to {last_date}, the account placed "
        f"{len(trades)} trades with a {acct.win_rate:.0%} win rate, generating "
        f"a net {profit_loss_word} of ${abs(acct.total_pnl):,.2f} "
        f"({total_return_pct:+.1f}% return on ${acct.starting_capital:,.0f} starting capital). "
        f"Winners averaged ${avg_winner:+,.2f} while losers averaged ${avg_loser:+,.2f}. "
        f"Pricing is sourced from {pricing_label} with $0.65/contract/leg commission "
        f"and 2% bid-ask slippage applied."
    )

    lines = [
        f"Account: {acct.name}",
        f"Type: {acct.account_type}",
        f"Strategy: {acct.strategy}",
        f"Period: {first_date} to {last_date}",
        f"Starting capital: ${acct.starting_capital:,.0f}",
        f"Final equity: ${acct.equity:,.2f}",
        f"Total return: {total_return_pct:+.1f}%",
        f"",
        f"Trades: {len(trades)} ({winners}W / {losers}L)",
        f"Win rate: {acct.win_rate:.1%}",
        f"Avg winner: ${avg_winner:+,.2f}",
        f"Avg loser: ${avg_loser:+,.2f}",
        f"",
        f"Buffer: {buffer_str}",
        f"Spread type: Put credit spread",
        f"DTE: 30 open / 14 close",
        f"Entry: Pullback (3% from 20-day high)",
        f"Tickers: {', '.join(tickers)}",
        f"",
        f"Data source: Theta Data Standard (real historical bid/ask)",
        f"Pricing: {'Live yfinance chains' if acct.account_type == 'live' else 'Theta Data EOD option chains'}",
        f"Friction: $0.65/contract/leg + 2% bid-ask slippage",
        f"Generated: {acct.created_date[:19] if acct.created_date else 'N/A'}",
        f"",
        f"---",
        f"",
        narrative,
    ]
    return "\n".join(lines)


def _start_date(acct: SimulatedAccount) -> str:
    """Derive the effective start date of an account."""
    if acct.equity_curve:
        return acct.equity_curve[0].date[:10]
    if acct.trades:
        return min(t.entry_date[:10] for t in acct.trades)
    return acct.created_date[:10] if acct.created_date else ""


def _to_summary(acct: SimulatedAccount, file_id: str | None = None) -> AccountSummary:
    return AccountSummary(
        name=file_id or acct.name,
        strategy=acct.strategy,
        account_type=acct.account_type,
        starting_capital=acct.starting_capital,
        equity=acct.equity,
        balance=acct.balance,
        locked=acct.locked,
        total_pnl=acct.total_pnl,
        total_return_pct=acct.total_pnl / acct.starting_capital * 100 if acct.starting_capital > 0 else 0,
        total_trades=acct.total_trades_count,
        win_rate=acct.win_rate,
        open_positions=len(acct.positions),
        last_advanced_date=acct.last_advanced_date,
        created_date=_start_date(acct),
    )
