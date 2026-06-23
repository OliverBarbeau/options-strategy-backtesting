"""Accounting invariant tests for SimulatedAccount.

The most important invariant: after a sequence of open/close operations,
the sum of trade P/L must equal (equity - starting_capital). If these don't
match, there's a double-count bug somewhere.

This test class was added after discovering a credit-double-count bug in
close_position where the balance mutation re-added the credit that was
already added at open time.
"""

from __future__ import annotations

import tempfile

import pytest

from tradelab.account import SimulatedAccount


@pytest.fixture
def account(tmp_path):
    path = tmp_path / "test_invariants.json"
    return SimulatedAccount.load_or_create(
        str(path), starting_capital=10000.0, name="test_invariants"
    )


class TestAccountingInvariant:
    """Sum of trade P/L must equal equity change."""

    def test_single_winning_trade(self, account):
        # Open a put credit spread: sell $95, buy $90, receive $1/share credit
        # Collateral = $5 width * 100 = $500, credit = $100
        account.open_position(
            ticker="TEST",
            date="2024-01-01",
            entry_price=100.0,
            short_strike=95.0,
            long_strike=90.0,
            contracts=1,
            credit_per_contract=100.0,
            collateral_per_contract=500.0,
            close_target_date="2024-01-15",
        )

        # Close for $30 close cost (winner: keeps $70 minus commissions)
        pos = account.positions[0]
        account.close_position(
            pos_id=pos.id,
            date="2024-01-15",
            exit_price=100.0,
            close_cost=30.0,
        )

        trade_pnl = account.trades[0].pnl
        equity_change = account.equity - 10000.0
        assert trade_pnl == pytest.approx(equity_change, abs=0.01), (
            f"Trade P/L {trade_pnl} should match equity change {equity_change}"
        )

    def test_single_losing_trade(self, account):
        account.open_position(
            ticker="TEST",
            date="2024-01-01",
            entry_price=100.0,
            short_strike=95.0,
            long_strike=90.0,
            contracts=1,
            credit_per_contract=100.0,
            collateral_per_contract=500.0,
            close_target_date="2024-01-15",
        )

        # Close for $400 -- big loser
        pos = account.positions[0]
        account.close_position(
            pos_id=pos.id,
            date="2024-01-15",
            exit_price=85.0,
            close_cost=400.0,
        )

        trade_pnl = account.trades[0].pnl
        equity_change = account.equity - 10000.0
        assert trade_pnl == pytest.approx(equity_change, abs=0.01)
        assert trade_pnl < 0  # loser

    def test_multiple_trades_sum_equals_equity_change(self, account):
        trades_to_run = [
            # (short_strike, long_strike, credit, collateral, close_cost)
            (95, 90, 80, 500, 20),   # +$60 gross
            (100, 95, 90, 500, 200), # -$110 gross
            (110, 105, 50, 500, 10), # +$40 gross
            (120, 115, 70, 500, 100),# -$30 gross
        ]

        for i, (sk, lk, credit, col, close_cost) in enumerate(trades_to_run):
            account.open_position(
                ticker="TEST",
                date=f"2024-01-{i*3+1:02d}",
                entry_price=sk + 5,
                short_strike=sk,
                long_strike=lk,
                contracts=1,
                credit_per_contract=credit,
                collateral_per_contract=col,
                close_target_date=f"2024-01-{i*3+2:02d}",
            )
            pos = account.positions[-1]
            account.close_position(
                pos_id=pos.id,
                date=f"2024-01-{i*3+2:02d}",
                exit_price=sk + 5,
                close_cost=close_cost,
            )

        sum_pnl = sum(t.pnl for t in account.trades)
        equity_change = account.equity - 10000.0
        assert sum_pnl == pytest.approx(equity_change, abs=0.01), (
            f"Sum of 4 trade P/Ls ({sum_pnl}) should equal "
            f"equity change ({equity_change})"
        )
        assert account.locked == 0
        assert len(account.positions) == 0

    def test_multiple_contracts(self, account):
        account.open_position(
            ticker="TEST",
            date="2024-01-01",
            entry_price=100.0,
            short_strike=95.0,
            long_strike=90.0,
            contracts=5,  # 5 contracts
            credit_per_contract=100.0,  # $100 each
            collateral_per_contract=500.0,  # $500 each
            close_target_date="2024-01-15",
        )

        assert account.locked == 2500.0  # 5 * $500

        pos = account.positions[0]
        account.close_position(
            pos_id=pos.id,
            date="2024-01-15",
            exit_price=100.0,
            close_cost=150.0,  # 5 * $30 intrinsic
        )

        trade_pnl = account.trades[0].pnl
        equity_change = account.equity - 10000.0
        assert trade_pnl == pytest.approx(equity_change, abs=0.01)
        assert account.locked == 0

    def test_balance_is_not_inflated_during_open(self, account):
        """After opening a position, balance should decrease by collateral
        but increase by net_credit. Net effect: balance = starting - col + net_credit."""
        starting = account.balance
        account.open_position(
            ticker="TEST",
            date="2024-01-01",
            entry_price=100.0,
            short_strike=95.0,
            long_strike=90.0,
            contracts=1,
            credit_per_contract=100.0,
            collateral_per_contract=500.0,
            close_target_date="2024-01-15",
        )

        # Account charges 2% slippage by default
        expected_net_credit = 100.0 * (1 - 0.02)
        expected_balance = starting - 500.0 + expected_net_credit
        assert account.balance == pytest.approx(expected_balance, abs=0.01)
        assert account.locked == 500.0

    def test_equity_increases_by_net_credit_at_open(self, account):
        """Opening a position adds net credit to balance immediately.

        This is because the credit IS cash in hand once received. The slippage
        cost only materializes on the full round-trip when the position is
        closed and the close_cost is paid out of balance.

        At open:
          balance: -= collateral, += net_credit (credit - slippage)
          locked:  += collateral
          equity:  += net_credit (temporarily "ahead" by this amount)
        """
        starting_equity = account.equity
        account.open_position(
            ticker="TEST",
            date="2024-01-01",
            entry_price=100.0,
            short_strike=95.0,
            long_strike=90.0,
            contracts=1,
            credit_per_contract=100.0,  # gross
            collateral_per_contract=500.0,
            close_target_date="2024-01-15",
        )
        # Net credit after 2% slippage = 98
        expected_net_credit = 100.0 * 0.98
        assert account.equity == pytest.approx(
            starting_equity + expected_net_credit, abs=0.01
        )
        assert account.locked == 500.0
