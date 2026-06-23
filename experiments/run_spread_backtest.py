"""Example: backtest rolling put credit spreads on SPY.

Usage:
    export FINNHUB_API_KEY=your_key_here
    python -m examples.run_spread_backtest
"""

from tradelab.data import StockDataFetcher
from tradelab.analysis import ProbabilityEngine
from tradelab.strategies.credit_spreads import RollingCreditSpreadStrategy


def main():
    ticker = "SPY"
    start = "15/01/2015"
    end = "02/10/2023"

    # --- Fetch data ---
    fetcher = StockDataFetcher()
    df = fetcher.fetch(ticker, start, end, columns=["c"])
    print(f"Loaded {len(df)} trading days for {ticker}\n")

    # --- Probability analysis ---
    engine = ProbabilityEngine(df)
    print(engine.summary(ticker, offset=260))
    print()

    table = engine.strike_probability_table(offset=260, adjust_start=0.90, adjust_end=1.05)
    print(table.to_string(index=False))
    print()

    # --- Run rolling spread strategy ---
    strategy = RollingCreditSpreadStrategy(
        max_spreads=4,
        offset_days=274,
        buffer=0.019,
        credit_ratio=0.20,
        loss_ratio=0.95,
    )
    result = strategy.run(df, initial_balance=1_000)
    print(result.summary())


if __name__ == "__main__":
    main()
