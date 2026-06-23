"""Example: backtest a simple momentum signal on crypto.

Usage:
    export FINNHUB_API_KEY=your_key_here
    python -m examples.run_crypto_backtest
"""

import pandas as pd

from tradelab.data import CryptoDataFetcher
from tradelab.models import Position
from tradelab.backtester import Backtester


def sma_crossover_signal(index, row, current_pos):
    """Go long when fast SMA > slow SMA, else flat."""
    if row["sma_fast"] > row["sma_slow"]:
        return Position.LONG
    return None


def main():
    symbol = "BINANCE:BTCUSDT"
    start = "01/01/2021"
    end = "01/01/2023"

    # --- Fetch data ---
    fetcher = CryptoDataFetcher()
    df = fetcher.fetch(symbol, start, end, resolution="60")
    print(f"Loaded {len(df)} candles for {symbol}\n")

    # --- Add indicators ---
    df["sma_fast"] = df["c"].rolling(window=24).mean()   # 24h
    df["sma_slow"] = df["c"].rolling(window=168).mean()  # 7d
    df = df.dropna()

    # --- Backtest ---
    bt = Backtester(capital=10_000, leverage=2, allow_short=False)
    result = bt.run(df, price_col="c", signal_fn=sma_crossover_signal)

    print(result.summary())


if __name__ == "__main__":
    main()
