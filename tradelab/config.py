import os


class Config:
    """Central configuration loaded from environment variables."""

    FINNHUB_API_KEY: str = os.environ.get("FINNHUB_API_KEY", "")
    TRADING_DAYS_PER_YEAR: int = 252
    CALENDAR_DAYS_PER_YEAR: int = 365

    @classmethod
    def require_finnhub_key(cls) -> str:
        if not cls.FINNHUB_API_KEY:
            raise EnvironmentError(
                "FINNHUB_API_KEY is not set. "
                "Export it or add it to a .env file."
            )
        return cls.FINNHUB_API_KEY
