"""FastAPI configuration via environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "TradeLab API"
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    accounts_dir: str = "accounts"
    cache_dir: str = "data/cache"

    model_config = {"env_prefix": "TRADELAB_"}


settings = Settings()
