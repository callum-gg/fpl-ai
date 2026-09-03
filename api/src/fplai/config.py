"""Settings: .env -> Settings. DB-backed settings layer on top lives in settings_store.py.

Precedence (11-config.md): .env -> settings(global) -> settings(squad) -> request override.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

SECRET_RE = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD)", re.IGNORECASE)

# api/src/fplai/config.py -> fplai -> src -> api -> the repo root, which is also /app in
# the container. Relative data paths anchor here rather than on whatever directory the
# process happened to start in.
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _anchored(p: Path) -> Path:
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def _csv(v: str | list[str] | None) -> list[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return v
    return [s.strip() for s in v.split(",") if s.strip()]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", "../.env", "../../.env"), extra="ignore", case_sensitive=False
    )

    # Core
    app_env: str = "local"
    tz: str = "Europe/London"
    bind_host: str = "127.0.0.1"
    api_port: int = 8000
    web_port: int = 5173
    log_level: str = "INFO"
    data_dir: Path = Path("./data")
    database_url: str = "sqlite:///./data/fplai.db"
    current_season: str = "2026-27"

    app_auth_mode: str = "none"
    app_token: str = ""
    allowed_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:5173"]
    )

    # LLM
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_api_key: str = ""
    llm_default_model: str = ""
    llm_timeout_s: int = 120
    llm_max_retries: int = 2
    llm_cache_enabled: bool = True
    llm_alt_base_url: str = ""
    llm_alt_api_key: str = ""

    ollama_enabled: bool = False
    ollama_base_url: str = "http://ollama:11434/v1"
    ollama_api_key: str = "ollama"

    embedding_provider: str = "local"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_device: str = "auto"
    embedding_dim: int = 384

    # FPL
    fpl_entry_ids: Annotated[list[str], NoDecode] = Field(default_factory=list)
    fpl_write_enabled: bool = False
    fpl_email: str = ""
    fpl_password: str = ""
    fpl_user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    # FPL replaced its password login with PingOne SSO, so email/password above can no
    # longer authenticate anything. Paste a refresh token instead — see docs/02.
    fpl_refresh_token: str = ""

    # Key-gated sources
    odds_api_key: str = ""
    odds_api_regions: str = "uk"
    odds_api_markets: str = "h2h,totals"
    betfair_app_key: str = ""
    betfair_username: str = ""
    betfair_password: str = ""
    betfair_cert_path: str = ""
    betfair_key_path: str = ""
    api_football_key: str = ""
    api_football_host: str = "v3.football.api-sports.io"
    sportmonks_api_key: str = ""
    youtube_api_key: str = ""
    supadata_api_key: str = ""
    apify_token: str = ""
    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    reddit_user_agent: str = "fpl-ai/1.0"
    bluesky_handle: str = ""
    bluesky_app_password: str = ""
    football_data_org_key: str = ""

    # X
    x_enabled: bool = True
    x_methods: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["syndication", "nitter", "twscrape"]
    )
    nitter_instances: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["https://nitter.net"]
    )
    twscrape_accounts_file: str = "./data/secrets/twscrape_accounts.json"

    # Scraping
    scrape_enabled: bool = True
    scrape_min_delay_ms: int = 1200
    scrape_max_concurrency: int = 4
    http_proxy_url: str = ""

    # Scheduler
    apscheduler_enabled: bool = True
    ingest_on_startup: bool = False
    deadline_turbo_hours: int = 6

    # Notifications
    discord_webhook_url: str = ""
    discord_digest_hour: int = 8
    notify_price_changes: bool = True
    notify_injury_to_owned: bool = True
    notify_deadline_hours: Annotated[list[int], NoDecode] = Field(default_factory=lambda: [24, 2])

    # Modelling
    sim_iterations: int = 10000
    model_auto_promote: bool = True
    train_season_decay: float = 0.72
    optimiser_solver: str = "CBC"
    optimiser_time_limit_s: int = 60

    @field_validator(
        "allowed_origins", "fpl_entry_ids", "x_methods", "nitter_instances", mode="before"
    )
    @classmethod
    def _split_csv(cls, v):
        return _csv(v)

    @field_validator("notify_deadline_hours", mode="before")
    @classmethod
    def _split_int_csv(cls, v):
        if isinstance(v, str):
            return [int(x) for x in _csv(v)]
        return v

    @field_validator("data_dir", mode="after")
    @classmethod
    def _anchor_data_dir(cls, v: Path) -> Path:
        """Resolve a relative data dir against the repo root, never the shell's cwd.

        `get_settings()` mkdirs this eagerly, so with the default `./data` every process
        that imported fplai from somewhere else quietly grew its own state tree there —
        which is how 394 MB of stray database ended up at `api/src/data/fplai.db` and how
        a run could read a database nobody was writing to.
        """
        return _anchored(v)

    @property
    def db_path(self) -> Path:
        url = self.database_url
        if url.startswith("sqlite:///"):
            raw = Path(url.removeprefix("sqlite:///"))
        elif url.startswith("sqlite://"):
            raw = Path(url.removeprefix("sqlite://"))
        else:
            raw = Path(url)
        return _anchored(raw)

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    def redacted(self) -> dict:
        """Env view safe to send to the UI. 01-architecture.md: never log/serve secrets."""
        out = {}
        for name, value in self.model_dump().items():
            if SECRET_RE.search(name):
                out[name] = "***set***" if value else None
            else:
                out[name] = str(value) if isinstance(value, Path) else value
        return out

    def has_key(self, *names: str) -> bool:
        return all(bool(getattr(self, n, "")) for n in names)


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    for d in (s.data_dir, s.raw_dir, s.models_dir):
        d.mkdir(parents=True, exist_ok=True)
    return s
