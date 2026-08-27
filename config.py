"""Central configuration for the fraud-detection services.

All tunable settings (Kafka address, topic, model/DB paths, dashboard port, and
which transport to use) live here and are read from the environment / a `.env`
file. Modules import :func:`load_settings` instead of hard-coding values, so the
same code runs locally, in Docker, or against a different broker by swapping the
`.env` only.

Run ``copy .env.example .env`` and edit to override defaults.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Settings:
    """Resolved runtime configuration (all paths are absolute)."""

    kafka_bootstrap_servers: str
    kafka_topic: str
    model_path: Path
    explainer_path: Path
    db_path: Path
    dashboard_port: int
    transport: str

    @property
    def kafka_enabled(self) -> bool:
        return self.transport == "kafka"


def _path(value: str, default: Path) -> Path:
    return Path(value).resolve() if value else default


def load_settings() -> Settings:
    """Build :class:`Settings` from environment variables with sane defaults."""
    return Settings(
        kafka_bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
        kafka_topic=os.getenv("KAFKA_TOPIC", "transactions"),
        model_path=_path(os.getenv("MODEL_PATH", ""), ROOT / "consumer" / "model.pkl"),
        explainer_path=_path(os.getenv("EXPLAINER_PATH", ""), ROOT / "consumer" / "explainer.pkl"),
        db_path=_path(os.getenv("DB_PATH", ""), ROOT / "fraud_results.db"),
        dashboard_port=int(os.getenv("DASHBOARD_PORT", "8501")),
        transport=os.getenv("TRANSPORT", "kafka"),
    )
