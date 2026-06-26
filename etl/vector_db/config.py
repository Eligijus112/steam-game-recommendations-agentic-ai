"""Configuration for the vector store ETL.

All values are sourced from environment variables (loaded from a local ``.env``
file when present) using the ``GAMEREC_`` prefix, so the same settings drive
both the ingestion scripts and any downstream services.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


class ConfigError(ValueError):
    """Raised when the environment configuration is missing or invalid."""


def _get_int(name: str, default: int) -> int:
    """Read a positive integer env var, falling back to ``default``."""
    raw_value = os.environ.get(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    try:
        value = int(raw_value)
    except ValueError as error:
        raise ConfigError(f"{name} must be an integer, got {raw_value!r}") from error
    if value <= 0:
        raise ConfigError(f"{name} must be a positive integer, got {value}")
    return value


@dataclass(frozen=True)
class VectorDbConfig:
    """Connection and embedding settings for populating the vector store."""

    qdrant_url: str
    qdrant_api_key: str | None
    collection_name: str
    embedding_model: str
    # Empty string means "auto-detect" (cuda when available, otherwise cpu).
    embedding_device: str
    embedding_batch_size: int
    upsert_batch_size: int


def load_config() -> VectorDbConfig:
    """Build a :class:`VectorDbConfig` from the current environment.

    Loads ``.env`` (without overriding already-exported variables) and validates
    the required values. Raises :class:`ConfigError` on missing/invalid input.
    """
    load_dotenv(override=False)

    qdrant_url = os.environ.get("GAMEREC_QDRANT_URL", "").strip()
    if not qdrant_url:
        raise ConfigError("GAMEREC_QDRANT_URL is required (e.g. http://localhost:6333)")

    collection_name = os.environ.get("GAMEREC_COLLECTION_NAME", "").strip()
    if not collection_name:
        raise ConfigError("GAMEREC_COLLECTION_NAME is required")

    embedding_model = os.environ.get("GAMEREC_EMBEDDING_MODEL", "").strip()
    if not embedding_model:
        raise ConfigError("GAMEREC_EMBEDDING_MODEL is required")

    api_key = os.environ.get("GAMEREC_QDRANT_API_KEY", "").strip() or None

    return VectorDbConfig(
        qdrant_url=qdrant_url,
        qdrant_api_key=api_key,
        collection_name=collection_name,
        embedding_model=embedding_model,
        embedding_device=os.environ.get("GAMEREC_EMBEDDING_DEVICE", "").strip(),
        embedding_batch_size=_get_int("GAMEREC_EMBEDDING_BATCH_SIZE", 256),
        upsert_batch_size=_get_int("GAMEREC_UPSERT_BATCH_SIZE", 512),
    )
