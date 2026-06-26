"""Populate the Qdrant vector store from a Steam games CSV.

The ``game_description`` column is embedded into a dense vector; every other
column is stored verbatim as point payload (metadata). The embedded description
is also kept in the payload so it can be shown alongside search results.

Run as a module from the project root, e.g.::

    python -m etl.vector_db.ingest_games --csv data/steam_games.csv --recreate

Heavy dependencies (``sentence-transformers`` / ``torch``) are imported lazily
so the module can be imported without them being installed.
"""

from __future__ import annotations

import argparse
import logging
import math
import uuid
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)
from tqdm import tqdm

from etl.vector_db.config import VectorDbConfig, load_config

logger = logging.getLogger("etl.vector_db.ingest")

DEFAULT_CSV_PATH = Path("data/steam_games.csv")
DEFAULT_DESCRIPTION_COLUMN = "detailed_description"
# Columns checked (case-insensitively) when no id column is supplied. rawg_id is
# preferred because it is the only fully-populated, unique key in the dataset
# (steam_appid has nulls and duplicates).
ID_COLUMN_CANDIDATES = ("rawg_id", "app_id", "appid", "steam_appid", "id")
# Stable namespace so string ids map to the same UUID across runs.
ID_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00cf4fc964ff")
# Columns whose cells hold multiple values joined by a separator. They are stored
# as arrays so each value is individually filterable (Qdrant matches any element).
MULTI_VALUE_COLUMNS = ("genres", "tags")
MULTI_VALUE_SEPARATOR = ";"
# Payload fields to index so metadata filters stay fast as the dataset grows.
PAYLOAD_INDEXES: dict[str, PayloadSchemaType] = {
    "tags": PayloadSchemaType.KEYWORD,
    "genres": PayloadSchemaType.KEYWORD,
    "genre": PayloadSchemaType.KEYWORD,
    "developer": PayloadSchemaType.KEYWORD,
    "publisher": PayloadSchemaType.KEYWORD,
    "is_free": PayloadSchemaType.BOOL,
    "price_usd": PayloadSchemaType.FLOAT,
    "rating": PayloadSchemaType.FLOAT,
    "metacritic": PayloadSchemaType.FLOAT,
}


class IngestionError(RuntimeError):
    """Raised when the ingestion pipeline cannot proceed."""


def resolve_device(configured_device: str) -> str:
    """Return the torch device to embed on.

    An explicit ``configured_device`` wins; otherwise pick ``cuda`` when a GPU
    is available and fall back to ``cpu``.
    """
    if configured_device:
        return configured_device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def load_games(csv_path: Path, description_column: str) -> pd.DataFrame:
    """Load the games CSV and validate the description column exists.

    Rows with a missing/blank description are dropped (they cannot be embedded).
    """
    if not csv_path.is_file():
        raise IngestionError(f"CSV file not found: {csv_path}")

    dataframe = pd.read_csv(csv_path)
    if description_column not in dataframe.columns:
        raise IngestionError(
            f"Column {description_column!r} not found in {csv_path}. "
            f"Available columns: {list(dataframe.columns)}"
        )

    total_rows = len(dataframe)
    descriptions = dataframe[description_column].astype("string").str.strip()
    dataframe = dataframe[descriptions.notna() & (descriptions != "")]
    dropped = total_rows - len(dataframe)
    if dropped:
        logger.warning("Skipped %d row(s) with an empty %s", dropped, description_column)
    if dataframe.empty:
        raise IngestionError(f"No rows with a non-empty {description_column!r} to ingest")

    return dataframe.reset_index(drop=True)


def _normalize_value(value: Any) -> Any:
    """Convert a cell into a JSON-serializable payload value.

    Handles pandas/numpy NaN (-> ``None``) and numpy scalar types (-> native).
    """
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        # Array-like values are not NaN-checkable; fall through and keep them.
        pass
    # numpy scalars expose ``.item()`` to return the native Python equivalent.
    if hasattr(value, "item"):
        return value.item()
    return value


def _split_multi_value(value: Any) -> list[str] | None:
    """Split a separator-joined cell into a list of trimmed, non-empty values.

    Returns ``None`` when the cell is empty/NaN so the field is simply absent
    rather than an empty list.
    """
    normalized = _normalize_value(value)
    if not isinstance(normalized, str):
        return None
    items = [part.strip() for part in normalized.split(MULTI_VALUE_SEPARATOR)]
    items = [part for part in items if part]
    return items or None


def build_payloads(dataframe: pd.DataFrame) -> list[dict[str, Any]]:
    """Turn each row into a payload dict with JSON-safe values.

    Columns in :data:`MULTI_VALUE_COLUMNS` are stored as arrays; all others are
    stored as scalar values.
    """
    payloads: list[dict[str, Any]] = []
    for _, row in dataframe.iterrows():
        payload: dict[str, Any] = {}
        for column in dataframe.columns:
            if column in MULTI_VALUE_COLUMNS:
                payload[column] = _split_multi_value(row[column])
            else:
                payload[column] = _normalize_value(row[column])
        payloads.append(payload)
    return payloads


def build_point_ids(dataframe: pd.DataFrame, id_column: str | None) -> list[Any]:
    """Resolve a stable Qdrant point id for every row.

    Priority: explicit ``id_column`` -> auto-detected id column -> row index.
    Integer ids are used directly; non-integer ids become deterministic UUIDs.
    """
    resolved_column = id_column
    if resolved_column is None:
        lower_to_actual = {name.lower(): name for name in dataframe.columns}
        for candidate in ID_COLUMN_CANDIDATES:
            if candidate in lower_to_actual:
                resolved_column = lower_to_actual[candidate]
                break

    if resolved_column is None:
        logger.info("No id column found; using sequential row indices as ids")
        return list(range(len(dataframe)))

    if resolved_column not in dataframe.columns:
        raise IngestionError(f"Requested id column {resolved_column!r} not in CSV")

    logger.info("Using %r as the point id column", resolved_column)
    numeric_ids = pd.to_numeric(dataframe[resolved_column], errors="coerce")
    if numeric_ids.notna().all() and (numeric_ids % 1 == 0).all():
        return [int(value) for value in numeric_ids]

    return [
        str(uuid.uuid5(ID_NAMESPACE, str(value)))
        for value in dataframe[resolved_column]
    ]


def ensure_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
    recreate: bool,
) -> None:
    """Create the collection if needed (or recreate it when requested)."""
    exists = client.collection_exists(collection_name)
    if exists and recreate:
        logger.info("Dropping existing collection %r", collection_name)
        client.delete_collection(collection_name)
        exists = False

    if not exists:
        logger.info(
            "Creating collection %r (size=%d, distance=cosine)",
            collection_name,
            vector_size,
        )
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


def create_payload_indexes(client: QdrantClient, collection_name: str) -> None:
    """Create payload indexes for the fields commonly used in metadata filters.

    Re-running is safe: Qdrant treats creating an existing index as a no-op.
    """
    for field_name, schema_type in PAYLOAD_INDEXES.items():
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=schema_type,
        )
        logger.info("Indexed payload field %r (%s)", field_name, schema_type.value)


def _chunk_indices(total: int, chunk_size: int) -> Iterator[tuple[int, int]]:
    """Yield ``(start, end)`` row ranges of at most ``chunk_size`` items."""
    for start in range(0, total, chunk_size):
        yield start, min(start + chunk_size, total)


def run_ingestion(
    config: VectorDbConfig,
    csv_path: Path,
    description_column: str,
    id_column: str | None,
    recreate: bool,
    limit: int | None,
) -> int:
    """Embed descriptions and upsert all rows into Qdrant.

    Returns the number of points written.
    """
    from sentence_transformers import SentenceTransformer

    dataframe = load_games(csv_path, description_column)
    if limit is not None:
        dataframe = dataframe.head(limit).reset_index(drop=True)
    logger.info("Loaded %d game(s) from %s", len(dataframe), csv_path)

    device = resolve_device(config.embedding_device)
    logger.info("Loading embedding model %r on %s", config.embedding_model, device)
    model = SentenceTransformer(config.embedding_model, device=device)
    vector_size = model.get_embedding_dimension()

    payloads = build_payloads(dataframe)
    point_ids = build_point_ids(dataframe, id_column)
    descriptions = dataframe[description_column].astype(str).tolist()

    client = QdrantClient(url=config.qdrant_url, api_key=config.qdrant_api_key)
    ensure_collection(client, config.collection_name, vector_size, recreate)
    create_payload_indexes(client, config.collection_name)

    total = len(dataframe)
    written = 0
    with tqdm(total=total, desc="Embedding + upserting", unit="game") as progress:
        for start, end in _chunk_indices(total, config.upsert_batch_size):
            vectors = model.encode(
                descriptions[start:end],
                batch_size=config.embedding_batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            points = [
                PointStruct(
                    id=point_ids[index],
                    vector=vectors[index - start].tolist(),
                    payload=payloads[index],
                )
                for index in range(start, end)
            ]
            client.upsert(collection_name=config.collection_name, points=points)
            written += len(points)
            progress.update(len(points))

    return written


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the ingestion script."""
    parser = argparse.ArgumentParser(
        description="Populate the Qdrant vector store from a Steam games CSV.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help=f"Path to the games CSV (default: {DEFAULT_CSV_PATH}).",
    )
    parser.add_argument(
        "--description-column",
        default=DEFAULT_DESCRIPTION_COLUMN,
        help=f"Column to embed (default: {DEFAULT_DESCRIPTION_COLUMN}).",
    )
    parser.add_argument(
        "--id-column",
        default=None,
        help="Column to use as point id (default: auto-detect, else row index).",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate the collection before ingesting.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only ingest the first N rows (useful for testing).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # The HTTP client logs every request at INFO; keep ingestion output readable.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    args = parse_args(argv)
    try:
        config = load_config()
        written = run_ingestion(
            config=config,
            csv_path=args.csv,
            description_column=args.description_column,
            id_column=args.id_column,
            recreate=args.recreate,
            limit=args.limit,
        )
    except IngestionError as error:
        logger.error("Ingestion failed: %s", error)
        return 1
    logger.info("Done. Wrote %d points to %r", written, config.collection_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
