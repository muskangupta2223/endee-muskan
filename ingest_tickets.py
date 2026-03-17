"""
Data ingestion script — run this once before starting the API server.

Reads cleaned_tickets.csv, generates embeddings for each ticket description,
and upserts all vectors into the Endee vector index.

Usage:
    python ingest_tickets.py
"""

import logging
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from backend.embedder import embed_batch
from backend.endee_client import INDEX_NAME, DIMENSION, check_connection, create_index, insert_batch, invalidate_cache

CSV_PATH   = "data/cleaned_tickets.csv"
BATCH_SIZE = 500  # Number of tickets to embed and insert per iteration (SDK max is 1000)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def load_csv(path: str) -> pd.DataFrame:
    """
    Load and validate the ticket CSV file.
    Drops rows where description, team, or resolution is missing
    since incomplete records can't be used for RAG retrieval.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {path}\n  → Place cleaned_tickets.csv in the data/ directory.")

    df = pd.read_csv(path)

    # Validate that all required columns are present
    required = ["ticket_id", "description", "team", "resolution"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    before = len(df)
    df = df.dropna(subset=["description", "team", "resolution"])
    if before - len(df):
        logger.warning(f"Dropped {before - len(df)} rows with missing data")

    logger.info(f"Loaded {len(df):,} valid tickets")
    return df


def ingest(df: pd.DataFrame) -> tuple[int, int]:
    """
    Embed and insert all tickets into Endee in batches.

    For each batch:
      1. Generate embeddings for all descriptions in one call (faster than one-by-one)
      2. Build the vector list with id, embedding, and metadata
      3. Insert the batch via msgpack POST to Endee

    Returns (inserted_count, failed_count).
    """
    inserted, failed = 0, 0

    with tqdm(total=len(df), unit="ticket", desc="Ingesting") as bar:
        for start in range(0, len(df), BATCH_SIZE):
            chunk = df.iloc[start: start + BATCH_SIZE]
            try:
                # Embed the entire chunk at once — much faster than per-row encoding
                embeddings = embed_batch(chunk["description"].tolist(), normalize=True)

                vectors = [
                    {
                        "id":       str(row["ticket_id"]),
                        "values":   emb,
                        "metadata": {
                            "team":       str(row["team"]),
                            "resolution": str(row["resolution"])
                        },
                    }
                    for (_, row), emb in zip(chunk.iterrows(), embeddings)
                ]

                insert_batch(vectors)
                inserted += len(vectors)

            except Exception as e:
                # Log the error and continue — a single bad batch shouldn't stop everything
                logger.error(f"Batch {start}–{start + len(chunk)} failed: {e}")
                failed += len(chunk)

            bar.update(len(chunk))

    return inserted, failed


def main():
    print("\n╔══════════════════════════════════════════════╗")
    print("║   SmartSupport AI  —  Data Ingestion         ║")
    print("╚══════════════════════════════════════════════╝")

    # 1. Make sure Endee is running before we start
    print("\n[1/4]  Checking Endee...")
    if not check_connection():
        print("  ❌  Endee not running. Start it:")
        print("       cd ~/endee && export NDD_DATA_DIR=$(pwd)/data && ./build/ndd-avx2")
        sys.exit(1)
    print("  ✅  Endee running at localhost:8080")

    # 2. Load and validate the CSV
    print("\n[2/4]  Loading CSV...")
    try:
        df = load_csv(CSV_PATH)
        print(f"  ✅  {len(df):,} tickets ready")
    except Exception as e:
        print(f"  ❌  {e}")
        sys.exit(1)

    # 3. Create the vector index (idempotent — safe to run if it already exists)
    print("\n[3/4]  Creating index...")
    try:
        print(f"  ✅  {create_index(INDEX_NAME, DIMENSION, 'cosine')}")
        invalidate_cache()
    except Exception as e:
        print(f"  ❌  {e}")
        sys.exit(1)

    # 4. Run the main ingestion loop
    print(f"\n[4/4]  Ingesting {len(df):,} tickets (batch={BATCH_SIZE})...\n")
    try:
        inserted, failed = ingest(df)
    except KeyboardInterrupt:
        print("\n  ⚠️  Interrupted")
        sys.exit(1)

    print(f"\n╔══════════════════════════════════════════════╗")
    print(f"║  ✅  Inserted : {inserted:>6,} tickets               ║")
    if failed:
        print(f"║  ⚠️  Failed   : {failed:>6,} tickets               ║")
    print(f"╚══════════════════════════════════════════════╝")

    if inserted == 0:
        print("\n❌  No vectors inserted — check the errors above.")
        sys.exit(1)

    print("\n🎉  Done! Start the API server:\n     uvicorn backend.main:app --reload\n")


if __name__ == "__main__":
    main()