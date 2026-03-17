"""
Endee vector database client — direct HTTP implementation.

We bypass the official endee Python SDK because it pins pydantic==1.x,
which conflicts with FastAPI's requirement for pydantic v2.
Instead, we replicate the exact same HTTP calls by reading the SDK source.

Confirmed wire formats (from endee==0.1.10 source):
  CREATE  POST /api/v1/index/create
          JSON body: {"index_name", "dim", "space_type", "M", "ef_con",
                      "checksum": -1, "precision": "int8d", "version": 1}

  INSERT  POST /api/v1/index/{name}/vector/insert
          Content-Type: application/msgpack
          Each vector packed as: [id, zlib(orjson(meta)), filter_json, norm, vector]

  SEARCH  POST /api/v1/index/{name}/search
          JSON body: {"k", "ef", "vector", "include_vectors"}
          Response: msgpack list of [similarity, id, meta_bytes, filter_str, norm, ...]

  INFO    GET  /api/v1/index/{name}/info
  LIST    GET  /api/v1/index/list
"""

import logging
import zlib
import math
from typing import List, Dict, Optional

import msgpack
import orjson
import requests
from requests.adapters import HTTPAdapter, Retry

logger = logging.getLogger(__name__)

# --- Configuration ---
INDEX_NAME = "tickets"
DIMENSION  = 384
BASE_URL   = "http://127.0.0.1:8080/api/v1"

# HNSW algorithm defaults (taken directly from SDK constants.py)
DEFAULT_M      = 16    # Number of bi-directional links per HNSW node
DEFAULT_EF_CON = 128   # Candidate list size during index construction
DEFAULT_EF     = 128   # Candidate list size during search queries
PRECISION      = "int8d"  # 8-bit integer quantization (good balance of speed/accuracy)
CHECKSUM       = -1    # Required sentinel value — tells Endee to compute its own checksum

# Shared HTTP session — reused across all requests for connection pooling
_session: Optional[requests.Session] = None


def _get_session() -> requests.Session:
    """Return a shared requests.Session with retry logic and connection pooling."""
    global _session
    if _session is None:
        s = requests.Session()
        # Automatically retry on transient server errors and rate limits
        retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
        s.mount("http://",  HTTPAdapter(max_retries=retry, pool_maxsize=10))
        s.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=10))
        _session = s
    return _session


# --- Compression helpers (matches SDK compression.py exactly) ---

def _json_zip(data: dict) -> bytes:
    """Compress a metadata dict to bytes using orjson + zlib (SDK storage format)."""
    return b"" if not data else zlib.compress(orjson.dumps(data))


def _json_unzip(data: bytes) -> dict:
    """Decompress metadata bytes back into a Python dict."""
    return {} if not data else orjson.loads(zlib.decompress(data))


def _normalise(vec: List[float]) -> List[float]:
    """
    L2-normalise a vector so its magnitude equals 1.
    This is required before inserting into or querying a cosine index,
    because cosine similarity assumes unit vectors.
    """
    norm = math.sqrt(sum(x * x for x in vec))
    return vec if norm < 1e-10 else [x / norm for x in vec]


# --- Public API ---

def create_index(index_name: str = INDEX_NAME, dimension: int = DIMENSION, space_type: str = "cosine") -> str:
    """
    Create a new vector index in Endee.
    Safe to call multiple times — handles 409 Conflict if it already exists.
    """
    r = _get_session().post(f"{BASE_URL}/index/create", json={
        "index_name": index_name,
        "dim":        dimension,
        "space_type": space_type,
        "M":          DEFAULT_M,
        "ef_con":     DEFAULT_EF_CON,
        "checksum":   CHECKSUM,
        "precision":  PRECISION,
        "version":    1,
    }, timeout=10)

    if r.status_code == 200:
        logger.info(f"Index '{index_name}' created")
        return f"Index '{index_name}' created"
    elif r.status_code == 409:
        # Index already exists — this is fine, just skip
        logger.info(f"Index '{index_name}' already exists")
        return f"Index '{index_name}' already exists"
    else:
        raise RuntimeError(f"create_index failed {r.status_code}: {r.text}")


def insert_batch(vectors: List[Dict], index_name: str = INDEX_NAME) -> str:
    """
    Insert a batch of vectors into Endee via msgpack.

    Expected input format:
        [{"id": str, "values": [float, ...], "metadata": {"team": str, "resolution": str}}]

    Each vector is packed into Endee's wire format:
        [id, zlib(meta), filter_json, original_norm, normalised_vector]
    """
    if not vectors:
        return "Empty batch"

    batch = []
    for v in vectors:
        raw_vec = v["values"]
        batch.append([
            str(v["id"]),
            _json_zip(v.get("metadata", {})),   # Compress metadata for storage
            orjson.dumps({}).decode(),           # Empty filter (unused in basic RAG)
            float(math.sqrt(sum(x * x for x in raw_vec))),  # Original L2 norm (stored for retrieval)
            _normalise(raw_vec),                 # Unit vector for cosine similarity
        ])

    r = _get_session().post(
        f"{BASE_URL}/index/{index_name}/vector/insert",
        data=msgpack.packb(batch, use_bin_type=True, use_single_float=True),
        headers={"Content-Type": "application/msgpack"},
        timeout=30,
    )
    if r.status_code != 200:
        raise RuntimeError(f"insert_batch failed {r.status_code}: {r.text}")

    logger.debug(f"Inserted {len(vectors)} vectors into '{index_name}'")
    return "Vectors inserted successfully"


def search(vector: List[float], top_k: int = 5, index_name: str = INDEX_NAME) -> dict:
    """
    Find the top-k most similar vectors to the query.

    Endee returns a msgpack-encoded list sorted by similarity (highest first).
    Each item: [similarity, id, meta_bytes, filter_str, norm, ...]

    Returns:
        {"results": [{"id": str, "score": float, "metadata": {"team": ..., "resolution": ...}}]}
    """
    if not vector:
        raise ValueError("Empty query vector")

    r = _get_session().post(f"{BASE_URL}/index/{index_name}/search", json={
        "k":               top_k,
        "ef":              DEFAULT_EF,
        "vector":          _normalise(vector),  # Must be unit vector for cosine index
        "include_vectors": False,               # We only need metadata, not the raw vectors
    }, timeout=10)

    if r.status_code != 200:
        raise RuntimeError(f"search failed {r.status_code}: {r.text}")

    raw = msgpack.unpackb(r.content, raw=False)

    # Unpack each result and decompress the stored metadata
    results = [
        {
            "id":       item[1],
            "score":    item[0],                                    # Cosine similarity score (0–1)
            "metadata": _json_unzip(item[2]) if item[2] else {}     # team + resolution
        }
        for item in raw[:top_k]
    ]

    logger.debug(f"Search returned {len(results)} results")
    return {"results": results}


def list_indexes() -> list:
    """Return a list of all indexes currently stored in Endee."""
    r = _get_session().get(f"{BASE_URL}/index/list", timeout=5)
    r.raise_for_status()
    return r.json()


def get_index_info(index_name: str = INDEX_NAME) -> dict:
    """
    Fetch metadata for a specific index (dimension, count, space type, etc.).
    Raises a descriptive RuntimeError if the index doesn't exist yet.
    """
    r = _get_session().get(f"{BASE_URL}/index/{index_name}/info", timeout=5)
    if r.status_code == 404:
        raise RuntimeError(f"Index '{index_name}' does not exist. Run: python ingest_tickets.py")
    r.raise_for_status()
    return r.json()


def check_connection() -> bool:
    """Return True if the Endee server is reachable at localhost:8080."""
    try:
        return requests.get("http://localhost:8080/health", timeout=2).status_code == 200
    except Exception:
        return False


def invalidate_cache():
    """No-op — kept for API compatibility with earlier SDK-based version."""
    pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    print("=" * 50)

    if not check_connection():
        print("❌ Endee not running. Start: cd ~/endee && ./build/ndd-avx2")
        exit(1)
    print("✅ Endee running")
    print(f"✅ {create_index()}")

    result = insert_batch([{
        "id": "selftest_001",
        "values": [0.01] * DIMENSION,
        "metadata": {"team": "IT Support", "resolution": "Restart the service"}
    }])
    print(f"✅ {result}")

    hits = search([0.01] * DIMENSION, top_k=1)["results"]
    print(f"✅ Search: {len(hits)} result(s)")
    if hits:
        print(f"   id={hits[0]['id']}  score={hits[0]['score']:.4f}  team={hits[0]['metadata'].get('team')}")
    print("=" * 50)