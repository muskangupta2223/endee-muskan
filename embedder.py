"""
Embedding module — converts text into 384-dimensional vectors using
the all-MiniLM-L6-v2 SentenceTransformer model.
"""

from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # Output dimension for this model


@lru_cache(maxsize=1)
def get_model():
    """
    Load the SentenceTransformer model once and cache it.
    The @lru_cache ensures we don't reload it on every request.
    """
    from sentence_transformers import SentenceTransformer
    logger.info(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    logger.info(f"Model loaded (dim={EMBEDDING_DIM})")
    return model


def embed_text(text: str, normalize: bool = True) -> list:
    """
    Convert a single text string into a 384-dim float vector.
    normalize=True is required for cosine similarity to work correctly.
    """
    if not text or not text.strip():
        raise ValueError("Cannot embed empty text")
    return get_model().encode(
        text,
        normalize_embeddings=normalize,
        show_progress_bar=False
    ).tolist()


def embed_batch(texts: list, normalize: bool = True, batch_size: int = 32) -> list:
    """
    Convert a list of texts into vectors in one batched call.
    Much faster than calling embed_text() in a loop for large datasets.
    Shows a progress bar automatically when processing >100 texts.
    """
    if not texts:
        raise ValueError("Cannot embed empty list")
    vectors = get_model().encode(
        texts,
        normalize_embeddings=normalize,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 100
    )
    return [v.tolist() for v in vectors]


def get_embedding_dimension() -> int:
    """Return the vector size produced by this model (384)."""
    return EMBEDDING_DIM


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    vec = embed_text("This is a test support ticket")
    print(f"✅ Single embedding: dim={len(vec)}")
    vecs = embed_batch(["Payment issue", "Account locked", "Password reset"])
    print(f"✅ Batch embedding: {len(vecs)} vectors")
    try:
        embed_text("")
    except ValueError:
        print("✅ Empty text validation working")