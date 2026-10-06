"""Sentence embeddings via all-MiniLM-L6-v2 (384-dim). Model loaded once and reused."""
from __future__ import annotations

import logging
from typing import List

import numpy as np

log = logging.getLogger(__name__)

_MODEL = None
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DIM = 384


def get_model(name: str = MODEL_NAME):
    global _MODEL, MODEL_NAME
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer

        log.info("Loading SentenceTransformer %s ...", name)
        _MODEL = SentenceTransformer(name)
        MODEL_NAME = name
        log.info("Embedding model ready (dim=%s).", _MODEL.get_sentence_embedding_dimension())
    return _MODEL


def embed_texts(texts: List[str], normalize: bool = True) -> np.ndarray:
    """Embed a batch of texts -> (n, 384) float32 array."""
    model = get_model()
    embs = model.encode(texts, normalize_embeddings=normalize, show_progress_bar=False)
    return np.asarray(embs, dtype=np.float32)


def embed_text(text: str) -> List[float]:
    return embed_texts([text])[0].tolist()


def embed_chunks(chunks: List[dict], text_key: str = "text",
                 batch_size: int = 64) -> List[dict]:
    """Attach 'embedding' (list[float], 384-dim) to each chunk dict, in batches."""
    model = get_model()
    texts = [(c.get(text_key) or "") for c in chunks]
    out_embs = model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                            show_progress_bar=True)
    for c, e in zip(chunks, out_embs):
        c["embedding"] = np.asarray(e, dtype=np.float32).tolist()
    log.info("Embedded %d chunks (dim=%d).", len(chunks), len(chunks[0]["embedding"]) if chunks else 0)
    return chunks
