"""Chunking unit tests (no pytest / no model download required)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from unittest.mock import patch
from src import chunker


def test_fixed_respects_token_limit():
    text = " ".join(f"w{i}" for i in range(250))
    cs = chunker.chunk_fixed(text, "a1", 100)
    assert len(cs) == 3 and all(c["strategy"] == "fixed" for c in cs)
    assert all(c["token_estimate"] <= 100 for c in cs)


def test_sentence_merges_to_limit():
    text = "Diabetes is common. Metformin treats diabetes. Cardiovascular risk matters. TP53 regulates cancer."
    cs = chunker.chunk_sentence(text, "a1", 100)
    assert len(cs) >= 1 and all("chunk_id" in c and "article_id" in c for c in cs)


def test_semantic_uses_clustering(monkeypatch=None):
    import numpy as np

    sents = ["Diabetes treatment with metformin.", "Metformin lowers blood sugar.",
             "TP53 mutations cause cancer.", "Cancer therapy targets TP53."]
    vecs = np.array([[1, 0]] * 2 + [[0, 1]] * 2, dtype=np.float32)
    with patch("src.chunker.split_sentences", return_value=sents):
        with patch("src.embeddings.embed_texts", return_value=vecs):
            cs = chunker.chunk_semantic(" ".join(sents), "a1", 100)
    assert len(cs) >= 2 and all(c["strategy"] == "semantic" for c in cs)


if __name__ == "__main__":
    test_fixed_respects_token_limit()
    test_sentence_merges_to_limit()
    test_semantic_uses_clustering()
    print("chunking OK")
