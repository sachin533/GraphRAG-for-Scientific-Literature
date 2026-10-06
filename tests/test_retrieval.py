"""Retrieval combine/dedupe logic (no Neo4j needed)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.retrieval import combine_and_rerank


def test_dedupe_and_scoring():
    vec = [{"chunk_id": "c1", "article_id": "a1", "text": "t1", "vector_score": 0.9, "entities": []},
           {"chunk_id": "c2", "article_id": "a1", "text": "t2", "vector_score": 0.7, "entities": []}]
    gra = [{"chunk_id": "c2", "article_id": "a1", "text": "t2", "graph_score": 1.0, "entities": ["Diabetes"]},
           {"chunk_id": "c3", "article_id": "a2", "text": "t3", "graph_score": 0.5, "entities": []}]
    out = combine_and_rerank(vec, gra, top_k=3)
    ids = [r["chunk_id"] for r in out]
    # c2 has vector 0.7 + graph 1.0 -> 0.79, ranks above c1 (0.63 vector-only)
    assert ids[0] == "c2" and set(ids) == {"c1", "c2", "c3"}
    assert all("combined_score" in r for r in out)
