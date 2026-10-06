"""RAG pipeline test with stubbed retrieval + LLM (no network, no pytest needed)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from unittest.mock import patch


def test_answer_merges(monkeypatch=None):
    from src import rag_pipeline as P

    fake_ctx = [{"chunk_id": "c1", "article_id": "a1", "text": "Metformin treats diabetes.",
                 "vector_score": 0.9, "graph_score": 0.0,
                 "combined_score": 0.63, "entities": ["Metformin"]}]
    with patch("src.retrieval.retrieve", return_value=fake_ctx):
        with patch("src.llm.generate",
                   return_value={"answer": "Metformin treats diabetes.",
                                 "provider": "test", "model": "test"}):
            out = P.answer("What treats diabetes?", top_k=1)
    assert "Metformin" in out["answer"] and len(out["contexts"]) == 1
    return True


if __name__ == "__main__":
    test_answer_merges()
    print("rag OK")
