"""Embedding tests: dimension 384, reuse of singleton model."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_embed_dim():
    from src import embeddings as E

    class Fake:
        def get_sentence_embedding_dimension(self):
            return 384

        def encode(self, texts, **kw):
            import numpy as np

            return np.zeros((len(texts), 384), dtype=np.float32)

    E._MODEL = Fake()
    v = E.embed_text("diabetes and metformin")
    assert len(v) == 384
    E._MODEL = None
