"""Three chunking strategies for comparison.

A. fixed-token baseline: ~100 tokens (whitespace estimate), no overlap.
B. sentence-boundary: split into sentences, greedily merge up to token limit.
C. semantic clustering: sentence embeddings (all-MiniLM-L6-v2) + Agglomerative
   Clustering (cosine) or HDBSCAN when available. Each cluster -> one chunk,
   sentences re-ordered to original position; tiny leftover clusters merged.

Output JSONL rows: {article_id, chunk_id, text, strategy, ...metadata}.
No fixed-character-length splitting: all strategies respect token/sentence/semantic units.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List

log = logging.getLogger(__name__)


def estimate_tokens(text: str) -> int:
    """Whitespace token estimate (~matches MiniLM wordpieces closely enough for chunking)."""
    return len(text.split())


def split_sentences(text: str) -> List[str]:
    """Sentence split: spaCy sentencizer if available, else regex fallback."""
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []
    try:
        import spacy

        try:
            nlp = spacy.blank("en")
            nlp.add_pipe("sentencizer")
            return [s.text.strip() for s in nlp(text).sents if s.text.strip()]
        except Exception:
            pass
    except Exception:
        pass
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_fixed(text: str, article_id: str, token_limit: int = 100) -> List[Dict[str, Any]]:
    words = (text or "").split()
    chunks = []
    for i in range(0, len(words), token_limit):
        piece = " ".join(words[i:i + token_limit])
        if piece.strip():
            chunks.append(_mk(article_id, len(chunks), piece, "fixed", token_limit))
    return chunks


def chunk_sentence(text: str, article_id: str, token_limit: int = 100) -> List[Dict[str, Any]]:
    sentences = split_sentences(text)
    chunks: List[Dict[str, Any]] = []
    cur: List[str] = []
    cur_tok = 0
    for s in sentences:
        t = estimate_tokens(s)
        if cur and cur_tok + t > token_limit:
            chunks.append(_mk(article_id, len(chunks), " ".join(cur), "sentence", token_limit))
            cur, cur_tok = [], 0
        cur.append(s)
        cur_tok += t
    if cur:
        chunks.append(_mk(article_id, len(chunks), " ".join(cur), "sentence", token_limit))
    return chunks


def chunk_semantic(text: str, article_id: str, token_limit: int = 100,
                   min_cluster: int = 2) -> List[Dict[str, Any]]:
    """Semantic chunking: embed sentences, cluster, one chunk per cluster.

    Uses AgglomerativeClustering(cosine, distance_threshold) by default because it
    needs no pre-set k and works on small per-abstract sentence sets. If hdbscan
    is installed it is tried first (min_cluster_size=min_cluster).
    """
    from .embeddings import embed_texts

    sentences = split_sentences(text)
    if len(sentences) <= 1:
        return [_mk(article_id, 0, text.strip(), "semantic", token_limit)] if text.strip() else []
    if len(sentences) <= 3:
        # Too few sentences to cluster meaningfully -> sentence strategy.
        out = chunk_sentence(text, article_id, token_limit)
        for c in out:
            c["strategy"] = "semantic"
        return out

    embs = embed_texts(sentences)  # (n, 384), L2-normalised
    import numpy as np

    labels: Any = None
    try:
        import hdbscan  # type: ignore

        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster, metric="euclidean")
        labels = clusterer.fit_predict(embs)
        log.debug("HDBSCAN labels: %s", sorted(set(int(x) for x in labels)))
    except Exception as exc:
        log.debug("HDBSCAN unavailable/failed (%s); using Agglomerative.", exc)
        labels = None
    if labels is None or (hasattr(labels, "__len__") and set(int(x) for x in labels) <= { -1 }):
        from sklearn.cluster import AgglomerativeClustering

        # distance_threshold ~0.5 cosine distance groups near-paraphrases.
        cl = AgglomerativeClustering(
            n_clusters=None, metric="cosine", linkage="average", distance_threshold=0.5
        )
        labels = cl.fit_predict(embs)

    # Group sentence indices by label; noise (-1) each becomes its own chunk.
    groups: Dict[int, List[int]] = {}
    for idx, lab in enumerate(labels):
        groups.setdefault(int(lab), []).append(idx)
    chunks: List[Dict[str, Any]] = []
    for lab in sorted(groups):
        idxs = sorted(groups[lab])
        if lab == -1:
            for j in idxs:  # noise: single-sentence chunks
                chunks.append(_mk(article_id, len(chunks), sentences[j], "semantic", token_limit,
                                  cluster=lab))
        else:
            # Respect token limit: split oversized clusters greedily in original order.
            cur: List[str] = []
            cur_tok = 0
            for j in idxs:
                t = estimate_tokens(sentences[j])
                if cur and cur_tok + t > token_limit * 2:  # semantic chunks may be larger
                    chunks.append(_mk(article_id, len(chunks), " ".join(cur), "semantic",
                                      token_limit, cluster=lab))
                    cur, cur_tok = [], 0
                cur.append(sentences[j])
                cur_tok += t
            if cur:
                chunks.append(_mk(article_id, len(chunks), " ".join(cur), "semantic",
                                  token_limit, cluster=lab))
    # Order chunks by first occurrence in the abstract for readability.
    return chunks


def _mk(article_id: str, i: int, text: str, strategy: str, token_limit: int,
        **extra: Any) -> Dict[str, Any]:
    return {
        "article_id": article_id,
        "chunk_id": f"{article_id}::{strategy}::{i:04d}",
        "text": text,
        "strategy": strategy,
        "token_estimate": estimate_tokens(text),
        "token_limit": token_limit,
        **extra,
    }


def chunk_articles(articles: List[Dict[str, Any]], strategy: str = "semantic",
                   token_limit: int = 100) -> List[Dict[str, Any]]:
    fn = {"fixed": chunk_fixed, "sentence": chunk_sentence, "semantic": chunk_semantic}[strategy]
    all_chunks: List[Dict[str, Any]] = []
    for a in articles:
        body = ((a.get("title") or "") + ". " + (a.get("abstract") or "")).strip()
        if strategy == "semantic":
            cs = fn(body, a["article_id"], token_limit)
        else:
            cs = fn(body, a["article_id"], token_limit)
        for c in cs:
            c["pub_date"] = a.get("pub_date", "")
            c["title"] = a.get("title", "")[:200]
        all_chunks.extend(cs)
    log.info("strategy=%s articles=%d chunks=%d", strategy, len(articles), len(all_chunks))
    return all_chunks


def save_chunks(chunks: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    log.info("Saved %d chunks -> %s", len(chunks), path)
