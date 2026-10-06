"""Graph-enhanced retrieval: Neo4j vector search + shared-entity expansion + re-rank.

Pipeline per query:
  1. embed query (same all-MiniLM-L6-v2 model)
  2. Neo4j vector index search -> initial top_k (REAL vector search, not a Python list scan)
  3. optional graph expansion (depth<=2): Chunk-[:MENTIONS]->Entity<-[:MENTIONS]-Chunk
  4. dedupe + combined scoring: alpha*vector + beta*graph [+ gamma*centrality]
  5. return ranked context chunks with scores + entities

Each result: {chunk_id, article_id, text, vector_score, graph_score,
combined_score, entities}
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from .config import settings

log = logging.getLogger(__name__)


def vector_search(query_embedding: List[float], top_k: int = 5,
                  candidates: int | None = None, driver=None) -> List[Dict[str, Any]]:
    """REAL Neo4j vector-index query (db.index.vector.queryNodes)."""
    from .neo4j_loader import VECTOR_INDEX

    drv = driver
    if drv is None:
        from .neo4j_loader import get_driver

        drv = get_driver()
    k = candidates or max(top_k * 10, settings.vector_candidates)
    stmt = (
        "CALL db.index.vector.queryNodes($index, $k, $emb) YIELD node, score "
        "OPTIONAL MATCH (node)-[:MENTIONS]->(e:Entity) "
        "RETURN node.chunk_id AS chunk_id, node.article_id AS article_id, "
        "node.text AS text, score AS vector_score, "
        "collect(DISTINCT e.display_name) AS entities"
    )
    with drv.session(database=settings.neo4j_database) as s:
        try:
            recs = s.run(stmt, index=VECTOR_INDEX, k=k, emb=query_embedding)
            out = [dict(r) for r in recs]
        except Exception as exc:
            if "no such vector" in str(exc).lower() or "no such index" in str(exc).lower():
                raise RuntimeError(
                    "The Neo4j vector index 'chunk_embedding' does not exist yet. "
                    "The knowledge graph has not been built: run "
                    "`py scripts/create_chunks.py` then "
                    "`py scripts/build_graph.py --strategy semantic --clear`."
                ) from exc
            raise
    # Keep top_k by vector score
    out.sort(key=lambda r: r.get("vector_score", 0) or 0, reverse=True)
    log.info("Vector search: %d candidates, keeping top %d.", len(out), top_k)
    return out[:top_k]


def graph_expand(seed_chunk_ids: List[str], driver=None,
                 limit_per_seed: int = 5) -> List[Dict[str, Any]]:
    """Depth<=2 traversal via shared entities. Returns neighbour chunks with graph_score.

    graph_score = 1 / (1 + shared_rank): chunks sharing more/larger-overlap entities
    rank higher. Implemented as count of shared entities normalised.
    """
    if not seed_chunk_ids:
        return []
    drv = driver
    if drv is None:
        from .neo4j_loader import get_driver

        drv = get_driver()
    stmt = (
        "UNWIND $seeds AS sid "
        "MATCH (s:Chunk {chunk_id: sid})-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(n:Chunk) "
        "WHERE n.chunk_id <> sid "
        "WITH n, count(DISTINCT e) AS shared, collect(DISTINCT e.display_name) AS via "
        "OPTIONAL MATCH (n)-[:MENTIONS]->(e2:Entity) "
        "RETURN n.chunk_id AS chunk_id, n.article_id AS article_id, n.text AS text, "
        "shared, via, collect(DISTINCT e2.display_name) AS entities "
        "ORDER BY shared DESC LIMIT $lim"
    )
    with drv.session(database=settings.neo4j_database) as s:
        rows = [dict(r) for r in s.run(stmt, seeds=seed_chunk_ids,
                                       lim=limit_per_seed * len(seed_chunk_ids))]
    max_shared = max([r.get("shared", 1) for r in rows], default=1)
    for r in rows:
        r["graph_score"] = float(r.pop("shared", 1)) / float(max_shared)
        r["vector_score"] = 0.0
        r["via_entities"] = r.pop("via", [])
    log.info("Graph expansion: %d seeds -> %d neighbours.", len(seed_chunk_ids), len(rows))
    return rows


def combine_and_rerank(vector_hits: List[Dict[str, Any]],
                       graph_hits: List[Dict[str, Any]],
                       top_k: int = 5, alpha: float = 0.7,
                       beta: float = 0.3,
                       centrality: Dict[str, float] | None = None,
                       gamma: float = 0.0) -> List[Dict[str, Any]]:
    """Dedupe by chunk_id, combined = alpha*vector + beta*graph (+ gamma*centrality)."""
    pool: Dict[str, Dict[str, Any]] = {}
    for h in vector_hits:
        c = dict(h)
        c["vector_score"] = float(c.get("vector_score") or 0.0)
        c.setdefault("graph_score", 0.0)
        pool[c["chunk_id"]] = c
    for h in graph_hits:
        cid = h["chunk_id"]
        if cid in pool:
            pool[cid]["graph_score"] = max(float(pool[cid].get("graph_score") or 0.0),
                                          float(h.get("graph_score") or 0.0))
            via = set(pool[cid].get("via_entities", []) or []) | set(h.get("via_entities", []) or [])
            pool[cid]["via_entities"] = sorted(via)
        else:
            c = dict(h)
            c["vector_score"] = 0.0
            c["graph_score"] = float(c.get("graph_score") or 0.0)
            pool[cid] = c
    for cid, c in pool.items():
        cent = float((centrality or {}).get(cid, 0.0))
        c["centrality"] = cent
        c["combined_score"] = alpha * c["vector_score"] + beta * c["graph_score"] + gamma * cent
        c.setdefault("entities", [])
    ranked = sorted(pool.values(), key=lambda r: r["combined_score"], reverse=True)
    # Filter: drop pure-graph neighbours with negligible signal only when we have enough vector hits.
    if len([r for r in ranked if r["vector_score"] > 0]) >= top_k:
        ranked = [r for r in ranked if r["vector_score"] > 0 or r["graph_score"] >= 0.5]
    return ranked[:top_k]


def retrieve(query: str, top_k: int = 5, expand_graph: bool = True,
             alpha: float = 0.7, beta: float = 0.3,
             use_centrality: bool = False, driver=None) -> List[Dict[str, Any]]:
    """Main entry used by the pipeline/UI: embed -> vector -> expand -> rerank."""
    from .embeddings import embed_text
    from .graph_reranker import centrality_scores

    if not query or not query.strip():
        raise ValueError("Empty query")
    qemb = embed_text(query)
    vec_hits = vector_search(qemb, top_k=top_k, driver=driver)
    graph_hits: List[Dict[str, Any]] = []
    if expand_graph and vec_hits:
        graph_hits = graph_expand([h["chunk_id"] for h in vec_hits], driver=driver)
    cent = None
    gamma = 0.0
    if use_centrality and (vec_hits or graph_hits):
        cent = centrality_scores(vec_hits + graph_hits, driver=driver)
        gamma = 0.1
    ranked = combine_and_rerank(vec_hits, graph_hits, top_k=top_k,
                               alpha=alpha, beta=beta, centrality=cent, gamma=gamma)
    log.info("retrieve: q=%.60s vec=%d graph=%d final=%d expand=%s",
             query, len(vec_hits), len(graph_hits), len(ranked), expand_graph)
    return ranked
