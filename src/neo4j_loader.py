"""Neo4j knowledge-graph loader: schema, batch UNWIND inserts, vector index.

Schema:
  (:Article {article_id, title, pub_date})
  (:Chunk {chunk_id, text, embedding, strategy, article_id})
  (:Entity {name, type})   // name key is lower-cased for dedup; display_name keeps original
  (:Article)-[:HAS_CHUNK]->(:Chunk)
  (:Chunk)-[:MENTIONS]->(:Entity)
  (:Chunk)-[:SEMANTIC_SIMILAR {score}]->(:Chunk)   // cosine > threshold, within-article

All writes use UNWIND batches (no row-by-row thousands of queries).
Credentials only from env (.env) / Streamlit secrets -- never hardcoded.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np

from .config import settings

log = logging.getLogger(__name__)

VECTOR_INDEX = "chunk_embedding"
VECTOR_DIM = 384


def get_driver():
    from neo4j import GraphDatabase

    if not settings.neo4j_password:
        log.warning("NEO4J_PASSWORD is empty -- connection will likely fail. Set it in .env.")
    return GraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
    )


def init_schema(driver=None) -> None:
    """Constraints + indexes (idempotent)."""
    drv = driver or get_driver()
    stmts = [
        "CREATE CONSTRAINT article_id IF NOT EXISTS FOR (a:Article) REQUIRE a.article_id IS UNIQUE",
        "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
        "CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
        "CREATE INDEX chunk_article IF NOT EXISTS FOR (c:Chunk) ON (c.article_id)",
    ]
    with drv.session(database=settings.neo4j_database) as s:
        for st in stmts:
            try:
                s.run(st).consume()
            except Exception as exc:
                log.warning("Schema stmt failed [%s]: %s", st[:60], exc)
    log.info("Schema constraints/indexes ensured.")


def create_vector_index(driver=None, dim: int = VECTOR_DIM) -> None:
    drv = driver or get_driver()
    stmt = (
        f"CREATE VECTOR INDEX {VECTOR_INDEX} IF NOT EXISTS FOR (c:Chunk) ON (c.embedding) "
        f"OPTIONS {{indexConfig: {{`vector.dimensions`: {dim}, `vector.similarity_function`: 'cosine'}}}}"
    )
    with drv.session(database=settings.neo4j_database) as s:
        s.run(stmt).consume()
    log.info("Vector index '%s' ensured (dim=%d, cosine).", VECTOR_INDEX, dim)


def clear_graph(driver=None) -> None:
    drv = driver or get_driver()
    with drv.session(database=settings.neo4j_database) as s:
        res = s.run("MATCH (n) DETACH DELETE n RETURN count(n) AS d").single()
    log.warning("Graph cleared (%s nodes deleted).", res["d"] if res else "?")


def _batches(rows: List[dict], size: int = 500):
    for i in range(0, len(rows), size):
        yield rows[i:i + size]


def insert_articles(articles: List[Dict[str, Any]], driver=None) -> int:
    drv = driver or get_driver()
    rows = [{"article_id": a["article_id"], "title": a.get("title", "")[:1000],
             "pub_date": str(a.get("pub_date", ""))} for a in articles]
    with drv.session(database=settings.neo4j_database) as s:
        for b in _batches(rows):
            s.run("UNWIND $rows AS r MERGE (a:Article {article_id: r.article_id}) "
                  "SET a.title=r.title, a.pub_date=r.pub_date", rows=b).consume()
    log.info("Upserted %d Article nodes.", len(rows))
    return len(rows)


def insert_chunks(chunks: List[Dict[str, Any]], driver=None) -> int:
    """Insert Chunk nodes + HAS_CHUNK edges. Requires 'embedding' on each chunk."""
    drv = driver or get_driver()
    rows = [{"chunk_id": c["chunk_id"], "text": c.get("text", ""),
             "embedding": [float(x) for x in c["embedding"]],
             "strategy": c.get("strategy", ""), "article_id": c.get("article_id", "")}
            for c in chunks]
    with drv.session(database=settings.neo4j_database) as s:
        for b in _batches(rows, 200):
            s.run("UNWIND $rows AS r MERGE (c:Chunk {chunk_id: r.chunk_id}) "
                  "SET c.text=r.text, c.embedding=r.embedding, "
                  "c.strategy=r.strategy, c.article_id=r.article_id", rows=b).consume()
            s.run("UNWIND $rows AS r MATCH (a:Article {article_id: r.article_id}) "
                  "MATCH (c:Chunk {chunk_id: r.chunk_id}) "
                  "MERGE (a)-[:HAS_CHUNK]->(c)", rows=b).consume()
    log.info("Upserted %d Chunk nodes + HAS_CHUNK edges.", len(rows))
    return len(rows)


def insert_entities(chunks: List[Dict[str, Any]], driver=None) -> int:
    """Insert Entity nodes + MENTIONS edges from chunk['entities'] (batched UNWIND)."""
    drv = driver or get_driver()
    ent_rows, men_rows = [], []
    for c in chunks:
        for e in (c.get("entities") or []):
            name = (e.get("name") or "").strip()
            if not name:
                continue
            ent_rows.append({"name": name.lower(), "display": name,
                             "type": e.get("type", "ENTITY")})
            men_rows.append({"chunk_id": c["chunk_id"], "name": name.lower()})
    # dedup entities by key
    ded = {}
    for r in ent_rows:
        ded.setdefault(r["name"], r)
    ent_rows = list(ded.values())
    with drv.session(database=settings.neo4j_database) as s:
        for b in _batches(ent_rows):
            s.run("UNWIND $rows AS r MERGE (e:Entity {name: r.name}) "
                  "SET e.display_name=r.display, e.type=r.type", rows=b).consume()
        for b in _batches(men_rows, 1000):
            s.run("UNWIND $rows AS r MATCH (c:Chunk {chunk_id: r.chunk_id}) "
                  "MATCH (e:Entity {name: r.name}) MERGE (c)-[:MENTIONS]->(e)",
                  rows=b).consume()
    log.info("Upserted %d entities, %d MENTIONS edges.", len(ent_rows), len(men_rows))
    return len(ent_rows)


def insert_semantic_similar(chunks: List[Dict[str, Any]], driver=None,
                            threshold: float | None = None) -> int:
    """Create SEMANTIC_SIMILAR edges for within-article chunk pairs with cosine > threshold.

    Within-article only: keeps edge count tractable (O(k^2) per article, not global).
    """
    thr = threshold if threshold is not None else settings.semantic_sim_threshold
    from collections import defaultdict

    by_art: Dict[str, List[dict]] = defaultdict(list)
    for c in chunks:
        if "embedding" in c:
            by_art[c["article_id"]].append(c)
    pairs = []
    for aid, cs in by_art.items():
        if len(cs) < 2:
            continue
        M = np.asarray([c["embedding"] for c in cs], dtype=np.float32)
        M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
        sim = M @ M.T
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                sc = float(sim[i, j])
                if sc > thr:
                    pairs.append({"a": cs[i]["chunk_id"], "b": cs[j]["chunk_id"],
                                  "score": sc})
    drv = driver or get_driver()
    with drv.session(database=settings.neo4j_database) as s:
        for b in _batches(pairs, 1000):
            s.run("UNWIND $rows AS r MATCH (x:Chunk {chunk_id: r.a}) "
                  "MATCH (y:Chunk {chunk_id: r.b}) "
                  "MERGE (x)-[s:SEMANTIC_SIMILAR]-(y) SET s.score=r.score", rows=b).consume()
    log.info("Created %d SEMANTIC_SIMILAR edges (threshold=%.2f).", len(pairs), thr)
    return len(pairs)


def stats(driver=None) -> Dict[str, int]:
    drv = driver or get_driver()
    q = {"articles": "MATCH (a:Article) RETURN count(a) AS n",
         "chunks": "MATCH (c:Chunk) RETURN count(c) AS n",
         "entities": "MATCH (e:Entity) RETURN count(e) AS n",
         "has_chunk": "MATCH ()-[r:HAS_CHUNK]->() RETURN count(r) AS n",
         "mentions": "MATCH ()-[r:MENTIONS]->() RETURN count(r) AS n",
         "sem_sim": "MATCH ()-[r:SEMANTIC_SIMILAR]-() RETURN count(r)/2 AS n"}
    out = {}
    with drv.session(database=settings.neo4j_database) as s:
        for k, stmt in q.items():
            try:
                out[k] = int(s.run(stmt).single()["n"])
            except Exception as exc:
                out[k] = -1
                log.warning("stats(%s) failed: %s", k, exc)
    log.info("Graph stats: %s", out)
    return out


def build_full_graph(articles, chunks, driver=None, clear: bool = False) -> Dict[str, int]:
    drv = driver or get_driver()
    init_schema(drv)
    if clear:
        clear_graph(drv)
    insert_articles(articles, drv)
    insert_chunks(chunks, drv)
    insert_entities(chunks, drv)
    insert_semantic_similar(chunks, drv)
    create_vector_index(drv)
    return stats(drv)
