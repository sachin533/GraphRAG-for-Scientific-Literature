"""Graph-based re-ranking: GDS PageRank when available, else NetworkX.

Builds the retrieved subgraph (chunks + shared entities), scores chunk nodes by
PageRank/Betweenness, and returns {chunk_id: normalised centrality in [0,1]}.
Falls back gracefully when GDS is not installed -- scoring stays understandable:
final = alpha*vector + beta*graph + gamma*centrality (see retrieval.combine_and_rerank).
"""
from __future__ import annotations

import logging
from typing import Dict, List

log = logging.getLogger(__name__)


def centrality_scores(nodes: List[Dict], driver=None,
                      method: str = "pagerank") -> Dict[str, float]:
    """Try Neo4j GDS first; fall back to local NetworkX on the retrieved subgraph."""
    try:
        return _gds_centrality(nodes, driver, method)
    except Exception as exc:
        log.info("GDS centrality unavailable (%s); using NetworkX fallback.", exc)
        return _networkx_centrality(nodes, method)


def _gds_centrality(nodes, driver, method: str = "pagerank") -> Dict[str, float]:
    if driver is None:
        from .neo4j_loader import get_driver

        driver = get_driver()
    from .config import settings

    ids = [n["chunk_id"] for n in nodes if n.get("chunk_id")]
    if not ids:
        return {}
    algo = "gds.pageRank.stream" if method == "pagerank" else "gds.betweenness.stream"
    with driver.session(database=settings.neo4j_database) as s:
        # Verify GDS exists
        s.run("CALL gds.list() YIELD name RETURN name LIMIT 1").consume()
        rows = s.run(
            "MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(d:Chunk) "
            "WHERE c.chunk_id IN $ids AND d.chunk_id IN $ids "
            "WITH gds.graph.project('retr-sub', c, d) AS g "
            f"CALL {algo}(g) YIELD nodeId, score "
            "RETURN gds.util.asNode(nodeId).chunk_id AS cid, score",
            ids=ids,
        )
        out = {r["cid"]: float(r["score"]) for r in rows}
    return _normalise(out)


def _networkx_centrality(nodes: List[Dict], method: str = "pagerank") -> Dict[str, float]:
    import networkx as nx

    G = nx.Graph()
    chunk_ids = [n.get("chunk_id") for n in nodes if n.get("chunk_id")]
    G.add_nodes_from(chunk_ids)
    # Edge between chunks sharing a via_entity (recorded during expansion) or entity overlap.
    ent_of = {}
    for n in nodes:
        ents = set(n.get("entities") or []) | set(n.get("via_entities") or [])
        ent_of[n.get("chunk_id")] = ents
    for i in range(len(chunk_ids)):
        for j in range(i + 1, len(chunk_ids)):
            if ent_of.get(chunk_ids[i]) & ent_of.get(chunk_ids[j]):
                G.add_edge(chunk_ids[i], chunk_ids[j])
    if len(G.edges) == 0:
        return {cid: 0.0 for cid in chunk_ids}
    scores = (nx.pagerank(G) if method == "pagerank"
              else nx.betweenness_centrality(G))
    return _normalise({k: float(v) for k, v in scores.items()})


def _normalise(d: Dict[str, float]) -> Dict[str, float]:
    if not d:
        return {}
    mx = max(d.values())
    if mx <= 0:
        return {k: 0.0 for k in d}
    return {k: v / mx for k, v in d.items()}
