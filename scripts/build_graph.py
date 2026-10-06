"""Build Neo4j graph:  py scripts/build_graph.py [--strategy semantic] [--clear]"""
import argparse, json, logging, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data_loader import load_or_download  # noqa: E402
from src.embeddings import embed_chunks  # noqa: E402
from src.entity_extractor import extract_for_chunks  # noqa: E402
from src.neo4j_loader import build_full_graph, get_driver  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="semantic", choices=["fixed", "sentence", "semantic"])
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--clear", action="store_true")
    ap.add_argument("--chunks", default="")
    a = ap.parse_args()
    if a.chunks and os.path.exists(a.chunks):
        chunks = [json.loads(l) for l in open(a.chunks, encoding="utf-8")]
        arts = load_or_download()["articles"]
    else:
        from src.chunker import chunk_articles

        arts = load_or_download()["articles"]
        chunks = chunk_articles(arts, strategy=a.strategy, token_limit=a.limit)
    if not chunks or "embedding" not in chunks[0]:
        embed_chunks(chunks)
    if not chunks or "entities" not in chunks[0]:
        extract_for_chunks(chunks)
    stats = build_full_graph(arts, chunks, driver=get_driver(), clear=a.clear)
    print(stats)
