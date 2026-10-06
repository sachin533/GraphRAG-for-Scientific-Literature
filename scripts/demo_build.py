"""Fast demo build: first N articles -> sentence chunks -> Neo4j + vector index.

Gives a working end-to-end system in minutes while the full 5000-article
semantic build runs separately.  Usage:  py scripts/demo_build.py [--n 500]
"""
import argparse, logging, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    a = ap.parse_args()
    from src.data_loader import load_or_download
    from src.chunker import chunk_articles
    from src.embeddings import embed_chunks
    from src.entity_extractor import extract_for_chunks
    from src.neo4j_loader import build_full_graph, get_driver

    arts = load_or_download(n_articles=a.n)["articles"][:a.n]
    log.info("Demo articles: %d", len(arts))
    chunks = chunk_articles(arts, strategy="sentence", token_limit=100)
    embed_chunks(chunks)
    extract_for_chunks(chunks)
    print(build_full_graph(arts, chunks, driver=get_driver(), clear=True))
