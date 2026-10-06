"""Create chunk files:  py scripts/create_chunks.py [--n 5000] [--limit 100]"""
import argparse, logging, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data_loader import load_or_download  # noqa: E402
from src.chunker import chunk_articles, save_chunks  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--limit", type=int, default=100)
    a = ap.parse_args()
    arts = load_or_download(n_articles=a.n)["articles"]
    for strat in ("fixed", "sentence", "semantic"):
        chunks = chunk_articles(arts, strategy=strat, token_limit=a.limit)
        save_chunks(chunks, f"data/chunks/{strat}.jsonl")
        print(f"{strat}: {len(chunks)} chunks")
