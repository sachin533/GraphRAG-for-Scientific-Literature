"""Download PubMed abstracts + PubMedQA:  py scripts/download_data.py [--n 5000]"""
import argparse, logging, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data_loader import load_or_download  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--no-cache", action="store_true")
    a = ap.parse_args()
    out = load_or_download(n_articles=a.n, use_cache=not a.no_cache)
    print(f"articles={len(out['articles'])} pubmedqa={len(out['pubmedqa'])} matched={len(out['matched'])}")
