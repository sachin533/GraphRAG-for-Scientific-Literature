"""Create Neo4j vector index:  py scripts/create_vector_index.py"""
import logging, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.neo4j_loader import create_vector_index, get_driver  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

if __name__ == "__main__":
    create_vector_index(get_driver())
    print("Vector index ensured: chunk_embedding (384, cosine).")
