"""Phase-7 comparison: 10 PubMedQA questions, answers with vs without graph.

Usage: py scripts/compare_10.py [--n 10]
Writes: evaluation/comparison_10_questions.csv (Question, Answer without graph,
Answer with graph, Retrieved chunks, Relevant article, Observations)
"""
import csv, logging, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)


def main(n: int = 10):
    from src.data_loader import load_or_download
    from src import rag_pipeline as P

    matched = load_or_download()["matched"][:n]
    os.makedirs("evaluation", exist_ok=True)
    path = "evaluation/comparison_10_questions.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Question", "Answer without graph", "Answer with graph",
                    "Retrieved chunks", "Relevant article", "Observations"])
        for m in matched:
            q = m["question"]
            try:
                off = P.answer(q, top_k=5, expand_graph=False)
            except Exception as exc:
                off = {"answer": f"ERROR: {exc}", "contexts": []}
            try:
                on = P.answer(q, top_k=5, expand_graph=True)
            except Exception as exc:
                on = {"answer": f"ERROR: {exc}", "contexts": []}
            ids_off = [c.get("chunk_id", "") for c in off.get("contexts", [])]
            ids_on = [c.get("chunk_id", "") for c in on.get("contexts", [])]
            new = sorted(set(ids_on) - set(ids_off))
            obs = f"graph added {len(new)} chunks via shared entities: {new[:3]}" if new \
                else "graph added no new chunks"
            w.writerow([q, off.get("answer", ""), on.get("answer", ""),
                        "; ".join(ids_on),
                        m.get("matched_article_id", ""), obs])
            log.info("Q done: %.60s | %s", q, obs)
    print(f"Wrote {path}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    main(ap.parse_args().n)
