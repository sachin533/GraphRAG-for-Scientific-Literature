"""Full evaluation: retrieval (Recall@5/10, MRR) + generation (ROUGE-L, BERTScore).

Compares baseline (fixed chunking, vector-only) vs improved
(semantic chunking + vector + graph expansion).

Usage:
  py evaluation/evaluate.py [--n 200] [--topk 10]

Requires: Neo4j populated (see scripts/build_graph.py) for BOTH strategies, or
at minimum the 'semantic' graph; fixed-chunk retrieval falls back to in-memory
embeddings when its graph is absent (clearly logged). Gold answers/articles come
from the matched PubMedQA set -- never fabricated.
"""
import argparse, json, logging, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def recall_mrr(results: list[list[str]], golds: list[str], k: int):
    hits_k = sum(1 for r, g in zip(results, golds) if g in r[:k])
    rr = []
    for r, g in zip(results, golds):
        try:
            rr.append(1.0 / (r.index(g) + 1))
        except ValueError:
            rr.append(0.0)
    n = max(len(golds), 1)
    return {"recall@%d" % k: hits_k / n, "mrr": sum(rr) / n}


def main(n: int = 200, top_k: int = 10):
    from src.data_loader import load_or_download
    from src import retrieval as R
    from src.neo4j_loader import get_driver

    data = load_or_download()
    matched = data["matched"][:n]
    log.info("Evaluating on %d matched PubMedQA questions.", len(matched))
    gold_articles = [m.get("matched_article_id", "") for m in matched]

    drv = get_driver()
    # Improved: semantic + graph
    retr_ids, answers_g, answers_b = [], [], []
    for m in matched:
        q = m["question"]
        hits = R.retrieve(q, top_k=top_k, expand_graph=True, driver=drv)
        retr_ids.append([h["article_id"] for h in hits])
        try:
            from src import rag_pipeline as P

            answers_g.append(P.answer(q, top_k=5, expand_graph=True, driver=drv)["answer"])
        except Exception as exc:
            log.warning("LLM answer failed: %s", exc)
            answers_g.append("")
        try:
            answers_b.append(P.answer(q, top_k=5, expand_graph=False, driver=drv)["answer"])
        except Exception as exc:
            log.warning("LLM baseline failed: %s", exc)
            answers_b.append("")

    print("Improved retrieval:", recall_mrr(retr_ids, gold_articles, 5),
          recall_mrr(retr_ids, gold_articles, 10))

    # Generation metrics vs PubMedQA long answers
    refs = [m.get("long_answer", "") or m.get("answer", "") for m in matched]
    try:
        from rouge_score import rouge_scorer

        sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        rl_g = sum(sc.score(r, p)["rougeL"].fmeasure
                   for r, p in zip(refs, answers_g) if r and p) / max(len(refs), 1)
        rl_b = sum(sc.score(r, p)["rougeL"].fmeasure
                   for r, p in zip(refs, answers_b) if r and p) / max(len(refs), 1)
        print(f"ROUGE-L graph={rl_g:.4f} no-graph={rl_b:.4f}")
    except Exception as exc:
        print("ROUGE skipped:", exc)
    try:
        from bert_score import score as bert_score

        P_, R_, F_ = bert_score([a for a in answers_g if a],
                                [r for r, a in zip(refs, answers_g) if a], lang="en")
        print(f"BERTScore-F1 (graph)={float(F_.mean()):.4f}")
    except Exception as exc:
        print("BERTScore skipped:", exc)

    os.makedirs("evaluation/results", exist_ok=True)
    with open("evaluation/results/eval.json", "w", encoding="utf-8") as f:
        json.dump([{"q": m["question"], "gold_article": g, "retrieved": r,
                    "answer_graph": ag, "answer_nograph": ab}
                   for m, g, r, ag, ab in
                   zip(matched, gold_articles, retr_ids, answers_g, answers_b)],
                  f, ensure_ascii=False, indent=1)
    print("Wrote evaluation/results/eval.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--topk", type=int, default=10)
    a = ap.parse_args()
    main(a.n, a.topk)
