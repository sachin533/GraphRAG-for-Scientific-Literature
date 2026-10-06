"""Full RAG pipeline: retrieve (vector + graph) -> LLM answer.

  answer(question, top_k, expand_graph, decompose, use_centrality)
    -> {'question','answer','provider','model','contexts','subquestions'}
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from . import retrieval as _ret
from . import llm as _llm

log = logging.getLogger(__name__)


def answer(question: str, top_k: int = 5, expand_graph: bool = True,
           decompose: bool = False, use_centrality: bool = False,
           driver=None) -> Dict[str, Any]:
    if not question or not question.strip():
        raise ValueError("Empty question")
    subqs: List[str] = [question]
    if decompose:
        from .query_decomposition import decompose as _dec

        subqs = _dec(question)
    # Retrieve per subquestion, merge with dedupe (keep best combined_score).
    pool: Dict[str, Dict] = {}
    for sq in subqs:
        for h in _ret.retrieve(sq, top_k=top_k, expand_graph=expand_graph,
                               use_centrality=use_centrality, driver=driver):
            cid = h["chunk_id"]
            if cid not in pool or h.get("combined_score", 0) > pool[cid].get("combined_score", 0):
                pool[cid] = h
    contexts = sorted(pool.values(), key=lambda r: r.get("combined_score", 0),
                      reverse=True)[:top_k]
    if not contexts:
        log.warning("No context retrieved for: %.80s", question)
        return {"question": question, "answer": "The evidence is insufficient: "
                "no relevant abstracts were retrieved.", "provider": "none",
                "model": "", "contexts": [], "subquestions": subqs}
    gen = _llm.generate(question, contexts)
    return {"question": question, "answer": gen["answer"], "provider": gen["provider"],
            "model": gen["model"], "contexts": contexts, "subquestions": subqs}
