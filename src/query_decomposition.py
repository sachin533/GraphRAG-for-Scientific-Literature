"""Optional query decomposition for complex questions.

Complex question -> 2-4 focused subquestions -> retrieve per subquestion -> merge.
Simple questions pass through untouched. Decomposition uses the LLM when available
with a deterministic heuristic fallback (splitting on conjunctions/comparisons).
"""
from __future__ import annotations

import logging
import re
from typing import List

log = logging.getLogger(__name__)

_COMPLEX_HINTS = re.compile(
    r"\b(relationship between|compare|versus| vs\.? |difference|effect of|impact of|"
    r"association|interaction|mechanism|treatment and|risk of)\b", re.I)


def is_complex(question: str) -> bool:
    q = (question or "").strip()
    return len(q.split()) >= 15 or bool(_COMPLEX_HINTS.search(q)) or q.count("?") > 1


def heuristic_split(question: str, max_sub: int = 3) -> List[str]:
    parts = re.split(r"\s+and\s+|\s+vs\.?\s+|\?\s*|\s*;\s*", question.strip())
    out = [p.strip(" .?") + "?" for p in parts if len(p.split()) >= 4][:max_sub]
    return out or [question]


def decompose(question: str, max_sub: int = 3, use_llm: bool = True) -> List[str]:
    """Return subquestions (or [question] when simple / LLM unavailable)."""
    if not is_complex(question):
        return [question]
    if use_llm:
        try:
            from . import llm as _llm

            msgs = [{"role": "system",
                     "content": "Split the biomedical question into 2-4 focused subquestions, "
                                "one per line, no numbering."},
                    {"role": "user", "content": question}]
            # Reuse provider clients lightly: call generate with empty context.
            res = _llm.generate(question="Split into subquestions:\n" + question,
                                contexts=[], max_tokens=200)
            lines = [ln.strip("- •1234567890. ") for ln in res["answer"].splitlines()]
            subs = [ln if ln.endswith("?") else ln + "?" for ln in lines if len(ln.split()) >= 4]
            if 2 <= len(subs) <= max_sub + 1:
                log.info("Decomposed into %d subquestions via LLM.", len(subs))
                return subs[:max_sub]
        except Exception as exc:
            log.info("LLM decomposition failed (%s); using heuristic.", exc)
    subs = heuristic_split(question, max_sub)
    log.info("Decomposed into %d subquestions (heuristic).", len(subs))
    return subs
