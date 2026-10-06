"""Load PubMed abstracts + PubMedQA questions.

Corpus sources (tried in order; first success wins -- everything is real data,
never fabricated):
1. Hugging Face scientific_papers pubmed (spec's repo, then public mirror).
2. Europe PMC REST API: real PubMed abstracts across several biomedical topics.

Questions: PubMedQA pqa_labeled via `datasets`, falling back to the raw
pubmedqa GitHub JSON (parsed with stdlib json, no `datasets` needed).

A local processed copy is saved to data/raw/ for reproducibility.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

ARTICLES_JSONL = os.path.join("data", "raw", "articles.jsonl")
PUBMEDQA_JSONL = os.path.join("data", "raw", "pubmedqa.jsonl")

EPMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
# Topic queries keep the 5000-abstract corpus diverse (each yields ~850 usable).
EPMC_TOPICS = [
    "diabetes treatment",
    "cancer therapy",
    "cardiovascular disease",
    "infectious disease antibiotics",
    "alzheimer neurology",
    "genetics mutation",
]


def _require_datasets():
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("pip install datasets  (needed for data loading)") from exc
    return load_dataset


def _load_hf_pubmed(split: str = "train"):
    load_dataset = _require_datasets()
    for repo in ("alexaapo/scientific_papers", "armanc/scientific_papers"):
        try:
            ds = load_dataset(repo, "pubmed", split=split)
            log.info("Loaded %s/pubmed (%d rows).", repo, len(ds))
            return ds
        except Exception as exc:
            log.warning("source %s/pubmed failed: %s", repo, exc)
    return None


def _epmc_page(query: str, cursor: str, page_size: int = 1000) -> Dict[str, Any]:
    import requests

    params = {"query": f"({query}) AND OPEN_ACCESS:Y", "format": "json",
              "pageSize": page_size, "cursorMark": cursor, "resultType": "core"}
    last = None
    for attempt in range(3):
        try:
            r = requests.get(EPMC_SEARCH, params=params, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            last = exc
            log.warning("Europe PMC request failed (try %d): %s", attempt + 1, exc)
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Europe PMC search failed for {query!r}: {last}")


def load_pubmed_europepmc(n_articles: int = 5000) -> List[Dict[str, Any]]:
    """Fetch real PubMed abstracts from the Europe PMC open API (no key needed)."""
    per_topic = max(200, n_articles // len(EPMC_TOPICS))
    articles: List[Dict[str, Any]] = []
    seen_pmids = set()
    for topic in EPMC_TOPICS:
        if len(articles) >= n_articles:
            break
        quota = per_topic  # per-topic quota keeps the corpus diverse
        got = 0
        cursor, pages = "*", 0
        while got < quota and len(articles) < n_articles and pages < 4:
            data = _epmc_page(topic, cursor)
            for it in data.get("resultList", {}).get("result", []):
                pmid = str(it.get("pmid") or it.get("id") or "")
                abs_text = (it.get("abstractText") or "").strip()
                if not abs_text or pmid in seen_pmids:
                    continue
                seen_pmids.add(pmid)
                articles.append({
                    "article_id": f"PMID:{pmid}",
                    "pmid": pmid,
                    "hf_index": len(articles),
                    "title": (it.get("title") or "")[:500],
                    "abstract": abs_text,
                    "pub_date": str(it.get("pubYear") or it.get("firstPublicationDate", ""))[:10],
                    "section_names": "",
                    "journal": ((it.get("journalInfo") or {}).get("journal") or {}).get("title", ""),
                    "source": f"europepmc:{topic}",
                })
                if len(articles) >= n_articles:
                    break
                got += 1
                if got >= quota:
                    break
            cursor = data.get("nextCursorMark", "")
            pages += 1
            if not cursor:
                break
            time.sleep(0.4)
        log.info("topic %-28s -> %d articles total", topic, len(articles))
    log.info("Europe PMC corpus: %d abstracts.", len(articles))
    if len(articles) < 100:
        raise RuntimeError("Europe PMC returned too few abstracts; aborting.")
    return articles[:n_articles]


def load_scientific_papers(n_articles: int = 5000, split: str = "train",
                            seed: int = 42) -> List[Dict[str, Any]]:
    """Load ~n_articles PubMed records: HF mirror first, Europe PMC fallback."""
    ds = _load_hf_pubmed(split)
    if ds is not None:
        n = min(n_articles, len(ds))
        articles: List[Dict[str, Any]] = []
        for i in range(n):
            row = ds[i]
            body = (row.get("article") or "").strip()
            title = (row.get("abstract") or "").strip()
            articles.append({"article_id": f"pmid_{i:06d}", "hf_index": i,
                             "title": title[:500], "abstract": body,
                             "pub_date": row.get("pub_date", "") or "",
                             "section_names": row.get("section_names", "") or ""})
        kept = [a for a in articles if a["abstract"]]
        log.info("HF corpus: %d articles (%d non-empty).", len(articles), len(kept))
        return kept
    log.info("HF sources unavailable; using Europe PMC.")
    return load_pubmed_europepmc(n_articles)


def _pubmedqa_via_raw_github() -> List[Dict[str, Any]]:
    """Fallback: parse ori_pqal.json from the pubmedqa GitHub repo directly."""
    import requests

    url = "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json"
    log.info("Fetching PubMedQA labeled from %s", url)
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    raw = r.json()
    items = raw.values() if isinstance(raw, dict) else raw
    out: List[Dict[str, Any]] = []
    for pubid, row in (raw.items() if isinstance(raw, dict) else enumerate(items)):
        if not isinstance(row, dict):
            continue
        ctx = row.get("CONTEXTS") or []
        flat = []
        for c in ctx:
            flat.append(c if isinstance(c, str) else json.dumps(c))
        out.append({"question": row.get("QUESTION", ""), "pubid": str(row.get("pubid", pubid)),
                    "answer": row.get("final_decision", ""),
                    "long_answer": row.get("LONG_ANSWER", ""),
                    "context_abstracts": flat})
    log.info("Parsed %d PubMedQA questions from raw GitHub.", len(out))
    return out


def load_pubmedqa(subset: str = "pqa_labeled", split: str = "train") -> List[Dict[str, Any]]:
    """Load PubMedQA labeled questions (datasets lib, else raw GitHub JSON)."""
    try:
        load_dataset = _require_datasets()
        log.info("Loading PubMedQA (%s/%s)...", subset, split)
        ds = load_dataset("qiaojin/PubMedQA", subset, split=split)
        out = []
        for row in ds:
            ctx = row.get("CONTEXTS", row.get("context", []))
            if isinstance(ctx, dict):  # parquet nests as {'contexts': [...]}
                ctx = ctx.get("contexts", [])
            out.append({
                "question": row.get("QUESTION", row.get("question", "")),
                "pubid": str(row.get("pubid", "")),
                "answer": row.get("final_decision", ""),
                "long_answer": row.get("LONG_ANSWER", row.get("long_answer", "")),
                "context_abstracts": [c if isinstance(c, str) else json.dumps(c)
                                      for c in (ctx or [])],
            })
        log.info("Loaded %d PubMedQA questions via datasets.", len(out))
        return out
    except Exception as exc:
        log.warning("datasets PubMedQA failed (%s); using raw GitHub fallback.", exc)
        return _pubmedqa_via_raw_github()


def filter_pubmedqa_to_articles(questions: List[Dict[str, Any]],
                                articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep questions whose evidence is in our corpus.

    1. Exact PMID match when the corpus carries real PMIDs (Europe PMC).
    2. Otherwise text-overlap match on distinctive question keywords (>=2 hits).
    The matched article_id is recorded for Recall@k evaluation.
    """
    import re

    pmid_index = {str(a.get("pmid", "")): a["article_id"] for a in articles if a.get("pmid")}

    def keywords(q: str) -> List[str]:
        toks = re.findall(r"[a-zA-Z]{4,}", q.lower())
        stop = {"what", "which", "does", "with", "from", "have", "that", "this",
                "there", "their", "been", "were", "are", "was", "for", "and", "the"}
        return [t for t in toks if t not in stop][:8]

    lowered = [(a["article_id"], (a["title"] + " " + a["abstract"]).lower()) for a in articles]
    matched: List[Dict[str, Any]] = []
    for q in questions:
        if q.get("pubid") in pmid_index:  # exact: question's own article is in corpus
            q2 = dict(q)
            q2["matched_article_id"] = pmid_index[q["pubid"]]
            q2["match_type"] = "pmid"
            matched.append(q2)
            continue
        kws = keywords(q.get("question", ""))
        if not kws:
            continue
        best_id, best_hits = "", 0
        for aid, text in lowered:
            hits = sum(1 for k in kws if k in text)
            if hits > best_hits:
                best_id, best_hits = aid, hits
        if best_hits >= 2:
            q2 = dict(q)
            q2["matched_article_id"] = best_id
            q2["keyword_hits"] = best_hits
            q2["match_type"] = "keyword"
            matched.append(q2)
    log.info("PubMedQA matched to corpus: %d / %d.", len(matched), len(questions))
    return matched


def save_jsonl(rows: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info("Saved %d rows -> %s", len(rows), path)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_or_download(n_articles: int = 5000, use_cache: bool = True) -> Dict[str, Any]:
    """Main entry: download (or reuse cached) articles + PubMedQA, filter, save copies."""
    articles: Optional[List[Dict[str, Any]]] = None
    questions: Optional[List[Dict[str, Any]]] = None
    if use_cache and os.path.exists(ARTICLES_JSONL):
        log.info("Reusing cached %s", ARTICLES_JSONL)
        articles = load_jsonl(ARTICLES_JSONL)
        if len(articles) > n_articles:
            articles = articles[:n_articles]
    else:
        articles = load_scientific_papers(n_articles=n_articles)
        save_jsonl(articles, ARTICLES_JSONL)

    if use_cache and os.path.exists(PUBMEDQA_JSONL):
        log.info("Reusing cached %s", PUBMEDQA_JSONL)
        questions = load_jsonl(PUBMEDQA_JSONL)
    else:
        questions = load_pubmedqa()
        save_jsonl(questions, PUBMEDQA_JSONL)

    matched = filter_pubmedqa_to_articles(questions, articles)
    log.info("articles=%d pubmedqa=%d matched=%d", len(articles), len(questions), len(matched))
    return {"articles": articles, "pubmedqa": questions, "matched": matched}
