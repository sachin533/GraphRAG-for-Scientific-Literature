"""Scientific entity extraction with SciSpacy / spaCy.

Tries models in order: SPACY_MODEL env (default en_core_sci_sm) -> en_core_web_sm.
Never hardcodes entities: everything comes from the NLP pipeline.
Output: [{'name': ..., 'type': ...}] deduplicated, case-normalised for graph keys.
"""
from __future__ import annotations

import logging
from typing import Dict, List

log = logging.getLogger(__name__)

_NLP = None
_ACTIVE_MODEL = ""


def get_nlp():
    global _NLP, _ACTIVE_MODEL
    if _NLP is not None:
        return _NLP
    import spacy
    from .config import settings

    candidates = [settings.spacy_model, "en_core_sci_sm", "en_core_web_sm"]
    seen = []
    for name in candidates:
        if not name or name in seen:
            continue
        seen.append(name)
        try:
            _NLP = spacy.load(name)
            _ACTIVE_MODEL = name
            log.info("Entity extractor loaded: %s", name)
            return _NLP
        except OSError:
            log.warning("spaCy model '%s' not installed; trying next.", name)
    # Last resort: blank English + sentencizer (entity-less, caller handles empty list).
    import spacy as _sp

    _NLP = _sp.blank("en")
    try:
        _NLP.add_pipe("sentencizer")
    except Exception:
        pass
    _ACTIVE_MODEL = "blank(en)"
    log.warning("No NER model available; entity extraction will return []. "
                "Install with: pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz")
    return _NLP


def active_model() -> str:
    if _NLP is None:
        get_nlp()
    return _ACTIVE_MODEL


def extract_entities(text: str, max_ents: int = 20) -> List[Dict[str, str]]:
    """Extract entities from a chunk of text. Real NLP output, never hardcoded."""
    nlp = get_nlp()
    if not text or not text.strip():
        return []
    doc = nlp(text[:5000])  # cap for speed
    seen, out = set(), []
    for ent in getattr(doc, "ents", []):
        name = " ".join(ent.text.strip().split())
        if len(name) < 2 or len(name) > 80:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append({"name": name, "type": ent.label_ or "ENTITY"})
        if len(out) >= max_ents:
            break
    return out


def extract_for_chunks(chunks: List[Dict], text_key: str = "text") -> List[Dict]:
    """Attach 'entities' list to each chunk dict (with progress logging)."""
    get_nlp()
    total_ents = 0
    for i, c in enumerate(chunks):
        ents = extract_entities(c.get(text_key) or "")
        c["entities"] = ents
        total_ents += len(ents)
        if (i + 1) % 500 == 0:
            log.info("Entity extraction: %d/%d chunks, %d entities so far.",
                     i + 1, len(chunks), total_ents)
    log.info("Entity extraction done: %d chunks, %d entity mentions (%s).",
             len(chunks), total_ents, active_model())
    return chunks
