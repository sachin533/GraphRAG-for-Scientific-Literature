"""Configurable LLM client: OpenAI / Groq / Ollama. No hardcoded keys.

Provider resolution: explicit LLM_PROVIDER env, else first available key
(OpenAI > Groq), else Ollama (local). All errors are raised with clear messages
(never silently swallowed) so the UI can display them.
"""
from __future__ import annotations

import logging

from .config import settings

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a careful biomedical assistant. Answer ONLY using the provided scientific "
    "context excerpts. Rules:\n"
    "- Do not invent facts, numbers, or citations.\n"
    "- If the context lacks enough evidence, say the evidence is insufficient.\n"
    "- Prefer information from the retrieved abstracts; quote chunk/article ids when useful.\n"
    "- Be concise and precise. Never claim you searched the internet.\n"
    "- Never fabricate references, DOIs, or PubMed IDs."
)


def build_prompt(question: str, contexts: list[dict]) -> list[dict]:
    ctx_lines = []
    for i, c in enumerate(contexts, 1):
        ctx_lines.append(
            f"[Chunk {i} | {c.get('chunk_id', '?')} | article {c.get('article_id', '?')} "
            f"| score {c.get('combined_score', c.get('vector_score', 0)):.3f}]\n{c.get('text', '')}"
        )
    context_block = "\n\n".join(ctx_lines) if ctx_lines else "(no context retrieved)"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",
         "content": f"Scientific context:\n{context_block}\n\nQuestion: {question}\n\n"
                    "Answer using only the context above. If insufficient, say so."},
    ]


def provider_name() -> str:
    return settings.resolve_llm_provider()


def generate(question: str, contexts: list[dict], max_tokens: int = 512,
             temperature: float = 0.1) -> dict:
    """Generate an answer. Returns {'answer', 'provider', 'model'}."""
    prov = provider_name()
    messages = build_prompt(question, contexts)
    log.info("LLM generate via %s (%d context chunks).", prov, len(contexts))
    if prov == "openai":
        return _gen_openai(messages, max_tokens, temperature)
    if prov == "groq":
        return _gen_groq(messages, max_tokens, temperature)
    return _gen_ollama(messages, max_tokens, temperature)


def _gen_openai(messages, max_tokens, temperature):
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set (and LLM_PROVIDER=openai).")
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)
    resp = client.chat.completions.create(model=settings.openai_model, messages=messages,
                                          max_tokens=max_tokens, temperature=temperature)
    return {"answer": resp.choices[0].message.content.strip(),
            "provider": "openai", "model": settings.openai_model}


def _gen_groq(messages, max_tokens, temperature):
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not set (and LLM_PROVIDER=groq).")
    from groq import Groq

    client = Groq(api_key=settings.groq_api_key)
    resp = client.chat.completions.create(model=settings.groq_model, messages=messages,
                                          max_tokens=max_tokens, temperature=temperature)
    return {"answer": resp.choices[0].message.content.strip(),
            "provider": "groq", "model": settings.groq_model}


def _gen_ollama(messages, max_tokens, temperature):
    import requests

    url = settings.ollama_host.rstrip("/") + "/api/chat"
    payload = {"model": settings.ollama_model, "messages": messages,
               "stream": False, "options": {"num_predict": max_tokens, "temperature": temperature}}
    try:
        r = requests.post(url, json=payload, timeout=120)
        r.raise_for_status()
    except Exception as exc:
        raise RuntimeError(
            f"Ollama request failed at {url} (model={settings.ollama_model}). "
            "Is Ollama running? Set OLLAMA_HOST/OLLAMA_MODEL or use another provider."
        ) from exc
    data = r.json()
    text = (data.get("message") or {}).get("content", "").strip()
    if not text:
        raise RuntimeError(f"Empty response from Ollama: {data}")
    return {"answer": text, "provider": "ollama", "model": settings.ollama_model}
