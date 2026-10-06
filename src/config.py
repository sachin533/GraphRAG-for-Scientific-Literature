"""Central configuration. Reads env vars / .env locally and Streamlit secrets on Cloud.

Never hardcode credentials. Use environment variables locally (.env) and
Streamlit Secrets (st.secrets) when deployed to Streamlit Cloud.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


def _get(key: str, default: str = "") -> str:
    # 1. Streamlit secrets (Cloud) take precedence when available.
    try:
        import streamlit as st  # lazy, optional

        if hasattr(st, "secrets") and key in st.secrets:
            val = st.secrets[key]
            if val is not None and str(val) != "":
                return str(val)
    except Exception:
        pass
    return os.getenv(key, default)


@dataclass
class Settings:
    neo4j_uri: str = field(default_factory=lambda: _get("NEO4J_URI", "bolt://localhost:7687"))
    neo4j_user: str = field(default_factory=lambda: _get("NEO4J_USER", "neo4j"))
    neo4j_password: str = field(default_factory=lambda: _get("NEO4J_PASSWORD", ""))
    neo4j_database: str = field(default_factory=lambda: _get("NEO4J_DATABASE", "neo4j"))

    llm_provider: str = field(default_factory=lambda: _get("LLM_PROVIDER", "").lower().strip())
    openai_api_key: str = field(default_factory=lambda: _get("OPENAI_API_KEY", ""))
    openai_model: str = field(default_factory=lambda: _get("OPENAI_MODEL", "gpt-4o-mini"))
    groq_api_key: str = field(default_factory=lambda: _get("GROQ_API_KEY", ""))
    groq_model: str = field(default_factory=lambda: _get("GROQ_MODEL", "llama-3.3-70b-versatile"))
    ollama_host: str = field(default_factory=lambda: _get("OLLAMA_HOST", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: _get("OLLAMA_MODEL", "llama3.1"))

    top_k: int = field(default_factory=lambda: int(_get("TOP_K", "5") or 5))
    expand_graph: bool = field(
        default_factory=lambda: (_get("EXPAND_GRAPH", "true").lower() not in ("0", "false", "no"))
    )
    semantic_sim_threshold: float = field(
        default_factory=lambda: float(_get("SEMANTIC_SIM_THRESHOLD", "0.85") or 0.85)
    )
    vector_candidates: int = field(
        default_factory=lambda: int(_get("VECTOR_CANDIDATES", "50") or 50)
    )

    embedding_model: str = field(
        default_factory=lambda: _get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )
    spacy_model: str = field(default_factory=lambda: _get("SPACY_MODEL", "en_core_sci_sm"))
    n_articles: int = field(default_factory=lambda: int(_get("N_ARTICLES", "5000") or 5000))
    chunk_token_limit: int = field(
        default_factory=lambda: int(_get("CHUNK_TOKEN_LIMIT", "100") or 100)
    )

    def resolve_llm_provider(self) -> str:
        """Return 'openai' | 'groq' | 'ollama'. Raises if none configured."""
        if self.llm_provider in ("openai", "groq", "ollama"):
            return self.llm_provider
        if self.openai_api_key:
            return "openai"
        if self.groq_api_key:
            return "groq"
        # Ollama needs no key; use it as last-resort default (may fail at call time with clear error).
        return "ollama"


settings = Settings()
