"""Streamlit app: GraphRAG for Scientific Literature.  Run:  streamlit run app.py"""
import logging

import streamlit as st

from src.config import settings
from src.neo4j_loader import get_driver, stats

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

st.set_page_config(page_title="GraphRAG for Scientific Literature", layout="wide")
st.title("GraphRAG for Scientific Literature")
st.caption("Vector search (Neo4j) + shared-entity graph expansion + LLM answer over PubMed abstracts.")

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Top K", 1, 15, settings.top_k)
    expand = st.checkbox("Graph expansion", value=settings.expand_graph)
    decompose = st.checkbox("Query decomposition (complex Qs)", value=False)
    centrality = st.checkbox("Graph centrality re-ranking", value=False)
    provider = st.text_input("LLM provider (blank=auto)", value=settings.llm_provider or "")
    if provider:
        settings.llm_provider = provider.lower().strip()
    try:
        from src.llm import provider_name

        st.info(f"LLM provider: **{provider_name()}**")
    except Exception as exc:
        st.warning(str(exc))
    if st.button("Graph stats"):
        try:
            st.json(stats(get_driver()))
        except Exception as exc:
            st.error(f"Neo4j unavailable: {exc}")

question = st.text_area("Scientific question",
                        placeholder="e.g. What is the relationship between diabetes treatment and cardiovascular risk?")
ask = st.button("Ask", type="primary")

if ask and question.strip():
    with st.spinner("Retrieving + generating..."):
        try:
            from src import rag_pipeline as P

            out = P.answer(question, top_k=top_k, expand_graph=expand,
                           decompose=decompose, use_centrality=centrality)
        except Exception as exc:
            log.exception("Pipeline failed")
            st.error(f"Pipeline error: {exc}")
            st.stop()
    st.subheader("Answer")
    st.write(out["answer"])
    st.caption(f"provider={out.get('provider')} model={out.get('model')} "
               f"subquestions={out.get('subquestions')}")
    st.subheader("Retrieved Sources")
    for i, c in enumerate(out.get("contexts", []), 1):
        with st.expander(f"Chunk {i}: {c.get('chunk_id')} (score {c.get('combined_score', 0):.3f})"):
            st.write(f"**Article:** {c.get('article_id')}  | vector={c.get('vector_score', 0):.3f} "
                     f"graph={c.get('graph_score', 0):.3f} combined={c.get('combined_score', 0):.3f}")
            st.write(c.get("text", ""))
            ents = c.get("entities") or []
            via = c.get("via_entities") or []
            if ents:
                st.write("Entities: " + ", ".join(map(str, ents[:15])))
            if via:
                st.write("Connected via: " + ", ".join(map(str, via[:15])))
    st.subheader("Knowledge Graph")
    try:
        from pyvis.network import Network
        import streamlit.components.v1 as components
        import tempfile, os

        net = Network(height="500px", width="100%", notebook=False)
        for c in out.get("contexts", []):
            cid = c.get("chunk_id", "?")
            net.add_node(cid, label=cid[-12:], title=(c.get("text", "")[:300]),
                         color="#5b9bd5")
            for e in (c.get("entities") or [])[:6] + (c.get("via_entities") or [])[:6]:
                name = str(e)
                try:
                    net.add_node("E:" + name, label=name[:20], color="#ed7d31")
                except Exception:
                    pass
                net.add_edge(cid, "E:" + name)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as f:
            net.save_graph(f.name)
            html = open(f.name, encoding="utf-8").read()
            os.unlink(f.name)
        components.html(html, height=520)
    except Exception as exc:
        st.warning(f"Graph visualisation unavailable: {exc}")
elif ask:
    st.warning("Enter a question first.")
