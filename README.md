# GraphRAG for Scientific Literature

Answer complex biomedical questions over ~5000 PubMed abstracts using **vector search + knowledge-graph expansion + LLM generation**.

## 1. What is GraphRAG?

GraphRAG = Retrieval-Augmented Generation where retrieved passages are connected through an explicit **knowledge graph** (here: chunks ↔ scientific entities). The graph lets the system discover evidence that shares entities with the initial vector hits, even when word overlap is low.

## 2. Why normal RAG is not enough here

Plain vector RAG returns the k nearest chunks to the question and stops. In science, the answer often spans papers: drug A ↔ disease B in one abstract, disease B ↔ gene C in another. Vector-only retrieval misses the second hop. Graph expansion follows `Chunk→Entity←Chunk` links (depth ≤ 2) to recover it.

## 3. Semantic chunking

- **fixed** (~100 tokens, no overlap): baseline.
- **sentence**: merge whole sentences up to the token limit (no mid-sentence cuts).
- **semantic**: embed sentences with `all-MiniLM-L6-v2`, cluster with HDBSCAN (fallback: AgglomerativeClustering/cosine), one chunk per cluster. Paraphrases stay together; topic shifts split.

## 4. Why all-MiniLM-L6-v2

384-dim, fast on CPU, strong for semantic similarity benchmarks, and small enough to store one vector per chunk in Neo4j. The **same** model embeds questions, so query and chunks live in one space.

## 5. Embeddings

Dense vectors capturing meaning (`src/embeddings.py`, singleton model, `embed_text`/`embed_chunks`). Cosine similarity in Neo4j's vector index ranks chunks by semantic closeness to the question.

## 6. Why Neo4j

One store for (a) the knowledge graph (articles/chunks/entities + relations), (b) the **vector index** on `Chunk.embedding`, and (c) graph algorithms (GDS PageRank optional). No fake Python list search: retrieval calls `db.index.vector.queryNodes`.

## 7. Graph schema

```
(:Article {article_id, title, pub_date})
(:Chunk {chunk_id, text, embedding(384), strategy, article_id})
(:Entity {name(lowercased key), display_name, type})

(:Article)-[:HAS_CHUNK]->(:Chunk)
(:Chunk)-[:MENTIONS]->(:Entity)
(:Chunk)-[:SEMANTIC_SIMILAR {score}]->(:Chunk)   # within-article, cosine > 0.85
```

## 8. Vector search

`CALL db.index.vector.queryNodes('chunk_embedding', k, $qemb)` returns top-k chunks with cosine scores. Real index, 384 dims, cosine.

## 9. Graph expansion

For each seed chunk: `MATCH (s)-[:MENTIONS]->(e)<-[:MENTIONS]-(n)` (depth ≤ 2). Neighbours sharing entities get `graph_score = shared/max_shared`. `combined = 0.7·vector + 0.3·graph (+ 0.1·centrality optional)`, deduped by `chunk_id`.

## 10. Final context

Top-k merged chunks with `{chunk_id, article_id, text, vector_score, graph_score, combined_score, entities}`.

## 11. LLM answer

`SYSTEM (grounding rules) + CONTEXT + QUESTION → LLM`. Prompt forbids invention; insufficient evidence → "evidence is insufficient". Provider configurable: OpenAI / Groq / Ollama via env or Streamlit secrets.

## 12. Architecture

```
OFFLINE: PubMed abstracts → chunk → embed → entities → Neo4j (+vector index, SEMANTIC_SIMILAR)
ONLINE:  question → embed → Neo4j vector search → graph expansion → dedupe/re-rank
         → context → LLM → answer  (+ Streamlit UI, eval on PubMedQA)
```

## 13. Installation

```powershell
py -m venv .venv; .\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
py -m spacy download en_core_web_sm   # fallback NER
# optional SciSpacy biomedical NER:
py -m pip install "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz"
copy .env.example .env   # then fill secrets
```

## 14. Neo4j setup

- Easiest: [Neo4j AuraDB](https://neo4j.com/cloud/aura/) free tier, or local Neo4j Desktop / Docker:
  `docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/changeme neo4j:5`
- Set `NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD` in `.env` (locally) or Streamlit Secrets (Cloud).

## 15. Dataset setup

```powershell
py scripts/download_data.py --n 5000
```

## 16. Build the graph

```powershell
py scripts/create_chunks.py --n 5000
py scripts/build_graph.py --strategy semantic --clear
```

## 17. Vector index

Created automatically by `build_graph.py`; or `py scripts/create_vector_index.py`.

## 18. Evaluation

```powershell
py evaluation/evaluate.py --n 200 --topk 10   # Recall@5/10, MRR, ROUGE-L, BERTScore
```

## 19. Streamlit

```powershell
streamlit run app.py
```

## 20. Example questions

- What is the relationship between diabetes treatment and cardiovascular risk?
- Does metformin reduce cancer incidence in diabetic patients?
- Which genes are associated with breast cancer prognosis?
- Is TP53 mutation predictive of chemotherapy response?

## 21. Known limitations

- `scientific_papers` row IDs are not real PubMed IDs; PubMedQA overlap is estimated by keyword match (logged, inspectable).
- `SEMANTIC_SIMILAR` is within-article only (global O(n²) is infeasible at 5000 abstracts).
- SciSpacy model is large; the `en_core_web_sm` fallback extracts fewer biomedical entities.
- Evaluation needs a populated Neo4j + an LLM key; otherwise retrieval-only metrics still run.
- No internet search: answers use only retrieved abstracts by design.
