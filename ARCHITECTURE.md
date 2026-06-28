# ResearchLens — Architecture

## System Pipeline

Every user query passes through six sequential layers:

```
User Query
    │
    ▼
┌─────────────────────────────────────┐
│         INGESTION LAYER             │
│  arXiv PDFs → PyMuPDF → clean text  │
│  Fixed chunks (512 tokens, 50 overlap)│
│  Semantic chunks (sentence boundaries)│
│  → ChromaDB (dense) + BM25 (sparse)  │
└──────────────────┬──────────────────┘
                   │
    ▼
┌─────────────────────────────────────┐
│         QUERY ROUTER                │
│  Keyword classifier (primary)       │
│  Mistral-7B LLM fallback (ambiguous)│
│  → factual / comparative /          │
│     consensus / procedural          │
└──────────────────┬──────────────────┘
                   │
         ┌─────────┴──────────┐
         │                    │
    ▼               ▼
┌──────────────┐  ┌──────────────────────┐
│ Strategy A/B │  │ Strategy C/D         │
│ Naive/Semantic│  │ Hybrid+Reranker      │
│ cosine sim   │  │ BM25+Dense via RRF   │
│              │  │ + cross-encoder      │
└──────┬───────┘  └──────────┬───────────┘
       └─────────┬───────────┘
                 │
    ▼
┌─────────────────────────────────────┐
│         GENERATION LAYER            │
│  Mistral-7B-Instruct-v0.2           │
│  Kaggle T4 GPU · fp16               │
│  max_new_tokens=300 · temp=0.3      │
│  Grounded prompt: context-only      │
└──────────────────┬──────────────────┘
                   │
    ▼
┌─────────────────────────────────────┐
│     UNCERTAINTY QUANTIFICATION      │
│  Signal 1: retrieval confidence     │
│    (mean cosine sim of top-k chunks)│
│  Signal 2: faithfulness proxy       │
│    (token overlap answer↔context)   │
│  Signal 3: consistency              │
│    (2x generation cosine similarity)│
│  → 0–100 score → GREEN/YELLOW/RED  │
│  RED (<50): refuse to answer        │
└──────────────────┬──────────────────┘
                   │
    ▼
┌─────────────────────────────────────┐
│         EVALUATION LAYER            │
│  50 QASPER questions                │
│  Custom metrics (no external API)   │
│  Faithfulness · Precision · Relevance│
│  Latency benchmark per strategy     │
└─────────────────────────────────────┘
```

---

## Four Retrieval Strategies

### Strategy A — Naive RAG
- Chunk: fixed 512-word windows, 50-word overlap
- Index: ChromaDB with cosine similarity
- Retrieve: embed query → cosine search → top-k chunks
- Best for: simple factual questions

### Strategy B — Semantic RAG
- Chunk: sentence-boundary splits, 50–400 words per chunk
- Index: ChromaDB with cosine similarity
- Retrieve: same as A but on sentence-coherent chunks
- Best for: procedural questions, narrative explanations

### Strategy C — Hybrid RAG
- Chunk: fixed chunks (same as A)
- Index: ChromaDB (dense) + BM25 (sparse)
- Retrieve: run both searches, merge via Reciprocal Rank Fusion
- RRF score: `1/(rank_dense + 60) + 1/(rank_bm25 + 60)`
- Best for: queries with specific technical terms

### Strategy D — Hybrid + Reranker
- Retrieve: Strategy C with top_k=20 candidates
- Rerank: cross-encoder/ms-marco-MiniLM-L-6-v2 scores all 20 pairs
- Return: top-k by cross-encoder score
- Best for: complex comparative questions where precision matters most

---

## Query Router

Two-tier classification:

**Tier 1 — Keyword classifier (95% confidence)**
Signal word lookup across 4 query type dictionaries. Runs in microseconds.

**Tier 2 — LLM fallback (85% confidence)**
When keyword confidence < 0.7, routes to Mistral-7B with a zero-shot prompt asking for a one-word category label. Adds ~200ms latency.

**Routing map:**
```
factual    → Strategy A (Naive RAG)
comparative → Strategy D (Hybrid + Reranker)
consensus  → Strategy C (Hybrid RAG) + contradiction handler
procedural → Strategy B (Semantic RAG)
```

---

## Contradiction Detection Pipeline

Activated for consensus query type:

```
1. Retrieve top-20 chunks across all papers (hybrid search)
2. Extract factual claims with spaCy
   - Filter: 15–55 words, must have subject + verb
   - Remove: citations, URLs, figure captions
3. Build cross-paper pairs
   - Top-3 most topic-relevant claims per paper
   - All pairs across different papers
4. Pre-filter by embedding similarity (threshold: 0.40)
   - Eliminates pairs too dissimilar to plausibly contradict
5. Run NLI on filtered pairs (facebook/bart-large-mnli)
   - Both directions: A→B and B→A
   - Average contradiction scores
6. Surface pairs above contradiction threshold (0.45)
```

---

## Uncertainty Quantification Detail

Three signals combined into a single 0–100 score:

```python
confidence = (
    retrieval_score * 0.4 +   # mean cosine sim of top-k
    faithfulness    * 0.3 +   # token overlap answer↔context
    consistency     * 0.3     # cosine sim between 2 generation runs
) * 100
```

**Known limitation:** hybrid_rag retrieval scores (RRF values ~0.016) are not comparable to naive/semantic cosine scores (0.5–0.7). This artificially depresses confidence for hybrid strategy answers. Fix: normalise per-strategy before combining.

---

## Models

| Model | Parameters | Role | Runs On |
|-------|-----------|------|---------|
| Mistral-7B-Instruct-v0.2 | 7B | Answer generation | Kaggle T4 GPU (fp16) |
| all-MiniLM-L6-v2 | 22M | Embeddings + clustering | CPU / GPU |
| ms-marco-MiniLM-L-6-v2 | 22M | Cross-encoder reranking | CPU / GPU |
| facebook/bart-large-mnli | 400M | NLI contradiction detection | GPU preferred |
| en_core_web_sm (spaCy) | 12MB | Claim extraction (grammar) | CPU |

---

## Data

**Knowledge base:** 100 ML/AI papers from arXiv, fetched programmatically via the `arxiv` Python library. Topics: large language models, RAG, transformers, fine-tuning, attention mechanisms.

**Evaluation set:** 50 QA pairs from QASPER (allenai/qasper), filtered to free-form answers with length > 10 characters.

**Chunking stats:**
- Fixed chunks: 3,205 total, ~512 words average
- Semantic chunks: 3,427 total, ~400 words average

---

## Infrastructure

Everything runs on free tiers:

```
Kaggle Notebooks (T4 GPU, 30 hrs/week)
    └── Experiments, model inference, index building

Kaggle Datasets
    └── Persistent storage for indexes across sessions

GitHub
    └── Code + notebooks + results

Streamlit Cloud
    └── Public demo (retrieval only, no live generation)
```

Total cost: $0.
