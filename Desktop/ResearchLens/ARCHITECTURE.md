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
│  50 type-balanced questions         │
│  (13F · 13C · 12S · 12P)           │
│  Faithfulness · Correctness · AUC   │
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

## Generation Layer

**Interactive pipeline (Notebooks 05/06, Streamlit demo):** `max_new_tokens=300`, `temperature=0.3`, prompt cap 3500 tokens. Temperature > 0 is required for the consistency signal in uncertainty quantification (needs two stochastic samples to compare).

**Evaluation run (Notebook 07 v2):** `max_new_tokens=150`, `temperature=0.1`, `batch_size=4` with left-padding. Shorter output limit and near-deterministic temperature chosen for throughput (250 generations × 4 strategies) and measurement consistency, not quality. These settings apply only to evaluation and do not reflect what the interactive pipeline serves.

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

**Strategy-aware normalisation (v2):** hybrid_rerank scores use sigmoid; hybrid_rag uses RRF_MAX normalisation; naive/semantic use cosine clamp to [0,1]. Prevents confidence > 100 for reranker answers and artificially low confidence for hybrid answers.

**Measured calibration:** ROC-AUC = 0.782 on 50 type-balanced questions. GREEN band (confidence ≥ 80): 88.9% correct. RED band (confidence < 50): 0.0% correct. Gap: +88.9pp.

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

**Knowledge base:** 505 ML/AI papers from arXiv, fetched programmatically via the `arxiv` Python library. Topics: large language models (0–168), RAG systems (169–337), fine-tuning and attention (338–504).

**Evaluation set:** 50 type-balanced questions hand-authored against the 505-paper corpus — 13 factual, 13 comparative, 12 consensus, 12 procedural — with embedded `scoring_config` (formula: sentence_mean_of_max_cosine; thresholds: 0.55 factual/comparative/procedural, 0.45 consensus).

**Chunking stats (post reference-section filtering):**
- Fixed chunks: 8,933 total, ~512 words average
- Semantic chunks: 9,561 total, ~400 words average
- Reference pages removed: 39.0% of raw pages (page-level cascade on standalone "References" header; pre-header content on cutoff page salvaged)

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
