# ResearchLens

An intelligent research paper analysis system that routes queries to optimal retrieval strategies, quantifies answer confidence, and surfaces contradictions across papers — evaluated rigorously with real metrics.

**Built entirely on free infrastructure. Zero cost.**

---

## What Makes This Different From Basic RAG

Most RAG projects upload a PDF and ask questions. ResearchLens goes further:

| Basic RAG | ResearchLens |
|-----------|-------------|
| Single retrieval strategy | 4 strategies + automatic routing |
| "It works great!" (vibes) | Measured accuracy across 50 QASPER questions |
| No confidence signal | Calibrated uncertainty score per answer |
| Collapses contradictions | Surfaces and explains when papers disagree |

---

## Four Core Components

**1. Adaptive Query Routing**
Classifies every incoming question as factual / comparative / consensus / procedural and dispatches it to the retrieval strategy most likely to succeed. Routing accuracy: 82% on a 50-question type-balanced evaluation set using an embedding-centroid classifier — an earlier 72% figure used keyword matching, and an even earlier 100% figure came from a self-authored keyword test that shared an author with the classifier (documented honestly; see calibration section below).

**2. Multi-Strategy Retrieval**
Four distinct RAG pipelines implemented and compared:
- Strategy A — Naive RAG: fixed 512-token chunks + cosine similarity (baseline)
- Strategy B — Semantic RAG: sentence-boundary chunking for narrative coherence
- Strategy C — Hybrid RAG: BM25 + dense retrieval fused via Reciprocal Rank Fusion
- Strategy D — Hybrid + Reranker: C + cross-encoder re-scoring of top-20 candidates

**3. Uncertainty Quantification**
Every answer carries a 0–100 confidence score from three signals:
- Retrieval confidence: cosine similarity of top-k chunks to query
- Faithfulness proxy: token overlap between answer and retrieved context
- Consistency: semantic similarity between two generation runs at temperature > 0

Answers below 50 confidence are flagged or refused rather than served.

**4. Contradiction Detection**
When a consensus query arrives ("do papers agree on X?"), the system extracts factual claims across papers, groups semantically similar claims, and runs NLI (BART-large-mnli) to surface genuine disagreements with source attribution.

---

## Results

Evaluated on 70 type-balanced questions (13 factual · 13 comparative · 12 consensus · 32 procedural) against the 505-paper corpus, using Mistral-7B-Instruct-v0.2:

| Strategy | Faithfulness ↑ | Ctx Precision ↑ | Ans Relevance ↑ | Correctness ↑ | % Correct ↑ | Latency |
|----------|---------------|----------------|----------------|--------------|------------|---------|
| A · Naive RAG | 0.731 | 1.000† | 0.532 | 0.467 | 37% | 9,480ms |
| B · Semantic RAG | 0.755 | 1.000† | 0.555 | 0.501 | 43% | 9,747ms |
| C · Hybrid RAG | 0.807 | 0.871 | 0.528 | 0.481 | 39% | 11,137ms |
| D · Hybrid + Reranker | **0.845** | 0.894 | 0.563 | **0.544** | **54%** | 11,536ms |
| Adaptive Router | 0.783 | **0.946** | 0.549 | 0.506 | 43% | 10,735ms |

† Context Precision ≈ 1.0 for naive/semantic is a threshold artifact — questions grounded in this specific corpus push all chunk cosine similarities above the 0.4 threshold. Correctness (sentence_mean_of_max_cosine) is the primary quality signal.

**Findings (70-question set):** Hybrid + Reranker leads on both faithfulness (0.845) and correctness (0.544, 54% correct) — the reranker's value is clearest on the harder, procedural-heavy set. Absolute correctness is lower than the earlier 50-question set, and a like-for-like per-type check confirms this is the eval getting harder, not a regression: factual/comparative/consensus are the *same questions* in both sets and their correctness held steady within run-to-run variance (±1 question, inconsistent direction across strategies). The 20 newly-added procedural questions are simply harder — 15–20% answerable versus 66–83% for the original 12 — while the original 12 held under a fixed strategy (83%→75%). Router accuracy on this set is **74%** (52/70): the embedding-centroid router scored 82% on the 50-question set its centroids were built from, and 74% here where the 20 added procedural questions are **out-of-sample** for the centroids — a more conservative, more defensible generalization estimate.

### Confidence Calibration

Confidence weights fitted by logistic regression on the eval set, extended with a procedural×answer-relevance interaction term to address a diagnosed failure mode. On an expanded 70-question set (32 procedural, up from 12), validated against actual answer correctness (ROC-AUC = **0.796**):

| Band | Confidence | N | % Correct | Interpretation |
|------|-----------|---|-----------|----------------|
| GREEN | ≥ 80 | 3 | 33% | High confidence (rarely expressed) |
| YELLOW | 50–79 | 26 | 73% | Moderate confidence |
| RED | < 50 | 41 | 24% | Low confidence — refusal appropriate |

The interaction coefficient (`is_procedural × answer_relevance`) strengthened from −0.064 at n=12 procedural to **−0.610** at n=32 — same sign, ~9.5× larger — and the extended model's AUC edge over the plain logistic regression grew from +0.01 to +0.06. The payoff: **no procedural answer now reaches the GREEN band**, closing the diagnosed failure mode where high-relevance-but-wrong procedural answers were scored over-confidently. Honest caveats: the GREEN band is small (n=3) so its purity is noise-dominated, and overall correctness is lower than on the 50-question set because procedural — the hardest query type at 25% correct — now makes up 46% of the eval rather than 24%. YELLOW is well-calibrated (73%) and RED correctly concentrates wrong answers.

> All figures use Mistral-7B-Instruct-v0.2, the project's consistent generator. An earlier refit on v0.1 was discarded as a methodology confound.

---

## Tech Stack

| Component | Tool | Cost |
|-----------|------|------|
| Compute | Kaggle T4 GPU (30 hrs/week) | Free |
| Generation | Mistral-7B-Instruct-v0.2 | Free |
| Embeddings | all-MiniLM-L6-v2 | Free |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | Free |
| NLI | facebook/bart-large-mnli | Free |
| Vector store | ChromaDB | Free |
| Keyword search | BM25 (rank_bm25) | Free |
| Papers | arXiv API (505 ML/AI papers) | Free |
| Evaluation | QASPER benchmark + custom metrics | Free |
| Demo | Streamlit Cloud | Free |

---

## Repository Structure

```
researchlens/
├── notebooks/
│   ├── 01_data_collection.ipynb       — arXiv fetch + QASPER load
│   ├── 02_ingestion_pipeline.ipynb    — parse + chunk + embed + ChromaDB + BM25
│   ├── 03_retrieval_strategies.ipynb  — all 4 strategies + latency benchmark
│   ├── 04_query_router.ipynb          — classifier + routing + accuracy eval
│   ├── 05_generation_uncertainty.ipynb — Mistral inference + confidence scoring
│   ├── 06_contradiction_detection.ipynb — NLI pipeline + cross-paper analysis
│   └── 07_evaluation.ipynb            — full comparison table + custom metrics
│
├── app/
│   ├── streamlit_app.py               — demo UI
│   ├── requirements.txt
│   └── data/                          — indexes + evaluation results
│
└── results/
    ├── comparison_table.csv           — strategy comparison with real numbers
    └── full_eval_results.json         — complete evaluation output
```

---

## Running the Demo Locally

```bash
git clone https://github.com/HarshAmin782/researchlens.git
cd researchlens/app
pip install -r requirements.txt
streamlit run streamlit_app.py
```

First run takes 2–3 minutes to rebuild ChromaDB from the saved indexes. After that it's cached.

---

## Reproducing the Experiments

All experiments run on Kaggle free tier (T4 GPU, 30 hrs/week). Run notebooks in order:

1. Open each notebook on Kaggle
2. Add `researchlens-dataset` as input (contains pre-built indexes)
3. Enable GPU T4 for notebooks 02–07
4. Run all cells

GPU budget used: approximately 28 of the 30 hr/week free limit across all 7 notebooks.

---

## Design Decisions

**Why not a standard RAG project?**
Basic RAG is commoditised. Adding adaptive routing, uncertainty quantification, and contradiction detection makes the system answer three questions a senior engineer immediately asks: which retrieval strategy should I use, how do I know the answer is correct, and what happens when papers disagree?

**Why Kaggle over local?**
The device used for development cannot run 7B-parameter models. Kaggle provides a free T4 GPU that runs Mistral-7B in fp16 comfortably. Using cloud GPUs is industry-standard practice.

**Why Mistral-7B and not GPT-4?**
GPT-4 costs money. Mistral-7B is open-source, runs on Kaggle's free GPU, and produces competitive results on retrieval-augmented tasks. The choice demonstrates cost vs capability awareness.

**Why custom evaluation metrics?**
RAGAS 0.4 requires an external LLM API for all metrics and the free tier (Groq) hits rate limits when evaluating 50 samples in parallel. Custom metrics — token overlap faithfulness, embedding cosine precision, query-answer relevance — run entirely on the T4 with no API dependency and produce reproducible, explainable numbers.

---

## What's Next

- ~~Expand corpus from 100 to 500+ papers~~ ✓ Done — 505 papers, 18,494 chunks (8,933 fixed + 9,561 semantic; reference sections filtered at ingestion)
- ~~Type-balanced evaluation set~~ ✓ Done — 50 questions across 4 query types, corpus-grounded
- ~~Confidence calibration~~ ✓ Done — AUC 0.782, GREEN/YELLOW/RED bands validated
- ~~Embedding-centroid router~~ ✓ Done — 82% accuracy (vs 72% keyword baseline); procedural routing 25% → 92%
- ~~Logistic-regression confidence combiner~~ ✓ Done — AUC 0.796 (extended LR with procedural×relevance interaction term); interaction coefficient −0.610 on an expanded 32-procedural eval set (up from −0.064 at n=12); no procedural answer reaches false-high confidence
- Real-time arXiv paper fetching during queries

---

## About

Built as a portfolio project demonstrating ML engineering depth: system design, multi-strategy retrieval, LLM inference, uncertainty quantification, and rigorous evaluation. Part of a two-project series alongside a full-stack Content Performance Analyzer.
