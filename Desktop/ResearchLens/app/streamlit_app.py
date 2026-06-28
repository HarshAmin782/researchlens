import streamlit as st
import json
import pickle
import os
import numpy as np
import re
from pathlib import Path

st.set_page_config(
    page_title="ResearchLens",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400;1,600&family=Source+Serif+4:ital,wght@0,300;0,400;0,600;1,300;1,400&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Source Serif 4', Georgia, serif;
    background-color: #F7F4EF;
}

.stApp { background-color: #F7F4EF; }

header[data-testid="stHeader"] {
    background-color: #F7F4EF !important;
    border-bottom: 1px solid #C8BBA8 !important;
}

header[data-testid="stHeader"] button,
header[data-testid="stHeader"] a,
header[data-testid="stHeader"] span {
    color: #1A1208 !important;
}

.stMainBlockContainer { background-color: #F7F4EF !important; }

section[data-testid="stSidebar"] {
    background-color: #EDE8DF;
    border-right: 1px solid #C8BBA8;
}

.masthead {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: #1A1208;
    letter-spacing: -0.01em;
    line-height: 1.1;
}

.vol-line {
    font-size: 0.62rem;
    color: #8B7355;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}

.rule-double {
    border-top: 2.5px solid #1A1208;
    border-bottom: 1px solid #1A1208;
    height: 4px;
    margin: 0.5rem 0 1.2rem;
}

.rule-single {
    border-top: 1px solid #C8BBA8;
    margin: 1rem 0;
}

.section-label {
    font-size: 0.6rem;
    color: #8B7355;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.4rem;
    font-family: 'JetBrains Mono', monospace;
}

.query-display {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 1.2rem;
    font-style: italic;
    color: #1A1208;
    margin-bottom: 0.3rem;
}

.type-badge {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    padding: 0.15rem 0.6rem;
    border-radius: 2px;
    border: 1px solid #C8BBA8;
    color: #5C4A35;
    background: #EDE8DF;
    margin-right: 0.4rem;
    font-style: normal;
}

.router-note {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: #8B7355;
    margin: 0.4rem 0 1rem;
}

.confidence-row {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    background: white;
    border: 1px solid #C8BBA8;
    padding: 0.5rem 0.8rem;
    margin-bottom: 1.2rem;
}

.chunk-card {
    background: white;
    border: 1px solid #C8BBA8;
    border-left: 3px solid #1A1208;
    padding: 0.85rem 1rem;
    margin-bottom: 0.8rem;
}

.chunk-card-secondary {
    border-left-color: #8B7355;
}

.chunk-paper {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 0.78rem;
    font-style: italic;
    color: #4A3728;
    margin-bottom: 0.3rem;
}

.chunk-score {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: #8B7355;
}

.chunk-text {
    font-size: 0.82rem;
    color: #3D2E1E;
    line-height: 1.7;
    margin-top: 0.4rem;
}

.col-header {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 0.85rem;
    font-weight: 700;
    color: #1A1208;
    border-bottom: 2px solid #1A1208;
    padding-bottom: 0.3rem;
    margin-bottom: 0.8rem;
}

.stats-strip {
    display: flex;
    gap: 0;
    border: 1px solid #C8BBA8;
    background: white;
    margin-bottom: 1.5rem;
}

.stat-cell {
    flex: 1;
    padding: 0.7rem 0.8rem;
    border-right: 1px solid #C8BBA8;
    text-align: center;
}

.stat-cell:last-child { border-right: none; }

.stat-num {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #1A1208;
    display: block;
}

.stat-cap {
    font-size: 0.6rem;
    color: #8B7355;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-family: 'JetBrains Mono', monospace;
}

.pull-quote {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 1.1rem;
    font-style: italic;
    color: #1A1208;
    border-top: 2px solid #1A1208;
    border-bottom: 1px solid #1A1208;
    padding: 0.7rem 0;
    margin: 1rem 0;
    text-align: center;
}

.results-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.8rem;
    background: white;
    border: 1px solid #C8BBA8;
}

.results-table th {
    background: #1A1208;
    color: #F7F4EF;
    padding: 0.5rem 0.8rem;
    text-align: left;
    font-weight: 600;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-family: 'JetBrains Mono', monospace;
}

.results-table td {
    padding: 0.55rem 0.8rem;
    border-bottom: 1px solid #EDE8DF;
    color: #3D2E1E;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
}

.results-table tr:hover td { background: #F0EBE1; }

.best-val {
    color: #1A1208;
    font-weight: 600;
    text-decoration: underline;
    text-underline-offset: 3px;
}

.strategy-name {
    font-family: 'Playfair Display', Georgia, serif;
    font-style: italic;
    color: #4A3728;
    font-size: 0.78rem;
}

.contra-card {
    background: white;
    border: 1px solid #C8BBA8;
    border-left: 3px solid #6B0000;
    padding: 1rem 1.1rem;
    margin-bottom: 0.9rem;
}

.paper-byline {
    font-size: 0.62rem;
    color: #8B7355;
    font-family: 'JetBrains Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.3rem;
}

.claim-text {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 0.85rem;
    font-style: italic;
    color: #3D2E1E;
    line-height: 1.6;
    margin-bottom: 0.5rem;
}

.vs-divider {
    text-align: center;
    color: #6B0000;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    margin: 0.4rem 0;
}

.empty-state {
    text-align: center;
    padding: 5rem 2rem;
}

.empty-head {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 1.1rem;
    font-style: italic;
    color: #8B7355;
    margin-bottom: 0.4rem;
}

.empty-hint {
    font-size: 0.78rem;
    color: #B5A898;
    font-family: 'JetBrains Mono', monospace;
}

.footnote {
    font-size: 0.68rem;
    color: #8B7355;
    line-height: 1.8;
    font-family: 'JetBrains Mono', monospace;
    border-top: 1px solid #C8BBA8;
    padding-top: 0.8rem;
    margin-top: 1.2rem;
}

.stTextInput > div > div > input {
    background-color: white !important;
    color: #1A1208 !important;
    border: 1px solid #C8BBA8 !important;
    border-radius: 2px !important;
    font-family: 'Source Serif 4', Georgia, serif !important;
    font-size: 0.85rem !important;
}

.stSelectbox > div > div {
    background-color: white !important;
    border: 1px solid #C8BBA8 !important;
    border-radius: 2px !important;
    color: #1A1208 !important;
}

.stButton > button {
    background-color: #1A1208 !important;
    color: #F7F4EF !important;
    border: none !important;
    border-radius: 2px !important;
    font-family: 'Playfair Display', Georgia, serif !important;
    font-style: italic !important;
    font-size: 0.9rem !important;
    width: 100% !important;
    padding: 0.5rem !important;
}

.stButton > button:hover {
    background-color: #3D2E1E !important;
}

.stTabs [data-baseweb="tab-list"] {
    background-color: #F7F4EF;
    border-bottom: 2px solid #1A1208;
    gap: 0;
}

.stTabs [data-baseweb="tab"] {
    color: #8B7355;
    font-size: 0.82rem;
    padding: 0.5rem 1.2rem;
    font-family: 'Playfair Display', Georgia, serif;
}

.stTabs [aria-selected="true"] {
    color: #1A1208 !important;
    background: #1A1208 !important;
    color: #F7F4EF !important;
}

.stInfo {
    background: #EDE8DF;
    border: 1px solid #C8BBA8;
    color: #5C4A35;
}
</style>
""", unsafe_allow_html=True)

DATA_DIR = Path(__file__).parent / 'data'


@st.cache_resource(show_spinner="Setting the type — indexing papers, first run takes ~2 minutes...")
def load_indexes():
    with open(DATA_DIR / 'bm25_index.pkl', 'rb') as f:
        bm25_index = pickle.load(f)
    with open(DATA_DIR / 'bm25_metadata.pkl', 'rb') as f:
        bm25_metadata = pickle.load(f)
    with open(DATA_DIR / 'fixed_chunks.pkl', 'rb') as f:
        fixed_chunks = pickle.load(f)
    with open(DATA_DIR / 'semantic_chunks.pkl', 'rb') as f:
        semantic_chunks = pickle.load(f)

    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')

    import chromadb
    chroma_path = DATA_DIR / 'chroma_db'
    chroma_path.mkdir(exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(chroma_path))
    existing = [c.name for c in chroma_client.list_collections()]

    if 'fixed_chunks' not in existing:
        fixed_col = chroma_client.create_collection(name="fixed_chunks", metadata={"hnsw:space": "cosine"})
        texts = [c['text'] for c in fixed_chunks]
        embeddings = embed_model.encode(texts, batch_size=64, show_progress_bar=False)
        for i in range(0, len(fixed_chunks), 500):
            bc = fixed_chunks[i:i+500]
            be = embeddings[i:i+500]
            fixed_col.add(
                ids=[c['chunk_id'] for c in bc],
                documents=[c['text'] for c in bc],
                embeddings=be.tolist(),
                metadatas=[{'paper_id': c['paper_id'], 'title': c['title'], 'page_num': c['page_num'], 'chunk_id': c['chunk_id']} for c in bc]
            )
    else:
        fixed_col = chroma_client.get_collection("fixed_chunks")

    if 'semantic_chunks' not in existing:
        sem_col = chroma_client.create_collection(name="semantic_chunks", metadata={"hnsw:space": "cosine"})
        texts = [c['text'] for c in semantic_chunks]
        embeddings = embed_model.encode(texts, batch_size=64, show_progress_bar=False)
        for i in range(0, len(semantic_chunks), 500):
            bc = semantic_chunks[i:i+500]
            be = embeddings[i:i+500]
            sem_col.add(
                ids=[c['chunk_id'] for c in bc],
                documents=[c['text'] for c in bc],
                embeddings=be.tolist(),
                metadatas=[{'paper_id': c['paper_id'], 'title': c['title'], 'page_num': c['page_num'], 'chunk_id': c['chunk_id']} for c in bc]
            )
    else:
        sem_col = chroma_client.get_collection("semantic_chunks")

    return bm25_index, bm25_metadata, fixed_chunks, semantic_chunks, embed_model, fixed_col, sem_col


@st.cache_data
def load_eval_data():
    with open(DATA_DIR / 'comparison_table.json', 'r') as f:
        comparison = json.load(f)
    with open(DATA_DIR / 'contradiction_results.json', 'r') as f:
        contradictions = json.load(f)
    return comparison, contradictions


def naive_retrieve(query, embed_model, fixed_col, top_k=5):
    qe = embed_model.encode(query).tolist()
    r = fixed_col.query(query_embeddings=[qe], n_results=top_k, include=['documents', 'distances', 'metadatas'])
    return [{'text': d, 'score': round(1-s, 3), 'title': m['title']}
            for d, s, m in zip(r['documents'][0], r['distances'][0], r['metadatas'][0])]


def semantic_retrieve(query, embed_model, sem_col, top_k=5):
    qe = embed_model.encode(query).tolist()
    r = sem_col.query(query_embeddings=[qe], n_results=top_k, include=['documents', 'distances', 'metadatas'])
    return [{'text': d, 'score': round(1-s, 3), 'title': m['title']}
            for d, s, m in zip(r['documents'][0], r['distances'][0], r['metadatas'][0])]


def hybrid_retrieve(query, embed_model, fixed_col, bm25_index, bm25_metadata, top_k=5):
    fetch_k = top_k * 4
    qe = embed_model.encode(query).tolist()
    dr = fixed_col.query(query_embeddings=[qe], n_results=fetch_k, include=['documents', 'distances', 'metadatas'])
    dense_hits = {}
    for rank, (d, s, m) in enumerate(zip(dr['documents'][0], dr['distances'][0], dr['metadatas'][0])):
        cid = m.get('chunk_id', f"d_{rank}")
        dense_hits[cid] = {'rank': rank, 'text': d, 'score': 1-s, 'title': m['title']}

    bm25_scores = bm25_index.get_scores(query.lower().split())
    top_idx = np.argsort(bm25_scores)[::-1][:fetch_k]
    bm25_hits = {}
    for rank, idx in enumerate(top_idx):
        cid = bm25_metadata[idx]['chunk_id']
        bm25_hits[cid] = {'rank': rank, 'text': bm25_metadata[idx]['text'],
                          'score': bm25_scores[idx], 'title': bm25_metadata[idx]['title']}

    k = 60
    all_ids = set(dense_hits) | set(bm25_hits)
    rrf = {cid: (1/(dense_hits[cid]['rank']+k) if cid in dense_hits else 0) +
                (1/(bm25_hits[cid]['rank']+k) if cid in bm25_hits else 0)
           for cid in all_ids}
    top_ids = sorted(rrf, key=rrf.get, reverse=True)[:top_k]
    return [{'text': (dense_hits.get(cid) or bm25_hits[cid])['text'],
             'score': round(rrf[cid], 4),
             'title': (dense_hits.get(cid) or bm25_hits[cid])['title']}
            for cid in top_ids]


def classify_query(query):
    q = query.lower()
    signals = {
        'consensus': ['do papers agree', 'is there consensus', 'controversy', 'disagree', 'contradiction', 'conflicting', 'debate'],
        'comparative': ['compare', 'difference between', 'versus', ' vs ', 'better than', 'trade-off', 'contrast', 'differ'],
        'procedural': ['how to', 'how do i', 'steps to', 'implement', 'build'],
        'factual': ['what is', 'what are', 'define', 'explain', 'describe']
    }
    for qtype, words in signals.items():
        for w in words:
            if w in q:
                return qtype
    return 'factual'


def render_chunk(chunk, idx):
    card_class = "chunk-card" if idx == 0 else "chunk-card chunk-card-secondary"
    st.markdown(f"""
    <div class="{card_class}">
        <div style="display:flex;justify-content:space-between;align-items:baseline">
            <div class="chunk-paper">{chunk['title'][:68]}</div>
            <span class="chunk-score">{chunk['score']}</span>
        </div>
        <div class="chunk-text">{chunk['text'][:440]}{'…' if len(chunk['text']) > 440 else ''}</div>
    </div>
    """, unsafe_allow_html=True)


with st.sidebar:
    st.markdown('<div class="masthead">ResearchLens</div>', unsafe_allow_html=True)
    st.markdown('<div class="vol-line">Vol. I · ML Paper Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="rule-double"></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">Query</div>', unsafe_allow_html=True)
    query = st.text_input("", placeholder="What is LoRA?", label_visibility="collapsed", key="query_input")

    st.markdown('<div class="section-label" style="margin-top:0.8rem">Retrieval mode</div>', unsafe_allow_html=True)
    mode = st.selectbox("", [
        "Auto — Adaptive Router",
        "A · Naive RAG",
        "B · Semantic RAG",
        "C · Hybrid RAG",
        "Compare All 4 Strategies"
    ], label_visibility="collapsed")

    st.button("Search →")

    st.markdown('<div class="rule-single"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.7rem;color:#8B7355;line-height:2;font-family:'JetBrains Mono',monospace">
        Corpus · 100 arXiv papers<br>
        Chunks · 3,205 fixed · 3,427 semantic<br>
        Generator · Mistral-7B-Instruct<br>
        Embeddings · all-MiniLM-L6-v2<br>
        NLI · bart-large-mnli
    </div>
    """, unsafe_allow_html=True)


tab1, tab2, tab3 = st.tabs(["Search", "Results Table", "Contradictions"])

with tab1:
    if not query:
        st.markdown("""
        <div class="empty-state">
            <div style="font-size:2rem;margin-bottom:0.8rem">🔬</div>
            <div class="empty-head">Ask a question about ML research</div>
            <div class="empty-hint">Try: "What is the difference between LoRA and full fine-tuning?"</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        bm25_index, bm25_metadata, fixed_chunks, semantic_chunks, embed_model, fixed_col, sem_col = load_indexes()

        query_type = classify_query(query)

        st.markdown(f"""
        <div style="margin-bottom:1rem">
            <div class="query-display">"{query}"</div>
            <span class="type-badge">{query_type} query</span>
        </div>
        """, unsafe_allow_html=True)

        type_to_strategy = {
            'factual': 'naive', 'procedural': 'semantic',
            'comparative': 'hybrid', 'consensus': 'hybrid'
        }

        if mode == "Compare All 4 Strategies":
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown('<div class="col-header">A · Naive RAG</div>', unsafe_allow_html=True)
                for i, c in enumerate(naive_retrieve(query, embed_model, fixed_col, top_k=3)):
                    render_chunk(c, i)
                st.markdown('<div class="col-header" style="margin-top:1.2rem">C · Hybrid RAG</div>', unsafe_allow_html=True)
                for i, c in enumerate(hybrid_retrieve(query, embed_model, fixed_col, bm25_index, bm25_metadata, top_k=3)):
                    render_chunk(c, i)
            with col_b:
                st.markdown('<div class="col-header">B · Semantic RAG</div>', unsafe_allow_html=True)
                for i, c in enumerate(semantic_retrieve(query, embed_model, sem_col, top_k=3)):
                    render_chunk(c, i)
                st.markdown('<div class="col-header" style="margin-top:1.2rem">D · Hybrid + Reranker</div>', unsafe_allow_html=True)
                st.markdown("""
                <div style="background:white;border:1px solid #C8BBA8;padding:0.7rem 0.9rem;
                    font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:#8B7355">
                    Cross-encoder reranker requires GPU. Showing Hybrid results as proxy.
                    See Results Table for measured reranker metrics.
                </div>
                """, unsafe_allow_html=True)
                for i, c in enumerate(hybrid_retrieve(query, embed_model, fixed_col, bm25_index, bm25_metadata, top_k=3)):
                    render_chunk(c, i)
        else:
            mode_map = {
                "Auto — Adaptive Router": None,
                "A · Naive RAG": "naive",
                "B · Semantic RAG": "semantic",
                "C · Hybrid RAG": "hybrid",
            }
            selected = mode_map.get(mode)
            if selected is None:
                selected = type_to_strategy.get(query_type, 'naive')
                st.markdown(f"""
                <div class="router-note">
                    router → <em>{selected}</em> · query classified as <em>{query_type}</em>
                </div>
                """, unsafe_allow_html=True)

            if selected == 'naive':
                results = naive_retrieve(query, embed_model, fixed_col)
            elif selected == 'semantic':
                results = semantic_retrieve(query, embed_model, sem_col)
            else:
                results = hybrid_retrieve(query, embed_model, fixed_col, bm25_index, bm25_metadata)

            avg_score = float(np.mean([r['score'] for r in results]))
            conf = min(int(avg_score * 150), 100)
            bar_color = '#2D6A2D' if conf >= 70 else '#8B6914' if conf >= 45 else '#6B0000'
            badge = 'High confidence' if conf >= 70 else 'Moderate confidence' if conf >= 45 else 'Low confidence'

            st.markdown(f"""
            <div class="confidence-row">
                <div style="flex:1;background:#EDE8DF;height:3px;overflow:hidden">
                    <div style="width:{conf}%;height:100%;background:{bar_color}"></div>
                </div>
                <span style="color:{bar_color};font-family:'JetBrains Mono',monospace;
                    font-size:0.68rem;white-space:nowrap">{badge} · {conf}/100</span>
            </div>
            """, unsafe_allow_html=True)

            for i, chunk in enumerate(results):
                render_chunk(chunk, i)

with tab2:
    try:
        comparison, _ = load_eval_data()

        st.markdown("""
        <div class="stats-strip">
            <div class="stat-cell">
                <span class="stat-num">100</span>
                <span class="stat-cap">arXiv papers</span>
            </div>
            <div class="stat-cell">
                <span class="stat-num">50</span>
                <span class="stat-cap">QASPER questions</span>
            </div>
            <div class="stat-cell">
                <span class="stat-num">4</span>
                <span class="stat-cap">strategies tested</span>
            </div>
            <div class="stat-cell">
                <span class="stat-num">+9.2%</span>
                <span class="stat-cap">faithfulness gain</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="pull-quote">
            Hybrid + Reranker achieves 9.2% higher faithfulness than Naive RAG at only 14% latency overhead.
        </div>
        """, unsafe_allow_html=True)

        import pandas as pd

        df_display = pd.DataFrame([{
            'Strategy': r['strategy'],
            'Faithfulness': r['faithfulness'],
            'Context Precision': r['context_precision'],
            'Answer Relevance': r['answer_relevance'],
            'Latency (ms)': r['latency_ms']
        } for r in comparison])

        st.dataframe(
            df_display.style
                .highlight_max(
                    subset=['Faithfulness', 'Context Precision', 'Answer Relevance'],
                    color='#C8BBA8'
                )
                .format({
                    'Faithfulness': '{:.3f}',
                    'Context Precision': '{:.3f}',
                    'Answer Relevance': '{:.3f}',
                    'Latency (ms)': '{:,}'
                }),
            use_container_width=True,
            hide_index=True
        )

        st.markdown("""
        <div class="footnote">
            † Faithfulness — token overlap between answer and retrieved context<br>
            † Context Precision — fraction of retrieved chunks with cosine similarity &gt; 0.4 to query<br>
            † Answer Relevance — cosine similarity between query embedding and answer embedding<br>
            † Generator — Mistral-7B-Instruct-v0.2 on Kaggle T4 GPU · best values underlined
        </div>
        """, unsafe_allow_html=True)

    except Exception:
        st.info("Add comparison_table.json to the app/data/ folder.")

with tab3:
    try:
        _, contradictions = load_eval_data()

        total = sum(len(t['contradictions']) for t in contradictions)
        st.markdown(f"""
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:#8B7355;
            margin-bottom:1rem;border-bottom:1px solid #C8BBA8;padding-bottom:0.6rem">
            {total} contradictions detected across 3 topics · NLI: facebook/bart-large-mnli · threshold 0.45
        </div>
        """, unsafe_allow_html=True)

        for topic_result in contradictions:
            if not topic_result['contradictions']:
                st.markdown(f"""
                <div style="font-size:0.78rem;color:#8B7355;margin:0.8rem 0;
                    font-family:'JetBrains Mono',monospace">
                    "{topic_result['topic']}" — no contradictions detected above threshold
                </div>
                """, unsafe_allow_html=True)
                continue

            st.markdown(f"""
            <div style="margin:1.2rem 0 0.7rem">
                <span style="font-family:'Playfair Display',Georgia,serif;font-size:0.95rem;
                    font-style:italic;color:#1A1208">"{topic_result['topic']}"</span>
                <span style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;
                    color:#8B7355;margin-left:0.5rem">{len(topic_result['contradictions'])} found</span>
            </div>
            """, unsafe_allow_html=True)

            for c in topic_result['contradictions'][:3]:
                score = c['contradiction_score']
                score_color = '#6B0000' if score > 0.45 else '#8B6914'
                st.markdown(f"""
                <div class="contra-card">
                    <div style="display:flex;justify-content:space-between;margin-bottom:0.6rem">
                        <span style="font-family:'JetBrains Mono',monospace;font-size:0.62rem;color:#8B7355">
                            contradiction score
                        </span>
                        <span style="color:{score_color};font-family:'JetBrains Mono',monospace;
                            font-size:0.7rem;font-weight:600">{score}</span>
                    </div>
                    <div class="paper-byline">Paper A · {c['paper_a'][:60]}</div>
                    <div class="claim-text">"{c['claim_a'][:290]}"</div>
                    <div class="vs-divider">↕ &nbsp; contradicts &nbsp; ↕</div>
                    <div class="paper-byline" style="margin-top:0.5rem">Paper B · {c['paper_b'][:60]}</div>
                    <div class="claim-text">"{c['claim_b'][:290]}"</div>
                </div>
                """, unsafe_allow_html=True)

    except Exception:
        st.info("Add contradiction_results.json to the app/data/ folder.")