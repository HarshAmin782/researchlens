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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background-color: #0A0F1E; color: #E2E8F0; }

section[data-testid="stSidebar"] {
    background-color: #0D1220;
    border-right: 1px solid #1E293B;
}

.main-title {
    font-size: 1.6rem;
    font-weight: 600;
    color: #F8FAFC;
    letter-spacing: -0.02em;
    line-height: 1.2;
    margin-bottom: 0.15rem;
}

.main-sub {
    font-size: 0.78rem;
    color: #475569;
    margin-bottom: 1.5rem;
    font-weight: 400;
}

.section-label {
    font-size: 0.65rem;
    font-weight: 600;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.5rem;
    margin-top: 1.2rem;
}

.query-display {
    font-size: 1.05rem;
    color: #F1F5F9;
    font-weight: 500;
    margin-bottom: 0.4rem;
}

.type-badge {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    padding: 0.18rem 0.55rem;
    border-radius: 3px;
    font-weight: 500;
    margin-right: 0.4rem;
}

.router-note {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #475569;
    margin: 0.6rem 0 1.2rem;
}

.confidence-bar-wrap {
    background: #1E293B;
    border-radius: 6px;
    padding: 0.7rem 1rem;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}

.chunk-card {
    background: #111827;
    border: 1px solid #1E293B;
    border-radius: 6px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.7rem;
}

.chunk-meta {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
}

.chunk-title {
    font-size: 0.72rem;
    color: #60A5FA;
    font-family: 'JetBrains Mono', monospace;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 75%;
}

.chunk-score {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    padding: 0.12rem 0.45rem;
    border-radius: 3px;
}

.chunk-text {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.74rem;
    color: #94A3B8;
    line-height: 1.65;
    border-left: 2px solid #1E40AF;
    padding-left: 0.7rem;
}

.col-label {
    font-size: 0.7rem;
    font-weight: 600;
    color: #334155;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.7rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #1E293B;
}

.stats-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.5rem;
}

.stat-box {
    background: #111827;
    border: 1px solid #1E293B;
    border-radius: 6px;
    padding: 0.7rem 1rem;
    flex: 1;
    text-align: center;
}

.stat-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.3rem;
    font-weight: 600;
    color: #3B82F6;
    display: block;
}

.stat-label {
    font-size: 0.65rem;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}

.results-table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
}

.results-table th {
    background: #111827;
    color: #64748B;
    padding: 0.6rem 0.9rem;
    text-align: left;
    font-weight: 500;
    border-bottom: 1px solid #1E293B;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

.results-table td {
    padding: 0.65rem 0.9rem;
    border-bottom: 1px solid #0F172A;
    color: #CBD5E1;
}

.results-table tr:hover td { background: #111827; }

.best-val { color: #34D399; font-weight: 600; }

.contra-card {
    background: #111827;
    border: 1px solid #1F2937;
    border-left: 3px solid #7F1D1D;
    border-radius: 0 6px 6px 0;
    padding: 1rem 1.1rem;
    margin-bottom: 0.9rem;
}

.paper-tag {
    font-size: 0.65rem;
    color: #60A5FA;
    font-family: 'JetBrains Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.3rem;
}

.claim {
    font-size: 0.83rem;
    color: #CBD5E1;
    line-height: 1.55;
    font-style: italic;
    margin-bottom: 0.6rem;
}

.vs-label {
    text-align: center;
    color: #EF4444;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    margin: 0.5rem 0;
    opacity: 0.7;
}

.empty-state {
    text-align: center;
    padding: 5rem 2rem;
    color: #1E293B;
}

.empty-icon { font-size: 2.5rem; margin-bottom: 0.8rem; }

.empty-title {
    font-size: 1rem;
    font-weight: 500;
    color: #334155;
    margin-bottom: 0.4rem;
}

.empty-hint { font-size: 0.8rem; color: #1E293B; }

.stTextInput > div > div > input {
    background-color: #111827 !important;
    color: #E2E8F0 !important;
    border: 1px solid #1E293B !important;
    border-radius: 5px !important;
    font-size: 0.85rem !important;
}

.stSelectbox > div > div {
    background-color: #111827 !important;
    border: 1px solid #1E293B !important;
    border-radius: 5px !important;
}

.stButton > button {
    background-color: #1D4ED8 !important;
    color: white !important;
    border: none !important;
    border-radius: 5px !important;
    font-weight: 500 !important;
    width: 100% !important;
    font-size: 0.85rem !important;
}

.stButton > button:hover { background-color: #1E40AF !important; }

.stTabs [data-baseweb="tab-list"] {
    background-color: #0A0F1E;
    border-bottom: 1px solid #1E293B;
    gap: 0;
}

.stTabs [data-baseweb="tab"] {
    color: #475569;
    font-size: 0.82rem;
    padding: 0.6rem 1.2rem;
}

.stTabs [aria-selected="true"] {
    color: #3B82F6 !important;
    border-bottom: 2px solid #3B82F6 !important;
    background: transparent !important;
}

div[data-testid="stSpinner"] { color: #3B82F6; }
</style>
""", unsafe_allow_html=True)

DATA_DIR = Path(__file__).parent / 'data'


@st.cache_resource(show_spinner="Loading indexes — first run takes ~2 minutes...")
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
        fixed_col = chroma_client.create_collection(
            name="fixed_chunks", metadata={"hnsw:space": "cosine"}
        )
        texts = [c['text'] for c in fixed_chunks]
        embeddings = embed_model.encode(texts, batch_size=64, show_progress_bar=False)
        batch = 500
        for i in range(0, len(fixed_chunks), batch):
            bc = fixed_chunks[i:i+batch]
            be = embeddings[i:i+batch]
            fixed_col.add(
                ids=[c['chunk_id'] for c in bc],
                documents=[c['text'] for c in bc],
                embeddings=be.tolist(),
                metadatas=[{'paper_id': c['paper_id'], 'title': c['title'],
                            'page_num': c['page_num'], 'chunk_id': c['chunk_id']} for c in bc]
            )
    else:
        fixed_col = chroma_client.get_collection("fixed_chunks")

    if 'semantic_chunks' not in existing:
        sem_col = chroma_client.create_collection(
            name="semantic_chunks", metadata={"hnsw:space": "cosine"}
        )
        texts = [c['text'] for c in semantic_chunks]
        embeddings = embed_model.encode(texts, batch_size=64, show_progress_bar=False)
        batch = 500
        for i in range(0, len(semantic_chunks), batch):
            bc = semantic_chunks[i:i+batch]
            be = embeddings[i:i+batch]
            sem_col.add(
                ids=[c['chunk_id'] for c in bc],
                documents=[c['text'] for c in bc],
                embeddings=be.tolist(),
                metadatas=[{'paper_id': c['paper_id'], 'title': c['title'],
                            'page_num': c['page_num'], 'chunk_id': c['chunk_id']} for c in bc]
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
    r = fixed_col.query(query_embeddings=[qe], n_results=top_k,
                        include=['documents', 'distances', 'metadatas'])
    return [{'text': d, 'score': round(1-s, 3), 'title': m['title']}
            for d, s, m in zip(r['documents'][0], r['distances'][0], r['metadatas'][0])]


def semantic_retrieve(query, embed_model, sem_col, top_k=5):
    qe = embed_model.encode(query).tolist()
    r = sem_col.query(query_embeddings=[qe], n_results=top_k,
                      include=['documents', 'distances', 'metadatas'])
    return [{'text': d, 'score': round(1-s, 3), 'title': m['title']}
            for d, s, m in zip(r['documents'][0], r['distances'][0], r['metadatas'][0])]


def hybrid_retrieve(query, embed_model, fixed_col, bm25_index, bm25_metadata, top_k=5):
    fetch_k = top_k * 4
    qe = embed_model.encode(query).tolist()
    dr = fixed_col.query(query_embeddings=[qe], n_results=fetch_k,
                         include=['documents', 'distances', 'metadatas'])
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
        'consensus': ['do papers agree', 'is there consensus', 'controversy',
                      'disagree', 'contradiction', 'conflicting', 'debate'],
        'comparative': ['compare', 'difference between', 'versus', ' vs ',
                        'better than', 'trade-off', 'contrast', 'differ'],
        'procedural': ['how to', 'how do i', 'steps to', 'implement', 'build'],
        'factual': ['what is', 'what are', 'define', 'explain', 'describe']
    }
    for qtype, words in signals.items():
        for w in words:
            if w in q:
                return qtype
    return 'factual'


def render_chunk(chunk, strategy_label):
    score = chunk['score']
    score_color = '#34D399' if score > 0.55 else '#FBBF24' if score > 0.4 else '#F87171'
    st.markdown(f"""
    <div class="chunk-card">
        <div class="chunk-meta">
            <div class="chunk-title">{chunk['title'][:65]}</div>
            <span class="chunk-score" style="background:{score_color}22;color:{score_color}">
                {score}
            </span>
        </div>
        <div class="chunk-text">{chunk['text'][:420]}{'...' if len(chunk['text']) > 420 else ''}</div>
    </div>
    """, unsafe_allow_html=True)


with st.sidebar:
    st.markdown('<div class="main-title">🔬 ResearchLens</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-sub">Intelligent ML Paper Analysis</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">Ask a question</div>', unsafe_allow_html=True)
    query = st.text_input(
        "", placeholder="What is LoRA?", label_visibility="collapsed",
        key="query_input"
    )

    st.markdown('<div class="section-label">Retrieval mode</div>', unsafe_allow_html=True)
    mode = st.selectbox("", [
        "Auto — Adaptive Router",
        "A · Naive RAG",
        "B · Semantic RAG",
        "C · Hybrid RAG",
        "Compare All 4 Strategies"
    ], label_visibility="collapsed")

    search = st.button("Search")

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.72rem;color:#334155;line-height:1.7">
        <span style="color:#475569;font-weight:500">Corpus</span><br>
        100 ML/AI papers · arXiv<br><br>
        <span style="color:#475569;font-weight:500">Strategies</span><br>
        Naive · Semantic · Hybrid · Reranker<br><br>
        <span style="color:#475569;font-weight:500">Evaluated on</span><br>
        50 QASPER questions<br><br>
        <span style="color:#475569;font-weight:500">Generator</span><br>
        Mistral-7B-Instruct-v0.2
    </div>
    """, unsafe_allow_html=True)


tab1, tab2, tab3 = st.tabs(["🔍  Search", "📊  Results Table", "⚡  Contradictions"])

with tab1:
    if not query:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">🔬</div>
            <div class="empty-title">Search across 100 ML research papers</div>
            <div class="empty-hint">Try: "What is the difference between LoRA and full fine-tuning?"</div>
        </div>
        """, unsafe_allow_html=True)

    else:
        bm25_index, bm25_metadata, fixed_chunks, semantic_chunks, embed_model, fixed_col, sem_col = load_indexes()

        query_type = classify_query(query)
        type_styles = {
            'factual':     ('#3B82F6', '#1E3A5F'),
            'comparative': ('#8B5CF6', '#2E1065'),
            'consensus':   ('#EF4444', '#450A0A'),
            'procedural':  ('#10B981', '#022C22'),
        }
        tc, tb = type_styles.get(query_type, ('#3B82F6', '#1E3A5F'))

        st.markdown(f"""
        <div style="margin-bottom:1rem">
            <div class="query-display">"{query}"</div>
            <span class="type-badge" style="background:{tb};color:{tc};border:1px solid {tc}44">
                {query_type}
            </span>
        </div>
        """, unsafe_allow_html=True)

        type_to_strategy = {
            'factual': 'naive', 'procedural': 'semantic',
            'comparative': 'hybrid', 'consensus': 'hybrid'
        }

        if mode == "Compare All 4 Strategies":
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown('<div class="col-label">A · Naive RAG</div>', unsafe_allow_html=True)
                for c in naive_retrieve(query, embed_model, fixed_col, top_k=3):
                    render_chunk(c, "naive")
                st.markdown('<div class="col-label" style="margin-top:1rem">C · Hybrid RAG</div>', unsafe_allow_html=True)
                for c in hybrid_retrieve(query, embed_model, fixed_col, bm25_index, bm25_metadata, top_k=3):
                    render_chunk(c, "hybrid")
            with col_b:
                st.markdown('<div class="col-label">B · Semantic RAG</div>', unsafe_allow_html=True)
                for c in semantic_retrieve(query, embed_model, sem_col, top_k=3):
                    render_chunk(c, "semantic")
                st.markdown('<div class="col-label" style="margin-top:1rem">D · Hybrid + Reranker</div>', unsafe_allow_html=True)
                st.markdown("""
                <div style="font-size:0.75rem;color:#475569;font-family:monospace;
                    padding:0.6rem;background:#111827;border-radius:4px;margin-bottom:0.5rem">
                    Cross-encoder reranker requires GPU — showing Hybrid results as proxy.
                    See comparison table for measured reranker metrics.
                </div>
                """, unsafe_allow_html=True)
                for c in hybrid_retrieve(query, embed_model, fixed_col, bm25_index, bm25_metadata, top_k=3):
                    render_chunk(c, "hybrid+rerank")

        else:
            mode_map = {
                "Auto — Adaptive Router": None,
                "A · Naive RAG": "naive",
                "B · Semantic RAG": "semantic",
                "C · Hybrid RAG": "hybrid",
            }
            selected = mode_map[mode]

            if selected is None:
                selected = type_to_strategy.get(query_type, 'naive')
                st.markdown(f"""
                <div class="router-note">
                    → router: <span style="color:#3B82F6">{selected}</span>
                    &nbsp;·&nbsp; query type: <span style="color:{tc}">{query_type}</span>
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
            bar_color = '#10B981' if conf >= 70 else '#F59E0B' if conf >= 45 else '#EF4444'
            badge = 'HIGH' if conf >= 70 else 'MODERATE' if conf >= 45 else 'LOW'

            st.markdown(f"""
            <div style="background:#111827;border:1px solid #1E293B;border-radius:6px;
                padding:0.7rem 1rem;margin-bottom:1.2rem;display:flex;align-items:center;gap:1rem">
                <div style="flex:1;background:#0A0F1E;border-radius:3px;height:4px;overflow:hidden">
                    <div style="width:{conf}%;height:100%;background:{bar_color};border-radius:3px"></div>
                </div>
                <span style="color:{bar_color};font-family:monospace;font-size:0.7rem;
                    font-weight:600;white-space:nowrap">{badge} · {conf}/100</span>
            </div>
            """, unsafe_allow_html=True)

            for chunk in results:
                render_chunk(chunk, selected)

with tab2:
    try:
        comparison, _ = load_eval_data()

        st.markdown("""
        <div class="stats-row">
            <div class="stat-box">
                <span class="stat-val">100</span>
                <span class="stat-label">arXiv Papers</span>
            </div>
            <div class="stat-box">
                <span class="stat-val">50</span>
                <span class="stat-label">QASPER Questions</span>
            </div>
            <div class="stat-box">
                <span class="stat-val">4</span>
                <span class="stat-label">Strategies Compared</span>
            </div>
            <div class="stat-box">
                <span class="stat-val">+9.2%</span>
                <span class="stat-label">Faithfulness Gain</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        best_faith = max(r['faithfulness'] for r in comparison)
        best_prec = max(r['context_precision'] for r in comparison)
        best_rel = max(r['answer_relevance'] for r in comparison)

        table_html = """
        <table class="results-table">
        <tr>
            <th>Strategy</th>
            <th>Faithfulness ↑</th>
            <th>Context Precision ↑</th>
            <th>Answer Relevance ↑</th>
            <th>Avg Latency</th>
        </tr>
        """
        for row in comparison:
            f_class = 'best-val' if row['faithfulness'] == best_faith else ''
            p_class = 'best-val' if row['context_precision'] == best_prec else ''
            r_class = 'best-val' if row['answer_relevance'] == best_rel else ''
            table_html += f"""
            <tr>
                <td style="color:#60A5FA;font-family:monospace">{row['strategy']}</td>
                <td class="{f_class}">{row['faithfulness']}</td>
                <td class="{p_class}">{row['context_precision']}</td>
                <td class="{r_class}">{row['answer_relevance']}</td>
                <td style="color:#475569">{row['latency_ms']}ms</td>
            </tr>
            """
        table_html += "</table>"
        st.markdown(table_html, unsafe_allow_html=True)

        st.markdown("""
        <div style="margin-top:1.2rem;font-size:0.72rem;color:#334155;line-height:1.9;
            font-family:'JetBrains Mono',monospace">
            Faithfulness — token overlap between answer and retrieved context<br>
            Context Precision — fraction of retrieved chunks with cosine sim > 0.4 to query<br>
            Answer Relevance — cosine similarity between query and answer embeddings<br>
            Generator — Mistral-7B-Instruct-v0.2 · Evaluator — custom metrics on Kaggle T4
        </div>
        """, unsafe_allow_html=True)

    except Exception:
        st.info("Add comparison_table.json to the app/data/ folder.")

with tab3:
    try:
        _, contradictions = load_eval_data()

        total = sum(len(t['contradictions']) for t in contradictions)
        st.markdown(f"""
        <div style="font-size:0.78rem;color:#475569;margin-bottom:1.2rem;font-family:monospace">
            {total} contradictions detected across 3 topics · NLI model: facebook/bart-large-mnli
        </div>
        """, unsafe_allow_html=True)

        for topic_result in contradictions:
            if not topic_result['contradictions']:
                st.markdown(f"""
                <div style="font-size:0.8rem;color:#334155;margin:0.8rem 0;font-family:monospace">
                    Topic: "{topic_result['topic']}" — no contradictions above threshold
                </div>
                """, unsafe_allow_html=True)
                continue

            st.markdown(f"""
            <div style="font-size:0.85rem;font-weight:500;color:#E2E8F0;margin:1.2rem 0 0.7rem">
                "{topic_result['topic']}"
                <span style="color:#475569;font-size:0.72rem;font-family:monospace;margin-left:0.5rem">
                    {len(topic_result['contradictions'])} found
                </span>
            </div>
            """, unsafe_allow_html=True)

            for c in topic_result['contradictions'][:3]:
                score = c['contradiction_score']
                score_color = '#EF4444' if score > 0.45 else '#F59E0B'
                st.markdown(f"""
                <div class="contra-card">
                    <div style="display:flex;justify-content:space-between;margin-bottom:0.7rem">
                        <span style="font-family:monospace;font-size:0.65rem;color:#475569">
                            contradiction score
                        </span>
                        <span style="color:{score_color};font-family:monospace;
                            font-size:0.72rem;font-weight:600">{score}</span>
                    </div>
                    <div class="paper-tag">Paper A · {c['paper_a'][:58]}</div>
                    <div class="claim">"{c['claim_a'][:280]}"</div>
                    <div class="vs-label">↕ &nbsp; CONTRADICTS &nbsp; ↕</div>
                    <div class="paper-tag" style="margin-top:0.6rem">Paper B · {c['paper_b'][:58]}</div>
                    <div class="claim">"{c['claim_b'][:280]}"</div>
                </div>
                """, unsafe_allow_html=True)

    except Exception:
        st.info("Add contradiction_results.json to the app/data/ folder.")
