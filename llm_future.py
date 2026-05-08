# llm_future.py
# ─────────────────────────────────────────────────────────────────────────────
# LLM / RAG LAYER — saved for future re-integration
# Removed from the main Streamlit app on 2026-04-25 to simplify the demo.
#
# TO RE-INTEGRATE:
#   1. Restore llm/index/build_index.py (see INDEXER section below)
#   2. Run: python3 llm/index/build_index.py
#   3. In app.py, add Tab 3 code (see TAB3 section below)
#   4. pip install anthropic>=0.97.0
#   5. Add anthropic>=0.97.0 back to requirements.txt
#   6. Run: ANTHROPIC_API_KEY=sk-ant-... streamlit run app.py
# ─────────────────────────────────────────────────────────────────────────────

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — INDEX BUILDER  (was llm/index/build_index.py)
# ══════════════════════════════════════════════════════════════════════════════
#
# Run standalone to rebuild the chunk index:
#   python3 -c "exec(open('llm_future.py').read()); build_index()"
#
# Outputs: llm/index/chunks.json

BUILD_INDEX_CODE = '''
# Phase: 6
# Purpose: Build a text chunk index over project artifacts for the RAG Q&A layer.
# Inputs: artifacts/*.md, artifacts/*.json, artifacts/*.csv, data/catalog/phase5_catalog.csv
# Outputs: llm/index/chunks.json

import json
import re
import math
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np

ROOT  = Path(__file__).resolve().parent
IDX   = ROOT / "llm" / "index"
IDX.mkdir(parents=True, exist_ok=True)
ART   = ROOT / "artifacts"

CHUNK_SIZE  = 400   # words per chunk
CHUNK_STEP  = 200   # sliding window stride (50% overlap)

def read_text(p): return p.read_text()
def read_json_pretty(p):
    with open(p) as f:
        return json.dumps(json.load(f), indent=2)

def read_evaluation_metrics(p):
    with open(p) as f:
        d = json.load(f)
    lines = [f"Phase {d[\'phase\']} Evaluation Metrics — {d.get(\'pilot_window\',\'\')}",
             f"Stations: {d.get(\'n_stations\',\'\')}, Catalog: {d.get(\'catalog_source\',\'\')}"]
    for mk, m in d["models"].items():
        lines.append(f"\\n{m[\'model\']}:")
        lines.append(f"  recall={m[\'recall\']}  TP={m[\'TP\']}  FN={m[\'FN\']}")
        lines.append(f"  n_event_windows={m[\'n_event_windows\']}")
        if m.get("p_pick_mae_s"):
            lines.append(f"  P-pick MAE={m[\'p_pick_mae_s\']}s (upper bound)")
        for band, bm in m.get("by_magnitude", {}).items():
            lines.append(f"  {band}: recall={bm[\'recall\']}  TP={bm[\'TP\']}  n={bm[\'n_events\']}")
    return "\\n".join(lines)

def read_detection_summary(p):
    with open(p) as f:
        d = json.load(f)
    results = d["results"]
    n = len(results)
    models = ["stalta", "phasenet", "eqtransformer", "gpd"]
    model_labels = {"stalta":"STA/LTA","phasenet":"PhaseNet",
                    "eqtransformer":"EQTransformer","gpd":"GPD"}
    lines = [f"Detection Results Summary (Phase {d[\'phase\']}, mode={d[\'mode\']}, n={n} windows)"]
    for mk in models:
        det = [r for r in results if r[mk].get("detected", False)]
        lines.append(f"\\n{model_labels[mk]}: {len(det)}/{n} detected ({100*len(det)/n:.1f}%)")
        p_picks = [r[mk]["p_pick_s"] for r in results if r[mk].get("p_pick_s") is not None]
        if p_picks:
            arr = np.array(p_picks)
            lines.append(f"  P-pick: mean={arr.mean():.1f}s  std={arr.std():.1f}s")
    lines.append("\\n(Full per-event data in artifacts/detection_results.json)")
    return "\\n".join(lines)

def read_csv_summary(p, max_rows=30):
    df = pd.read_csv(p)
    lines = [f"File: {p.name}", f"Shape: {df.shape[0]} rows × {df.shape[1]} columns",
             f"Columns: {list(df.columns)}", "", df.head(max_rows).to_string(index=False)]
    return "\\n".join(lines)

def read_phase5_catalog_summary(p):
    df = pd.read_csv(p)
    df["magnitude"] = pd.to_numeric(df["magnitude"], errors="coerce")
    lines = [f"Phase 5 Catalog — {p.name}", f"Total events: {len(df)}",
             f"Date range: {df[\'origin_time\'].min()} to {df[\'origin_time\'].max()}",
             "Magnitude distribution:"]
    for lo, hi, lbl in [(2,3,"M 2-3"),(3,4,"M 3-4"),(4,99,"M>=4")]:
        n = ((df.magnitude>=lo)&(df.magnitude<hi)).sum()
        lines.append(f"  {lbl}: {n} events")
    return "\\n".join(lines)

SOURCES = [
    (ART/"planning.md",             "artifacts/planning.md",              read_text),
    (ART/"data_access_log.md",      "artifacts/data_access_log.md",       read_text),
    (ART/"methodology_notes.md",    "artifacts/methodology_notes.md",     read_text),
    (ART/"limitations.md",          "artifacts/limitations.md",           read_text),
    (ART/"figure_captions.md",      "artifacts/figure_captions.md",       read_text),
    (ART/"evaluation_metrics.json", "artifacts/evaluation_metrics.json",  read_evaluation_metrics),
    (ART/"detection_results.json",  "artifacts/detection_results.json",   read_detection_summary),
    (ART/"station_summary.csv",     "artifacts/station_summary.csv",      read_csv_summary),
    (ART/"phase5_station_list.csv", "artifacts/phase5_station_list.csv",  read_csv_summary),
    (ART/"phase5_gap_report.csv",   "artifacts/phase5_gap_report.csv",    read_csv_summary),
    (ART/"waveform_inventory.csv",  "artifacts/waveform_inventory.csv",   lambda p: read_csv_summary(p, 10)),
    (ROOT/"data"/"catalog"/"phase5_catalog.csv",
                                    "data/catalog/phase5_catalog.csv",    read_phase5_catalog_summary),
]

p4_backup = ART / "phase4_backup" / "evaluation_metrics.json"
if p4_backup.exists():
    SOURCES.append((p4_backup, "artifacts/phase4_backup/evaluation_metrics.json",
                    read_evaluation_metrics))

tbdy = ART / "tbdy2018_provisions.md"
if tbdy.exists():
    SOURCES.append((tbdy, "artifacts/tbdy2018_provisions.md", read_text))

STOPWORDS = {
    "the","a","an","and","or","in","of","to","is","are","was","were",
    "it","its","this","that","with","for","from","by","at","on","as",
    "be","has","have","had","not","no","but","if","which","all","one",
    "can","will","may","also","into","more","than","any","such","each",
}

def chunk_text(label, text, size=CHUNK_SIZE, step=CHUNK_STEP):
    words = text.split()
    chunks = []
    for i in range(0, max(1, len(words)-size+1), step):
        chunks.append({"source": label, "text": " ".join(words[i:i+size]), "char_start": i})
    if not chunks:
        chunks.append({"source": label, "text": text[:2000], "char_start": 0})
    return chunks

def tokenize_clean(text):
    return [w for w in re.findall(r\'\\b[a-z]\\w*\\b\', text.lower())
            if w not in STOPWORDS and len(w) > 1]

all_chunks = []
for path, label, reader in SOURCES:
    if not path.exists():
        continue
    text = reader(path)
    all_chunks.extend(chunk_text(label, text))

N = len(all_chunks)
doc_freq = Counter()
tokenized = []
for ck in all_chunks:
    toks = tokenize_clean(ck["text"])
    tokenized.append(toks)
    doc_freq.update(set(toks))

vocab_terms = [t for t, df in doc_freq.items() if 1 < df < N * 0.90]
vocab_terms.sort(key=lambda t: -doc_freq[t])
vocab_terms = vocab_terms[:8000]
vocab = {t: i for i, t in enumerate(vocab_terms)}
V = len(vocab)
idf = np.log((N+1)/(np.array([doc_freq[t] for t in vocab_terms])+1))+1.0
tfidf_matrix = np.zeros((N, V), dtype=np.float32)
for j, toks in enumerate(tokenized):
    tf = Counter(toks)
    for t, cnt in tf.items():
        if t in vocab:
            tfidf_matrix[j, vocab[t]] = cnt * idf[vocab[t]]
norms = np.linalg.norm(tfidf_matrix, axis=1, keepdims=True)
norms[norms == 0] = 1
tfidf_matrix /= norms

with open(IDX / "chunks.json", "w") as f:
    json.dump({"chunks": all_chunks, "vocab": vocab_terms,
               "idf": idf.tolist(), "tfidf_matrix": tfidf_matrix.tolist()}, f)
print(f"Index saved: {len(all_chunks)} chunks, {V} vocab terms")
'''


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — TAB 3 STREAMLIT CODE  (paste into app.py when re-integrating)
# ══════════════════════════════════════════════════════════════════════════════
#
# Prerequisites before pasting:
#   - rag_idx  = load_rag_index()  (add to @st.cache_resource block in app.py)
#   - pip install anthropic>=0.97.0
#   - Add:  import anthropic, import os  to app.py imports

TAB3_STREAMLIT_CODE = '''
# ── Add these imports to the top of app.py ──────────────────────
import anthropic
import os

# ── Add this cached loader alongside load_catalog() etc. ────────
@st.cache_resource
def load_rag_index():
    idx_path = ROOT / "llm" / "index" / "chunks.json"
    if not idx_path.exists():
        return None
    with open(idx_path) as f:
        return json.load(f)

rag_idx = load_rag_index()

# ── Replace the two-tab layout line with three tabs: ────────────
# tab1, tab2, tab3 = st.tabs(["🗺  Event Map", "〰  Waveform Viewer", "💬  LLM Q&A"])

# ── Tab 3 body ──────────────────────────────────────────────────
with tab3:
    st.subheader("Grounded Q&A — Seismic Detection Pipeline")
    st.caption(
        "This Q&A system answers **only** from indexed project artifacts. "
        "Every answer cites the specific file it draws from. "
        "Questions outside the artifact scope are declined."
    )

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        api_key = st.text_input(
            "Anthropic API key (not stored, session only)",
            type="password",
        )

    if not api_key:
        st.info("Enter an Anthropic API key above to enable Q&A.")
    elif rag_idx is None:
        st.error("RAG index not found. Run `python3 llm/index/build_index.py` first.")
    else:
        vocab     = rag_idx["vocab"]
        idf_arr   = np.array(rag_idx["idf"], dtype=np.float32)
        tfidf_mat = np.array(rag_idx["tfidf_matrix"], dtype=np.float32)
        chunks    = rag_idx["chunks"]
        vocab_map = {t: i for i, t in enumerate(vocab)}
        STOPWORDS_QA = {"the","a","an","and","or","in","of","to","is","are","was","were",
                        "it","its","this","that","with","for","from","by","at","on","as"}

        def retrieve(query, top_k=5):
            from collections import Counter as _Counter
            words = [w for w in re.findall(r\'\\b[a-z]\\w*\\b\', query.lower())
                     if w not in STOPWORDS_QA and len(w) > 1]
            tf = _Counter(words)
            qvec = np.zeros(len(vocab), dtype=np.float32)
            for t, cnt in tf.items():
                if t in vocab_map:
                    qvec[vocab_map[t]] = cnt * idf_arr[vocab_map[t]]
            norm = np.linalg.norm(qvec)
            if norm == 0:
                return []
            qvec /= norm
            scores = tfidf_mat @ qvec
            top_idx = np.argsort(scores)[::-1][:top_k]
            return [(float(scores[i]), chunks[i]) for i in top_idx if scores[i] > 0.01]

        SYSTEM_PROMPT = (
            "You are the Q&A assistant for the Seismic AI Pipeline project by "
            "Bertuğ Taş, Kadir Emir Yücel, Melih Takyaci, Emre Özdemir, and Efendi Nasiboğlu "
            "(Dokuz Eylül University, III. National Basic Sciences Youth Symposium 2026).\\n\\n"
            "STRICT RULES:\\n"
            "1. Answer ONLY from the provided artifact context.\\n"
            "2. Every factual claim must be followed by (source: filename).\\n"
            "3. If the answer cannot be found, respond: \\"This question is outside the scope of "
            "the indexed project artifacts.\\"\\n"
            "4. NEVER use \\"earthquake prediction\\" or \\"earthquake forecasting\\". "
            "Use: \\"automatic seismic event detection\\".\\n"
            "5. When reporting metrics, always state which phase and evaluation subset."
        )

        suggestions = [
            "What is the recall of PhaseNet on M 2–3 events in Phase 5?",
            "Why is EQTransformer recall so low compared to PhaseNet?",
            "What is Limitation L10 and how does it affect Phase 5 results?",
            "How were the Phase 5 stations selected?",
            "What preprocessing steps are applied to each waveform window?",
            "What is the detection criterion used for all models?",
            "How does Phase 5 recall compare to Phase 4 pilot recall?",
        ]
        cols = st.columns(2)
        for i, sug in enumerate(suggestions):
            if cols[i % 2].button(sug, key=f"sug_{i}", use_container_width=True):
                st.session_state["qa_input"] = sug

        st.markdown("---")
        query = st.text_area("Your question",
                             value=st.session_state.get("qa_input", ""), height=80)

        if st.button("Ask", type="primary", disabled=not query.strip()):
            with st.spinner("Retrieving and generating..."):
                retrieved = retrieve(query, top_k=5)
                context = "\\n\\n---\\n\\n".join(
                    f"[Source: {ck[\'source\']}]\\n{ck[\'text\']}" for _, ck in retrieved
                )
                client = anthropic.Anthropic(api_key=api_key)
                response = client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=1024,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user",
                               "content": f"<context>\\n{context}\\n</context>\\n\\nQuestion: {query}"}],
                )
                st.markdown("### Answer")
                st.markdown(response.content[0].text)
            if retrieved:
                with st.expander("Retrieved context"):
                    for score, ck in retrieved:
                        st.markdown(f"`{ck[\'source\']}` — score: {score:.3f}")
                        st.text(ck["text"][:400] + "…")
                        st.markdown("---")
'''


if __name__ == "__main__":
    print("llm_future.py — LLM/RAG code saved for future re-integration.")
    print("See BUILD_INDEX_CODE and TAB3_STREAMLIT_CODE strings above.")
    print()
    print("To rebuild the index when re-integrating:")
    print("  mkdir -p llm/index")
    print("  python3 -c \"exec(open('llm_future.py').read()); exec(BUILD_INDEX_CODE)\"")
