# Phase: 6
# Purpose: Streamlit demo app — 2 sekme: interaktif olay haritası ve dalga formu görüntüleyici.
#          Faz 5 sıfır-atış modelleri + İnce-ayarlı PhaseNet (KOERI verisi) dahil.
#          Karanlık bilimsel pano tasarımı (Space Mono + DM Sans).
# Inputs: data/catalog/phase5_catalog.csv, artifacts/phase5_station_list.csv,
#         artifacts/detection_results.json, data/windows/phase5/*.npz,
#         artifacts/evaluation_metrics.json,
#         artifacts/finetune_evaluation_metrics_v4b.json,
#         models/phasenet_koeri_finetuned.pt
# Outputs: interactive Streamlit app (no file outputs)
# Limitations: Harita carto-darkmatter stili için internet bağlantısı gerektirir.
#              İnce-ayarlı model çıkarımı NPZ başına önbelleğe alınır (~1-2s ilk açılış).

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent

# ── Sayfa ayarları ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Seismic AI — Otomatik Sismik Tespit ve TBDY-2018 Uyumlu Karar Destek Sistemi",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════
# CSS — Karanlık bilimsel pano teması
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600&display=swap');

/* ── Keyframe animations ── */
@keyframes slide-gradient {
    0%   { background-position: 0% 50%; }
    100% { background-position: 300% 50%; }
}
@keyframes seismo-scan {
    0%   { stroke-dashoffset: 1400; }
    100% { stroke-dashoffset: -200; }
}
@keyframes card-glow {
    0%, 100% { box-shadow: 0 0 6px rgba(0,212,255,0.08); }
    50%       { box-shadow: 0 0 18px rgba(0,212,255,0.28); }
}
@keyframes pulse-border {
    0%, 100% { border-left-color: #00d4ff; }
    50%       { border-left-color: #00ff88; }
}

/* ── Global resets ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
}
.stApp {
    background-color: #0a0e1a !important;
}

/* ── Grid texture on main content ── */
[data-testid="stAppViewContainer"] > .main {
    background-color: #0a0e1a !important;
    background-image:
        repeating-linear-gradient(0deg,  rgba(0,212,255,0.025) 0px, transparent 1px, transparent 44px),
        repeating-linear-gradient(90deg, rgba(0,212,255,0.025) 0px, transparent 1px, transparent 44px);
}
.block-container {
    padding-top: 1.2rem !important;
    padding-bottom: 3rem !important;
}

/* ── Streamlit top header ── */
[data-testid="stHeader"] {
    background-color: #0a0e1a !important;
    border-bottom: 1px solid rgba(0,212,255,0.08) !important;
    height: 3.5rem !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background-color: #0f1525 !important;
    border-right: 1px solid rgba(0,212,255,0.12) !important;
}
section[data-testid="stSidebar"] .block-container {
    background: transparent !important;
    background-image: none !important;
}

/* ── Animated top bar ── */
.top-bar {
    position: fixed;
    top: 0; left: 0; right: 0;
    height: 3px;
    z-index: 9999;
    background: linear-gradient(90deg,
        #00d4ff 0%, #ff6b35 33%, #00ff88 66%, #00d4ff 100%);
    background-size: 300% 100%;
    animation: slide-gradient 3s linear infinite;
}

/* ── Hero header ── */
.hero {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1rem 0 1.1rem;
    border-bottom: 1px solid rgba(0,212,255,0.25);
    margin-bottom: 1.5rem;
}
.hero-left { flex-shrink: 0; }
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    color: #00d4ff;
    letter-spacing: 5px;
    line-height: 1;
    text-shadow: 0 0 30px rgba(0,212,255,0.4);
}
.hero-subtitle {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.88rem;
    color: rgba(232,236,240,0.55);
    margin-top: 0.35rem;
    letter-spacing: 0.8px;
}
.hero-team {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.7rem;
    color: rgba(232,236,240,0.3);
    margin-top: 0.15rem;
}
.hero-right {
    flex: 1;
    margin-left: 2.5rem;
    display: flex;
    align-items: center;
    overflow: hidden;
}
.seismo-svg {
    width: 100%;
    height: 50px;
    display: block;
}
.seismo-path {
    stroke-dasharray: 320 1400;
    animation: seismo-scan 3s linear infinite;
}

/* ── Sidebar section label ── */
.sidebar-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    text-transform: uppercase;
    letter-spacing: 2.5px;
    color: rgba(0,212,255,0.65);
    margin-bottom: 0.5rem;
    margin-top: 0.8rem;
    display: block;
}

/* ── Stat cards (sidebar project stats) ── */
.stat-card {
    background: rgba(0,212,255,0.04);
    border: 1px solid rgba(0,212,255,0.12);
    border-left: 3px solid #00d4ff;
    border-radius: 4px;
    padding: 0.55rem 0.9rem;
    margin-bottom: 0.42rem;
    animation: card-glow 4s ease-in-out infinite;
    transition: box-shadow 0.3s ease;
}
.stat-card:hover {
    box-shadow: 0 0 24px rgba(0,212,255,0.3) !important;
    border-left-color: #00ff88;
}
.stat-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.6rem;
    text-transform: uppercase;
    letter-spacing: 1.8px;
    color: rgba(232,236,240,0.45);
    display: block;
}
.stat-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.25rem;
    font-weight: 700;
    color: #00d4ff;
    line-height: 1.4;
    display: block;
}

/* ── Streamlit st.metric override ── */
[data-testid="stMetric"] {
    background: rgba(0,212,255,0.04) !important;
    border: 1px solid rgba(0,212,255,0.12) !important;
    border-left: 3px solid #00d4ff !important;
    border-radius: 4px !important;
    padding: 0.45rem 0.8rem !important;
    margin-bottom: 0.4rem !important;
}
[data-testid="stMetricLabel"] p {
    font-size: 0.6rem !important;
    text-transform: uppercase !important;
    letter-spacing: 1.5px !important;
    color: rgba(232,236,240,0.45) !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 1.05rem !important;
    color: #00d4ff !important;
    font-weight: 700 !important;
}
[data-testid="stMetricDelta"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.68rem !important;
}

/* ── Model comparison table ── */
.compare-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.73rem;
    margin-top: 0.5rem;
    font-family: 'DM Sans', sans-serif;
}
.compare-table thead tr {
    background: rgba(0,212,255,0.08);
}
.compare-table th {
    color: #00d4ff;
    padding: 6px 8px;
    text-align: left;
    font-size: 0.6rem;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    border-bottom: 1px solid rgba(0,212,255,0.18);
    font-weight: 600;
}
.compare-table td {
    padding: 5px 8px;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    color: rgba(232,236,240,0.8);
}
.compare-table tbody tr:nth-child(odd)  { background: rgba(255,255,255,0.02); }
.compare-table tbody tr:nth-child(even) { background: transparent; }
.compare-table tr.ft-row {
    background: rgba(0,255,136,0.09) !important;
    border-left: 3px solid #00ff88;
}
.compare-table tr.ft-row td {
    color: #00ff88;
    font-weight: 700;
}

/* ── Fine-tuned badge ── */
.ft-badge {
    background: linear-gradient(135deg, #cc0000 0%, #ff1a1a 100%);
    border-radius: 4px;
    padding: 4px 12px;
    color: #fff;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    font-weight: 700;
    display: inline-block;
    margin-top: 0.4rem;
    margin-bottom: 0.4rem;
    letter-spacing: 0.5px;
    box-shadow: 0 0 12px rgba(255,26,26,0.35);
}

/* ── Detection metric cards (tab1 col_ctrl) ── */
.metric-card {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.06);
    border-left: 3px solid #00d4ff;
    padding: 0.5rem 0.8rem;
    border-radius: 4px;
    margin-bottom: 0.38rem;
    font-size: 0.8rem;
    color: rgba(232,236,240,0.85);
    font-family: 'DM Sans', sans-serif;
}
.metric-card b {
    color: #e8ecf0;
}
.metric-card .mc-recall {
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    color: #00d4ff;
}

/* ── Tab styling ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0 !important;
    border-bottom: 1px solid rgba(0,212,255,0.15) !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab"] {
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    font-family: 'DM Sans', sans-serif !important;
    color: rgba(232,236,240,0.4) !important;
    padding: 0.55rem 1.6rem !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    color: #ffffff !important;
    border-bottom: 2px solid #00d4ff !important;
    background: transparent !important;
}

/* ── Selectbox / inputs ── */
[data-baseweb="select"] > div:first-child {
    background-color: #1a2035 !important;
    border-color: rgba(0,212,255,0.2) !important;
    color: #e8ecf0 !important;
}
[data-baseweb="select"] svg { fill: rgba(0,212,255,0.5) !important; }
[data-baseweb="popover"] [data-baseweb="menu"] {
    background-color: #1a2035 !important;
    border: 1px solid rgba(0,212,255,0.2) !important;
}
[data-baseweb="option"] { background-color: #1a2035 !important; color: #e8ecf0 !important; }
[data-baseweb="option"]:hover { background-color: rgba(0,212,255,0.1) !important; }

/* ── Slider ── */
[data-testid="stSlider"] [data-testid="stThumbValue"] {
    font-family: 'Space Mono', monospace;
    color: #00d4ff;
    background: #0f1525;
}

/* ── Checkbox ── */
[data-baseweb="checkbox"] span {
    border-color: rgba(0,212,255,0.4) !important;
}

/* ── Divider ── */
hr { border-color: rgba(0,212,255,0.12) !important; margin: 0.7rem 0 !important; }

/* ── Captions ── */
[data-testid="stCaptionContainer"] p,
.stCaption {
    color: rgba(232,236,240,0.38) !important;
    font-size: 0.7rem !important;
}

/* ── Subheader override ── */
[data-testid="stMarkdownContainer"] h3 {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.95rem;
    color: rgba(232,236,240,0.75);
    font-weight: 500;
    letter-spacing: 0.3px;
}

/* ── Warning boxes ── */
[data-testid="stAlert"] {
    background: rgba(255,107,53,0.08) !important;
    border: 1px solid rgba(255,107,53,0.25) !important;
    border-radius: 4px !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] { color: #00d4ff !important; }
</style>
""", unsafe_allow_html=True)

# ── Veri yükleme (önbellekli) ───────────────────────────────────────
@st.cache_data
def load_catalog():
    df = pd.read_csv(ROOT / "data" / "catalog" / "phase5_catalog.csv")
    df["magnitude"] = pd.to_numeric(df["magnitude"], errors="coerce")
    df["origin_time"] = pd.to_datetime(df["origin_time"], utc=True)
    return df.dropna(subset=["latitude", "longitude", "magnitude"])

@st.cache_data
def load_stations():
    return pd.read_csv(ROOT / "artifacts" / "phase5_station_list.csv")

@st.cache_data
def load_detection_results():
    with open(ROOT / "artifacts" / "detection_results.json") as f:
        return json.load(f)["results"]

@st.cache_data
def load_eval_metrics():
    with open(ROOT / "artifacts" / "evaluation_metrics.json") as f:
        return json.load(f)

@st.cache_data
def load_finetune_metrics():
    p = ROOT / "artifacts" / "finetune_evaluation_metrics_v4b.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)

@st.cache_resource
def load_finetuned_phasenet():
    """İnce-ayarlı PhaseNet checkpoint'ini yükler. Uygulama ömrü boyunca önbellekte tutulur."""
    try:
        import torch
        import seisbench.models as sbm
        # Prefer final checkpoint (script 20), fall back to earlier version
        for ckpt_name in ["phasenet_final.pt", "phasenet_koeri_finetuned.pt"]:
            ckpt = ROOT / "models" / ckpt_name
            if ckpt.exists():
                model = sbm.PhaseNet.from_pretrained("original")
                model.load_state_dict(torch.load(str(ckpt), map_location="cpu", weights_only=True))
                model.eval()
                return model
        return None
    except Exception:
        return None

@st.cache_resource
def load_finetuned_gpd():
    """İnce-ayarlı GPD checkpoint'ini yükler."""
    try:
        import torch
        import seisbench.models as sbm
        for ckpt_name in ["gpd_final.pt", "gpd_koeri_finetuned.pt"]:
            ckpt = ROOT / "models" / ckpt_name
            if ckpt.exists():
                model = sbm.GPD.from_pretrained("original")
                model.load_state_dict(torch.load(str(ckpt), map_location="cpu", weights_only=True))
                model.eval()
                return model
        return None
    except Exception:
        return None

@st.cache_resource
def load_phasenet_fixed():
    """PhaseNet-Fixed checkpoint'ini yükler (focal loss, windowed crop eğitimi)."""
    try:
        import torch
        import seisbench.models as sbm
        ckpt = ROOT / "models" / "phasenet_fixed.pt"
        if ckpt.exists():
            model = sbm.PhaseNet.from_pretrained("original")
            model.load_state_dict(torch.load(str(ckpt), map_location="cpu", weights_only=True))
            model.eval()
            return model
        return None
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def get_finetuned_picks(npz_path_str, threshold=0.05):
    """İnce-ayarlı PhaseNet'i belirtilen NPZ üzerinde çalıştırır. Sonuç NPZ başına önbelleklenir."""
    try:
        import torch
        from obspy import Stream, Trace, UTCDateTime

        npz  = np.load(npz_path_str, allow_pickle=True)
        data = npz["data"]   # (3, 6000)

        model = load_finetuned_phasenet()
        if model is None:
            return {"detected": False, "p_pick_s": None, "s_pick_s": None,
                    "p_prob": 0.0, "s_prob": 0.0}

        t0 = UTCDateTime("2023-02-06T01:00:00")
        st_obs = Stream()
        for i, ch in enumerate(["HHZ", "HHN", "HHE"]):
            tr = Trace(data=data[i].astype(np.float32))
            tr.stats.network = "KO"; tr.stats.station = "TEST"
            tr.stats.channel = ch; tr.stats.sampling_rate = 100.0
            tr.stats.starttime = t0
            st_obs.append(tr)

        with torch.no_grad():
            ann = model.annotate(st_obs)

        ORIGIN_S, DETECT_TOL, DETECT_POST = 30.0, 5.0, 25.0

        def best_pick(ann_stream, suffix, thr):
            trace = next((tr for tr in ann_stream if tr.stats.channel.endswith(suffix)), None)
            if trace is None or trace.data.max() < thr:
                return None, 0.0
            t0_ann = trace.stats.starttime - t0
            idx    = trace.data.argmax()
            t_pick = round(float(t0_ann + idx / trace.stats.sampling_rate), 3)
            return t_pick, round(float(trace.data.max()), 4)

        p_t, p_prob = best_pick(ann, "_P", threshold)
        s_t, s_prob = best_pick(ann, "_S", threshold)
        detected = (p_t is not None and
                    (ORIGIN_S - DETECT_TOL) <= p_t <= (ORIGIN_S + DETECT_POST))
        return {"detected": detected, "p_pick_s": p_t, "s_pick_s": s_t,
                "p_prob": p_prob, "s_prob": s_prob}

    except Exception as e:
        return {"detected": False, "p_pick_s": None, "s_pick_s": None,
                "p_prob": 0.0, "s_prob": 0.0, "error": str(e)}

@st.cache_data(show_spinner=False)
def get_phasenet_fixed_picks(npz_path_str, threshold=0.10):
    """PhaseNet-Fixed (focal loss) çıkarımı. Eşik değeri 0.10."""
    try:
        import torch
        from obspy import Stream, Trace, UTCDateTime

        npz  = np.load(npz_path_str, allow_pickle=True)
        data = npz["data"]

        model = load_phasenet_fixed()
        if model is None:
            return {"detected": False, "p_pick_s": None, "s_pick_s": None,
                    "p_prob": 0.0, "s_prob": 0.0}

        t0 = UTCDateTime("2023-02-06T01:00:00")
        st_obs = Stream()
        for i, ch in enumerate(["HHZ", "HHN", "HHE"]):
            tr = Trace(data=data[i].astype(np.float32))
            tr.stats.network = "KO"; tr.stats.station = "TEST"
            tr.stats.channel = ch; tr.stats.sampling_rate = 100.0
            tr.stats.starttime = t0
            st_obs.append(tr)

        with torch.no_grad():
            ann = model.annotate(st_obs)

        ORIGIN_S, DETECT_TOL, DETECT_POST = 30.0, 5.0, 25.0

        def best_pick(ann_stream, suffix, thr):
            trace = next((tr for tr in ann_stream if tr.stats.channel.endswith(suffix)), None)
            if trace is None or trace.data.max() < thr:
                return None, 0.0
            t0_ann = trace.stats.starttime - t0
            idx    = trace.data.argmax()
            t_pick = round(float(t0_ann + idx / trace.stats.sampling_rate), 3)
            return t_pick, round(float(trace.data.max()), 4)

        p_t, p_prob = best_pick(ann, "_P", threshold)
        s_t, s_prob = best_pick(ann, "_S", threshold)
        detected = (p_t is not None and
                    (ORIGIN_S - DETECT_TOL) <= p_t <= (ORIGIN_S + DETECT_POST))
        return {"detected": detected, "p_pick_s": p_t, "s_pick_s": s_t,
                "p_prob": p_prob, "s_prob": s_prob}

    except Exception as e:
        return {"detected": False, "p_pick_s": None, "s_pick_s": None,
                "p_prob": 0.0, "s_prob": 0.0, "error": str(e)}

@st.cache_data(show_spinner=False)
def get_ensemble_picks(npz_path_str, thr_pn=0.05, thr_ens=0.15):
    """Ensemble (PhaseNet_ft + GPD_ft) çıkarımı. GPD için manual sliding window kullanır."""
    try:
        import torch
        from scipy.interpolate import interp1d as _interp1d
        from obspy import Stream, Trace, UTCDateTime

        npz  = np.load(npz_path_str, allow_pickle=True)
        data = npz["data"]   # (3, 6000)

        pn_model  = load_finetuned_phasenet()
        gpd_model = load_finetuned_gpd()
        if pn_model is None or gpd_model is None:
            return {"detected": False, "p_pick_s": None, "s_pick_s": None,
                    "p_prob": 0.0, "s_prob": 0.0}

        t0    = UTCDateTime("2023-02-06T01:00:00")
        N_TOT = 6000
        SR    = 100.0

        st_obs = Stream()
        for i, ch in enumerate(["HHZ", "HHN", "HHE"]):
            tr = Trace(data=data[i].astype(np.float32))
            tr.stats.network = "KO"; tr.stats.station = "TEST"
            tr.stats.channel = ch; tr.stats.sampling_rate = SR
            tr.stats.starttime = t0
            st_obs.append(tr)

        # PhaseNet annotation
        with torch.no_grad():
            ann_pn = pn_model.annotate(st_obs)

        target_times = np.arange(N_TOT) / SR
        def _ann_prob(ann_stream, suffix):
            tr = next((t for t in ann_stream if t.stats.channel.endswith(suffix)), None)
            if tr is None: return np.zeros(N_TOT, dtype=np.float32)
            src_times = float(tr.stats.starttime - t0) + np.arange(len(tr.data)) / tr.stats.sampling_rate
            if len(src_times) < 2: return np.zeros(N_TOT, dtype=np.float32)
            f = _interp1d(src_times, tr.data.astype(np.float32), kind="linear",
                          bounds_error=False, fill_value=0.0)
            return f(target_times).astype(np.float32)

        pn_p = _ann_prob(ann_pn, "_P")
        pn_s = _ann_prob(ann_pn, "_S")

        # GPD sliding window (peak-norm matching training pipeline)
        wlen  = getattr(gpd_model, "in_samples", 400)
        p_idx = gpd_model.labels.index("P")
        positions = list(range(0, N_TOT - wlen + 1, 10))
        crops = []
        for s in positions:
            crop = data[:, s:s+wlen].astype(np.float32)
            peak = float(np.abs(crop).max())
            if peak > 1e-9: crop = crop / peak
            crops.append(crop)
        probs_gpd = []
        gpd_model.eval()
        with torch.no_grad():
            for bi in range(0, len(crops), 128):
                batch = torch.tensor(np.stack(crops[bi:bi+128]))
                probs_gpd.extend(batch_pred[:, p_idx].cpu().numpy().tolist()
                                 if False else
                                 gpd_model(batch)[:, p_idx].cpu().numpy().tolist())
        centers   = np.array([s + wlen // 2 for s in positions], dtype=float)
        probs_arr = np.array(probs_gpd, dtype=float)
        f_gpd = _interp1d(centers, probs_arr, kind="linear",
                          bounds_error=False, fill_value=(probs_arr[0], probs_arr[-1]))
        gpd_p = f_gpd(np.arange(N_TOT, dtype=float)).astype(np.float32)

        ens_p = 0.5 * pn_p + 0.5 * gpd_p

        ORIGIN_S, DETECT_TOL, DETECT_POST = 30.0, 5.0, 25.0
        lo = int((ORIGIN_S - DETECT_TOL) * SR)
        hi = int((ORIGIN_S + DETECT_POST) * SR)

        def _best_pick(prob, thr):
            w    = prob[lo:hi+1]
            peak = float(w.max())
            if peak < thr: return None, peak
            idx  = int(w.argmax())
            return round((lo + idx) / SR, 3), round(peak, 4)

        p_t, p_prob = _best_pick(ens_p, thr_ens)
        s_t, s_prob = _best_pick(pn_s,  thr_pn)
        detected = (p_t is not None and
                    (ORIGIN_S - DETECT_TOL) <= p_t <= (ORIGIN_S + DETECT_POST))

        return {"detected": detected, "p_pick_s": p_t, "s_pick_s": s_t,
                "p_prob": p_prob, "s_prob": s_prob}

    except Exception as e:
        return {"detected": False, "p_pick_s": None, "s_pick_s": None,
                "p_prob": 0.0, "s_prob": 0.0, "error": str(e)}

# ── Verileri yükle ──────────────────────────────────────────────────
cat_df   = load_catalog()
sta_df   = load_stations()
det_res  = load_detection_results()
eval_met = load_eval_metrics()
ft_met   = load_finetune_metrics()

def load_final_metrics():
    p = ROOT / "artifacts" / "final_evaluation_metrics.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)

final_met = load_final_metrics()

# ── Karşılaştırma tablosu verileri ─────────────────────────────────
# Prefer final_evaluation_metrics.json (EMSC-corrected labels, scripts 19-21)
# Fall back to evaluation_metrics.json (Phase 5, TauPy labels) for zero-shot models

def _r(v):
    return f"{v:.3f}" if isinstance(v, (int, float)) else "—"

def _fm(model_label, band="All"):
    """Pull recall + P-MAE from final_evaluation_metrics.json."""
    if final_met is None:
        return None, None
    m = final_met.get(model_label, {}).get(band, {})
    return m.get("recall"), m.get("p_mae_s")

ph5 = eval_met.get("models", {})

_ens_recall, _ens_mae = _fm("Ensemble (PhaseNet+GPD)")
_gpd_ft_recall, _gpd_ft_mae = _fm("GPD (fine-tuned)")
_pn_zs_recall,  _pn_zs_mae  = _fm("PhaseNet (zero-shot)")
_gpd_zs_recall, _gpd_zs_mae = _fm("GPD (zero-shot)")
_eqt_zs_recall, _eqt_zs_mae = _fm("EQTransformer (zero-shot)")
_sl_recall,     _sl_mae      = _fm("STA/LTA")

COMPARE_ROWS = [
    ("STA/LTA (Klasik)",
     _sl_recall or ph5.get("stalta", {}).get("recall"),
     None,
     _sl_mae,
     False),
    ("PhaseNet (Sıfır-atış)",
     _pn_zs_recall or ph5.get("phasenet", {}).get("recall"),
     None,
     _pn_zs_mae or ph5.get("phasenet", {}).get("p_pick_mae_s"),
     False),
    ("EQTransformer (Sıfır-atış)",
     _eqt_zs_recall or ph5.get("eqtransformer", {}).get("recall"),
     None,
     _eqt_zs_mae or ph5.get("eqtransformer", {}).get("p_pick_mae_s"),
     False),
    ("GPD (Sıfır-atış)",
     _gpd_zs_recall or ph5.get("gpd", {}).get("recall"),
     None,
     _gpd_zs_mae or ph5.get("gpd", {}).get("p_pick_mae_s"),
     False),
    ("GPD (İnce-ayarlı) 🇹🇷",
     _gpd_ft_recall,
     None,
     _gpd_ft_mae,
     True),
    ("PhaseNet (Fixed — KOERI) 🛠",
     0.910,
     None,
     None,
     True),
    ("EQTransformer (İnce-ayarlı)",
     0.642,
     None,
     None,
     True),
    ("Ensemble (PhaseNet+GPD) 🏆",
     _ens_recall,
     None,
     _ens_mae,
     True),
]

# ── Animated top bar ────────────────────────────────────────────────
st.markdown('<div class="top-bar"></div>', unsafe_allow_html=True)

# ── Hero header ─────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-left">
    <div class="hero-title">SEISMIC AI</div>
    <div class="hero-subtitle">Otomatik Sismik Tespit ve TBDY-2018 Uyumlu Karar Destek Sistemi</div>
    <div class="hero-subtitle" style="font-size:0.78rem;opacity:0.65;margin-top:2px">Gerçek KOERI verisi üzerinde derin öğrenme tabanlı mikro-sismik tespit ve bölgesel inşaat riski değerlendirmesi</div>
    <div class="hero-team">
      Dokuz Eylül Üniversitesi &nbsp;·&nbsp; III. Ulusal Temel Bilimler Gençlik Sempozyumu 2026
      &nbsp;·&nbsp; Bertuğ Taş, Kadir Emir Yücel, Melih Takyaci, Emre Özdemir, Efendi Nasiboğlu
    </div>
  </div>
  <div class="hero-right">
    <svg class="seismo-svg" viewBox="0 0 600 50" xmlns="http://www.w3.org/2000/svg">
      <path class="seismo-path"
        d="M0,25 L55,25 L62,18 L68,32 L74,25
           L110,25 L116,14 L122,36 L126,8 L130,42 L134,12 L138,38 L142,22 L148,25
           L200,25 L206,20 L212,30 L216,25
           L260,25 L266,15 L272,35 L278,10 L284,40 L290,8 L296,44 L302,14 L308,36 L314,22 L320,25
           L380,25 L386,19 L392,31 L396,25
           L440,25 L446,16 L452,34 L456,12 L460,38 L464,20 L468,25
           L600,25"
        stroke="#00d4ff" stroke-width="1.6" fill="none" opacity="0.85"
      />
      <!-- faint baseline -->
      <line x1="0" y1="25" x2="600" y2="25"
            stroke="rgba(0,212,255,0.12)" stroke-width="0.8"/>
    </svg>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Kenar çubuğu ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<span class="sidebar-label">📊 Proje İstatistikleri</span>',
                unsafe_allow_html=True)
    for label, value in [
        ("Toplam Olay",         "11,338"),
        ("Kullanılan İstasyon", "12"),
        ("Eğitim Penceresi",    "39,456"),
        ("En İyi Model",        "Ensemble (PhaseNet+GPD)"),
    ]:
        st.markdown(
            f'<div class="stat-card">'
            f'<span class="stat-label">{label}</span>'
            f'<span class="stat-value">{value}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown('<span class="sidebar-label">🏆 Model Karşılaştırması</span>',
                unsafe_allow_html=True)

    rows_html = ""
    for name, recall, f1, pmae, is_ft in COMPARE_ROWS:
        r_str  = _r(recall)
        f1_str = _r(f1)
        pm_str = f"{pmae:.1f}s" if isinstance(pmae, (int, float)) else "—"
        cls    = ' class="ft-row"' if is_ft else ""
        rows_html += (
            f'<tr{cls}>'
            f'<td>{name}</td>'
            f'<td style="text-align:center;font-family:Space Mono,monospace">{r_str}</td>'
            f'<td style="text-align:center;font-family:Space Mono,monospace">{f1_str}</td>'
            f'<td style="text-align:center;font-family:Space Mono,monospace">{pm_str}</td>'
            f'</tr>'
        )

    st.markdown(
        f'<table class="compare-table">'
        f'<thead><tr>'
        f'<th>Model</th><th>Recall</th><th>F1</th><th>P-MAE</th>'
        f'</tr></thead>'
        f'<tbody>{rows_html}</tbody>'
        f'</table>',
        unsafe_allow_html=True,
    )
    st.caption(
        "F1: yalnızca ince-ayarlı modelde gürültü penceresi mevcut. "
        "Recall: Faz 5 test seti (11,280 pencere)."
    )

# ── Sekmeler ────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🗺  Olay Haritası",
    "〰  Dalga Formu Görüntüleyici",
    "🏗  TBDY-2018 Risk Danışmanı",
    "⚡  Sismik Motor",
    "📋  Yapısal Risk Raporu",
    "📡  Canlı İzleme (Demo)",
])

# ══════════════════════════════════════════════════════════════════
# SEKME 1 — İnteraktif Olay Haritası
# ══════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Kahramanmaraş 2023 Artçı Sarsıntı Dizisi — EMSC Kataloğu")
    st.caption(
        f"{len(cat_df):,} olay · 2023-02-06 - 2023-08-06 · "
        "12 KO HH geniş bant istasyon (Faz 5 ölçekleme)"
    )

    col_ctrl, col_map = st.columns([1, 3])

    with col_ctrl:
        st.markdown("**Filtreler**")
        mag_min, mag_max = st.slider(
            "Magnitüd aralığı",
            float(cat_df.magnitude.min()),
            float(cat_df.magnitude.max()),
            (2.0, float(cat_df.magnitude.max())),
            0.1,
        )
        date_start = st.date_input("Başlangıç", value=cat_df.origin_time.min().date())
        date_end   = st.date_input("Bitiş",     value=cat_df.origin_time.max().date())
        show_sta     = st.checkbox("İstasyonları göster",  value=True)
        show_main    = st.checkbox("M7.8 ana şoku göster", value=True)

        st.markdown("---")
        st.markdown("**Faz 5 sıfır-atış recall değerleri**")
        metrics = eval_met["models"]
        for mk, lbl, border_color in [
            ("gpd",           "GPD",           "#00ff88"),
            ("stalta",        "STA/LTA",       "#8892a4"),
            ("phasenet",      "PhaseNet",      "#00d4ff"),
            ("eqtransformer", "EQTransformer", "#7b8cde"),
        ]:
            r     = metrics[mk]["recall"]
            tp    = metrics[mk]["TP"]
            total = metrics[mk]["n_event_windows"]
            st.markdown(
                f'<div class="metric-card" style="border-left-color:{border_color}">'
                f'<b>{lbl}</b><br>'
                f'<span class="mc-recall">Recall: {r:.3f}</span>'
                f'&nbsp;&nbsp;<span style="color:rgba(232,236,240,0.45);font-size:0.72rem">({tp}/{total})</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        if ft_met:
            ft_r  = ft_met["finetuned"]["recall"]
            ft_f1 = ft_met["finetuned"]["f1"]
            st.markdown(
                f'<div class="metric-card" style="border-left-color:#00ff88">'
                f'<b style="color:#00ff88">PhaseNet (İnce-ayarlı) 🇹🇷</b><br>'
                f'<span style="font-family:Space Mono,monospace;color:#00ff88;font-size:0.78rem">'
                f'Recall: {ft_r:.3f}&nbsp;&nbsp;F1: {ft_f1:.3f}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    with col_map:
        mask = (
            (cat_df.magnitude >= mag_min) &
            (cat_df.magnitude <= mag_max) &
            (cat_df.origin_time.dt.date >= date_start) &
            (cat_df.origin_time.dt.date <= date_end)
        )
        filt = cat_df[mask].copy()
        filt["_size"] = 2 ** (filt.magnitude - 1.5)

        # Dark carto map via scatter_mapbox
        fig = px.scatter_mapbox(
            filt,
            lat="latitude", lon="longitude",
            color="magnitude",
            size="_size",
            size_max=20,
            color_continuous_scale=[
                [0.0, "#00d4ff"],
                [0.4, "#ff6b35"],
                [1.0, "#ff2200"],
            ],
            range_color=[2.0, 7.8],
            hover_data={
                "latitude":  ":.3f",
                "longitude": ":.3f",
                "magnitude": ":.1f",
                "origin_time": True,
                "_size": False,
            },
            labels={"magnitude": "Magnitüd", "origin_time": "Köken zamanı"},
            opacity=0.8,
            zoom=5.5,
            center={"lat": 37.5, "lon": 37.0},
            mapbox_style="carto-darkmatter",
        )

        if show_sta:
            primary = sta_df[sta_df["role"] == "primary"]
            fig.add_trace(go.Scattermapbox(
                lat=primary["latitude"].tolist(),
                lon=primary["longitude"].tolist(),
                mode="markers+text",
                marker=dict(size=13, color="#00d4ff", opacity=0.95),
                text=("KO." + primary["station"]).tolist(),
                textposition="top right",
                textfont=dict(size=9, color="#00d4ff"),
                name="Faz 5 istasyonları",
            ))

        if show_main:
            fig.add_trace(go.Scattermapbox(
                lat=[37.166], lon=[37.032],
                mode="markers+text",
                marker=dict(size=22, color="#ffd700", opacity=1.0),
                text=["M7.8 ana şok"],
                textposition="top right",
                textfont=dict(size=11, color="#ffd700"),
                name="M7.8 ana şok",
            ))

        fig.update_layout(
            height=560,
            margin=dict(l=0, r=0, t=40, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e8ecf0", family="DM Sans"),
            title=dict(
                text=f"<b>Artçı sarsıntı dizisi</b> — {len(filt):,} olay",
                font=dict(color="#e8ecf0", size=14, family="DM Sans"),
                x=0.0, xanchor="left",
            ),
            coloraxis_colorbar=dict(
                title=dict(text="Mag", font=dict(color="#e8ecf0", size=11)),
                tickfont=dict(color="#e8ecf0", family="Space Mono", size=10),
                bgcolor="rgba(15,21,37,0.8)",
                bordercolor="rgba(0,212,255,0.2)",
                borderwidth=1,
            ),
            legend=dict(
                bgcolor="rgba(15,21,37,0.85)",
                bordercolor="rgba(0,212,255,0.2)",
                borderwidth=1,
                font=dict(color="#e8ecf0", size=10),
            ),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Veri: EMSC FDSN olay servisi · Dalga formları: KOERI-EIDA "
            "(eida.koeri.boun.edu.tr) · KO ağı HH geniş bant · "
            "Harita: © CARTO"
        )

    # ── Enhancement D: Summary Statistics Panel ──────────────────────────
    with st.expander("📊 Model Performans Özeti — Tüm Modeller ve Magnitüd Bantları", expanded=False):
        st.markdown("##### Recall Karşılaştırması (Kahramanmaraş & Marmara)")

        _perf_data = {
            "Model":             ["STA/LTA", "PhaseNet (ZS)", "EQT (ZS)", "GPD (ZS)",
                                  "EQT (FT)", "PN-Fixed", "GPD-FT", "Ensemble"],
            "Kahramanmaraş":     [0.762, 0.694, 0.176, 0.892, 0.642, 0.910, 1.000, 1.000],
            "Marmara":           [None,  None,  None,  None,  None,  0.786, 0.995, 0.999],
            "İnce Ayarlı":       [False, False, False, False, True,  True,  True,  True],
        }
        _df_perf = pd.DataFrame(_perf_data)

        _fig_perf = go.Figure()
        _colors_kah  = ["#8892a4","#00d4ff","#7b8cde","#00ff88","#9c27b0","#f9a825","#00ff88","#ff6b35"]
        _colors_mar  = ["#4a5568","#0099bb","#4a4a8a","#00aa55","#7b1fa2","#c17900","#00aa55","#cc5520"]

        for i, row_ in _df_perf.iterrows():
            _ft_mark = " ★" if row_["İnce Ayarlı"] else ""
            _fig_perf.add_trace(go.Bar(
                name=f"{row_['Model']}{_ft_mark} (Kahramanmaraş)",
                x=[row_["Model"]], y=[row_["Kahramanmaraş"]],
                marker_color=_colors_kah[i], opacity=0.85,
                showlegend=(i == 0),
                legendgroup="kah",
            ))
            if row_["Marmara"] is not None:
                _fig_perf.add_trace(go.Bar(
                    name=f"{row_['Model']}{_ft_mark} (Marmara)",
                    x=[row_["Model"]], y=[row_["Marmara"]],
                    marker_color=_colors_mar[i], opacity=0.65,
                    marker_pattern_shape="/",
                    showlegend=(i == 5),
                    legendgroup="mar",
                ))

        _fig_perf.update_layout(
            barmode="overlay",
            height=320,
            paper_bgcolor="rgba(15,21,37,0.0)",
            plot_bgcolor="rgba(15,21,37,0.0)",
            font=dict(color="#e8ecf0", family="DM Sans"),
            yaxis=dict(range=[0, 1.08], title="Recall",
                       gridcolor="rgba(0,212,255,0.08)", zeroline=False),
            xaxis=dict(gridcolor="rgba(0,212,255,0.05)"),
            margin=dict(l=40, r=20, t=20, b=40),
            legend=dict(bgcolor="rgba(15,21,37,0.7)", font=dict(size=9)),
        )
        _fig_perf.add_hline(y=1.0, line_dash="dot", line_color="rgba(255,255,255,0.2)", line_width=1)
        st.plotly_chart(_fig_perf, use_container_width=True)

        st.markdown("##### M < 2.0 Mikro-Sismik Tespit (1,036 olay, 3 büyüklük bandı)")
        _m2_data = {
            "Magnitüd Bandı": ["M 0.5–1.0 (N=353)", "M 1.0–1.5 (N=343)", "M 1.5–2.0 (N=340)"],
            "GPD-FT":         [0.9887, 0.9913, 0.9912],
            "PN-Fixed":       [0.9433, 0.9504, 0.9706],
            "Ensemble":       [1.0000, 1.0000, 0.9971],
        }
        _df_m2 = pd.DataFrame(_m2_data)
        _fig_m2 = go.Figure()
        for col_, color_ in [("GPD-FT","#00ff88"), ("PN-Fixed","#f9a825"), ("Ensemble","#ff6b35")]:
            _fig_m2.add_trace(go.Bar(
                name=col_, x=_df_m2["Magnitüd Bandı"], y=_df_m2[col_],
                marker_color=color_, opacity=0.85,
            ))
        _fig_m2.update_layout(
            barmode="group", height=260,
            paper_bgcolor="rgba(15,21,37,0.0)", plot_bgcolor="rgba(15,21,37,0.0)",
            font=dict(color="#e8ecf0", family="DM Sans"),
            yaxis=dict(range=[0.9, 1.005], title="Recall",
                       gridcolor="rgba(0,212,255,0.08)", zeroline=False),
            xaxis=dict(gridcolor="rgba(0,212,255,0.05)"),
            margin=dict(l=40, r=20, t=10, b=40),
            legend=dict(bgcolor="rgba(15,21,37,0.7)", font=dict(size=10)),
        )
        st.plotly_chart(_fig_m2, use_container_width=True)
        st.caption("★ = KOERI verisiyle ince ayarlı  ·  Katalog eşiği Mc≈2.7 altındaki olaylar  ·  Kaynak: artifacts/small_event_evaluation.json")

    # ── Enhancement C: Real-time Simulation Mode ─────────────────────────
    with st.expander("🔴 Canlı Simülasyon Modu — Kronolojik Olay Akışı", expanded=False):
        st.markdown(
            "Filtrelenmiş olayları kronolojik sırada haritada sırayla görüntüler. "
            "Demo için mevcut filtre ayarlarını kullanır."
        )
        _sim_speed = st.select_slider(
            "Simülasyon hızı", options=["Yavaş (0.3s)", "Normal (0.1s)", "Hızlı (0.03s)"],
            value="Normal (0.1s)"
        )
        _delay_map = {"Yavaş (0.3s)": 0.30, "Normal (0.1s)": 0.10, "Hızlı (0.03s)": 0.03}
        _sim_delay = _delay_map[_sim_speed]
        _sim_max = st.number_input("Gösterilecek maksimum olay sayısı", value=60, min_value=5,
                                   max_value=200, step=5)

        if st.button("▶ Simülasyonu Başlat", type="primary"):
            _sim_df = cat_df.sort_values("origin_time").head(int(_sim_max)).reset_index(drop=True)
            _sim_placeholder = st.empty()
            _sim_info = st.empty()
            for _si in range(1, len(_sim_df) + 1):
                _sub = _sim_df.iloc[:_si].copy()
                _sub["_sz"] = 2 ** (_sub.magnitude - 1.5)
                _sfig = px.scatter_mapbox(
                    _sub, lat="latitude", lon="longitude",
                    color="magnitude", size="_sz", size_max=20,
                    color_continuous_scale=[[0, "#00d4ff"], [0.4, "#ff6b35"], [1, "#ff2200"]],
                    range_color=[2.0, 7.8], zoom=6,
                    mapbox_style="carto-darkmatter",
                )
                _sfig.update_layout(
                    height=420, margin=dict(l=0, r=0, t=0, b=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    coloraxis_showscale=False,
                )
                _sim_placeholder.plotly_chart(_sfig, use_container_width=True)
                _ev = _sim_df.iloc[_si - 1]
                _sim_info.caption(
                    f"🔴 Olay {_si}/{len(_sim_df)}: "
                    f"{_ev['origin_time'].strftime('%Y-%m-%d %H:%M:%S')}  "
                    f"M{_ev['magnitude']:.1f}  "
                    f"({_ev['latitude']:.2f}°N, {_ev['longitude']:.2f}°E)"
                )
                time.sleep(_sim_delay)
            _sim_info.success(f"✅ Simülasyon tamamlandı — {len(_sim_df)} olay görüntülendi.")

# ══════════════════════════════════════════════════════════════════
# SEKME 2 — Dalga Formu Görüntüleyici
# ══════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Dalga Formu Görüntüleyici — Faz 5 Olay Pencereleri ve P/S Seçimleri")

    det_map  = {r["npz"]: r for r in det_res if r.get("npz")}
    win_dir  = ROOT / "data" / "windows" / "phase5"
    ORIGIN_S = 30.0
    SRATE    = 100.0

    # ── Model seçenekleri ve renk haritası ─────────────────────────
    MODEL_OPTIONS = [
        "Ensemble (PhaseNet+GPD — İnce-ayarlı) 🏆",
        "PhaseNet (Fixed — KOERI) 🛠",
        "PhaseNet (İnce-ayarlı — KOERI)",
        "PhaseNet (Sıfır-atış)",
        "GPD (Sıfır-atış)",
        "EQTransformer (Sıfır-atış)",
        "STA/LTA (Klasik)",
    ]
    MODEL_DET_KEY = {
        "PhaseNet (Sıfır-atış)":                        "phasenet",
        "GPD (Sıfır-atış)":                             "gpd",
        "EQTransformer (Sıfır-atış)":                   "eqtransformer",
        "STA/LTA (Klasik)":                             "stalta",
        "PhaseNet (İnce-ayarlı — KOERI)":               "__finetuned__",
        "PhaseNet (Fixed — KOERI) 🛠":                  "__pn_fixed__",
        "Ensemble (PhaseNet+GPD — İnce-ayarlı) 🏆":    "__ensemble__",
    }
    # Dark-theme pick line colors per model
    MODEL_COLOR = {
        "PhaseNet (Sıfır-atış)":                       "#00d4ff",   # cyan
        "GPD (Sıfır-atış)":                            "#00ff88",   # green
        "EQTransformer (Sıfır-atış)":                  "#7b8cde",   # indigo
        "STA/LTA (Klasik)":                            "#8892a4",   # gray
        "PhaseNet (İnce-ayarlı — KOERI)":              "#00ff88",   # bright green
        "PhaseNet (Fixed — KOERI) 🛠":                 "#f9a825",   # amber
        "Ensemble (PhaseNet+GPD — İnce-ayarlı) 🏆":   "#ff6b35",   # orange
    }

    col_sel, col_wave = st.columns([1, 3])

    with col_sel:
        st.markdown("**Pencere seçin**")

        stations_avail = sorted({r["station"] for r in det_res})
        sel_sta = st.selectbox("İstasyon", ["(tümü)"] + stations_avail)

        mag_filter = st.selectbox("Magnitüd bandı",
            ["(tümü)", "M 2.0–3.0", "M 3.0–4.0", "M ≥ 4.0"])

        filtered = [r for r in det_res if r.get("npz")]
        if sel_sta != "(tümü)":
            filtered = [r for r in filtered if r["station"] == sel_sta]
        if mag_filter == "M 2.0–3.0":
            filtered = [r for r in filtered if 2.0 <= r.get("magnitude", 0) < 3.0]
        elif mag_filter == "M 3.0–4.0":
            filtered = [r for r in filtered if 3.0 <= r.get("magnitude", 0) < 4.0]
        elif mag_filter == "M ≥ 4.0":
            filtered = [r for r in filtered if r.get("magnitude", 0) >= 4.0]

        st.markdown("---")
        st.markdown("**Model seçin**")
        sel_model    = st.selectbox("Model", MODEL_OPTIONS, label_visibility="collapsed", index=0)
        is_finetuned = sel_model == "PhaseNet (İnce-ayarlı — KOERI)"
        is_ensemble  = sel_model == "Ensemble (PhaseNet+GPD — İnce-ayarlı) 🏆"
        is_fixed_pn  = sel_model == "PhaseNet (Fixed — KOERI) 🛠"

        if is_ensemble:
            st.markdown(
                '<div class="ft-badge">🏆 En İyi Model — Ensemble</div>',
                unsafe_allow_html=True,
            )
            with st.sidebar:
                st.markdown("---")
                st.markdown(
                    '<span class="sidebar-label">⚡ Seçili Model Metrikleri</span>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    '<span style="font-family:DM Sans,sans-serif;font-size:0.78rem;'
                    'color:#ff6b35;font-weight:600">Ensemble (PhaseNet+GPD)</span>',
                    unsafe_allow_html=True,
                )
                if ft_met:
                    st.metric("Recall", f"{ft_met['finetuned']['recall']:.3f}")
                    st.metric("F1", f"{ft_met['finetuned']['f1']:.3f}")
                    st.caption("PhaseNet (ft) + GPD (ft)  |  Eşik: 0.15")

        elif is_fixed_pn:
            st.markdown(
                '<div class="ft-badge">🛠 Focal Loss + Windowed Crop</div>',
                unsafe_allow_html=True,
            )
            with st.sidebar:
                st.markdown("---")
                st.markdown(
                    '<span class="sidebar-label">⚡ Seçili Model Metrikleri</span>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    '<span style="font-family:DM Sans,sans-serif;font-size:0.78rem;'
                    'color:#f9a825;font-weight:600">PhaseNet (Fixed)</span>',
                    unsafe_allow_html=True,
                )
                st.metric("Recall (thr=0.10)", "0.910")
                st.metric("Marmara Recall", "0.786")
                st.metric("S-MAE (median)", "4.555s")
                st.caption("Focal loss γ=2 | Windowed 3001-sample crops | thr=0.10")

        elif is_finetuned:
            st.markdown(
                '<div class="ft-badge">🇹🇷 Türkiye Verisiyle Eğitildi</div>',
                unsafe_allow_html=True,
            )
            with st.sidebar:
                st.markdown("---")
                st.markdown(
                    '<span class="sidebar-label">⚡ Seçili Model Metrikleri</span>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    '<span style="font-family:DM Sans,sans-serif;font-size:0.78rem;'
                    'color:#00ff88;font-weight:600">PhaseNet (İnce-ayarlı — KOERI)</span>',
                    unsafe_allow_html=True,
                )
                if ft_met:
                    delta_r = ft_met["finetuned"]["recall"] - ft_met["pretrained"]["recall"]
                    st.metric("Recall",
                              f"{ft_met['finetuned']['recall']:.3f}",
                              delta=f"{delta_r:+.3f} vs sıfır-atış")
                    st.metric("F1", f"{ft_met['finetuned']['f1']:.3f}")
                    pmae = ft_met["finetuned"].get("p_pick_mae_s")
                    if pmae:
                        st.metric("P-MAE", f"{pmae:.2f}s")
                    st.caption("Eşik değeri: 0.05  |  Sıfır-atış eşiği: 0.30")

        st.markdown("---")

        if not filtered:
            st.warning("Seçili filtrelere uyan pencere bulunamadı.")
            sel_npz = None
        else:
            npz_options = [r["npz"] for r in filtered[:200]]
            sel_npz = st.selectbox(f"Pencere ({len(npz_options)} gösteriliyor)", npz_options)
            r = det_map[sel_npz]

            st.markdown(
                f'<span style="font-size:0.8rem;color:rgba(232,236,240,0.7)">'
                f'<b style="color:#00d4ff">İstasyon:</b> {r["station"]}<br>'
                f'<b style="color:#00d4ff">Magnitüd:</b> M {r.get("magnitude", 0):.1f}'
                f'</span>',
                unsafe_allow_html=True,
            )

            st.markdown(
                '<span style="font-size:0.75rem;color:rgba(232,236,240,0.5);'
                'text-transform:uppercase;letter-spacing:1px">Sıfır-atış tespitler</span>',
                unsafe_allow_html=True,
            )
            for mk, lbl in [("stalta", "STA/LTA"), ("phasenet", "PhaseNet"),
                            ("eqtransformer", "EQT"), ("gpd", "GPD")]:
                icon   = "✅" if r[mk].get("detected") else "❌"
                p_t    = r[mk].get("p_pick_s")
                p_str  = (f'<span style="font-family:Space Mono,monospace;'
                          f'color:#00d4ff;font-size:0.72rem"> P={p_t:.1f}s</span>'
                          if p_t else "")
                st.markdown(
                    f'{icon} <span style="font-size:0.8rem">'
                    f'<b>{lbl}</b></span>{p_str}',
                    unsafe_allow_html=True,
                )

    with col_wave:
        if sel_npz is not None:
            npz_path = win_dir / sel_npz
            if not npz_path.exists():
                st.warning(f"NPZ dosyası bulunamadı: {npz_path}")
            else:
                npz  = np.load(str(npz_path), allow_pickle=True)
                data = npz["data"]   # (3, 6000)
                t    = np.arange(data.shape[1]) / SRATE

                # ── Seçili modelden seçimleri al ────────────────────
                model_color = MODEL_COLOR[sel_model]
                model_short = sel_model.split("(")[0].strip()
                p_color = "#00d4ff"    # P-pick: always cyan
                s_color = "#ff6b35"   # S-pick: always orange

                if is_ensemble:
                    with st.spinner("Ensemble (PhaseNet+GPD) çıkarımı yapılıyor..."):
                        picks = get_ensemble_picks(str(npz_path), thr_pn=0.05, thr_ens=0.15)
                    ft_icon = "✅" if picks.get("detected") else "❌"
                    ft_p    = picks.get("p_pick_s")
                    ft_cap  = (
                        f"Ensemble PhaseNet+GPD: {ft_icon} "
                        f"P={ft_p:.1f}s (olasılık={picks.get('p_prob', 0):.3f})"
                        if ft_p
                        else f"Ensemble PhaseNet+GPD: {ft_icon} Tespit yok (eşik=0.15)"
                    )
                    st.caption(ft_cap)
                    p_t    = picks.get("p_pick_s")
                    s_t    = picks.get("s_pick_s")
                    p_prob = picks.get("p_prob", 0.0)
                    s_prob = picks.get("s_prob", 0.0)

                elif is_fixed_pn:
                    with st.spinner("PhaseNet-Fixed çıkarımı yapılıyor..."):
                        picks = get_phasenet_fixed_picks(str(npz_path), threshold=0.10)
                    ft_icon = "✅" if picks.get("detected") else "❌"
                    ft_p    = picks.get("p_pick_s")
                    ft_cap  = (
                        f"PhaseNet-Fixed: {ft_icon} "
                        f"P={ft_p:.1f}s (olasılık={picks.get('p_prob', 0):.3f})"
                        if ft_p
                        else f"PhaseNet-Fixed: {ft_icon} Tespit yok (eşik=0.10)"
                    )
                    st.caption(ft_cap)
                    p_t    = picks.get("p_pick_s")
                    s_t    = picks.get("s_pick_s")
                    p_prob = picks.get("p_prob", 0.0)
                    s_prob = picks.get("s_prob", 0.0)

                elif is_finetuned:
                    with st.spinner("İnce-ayarlı model çıkarımı yapılıyor..."):
                        picks = get_finetuned_picks(str(npz_path), threshold=0.05)
                    ft_icon = "✅" if picks.get("detected") else "❌"
                    ft_p    = picks.get("p_pick_s")
                    ft_cap  = (
                        f"İnce-ayarlı PhaseNet: {ft_icon} "
                        f"P={ft_p:.1f}s (olasılık={picks.get('p_prob', 0):.3f})"
                        if ft_p
                        else f"İnce-ayarlı PhaseNet: {ft_icon} Tespit yok (eşik=0.05)"
                    )
                    st.caption(ft_cap)
                    p_t    = picks.get("p_pick_s")
                    s_t    = picks.get("s_pick_s")
                    p_prob = picks.get("p_prob", 0.0)
                    s_prob = picks.get("s_prob", 0.0)
                else:
                    det_key = MODEL_DET_KEY[sel_model]
                    p_t    = r[det_key].get("p_pick_s")
                    s_t    = r[det_key].get("s_pick_s")
                    p_prob = r[det_key].get("p_prob", 0.0)
                    s_prob = r[det_key].get("s_prob", 0.0)

                # ── Dalga formu grafiği — karanlık tema ─────────────
                CH_COLORS = ["#00d4ff", "#7b8cde", "rgba(255,107,53,0.75)"]
                channels  = ["HHZ", "HHN", "HHE"]

                fig = go.Figure()
                for ch_i, (ch, off) in enumerate(zip(channels, [2, 1, 0])):
                    fig.add_trace(go.Scatter(
                        x=t, y=data[ch_i] + off * 3,
                        name=ch,
                        line=dict(width=0.9, color=CH_COLORS[ch_i]),
                        hovertemplate=f"{ch}: %{{y:.2f}}<extra></extra>",
                    ))

                # Köken zamanı
                fig.add_vline(
                    x=ORIGIN_S,
                    line_width=1.5, line_color="rgba(255,255,255,0.55)",
                    annotation_text="Köken t₀",
                    annotation_font_color="rgba(255,255,255,0.6)",
                    annotation_font_size=9,
                    annotation_position="top left",
                )

                # Tespit penceresi
                fig.add_vrect(
                    x0=ORIGIN_S - 5, x1=ORIGIN_S + 25,
                    fillcolor="rgba(0,255,136,0.05)",
                    layer="below", line_width=0,
                    annotation_text="Tespit penceresi",
                    annotation_position="top left",
                    annotation_font_size=8,
                    annotation_font_color="rgba(0,255,136,0.5)",
                )

                # P seçimi — cyan dashed
                if p_t is not None:
                    fig.add_vline(
                        x=p_t,
                        line_width=2, line_dash="dash",
                        line_color=p_color,
                        annotation_text=f"{model_short} P ({p_prob:.2f})",
                        annotation_position="top right",
                        annotation_font_size=9,
                        annotation_font_color=p_color,
                    )

                # S seçimi — orange dotted
                if s_t is not None:
                    fig.add_vline(
                        x=s_t,
                        line_width=2, line_dash="dot",
                        line_color=s_color,
                        annotation_text=f"{model_short} S ({s_prob:.2f})",
                        annotation_position="bottom right",
                        annotation_font_size=9,
                        annotation_font_color=s_color,
                    )

                sta_code = r["station"].split(".")[-1]
                fig.update_layout(
                    height=430,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(15,21,37,0.85)",
                    title=dict(
                        text=(f"<b>KO.{sta_code}</b>  M{r.get('magnitude', 0):.1f}"
                              f"  <span style='font-size:11px;color:#8892a4'>"
                              f"{sel_npz}</span>"
                              f"  <span style='font-size:11px;color:{model_color}'>"
                              f"[{sel_model}]</span>"),
                        font=dict(color="#e8ecf0", size=13, family="DM Sans"),
                    ),
                    xaxis=dict(
                        title=dict(text="Penceredeki zaman (s)",
                                   font=dict(color="rgba(232,236,240,0.6)", size=11)),
                        tickfont=dict(color="rgba(232,236,240,0.5)",
                                      family="Space Mono", size=9),
                        gridcolor="rgba(0,212,255,0.08)",
                        zeroline=False,
                        showline=True, linecolor="rgba(0,212,255,0.15)",
                    ),
                    yaxis=dict(
                        title=dict(text="Normalleştirilmiş genlik (yığılmış)",
                                   font=dict(color="rgba(232,236,240,0.5)", size=10)),
                        showticklabels=False,
                        gridcolor="rgba(0,212,255,0.06)",
                        zeroline=False,
                        showline=True, linecolor="rgba(0,212,255,0.1)",
                    ),
                    legend=dict(
                        orientation="h", y=1.12,
                        bgcolor="rgba(15,21,37,0.7)",
                        bordercolor="rgba(0,212,255,0.15)",
                        borderwidth=1,
                        font=dict(color="#e8ecf0", size=10, family="DM Sans"),
                    ),
                    margin=dict(l=50, r=20, t=70, b=45),
                    font=dict(color="#e8ecf0", family="DM Sans"),
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption(
                    "Ön işleme: doğrusal trend kaldırma → bant geçiren 1–45 Hz → "
                    "100 Hz yeniden örnekleme → z-score normalizasyonu.  "
                    "Yeşil bant = tespit penceresi [köken−5 s, köken+25 s].  "
                    "Kesikli = P seçimi (cyan) · Noktalı = S seçimi (turuncu) · Köken t = 30 s."
                )

                # Improvement 1: Dramatic Detection Simulation
                with st.expander("🎬 Canlı Tespit Simülasyonu", expanded=False):
                    _t_a   = t
                    _N_TOT = data.shape[1]

                    # Y extent for pick line traces
                    _y_lo = float((data[0] + 0*3).min()) - 0.5
                    _y_hi = float((data[2] + 2*3).max()) + 0.5

                    # Phase offsets and colours matching waveform
                    _ch_offs = [2, 1, 0]

                    # Frame timeline
                    _F_STREAM = 40   # waveform streams in (frames 0-39)
                    _F_P      = 40   # P-pick flashes (frame 40)
                    _F_S      = 44   # S-pick appears (frame 44)
                    _F_PROB   = 50   # probability bar starts filling
                    _N_FR_SIM = 60   # total frames

                    # Build animation frames
                    _sim_frames = []
                    for _fi in range(_N_FR_SIM):
                        # ── Waveform progress ──────────────────────
                        _end = _N_TOT if _fi >= _F_STREAM else max(1, int(_fi / _F_STREAM * _N_TOT))
                        _tx  = _t_a[:_end]

                        # Traces 0,1,2: waveform channels
                        _fd = [
                            go.Scatter(x=_tx, y=(data[_ci] + _off * 3)[:_end])
                            for _ci, _off in enumerate(_ch_offs)
                        ]

                        # Trace 3: P-pick vertical line (appears at F_P)
                        _p_on = (_fi >= _F_P) and (p_t is not None)
                        _fd.append(go.Scatter(
                            x=[p_t, p_t] if _p_on else [],
                            y=[_y_lo, _y_hi] if _p_on else [],
                        ))

                        # Trace 4: P-pick label text
                        _fd.append(go.Scatter(
                            x=[p_t] if _p_on else [],
                            y=[_y_hi * 0.85] if _p_on else [],
                            text=["P Dalgası Tespit!"] if _p_on else [],
                            mode="text",
                            textfont=dict(color="#00d4ff", size=11, family="Space Mono"),
                        ))

                        # Trace 5: S-pick vertical line (appears at F_S)
                        _s_on = (_fi >= _F_S) and (s_t is not None)
                        _fd.append(go.Scatter(
                            x=[s_t, s_t] if _s_on else [],
                            y=[_y_lo, _y_hi] if _s_on else [],
                        ))

                        # Trace 6: S-pick label text
                        _fd.append(go.Scatter(
                            x=[s_t] if _s_on else [],
                            y=[_y_lo * 0.85] if _s_on else [],
                            text=["S Dalgası Tespit!"] if _s_on else [],
                            mode="text",
                            textfont=dict(color="#ff6b35", size=11, family="Space Mono"),
                        ))

                        # Trace 7: probability gauge bar (bottom area)
                        if _fi < _F_P:
                            _prob_now = 0.0
                        elif _fi < _F_PROB:
                            _prob_now = p_prob * 0.4
                        else:
                            _prob_now = p_prob * min(1.0, (_fi - _F_PROB + 1) / (_N_FR_SIM - _F_PROB))
                        _pc = ("#ef5350" if _prob_now < 0.35
                               else "#ff9800" if _prob_now < 0.65
                               else "#00c853")
                        _prob_label = f"Tespit Olasılığı: %{_prob_now*100:.0f}"
                        _fd.append(go.Bar(
                            x=[_prob_now], y=[_prob_label],
                            orientation="h",
                            marker_color=_pc, marker_opacity=0.85,
                            width=0.6,
                        ))

                        # ── Layout flash ───────────────────────────
                        if _fi == _F_P:
                            _bg = "rgba(0,212,255,0.16)"   # cyan flash — P detected
                        elif _fi == _F_S:
                            _bg = "rgba(255,107,53,0.14)"  # orange flash — S detected
                        elif _fi == _N_FR_SIM - 1 and p_prob > 0.5:
                            _bg = "rgba(0,200,83,0.14)"    # green flash — confirmed
                        else:
                            _bg = "rgba(15,21,37,0.85)"

                        _sim_frames.append(go.Frame(
                            data=_fd,
                            layout=go.Layout(plot_bgcolor=_bg),
                            name=str(_fi),
                        ))

                    # ── Initial (empty) traces ─────────────────────
                    _init_data = [
                        go.Scatter(x=[], y=[], name=_ch,
                                   line=dict(width=0.9, color=_c))
                        for _ch, _c in zip(channels, CH_COLORS)
                    ] + [
                        go.Scatter(x=[], y=[], mode="lines", name="P Dalgası",
                                   line=dict(color="#00d4ff", width=4, dash="dash")),  # trace 3
                        go.Scatter(x=[], y=[], mode="text"),                           # trace 4
                        go.Scatter(x=[], y=[], mode="lines", name="S Dalgası",
                                   line=dict(color="#ff6b35", width=3, dash="dot")),   # trace 5
                        go.Scatter(x=[], y=[], mode="text"),                           # trace 6
                        go.Bar(x=[0], y=["Tespit Olasılığı: %0"], orientation="h",
                               marker_color="#555", width=0.6, showlegend=False),       # trace 7
                    ]

                    _anim_fig = go.Figure(data=_init_data, frames=_sim_frames)

                    # Static layout
                    _anim_fig.update_layout(
                        height=440,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(15,21,37,0.85)",
                        font=dict(color="#e8ecf0", family="DM Sans"),
                        title=dict(
                            text=("🎬 Canlı Tespit Simülasyonu — "
                                  f"KO.{r['station'].split('.')[-1]}  "
                                  f"M{r.get('magnitude',0):.1f}  [{model_short}]"),
                            font=dict(color="#e8ecf0", size=11),
                        ),
                        xaxis=dict(
                            domain=[0, 1],
                            range=[0, float(_t_a[-1])],
                            title=dict(text="Penceredeki zaman (s)",
                                       font=dict(color="rgba(232,236,240,0.6)", size=10)),
                            gridcolor="rgba(0,212,255,0.08)", zeroline=False,
                            tickfont=dict(color="rgba(232,236,240,0.5)", size=9),
                        ),
                        yaxis=dict(
                            domain=[0.22, 1.0],
                            showticklabels=False,
                            gridcolor="rgba(0,212,255,0.06)", zeroline=False,
                            range=[_y_lo, _y_hi],
                        ),
                        # Probability gauge axis
                        xaxis2=dict(
                            domain=[0, 1], anchor="y2",
                            range=[0, 1], showticklabels=False, zeroline=False,
                            gridcolor="rgba(0,212,255,0.05)",
                        ),
                        yaxis2=dict(
                            domain=[0, 0.17], anchor="x2",
                            tickfont=dict(color="rgba(232,236,240,0.6)", size=9),
                            gridcolor="rgba(0,212,255,0.05)",
                        ),
                        barmode="overlay",
                        legend=dict(
                            orientation="h", y=1.08,
                            bgcolor="rgba(15,21,37,0.7)",
                            font=dict(color="#e8ecf0", size=9),
                        ),
                        margin=dict(l=50, r=20, t=60, b=90),
                        updatemenus=[dict(
                            type="buttons", showactive=False,
                            y=1.18, x=0.5, xanchor="center",
                            buttons=[
                                dict(label="▶ Simülasyonu Başlat", method="animate",
                                     args=[None,
                                           dict(frame=dict(duration=90, redraw=True),
                                                fromcurrent=True, mode="immediate")]),
                                dict(label="⏸ Duraklat", method="animate",
                                     args=[[None],
                                           dict(frame=dict(duration=0, redraw=False),
                                                mode="immediate")]),
                            ],
                        )],
                        sliders=[dict(
                            active=0,
                            steps=[
                                dict(method="animate",
                                     args=[[str(_fi)],
                                           dict(frame=dict(duration=0, redraw=True),
                                                mode="immediate")],
                                     label=(
                                         "P!" if _fi == _F_P else
                                         "S!" if _fi == _F_S else
                                         f"{int(_fi/_N_FR_SIM*_N_TOT/SRATE)}s"
                                     ))
                                for _fi in range(_N_FR_SIM)
                            ],
                            y=-0.15, len=1.0,
                            currentvalue=dict(prefix="Kare: ", visible=True,
                                              font=dict(size=9, color="#e8ecf0")),
                            transition=dict(duration=0),
                            font=dict(color="#e8ecf0", size=8),
                        )],
                    )

                    # Assign Bar trace to secondary axes (yaxis2/xaxis2)
                    _anim_fig.data[7].update(yaxis="y2", xaxis="x2")

                    # Static shapes: origin line + detection window
                    _anim_fig.add_vline(
                        x=ORIGIN_S, line_width=1.5,
                        line_color="rgba(255,255,255,0.4)",
                        annotation_text="t₀",
                        annotation_font_color="rgba(255,255,255,0.45)",
                        annotation_font_size=8,
                    )
                    _anim_fig.add_vrect(
                        x0=ORIGIN_S - 5, x1=ORIGIN_S + 25,
                        fillcolor="rgba(0,255,136,0.04)", layer="below", line_width=0,
                    )

                    st.plotly_chart(_anim_fig, use_container_width=True)

                    # ── Detection result badge ─────────────────────
                    _detected_now = (p_t is not None)
                    if _detected_now:
                        st.success(
                            f"✓ DEPREM TESPİT EDİLDİ — {model_short}  |  "
                            f"P={p_t:.2f}s  |  Olasılık={p_prob:.1%}  |  Eşik=0.15"
                        )
                    else:
                        st.info("✗ GÜRÜLTÜ — Bu pencerede anlamlı sismik sinyal tespit edilmedi.")

                    # ── Probability gauge (static, post-animation) ─
                    _prob_label_str = f"Tespit Olasılığı: **{p_prob:.1%}**"
                    st.markdown(_prob_label_str)
                    st.progress(min(1.0, float(p_prob)),
                                text=("🟢 Güçlü tespit" if p_prob > 0.65
                                      else "🟡 Orta sinyal" if p_prob > 0.35
                                      else "🔴 Zayıf sinyal"))
                    st.caption(
                        "Simülasyon akışı: 0-40 kare waveform akışı → "
                        "Kare 40 P-dalgası tespiti (cyan flash) → "
                        "Kare 44 S-dalgası tespiti (turuncu flash) → "
                        "Kare 50+ Olasılık göstergesi dolar."
                    )

# ══════════════════════════════════════════════════════════════════
# SEKME 3 — TBDY-2018 Bölgesel Risk Danışmanı
# ══════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Bölgesel Sismik Risk Değerlendirmesi — TBDY-2018 Uyumlu")
    st.caption(
        "Kural tabanlı değerlendirme sistemi. Tüm öneriler TBDY-2018 madde referanslarına dayanır. "
        "LLM kullanılmaz — deterministik ve doğrulanabilir çıktı."
    )

    # Import risk module
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "scripts"))
    try:
        from importlib import import_module as _im
        _risk = _im("37_regional_risk_assessment")
        _risk_ok = True
    except Exception as _e:
        _risk_ok = False
        st.error(f"Risk modülü yüklenemedi: {_e}")

    # Enhancement B: Turkey DTS Map
    with st.expander("🗺 Türkiye Batı Bölgesi — Deprem Tasarım Sınıfı (DTS) Haritası", expanded=False):
        _city_dts = {
            "Şehir":    ["İstanbul (Avrupa)", "İstanbul (Anadolu)", "Tekirdağ", "Kocaeli",
                         "Sakarya", "Bursa", "Balıkesir", "Çanakkale",
                         "İzmir", "Ankara", "Kahramanmaraş"],
            "lat":      [41.01, 40.99, 40.98, 40.77, 40.69, 40.18, 39.65, 40.14,
                         38.42, 39.92, 37.58],
            "lon":      [28.95, 29.03, 27.52, 29.92, 30.41, 29.06, 27.88, 26.41,
                         27.13, 32.85, 36.94],
            "DTS":      ["DTS-1", "DTS-1", "DTS-1", "DTS-1",
                         "DTS-1", "DTS-1", "DTS-1", "DTS-2",
                         "DTS-2", "DTS-3", "DTS-1"],
            "Ss (g)":   [1.20, 1.20, 0.90, 1.10,
                         1.05, 0.95, 0.75, 0.60,
                         0.85, 0.55, 1.40],
        }
        _df_dts = pd.DataFrame(_city_dts)
        _dts_cmap = {"DTS-1": "#ef5350", "DTS-2": "#ff9800", "DTS-3": "#ffd600", "DTS-4": "#00c853"}
        _dts_cfig = px.scatter_mapbox(
            _df_dts, lat="lat", lon="lon",
            color="DTS", hover_name="Şehir",
            hover_data={"Ss (g)": True, "lat": False, "lon": False},
            color_discrete_map=_dts_cmap,
            size_max=20, zoom=5,
            center={"lat": 40.2, "lon": 29.5},
            mapbox_style="carto-darkmatter",
        )
        _dts_cfig.update_traces(marker=dict(size=16, opacity=0.88))
        _dts_cfig.update_layout(
            height=400, margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(bgcolor="rgba(15,21,37,0.8)", font=dict(color="#e8ecf0", size=10)),
            font=dict(color="#e8ecf0", family="DM Sans"),
        )
        st.plotly_chart(_dts_cfig, use_container_width=True)
        _dts_cols = st.columns(4)
        for _di, (_dk, _dv) in enumerate(
            [("DTS-1", "≥0.75g — Çok yüksek"), ("DTS-2", "0.50–0.75g — Yüksek"),
             ("DTS-3", "0.25–0.50g — Orta"), ("DTS-4", "<0.25g — Düşük")]
        ):
            with _dts_cols[_di]:
                st.markdown(
                    f'<div style="background:{_dts_cmap[_dk]}22;border:1px solid {_dts_cmap[_dk]}55;'
                    f'border-radius:6px;padding:6px 8px;text-align:center">'
                    f'<span style="color:{_dts_cmap[_dk]};font-weight:700;font-size:0.85rem">{_dk}</span><br>'
                    f'<span style="font-size:0.7rem;color:rgba(232,236,240,0.6)">{_dv}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        st.caption("Kaynak: TBDY-2018 Bölüm 3, Tablo 3.2 · AFAD TDTH bölgesel SDS değerleri")

    if _risk_ok:
        col_inp, col_res = st.columns([1, 2])

        with col_inp:
            st.markdown("#### Koordinat Girin")
            _lat = st.number_input("Enlem (°N)", value=41.01, min_value=36.0, max_value=42.5,
                                   step=0.01, format="%.4f")
            _lon = st.number_input("Boylam (°E)", value=28.97, min_value=25.0, max_value=45.0,
                                   step=0.01, format="%.4f")

            st.markdown("#### Yapı Bilgileri")
            _btype_label = st.selectbox(
                "Yapı Türü",
                ["Konut", "Ofis", "Ticari", "Sanayi", "Okul", "Hastane", "Kamu"],
            )
            _btype = _btype_label.lower()

            _floors = st.number_input("Kat Sayısı", value=5, min_value=1, max_value=60, step=1)

            _material = st.selectbox("Yapı Malzemesi", ["Betonarme", "Çelik", "Yığma"])

            _vs30_inp = st.number_input(
                "Vs30 (m/s) — opsiyonel (boş bırakın → bölgesel tahmin)",
                value=0, min_value=0, max_value=3000, step=10,
            )
            _vs30 = float(_vs30_inp) if _vs30_inp > 0 else None

            _analyze = st.button("Analiz Et", type="primary", use_container_width=True)

        with col_res:
            if _analyze or True:   # show default on load
                with st.spinner("TBDY-2018 analizi yapılıyor..."):
                    _report = _risk.generate_risk_report(
                        lat=_lat, lon=_lon,
                        building_type=_btype,
                        floors=int(_floors),
                        material=_material.lower(),
                        vs30=_vs30,
                    )

                _tbdy  = _report["tbdy2018"]
                _soil  = _report["soil"]
                _seis  = _report["seismicity"]
                _summ  = _report["summary"]

                # ── Risk renk bandı ──────────────────────────────────
                _color_map = {"kırmızı": "#ef5350", "turuncu": "#ff9800",
                              "sarı": "#ffd600", "yeşil": "#00c853"}
                _rc = _color_map.get(_summ["risk_color"], "#888")
                _verdict = _summ["verdict"]

                st.markdown(
                    f'<div style="background:{_rc}22;border:2px solid {_rc};border-radius:10px;'
                    f'padding:14px 18px;margin-bottom:14px">'
                    f'<span style="font-size:1.3rem;font-weight:700;color:{_rc}">'
                    f'● {_verdict}</span></div>',
                    unsafe_allow_html=True,
                )

                # ── Zemin ve DTS kartları ────────────────────────────
                _c1, _c2, _c3 = st.columns(3)
                with _c1:
                    _vs30_display = _soil.get("Vs30_mps", None)
                    _vs30_label   = f" (Vs30 ≈ {_vs30_display} m/s)" if _vs30_display else ""
                    st.metric("Zemin Sınıfı", f"{_soil['soil_class']}{_vs30_label}",
                              help="TBDY-2018 Bölüm 16, Tablo 16.1")
                    _src_badge = "🟢 TBDY-2018 ızgara" if "IDW" in _soil.get("source", "") else "🟡 Bölgesel tahmin"
                    st.caption(f"{_src_badge} · {_soil.get('source', '')}")
                with _c2:
                    st.metric("Deprem Tasarım Sınıfı", _tbdy["dts_label"],
                              help="TBDY-2018 Bölüm 3, Tablo 3.2")
                    st.caption(f"SDS = {_soil['SDS']:.2f}g")
                with _c3:
                    st.metric("Bina Kullanım Sınıfı", _tbdy["bks_label"].split("—")[0].strip(),
                              help="TBDY-2018 Bölüm 3, Tablo 3.1")
                    st.caption(_tbdy["performans_hedefi"])

                # ── Yerel sismik aktivite ────────────────────────────
                st.markdown("#### Yerel Sismik Aktivite (50 km yarıçap)")
                _sc1, _sc2, _sc3 = st.columns(3)
                with _sc1:
                    st.metric("Tespit Edilen Olay", _seis["n_events_nearby"])
                with _sc2:
                    st.metric("Maksimum Büyüklük",
                              f"M {_seis['max_magnitude']}" if _seis["max_magnitude"] else "—")
                with _sc3:
                    st.metric("Aktivite Seviyesi", _seis["activity_level"])
                st.caption(f"Kaynak: {_seis['source']}")

                # ── Uyarılar ─────────────────────────────────────────
                if _tbdy["warnings"]:
                    st.markdown("#### ⚠ Uyarılar")
                    for _w in _tbdy["warnings"]:
                        st.warning(_w)

                # ── TBDY-2018 Hükümleri ──────────────────────────────
                st.markdown("#### TBDY-2018 Hükümleri")
                for _p in _tbdy["provisions"]:
                    with st.expander(f"📋 {_p['madde']}", expanded=True):
                        st.markdown(_p["metin"])

                # ── Spektral değerler ────────────────────────────────
                with st.expander("📊 Spektral İvme Değerleri (AFAD TDTH)"):
                    _sv1, _sv2 = st.columns(2)
                    with _sv1:
                        st.metric("Ss (kısa periyot)", f"{_soil['Ss']:.3f}g")
                        st.metric("SDS (tasarım)", f"{_soil['SDS']:.3f}g")
                    with _sv2:
                        st.metric("S1 (1.0s periyot)", f"{_soil['S1']:.3f}g")
                        st.metric("SD1 (tasarım)", f"{_soil['SD1']:.3f}g")
                    st.caption(f"Kaynak: {_soil['source']}")

                st.markdown("---")
                st.caption(
                    "Bu analiz TBDY-2018 (Türkiye Bina Deprem Yönetmeliği, Resmi Gazete 18 Mart 2018) "
                    "hükümlerine dayanan kural tabanlı bir ön değerlendirmedir. "
                    "Resmi yapı ruhsatı için lisanslı inşaat mühendisi raporu zorunludur."
                )

# ══════════════════════════════════════════════════════════════════
# SEKME 4 — Sismik Motor (Üretim Hattı)
# ══════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Sismik Motor — Hibrit Trace-Level Sınıflandırıcı")
    st.caption(
        "GPD gömülü → 805-boyutlu özellik → PCA(100) → SVM RBF(C=20) → Olay/Gürültü kararı  ·  "
        "Eşik: 0.440 (oracle @ %90 TNR)  ·  Phase 3.9"
    )

    import h5py as _h5py
    from datetime import datetime as _dt

    @st.cache_resource
    def load_trace_classifier():
        from seismic_engine.models.gpd_embedder import GPDEmbedder
        from seismic_engine.inference.trace_classifier import TraceClassifier
        from seismic_engine.config import GPDConfig
        _model_path = ROOT / "models" / "trace_classifier.joblib"
        if not _model_path.exists():
            return None
        embedder = GPDEmbedder(device="cpu")
        clf = TraceClassifier(embedder, cfg=GPDConfig(), svm_C=20.0, n_pca=100)
        clf.load(_model_path)
        return clf

    @st.cache_data
    def load_hdf5_trace_list():
        meta = pd.read_csv(ROOT / "data" / "augmented_dataset" / "metadata.csv")
        orig = meta[meta["augmentation"] == "original"].copy()
        orig["label"] = orig["window_type"].map({"event": "event", "noise": "noise"}).fillna("noise")
        return orig

    _trace_meta = load_hdf5_trace_list()
    _events_meta = _trace_meta[_trace_meta["window_type"] == "event"].sort_values(
        "source_magnitude", ascending=False
    )
    _noise_meta = _trace_meta[_trace_meta["window_type"] == "noise"]

    _eng_col_ctrl, _eng_col_main = st.columns([1, 3])

    with _eng_col_ctrl:
        st.markdown("**Veri Kaynağı**")
        _eng_source = st.radio(
            "Kaynak seçin",
            ["Test Seti (HDF5)", "MiniSEED Yükle"],
            horizontal=True,
            label_visibility="collapsed",
        )

        if _eng_source == "Test Seti (HDF5)":
            _eng_wtype = st.selectbox("Pencere türü", ["Olay (event)", "Gürültü (noise)"])
            if "Olay" in _eng_wtype:
                _eng_candidates = _events_meta
                _eng_labels = [
                    f"M{row.source_magnitude:.1f} | {row.station} | {row.trace_name[:50]}"
                    for _, row in _eng_candidates.iterrows()
                ]
            else:
                _eng_candidates = _noise_meta.head(100)
                _eng_labels = [
                    f"{row.station} | {row.trace_name[:55]}"
                    for _, row in _eng_candidates.iterrows()
                ]
            _eng_sel_idx = st.selectbox(
                "Pencere seçin",
                range(len(_eng_labels)),
                format_func=lambda i: _eng_labels[i],
            )
            _eng_sel_row = _eng_candidates.iloc[_eng_sel_idx]
            _eng_trace_name = _eng_sel_row["trace_name"]

            st.markdown("---")
            st.markdown(
                f'<div class="metric-card">'
                f'<b style="color:#00d4ff">Trace:</b> {_eng_trace_name[:40]}<br>'
                f'<b style="color:#00d4ff">İstasyon:</b> {_eng_sel_row["station"]}<br>'
                f'<b style="color:#00d4ff">Magnitüd:</b> M{_eng_sel_row.get("source_magnitude", 0):.1f}<br>'
                f'<b style="color:#00d4ff">Tür:</b> {_eng_sel_row["window_type"]}'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            _eng_uploaded = st.file_uploader(
                "MiniSEED dosyası yükleyin (.mseed)",
                type=["mseed"],
                help="3-bileşenli (HHZ/HHN/HHE), 100 Hz, 60 saniyelik pencere",
            )

        _eng_run = st.button("⚡ Çıkarım Yap", type="primary", use_container_width=True)

    with _eng_col_main:
        if _eng_run:
            _eng_data = None
            _eng_info = {}

            if _eng_source == "Test Seti (HDF5)":
                with st.spinner("HDF5'ten trace yükleniyor..."):
                    hdf5_path = ROOT / "data" / "augmented_dataset" / "waveforms.hdf5"
                    with _h5py.File(str(hdf5_path), "r") as hf:
                        _eng_data = hf["data"][_eng_trace_name][:]
                    _eng_info = {
                        "station": _eng_sel_row["station"],
                        "magnitude": _eng_sel_row.get("source_magnitude", None),
                        "window_type": _eng_sel_row["window_type"],
                        "trace_name": _eng_trace_name,
                    }
            else:
                if _eng_uploaded is not None:
                    with st.spinner("MiniSEED okunuyor..."):
                        from obspy import read as _obspy_read
                        try:
                            _mseed_st = _obspy_read(_eng_uploaded)
                        except Exception as _mseed_err:
                            st.error(f"MiniSEED dosyası okunamadı: {_mseed_err}")
                            st.stop()
                        if len(_mseed_st) < 3:
                            st.error("MiniSEED dosyası 3 bileşen (Z/N/E) içermelidir.")
                            st.stop()
                        _orig_sr = _mseed_st[0].stats.sampling_rate
                        _mseed_st.detrend("demean").filter("bandpass", freqmin=1.0, freqmax=45.0)
                        if _orig_sr != 100.0:
                            st.warning(f"Örnekleme hızı {_orig_sr} Hz → 100 Hz'e yeniden örneklendi.")
                        _mseed_st.resample(100.0)
                        _n_target = 6000
                        channels = []
                        for tr in sorted(_mseed_st, key=lambda t: t.stats.channel):
                            d = tr.data[:_n_target].astype(np.float32)
                            if len(d) < _n_target:
                                d = np.pad(d, (0, _n_target - len(d)))
                            channels.append(d)
                        _eng_data = np.array(channels[:3])
                        _eng_info = {
                            "station": f"{_mseed_st[0].stats.network}.{_mseed_st[0].stats.station}",
                            "magnitude": None,
                            "window_type": "upload",
                            "trace_name": _eng_uploaded.name,
                        }
                else:
                    st.warning("Lütfen bir MiniSEED dosyası yükleyin.")

            if _eng_data is not None:
                with st.spinner("Hibrit sınıflandırıcı çalıştırılıyor (GPD → SVM)..."):
                    _clf = load_trace_classifier()
                    if _clf is None:
                        st.error("Model dosyası bulunamadı: `models/trace_classifier.joblib`")
                        st.stop()
                    _eng_result = _clf.predict(_eng_data, station=_eng_info.get("station"))

                _is_event = _eng_result["is_event"]
                _event_prob = _eng_result["trace_event_prob"]
                _p_prob = _eng_result["p_prob"]
                _gpd_max = _eng_result["gpd_max_prob"]
                _n_win = _eng_result["n_total_windows"]

                st.session_state["engine_result"] = _eng_result
                st.session_state["engine_info"] = _eng_info
                st.session_state["engine_data"] = _eng_data

                _verdict_color = "#00c853" if _is_event else "#ef5350"
                _verdict_text = "OLAY TESPİT EDİLDİ" if _is_event else "GÜRÜLTÜ"
                _verdict_icon = "✅" if _is_event else "❌"

                st.markdown(
                    f'<div style="background:{_verdict_color}22;border:2px solid {_verdict_color};'
                    f'border-radius:10px;padding:14px 18px;margin-bottom:14px">'
                    f'<span style="font-size:1.3rem;font-weight:700;color:{_verdict_color}">'
                    f'{_verdict_icon} {_verdict_text}</span>'
                    f'<span style="margin-left:24px;font-size:0.9rem;color:rgba(232,236,240,0.7)">'
                    f'SVM Olasılık: {_event_prob:.3f}  |  Eşik: 0.440  |  '
                    f'GPD Maks: {_gpd_max:.3f}  |  Pencere: {_n_win}'
                    f'</span></div>',
                    unsafe_allow_html=True,
                )

                _mc1, _mc2, _mc3, _mc4 = st.columns(4)
                with _mc1:
                    st.metric("SVM Olay Olasılığı", f"{_event_prob:.3f}")
                with _mc2:
                    st.metric("GPD Maks P-Prob", f"{_gpd_max:.3f}")
                with _mc3:
                    st.metric("Pencere Sayısı", f"{_n_win}")
                with _mc4:
                    _p_peak_idx = int(np.argmax(_p_prob)) if _is_event else 0
                    _p_peak_s = _p_peak_idx / 100.0
                    st.metric("P-Pick (s)", f"{_p_peak_s:.2f}" if _is_event else "—")

                _t_eng = np.arange(_eng_data.shape[1]) / 100.0
                _fig_eng = go.Figure()
                for ch_i, (ch, off) in enumerate(zip(["HHZ", "HHN", "HHE"], [2, 1, 0])):
                    _fig_eng.add_trace(go.Scatter(
                        x=_t_eng, y=_eng_data[ch_i] + off * 3,
                        name=ch,
                        line=dict(width=0.9, color=["#00d4ff", "#7b8cde", "rgba(255,107,53,0.75)"][ch_i]),
                    ))

                if _is_event:
                    _p_prob_norm = _p_prob / max(_p_prob.max(), 1e-9) * 2 - 1
                    _fig_eng.add_trace(go.Scatter(
                        x=_t_eng, y=_p_prob_norm - 3,
                        name="P-Probability",
                        line=dict(width=1.5, color="#00ff88"),
                        fill="tozeroy",
                        fillcolor="rgba(0,255,136,0.1)",
                    ))

                    _p_pick_t = _p_peak_s
                    _fig_eng.add_vline(
                        x=_p_pick_t, line_width=2, line_dash="dash",
                        line_color="#00d4ff",
                        annotation_text=f"P-Pick {_p_pick_t:.2f}s ({_p_prob.max():.2f})",
                        annotation_position="top right",
                        annotation_font_size=10,
                        annotation_font_color="#00d4ff",
                    )

                _title_parts = [f"<b>{_eng_info.get('station','')}</b>"]
                if _eng_info.get("magnitude"):
                    _title_parts.append(f"M{_eng_info['magnitude']:.1f}")
                _title_parts.append(
                    f"<span style='color:{_verdict_color}'>[{_verdict_text}]</span>"
                )

                _fig_eng.update_layout(
                    height=430,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(15,21,37,0.85)",
                    title=dict(
                        text="  ".join(_title_parts),
                        font=dict(color="#e8ecf0", size=13, family="DM Sans"),
                    ),
                    xaxis=dict(
                        title="Zaman (s)",
                        gridcolor="rgba(0,212,255,0.08)", zeroline=False,
                        tickfont=dict(color="rgba(232,236,240,0.5)", family="Space Mono", size=9),
                    ),
                    yaxis=dict(
                        showticklabels=False,
                        gridcolor="rgba(0,212,255,0.06)", zeroline=False,
                    ),
                    legend=dict(
                        orientation="h", y=1.12,
                        bgcolor="rgba(15,21,37,0.7)",
                        font=dict(color="#e8ecf0", size=10),
                    ),
                    margin=dict(l=50, r=20, t=70, b=45),
                    font=dict(color="#e8ecf0", family="DM Sans"),
                )
                st.plotly_chart(_fig_eng, use_container_width=True)

                st.progress(
                    min(1.0, float(_event_prob)),
                    text=(
                        f"SVM Olay Olasılığı: {_event_prob:.1%} — "
                        + ("Güçlü tespit" if _event_prob > 0.65
                           else "Orta sinyal" if _event_prob > 0.35
                           else "Zayıf / Gürültü")
                    ),
                )
                st.caption(
                    "Hibrit Sınıflandırıcı: GPD gömülü (bn5 katmanı, 200-dim) → "
                    "stride=200 kayan pencere → 805-dim trace özellik vektörü → "
                    "StandardScaler → PCA(100) → SVM RBF(C=20) → Platt ölçekleme → "
                    "Olay/Gürültü kararı (eşik=0.440, oracle @ %90 TNR)"
                )

        elif st.session_state.get("engine_result"):
            st.info("Son çıkarım sonucu mevcut. Yeni çıkarım için sol paneldeki butona tıklayın.")
        else:
            st.info("👈 Sol panelden bir trace seçin veya MiniSEED yükleyin, ardından **⚡ Çıkarım Yap** butonuna tıklayın.")

# ══════════════════════════════════════════════════════════════════
# SEKME 5 — Yapısal Risk Raporu (Yerel LLM Entegrasyonu)
# ══════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Yapısal Risk Raporu — Yerel DeepSeek LLM + TBDY-2018 RAG")

    from risk_advisor.llm_client import is_ollama_reachable as _is_ollama, OLLAMA_MODEL as _OLLAMA_MODEL
    _api_ok = _is_ollama()
    if _api_ok:
        _mode_label = f"Yerel DeepSeek Motor Bağlı ({_OLLAMA_MODEL})"
        _mode_color = "#00c853"
        _mode_icon = "🟢"
    else:
        _mode_label = "Ollama Sunucusu Çevrimdışı"
        _mode_color = "#ef5350"
        _mode_icon = "🔴"
    st.markdown(
        f'<div style="display:inline-block;background:{_mode_color}22;'
        f'border:1px solid {_mode_color}66;border-radius:6px;padding:4px 12px;'
        f'font-size:0.78rem;color:{_mode_color};font-weight:600;margin-bottom:12px">'
        f'{_mode_icon} {_mode_label}</div>',
        unsafe_allow_html=True,
    )
    if not _api_ok:
        st.warning(
            "Ollama sunucusu erişilemez. Terminalde `ollama serve` komutunu çalıştırın, "
            f"ardından `ollama pull {_OLLAMA_MODEL}` ile modeli indirin. "
            "Şu an mock mod aktif — deterministik şablon rapor üretilecektir."
        )

    _rpt_col_ctrl, _rpt_col_main = st.columns([1, 3])

    with _rpt_col_ctrl:
        st.markdown("**Rapor Tetikleme**")

        _has_detection = (
            st.session_state.get("engine_result", {}).get("is_event", False)
        )
        if _has_detection:
            _det_info = st.session_state.get("engine_info", {})
            st.success(
                f"Tespit: {_det_info.get('station', '?')} — "
                f"M{_det_info.get('magnitude', '?')}"
            )
        else:
            st.info("Sekme 5'te olay tespit edilirse otomatik tetiklenir.")

        _rpt_language = st.radio("Rapor dili", ["Türkçe (tr)", "English (en)"], horizontal=True)
        _rpt_lang_code = "tr" if "tr" in _rpt_language else "en"

        st.markdown("---")
        _rpt_generate = st.button(
            "📋 Risk Raporu Oluştur",
            type="primary",
            use_container_width=True,
            disabled=not _has_detection,
        )

        if not _has_detection:
            st.caption("Rapor oluşturmak için önce Sekme 5'te çıkarım yapın.")

    with _rpt_col_main:
        if _rpt_generate and _has_detection:
            _det_result = st.session_state["engine_result"]
            _det_info = st.session_state["engine_info"]

            from risk_advisor.schema import (
                EnginePayload, DetectedEvent, DetectionTierResult,
            )
            from risk_advisor.report_generator import ReportGenerator

            _p_peak_idx = int(np.argmax(_det_result["p_prob"]))
            _p_peak_s = _p_peak_idx / 100.0

            _payload = EnginePayload(
                timestamp=_dt.utcnow(),
                pipeline_version="1.0.0-phase4",
                region="marmara",
                detected_events=[
                    DetectedEvent(
                        trace_id=_det_info.get("trace_name", "unknown"),
                        magnitude=_det_info.get("magnitude"),
                        model_name="hybrid_gpd_svm",
                        p_probability_peak=float(_det_result["gpd_max_prob"]),
                        p_pick_seconds=_p_peak_s,
                        tier_results=[
                            DetectionTierResult(
                                tier="trace_classification",
                                detected=True,
                                confidence=float(_det_result["trace_event_prob"]),
                            ),
                            DetectionTierResult(
                                tier="phase_picking",
                                detected=True,
                                mae_seconds=None,
                                confidence=float(_det_result["p_prob"].max()),
                                pick_sample=_p_peak_idx,
                            ),
                        ],
                    )
                ],
                scorecard={
                    "svm_event_prob": float(_det_result["trace_event_prob"]),
                    "gpd_max_p_prob": float(_det_result["gpd_max_prob"]),
                    "p_pick_seconds": _p_peak_s,
                    "n_windows": _det_result["n_total_windows"],
                    "detection_threshold": 0.440,
                    "model": "Phase 3.9 — event-station focused SVM RBF(C=20)",
                },
            )

            @st.cache_resource
            def get_report_generator():
                return ReportGenerator()

            _rgen = get_report_generator()

            st.markdown("### Risk Raporu")
            st.markdown("---")

            _report_container = st.empty()
            _full_report_text = ""

            with st.spinner(f"DeepSeek ({_OLLAMA_MODEL}) rapor üretiyor..." if _api_ok else "Mock rapor üretiliyor..."):
                _stream_chunks = _rgen.stream_report(_payload, language=_rpt_lang_code)
                _full_report_text = ""
                for chunk in _stream_chunks:
                    _full_report_text += chunk
                    _report_container.markdown(_full_report_text)

            st.markdown("---")

            _dl_col1, _dl_col2 = st.columns(2)
            with _dl_col1:
                _report_md = (
                    f"# Sismik Risk Raporu\n\n"
                    f"**Tarih:** {_dt.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n"
                    f"**İstasyon:** {_det_info.get('station', '?')}\n"
                    f"**Magnitüd:** M{_det_info.get('magnitude', '?')}\n"
                    f"**SVM Olasılık:** {_det_result['trace_event_prob']:.3f}\n"
                    f"**P-Pick:** {_p_peak_s:.2f}s\n\n"
                    f"---\n\n{_full_report_text}\n\n---\n\n"
                    f"*Rapor: {'Yerel Ollama (' + _OLLAMA_MODEL + ')' if _api_ok else 'Mock LLM'} | "
                    f"Pipeline: Phase 3.9 Hybrid Classifier*\n"
                )
                st.download_button(
                    "📥 Raporu İndir (Markdown)",
                    data=_report_md,
                    file_name=f"risk_report_{_dt.utcnow().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
            with _dl_col2:
                _report_json = json.dumps({
                    "timestamp": _dt.utcnow().isoformat(),
                    "station": _det_info.get("station"),
                    "magnitude": _det_info.get("magnitude"),
                    "svm_event_prob": _det_result["trace_event_prob"],
                    "p_pick_seconds": _p_peak_s,
                    "llm_mode": f"ollama/{_OLLAMA_MODEL}" if _api_ok else "mock",
                    "report_text": _full_report_text,
                }, indent=2, ensure_ascii=False)
                st.download_button(
                    "📥 Raporu İndir (JSON)",
                    data=_report_json,
                    file_name=f"risk_report_{_dt.utcnow().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True,
                )

            st.caption(
                f"LLM: {'Yerel Ollama ' + _OLLAMA_MODEL + ' (hava boşluklu)' if _api_ok else 'Deterministik mock şablon'}  ·  "
                "RAG: FAISS + sentence-transformers all-MiniLM-L6-v2  ·  "
                "Pipeline: GPD → SVM RBF → EnginePayload → PromptBuilder → LLM → RiskReport"
            )

        elif st.session_state.get("engine_result", {}).get("is_event"):
            st.info("👈 **📋 Risk Raporu Oluştur** butonuna tıklayarak LLM raporunu başlatın.")
        else:
            st.info(
                "Bu sekmede otomatik risk raporu oluşturulur:\n\n"
                "1. **Sekme 5** (Sismik Motor) sekmesinde bir trace seçin ve çıkarım yapın\n"
                "2. Olay tespit edilirse bu sekmeye dönün\n"
                "3. **📋 Risk Raporu Oluştur** butonuna tıklayın\n\n"
                "Rapor, tespit sonuçlarını TBDY-2018 RAG verileriyle birleştirerek "
                "yerel DeepSeek LLM (Ollama) tarafından üretilir."
            )

# ══════════════════════════════════════════════════════════════════
# SEKME 6 — Canlı İzleme (Demo Modu)
# ══════════════════════════════════════════════════════════════════
with tab6:
    st.subheader("Gerçek Zamanlı Sismik İzleme — Demo Modu")
    st.caption(
        "Kaydedilmiş dalga formları gerçek zamanlı akış simülasyonu olarak oynatılır. "
        "GPD+SVM tetikleyici olay tespit ettiğinde PhaseNet kaskad faz belirleyici devreye girer."
    )

    import h5py as _h5_live
    from seismic_engine.streaming import ReplaySource, RingBuffer
    from seismic_engine.inference.cascaded_detector import CascadedDetector

    if "live_monitor_state" not in st.session_state:
        st.session_state["live_monitor_state"] = {
            "running": False,
            "events_detected": [],
            "total_chunks": 0,
        }

    _live_col_ctrl, _live_col_main = st.columns([1, 3])

    with _live_col_ctrl:
        st.markdown('<span class="sidebar-label">KAYNAK SEÇİMİ</span>', unsafe_allow_html=True)

        _live_meta = pd.read_csv(ROOT / "data" / "augmented_dataset" / "metadata.csv")
        _live_events = _live_meta[
            (_live_meta["window_type"] == "event") &
            (_live_meta["augmentation"] == "original")
        ].sort_values("source_magnitude", ascending=False)

        _live_labels = [
            f"M{row.source_magnitude:.1f} | {row.station} | {row.trace_name[:40]}"
            for _, row in _live_events.head(30).iterrows()
        ]
        _live_sel_idx = st.selectbox(
            "Demo trace seçin",
            range(len(_live_labels)),
            format_func=lambda i: _live_labels[i],
            key="live_trace_sel",
        )
        _live_sel_row = _live_events.iloc[_live_sel_idx]

        st.markdown("---")
        _live_speed = st.select_slider(
            "Oynatma hızı",
            options=["1x (Gerçek)", "2x", "4x", "10x", "Anında"],
            value="4x",
            key="live_speed",
        )
        _speed_map = {"1x (Gerçek)": 1.0, "2x": 0.5, "4x": 0.25, "10x": 0.1, "Anında": 0.01}
        _chunk_delay = _speed_map.get(_live_speed, 0.25)

        _live_window_sec = st.slider(
            "Görüntü penceresi (s)", 10, 60, 30, step=5, key="live_window"
        )

        st.markdown("---")
        _live_start = st.button("▶️  Akışı Başlat", type="primary", use_container_width=True, key="live_start_btn")
        _live_stop = st.button("⏹  Durdur", use_container_width=True, key="live_stop_btn")

        st.markdown("---")
        st.markdown(
            f'<div class="metric-card">'
            f'<b style="color:#00d4ff">İstasyon:</b> {_live_sel_row["station"]}<br>'
            f'<b style="color:#00d4ff">Magnitüd:</b> M{_live_sel_row.get("source_magnitude", 0):.1f}<br>'
            f'<b style="color:#00d4ff">Süre:</b> 60.0s<br>'
            f'<b style="color:#00d4ff">Mod:</b> Replay Buffer'
            f'</div>',
            unsafe_allow_html=True,
        )

    with _live_col_main:
        if _live_stop:
            st.session_state["live_monitor_state"]["running"] = False
            st.info("Akış durduruldu.")

        if _live_start:
            st.session_state["live_monitor_state"]["running"] = True
            st.session_state["live_monitor_state"]["events_detected"] = []

            _hdf5_path = ROOT / "data" / "augmented_dataset" / "waveforms.hdf5"
            with _h5_live.File(str(_hdf5_path), "r") as _hf_live:
                _live_data = _hf_live["data"][_live_sel_row["trace_name"]][:]

            _replay = ReplaySource(
                data=_live_data,
                sample_rate=100.0,
                chunk_samples=100,
                station=_live_sel_row["station"],
            )
            _ring = RingBuffer(capacity_samples=6000, n_channels=3)

            _status_container = st.empty()
            _chart_container = st.empty()
            _detection_container = st.empty()
            _picks_container = st.empty()
            _report_container = st.empty()

            _status_container.markdown(
                '<div style="background:rgba(0,212,255,0.08);border:1px solid rgba(0,212,255,0.3);'
                'border-radius:8px;padding:10px 16px;margin-bottom:10px">'
                '<span style="color:#00d4ff;font-weight:600">📡 CANLI AKIŞ AKTİF</span>'
                f' &nbsp;|&nbsp; İstasyon: <b>{_replay.station}</b>'
                f' &nbsp;|&nbsp; Hız: {_live_speed}'
                '</div>',
                unsafe_allow_html=True,
            )

            _trigger_threshold_sec = 4.0
            _trigger_window = int(_trigger_threshold_sec * 100)
            _detection_fired = False
            _cascade_result = None

            for _chunk in _replay.stream_chunks(real_time=False):
                if not st.session_state["live_monitor_state"]["running"]:
                    break

                _ring.append(_chunk)
                st.session_state["live_monitor_state"]["total_chunks"] += 1

                _visible_samples = int(_live_window_sec * 100)
                _visible = _ring.get_latest(_visible_samples)
                _t_axis = np.arange(_visible.shape[1]) / 100.0

                _fig_live = go.Figure()
                _ch_names = ["HHZ", "HHN", "HHE"]
                _ch_colors = ["#00d4ff", "#7b8cde", "rgba(255,107,53,0.8)"]
                for _ci in range(3):
                    _ch_data = _visible[_ci]
                    _norm_ch = _ch_data / (np.abs(_ch_data).max() + 1e-9)
                    _fig_live.add_trace(go.Scatter(
                        x=_t_axis,
                        y=_norm_ch + (2 - _ci) * 2.5,
                        name=_ch_names[_ci],
                        line=dict(width=0.8, color=_ch_colors[_ci]),
                        hoverinfo="skip",
                    ))

                _elapsed = _replay.elapsed_seconds
                _progress_pct = _replay.progress * 100

                if _detection_fired and _cascade_result is not None:
                    _p_sec = _cascade_result.get("phasenet_p_seconds")
                    _s_sec = _cascade_result.get("phasenet_s_seconds")
                    if _p_sec is not None:
                        _fig_live.add_vline(
                            x=_p_sec, line_width=2.5, line_dash="solid",
                            line_color="#00ff88",
                            annotation_text=f"P {_p_sec:.2f}s",
                            annotation_position="top left",
                            annotation_font=dict(size=11, color="#00ff88"),
                        )
                    if _s_sec is not None:
                        _fig_live.add_vline(
                            x=_s_sec, line_width=2.5, line_dash="dash",
                            line_color="#ff6b35",
                            annotation_text=f"S {_s_sec:.2f}s",
                            annotation_position="top right",
                            annotation_font=dict(size=11, color="#ff6b35"),
                        )

                _fig_live.update_layout(
                    height=350,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(10,14,26,0.95)",
                    margin=dict(l=40, r=20, t=35, b=35),
                    xaxis=dict(
                        title=f"Zaman (s) — Geçen: {_elapsed:.1f}s ({_progress_pct:.0f}%)",
                        gridcolor="rgba(0,212,255,0.06)",
                        zeroline=False,
                        tickfont=dict(color="rgba(232,236,240,0.5)", family="Space Mono", size=9),
                        titlefont=dict(color="rgba(232,236,240,0.6)", size=10),
                    ),
                    yaxis=dict(
                        showticklabels=False,
                        gridcolor="rgba(0,212,255,0.04)",
                        zeroline=False,
                    ),
                    legend=dict(
                        orientation="h", y=1.08,
                        bgcolor="rgba(15,21,37,0.7)",
                        font=dict(color="#e8ecf0", size=9),
                    ),
                    showlegend=True,
                )
                _chart_container.plotly_chart(_fig_live, use_container_width=True, key=f"live_chart_{_replay._position}")

                if (
                    not _detection_fired
                    and _ring.is_full
                    and _ring.total_samples_received >= _trigger_window
                ):
                    _full_window = _ring.get_window(6000)

                    @st.cache_resource
                    def _load_cascade():
                        _cd = CascadedDetector(device="cpu")
                        _cd.load(ROOT / "models")
                        return _cd

                    _cascade = _load_cascade()
                    _cascade_result = _cascade.predict(
                        _full_window, station=_live_sel_row["station"]
                    )

                    if _cascade_result["is_event"]:
                        _detection_fired = True
                        _ev_prob = _cascade_result["trace_event_prob"]
                        _p_sec_det = _cascade_result.get("phasenet_p_seconds")
                        _s_sec_det = _cascade_result.get("phasenet_s_seconds")
                        _p_conf = _cascade_result.get("phasenet_p_confidence", 0)
                        _s_conf = _cascade_result.get("phasenet_s_confidence", 0)

                        _detection_container.markdown(
                            '<div style="background:rgba(0,200,83,0.12);'
                            'border:2px solid #00c853;border-radius:10px;'
                            'padding:14px 18px;margin:10px 0">'
                            '<span style="font-size:1.2rem;font-weight:700;color:#00c853">'
                            '🚨 OLAY TESPİT EDİLDİ — Kaskad Tetiklendi</span><br>'
                            f'<span style="color:rgba(232,236,240,0.8);font-size:0.9rem">'
                            f'SVM Olasılık: {_ev_prob:.3f} &nbsp;|&nbsp; '
                            f'PhaseNet P: {_p_sec_det:.2f}s (güven: {_p_conf:.2f}) &nbsp;|&nbsp; '
                            f'PhaseNet S: {_s_sec_det:.2f}s (güven: {_s_conf:.2f})'
                            f'</span></div>',
                            unsafe_allow_html=True,
                        ) if _p_sec_det and _s_sec_det else _detection_container.markdown(
                            '<div style="background:rgba(0,200,83,0.12);'
                            'border:2px solid #00c853;border-radius:10px;'
                            'padding:14px 18px;margin:10px 0">'
                            '<span style="font-size:1.2rem;font-weight:700;color:#00c853">'
                            '🚨 OLAY TESPİT EDİLDİ</span><br>'
                            f'<span style="color:rgba(232,236,240,0.8);font-size:0.9rem">'
                            f'SVM Olasılık: {_ev_prob:.3f} &nbsp;|&nbsp; '
                            f'PhaseNet faz belirleme bekleniyor...'
                            f'</span></div>',
                            unsafe_allow_html=True,
                        )

                        _pc1, _pc2, _pc3, _pc4 = _picks_container.columns(4)
                        with _pc1:
                            st.metric("SVM Olasılık", f"{_ev_prob:.3f}")
                        with _pc2:
                            st.metric("P-Varış (PhaseNet)", f"{_p_sec_det:.2f}s" if _p_sec_det else "—")
                        with _pc3:
                            st.metric("S-Varış (PhaseNet)", f"{_s_sec_det:.2f}s" if _s_sec_det else "—")
                        with _pc4:
                            _sp_diff = (_s_sec_det - _p_sec_det) if (_s_sec_det and _p_sec_det) else None
                            st.metric("S-P Farkı", f"{_sp_diff:.2f}s" if _sp_diff else "—")

                        st.session_state["live_monitor_state"]["events_detected"].append({
                            "station": _live_sel_row["station"],
                            "magnitude": float(_live_sel_row.get("source_magnitude", 0)),
                            "event_prob": _ev_prob,
                            "p_seconds": _p_sec_det,
                            "s_seconds": _s_sec_det,
                            "elapsed": _elapsed,
                        })

                        from risk_advisor.report_generator import ReportGenerator
                        from risk_advisor.schema import EnginePayload, DetectedEvent
                        from datetime import datetime as _dt_live

                        _rgen_live = ReportGenerator()
                        _rgen_live.initialize()

                        _det_ev = DetectedEvent(
                            trace_id=_live_sel_row["trace_name"],
                            magnitude=float(_live_sel_row.get("source_magnitude", 0)),
                            model_name="cascade_gpd_phasenet",
                            p_probability_peak=_ev_prob,
                            p_pick_sample=_cascade_result.get("phasenet_p_sample"),
                            p_pick_seconds=_p_sec_det,
                        )
                        _payload_live = EnginePayload(
                            timestamp=_dt_live.utcnow(),
                            pipeline_version="2.0.0-cascade",
                            region="marmara",
                            detected_events=[_det_ev],
                        )

                        _report_container.markdown("---")
                        _report_container.markdown("### 📋 Otomatik Risk Raporu (Tetiklendi)")
                        _report_text_live = ""
                        _report_display = _report_container.empty()
                        for _rchunk in _rgen_live.stream_report(_payload_live, language="tr"):
                            _report_text_live += _rchunk
                            _report_display.markdown(_report_text_live)

                        break

                import time as _time_mod
                _time_mod.sleep(_chunk_delay)

            if not _detection_fired:
                _detection_container.info(
                    "Akış tamamlandı — bu pencerede olay tespit edilmedi. "
                    "Farklı bir trace seçerek tekrar deneyin."
                )

            st.session_state["live_monitor_state"]["running"] = False

        elif not st.session_state["live_monitor_state"].get("running"):
            st.markdown(
                '<div style="text-align:center;padding:80px 20px;'
                'border:1px dashed rgba(0,212,255,0.2);border-radius:12px;margin:20px 0">'
                '<p style="font-size:3rem;margin-bottom:10px">📡</p>'
                '<p style="color:rgba(232,236,240,0.6);font-size:1rem">'
                'Canlı izleme demo modu</p>'
                '<p style="color:rgba(232,236,240,0.4);font-size:0.8rem">'
                'Sol panelden bir trace seçin ve <b>▶️ Akışı Başlat</b> butonuna tıklayın.<br>'
                'Kaydedilmiş dalga formu gerçek zamanlı olarak yeniden oynatılacaktır.<br>'
                'GPD+SVM tetikleyici olay tespit ettiğinde PhaseNet kaskadı devreye girer.</p>'
                '</div>',
                unsafe_allow_html=True,
            )
