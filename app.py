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
tab1, tab2, tab3, tab4 = st.tabs([
    "🗺  Olay Haritası",
    "〰  Dalga Formu Görüntüleyici",
    "🏗  TBDY-2018 Risk Danışmanı",
    "🔴  Etki Senaryosu",
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
        show_hazard  = st.checkbox("🔴 Sismik Tehlike Katmanı (OSTA)", value=False,
                                   help="Olasılıksal Sismik Tehlike Analizi — P(M≥6, 50 yıl) "
                                        "— Deprem tahmini değildir")

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

        # ── OSTA Tehlike Katmanı ─────────────────────────────────────────
        if show_hazard:
            # Use v2 calibrated grid (3-zone ISC 1990-2023) if available, else v1
            _hgrid_v2 = ROOT / "artifacts" / "spatial_hazard_grid_v2.csv"
            _hgrid_v1 = ROOT / "artifacts" / "spatial_hazard_grid.csv"
            _hgrid_path = _hgrid_v2 if _hgrid_v2.exists() else _hgrid_v1
            _grid_ver   = "v2 (ISC 1990-2023)" if _hgrid_v2.exists() else "v1"
            if _hgrid_path.exists():
                _hg = pd.read_csv(_hgrid_path)
                # v2 grid already has Omori-corrected lambda_M6; v1 needs 50× correction
                if "P50yr_M6" in _hg.columns:
                    # v2: use P50yr_M6 directly (already background-corrected)
                    _hg["P50_M6_bg"] = _hg["P50yr_M6"]
                else:
                    _OMORI = 50.0
                    _hg["lambda_M6_bg"] = _hg["lambda_M6"] / _OMORI
                    _hg["P50_M6_bg"]    = 1 - np.exp(-_hg["lambda_M6_bg"] * 50)
                # Classify into hazard tiers for labeling
                def _htier(p):
                    if p < 0.20:  return "Düşük (<20%)"
                    if p < 0.50:  return "Orta (20-50%)"
                    if p < 0.80:  return "Yüksek (50-80%)"
                    return "Çok Yüksek (>80%)"
                _hg["tehlike_turu"] = _hg["P50_M6_bg"].apply(_htier)
                _hg["label"] = (
                    "P(M≥6,50yr)=" + (_hg["P50_M6_bg"] * 100).round(1).astype(str) + "%"
                    + "<br>b=" + _hg["b_value"].round(3).astype(str)
                    + "<br>N=" + _hg["n_events"].astype(str)
                )
                _tier_color = {
                    "Düşük (<20%)":       "#00c853",
                    "Orta (20-50%)":      "#ffd600",
                    "Yüksek (50-80%)":    "#ff9800",
                    "Çok Yüksek (>80%)":  "#ef5350",
                }
                for _tier, _tc in _tier_color.items():
                    _sub_h = _hg[_hg["tehlike_turu"] == _tier]
                    if len(_sub_h) == 0:
                        continue
                    fig.add_trace(go.Scattermapbox(
                        lat=_sub_h["lat_center"].tolist(),
                        lon=_sub_h["lon_center"].tolist(),
                        mode="markers",
                        marker=dict(size=18, color=_tc, opacity=0.45),
                        customdata=_sub_h[["P50_M6_bg", "b_value", "n_events"]].values,
                        hovertemplate=(
                            "<b>OSTA Tehlike Hücresi</b><br>"
                            "P(M≥6, 50yr): %{customdata[0]:.1%}<br>"
                            "b-değeri: %{customdata[1]:.3f}<br>"
                            "N olay: %{customdata[2]}<br>"
                            "<i>Uzun vadeli istatistik — deprem tahmini değildir</i>"
                            "<extra></extra>"
                        ),
                        name=f"OSTA: {_tier}",
                    ))
                fig.add_annotation(
                    text="🔴 OSTA Katmanı: P(M≥6,50yr) — Uzun vadeli istatistik",
                    xref="paper", yref="paper", x=0.01, y=0.01,
                    font=dict(color="#ff9800", size=9),
                    showarrow=False, bgcolor="rgba(15,21,37,0.7)",
                    bordercolor="#ff9800", borderwidth=1,
                )

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

    # Import risk + hazard modules
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "scripts"))
    try:
        from importlib import import_module as _im
        _risk = _im("37_regional_risk_assessment")
        _risk_ok = True
    except Exception as _e:
        _risk_ok = False
        st.error(f"Risk modülü yüklenemedi: {_e}")
    try:
        _haz = _im("43_hazard_calculator")
        _haz_ok = True
    except Exception as _e2:
        _haz_ok = False

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

                # ── OSTA: Sismik Tehlike Analizi ─────────────────────
                st.markdown("#### 📊 Sismik Tehlike Analizi (OSTA)")
                st.caption(
                    "Poisson istatistiğine dayalı uzun vadeli sismik tehlike tahmini. "
                    "Kısa vadeli deprem tahmini değildir. "
                    "v2 kalibrasyon: ISC 1990–2023 (46,254 olay), 3 bölge (KAF/DAF/OA), "
                    "2023 artçı şok dönemi hariç (2023-02-06 – 2023-08-06)."
                )
                if _haz_ok:
                    _h5  = _haz.get_hazard_probability(_lat, _lon, 5.0, 50)
                    _h6  = _haz.get_hazard_probability(_lat, _lon, 6.0, 50)
                    _h7  = _haz.get_hazard_probability(_lat, _lon, 7.0, 50)
                    _hcl = _haz.classify_hazard_level(_h5["probability"])
                    _hcl7 = _haz.classify_hazard_level(_h7["probability"])

                    _hc1, _hc2, _hc3 = st.columns(3)
                    with _hc1:
                        st.markdown(
                            f'<div style="background:{_hcl["color"]}22;'
                            f'border:1.5px solid {_hcl["color"]}66;'
                            f'border-radius:9px;padding:12px 14px;text-align:center">'
                            f'<div style="font-size:0.72rem;color:rgba(232,236,240,0.6);'
                            f'text-transform:uppercase;letter-spacing:1px">50 Yılda M≥5.0</div>'
                            f'<div style="font-size:1.6rem;font-weight:700;'
                            f'color:{_hcl["color"]}">{_h5["probability_pct"]:.0f}%</div>'
                            f'<div style="font-size:0.68rem;color:rgba(232,236,240,0.5)">'
                            f'T={_h5["return_period_yr"]:.0f} yıl</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    with _hc2:
                        st.markdown(
                            f'<div style="background:#ff980022;border:1.5px solid #ff980066;'
                            f'border-radius:9px;padding:12px 14px;text-align:center">'
                            f'<div style="font-size:0.72rem;color:rgba(232,236,240,0.6);'
                            f'text-transform:uppercase;letter-spacing:1px">50 Yılda M≥6.0</div>'
                            f'<div style="font-size:1.6rem;font-weight:700;color:#ff9800">'
                            f'{_h6["probability_pct"]:.0f}%</div>'
                            f'<div style="font-size:0.68rem;color:rgba(232,236,240,0.5)">'
                            f'T={_h6["return_period_yr"]:.0f} yıl</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    with _hc3:
                        st.markdown(
                            f'<div style="background:{_hcl7["color"]}22;'
                            f'border:1.5px solid {_hcl7["color"]}66;'
                            f'border-radius:9px;padding:12px 14px;text-align:center">'
                            f'<div style="font-size:0.72rem;color:rgba(232,236,240,0.6);'
                            f'text-transform:uppercase;letter-spacing:1px">50 Yılda M≥7.0</div>'
                            f'<div style="font-size:1.6rem;font-weight:700;'
                            f'color:{_hcl7["color"]}">{_h7["probability_pct"]:.0f}%</div>'
                            f'<div style="font-size:0.68rem;color:rgba(232,236,240,0.5)">'
                            f'T={_h7["return_period_yr"]:.0f} yıl</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                    # G-R curve mini-plot
                    with st.expander("📈 Gutenberg-Richter Eğrisi — Bu Konum için", expanded=False):
                        _gr_curve = _haz.get_gr_curve(_lat, _lon)
                        _gr_m   = _gr_curve["magnitude"]
                        _gr_lam = _gr_curve["annual_rate"]
                        _gr_fig = go.Figure()
                        _gr_fig.add_trace(go.Scatter(
                            x=_gr_m, y=_gr_lam,
                            mode="lines", name=f"G-R (b={_gr_curve['b_value']:.3f})",
                            line=dict(color="#ff6b35", width=2),
                        ))
                        # Poisson probability right axis: overlay as annotation boxes
                        for _m_mark, _m_color in [(5.0,"#00d4ff"), (6.0,"#ff9800"), (7.0,"#ef5350")]:
                            _lam_mark = _gr_curve["annual_rate"][
                                min(range(len(_gr_m)), key=lambda _i: abs(_gr_m[_i]-_m_mark))
                            ]
                            _gr_fig.add_vline(
                                x=_m_mark, line_dash="dot", line_color=_m_color, line_width=1.5,
                                annotation_text=f"M{_m_mark:.0f}",
                                annotation_font_color=_m_color, annotation_font_size=9,
                            )
                        _gr_fig.update_layout(
                            height=240, margin=dict(l=50, r=20, t=20, b=40),
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,21,37,0.85)",
                            font=dict(color="#e8ecf0", family="DM Sans"),
                            xaxis=dict(title=dict(text="Magnitüd", font=dict(size=10)),
                                       gridcolor="rgba(0,212,255,0.08)", zeroline=False,
                                       tickfont=dict(color="rgba(232,236,240,0.5)", size=9)),
                            yaxis=dict(title=dict(text="Yıllık oran λ(≥M)", font=dict(size=10)),
                                       type="log", gridcolor="rgba(0,212,255,0.08)",
                                       zeroline=False,
                                       tickfont=dict(color="rgba(232,236,240,0.5)", size=9)),
                            legend=dict(font=dict(size=9, color="#e8ecf0"),
                                       bgcolor="rgba(15,21,37,0.7)"),
                        )
                        st.plotly_chart(_gr_fig, use_container_width=True)
                        st.caption(
                            f"Bölge: {_h5['region']}  |  b-değeri: {_gr_curve['b_value']:.3f}  |  "
                            f"Kaynak: {_h5['source']}"
                        )

                    st.markdown(
                        '<p style="font-size:0.7rem;color:rgba(232,236,240,0.45);'
                        'margin-top:4px">⚠ Bu değerler Poisson istatistiğine dayalı uzun vadeli '
                        'sismik tehlike tahminidir. Kısa vadeli deprem tahmini değildir. '
                        'Kaynak: ISC 1990–2023 (46,254 olay), 3-bölge kalibrasyon (KAF/DAF/OA).</p>',
                        unsafe_allow_html=True,
                    )
                    st.markdown("---")
                else:
                    st.info("OSTA modülü yüklenemedi — `scripts/43_hazard_calculator.py` kontrolü yapın.")
                    st.markdown("---")

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
# SEKME 4 — Deprem Etki Senaryosu (Hipotetik Senaryo Simülasyonu)
# ══════════════════════════════════════════════════════════════════
with tab4:
    # ── Load impact calculator ────────────────────────────────────────────────
    try:
        import importlib.util as _ilu
        _spec47 = _ilu.spec_from_file_location(
            "impact_calc",
            ROOT / "scripts" / "47_impact_calculator.py"
        )
        _ic = _ilu.module_from_spec(_spec47)
        _spec47.loader.exec_module(_ic)
        _impact_ok = True
    except Exception as _ie:
        _impact_ok = False
        _impact_err = str(_ie)

    st.markdown("""
    <div style='background:linear-gradient(90deg,#7f0000,#c0392b);
                padding:12px 18px;border-radius:8px;margin-bottom:12px'>
      <span style='font-size:1.25rem;font-weight:700;color:white'>
        🔴 Deprem Etki Senaryosu — Hipotetik Senaryo Simülasyonu
      </span>
    </div>
    """, unsafe_allow_html=True)

    st.warning(
        "⚠️ **Bu araç olası etki tahmini üretir. Kesin tahmin değildir.**  \n"
        "FEMA HAZUS metodolojisi temel alınmıştır. 1999 Kocaeli (M7.6) ve 2023 "
        "Kahramanmaraş (M7.8) gerçek hasar verileriyle kalibre edilmiştir.  \n"
        "**Tahminler il düzeyi ortalamadır; ilçe/bina düzeyi hassasiyetinde değildir.**"
    )

    if not _impact_ok:
        st.error(f"İmpact hesaplayıcı yüklenemedi: {_impact_err}")
        st.stop()

    # ── Layout: left controls | right results ─────────────────────────────────
    _col_ctrl, _col_res = st.columns([1, 2], gap="large")

    with _col_ctrl:
        st.markdown("#### 📍 Deprem Senaryosu Tanımla")

        # Preset buttons
        st.markdown("**Hazır Senaryolar:**")
        _p1, _p2, _p3 = st.columns(3)
        _preset_key = st.session_state.get("impact_preset", None)
        with _p1:
            if st.button("🔁 1999\nKocaeli", use_container_width=True):
                st.session_state["impact_preset"]  = "kocaeli_1999"
                st.session_state["impact_lat"]     = 40.76
                st.session_state["impact_lon"]     = 29.97
                st.session_state["impact_mag"]     = 7.6
                st.session_state["impact_depth"]   = 17
                st.session_state["impact_time"]    = "Gece (02:00)"
                st.rerun()
        with _p2:
            if st.button("🏙️ Marmara\nM7.2", use_container_width=True):
                st.session_state["impact_preset"]  = "marmara_m72"
                st.session_state["impact_lat"]     = 40.80
                st.session_state["impact_lon"]     = 28.50
                st.session_state["impact_mag"]     = 7.2
                st.session_state["impact_depth"]   = 12
                st.session_state["impact_time"]    = "Gece (02:00)"
                st.rerun()
        with _p3:
            if st.button("🌊 İzmir\nM6.9", use_container_width=True):
                st.session_state["impact_preset"]  = "izmir_m69"
                st.session_state["impact_lat"]     = 38.35
                st.session_state["impact_lon"]     = 26.79
                st.session_state["impact_mag"]     = 6.9
                st.session_state["impact_depth"]   = 10
                st.session_state["impact_time"]    = "Gündüz (14:00)"
                st.rerun()

        st.markdown("---")

        # Manual inputs
        _sc_lat = st.number_input(
            "Merkez Üssü Enlemi (°N)",
            min_value=36.0, max_value=42.5, step=0.01,
            value=float(st.session_state.get("impact_lat", 40.76)),
            key="impact_lat_inp",
        )
        _sc_lon = st.number_input(
            "Merkez Üssü Boylamı (°E)",
            min_value=26.0, max_value=44.5, step=0.01,
            value=float(st.session_state.get("impact_lon", 29.97)),
            key="impact_lon_inp",
        )
        _sc_mag = st.slider(
            "Büyüklük (Mw)", min_value=5.0, max_value=8.0, step=0.1,
            value=float(st.session_state.get("impact_mag", 7.2)),
            key="impact_mag_sl",
        )
        _dep_options = [5, 10, 15, 20, 25, 30]
        _dep_raw = int(st.session_state.get("impact_depth", 15))
        _dep_val = min(_dep_options, key=lambda x: abs(x - _dep_raw))
        _sc_dep = st.select_slider(
            "Odak Derinliği (km)", options=_dep_options,
            value=_dep_val,
            key="impact_dep_sl",
        )
        _sc_time_lbl = st.radio(
            "Zaman", ["Gece (02:00)", "Gündüz (14:00)"],
            index=0 if st.session_state.get("impact_time","Gece")=="Gece (02:00)" else 1,
            key="impact_time_r",
            horizontal=True,
        )
        _sc_time = "night" if "Gece" in _sc_time_lbl else "day"

        st.markdown("---")
        _run_btn = st.button(
            "🔴 Senaryoyu Simüle Et",
            use_container_width=True,
            type="primary",
        )

        # Persistent disclaimer
        st.markdown(
            '<div style="border:1px solid #555;border-radius:6px;padding:10px;'
            'margin-top:12px;font-size:0.75rem;color:#aaa">'
            '⚠️ <b>Hipotetik Senaryo Simülasyonu</b><br>'
            'Bu araç istatistiksel etki tahmini üretir. Kesin tahmin değildir. '
            'FEMA HAZUS metodolojisi temel alınmıştır. 1999 Kocaeli (M7.6) ve '
            '2023 Kahramanmaraş (M7.8) verileriyle kalibre edilmiştir.'
            '</div>',
            unsafe_allow_html=True,
        )

    with _col_res:
        if _run_btn or st.session_state.get("impact_result"):
            if _run_btn:
                with st.spinner("Senaryo hesaplanıyor..."):
                    _res = _ic.run_scenario(_sc_lat, _sc_lon, _sc_mag, _sc_dep, _sc_time)
                st.session_state["impact_result"] = _res
            else:
                _res = st.session_state["impact_result"]

            if "error" in _res:
                st.error(f"Hesaplama hatası: {_res['error']}")
            else:
                _tot  = _res["totals"]
                _cas  = _tot["casualties"]
                _dmg  = _tot["buildings"]
                _eco  = _tot["economic"]
                _top  = _res["top_affected"]
                _prov = _res["province_results"]
                _sc   = _res["scenario"]

                # ── Section 1: Animated ShakeMap ─────────────────────────────
                st.markdown("#### 🗺️ ShakeMap — MMI Dağılımı")

                # Build animated concentric rings + province scatter
                _prov_df = pd.DataFrame(_prov)

                # Color per MMI bucket
                def _mmi_color(mmi):
                    if mmi < 3:   return "#e0e0e0"
                    if mmi < 5:   return "#a8e6cf"
                    if mmi < 6:   return "#ffd700"
                    if mmi < 7:   return "#ff9800"
                    if mmi < 8:   return "#ef5350"
                    if mmi < 9:   return "#b71c1c"
                    return "#4a0000"

                _prov_df["color"] = _prov_df["mmi"].apply(_mmi_color)
                _prov_df["label"] = (
                    _prov_df["province"] + "<br>MMI: " + _prov_df["mmi"].round(1).astype(str)
                    + "<br>PGA: " + (_prov_df["pga_g"] * 100).round(1).astype(str) + "% g"
                    + "<br>Mesafe: " + _prov_df["r_jb_km"].round(0).astype(str) + " km"
                )

                # Static ShakeMap figure (Plotly mapbox)
                _sm_fig = go.Figure()

                # Province centroids colored by MMI
                _sm_fig.add_trace(go.Scattermapbox(
                    lat=_prov_df["lat"].tolist(),
                    lon=_prov_df["lon"].tolist(),
                    mode="markers+text",
                    marker=dict(
                        size=(_prov_df["mmi"].clip(4, 10) - 3) * 4,
                        color=_prov_df["color"].tolist(),
                        opacity=0.80,
                    ),
                    text=_prov_df["province"].tolist(),
                    textposition="top center",
                    hovertext=_prov_df["label"].tolist(),
                    hoverinfo="text",
                    name="İller (MMI)",
                ))

                # Epicenter marker
                _sm_fig.add_trace(go.Scattermapbox(
                    lat=[_sc["lat"]], lon=[_sc["lon"]],
                    mode="markers+text",
                    marker=dict(size=22, color="#ff0000", symbol="star"),
                    text=[f"M{_sc['magnitude']}"],
                    textposition="top right",
                    hovertext=[f"Merkez Üssü<br>M{_sc['magnitude']}, {_sc['depth_km']}km"],
                    hoverinfo="text",
                    name="Merkez Üssü",
                ))

                _sm_fig.update_layout(
                    mapbox_style="carto-darkmatter",
                    mapbox=dict(center=dict(lat=_sc["lat"], lon=_sc["lon"]), zoom=5),
                    height=380, margin=dict(l=0, r=0, t=0, b=0),
                    legend=dict(orientation="h", y=-0.05),
                    paper_bgcolor="#0e1117",
                )
                st.plotly_chart(_sm_fig, use_container_width=True)

                # MMI legend
                _mmi_cols = st.columns(7)
                for _ci, (_label, _clr) in enumerate([
                    ("MMI 1-2", "#e0e0e0"), ("MMI 3-4", "#a8e6cf"),
                    ("MMI 5", "#ffd700"), ("MMI 6", "#ff9800"),
                    ("MMI 7", "#ef5350"), ("MMI 8", "#b71c1c"), ("MMI 9+", "#4a0000"),
                ]):
                    with _mmi_cols[_ci]:
                        st.markdown(
                            f'<div style="background:{_clr};color:{"#000" if _ci<3 else "#fff"};'
                            f'text-align:center;border-radius:4px;padding:2px 4px;'
                            f'font-size:0.65rem">{_label}</div>',
                            unsafe_allow_html=True,
                        )

                # ── Section 2: Four metric cards ─────────────────────────────
                st.markdown("#### 📊 Tahmini Etki Özeti")
                _mc1, _mc2, _mc3, _mc4 = st.columns(4)
                with _mc1:
                    st.metric(
                        "🏚 Hasar Gören Bina",
                        f"~{(_dmg['slight']+_dmg['moderate']+_dmg['extensive']):,.0f}",
                        help="Hafif + orta + ağır hasar"
                    )
                    st.caption(f"Ağır: {_dmg['extensive']:,}")
                with _mc2:
                    st.metric(
                        "🏗 Yıkılan / Çöken Bina",
                        f"~{_dmg['complete']:,.0f}",
                        help="Tam yıkım (complete collapse)"
                    )
                    st.caption(f"Kullanılamaz: {_dmg['total_unusable']:,}")
                with _mc3:
                    st.metric(
                        "💔 Can Kaybı Tahmini",
                        f"{_cas['fatality_low']:,} – {_cas['fatality_high']:,}",
                        help="Alt/üst tahmin (±40% belirsizlik)"
                    )
                    st.caption(f"Orta: {_cas['fatality_mid']:,}")
                with _mc4:
                    st.metric(
                        "💰 Ekonomik Kayıp",
                        f"~{_eco['total_loss_tl']/1e9:.0f} Mrd TL",
                        help="Dolaylı kayıplar dahil (1.5× çarpanı)"
                    )
                    st.caption(f"GSYİH'nın %{_eco['loss_pct_gdp']:.1f}")

                # Displaced + injured
                _mi1, _mi2 = st.columns(2)
                with _mi1:
                    st.metric("🏕 Yerinden Edilmiş", f"~{_cas['displaced']:,}")
                with _mi2:
                    st.metric("🏥 Yaralı Tahmini",
                              f"{_cas['injured_low']:,} – {_cas['injured_high']:,}")

                # ── Section 3: Affected Provinces Table ───────────────────────
                if _top:
                    st.markdown("#### 🔴 En Fazla Etkilenen İller")
                    _top_df = pd.DataFrame([{
                        "İl":            p["province"],
                        "MMI":           round(p["mmi"], 1),
                        "Nüfus":         f"{p['population']:,}",
                        "Can Kaybı":     f"{p['casualties']['fatality_mid']:,}",
                        "Yıkılan Bina":  f"{p['damage']['complete']:,}",
                        "Mesafe (km)":   round(p["r_jb_km"], 0),
                    } for p in _top])
                    st.dataframe(
                        _top_df,
                        use_container_width=True,
                        hide_index=True,
                    )

                # ── Section 4: Damage breakdown bar chart ─────────────────────
                _top8 = sorted(_prov, key=lambda x: x["mmi"], reverse=True)[:8]
                if any(p["damage"]["slight"] > 0 for p in _top8):
                    st.markdown("#### 📉 Hasar Dağılımı (En Çok Etkilenen 8 İl)")
                    _bar_fig = go.Figure()
                    _ds_config = [
                        ("complete",  "Çöken",  "#7b0000"),
                        ("extensive", "Ağır",   "#ef5350"),
                        ("moderate",  "Orta",   "#ff9800"),
                        ("slight",    "Hafif",  "#ffd700"),
                    ]
                    _pnames = [p["province"] for p in _top8]
                    for _ds_key, _ds_label, _ds_clr in _ds_config:
                        _vals = [p["damage"].get(_ds_key, 0) for p in _top8]
                        _bar_fig.add_trace(go.Bar(
                            name=_ds_label, x=_pnames, y=_vals,
                            marker_color=_ds_clr,
                        ))
                    _bar_fig.update_layout(
                        barmode="stack", height=280,
                        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                        font=dict(color="#e8ecf0"),
                        margin=dict(l=0, r=0, t=10, b=40),
                        legend=dict(orientation="h", y=1.12),
                        xaxis=dict(tickfont=dict(size=10)),
                        yaxis=dict(title="Bina Sayısı"),
                    )
                    st.plotly_chart(_bar_fig, use_container_width=True)

                # ── Section 5: TBDY-2018 implication ─────────────────────────
                st.markdown("#### 🏛 TBDY-2018 Bağlamı")
                _dts1_provs = [
                    p["province"] for p in _prov
                    if p["mmi"] >= 7.0
                ]
                if _dts1_provs:
                    st.info(
                        f"Bu senaryo şu illerde **DTS-1** tasarım kriterlerini "
                        f"zorunlu kılacak düzeyde sarsıntı üretmektedir:\n\n"
                        + ", ".join(_dts1_provs[:10])
                    )
                else:
                    st.info("Bu senaryo DTS-1 eşiğini (MMI≥7) aşan il bulunmamaktadır.")

                # Non-compliant building estimate (pre-1980 masonry in MMI≥6 zones)
                _noncompliant = sum(
                    p["damage"]["complete"] + p["damage"]["extensive"]
                    for p in _prov if p["mmi"] >= 6.0
                )
                _total_in_zone = sum(
                    p["total_buildings"] for p in _prov if p["mmi"] >= 6.0
                )
                if _total_in_zone > 0:
                    _nc_pct = 100 * _noncompliant / _total_in_zone
                    st.caption(
                        f"MMI≥6 bölgesindeki binaların tahminen **%{_nc_pct:.1f}'i** "
                        "ağır hasar veya yıkım riski taşımaktadır."
                    )

                # ── Section 6: Historical comparison ─────────────────────────
                st.markdown("#### 📚 Tarihsel Karşılaştırma")
                _hist = [
                    {
                        "Senaryo": f"Bu Senaryo (M{_sc['magnitude']})",
                        "Can Kaybı": f"{_cas['fatality_low']:,}–{_cas['fatality_high']:,}",
                        "Yıkılan Bina": f"~{_dmg['complete']:,}",
                        "Ekono. Kayıp": f"~{_eco['total_loss_tl']/1e9:.0f} Mrd TL",
                        "Not": "Model tahmini",
                    },
                    {
                        "Senaryo": "1999 Kocaeli M7.6 (gerçek)",
                        "Can Kaybı": "17,480",
                        "Yıkılan Bina": "18,373",
                        "Ekono. Kayıp": "~650 Mrd TL",
                        "Not": "AFAD resmi",
                    },
                    {
                        "Senaryo": "2023 Kahramanmaraş M7.8 (gerçek)",
                        "Can Kaybı": "53,537",
                        "Yıkılan Bina": "107,000+",
                        "Ekono. Kayıp": "~3,100 Mrd TL",
                        "Not": "AFAD resmi",
                    },
                ]
                st.dataframe(
                    pd.DataFrame(_hist),
                    use_container_width=True,
                    hide_index=True,
                )

                # Limitation note for Marmara scenario
                if abs(_sc["lon"] - 28.5) < 1.0 and abs(_sc["lat"] - 40.8) < 1.0:
                    st.info(
                        "ℹ️ **Marmara Senaryosu Not:** İl merkezi tabanlı model, "
                        "İstanbul'un Marmara Fayı'na yakın ilçelerinin maruziyetini "
                        "tam yansıtamamaktadır. Mühendislik çalışmaları (İMO 2019) bu senaryo "
                        "için İstanbul'da **30,000–50,000 can kaybı** öngörmektedir. "
                        "Bu araçtaki düşük değer il bazlı ortalama sınırlamasından kaynaklanır."
                    )

                st.caption(
                    "Kaynak: FEMA HAZUS-MH MR5 (2012), BA08 GMPE, "
                    "Worden et al. 2012 MMI dönüşümü. "
                    "Kalibrasyon: 1999 Kocaeli (oran: 1.01×), 2020 İzmir (oran: 0.75×). "
                    "**Hipotetik Senaryo Simülasyonu — Kesin tahmin değildir.**"
                )

        else:
            st.info("👈 Sol panelden bir hazır senaryo seçin veya parametreleri "
                    "girerek **🔴 Senaryoyu Simüle Et** butonuna tıklayın.")
