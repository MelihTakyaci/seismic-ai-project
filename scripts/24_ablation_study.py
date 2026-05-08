# Phase: Quality Audit — Step 3
# Purpose: Ablation study comparing 4 pipeline configurations to isolate the
#          contribution of each component: fine-tuning, EMSC re-labelling, ensemble.
#          Config A: GPD zero-shot (no fine-tuning)
#          Config B: GPD fine-tuned on TauPy labels (gpd_koeri_finetuned.pt)
#          Config C: GPD fine-tuned on EMSC-corrected labels (gpd_final.pt)
#          Config D: Full ensemble (PhaseNet_final + GPD_final)
# Inputs:  data/augmented_dataset/waveforms.hdf5, data/augmented_dataset/metadata.csv,
#          models/gpd_koeri_finetuned.pt, models/gpd_final.pt,
#          models/phasenet_final.pt (or phasenet_koeri_finetuned.pt)
# Outputs: artifacts/ablation_results.json, figures/ablation_comparison.png
# Limitations: PhaseNet (fine-tuned) excluded from ablation due to known loss-function
#              regression in script 20. Config D uses phasenet_final.pt for annotation
#              and gpd_final.pt for sliding window — recall is GPD-driven.

from pathlib import Path
import json

import h5py
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from obspy import Stream, Trace, UTCDateTime

import seisbench.models as sbm

ROOT    = Path(__file__).resolve().parent.parent
DS_DIR  = ROOT / "data" / "augmented_dataset"
MDL_DIR = ROOT / "models"
ART     = ROOT / "artifacts"
FIG_DIR = ROOT / "figures"

SRATE      = 100.0
N_TOTAL    = 6000
ORIGIN_S   = 30.0
DETECT_TOL = 5.0
DETECT_POST= 25.0
DETECT_LO  = int((ORIGIN_S - DETECT_TOL) * SRATE)
DETECT_HI  = int((ORIGIN_S + DETECT_POST) * SRATE)
T0_REF     = UTCDateTime("2023-02-06T01:00:00")

DEVICE = "cuda" if torch.cuda.is_available() else (
         "mps"  if torch.backends.mps.is_available() else "cpu")

MAG_BANDS = [
    ("All",       -np.inf, np.inf),
    ("M<2.0",     -np.inf, 2.0),
    ("M2.0-3.0",   2.0,    3.0),
    ("M3.0-4.0",   3.0,    4.0),
    ("M≥4.0",      4.0,   np.inf),
]

print("=" * 70)
print("ABLATION STUDY — 4 configurations")
print("=" * 70)

# ── Load data ──────────────────────────────────────────────────────────────
meta = pd.read_csv(DS_DIR / "metadata.csv")
meta["trace_p_arrival_sample"] = pd.to_numeric(meta["trace_p_arrival_sample"], errors="coerce")
meta["source_magnitude"]       = pd.to_numeric(meta["source_magnitude"],       errors="coerce")
if "augmentation" not in meta.columns:
    meta["augmentation"] = "original"

test_meta = meta[
    (meta["split"] == "test") & (meta["augmentation"] == "original")
].copy().reset_index(drop=True)
print(f"\n[1] Test windows: {len(test_meta)}")

# ── Load model configs ─────────────────────────────────────────────────────
print("\n[2] Loading models...")

# Config A: GPD zero-shot
gpd_zs = sbm.GPD.from_pretrained("original").to(DEVICE).eval()
print("    A: GPD zero-shot — loaded")

# Config B: GPD fine-tuned on TauPy labels
gpd_taupy = sbm.GPD.from_pretrained("original")
ckpt_b = MDL_DIR / "gpd_koeri_finetuned.pt"
if ckpt_b.exists():
    gpd_taupy.load_state_dict(torch.load(str(ckpt_b), map_location=DEVICE, weights_only=True))
    gpd_taupy.to(DEVICE).eval()
    print(f"    B: GPD fine-tuned (TauPy) — loaded from {ckpt_b.name}")
    config_b_available = True
else:
    print("    B: gpd_koeri_finetuned.pt not found — skipping")
    config_b_available = False

# Config C: GPD fine-tuned on EMSC labels
gpd_emsc = sbm.GPD.from_pretrained("original")
ckpt_c = MDL_DIR / "gpd_final.pt"
if ckpt_c.exists():
    gpd_emsc.load_state_dict(torch.load(str(ckpt_c), map_location=DEVICE, weights_only=True))
    gpd_emsc.to(DEVICE).eval()
    print(f"    C: GPD fine-tuned (EMSC) — loaded from {ckpt_c.name}")
    config_c_available = True
else:
    print("    C: gpd_final.pt not found — skipping")
    config_c_available = False

# Config D: Full ensemble (PhaseNet + GPD fine-tuned on EMSC)
pn_ft = sbm.PhaseNet.from_pretrained("original")
pn_ckpt = next((MDL_DIR/n for n in ["phasenet_final.pt","phasenet_koeri_finetuned.pt"]
                if (MDL_DIR/n).exists()), None)
if pn_ckpt and config_c_available:
    pn_ft.load_state_dict(torch.load(str(pn_ckpt), map_location=DEVICE, weights_only=True))
    pn_ft.to(DEVICE).eval()
    print(f"    D: Ensemble (PhaseNet {pn_ckpt.name} + GPD EMSC) — loaded")
    config_d_available = True
else:
    print("    D: Ensemble not available")
    config_d_available = False

GPD_WLEN  = getattr(gpd_zs, "in_samples", 400)
GPD_P_IDX = gpd_zs.labels.index("P")

# ── Helper functions ───────────────────────────────────────────────────────
def gpd_sliding_prob(model, data_3ch, wlen, p_idx, stride=10):
    positions = list(range(0, N_TOTAL - wlen + 1, stride))
    crops = []
    for s in positions:
        crop = data_3ch[:, s:s+wlen].astype(np.float32)
        peak = float(np.abs(crop).max())
        if peak > 1e-9: crop = crop / peak
        crops.append(crop)
    probs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(crops), 128):
            batch = torch.tensor(np.stack(crops[i:i+128])).to(DEVICE)
            probs.extend(model(batch)[:, p_idx].cpu().numpy().tolist())
    centers   = np.array([s + wlen // 2 for s in positions], dtype=float)
    probs_arr = np.array(probs, dtype=float)
    f = interp1d(centers, probs_arr, kind="linear",
                 bounds_error=False, fill_value=(probs_arr[0], probs_arr[-1]))
    return f(np.arange(N_TOTAL, dtype=float)).astype(np.float32)

def pn_ann_prob(model, data_3ch):
    st = Stream()
    for i, ch in enumerate(["HHZ", "HHN", "HHE"]):
        tr = Trace(data=data_3ch[i].astype(np.float32))
        tr.stats.network = "KO"; tr.stats.station = "TEST"
        tr.stats.channel = ch; tr.stats.sampling_rate = SRATE
        tr.stats.starttime = T0_REF
        st.append(tr)
    with torch.no_grad():
        ann = model.annotate(st)
    tr_p = next((t for t in ann if t.stats.channel.endswith("_P")), None)
    if tr_p is None: return np.zeros(N_TOTAL, dtype=np.float32)
    tgt = np.arange(N_TOTAL) / SRATE
    src = float(tr_p.stats.starttime - T0_REF) + np.arange(len(tr_p.data)) / tr_p.stats.sampling_rate
    if len(src) < 2: return np.zeros(N_TOTAL, dtype=np.float32)
    f = interp1d(src, tr_p.data.astype(np.float32), kind="linear", bounds_error=False, fill_value=0.0)
    return f(tgt).astype(np.float32)

def detect(prob, thr):
    return float(prob[DETECT_LO:DETECT_HI+1].max()) >= thr

def pick_s(prob, thr):
    w = prob[DETECT_LO:DETECT_HI+1]
    if w.max() < thr: return None
    return (DETECT_LO + int(w.argmax())) / SRATE

# ── Run inference ──────────────────────────────────────────────────────────
print("\n[3] Running inference on test set...")

# Store per-window results for all configs
CONFIGS = {
    "A_gpd_zs":    {"thr": 0.50, "available": True,             "label": "A: GPD (zero-shot)"},
    "B_gpd_taupy": {"thr": 0.50, "available": config_b_available,"label": "B: GPD (fine-tuned, TauPy labels)"},
    "C_gpd_emsc":  {"thr": 0.50, "available": config_c_available,"label": "C: GPD (fine-tuned, EMSC labels)"},
    "D_ensemble":  {"thr": 0.15, "available": config_d_available,"label": "D: Ensemble (PhaseNet+GPD, EMSC)"},
}

results = {k: [] for k in CONFIGS}
hdf5_path = DS_DIR / "waveforms.hdf5"

with h5py.File(hdf5_path, "r") as hf:
    for i, row in test_meta.iterrows():
        name = row["trace_name"]
        if name not in hf["data"]: continue
        data = hf["data"][name][:]
        mag  = row["source_magnitude"]
        ref_p= row["trace_p_arrival_sample"]

        gpd_a = gpd_sliding_prob(gpd_zs,  data, GPD_WLEN, GPD_P_IDX)
        gpd_b = gpd_sliding_prob(gpd_taupy,data,GPD_WLEN, GPD_P_IDX) if config_b_available else None
        gpd_c = gpd_sliding_prob(gpd_emsc, data, GPD_WLEN, GPD_P_IDX) if config_c_available else None
        pn_d  = pn_ann_prob(pn_ft, data) if config_d_available else None
        ens_d = (0.5 * pn_d + 0.5 * gpd_c) if (pn_d is not None and gpd_c is not None) else None

        for cfg_key, prob in [("A_gpd_zs", gpd_a), ("B_gpd_taupy", gpd_b),
                               ("C_gpd_emsc", gpd_c), ("D_ensemble", ens_d)]:
            thr = CONFIGS[cfg_key]["thr"]
            if prob is None or not CONFIGS[cfg_key]["available"]:
                results[cfg_key].append({"mag": float(mag) if pd.notna(mag) else None,
                                         "detected": None, "pick_s": None, "p_mae_s": None})
                continue
            det = detect(prob, thr)
            pk  = pick_s(prob, thr)
            mae = round(abs(pk - ref_p/SRATE), 3) if (pk and pd.notna(ref_p)) else None
            results[cfg_key].append({"mag": float(mag) if pd.notna(mag) else None,
                                     "detected": det, "pick_s": pk, "p_mae_s": mae})

        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(test_meta)}")

print(f"    Inference complete")

# ── Compute metrics ────────────────────────────────────────────────────────
print("\n[4] Computing metrics by magnitude band...")

def compute_band_metrics(cfg_key):
    rows = results[cfg_key]
    df   = pd.DataFrame(rows)
    if not CONFIGS[cfg_key]["available"] or df["detected"].isna().all():
        return {}
    out = {}
    for band, lo, hi in MAG_BANDS:
        mask = (df["mag"].fillna(0) >= lo) & (df["mag"].fillna(0) < hi)
        sub  = df[mask & df["detected"].notna()]
        if len(sub) == 0: continue
        n_det  = int(sub["detected"].sum())
        recall = round(n_det / len(sub), 4)
        mae_v  = sub["p_mae_s"].dropna()
        p_mae  = round(float(mae_v.mean()), 3) if len(mae_v) > 0 else None
        out[band] = {"n": len(sub), "n_detected": n_det, "recall": recall, "p_mae_s": p_mae}
    return out

ablation = {}
for cfg_key, cfg in CONFIGS.items():
    ablation[cfg_key] = {"label": cfg["label"], "threshold": cfg["thr"],
                          "available": cfg["available"],
                          "metrics": compute_band_metrics(cfg_key)}

print(f"\n  {'Config':<40} {'Recall(All)':>12} {'P-MAE(All)':>11}")
print(f"  {'-'*66}")
for cfg_key, d in ablation.items():
    if not d["available"]: continue
    am = d["metrics"].get("All", {})
    print(f"  {d['label']:<40} {am.get('recall','—'):>12}  {am.get('p_mae_s','—'):>10}")

# ── Incremental gain table ─────────────────────────────────────────────────
print("\n  Incremental gains:")
configs_avail = [k for k in ["A_gpd_zs","B_gpd_taupy","C_gpd_emsc","D_ensemble"]
                 if ablation[k]["available"]]
prev_recall = None
for cfg_key in configs_avail:
    am = ablation[cfg_key]["metrics"].get("All", {})
    r  = am.get("recall")
    if r is not None:
        delta = f"{r - prev_recall:+.4f}" if prev_recall is not None else "baseline"
        print(f"    {ablation[cfg_key]['label']:<45} recall={r:.4f}  {delta}")
        prev_recall = r

# ── Figure ─────────────────────────────────────────────────────────────────
print("\n[5] Generating ablation figure...")

BANDS_PLOT  = ["M<2.0", "M2.0-3.0", "M3.0-4.0", "M≥4.0"]
COLORS_CONF = ["#aaaaaa", "#ffaa44", "#00ff88", "#ff6b35"]
x  = np.arange(len(BANDS_PLOT))
n  = len(configs_avail)
w  = 0.18
offsets = np.linspace(-(n-1)/2, (n-1)/2, n) * w

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor("#0a0e1a")
for ax in axes:
    ax.set_facecolor("#0f1525")
    ax.tick_params(colors="#a0aabb")
    for sp in ax.spines.values(): sp.set_edgecolor("#2a3450")
    ax.xaxis.label.set_color("#a0aabb")
    ax.yaxis.label.set_color("#a0aabb")
    ax.title.set_color("#e8ecf0")

for ki, cfg_key in enumerate(configs_avail):
    recalls = [ablation[cfg_key]["metrics"].get(b, {}).get("recall", 0) for b in BANDS_PLOT]
    maes    = [ablation[cfg_key]["metrics"].get(b, {}).get("p_mae_s") or 0 for b in BANDS_PLOT]
    lbl     = ablation[cfg_key]["label"]
    col     = COLORS_CONF[ki % len(COLORS_CONF)]
    axes[0].bar(x + offsets[ki], recalls, w, label=lbl, color=col, alpha=0.85, edgecolor="none")
    axes[1].bar(x + offsets[ki], maes,    w, label=lbl, color=col, alpha=0.85, edgecolor="none")

for ax, title, ylabel in [
    (axes[0], "Recall by Magnitude Band", "Recall"),
    (axes[1], "P-Pick MAE by Magnitude Band", "P-MAE (s)"),
]:
    ax.set_xticks(x)
    ax.set_xticklabels(BANDS_PLOT, color="#a0aabb", fontsize=9)
    ax.set_ylabel(ylabel, color="#a0aabb")
    ax.set_title(title, color="#e8ecf0", fontsize=10)
    ax.legend(fontsize=7, labelcolor="#e8ecf0", facecolor="#1a2035",
              edgecolor="#2a3450", loc="lower right")
axes[0].set_ylim(0, 1.12)
axes[0].axhline(0.5, color="#555", lw=0.7, ls="--")

plt.suptitle("Ablation Study — Pipeline Component Contributions\n"
             "A→B: Fine-tuning  |  B→C: EMSC re-labelling  |  C→D: Ensemble",
             color="#e8ecf0", fontsize=11, y=1.03)
plt.tight_layout()
fig_path = FIG_DIR / "ablation_comparison.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor="#0a0e1a")
plt.close()
print(f"    Saved → {fig_path}")

# ── Save JSON ──────────────────────────────────────────────────────────────
with open(ART / "ablation_results.json", "w") as f:
    json.dump({"phase": "ablation_study", "n_test": len(test_meta),
               "configs": ablation}, f, indent=2)

print("\n" + "=" * 70)
print("ABLATION STUDY — COMPLETE")
print(f"  JSON:   artifacts/ablation_results.json")
print(f"  Figure: figures/ablation_comparison.png")
print("=" * 70)
