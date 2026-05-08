# Phase: Final Evaluation
# Purpose: Evaluate all 7 model variants on the test split:
#          (1) STA/LTA baseline, (2) PhaseNet zero-shot, (3) EQTransformer zero-shot,
#          (4) GPD zero-shot, (5) PhaseNet fine-tuned (phasenet_final.pt),
#          (6) GPD fine-tuned (gpd_final.pt), (7) Ensemble (PhaseNet_final + GPD_final).
#          Produces the definitive comparison with EMSC-corrected P-label references.
# Inputs:  data/augmented_dataset/waveforms.hdf5, data/augmented_dataset/metadata.csv
#          (post script-19 relabelling), models/phasenet_final.pt, models/gpd_final.pt
# Outputs: artifacts/final_evaluation_metrics.json, figures/final_model_comparison.png
# Limitations: Zero-shot models use annotate(); fine-tuned GPD uses manual sliding window
#              (annotate() normalization mismatch as established in script 17).
#              P-MAE now references EMSC analyst picks where available (median error ~0s
#              vs TauPy ~0.46s). STA/LTA P-pick is onset of first trigger window.
#              EQTransformer annotate() runtime is slow (~2-3s/window on CPU).

from pathlib import Path
import json
import time

import h5py
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from obspy import Stream, Trace, UTCDateTime
from obspy.signal.trigger import classic_sta_lta, trigger_onset

import seisbench.models as sbm

ROOT    = Path(__file__).resolve().parent.parent
DS_DIR  = ROOT / "data" / "augmented_dataset"
MDL_DIR = ROOT / "models"
ART     = ROOT / "artifacts"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ── Detection window parameters ────────────────────────────────────────────────
ORIGIN_S   = 30.0    # seconds: our P-label reference sits here (WIN_BEFORE in script 10)
DETECT_TOL = 5.0     # seconds before origin
DETECT_POST= 25.0    # seconds after origin
SRATE      = 100.0
N_TOTAL    = 6000
T0_REF     = UTCDateTime("2023-02-06T01:00:00")

# Thresholds (consistent with script 17)
THR_PHASENET_ZS  = 0.30   # zero-shot (pretrained)
THR_PHASENET_FT  = 0.05   # fine-tuned (calibrated)
THR_EQT_ZS       = 0.10   # zero-shot EQTransformer
THR_GPD_ZS       = 0.50   # zero-shot GPD
THR_GPD_FT       = 0.50   # fine-tuned GPD
THR_ENSEMBLE     = 0.15   # average of PhaseNet_ft + GPD_ft

DETECT_LO = int((ORIGIN_S - DETECT_TOL) * SRATE)   # 2500
DETECT_HI = int((ORIGIN_S + DETECT_POST) * SRATE)  # 5500

# STA/LTA parameters
STA_S, LTA_S = 1.0, 10.0
STALTA_THR   = 3.0

DEVICE = "cuda" if torch.cuda.is_available() else (
         "mps"  if torch.backends.mps.is_available() else "cpu")

print("=" * 70)
print("FINAL EVALUATION — All 7 model variants (EMSC-corrected labels)")
print(f"  Device: {DEVICE}")
print("=" * 70)

# ── Load metadata (post re-labelling) ─────────────────────────────────────────
print("\n[1] Loading test split (post EMSC re-labelling)...")
meta = pd.read_csv(DS_DIR / "metadata.csv")
meta["trace_p_arrival_sample"] = pd.to_numeric(meta["trace_p_arrival_sample"], errors="coerce")
meta["trace_s_arrival_sample"] = pd.to_numeric(meta["trace_s_arrival_sample"], errors="coerce")
meta["source_magnitude"]       = pd.to_numeric(meta["source_magnitude"],       errors="coerce")
if "augmentation" not in meta.columns:
    meta["augmentation"] = "original"

test_meta = meta[
    (meta["split"] == "test") & (meta["augmentation"] == "original")
].copy().reset_index(drop=True)
print(f"    Test windows: {len(test_meta)}")
print(f"    P-labeled   : {test_meta['trace_p_arrival_sample'].notna().sum()}")

# ── Load all models ────────────────────────────────────────────────────────────
print("\n[2] Loading models...")

# Zero-shot
pn_zs  = sbm.PhaseNet.from_pretrained("original").to(DEVICE).eval()
eqt_zs = sbm.EQTransformer.from_pretrained("original").to(DEVICE).eval()
gpd_zs = sbm.GPD.from_pretrained("original").to(DEVICE).eval()
print("    Zero-shot: PhaseNet, EQTransformer, GPD — loaded")

# Fine-tuned (phasenet_final.pt + gpd_final.pt)
pn_ft  = sbm.PhaseNet.from_pretrained("original")
gpd_ft = sbm.GPD.from_pretrained("original")

pn_ckpt  = MDL_DIR / "phasenet_final.pt"
gpd_ckpt = MDL_DIR / "gpd_final.pt"

# Fallback to earlier checkpoints if final not yet produced
if not pn_ckpt.exists():
    pn_ckpt_alt = MDL_DIR / "phasenet_koeri_finetuned.pt"
    if pn_ckpt_alt.exists():
        print(f"    [!] phasenet_final.pt not found — using {pn_ckpt_alt.name}")
        pn_ckpt = pn_ckpt_alt
    else:
        raise FileNotFoundError("No PhaseNet checkpoint found. Run script 20 first.")

if not gpd_ckpt.exists():
    gpd_ckpt_alt = MDL_DIR / "gpd_koeri_finetuned.pt"
    if gpd_ckpt_alt.exists():
        print(f"    [!] gpd_final.pt not found — using {gpd_ckpt_alt.name}")
        gpd_ckpt = gpd_ckpt_alt
    else:
        raise FileNotFoundError("No GPD checkpoint found. Run script 20 first.")

pn_ft.load_state_dict(torch.load(pn_ckpt,  map_location=DEVICE, weights_only=True))
gpd_ft.load_state_dict(torch.load(gpd_ckpt, map_location=DEVICE, weights_only=True))
pn_ft.to(DEVICE).eval()
gpd_ft.to(DEVICE).eval()
print(f"    Fine-tuned: PhaseNet ({pn_ckpt.name}) + GPD ({gpd_ckpt.name}) — loaded")

GPD_WLEN  = getattr(gpd_ft, "in_samples", 400)
GPD_P_IDX = gpd_ft.labels.index("P")

# ── Helper functions ───────────────────────────────────────────────────────────
def stream_from_array(data_3ch, sr=100.0, t0=T0_REF):
    st = Stream()
    for i, ch in enumerate(["HHZ", "HHN", "HHE"]):
        tr = Trace(data=data_3ch[i].astype(np.float32))
        tr.stats.network       = "KO"
        tr.stats.station       = "TEST"
        tr.stats.channel       = ch
        tr.stats.sampling_rate = sr
        tr.stats.starttime     = t0
        st.append(tr)
    return st

def ann_to_prob_array(ann_stream, suffix="_P", t0_ref=T0_REF, n_total=N_TOTAL, sr=SRATE):
    tr = next((t for t in ann_stream if t.stats.channel.endswith(suffix)), None)
    if tr is None:
        return np.zeros(n_total, dtype=np.float32)
    target_times = np.arange(n_total) / sr
    ann_start_s  = float(tr.stats.starttime - t0_ref)
    ann_sr       = tr.stats.sampling_rate
    src_times    = ann_start_s + np.arange(len(tr.data)) / ann_sr
    if len(src_times) < 2:
        return np.zeros(n_total, dtype=np.float32)
    f = interp1d(src_times, tr.data.astype(np.float32),
                 kind="linear", bounds_error=False, fill_value=0.0)
    return f(target_times).astype(np.float32)

def gpd_sliding_prob(model, data_3ch, wlen, p_idx, stride=10, n_total=N_TOTAL):
    """Peak-normalized sliding window — matches fine-tuning pipeline exactly."""
    N         = data_3ch.shape[1]
    positions = list(range(0, N - wlen + 1, stride))
    crops     = []
    for s in positions:
        crop = data_3ch[:, s:s+wlen].astype(np.float32)
        peak = float(np.abs(crop).max())
        if peak > 1e-9:
            crop = crop / peak
        crops.append(crop)
    probs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(crops), 128):
            batch = torch.tensor(np.stack(crops[i:i+128])).to(DEVICE)
            pred  = model(batch)
            probs.extend(pred[:, p_idx].cpu().numpy().tolist())
    centers   = np.array([s + wlen // 2 for s in positions], dtype=float)
    probs_arr = np.array(probs, dtype=float)
    f = interp1d(centers, probs_arr, kind="linear",
                 bounds_error=False, fill_value=(probs_arr[0], probs_arr[-1]))
    return f(np.arange(n_total, dtype=float)).astype(np.float32)

def stalta_prob(data_3ch, sta_s=STA_S, lta_s=LTA_S, sr=SRATE, n_total=N_TOTAL):
    """Run STA/LTA on the vertical channel; return normalized ratio as pseudo-prob."""
    nsta = int(sta_s * sr)
    nlta = int(lta_s * sr)
    cf   = classic_sta_lta(data_3ch[0], nsta, nlta)
    # Normalize to [0,1] using STALTA_THR as scale
    return np.clip(cf / (2 * STALTA_THR), 0.0, 1.0).astype(np.float32)

def detect_and_pick(prob_trace, threshold):
    """Returns (detected, pick_s, peak_prob)."""
    window = prob_trace[DETECT_LO:DETECT_HI + 1]
    peak   = float(window.max())
    if peak < threshold:
        return False, None, peak
    idx       = int(window.argmax())
    pick_s    = (DETECT_LO + idx) / SRATE
    return True, round(pick_s, 3), round(peak, 4)

# ── Run inference ──────────────────────────────────────────────────────────────
print("\n[3] Running inference on test set...")

MODEL_NAMES = [
    "stalta", "phasenet_zs", "eqt_zs", "gpd_zs", "phasenet_ft", "gpd_ft", "ensemble"
]
results = {m: [] for m in MODEL_NAMES}

hdf5_path = DS_DIR / "waveforms.hdf5"
t_start = time.time()

with h5py.File(hdf5_path, "r") as hf:
    for i, row in test_meta.iterrows():
        name       = row["trace_name"]
        if name not in hf["data"]:
            continue
        data       = hf["data"][name][:]          # (3, 6000)
        mag        = row["source_magnitude"]
        theo_p     = row["trace_p_arrival_sample"]   # EMSC-corrected where available

        stream = stream_from_array(data)

        # ── STA/LTA ──────────────────────────────────────────────────────────
        sl_prob = stalta_prob(data)
        det, pick_s, peak = detect_and_pick(sl_prob, STALTA_THR / (2 * STALTA_THR))
        p_mae = round(abs(pick_s - theo_p / SRATE), 3) if (pick_s is not None and pd.notna(theo_p)) else None
        results["stalta"].append({
            "trace_name": name, "magnitude": float(mag) if pd.notna(mag) else None,
            "detected": det, "pick_s": pick_s, "peak_prob": peak, "p_mae_s": p_mae
        })

        # ── PhaseNet zero-shot ────────────────────────────────────────────────
        with torch.no_grad():
            ann = pn_zs.annotate(stream)
        pn_zs_p = ann_to_prob_array(ann, suffix="_P")
        det, pick_s, peak = detect_and_pick(pn_zs_p, THR_PHASENET_ZS)
        p_mae = round(abs(pick_s - theo_p / SRATE), 3) if (pick_s is not None and pd.notna(theo_p)) else None
        results["phasenet_zs"].append({
            "trace_name": name, "magnitude": float(mag) if pd.notna(mag) else None,
            "detected": det, "pick_s": pick_s, "peak_prob": peak, "p_mae_s": p_mae
        })

        # ── EQTransformer zero-shot ───────────────────────────────────────────
        with torch.no_grad():
            ann = eqt_zs.annotate(stream)
        eqt_p = ann_to_prob_array(ann, suffix="_P")
        det, pick_s, peak = detect_and_pick(eqt_p, THR_EQT_ZS)
        p_mae = round(abs(pick_s - theo_p / SRATE), 3) if (pick_s is not None and pd.notna(theo_p)) else None
        results["eqt_zs"].append({
            "trace_name": name, "magnitude": float(mag) if pd.notna(mag) else None,
            "detected": det, "pick_s": pick_s, "peak_prob": peak, "p_mae_s": p_mae
        })

        # ── GPD zero-shot ─────────────────────────────────────────────────────
        gpd_zs_p = gpd_sliding_prob(gpd_zs, data, GPD_WLEN, gpd_zs.labels.index("P"))
        det, pick_s, peak = detect_and_pick(gpd_zs_p, THR_GPD_ZS)
        p_mae = round(abs(pick_s - theo_p / SRATE), 3) if (pick_s is not None and pd.notna(theo_p)) else None
        results["gpd_zs"].append({
            "trace_name": name, "magnitude": float(mag) if pd.notna(mag) else None,
            "detected": det, "pick_s": pick_s, "peak_prob": peak, "p_mae_s": p_mae
        })

        # ── PhaseNet fine-tuned ───────────────────────────────────────────────
        with torch.no_grad():
            ann = pn_ft.annotate(stream)
        pn_ft_p = ann_to_prob_array(ann, suffix="_P")
        det, pick_s, peak = detect_and_pick(pn_ft_p, THR_PHASENET_FT)
        p_mae = round(abs(pick_s - theo_p / SRATE), 3) if (pick_s is not None and pd.notna(theo_p)) else None
        results["phasenet_ft"].append({
            "trace_name": name, "magnitude": float(mag) if pd.notna(mag) else None,
            "detected": det, "pick_s": pick_s, "peak_prob": peak, "p_mae_s": p_mae
        })

        # ── GPD fine-tuned (manual sliding window) ────────────────────────────
        gpd_ft_p = gpd_sliding_prob(gpd_ft, data, GPD_WLEN, GPD_P_IDX)
        det, pick_s, peak = detect_and_pick(gpd_ft_p, THR_GPD_FT)
        p_mae = round(abs(pick_s - theo_p / SRATE), 3) if (pick_s is not None and pd.notna(theo_p)) else None
        results["gpd_ft"].append({
            "trace_name": name, "magnitude": float(mag) if pd.notna(mag) else None,
            "detected": det, "pick_s": pick_s, "peak_prob": peak, "p_mae_s": p_mae
        })

        # ── Ensemble ──────────────────────────────────────────────────────────
        ens_p = 0.5 * pn_ft_p + 0.5 * gpd_ft_p
        det, pick_s, peak = detect_and_pick(ens_p, THR_ENSEMBLE)
        p_mae = round(abs(pick_s - theo_p / SRATE), 3) if (pick_s is not None and pd.notna(theo_p)) else None
        results["ensemble"].append({
            "trace_name": name, "magnitude": float(mag) if pd.notna(mag) else None,
            "detected": det, "pick_s": pick_s, "peak_prob": peak, "p_mae_s": p_mae
        })

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t_start
            print(f"      {i+1}/{len(test_meta)}  ({elapsed:.0f}s elapsed)")

print(f"    Inference complete: {len(results['ensemble'])} windows in "
      f"{time.time()-t_start:.0f}s")

# ── Compute metrics ────────────────────────────────────────────────────────────
print("\n[4] Computing metrics...")

MAG_BANDS = [
    ("All",       -np.inf,  np.inf),
    ("M<2.0",     -np.inf,  2.0),
    ("M2.0-3.0",   2.0,     3.0),
    ("M3.0-4.0",   3.0,     4.0),
    ("M≥4.0",      4.0,    np.inf),
]

def compute_metrics(result_rows):
    df = pd.DataFrame(result_rows)
    out = {}
    for band, lo, hi in MAG_BANDS:
        mask = (df["magnitude"].fillna(0) >= lo) & (df["magnitude"].fillna(0) < hi)
        sub  = df[mask]
        if len(sub) == 0:
            continue
        n_det    = int(sub["detected"].sum())
        recall   = round(n_det / len(sub), 4)
        mae_vals = sub["p_mae_s"].dropna()
        p_mae    = round(float(mae_vals.mean()), 3) if len(mae_vals) > 0 else None
        out[band] = {
            "n_windows": len(sub),
            "n_detected": n_det,
            "recall": recall,
            "p_mae_s": p_mae,
        }
    return out

all_metrics = {m: compute_metrics(results[m]) for m in MODEL_NAMES}

MODEL_LABELS = {
    "stalta":       "STA/LTA",
    "phasenet_zs":  "PhaseNet (zero-shot)",
    "eqt_zs":       "EQTransformer (zero-shot)",
    "gpd_zs":       "GPD (zero-shot)",
    "phasenet_ft":  "PhaseNet (fine-tuned)",
    "gpd_ft":       "GPD (fine-tuned)",
    "ensemble":     "Ensemble (PhaseNet+GPD)",
}

print("\n  Summary (all test windows):")
print(f"  {'Model':<30}  {'Recall':>7}  {'Detected':>10}  {'P-MAE(s)':>9}")
print("  " + "-" * 65)
for m in MODEL_NAMES:
    am = all_metrics[m].get("All", {})
    print(f"  {MODEL_LABELS[m]:<30}  "
          f"{am.get('recall','—'):>7}  "
          f"{am.get('n_detected','—'):>4}/{am.get('n_windows','—'):<5}  "
          f"{am.get('p_mae_s','—'):>9}")

# ── Save JSON ──────────────────────────────────────────────────────────────────
print("\n[5] Saving final_evaluation_metrics.json...")
out = {
    "phase":       "final_evaluation",
    "label_source": "EMSC analyst picks (script 19) + TauPy fallback",
    "n_test_windows": len(results["ensemble"]),
    "thresholds": {
        "stalta":       STALTA_THR,
        "phasenet_zs":  THR_PHASENET_ZS,
        "eqt_zs":       THR_EQT_ZS,
        "gpd_zs":       THR_GPD_ZS,
        "phasenet_ft":  THR_PHASENET_FT,
        "gpd_ft":       THR_GPD_FT,
        "ensemble":     THR_ENSEMBLE,
    },
}
for m in MODEL_NAMES:
    out[MODEL_LABELS[m]] = all_metrics[m]

json_path = ART / "final_evaluation_metrics.json"
with open(json_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"    Saved → {json_path}")

# ── Figure ─────────────────────────────────────────────────────────────────────
print("\n[6] Generating final_model_comparison.png...")

BANDS_PLOT  = ["M<2.0", "M2.0-3.0", "M3.0-4.0", "M≥4.0"]
MODEL_PLOT  = ["stalta", "phasenet_zs", "eqt_zs", "gpd_zs",
               "phasenet_ft", "gpd_ft", "ensemble"]
COLORS      = {
    "stalta":      "#aaaaaa",
    "phasenet_zs": "#4488ff",
    "eqt_zs":      "#ff44aa",
    "gpd_zs":      "#ffaa44",
    "phasenet_ft": "#00d4ff",
    "gpd_ft":      "#00ff88",
    "ensemble":    "#ff6b35",
}

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor("#0a0e1a")
for ax in axes:
    ax.set_facecolor("#0f1525")
    ax.tick_params(colors="#a0aabb")
    ax.xaxis.label.set_color("#a0aabb")
    ax.yaxis.label.set_color("#a0aabb")
    ax.title.set_color("#e8ecf0")
    for sp in ax.spines.values():
        sp.set_edgecolor("#2a3450")

x  = np.arange(len(BANDS_PLOT))
n  = len(MODEL_PLOT)
w  = 0.10
offsets = np.linspace(-(n-1)/2, (n-1)/2, n) * w

# Recall
ax = axes[0]
for ki, m in enumerate(MODEL_PLOT):
    recalls = [all_metrics[m].get(b, {}).get("recall", 0) for b in BANDS_PLOT]
    ax.bar(x + offsets[ki], recalls, w, label=MODEL_LABELS[m],
           color=COLORS[m], alpha=0.85, edgecolor="none")
ax.set_xticks(x)
ax.set_xticklabels(BANDS_PLOT, color="#a0aabb", fontsize=9)
ax.set_ylim(0, 1.12)
ax.set_ylabel("Recall", color="#a0aabb")
ax.set_title("Recall by Magnitude Band", color="#e8ecf0", fontsize=11)
ax.axhline(0.5, color="#555", lw=0.7, ls="--", label="_hline")
ax.legend(fontsize=7.5, labelcolor="#e8ecf0",
          facecolor="#1a2035", edgecolor="#2a3450", ncol=2,
          loc="upper right")

# P-MAE
ax = axes[1]
for ki, m in enumerate(MODEL_PLOT):
    maes = [all_metrics[m].get(b, {}).get("p_mae_s") or 0 for b in BANDS_PLOT]
    ax.bar(x + offsets[ki], maes, w, label=MODEL_LABELS[m],
           color=COLORS[m], alpha=0.85, edgecolor="none")
ax.set_xticks(x)
ax.set_xticklabels(BANDS_PLOT, color="#a0aabb", fontsize=9)
ax.set_ylabel("P-pick MAE (seconds)", color="#a0aabb")
ax.set_title("P-Pick MAE by Magnitude Band\n(EMSC-corrected reference)",
             color="#e8ecf0", fontsize=11)
ax.legend(fontsize=7.5, labelcolor="#e8ecf0",
          facecolor="#1a2035", edgecolor="#2a3450", ncol=2,
          loc="upper right")

plt.suptitle("Final Model Comparison — Seismic Event Detection\n"
             "KOERI Kahramanmaraş 2023 Aftershock Sequence",
             color="#e8ecf0", fontsize=12, y=1.02)
plt.tight_layout()
fig_path = FIG_DIR / "final_model_comparison.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor="#0a0e1a")
plt.close()
print(f"    Saved → {fig_path}")

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("FINAL EVALUATION — COMPLETE")
print(f"  Test windows    : {len(results['ensemble'])}")
print(f"  JSON            : artifacts/final_evaluation_metrics.json")
print(f"  Figure          : figures/final_model_comparison.png")
ensemble_all = all_metrics["ensemble"].get("All", {})
pn_ft_all    = all_metrics["phasenet_ft"].get("All", {})
print(f"\n  Best model (Ensemble): recall={ensemble_all.get('recall','—')}  "
      f"P-MAE={ensemble_all.get('p_mae_s','—')}s")
print(f"  PhaseNet (ft) alone : recall={pn_ft_all.get('recall','—')}  "
      f"P-MAE={pn_ft_all.get('p_mae_s','—')}s")
print(f"\n  === FINAL PIPELINE COMPLETE ===")
print(f"  → Update Streamlit app.py with ensemble model support")
print("=" * 70)
