# Phase: Fine-tuning evaluation
# Purpose: Ensemble evaluation — average P-pick probability traces from fine-tuned
#          PhaseNet and fine-tuned GPD, then compare recall/precision/F1/MAE for:
#          (1) PhaseNet fine-tuned alone, (2) GPD fine-tuned alone, (3) Ensemble.
#          Uses original test-split windows from data/augmented_dataset/.
# Inputs:  data/augmented_dataset/waveforms.hdf5, data/augmented_dataset/metadata.csv,
#          models/phasenet_koeri_finetuned.pt, models/gpd_koeri_finetuned.pt
# Outputs: artifacts/ensemble_evaluation_metrics.json, figures/ensemble_comparison.png
# Limitations: GPD annotate() uses sliding 400-sample windows (stride auto-set by
#              SeisBench); ensemble is simple average of P-prob traces, no learned weights.
#              P-MAE uses TauPy theoretical picks as reference (same caveat as Phase 4/5).

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
FIG_DIR.mkdir(exist_ok=True)

# ── Detection parameters (identical to script 14) ─────────────────────────────
ORIGIN_S     = 30.0    # seconds from window start where event origin sits
DETECT_TOL   = 5.0     # seconds before origin counted as detection
DETECT_POST  = 25.0    # seconds after origin counted as detection
SRATE        = 100.0
N_TOTAL      = 6000    # full window length in samples
T0_REF       = UTCDateTime("2023-02-06T01:00:00")

# Thresholds — tuned per model
THR_PHASENET = 0.05    # fine-tuned PhaseNet (calibrated in script 14v4b)
THR_GPD      = 0.50    # GPD: default midpoint for 3-class softmax
THR_ENSEMBLE = 0.15    # averaged probability threshold

DEVICE = "cuda" if torch.cuda.is_available() else (
         "mps"  if torch.backends.mps.is_available() else "cpu")

print("=" * 70)
print("ENSEMBLE EVALUATION — PhaseNet (ft) + GPD (ft)")
print("=" * 70)

# ── Load models ────────────────────────────────────────────────────────────────
print("\n[1] Loading models...")

pn_ckpt  = MDL_DIR / "phasenet_koeri_finetuned.pt"
gpd_ckpt = MDL_DIR / "gpd_koeri_finetuned.pt"

for ckpt, name in [(pn_ckpt, "PhaseNet"), (gpd_ckpt, "GPD")]:
    if not ckpt.exists():
        raise FileNotFoundError(
            f"{name} checkpoint not found: {ckpt}\n"
            f"Run the corresponding fine-tuning script first.")

phasenet = sbm.PhaseNet.from_pretrained("original")
phasenet.load_state_dict(torch.load(pn_ckpt, map_location=DEVICE))
phasenet.to(DEVICE).eval()
print("    PhaseNet fine-tuned — loaded")

gpd = sbm.GPD.from_pretrained("original")
gpd.load_state_dict(torch.load(gpd_ckpt, map_location=DEVICE))
gpd.to(DEVICE).eval()
print("    GPD fine-tuned      — loaded")

GPD_WLEN  = getattr(gpd, "in_samples", 400)
GPD_P_IDX = gpd.labels.index("P")
print(f"    GPD in_samples={GPD_WLEN}, P_idx={GPD_P_IDX}")

# ── Load test split ────────────────────────────────────────────────────────────
print("\n[2] Loading test split from augmented_dataset...")
meta = pd.read_csv(DS_DIR / "metadata.csv")
meta["trace_p_arrival_sample"] = pd.to_numeric(meta["trace_p_arrival_sample"], errors="coerce")
meta["trace_s_arrival_sample"] = pd.to_numeric(meta["trace_s_arrival_sample"], errors="coerce")
meta["source_magnitude"]       = pd.to_numeric(meta["source_magnitude"],       errors="coerce")
if "augmentation" not in meta.columns:
    meta["augmentation"] = "original"

test_meta = meta[
    (meta["split"] == "test") & (meta["augmentation"] == "original")
].copy().reset_index(drop=True)
print(f"    Test windows (original): {len(test_meta)}")

# ── Helper: build ObsPy stream ─────────────────────────────────────────────────
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

# ── Helper: align PhaseNet annotation trace to fixed 6000-sample grid ─────────
def ann_to_prob_array(ann_stream, suffix="_P", t0_ref=T0_REF,
                      n_total=N_TOTAL, sr=SRATE):
    """
    Find the annotation trace whose channel ends with suffix and map it onto a
    zero-padded array of length n_total. The annotation may have a different
    start time and sampling rate from the input — we use the trace's own
    time axis for mapping.
    """
    tr = next((t for t in ann_stream if t.stats.channel.endswith(suffix)), None)
    if tr is None:
        return np.zeros(n_total, dtype=np.float32)

    # Build target time grid (in seconds from t0_ref)
    target_times = np.arange(n_total) / sr
    # Build source time grid for annotation trace
    ann_start_s = float(tr.stats.starttime - t0_ref)
    ann_sr      = tr.stats.sampling_rate
    src_times   = ann_start_s + np.arange(len(tr.data)) / ann_sr

    # Clamp source to valid range before interpolating
    if len(src_times) < 2:
        return np.zeros(n_total, dtype=np.float32)

    f = interp1d(src_times, tr.data.astype(np.float32),
                 kind="linear", bounds_error=False, fill_value=0.0)
    return f(target_times).astype(np.float32)

# ── Helper: GPD fine-tuned sliding window (uses same peak-norm as training) ────
_GPD_WLEN   = None   # set after model load
_GPD_P_IDX  = None

def gpd_sliding_prob(model, data_3ch, wlen, p_idx, stride=10, device="cpu",
                     n_total=N_TOTAL):
    """
    Slide the fine-tuned GPD window over the full (3, N) trace with peak-
    normalisation per crop — matching exactly the training data pipeline.
    Returns a (n_total,) P-probability trace at 100 Hz (interpolated).
    Using annotate() is intentionally avoided: SeisBench applies different
    normalisation than training, causing reversed P/N outputs.
    """
    N       = data_3ch.shape[1]
    positions = list(range(0, N - wlen + 1, stride))

    # Build normalised crops
    crops = []
    for s in positions:
        crop = data_3ch[:, s:s+wlen].astype(np.float32)
        peak = float(np.abs(crop).max())
        if peak > 1e-9:
            crop = crop / peak
        crops.append(crop)

    # Batch forward pass
    probs = []
    model.eval()
    bsz = 128
    with torch.no_grad():
        for i in range(0, len(crops), bsz):
            batch = torch.tensor(np.stack(crops[i:i+bsz])).to(device)
            pred  = model(batch)            # (B, 3)
            probs.extend(pred[:, p_idx].cpu().numpy().tolist())

    centers   = np.array([s + wlen // 2 for s in positions], dtype=float)
    probs_arr = np.array(probs, dtype=float)

    # Interpolate to n_total samples at 100 Hz
    target = np.arange(n_total, dtype=float)
    f = interp1d(centers, probs_arr, kind="linear",
                 bounds_error=False,
                 fill_value=(probs_arr[0], probs_arr[-1]))
    return f(target).astype(np.float32)

# ── Helper: detect + pick from a probability trace ────────────────────────────
DETECT_LO = int((ORIGIN_S - DETECT_TOL) * SRATE)   # sample 2500
DETECT_HI = int((ORIGIN_S + DETECT_POST) * SRATE)  # sample 5500

def detect_and_pick(prob_trace, threshold):
    """
    Returns (detected: bool, pick_s: float|None, peak_prob: float, p_mae_s: float|None).
    pick_s and p_mae_s are relative to window start (in seconds).
    """
    window = prob_trace[DETECT_LO:DETECT_HI + 1]
    peak   = float(window.max())
    if peak < threshold:
        return False, None, peak, None
    idx_in_window = int(window.argmax())
    pick_sample   = DETECT_LO + idx_in_window
    pick_s        = pick_sample / SRATE
    return True, round(pick_s, 3), round(peak, 4), None

# ── Run inference ──────────────────────────────────────────────────────────────
print("\n[3] Running inference on test set...")

rows_pn  = []   # PhaseNet fine-tuned alone
rows_gpd = []   # GPD fine-tuned alone
rows_ens = []   # Ensemble

hdf5_path = DS_DIR / "waveforms.hdf5"

with h5py.File(hdf5_path, "r") as hf:
    for i, row in test_meta.iterrows():
        name  = row["trace_name"]
        if name not in hf["data"]:
            continue

        data  = hf["data"][name][:]          # (3, 6000)
        mag   = row["source_magnitude"]
        theo_p_samp = row["trace_p_arrival_sample"]  # NaN if unavailable

        stream = stream_from_array(data)

        with torch.no_grad():
            ann_pn = phasenet.annotate(stream)

        # PhaseNet: use annotate() (its normalization matches pretraining)
        pn_p = ann_to_prob_array(ann_pn, suffix="_P")

        # GPD: manual sliding window with peak-norm (matches fine-tuning pipeline)
        # annotate() intentionally NOT used — different normalization causes
        # P/N output inversion with our fine-tuned weights.
        gpd_p = gpd_sliding_prob(gpd, data, GPD_WLEN, GPD_P_IDX, device=DEVICE)

        ens_p = 0.5 * pn_p + 0.5 * gpd_p

        # ── Per-model detection + pick ──────────────────────────────────────
        for prob, thr, result_list in [
            (pn_p,  THR_PHASENET, rows_pn),
            (gpd_p, THR_GPD,      rows_gpd),
            (ens_p, THR_ENSEMBLE, rows_ens),
        ]:
            detected, pick_s, peak_prob, _ = detect_and_pick(prob, thr)

            # P-MAE vs TauPy
            p_mae = None
            if pick_s is not None and not np.isnan(theo_p_samp):
                p_mae = round(abs(pick_s - theo_p_samp / SRATE), 3)

            result_list.append({
                "trace_name":  name,
                "magnitude":   float(mag) if not np.isnan(mag) else None,
                "detected":    detected,
                "pick_s":      pick_s,
                "peak_prob":   peak_prob,
                "p_mae_s":     p_mae,
            })

        if (i + 1) % 100 == 0:
            print(f"      {i+1}/{len(test_meta)} done...")

n_proc = len(rows_ens)
print(f"    Inference complete: {n_proc} windows")

# ── Compute metrics ────────────────────────────────────────────────────────────
print("\n[4] Computing metrics...")

MAG_BANDS = [
    ("All",      -np.inf,  np.inf),
    ("M<2.0",    -np.inf,  2.0),
    ("M2.0-3.0",  2.0,     3.0),
    ("M3.0-4.0",  3.0,     4.0),
    ("M≥4.0",     4.0,    np.inf),
]

def compute_metrics(result_rows):
    df = pd.DataFrame(result_rows)
    out = {}
    for band, lo, hi in MAG_BANDS:
        mask = (df["magnitude"].fillna(0) >= lo) & (df["magnitude"].fillna(0) < hi)
        sub  = df[mask]
        if len(sub) == 0:
            continue
        n_det  = int(sub["detected"].sum())
        recall = round(n_det / len(sub), 4)
        mae_vals = sub["p_mae_s"].dropna()
        p_mae    = round(float(mae_vals.mean()), 3) if len(mae_vals) > 0 else None
        out[band] = {
            "n_windows": len(sub),
            "n_detected": n_det,
            "recall": recall,
            "p_mae_s": p_mae,
        }
    return out

metrics_pn  = compute_metrics(rows_pn)
metrics_gpd = compute_metrics(rows_gpd)
metrics_ens = compute_metrics(rows_ens)

for label, m in [("PhaseNet (ft)", metrics_pn),
                  ("GPD (ft)",      metrics_gpd),
                  ("Ensemble",      metrics_ens)]:
    all_m = m.get("All", {})
    print(f"  {label:20s}  recall={all_m.get('recall','—')}"
          f"  det={all_m.get('n_detected','—')}/{all_m.get('n_windows','—')}"
          f"  P-MAE={all_m.get('p_mae_s','—')}s")

# ── Save JSON ──────────────────────────────────────────────────────────────────
out_json = {
    "phase":       "ensemble_evaluation",
    "n_test":      n_proc,
    "thresholds":  {
        "phasenet_ft": THR_PHASENET,
        "gpd_ft":      THR_GPD,
        "ensemble":    THR_ENSEMBLE,
    },
    "phasenet_finetuned": metrics_pn,
    "gpd_finetuned":      metrics_gpd,
    "ensemble":           metrics_ens,
}
json_path = ART / "ensemble_evaluation_metrics.json"
with open(json_path, "w") as f:
    json.dump(out_json, f, indent=2)
print(f"\n    Saved → {json_path}")

# ── Figure ─────────────────────────────────────────────────────────────────────
print("\n[5] Generating figure...")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor("#0a0e1a")
for ax in axes:
    ax.set_facecolor("#0f1525")
    ax.tick_params(colors="#a0aabb")
    ax.xaxis.label.set_color("#a0aabb")
    ax.yaxis.label.set_color("#a0aabb")
    ax.title.set_color("#e8ecf0")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a3450")

bands_plot = ["M<2.0", "M2.0-3.0", "M3.0-4.0", "M≥4.0"]
colors     = {"PhaseNet (ft)": "#00d4ff", "GPD (ft)": "#00ff88", "Ensemble": "#ff6b35"}
x          = np.arange(len(bands_plot))
w          = 0.25

# Recall by magnitude band
ax = axes[0]
for ki, (label, m) in enumerate([("PhaseNet (ft)", metrics_pn),
                                   ("GPD (ft)",      metrics_gpd),
                                   ("Ensemble",      metrics_ens)]):
    recalls = [m.get(b, {}).get("recall", 0) for b in bands_plot]
    ax.bar(x + (ki - 1) * w, recalls, w, label=label, color=colors[label],
           alpha=0.85, edgecolor="none")

ax.set_xticks(x)
ax.set_xticklabels(bands_plot, color="#a0aabb", fontsize=9)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Recall", color="#a0aabb")
ax.set_title("Recall by Magnitude Band", color="#e8ecf0")
ax.legend(fontsize=8, labelcolor="#e8ecf0",
          facecolor="#1a2035", edgecolor="#2a3450")
ax.axhline(0.5, color="#555", lw=0.7, ls="--")

# P-MAE comparison
ax = axes[1]
for ki, (label, m) in enumerate([("PhaseNet (ft)", metrics_pn),
                                   ("GPD (ft)",      metrics_gpd),
                                   ("Ensemble",      metrics_ens)]):
    maes = [m.get(b, {}).get("p_mae_s") or 0 for b in bands_plot]
    ax.bar(x + (ki - 1) * w, maes, w, label=label, color=colors[label],
           alpha=0.85, edgecolor="none")

ax.set_xticks(x)
ax.set_xticklabels(bands_plot, color="#a0aabb", fontsize=9)
ax.set_ylabel("P-pick MAE (s)", color="#a0aabb")
ax.set_title("P-Pick MAE by Magnitude Band", color="#e8ecf0")
ax.legend(fontsize=8, labelcolor="#e8ecf0",
          facecolor="#1a2035", edgecolor="#2a3450")

plt.suptitle("Ensemble vs Individual Fine-tuned Models", color="#e8ecf0",
             fontsize=12, y=1.02)
plt.tight_layout()
fig_path = FIG_DIR / "ensemble_comparison.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor="#0a0e1a")
plt.close()
print(f"    Saved → {fig_path}")

print("\n" + "=" * 70)
print("ENSEMBLE EVALUATION — COMPLETE")
print(f"  Metrics : artifacts/ensemble_evaluation_metrics.json")
print(f"  Figure  : figures/ensemble_comparison.png")
print("=" * 70)
