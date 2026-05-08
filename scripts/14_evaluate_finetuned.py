# Phase: Fine-tuning
# Purpose: Compare pretrained PhaseNet (zero-shot) vs fine-tuned PhaseNet on the
#          held-out test split. Computes recall, precision, F1, and FP/hour for
#          both models, stratified by magnitude band. Produces a side-by-side
#          comparison figure and saves all metrics to JSON.
# Inputs:  data/finetune_dataset/waveforms.hdf5, data/finetune_dataset/metadata.csv,
#          artifacts/finetune_split.json, models/phasenet_koeri_finetuned.pt
# Outputs: artifacts/finetune_evaluation_metrics.json,
#          figures/finetuned_vs_pretrained_comparison.png
# Limitations: Test set is 15% of ~5k windows (~750 samples). Precision uses noise
#              windows from the pilot dataset only (Phase 5 has none). FP/hour
#              estimated from pilot noise windows in test split; confidence intervals
#              will be wide given the small noise window count. Picking MAE uses
#              TauPy theoretical P as reference (same as training labels).

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from obspy import Stream, Trace, UTCDateTime
import seisbench.models as sbm
import seisbench.data as sbd

import os

ROOT    = Path(__file__).resolve().parent.parent
DS_DIR  = ROOT / "data" / "finetune_dataset"
ART     = ROOT / "artifacts"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

# Allow overriding output filenames via env vars (used for v2 run)
METRICS_OUT = os.environ.get("EVAL_METRICS_OUT", "finetune_evaluation_metrics.json")
FIGURE_OUT  = os.environ.get("EVAL_FIGURE_OUT",  "finetuned_vs_pretrained_comparison.png")

P_THRESHOLD = 0.3
ORIGIN_S    = 30.0
DETECT_TOL  = 5.0
DETECT_POST = 25.0
SRATE       = 100.0

DEVICE = "cuda" if torch.cuda.is_available() else (
         "mps"  if torch.backends.mps.is_available() else "cpu")

print("=" * 70)
print("EVALUATION — Pretrained vs Fine-tuned PhaseNet")
print("=" * 70)

# ── Step 1: Load test split metadata ──────────────────────────────────────────
print("\n[1] Loading test split...")

with open(ART / "finetune_split.json") as f:
    split_info = json.load(f)
test_names = set(split_info["test_trace_names"])

meta = pd.read_csv(DS_DIR / "metadata.csv")
meta["source_magnitude"] = pd.to_numeric(meta["source_magnitude"], errors="coerce")
test_meta = meta[meta["trace_name"].isin(test_names)].copy().reset_index(drop=True)
print(f"    Test windows: {len(test_meta)}")
print(f"    Event windows: {(test_meta['window_type']=='event').sum()}")
print(f"    Noise windows: {(test_meta['window_type']=='noise').sum()}")

# ── Step 2: Load both models ───────────────────────────────────────────────────
print("\n[2] Loading models...")

pretrained = sbm.PhaseNet.from_pretrained("original")
pretrained.to(DEVICE).eval()
print("    PhaseNet pretrained (original) — loaded")

ckpt_path = ROOT / "models" / "phasenet_koeri_finetuned.pt"
if not ckpt_path.exists():
    raise FileNotFoundError(
        f"Fine-tuned checkpoint not found: {ckpt_path}\n"
        "Run scripts/13_finetune_phasenet.py first."
    )
finetuned = sbm.PhaseNet.from_pretrained("original")   # load architecture
finetuned.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
finetuned.to(DEVICE).eval()
print("    PhaseNet fine-tuned (KOERI) — loaded")

# ── Step 3: Load waveforms and run inference ───────────────────────────────────
print("\n[3] Running inference on test set...")

import h5py
hdf5_path = DS_DIR / "waveforms.hdf5"

def stream_from_array(data_3ch, sampling_rate=100.0):
    """Build an ObsPy Stream from a (3, N) float32 array for model.annotate()."""
    t0  = UTCDateTime("2023-02-06T01:00:00")
    st  = Stream()
    for i, ch in enumerate(["HHZ", "HHN", "HHE"]):
        tr = Trace(data=data_3ch[i].astype(np.float32))
        tr.stats.network  = "KO"
        tr.stats.station  = "TEST"
        tr.stats.channel  = ch
        tr.stats.sampling_rate = sampling_rate
        tr.stats.starttime = t0
        st.append(tr)
    return st

def pick_from_annotation(ann_stream, t0_utc):
    """Extract P pick time (seconds from window start) from annotated stream."""
    p_tr = next((tr for tr in ann_stream if tr.stats.channel.endswith("_P")), None)
    if p_tr is None or p_tr.data.max() < P_THRESHOLD:
        return None, 0.0
    t0_ann = p_tr.stats.starttime - t0_utc
    idx = p_tr.data.argmax()
    pick_s = float(t0_ann + idx / p_tr.stats.sampling_rate)
    return round(pick_s, 3), round(float(p_tr.data.max()), 4)

t0_ref = UTCDateTime("2023-02-06T01:00:00")

results_pre  = []   # pretrained results
results_fine = []   # finetuned results

with h5py.File(hdf5_path, "r") as hf:
    for i, row in test_meta.iterrows():
        name = row["trace_name"]
        if name not in hf["data"]:
            continue

        data = hf["data"][name][:]         # (3, 6000)
        mag  = row["source_magnitude"]
        wtype = row["window_type"]

        theo_p = row["trace_p_arrival_sample"]   # NaN if unavailable
        theo_s = row["trace_s_arrival_sample"]

        stream = stream_from_array(data)

        for model_obj, result_list in [(pretrained, results_pre),
                                        (finetuned,  results_fine)]:
            with torch.no_grad():
                ann = model_obj.annotate(stream)

            p_t, p_prob = pick_from_annotation(ann, t0_ref)
            s_tr = next((tr for tr in ann if tr.stats.channel.endswith("_S")), None)
            s_t = s_prob = None
            if s_tr is not None and s_tr.data.max() >= P_THRESHOLD:
                t0_ann = s_tr.stats.starttime - t0_ref
                idx = s_tr.data.argmax()
                s_t = round(float(t0_ann + idx / s_tr.stats.sampling_rate), 3)
                s_prob = round(float(s_tr.data.max()), 4)

            # Detection: P pick in [origin-5s, origin+25s]
            detected = (p_t is not None and
                        (ORIGIN_S - DETECT_TOL) <= p_t <= (ORIGIN_S + DETECT_POST))

            # P-pick MAE vs TauPy theoretical (only where both available)
            p_mae = None
            if p_t is not None and not np.isnan(theo_p):
                p_mae = abs(p_t - theo_p / SRATE)   # theo_p is in samples

            result_list.append({
                "trace_name":  name,
                "magnitude":   float(mag) if not np.isnan(mag) else None,
                "window_type": wtype,
                "detected":    detected,
                "p_pick_s":    p_t,
                "s_pick_s":    s_t,
                "p_prob":      p_prob,
                "p_mae_s":     p_mae,
            })

        if (i + 1) % 100 == 0:
            print(f"      {i+1}/{len(test_meta)} done...")

print(f"    Inference complete: {len(results_pre)} windows")

# ── Step 4: Compute evaluation metrics ────────────────────────────────────────
print("\n[4] Computing metrics...")

bands = [
    (2.0, 3.0, "M 2.0-3.0"),
    (3.0, 4.0, "M 3.0-4.0"),
    (4.0, 99., "M>=4.0"),
]

WINDOW_LEN_S = 60.0
MODEL_NAMES  = {"pretrained": "PhaseNet (zero-shot)", "finetuned": "PhaseNet (fine-tuned)"}

def compute_metrics(results, label):
    ev  = [r for r in results if r["window_type"] == "event"]
    ns  = [r for r in results if r["window_type"] == "noise"]

    TP  = sum(1 for r in ev if r["detected"])
    FN  = len(ev) - TP
    FP  = sum(1 for r in ns if r["detected"])
    TN  = len(ns) - FP

    recall    = TP / len(ev) if ev else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    fp_per_hr = FP / (len(ns) * WINDOW_LEN_S / 3600) if ns else 0.0

    maes = [r["p_mae_s"] for r in ev if r["p_mae_s"] is not None]
    p_mae = float(np.mean(maes)) if maes else None

    band_metrics = {}
    for lo, hi, bl in bands:
        sub = [r for r in ev if r["magnitude"] is not None and lo <= r["magnitude"] < hi]
        b_tp = sum(1 for r in sub if r["detected"])
        band_metrics[bl] = {
            "n":      len(sub),
            "TP":     b_tp,
            "FN":     len(sub) - b_tp,
            "recall": round(b_tp / len(sub), 4) if sub else 0.0,
        }

    print(f"    {label}:")
    print(f"      recall={recall:.3f}  precision={precision:.3f}  F1={f1:.3f}  "
          f"FP/hr={fp_per_hr:.1f}"
          + (f"  P-MAE={p_mae:.2f}s" if p_mae else ""))
    for bl, bm in band_metrics.items():
        print(f"      {bl}: recall={bm['recall']:.3f}  (n={bm['n']})")

    return {
        "model":              label,
        "n_event_windows":    len(ev),
        "n_noise_windows":    len(ns),
        "TP": TP, "FN": FN, "FP": FP, "TN": TN,
        "recall":    round(recall,    4),
        "precision": round(precision, 4),
        "f1":        round(f1,        4),
        "fp_per_hour": round(fp_per_hr, 2),
        "p_pick_mae_s": round(p_mae, 4) if p_mae else None,
        "n_p_mae_samples": len(maes),
        "by_magnitude": band_metrics,
    }

metrics_pre  = compute_metrics(results_pre,  "PhaseNet (zero-shot)")
metrics_fine = compute_metrics(results_fine, "PhaseNet (fine-tuned)")

out_json = {
    "phase":          "fine-tuning",
    "test_n_total":   len(test_meta),
    "detection_criterion": f"P pick in [origin-{DETECT_TOL}s, origin+{DETECT_POST}s]",
    "pretrained":  metrics_pre,
    "finetuned":   metrics_fine,
}
with open(ART / METRICS_OUT, "w") as f:
    json.dump(out_json, f, indent=2)
print(f"\n    Saved artifacts/{METRICS_OUT}")

# ── Step 5: Comparison figure ──────────────────────────────────────────────────
print("\n[5] Generating comparison figure...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    "PhaseNet Zero-Shot vs Fine-tuned — Test Set Comparison\n"
    "KOERI Kahramanmaraş Data, Stratified 15% Test Split",
    fontsize=12, fontweight="bold"
)

band_labels = ["M 2–3", "M 3–4", "M≥4"]
band_keys   = [b[2] for b in bands]

x = np.arange(len(band_labels))
bw = 0.32
COLORS = {"pre": "#e74c3c", "fine": "#27ae60"}

metric_configs = [
    ("recall",    "Recall",    True),
    ("precision", "Precision", True),
    ("f1",        "F1 Score",  True),
]

for ax, (mkey, ylabel, _) in zip(axes, metric_configs):
    if mkey in ("precision", "f1"):
        # Overall value (not per-band for precision/F1 — use overall)
        vals_pre  = [metrics_pre["precision"]  if mkey == "precision" else metrics_pre["f1"]] * 3
        vals_fine = [metrics_fine["precision"] if mkey == "precision" else metrics_fine["f1"]] * 3
    else:
        vals_pre  = [metrics_pre["by_magnitude"][bk]["recall"]  for bk in band_keys]
        vals_fine = [metrics_fine["by_magnitude"][bk]["recall"] for bk in band_keys]

    b1 = ax.bar(x - bw / 2, vals_pre,  bw, label="Zero-shot",   color=COLORS["pre"],  alpha=0.85)
    b2 = ax.bar(x + bw / 2, vals_fine, bw, label="Fine-tuned",  color=COLORS["fine"], alpha=0.85)

    for bar, v in list(zip(b1, vals_pre)) + list(zip(b2, vals_fine)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(band_labels if mkey == "recall" else ["Overall"] * 3, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(ylabel, fontsize=11, fontweight="bold")

plt.tight_layout()
fig_path = FIG_DIR / FIGURE_OUT
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"    Saved → figures/{FIGURE_OUT}")

# Delta summary
delta_recall    = metrics_fine["recall"]    - metrics_pre["recall"]
delta_precision = metrics_fine["precision"] - metrics_pre["precision"]
delta_f1        = metrics_fine["f1"]        - metrics_pre["f1"]

print("\n" + "=" * 70)
print("EVALUATION — COMPLETE")
print(f"  Overall recall:    zero-shot={metrics_pre['recall']:.3f}  "
      f"fine-tuned={metrics_fine['recall']:.3f}  Δ={delta_recall:+.3f}")
print(f"  Overall precision: zero-shot={metrics_pre['precision']:.3f}  "
      f"fine-tuned={metrics_fine['precision']:.3f}  Δ={delta_precision:+.3f}")
print(f"  Overall F1:        zero-shot={metrics_pre['f1']:.3f}  "
      f"fine-tuned={metrics_fine['f1']:.3f}  Δ={delta_f1:+.3f}")
print(f"  artifacts/{METRICS_OUT}")
print(f"  figures/{FIGURE_OUT}")
print("=" * 70)
