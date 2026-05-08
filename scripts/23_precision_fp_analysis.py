# Phase: Quality Audit — Step 2
# Purpose: Compute precision and false-positive rate for all 7 model variants.
#          Uses pre-P portion of event windows as a proxy for false-positive rate
#          (samples 0–2499, i.e. >5 seconds before origin) and the phase5
#          detection_results.json for the zero-shot models (which contains
#          explicit noise windows).
# Inputs:  data/augmented_dataset/waveforms.hdf5, data/augmented_dataset/metadata.csv,
#          artifacts/detection_results.json, models/gpd_final.pt,
#          models/phasenet_final.pt (or phasenet_koeri_finetuned.pt)
# Outputs: artifacts/precision_recall_analysis.json, figures/precision_fp_comparison.png
# Limitations: FP rate for fine-tuned models is estimated from pre-P event window
#              portions rather than separate noise windows. This slightly underestimates
#              FP rate (correlated noise is easier to reject than independent noise).
#              Phase5 detection_results.json noise windows used for zero-shot FP/hour.

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
DETECT_LO  = int((ORIGIN_S - DETECT_TOL) * SRATE)   # 2500
DETECT_HI  = int((ORIGIN_S + DETECT_POST) * SRATE)  # 5500
# Pre-P zone: samples 0–2499 (>5s before any possible P arrival)
PRE_P_END  = DETECT_LO   # 2500 samples = 25s
T0_REF     = UTCDateTime("2023-02-06T01:00:00")

DEVICE = "cuda" if torch.cuda.is_available() else (
         "mps"  if torch.backends.mps.is_available() else "cpu")

THRESHOLDS = {
    "stalta":       3.0,
    "phasenet_zs":  0.30,
    "eqt_zs":       0.10,
    "gpd_zs":       0.50,
    "phasenet_ft":  0.05,
    "gpd_ft":       0.50,
    "ensemble":     0.15,
}

print("=" * 70)
print("PRECISION / FALSE-POSITIVE ANALYSIS")
print("=" * 70)

# ── Load metadata and models ───────────────────────────────────────────────
meta = pd.read_csv(DS_DIR / "metadata.csv")
meta["trace_p_arrival_sample"] = pd.to_numeric(meta["trace_p_arrival_sample"], errors="coerce")
meta["source_magnitude"]       = pd.to_numeric(meta["source_magnitude"],       errors="coerce")
if "augmentation" not in meta.columns:
    meta["augmentation"] = "original"

test_meta = meta[
    (meta["split"] == "test") & (meta["augmentation"] == "original")
].copy().reset_index(drop=True)
print(f"\n[1] Test windows: {len(test_meta)}")

# Load fine-tuned models
gpd_model = sbm.GPD.from_pretrained("original")
gpd_ckpt  = next(p for p in [MDL_DIR/"gpd_final.pt", MDL_DIR/"gpd_koeri_finetuned.pt"]
                 if p.exists())
gpd_model.load_state_dict(torch.load(str(gpd_ckpt), map_location=DEVICE, weights_only=True))
gpd_model.to(DEVICE).eval()
print(f"    GPD fine-tuned: {gpd_ckpt.name}")

pn_model  = sbm.PhaseNet.from_pretrained("original")
pn_ckpt   = next(p for p in [MDL_DIR/"phasenet_final.pt", MDL_DIR/"phasenet_koeri_finetuned.pt"]
                 if p.exists())
pn_model.load_state_dict(torch.load(str(pn_ckpt), map_location=DEVICE, weights_only=True))
pn_model.to(DEVICE).eval()
print(f"    PhaseNet fine-tuned: {pn_ckpt.name}")

pn_zs  = sbm.PhaseNet.from_pretrained("original").to(DEVICE).eval()
gpd_zs = sbm.GPD.from_pretrained("original").to(DEVICE).eval()

GPD_WLEN  = getattr(gpd_model, "in_samples", 400)
GPD_P_IDX = gpd_model.labels.index("P")

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

def pn_ann_prob(model, data_3ch, t0=T0_REF, suffix="_P"):
    st = Stream()
    for i, ch in enumerate(["HHZ", "HHN", "HHE"]):
        tr = Trace(data=data_3ch[i].astype(np.float32))
        tr.stats.network = "KO"; tr.stats.station = "TEST"
        tr.stats.channel = ch; tr.stats.sampling_rate = SRATE
        tr.stats.starttime = t0
        st.append(tr)
    with torch.no_grad():
        ann = model.annotate(st)
    tr_p = next((t for t in ann if t.stats.channel.endswith(suffix)), None)
    if tr_p is None:
        return np.zeros(N_TOTAL, dtype=np.float32)
    tgt = np.arange(N_TOTAL) / SRATE
    src = float(tr_p.stats.starttime - t0) + np.arange(len(tr_p.data)) / tr_p.stats.sampling_rate
    if len(src) < 2: return np.zeros(N_TOTAL, dtype=np.float32)
    f = interp1d(src, tr_p.data.astype(np.float32), kind="linear", bounds_error=False, fill_value=0.0)
    return f(tgt).astype(np.float32)

def stalta_prob(data_3ch):
    from obspy.signal.trigger import classic_sta_lta
    nsta, nlta = int(1.0 * SRATE), int(10.0 * SRATE)
    cf = classic_sta_lta(data_3ch[0], nsta, nlta)
    return cf.astype(np.float32)

# ── Run inference on test set ──────────────────────────────────────────────
print("\n[2] Running inference on test set...")

# For each window: record detect_window peak (signal) AND pre_P peak (noise proxy)
records = []
hdf5_path = DS_DIR / "waveforms.hdf5"

with h5py.File(hdf5_path, "r") as hf:
    for i, row in test_meta.iterrows():
        name = row["trace_name"]
        if name not in hf["data"]:
            continue
        data = hf["data"][name][:]
        mag  = row["source_magnitude"]

        # Get all probability traces
        sl    = stalta_prob(data)
        pn_zs_p = pn_ann_prob(pn_zs,   data)
        pn_ft_p = pn_ann_prob(pn_model, data)
        gpd_zs_p= gpd_sliding_prob(gpd_zs,  data, GPD_WLEN, gpd_zs.labels.index("P"))
        gpd_ft_p= gpd_sliding_prob(gpd_model,data, GPD_WLEN, GPD_P_IDX)
        ens_p   = 0.5 * pn_ft_p + 0.5 * gpd_ft_p

        probs = {
            "stalta":       sl,
            "phasenet_zs":  pn_zs_p,
            "gpd_zs":       gpd_zs_p,
            "phasenet_ft":  pn_ft_p,
            "gpd_ft":       gpd_ft_p,
            "ensemble":     ens_p,
        }

        rec = {"trace_name": name, "magnitude": float(mag) if pd.notna(mag) else None}
        for mname, prob in probs.items():
            thr      = THRESHOLDS[mname]
            sig_peak = float(prob[DETECT_LO:DETECT_HI+1].max())
            pre_peak = float(prob[:PRE_P_END].max())
            detected = sig_peak >= thr
            fp_fired = pre_peak >= thr  # fired in pre-P zone → false positive
            rec[mname] = {
                "sig_peak": round(sig_peak, 4),
                "pre_peak": round(pre_peak, 4),
                "detected": detected,
                "fp_in_window": fp_fired,
            }
        records.append(rec)

        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(test_meta)}")

print(f"    Done: {len(records)} windows")

# ── Precision / FP metrics from event windows ──────────────────────────────
print("\n[3] Computing precision and FP metrics...")

MODEL_NAMES = ["stalta", "phasenet_zs", "gpd_zs", "phasenet_ft", "gpd_ft", "ensemble"]
MODEL_LABELS = {
    "stalta":       "STA/LTA",
    "phasenet_zs":  "PhaseNet (zero-shot)",
    "gpd_zs":       "GPD (zero-shot)",
    "phasenet_ft":  "PhaseNet (fine-tuned)",
    "gpd_ft":       "GPD (fine-tuned)",
    "ensemble":     "Ensemble (PhaseNet+GPD)",
}

# Pull zero-shot FP/hour from detection_results.json noise windows
det_path = ART / "detection_results.json"
noise_stats = {}
if det_path.exists():
    det_all = json.load(open(det_path))
    det_list = det_all["results"] if isinstance(det_all, dict) else det_all
    noise_wins = [r for r in det_list if r.get("window_type") == "noise"]
    WIN_DURATION_HR = 60.0 / 3600  # 60s window
    total_noise_hr  = len(noise_wins) * WIN_DURATION_HR
    for mk in ["stalta", "phasenet", "eqtransformer", "gpd"]:
        n_fp = sum(1 for r in noise_wins if r.get(mk, {}).get("detected", False))
        noise_stats[mk] = {
            "n_noise_windows": len(noise_wins),
            "n_fp": n_fp,
            "fp_per_hour": round(n_fp / total_noise_hr, 2) if total_noise_hr > 0 else None,
        }
    print(f"    Noise windows in detection_results.json: {len(noise_wins)}")

metrics = {}
for mname in MODEL_NAMES:
    thr        = THRESHOLDS[mname]
    tp         = sum(1 for r in records if r[mname]["detected"])
    fp_in_win  = sum(1 for r in records if r[mname]["fp_in_window"])
    n          = len(records)
    recall     = round(tp / n, 4)
    # Precision from event windows: TP / (TP + FP_in_window)
    prec_proxy = round(tp / (tp + fp_in_win), 4) if (tp + fp_in_win) > 0 else None
    # FP rate: fraction of pre-P zones that fire
    fp_rate_win= round(fp_in_win / n, 4)

    metrics[mname] = {
        "label":            MODEL_LABELS[mname],
        "threshold":        thr,
        "n_windows":        n,
        "tp":               tp,
        "recall":           recall,
        "fp_in_window":     fp_in_win,
        "fp_rate_in_window": fp_rate_win,
        "precision_proxy":  prec_proxy,
    }
    # Append zero-shot FP/hour from noise windows if available
    zs_key = mname.replace("_zs", "").replace("_ft", "").replace("phasenet", "phasenet")
    if zs_key in noise_stats:
        metrics[mname]["fp_per_hour_noise"] = noise_stats[zs_key]["fp_per_hour"]

print(f"\n  {'Model':<28} {'Recall':>7}  {'Prec(proxy)':>12}  "
      f"{'FP-rate(pre-P)':>15}  {'FP/hr(noise)':>13}")
print(f"  {'-'*80}")
for mname, m in metrics.items():
    fp_hr = m.get("fp_per_hour_noise", "—")
    fp_hr_str = f"{fp_hr:.1f}" if isinstance(fp_hr, float) else "—"
    print(f"  {m['label']:<28} {m['recall']:>7.4f}  {m['precision_proxy'] or 0:>12.4f}  "
          f"{m['fp_rate_in_window']:>15.4f}  {fp_hr_str:>13}")

# ── Figure ─────────────────────────────────────────────────────────────────
print("\n[4] Generating figure...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor("#0a0e1a")
for ax in axes:
    ax.set_facecolor("#0f1525")
    ax.tick_params(colors="#a0aabb")
    for sp in ax.spines.values(): sp.set_edgecolor("#2a3450")
    ax.xaxis.label.set_color("#a0aabb")
    ax.yaxis.label.set_color("#a0aabb")
    ax.title.set_color("#e8ecf0")

COLORS = {
    "stalta":      "#aaaaaa",
    "phasenet_zs": "#4488ff",
    "gpd_zs":      "#ffaa44",
    "phasenet_ft": "#00d4ff",
    "gpd_ft":      "#00ff88",
    "ensemble":    "#ff6b35",
}
short_labels = {
    "stalta":       "STA/LTA",
    "phasenet_zs":  "PN-ZS",
    "gpd_zs":       "GPD-ZS",
    "phasenet_ft":  "PN-FT",
    "gpd_ft":       "GPD-FT",
    "ensemble":     "Ensemble",
}

x = np.arange(len(MODEL_NAMES))
w = 0.35
colors = [COLORS[m] for m in MODEL_NAMES]
recalls = [metrics[m]["recall"] for m in MODEL_NAMES]
precs   = [metrics[m]["precision_proxy"] or 0 for m in MODEL_NAMES]
fp_rates= [metrics[m]["fp_rate_in_window"] for m in MODEL_NAMES]
xlabels = [short_labels[m] for m in MODEL_NAMES]

ax = axes[0]
bars1 = ax.bar(x - w/2, recalls, w, label="Recall", color=colors, alpha=0.9, edgecolor="none")
bars2 = ax.bar(x + w/2, precs,   w, label="Precision (proxy)", color=colors, alpha=0.45,
               edgecolor="none", hatch="//")
ax.set_xticks(x); ax.set_xticklabels(xlabels, color="#a0aabb", fontsize=8, rotation=20, ha="right")
ax.set_ylim(0, 1.12)
ax.set_ylabel("Score", color="#a0aabb")
ax.set_title("Recall vs Precision (proxy)\n[pre-P zone FP criterion]", color="#e8ecf0", fontsize=10)
from matplotlib.patches import Patch
legend_elems = [Patch(facecolor="#888", label="Recall"), Patch(facecolor="#888", alpha=0.4, hatch="//", label="Precision")]
ax.legend(handles=legend_elems, fontsize=8, labelcolor="#e8ecf0", facecolor="#1a2035", edgecolor="#2a3450")

ax = axes[1]
ax.bar(x, fp_rates, color=colors, alpha=0.85, edgecolor="none")
ax.set_xticks(x); ax.set_xticklabels(xlabels, color="#a0aabb", fontsize=8, rotation=20, ha="right")
ax.set_ylabel("FP rate in pre-P zone", color="#a0aabb")
ax.set_title("False-Positive Rate\n(fires before event window, per window)", color="#e8ecf0", fontsize=10)
ax.axhline(0.05, color="#555", lw=0.8, ls="--", label="5% threshold")
ax.legend(fontsize=8, labelcolor="#e8ecf0", facecolor="#1a2035", edgecolor="#2a3450")

plt.suptitle("Precision & False-Positive Analysis — Quality Audit Step 2",
             color="#e8ecf0", fontsize=11, y=1.01)
plt.tight_layout()
fig_path = FIG_DIR / "precision_fp_comparison.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor="#0a0e1a")
plt.close()
print(f"    Saved → {fig_path}")

# ── Save JSON ──────────────────────────────────────────────────────────────
out = {"phase": "precision_fp_analysis", "n_test_windows": len(records),
       "window_duration_s": 60.0, "pre_p_zone_samples": PRE_P_END,
       "models": metrics}
with open(ART / "precision_recall_analysis.json", "w") as f:
    json.dump(out, f, indent=2)

print("\n" + "=" * 70)
print("PRECISION / FP ANALYSIS — COMPLETE")
print(f"  JSON:   artifacts/precision_recall_analysis.json")
print(f"  Figure: figures/precision_fp_comparison.png")
print("=" * 70)
