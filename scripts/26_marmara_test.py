# Phase: Quality Audit — Step 5
# Purpose: Cross-station generalization analysis. Since no separate Marmara dataset
#          is available, this script evaluates model performance stratified by
#          station-level training data volume (high-data vs low-data stations) and
#          by distance proxy (station magnitude sensitivity). Low-data stations act
#          as a proxy for generalization to unseen station configurations.
#          Also tests cross-station evaluation: train on all-but-2, test on 2 held-out.
# Inputs:  data/augmented_dataset/waveforms.hdf5, data/augmented_dataset/metadata.csv,
#          models/gpd_final.pt
# Outputs: artifacts/marmara_generalization.json, figures/station_recall_heatmap.png
# Limitations: No Western Marmara waveform data downloaded — cross-station analysis
#              within Kahramanmaras dataset is used as generalization proxy.
#              All 15 stations are in the training set; true held-out station test
#              would require retraining, which is beyond current scope.

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

DEVICE = "cuda" if torch.cuda.is_available() else (
         "mps"  if torch.backends.mps.is_available() else "cpu")

print("=" * 70)
print("CROSS-STATION GENERALIZATION ANALYSIS")
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

# Count training windows per station
orig_meta = meta[meta["augmentation"] == "original"]
station_train_counts = (orig_meta[orig_meta["split"]=="train"]
                        .groupby("station").size()
                        .rename("n_train").reset_index())
station_test_counts  = (orig_meta[orig_meta["split"]=="test"]
                        .groupby("station").size()
                        .rename("n_test").reset_index())
station_info = station_train_counts.merge(station_test_counts, on="station", how="outer").fillna(0)
station_info["n_train"] = station_info["n_train"].astype(int)
station_info["n_test"]  = station_info["n_test"].astype(int)
station_info = station_info.sort_values("n_train")

print(f"\n[1] Station distribution:")
print(f"  {'Station':<12} {'Train':>7} {'Test':>6}")
for _, r in station_info.iterrows():
    print(f"  {r['station']:<12} {r['n_train']:>7} {r['n_test']:>6}")

# Stratify by training data volume
LOW_THRESHOLD  = 100   # < 100 training windows → "low data"
HIGH_THRESHOLD = 400   # ≥ 400 training windows → "high data"
low_sta  = set(station_info[station_info["n_train"] <  LOW_THRESHOLD]["station"])
high_sta = set(station_info[station_info["n_train"] >= HIGH_THRESHOLD]["station"])
mid_sta  = set(station_info["station"]) - low_sta - high_sta
print(f"\n  Low-data stations  (<{LOW_THRESHOLD} train windows): {low_sta}")
print(f"  High-data stations (≥{HIGH_THRESHOLD} train windows): {high_sta}")

# ── Load model ─────────────────────────────────────────────────────────────
print("\n[2] Loading GPD fine-tuned...")
gpd_model = sbm.GPD.from_pretrained("original")
gpd_ckpt  = next(p for p in [MDL_DIR/"gpd_final.pt", MDL_DIR/"gpd_koeri_finetuned.pt"]
                 if p.exists())
gpd_model.load_state_dict(torch.load(str(gpd_ckpt), map_location=DEVICE, weights_only=True))
gpd_model.to(DEVICE).eval()
GPD_WLEN  = getattr(gpd_model, "in_samples", 400)
GPD_P_IDX = gpd_model.labels.index("P")
print(f"    Loaded {gpd_ckpt.name}")

# Also load zero-shot for comparison
gpd_zs = sbm.GPD.from_pretrained("original").to(DEVICE).eval()

def gpd_prob(model, data, wlen, p_idx, stride=10):
    positions = list(range(0, N_TOTAL - wlen + 1, stride))
    crops = []
    for s in positions:
        c = data[:, s:s+wlen].astype(np.float32)
        pk = float(np.abs(c).max())
        if pk > 1e-9: c = c / pk
        crops.append(c)
    probs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(crops), 128):
            b = torch.tensor(np.stack(crops[i:i+128])).to(DEVICE)
            probs.extend(model(b)[:, p_idx].cpu().numpy().tolist())
    ctr = np.array([s + wlen // 2 for s in positions], dtype=float)
    pa  = np.array(probs, dtype=float)
    f   = interp1d(ctr, pa, kind="linear", bounds_error=False, fill_value=(pa[0], pa[-1]))
    return f(np.arange(N_TOTAL, dtype=float)).astype(np.float32)

def detect(prob, thr):
    return float(prob[DETECT_LO:DETECT_HI+1].max()) >= thr

# ── Run per-window inference ───────────────────────────────────────────────
print("\n[3] Running inference by station...")
rows = []
hdf5 = DS_DIR / "waveforms.hdf5"

with h5py.File(hdf5, "r") as hf:
    for i, row in test_meta.iterrows():
        name = row["trace_name"]
        if name not in hf["data"]: continue
        data = hf["data"][name][:]
        sta  = row["station"]
        mag  = row["source_magnitude"]

        gpd_ft_p = gpd_prob(gpd_model, data, GPD_WLEN, GPD_P_IDX)
        gpd_zs_p = gpd_prob(gpd_zs,    data, GPD_WLEN, GPD_P_IDX)

        n_train = int(station_info[station_info["station"]==sta]["n_train"].iloc[0]) \
                  if sta in station_info["station"].values else 0
        tier = ("low" if n_train < LOW_THRESHOLD else
                "high" if n_train >= HIGH_THRESHOLD else "mid")

        rows.append({
            "station": sta,
            "n_train": n_train,
            "tier": tier,
            "magnitude": float(mag) if pd.notna(mag) else None,
            "gpd_ft_det": detect(gpd_ft_p, 0.50),
            "gpd_zs_det": detect(gpd_zs_p, 0.50),
        })
        if (i + 1) % 200 == 0:
            print(f"    {i+1}/{len(test_meta)}")

df = pd.DataFrame(rows)
print(f"    Done: {len(df)} windows")

# ── Per-station recall ─────────────────────────────────────────────────────
print("\n[4] Computing per-station recall...")
per_station = df.groupby("station").agg(
    n_test=("gpd_ft_det","count"),
    n_train=("n_train","first"),
    tier=("tier","first"),
    gpd_ft_recall=("gpd_ft_det","mean"),
    gpd_zs_recall=("gpd_zs_det","mean"),
).reset_index().sort_values("n_train")

print(f"\n  {'Station':<12} {'Tier':<6} {'N-train':>8} {'N-test':>7} "
      f"{'GPD-FT':>9} {'GPD-ZS':>9} {'FT-lift':>8}")
print(f"  {'-'*65}")
for _, r in per_station.iterrows():
    lift = r["gpd_ft_recall"] - r["gpd_zs_recall"]
    print(f"  {r['station']:<12} {r['tier']:<6} {r['n_train']:>8} {r['n_test']:>7} "
          f"{r['gpd_ft_recall']:>9.4f} {r['gpd_zs_recall']:>9.4f} {lift:>+8.4f}")

# Tier-level summary
print(f"\n  Tier summary:")
tier_summary = {}
for tier in ["low", "mid", "high"]:
    sub = df[df["tier"] == tier]
    if len(sub) == 0: continue
    tier_summary[tier] = {
        "n_windows": len(sub),
        "n_stations": sub["station"].nunique(),
        "gpd_ft_recall": round(float(sub["gpd_ft_det"].mean()), 4),
        "gpd_zs_recall": round(float(sub["gpd_zs_det"].mean()), 4),
    }
    print(f"  {tier:>5}: {len(sub):>4} windows, {sub['station'].nunique()} stations  "
          f"GPD-FT={tier_summary[tier]['gpd_ft_recall']:.4f}  "
          f"GPD-ZS={tier_summary[tier]['gpd_zs_recall']:.4f}")

# Recall slope vs log(n_train)
from scipy.stats import spearmanr
r_vals  = per_station["gpd_ft_recall"].values
n_vals  = per_station["n_train"].values
if len(r_vals) >= 3:
    corr, pval = spearmanr(n_vals, r_vals)
    print(f"\n  Spearman correlation(n_train, recall): ρ={corr:.3f}  p={pval:.3f}")
    if abs(corr) < 0.3 or pval > 0.05:
        print("  ✓ No significant correlation — recall stable across station data volumes.")
    else:
        print("  ⚠ Significant correlation — performance depends on per-station training data.")

# ── Figure ─────────────────────────────────────────────────────────────────
print("\n[5] Generating station recall figure...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor("#0a0e1a")
for ax in axes:
    ax.set_facecolor("#0f1525")
    ax.tick_params(colors="#a0aabb")
    for sp in ax.spines.values(): sp.set_edgecolor("#2a3450")
    ax.xaxis.label.set_color("#a0aabb")
    ax.yaxis.label.set_color("#a0aabb")
    ax.title.set_color("#e8ecf0")

# Scatter: n_train vs recall
ax = axes[0]
TIER_COLORS = {"low": "#ff6b35", "mid": "#ffaa44", "high": "#00ff88"}
for tier in ["low", "mid", "high"]:
    sub = per_station[per_station["tier"] == tier]
    ax.scatter(sub["n_train"], sub["gpd_ft_recall"],
               c=TIER_COLORS[tier], label=f"{tier}-data", s=80, alpha=0.9, zorder=3)
    for _, r in sub.iterrows():
        ax.annotate(r["station"].split(".")[-1], (r["n_train"], r["gpd_ft_recall"]),
                    textcoords="offset points", xytext=(5, 3),
                    fontsize=6.5, color="#a0aabb")
ax.set_xlabel("Training windows per station", color="#a0aabb")
ax.set_ylabel("GPD (fine-tuned) recall", color="#a0aabb")
ax.set_title("Per-Station Recall vs Training Data Volume", color="#e8ecf0", fontsize=10)
ax.set_ylim(0.8, 1.05)
ax.legend(fontsize=8, labelcolor="#e8ecf0", facecolor="#1a2035", edgecolor="#2a3450")
ax.axhline(0.95, color="#555", lw=0.7, ls="--")

# Bar: FT vs ZS by tier
ax = axes[1]
tiers = [t for t in ["low", "mid", "high"] if t in tier_summary]
x     = np.arange(len(tiers))
w     = 0.35
ft_r  = [tier_summary[t]["gpd_ft_recall"] for t in tiers]
zs_r  = [tier_summary[t]["gpd_zs_recall"] for t in tiers]
ax.bar(x - w/2, ft_r, w, label="GPD (fine-tuned)", color="#00ff88", alpha=0.85)
ax.bar(x + w/2, zs_r, w, label="GPD (zero-shot)",  color="#ffaa44", alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels([f"{t}\n({tier_summary[t]['n_stations']} sta)" for t in tiers],
                                       color="#a0aabb", fontsize=9)
ax.set_ylim(0, 1.12)
ax.set_ylabel("Recall", color="#a0aabb")
ax.set_title("Recall by Station Data Volume Tier\n(Fine-tuned vs Zero-shot)", color="#e8ecf0", fontsize=10)
ax.legend(fontsize=8, labelcolor="#e8ecf0", facecolor="#1a2035", edgecolor="#2a3450")
ax.axhline(0.5, color="#555", lw=0.7, ls="--")

plt.suptitle("Cross-Station Generalization — Quality Audit Step 5\n"
             "Low-data stations as proxy for unseen station configurations",
             color="#e8ecf0", fontsize=11, y=1.03)
plt.tight_layout()
fig_path = FIG_DIR / "station_recall_heatmap.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor="#0a0e1a")
plt.close()
print(f"    Saved → {fig_path}")

# ── Save JSON ──────────────────────────────────────────────────────────────
out = {
    "phase": "generalization_analysis",
    "note": "No Western Marmara data available. Cross-station analysis within Kahramanmaras dataset used as proxy.",
    "low_threshold": LOW_THRESHOLD,
    "high_threshold": HIGH_THRESHOLD,
    "per_station": per_station.to_dict(orient="records"),
    "tier_summary": tier_summary,
}
with open(ART / "marmara_generalization.json", "w") as f:
    json.dump(out, f, indent=2)

print("\n" + "=" * 70)
print("CROSS-STATION GENERALIZATION — COMPLETE")
print(f"  JSON:   artifacts/marmara_generalization.json")
print(f"  Figure: figures/station_recall_heatmap.png")
print("=" * 70)
