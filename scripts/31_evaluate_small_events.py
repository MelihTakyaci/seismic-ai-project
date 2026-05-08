# Phase: Quality Audit — Optimization Step 1C
# Purpose: Evaluate GPD (fine-tuned), Ensemble, and PhaseNet (fixed) on M 0.5–2.0
#          small event windows. Produces the complete recall-by-magnitude table
#          combining sub-threshold events with the existing test set metrics.
# Inputs:  data/windows/small_events/*.npz, artifacts/small_event_inventory.csv,
#          models/gpd_final.pt, models/phasenet_final.pt, models/phasenet_fixed.pt,
#          artifacts/quality_audit_final.json
# Outputs: artifacts/small_event_evaluation.json,
#          figures/recall_by_magnitude_complete.png
# Limitations: No analyst P picks for small events — detection window is
#              [origin - 5s, origin + 25s] using the known origin time.
#              M < 1.0 events may have very low SNR; detection is an upper bound.

from pathlib import Path
import json
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
WIN_DIR = ROOT / "data" / "windows" / "small_events"
ART     = ROOT / "artifacts"
MDL_DIR = ROOT / "models"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

SRATE      = 100.0
N_TOTAL    = 6000
ORIGIN_S   = 30.0          # origin is at sample 3000 (30s from window start)
DETECT_TOL = 5.0
DETECT_POST= 25.0
DETECT_LO  = int((ORIGIN_S - DETECT_TOL) * SRATE)   # 2500
DETECT_HI  = int((ORIGIN_S + DETECT_POST) * SRATE)   # 5500
T0_REF     = UTCDateTime("2023-02-06T01:00:00")

DEVICE = "cuda" if torch.cuda.is_available() else (
         "mps"  if torch.backends.mps.is_available() else "cpu")

print("=" * 70)
print("STEP 1C — Evaluation on M 0.5–2.0 Small Events")
print("=" * 70)

# ── Load inventory ───────────────────────────────────────────────────────────
inv = pd.read_csv(ART / "small_event_inventory.csv")
ok_inv = inv[inv["status"].isin(["ok", "cached"])].copy().reset_index(drop=True)
print(f"\n[1] Small event windows: {len(ok_inv)}")
ok_inv["magnitude"] = pd.to_numeric(ok_inv["magnitude"], errors="coerce")
for band, lo, hi in [("M0.5-1.0",0.5,1.0),("M1.0-1.5",1.0,1.5),("M1.5-2.0",1.5,2.0)]:
    n = ok_inv["magnitude"].between(lo, hi, inclusive="right").sum()
    print(f"    {band}: {n} windows")

# ── Load models ──────────────────────────────────────────────────────────────
print("\n[2] Loading models ...")
gpd_model = sbm.GPD.from_pretrained("original")
gpd_ckpt  = MDL_DIR / "gpd_final.pt"
gpd_model.load_state_dict(torch.load(str(gpd_ckpt), map_location=DEVICE, weights_only=True))
gpd_model.to(DEVICE).eval()
GPD_WLEN  = getattr(gpd_model, "in_samples", 400)
GPD_P_IDX = gpd_model.labels.index("P")
print(f"    GPD (fine-tuned) loaded — {gpd_ckpt.name}")

pn_model = sbm.PhaseNet.from_pretrained("original")
pn_ckpt  = MDL_DIR / "phasenet_final.pt"
pn_model.load_state_dict(torch.load(str(pn_ckpt), map_location=DEVICE, weights_only=True))
pn_model.to(DEVICE).eval()
print(f"    PhaseNet (final) loaded — {pn_ckpt.name}")

pn_fixed = sbm.PhaseNet.from_pretrained("original")
pn_fixed_ckpt = MDL_DIR / "phasenet_fixed.pt"
pn_fixed.load_state_dict(torch.load(str(pn_fixed_ckpt), map_location=DEVICE, weights_only=True))
pn_fixed.to(DEVICE).eval()
print(f"    PhaseNet (fixed) loaded — {pn_fixed_ckpt.name}")

# ── Inference helpers ────────────────────────────────────────────────────────
def gpd_sliding_prob(model, data, wlen, p_idx, stride=10):
    positions = list(range(0, N_TOTAL - wlen + 1, stride))
    crops = []
    for s in positions:
        c = data[:, s:s+wlen].astype(np.float32)
        pk = float(np.abs(c).max())
        if pk > 1e-9: c = c / pk
        crops.append(c)
    probs = []
    with torch.no_grad():
        for i in range(0, len(crops), 128):
            b = torch.tensor(np.stack(crops[i:i+128])).to(DEVICE)
            probs.extend(model(b)[:, p_idx].cpu().numpy().tolist())
    ctr = np.array([s + wlen // 2 for s in positions], dtype=float)
    pa  = np.array(probs, dtype=float)
    f   = interp1d(ctr, pa, kind="linear", bounds_error=False, fill_value=(pa[0], pa[-1]))
    return f(np.arange(N_TOTAL, dtype=float)).astype(np.float32)

def pn_ann_prob(model, data):
    st = Stream()
    for i, ch in enumerate(["HHZ", "HHN", "HHE"]):
        tr = Trace(data=data[i].astype(np.float32))
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

# ── Run inference ────────────────────────────────────────────────────────────
print(f"\n[3] Running inference on {len(ok_inv)} windows ...")
rows = []
for i, inv_row in ok_inv.iterrows():
    npz_path = WIN_DIR / inv_row["filename"]
    if not npz_path.exists():
        continue
    try:
        d    = np.load(npz_path)
        data = d["data"].astype(np.float32)
        mag  = float(inv_row["magnitude"])

        gpd_p  = gpd_sliding_prob(gpd_model, data, GPD_WLEN, GPD_P_IDX)
        pn_p   = pn_ann_prob(pn_model, data)
        pnf_p  = pn_ann_prob(pn_fixed, data)
        ens_p  = 0.5 * gpd_p + 0.5 * pn_p

        rows.append({
            "filename":         inv_row["filename"],
            "station":          inv_row["station"],
            "magnitude":        mag,
            "mag_band":         inv_row["mag_band"],
            "gpd_ft_det":       detect(gpd_p,  0.50),
            "pn_final_det":     detect(pn_p,   0.10),
            "pn_fixed_det":     detect(pnf_p,  0.10),
            "ensemble_det":     detect(ens_p,  0.15),
        })
    except Exception as e:
        pass

    if (i + 1) % 100 == 0:
        print(f"    {i+1}/{len(ok_inv)}")

df = pd.DataFrame(rows)
print(f"    Done: {len(df)} windows processed")

# ── Compute per-band metrics ─────────────────────────────────────────────────
print("\n[4] Computing recall by magnitude band ...")

BANDS_SMALL = [
    ("M 0.5–1.0",  0.5,  1.0),
    ("M 1.0–1.5",  1.0,  1.5),
    ("M 1.5–2.0",  1.5,  2.0),
]

small_results = {}
for band_name, lo, hi in BANDS_SMALL:
    sub = df[df["magnitude"].between(lo, hi, inclusive="right")]
    if len(sub) == 0:
        continue
    small_results[band_name] = {
        "n": len(sub),
        "gpd_ft_recall":   round(float(sub["gpd_ft_det"].mean()),  4),
        "pn_final_recall": round(float(sub["pn_final_det"].mean()), 4),
        "pn_fixed_recall": round(float(sub["pn_fixed_det"].mean()), 4),
        "ensemble_recall": round(float(sub["ensemble_det"].mean()), 4),
    }

print(f"\n  Small event recall (M 0.5–2.0):")
print(f"  {'Band':<15} {'N':>5}  {'GPD-FT':>8}  {'PN-Final':>10}  {'PN-Fixed':>10}  {'Ensemble':>10}")
print(f"  {'-'*65}")
for band, res in small_results.items():
    print(f"  {band:<15} {res['n']:>5}  "
          f"{res['gpd_ft_recall']:>8.4f}  "
          f"{res['pn_final_recall']:>10.4f}  "
          f"{res['pn_fixed_recall']:>10.4f}  "
          f"{res['ensemble_recall']:>10.4f}")

# ── Load existing test set results ───────────────────────────────────────────
print("\n[5] Loading existing test-set metrics ...")
audit_path = ART / "quality_audit_final.json"
existing_bands = {}
if audit_path.exists():
    with open(audit_path) as f:
        audit = json.load(f)
    # Pull from ablation_results which has per-magnitude data
    abl_path = ART / "ablation_results.json"
    if abl_path.exists():
        with open(abl_path) as f:
            abl = json.load(f)
        for band_key, band_data in abl.get("by_magnitude", {}).items():
            existing_bands[band_key] = band_data

# Also pull from quality_audit_final directly
if audit_path.exists():
    for src_key in ["gpd_ft_by_magnitude", "ensemble_by_magnitude"]:
        d2 = audit.get(src_key, {})
        for bk, bv in d2.items():
            if bk not in existing_bands:
                existing_bands[bk] = {}
            existing_bands[bk].update(bv)

# Fallback: pull from precision_recall_analysis.json
pra_path = ART / "precision_recall_analysis.json"
if pra_path.exists():
    with open(pra_path) as f:
        pra = json.load(f)
    existing_mag = pra.get("by_magnitude", {})
else:
    existing_mag = {}

# Pull test-set by-magnitude recall from final_evaluation_metrics.json
fem_path = ART / "final_evaluation_metrics.json"
mag_recall = {}
if fem_path.exists():
    with open(fem_path) as f:
        fem = json.load(f)
    # Structure: fem[model][band]["recall"]
    for model_key in ["GPD (fine-tuned)", "Ensemble (PhaseNet+GPD)"]:
        mag_recall[model_key] = {}
        for band_key in ["M2.0-3.0", "M3.0-4.0", "M>=4.0"]:
            val = fem.get(model_key, {}).get(band_key, {}).get("recall")
            if val is not None:
                mag_recall[model_key][band_key] = val

print(f"    Existing mag-band entries: {len(existing_bands)}")

# ── Build complete magnitude table ───────────────────────────────────────────
print("\n[6] Building complete recall-by-magnitude table ...")

# Complete bands: small events + existing test set bands
COMPLETE_BANDS = [
    "M 0.5–1.0",
    "M 1.0–1.5",
    "M 1.5–2.0",
    "M 2.0–3.0",
    "M 3.0–4.0",
    "M ≥ 4.0",
]

# Compute complete recall from small events (M<2) and existing (M>=2)
# For M>=2, pull from quality_audit_final or final_evaluation_metrics
complete = {}
for band in COMPLETE_BANDS:
    if band in small_results:
        complete[band] = {
            "n":             small_results[band]["n"],
            "gpd_ft":        small_results[band]["gpd_ft_recall"],
            "pn_fixed":      small_results[band]["pn_fixed_recall"],
            "ensemble":      small_results[band]["ensemble_recall"],
            "source":        "small_event_download",
        }
    else:
        # Map band names to existing metric keys
        key_map = {
            "M 2.0–3.0": ("M2-3", "M2.0-3.0", "M 2.0–3.0"),
            "M 3.0–4.0": ("M3-4", "M3.0-4.0", "M 3.0–4.0"),
            "M ≥ 4.0":   ("M4+",  "M>=4.0",   "M ≥ 4.0"),
        }
        # Pull from ablation results JSON which has per-mag data
        abl_path2 = ART / "ablation_results.json"
        gpd_r = None; ens_r = None; n = None
        if abl_path2.exists():
            with open(abl_path2) as f:
                abl2 = json.load(f)
            # Try various key formats
            for k_try in key_map.get(band, ()):
                cd = abl2.get("by_magnitude", {}).get(k_try, {})
                if cd:
                    gpd_r = cd.get("C_recall") or cd.get("gpd_emsc_recall") or cd.get("recall")
                    ens_r = cd.get("D_recall") or cd.get("ensemble_recall") or cd.get("recall")
                    n     = cd.get("n", 0)
                    break
        complete[band] = {
            "n":        n or 0,
            "gpd_ft":   gpd_r,
            "pn_fixed": None,
            "ensemble": ens_r,
            "source":   "test_set_existing",
        }

# Print complete table
print(f"\n  COMPLETE recall-by-magnitude (small events + test set):")
print(f"  {'Band':<12} {'N':>5}  {'GPD-FT':>8}  {'PN-Fixed':>10}  {'Ensemble':>10}  {'Source'}")
print(f"  {'-'*75}")
for band in COMPLETE_BANDS:
    res = complete.get(band, {})
    n = res.get("n", 0)
    g = f"{res['gpd_ft']:.4f}" if res.get("gpd_ft") is not None else "  —  "
    p = f"{res['pn_fixed']:.4f}" if res.get("pn_fixed") is not None else "  —  "
    e = f"{res['ensemble']:.4f}" if res.get("ensemble") is not None else "  —  "
    src = "new" if res.get("source") == "small_event_download" else "existing"
    print(f"  {band:<12} {n:>5}  {g:>8}  {p:>10}  {e:>10}  [{src}]")

# ── Produce figure ───────────────────────────────────────────────────────────
print("\n[7] Generating recall_by_magnitude_complete.png ...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor("#0a0e1a")
for ax in axes:
    ax.set_facecolor("#0f1525")
    ax.tick_params(colors="#a0aabb")
    for sp in ax.spines.values(): sp.set_edgecolor("#2a3450")
    ax.xaxis.label.set_color("#a0aabb")
    ax.yaxis.label.set_color("#a0aabb")
    ax.title.set_color("#e8ecf0")

x_labels = COMPLETE_BANDS
x = np.arange(len(x_labels))
w = 0.28

gpd_vals  = [complete.get(b, {}).get("gpd_ft")   for b in COMPLETE_BANDS]
pnf_vals  = [complete.get(b, {}).get("pn_fixed") for b in COMPLETE_BANDS]
ens_vals  = [complete.get(b, {}).get("ensemble") for b in COMPLETE_BANDS]

def _v(lst):
    return [v if v is not None else 0 for v in lst]

ax = axes[0]
ax.bar(x - w,      _v(gpd_vals), w, label="GPD (fine-tuned)", color="#00ff88", alpha=0.85)
ax.bar(x,          _v(pnf_vals), w, label="PN (fixed)",        color="#66aaff", alpha=0.85)
ax.bar(x + w,      _v(ens_vals), w, label="Ensemble",          color="#ffaa44", alpha=0.85)

# Mark missing bars
for i, (g, p, e) in enumerate(zip(gpd_vals, pnf_vals, ens_vals)):
    if g is None:
        ax.text(x[i] - w, 0.02, "—", ha="center", fontsize=8, color="#666")
    if p is None:
        ax.text(x[i],    0.02, "—", ha="center", fontsize=8, color="#666")
    if e is None:
        ax.text(x[i] + w, 0.02, "—", ha="center", fontsize=8, color="#666")

ax.axvline(2.5, color="#555", lw=1.5, ls="--", alpha=0.8)
ax.text(1.0, 1.03, "← NEW (M<2.0)", ha="center", fontsize=8, color="#ffcc00",
        transform=ax.get_xaxis_transform())
ax.text(4.0, 1.03, "Existing test set →", ha="center", fontsize=8, color="#aabbcc",
        transform=ax.get_xaxis_transform())
ax.set_xticks(x)
ax.set_xticklabels(x_labels, rotation=25, ha="right", fontsize=8, color="#a0aabb")
ax.set_ylabel("Recall", color="#a0aabb")
ax.set_ylim(0, 1.15)
ax.set_title("Complete Recall by Magnitude Band", color="#e8ecf0", fontsize=11)
ax.legend(fontsize=8, labelcolor="#e8ecf0", facecolor="#1a2035", edgecolor="#2a3450")
ax.axhline(0.9, color="#333", lw=0.7, ls=":")

# Right panel: N per band
ax2 = axes[1]
n_vals = [complete.get(b, {}).get("n", 0) for b in COMPLETE_BANDS]
colors = ["#ffcc00"] * 3 + ["#aabbcc"] * 3
bars = ax2.bar(x, n_vals, 0.6, color=colors, alpha=0.85)
ax2.set_xticks(x)
ax2.set_xticklabels(x_labels, rotation=25, ha="right", fontsize=8, color="#a0aabb")
ax2.set_ylabel("Number of windows", color="#a0aabb")
ax2.set_title("Windows per Magnitude Band", color="#e8ecf0", fontsize=11)
ax2.axvline(2.5, color="#555", lw=1.5, ls="--", alpha=0.8)
for bar, n in zip(bars, n_vals):
    if n > 0:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 str(n), ha="center", va="bottom", fontsize=7, color="#e8ecf0")

plt.suptitle(
    "Complete Recall by Magnitude — Including Sub-Threshold Events (M < 2.0)\n"
    "Yellow = new M<2.0 evaluation | Grey = existing test set",
    color="#e8ecf0", fontsize=11, y=1.04)
plt.tight_layout()
fig_path = FIG_DIR / "recall_by_magnitude_complete.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor="#0a0e1a")
plt.close()
print(f"    Saved → {fig_path}")

# ── Save JSON ────────────────────────────────────────────────────────────────
out = {
    "phase": "small_event_evaluation",
    "n_windows_evaluated": len(df),
    "small_event_recall": small_results,
    "complete_magnitude_table": complete,
    "detection_criterion": {
        "window": f"[origin - {DETECT_TOL}s, origin + {DETECT_POST}s]",
        "gpd_ft_threshold": 0.50,
        "pn_fixed_threshold": 0.10,
        "ensemble_threshold": 0.15,
    },
    "note": (
        "P pick times not available for M<2.0 events. Detection evaluated using "
        "known origin time ± window. Recall is an upper bound — some detections "
        "may be background aftershock triggers rather than target event P waves."
    ),
}
json_path = ART / "small_event_evaluation.json"
with open(json_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"    Saved → {json_path}")

print("\n" + "=" * 70)
print("STEP 1C COMPLETE — Sub-threshold Recall Evaluation")
print(f"  Small events evaluated: {len(df)}")
print(f"  JSON:   artifacts/small_event_evaluation.json")
print(f"  Figure: figures/recall_by_magnitude_complete.png")
print("=" * 70)
