# Phase: Quality Audit — Optimization Problem C (simplified, cached data)
# Purpose: Evaluate GPD-FT, PhaseNet-Fixed, and Ensemble on existing Marmara
#          windows (data/windows/marmara/, 5,838 windows, 28 KO stations).
#          Compares Kahramanmaras (training region) vs Marmara (deployment region)
#          recall to assess geographic generalization.
# Inputs:  data/windows/marmara/*.npz, models/gpd_final.pt, models/phasenet_fixed.pt
# Outputs: artifacts/marmara_extended_results.json, figures/kahramanmaras_vs_marmara_performance.png
# Limitations: No analyst P-pick labels in Marmara NPZs — uses origin-time-based
#              detection window [origin-5s, origin+25s] = samples [2500, 5500].
#              Origin assumed at sample 3000 (30s into 60s window, project convention).

from pathlib import Path
import json
import numpy as np
import torch
from scipy.interpolate import interp1d
from obspy import Stream, Trace, UTCDateTime
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seisbench.models as sbm

ROOT    = Path(__file__).resolve().parent.parent
WIN_DIR = ROOT / "data" / "windows" / "marmara"
MDL_DIR = ROOT / "models"
ART     = ROOT / "artifacts"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

SRATE      = 100.0
N_TOTAL    = 6000
ORIGIN_S   = 30.0
T0_REF     = UTCDateTime("2023-02-06T01:00:00")

DETECT_LO  = int((ORIGIN_S - 5.0) * SRATE)   # sample 2500
DETECT_HI  = int((ORIGIN_S + 25.0) * SRATE)  # sample 5500

DEVICE = "cuda" if torch.cuda.is_available() else (
         "mps"  if torch.backends.mps.is_available() else "cpu")

print("=" * 70)
print("PROBLEM C — Marmara Geographic Generalization (cached windows)")
print(f"  Device: {DEVICE}  |  Windows: {WIN_DIR}")
print("=" * 70)

# ── Load models ────────────────────────────────────────────────────────────────
print("\n[1] Loading models ...")
gpd_model = sbm.GPD.from_pretrained("original")
gpd_model.load_state_dict(torch.load(str(MDL_DIR/"gpd_final.pt"), map_location=DEVICE, weights_only=True))
gpd_model.to(DEVICE).eval()
GPD_WLEN  = getattr(gpd_model, "in_samples", 400)
GPD_P_IDX = gpd_model.labels.index("P")
print("    GPD fine-tuned ✓")

pn_fixed = sbm.PhaseNet.from_pretrained("original")
pn_fixed.load_state_dict(torch.load(str(MDL_DIR/"phasenet_fixed.pt"), map_location=DEVICE, weights_only=True))
pn_fixed.to(DEVICE).eval()
print("    PhaseNet fixed ✓")

# ── Helpers (same as scripts 21/31) ───────────────────────────────────────────
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

def pn_annotate(model, data):
    st = Stream()
    for i, ch in enumerate(["HHZ", "HHN", "HHE"]):
        tr = Trace(data=data[i].astype(np.float32))
        tr.stats.network = "KO"; tr.stats.station = "TEST"
        tr.stats.channel = ch;  tr.stats.sampling_rate = SRATE
        tr.stats.starttime = T0_REF; st.append(tr)
    with torch.no_grad(): ann = model.annotate(st)
    tgt = np.arange(N_TOTAL) / SRATE
    p_out = np.zeros(N_TOTAL, dtype=np.float32)
    for tr in ann:
        if not tr.stats.channel.endswith("_P"): continue
        src = float(tr.stats.starttime - T0_REF) + np.arange(len(tr.data)) / tr.stats.sampling_rate
        if len(src) < 2: continue
        f = interp1d(src, tr.data.astype(np.float32), kind="linear",
                     bounds_error=False, fill_value=0.0)
        p_out = f(tgt)
    return p_out

def detected(prob, lo, hi, thr):
    return float(prob[lo:hi+1].max()) >= thr

GPD_THR = 0.50
PNX_THR = 0.10
ENS_THR = 0.15

# ── Load and evaluate all windows ─────────────────────────────────────────────
print(f"\n[2] Scanning Marmara windows ...")
npz_files = sorted(WIN_DIR.glob("marmara_*.npz"))
print(f"    Found {len(npz_files)} windows")

rows = []
for idx, fp in enumerate(npz_files):
    d = np.load(fp, allow_pickle=True)
    data = d["data"].astype(np.float32)           # (3, 6000)
    mag  = float(d["magnitude"])
    stat = str(d["station"])

    # GPD fine-tuned
    gpd_prob = gpd_sliding_prob(gpd_model, data, GPD_WLEN, GPD_P_IDX)
    gpd_det  = detected(gpd_prob, DETECT_LO, DETECT_HI, GPD_THR)

    # PhaseNet fixed
    pnx_prob = pn_annotate(pn_fixed, data)
    pnx_det  = detected(pnx_prob, DETECT_LO, DETECT_HI, PNX_THR)

    # Ensemble: average probs, threshold
    ens_prob = (gpd_prob + pnx_prob) / 2.0
    ens_det  = detected(ens_prob, DETECT_LO, DETECT_HI, ENS_THR)

    rows.append({
        "station": stat, "magnitude": mag,
        "gpd_det": gpd_det, "pnx_det": pnx_det, "ens_det": ens_det,
    })

    if (idx + 1) % 200 == 0:
        n = len(rows)
        r_gpd = sum(r["gpd_det"] for r in rows) / n
        r_ens = sum(r["ens_det"] for r in rows) / n
        print(f"    {idx+1}/{len(npz_files)}  GPD={r_gpd:.3f}  Ens={r_ens:.3f}")

print(f"    Done: {len(rows)} windows")

# ── Aggregate ──────────────────────────────────────────────────────────────────
print("\n[3] Computing metrics ...")
import pandas as pd
df = pd.DataFrame(rows)
N = len(df)

def recall(col): return round(float(df[col].mean()), 4)

# Overall
r_gpd = recall("gpd_det")
r_pnx = recall("pnx_det")
r_ens = recall("ens_det")

print(f"\n  Marmara overall (N={N})")
print(f"    GPD-FT:          {r_gpd:.4f}")
print(f"    PhaseNet-Fixed:  {r_pnx:.4f}")
print(f"    Ensemble:        {r_ens:.4f}")

# By magnitude band
bands = [(0.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 10.0)]
band_results = {}
print(f"\n  By magnitude:")
for lo, hi in bands:
    sub = df[(df["magnitude"] >= lo) & (df["magnitude"] < hi)]
    label = f"M {lo:.1f}-{hi:.1f}"
    if len(sub) == 0:
        print(f"    {label}: N=0 (no data)")
        continue
    rg = round(float(sub["gpd_det"].mean()), 4)
    rp = round(float(sub["pnx_det"].mean()), 4)
    re = round(float(sub["ens_det"].mean()), 4)
    print(f"    {label}: N={len(sub):4d}  GPD={rg:.4f}  PN-Fixed={rp:.4f}  Ensemble={re:.4f}")
    band_results[label] = {"n": len(sub), "gpd": rg, "phasenet_fixed": rp, "ensemble": re}

# ── Kahramanmaras reference (from memory / final_evaluation_metrics.json) ──────
kah_ref = {
    "GPD-FT":         1.000,
    "PhaseNet-Fixed": 0.910,   # val recall at thr=0.10
    "Ensemble":       1.000,
}
mar_ref = {
    "GPD-FT":         r_gpd,
    "PhaseNet-Fixed": r_pnx,
    "Ensemble":       r_ens,
}

print("\n  Geographic generalization:")
print(f"  {'Model':<20} {'Kahramanmaras':>16}  {'Marmara':>10}  {'Delta':>8}")
print(f"  {'-'*60}")
for m in kah_ref:
    delta = mar_ref[m] - kah_ref[m]
    print(f"  {m:<20} {kah_ref[m]:>16.4f}  {mar_ref[m]:>10.4f}  {delta:>+8.4f}")

# ── Figure ────────────────────────────────────────────────────────────────────
print("\n[4] Producing figure ...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: overall bar comparison
models    = ["GPD-FT", "PhaseNet-Fixed", "Ensemble"]
kah_vals  = [kah_ref[m] for m in models]
mar_vals  = [mar_ref[m] for m in models]
x = np.arange(len(models))
w = 0.35
ax = axes[0]
ax.bar(x - w/2, kah_vals, w, label="Kahramanmaras (train)", color="#2196F3", alpha=0.85)
ax.bar(x + w/2, mar_vals, w, label="Marmara (deployment)", color="#FF5722", alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(models, rotation=10)
ax.set_ylabel("Recall"); ax.set_ylim(0, 1.05)
ax.set_title("Geographic Generalization: Kahramanmaras → Marmara")
ax.legend(); ax.grid(axis="y", alpha=0.3)
for xi, (kv, mv) in enumerate(zip(kah_vals, mar_vals)):
    ax.text(xi - w/2, kv + 0.01, f"{kv:.3f}", ha="center", fontsize=8)
    ax.text(xi + w/2, mv + 0.01, f"{mv:.3f}", ha="center", fontsize=8)

# Right: by magnitude band
band_labels = list(band_results.keys())
band_gpd = [band_results[b]["gpd"] for b in band_labels]
band_ens = [band_results[b]["ensemble"] for b in band_labels]
band_n   = [band_results[b]["n"]   for b in band_labels]
x2 = np.arange(len(band_labels))
ax2 = axes[1]
ax2.bar(x2 - w/2, band_gpd, w, label="GPD-FT",   color="#2196F3", alpha=0.85)
ax2.bar(x2 + w/2, band_ens, w, label="Ensemble", color="#4CAF50", alpha=0.85)
ax2.set_xticks(x2); ax2.set_xticklabels(band_labels, rotation=10)
ax2.set_ylabel("Recall"); ax2.set_ylim(0, 1.05)
ax2.set_title("Marmara Recall by Magnitude Band")
ax2.legend(); ax2.grid(axis="y", alpha=0.3)
for xi, (gv, ev, n) in enumerate(zip(band_gpd, band_ens, band_n)):
    ax2.text(xi - w/2, gv + 0.01, f"{gv:.3f}", ha="center", fontsize=7)
    ax2.text(xi + w/2, ev + 0.01, f"{ev:.3f}", ha="center", fontsize=7)
    ax2.text(xi, -0.06, f"N={n}", ha="center", fontsize=7, color="gray")

plt.tight_layout()
fig_path = FIG_DIR / "kahramanmaras_vs_marmara_performance.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"    Saved → {fig_path}")

# ── Save JSON ─────────────────────────────────────────────────────────────────
out = {
    "phase": "marmara_extended_evaluation",
    "n_windows": N,
    "n_stations": int(df["station"].nunique()),
    "stations": sorted(df["station"].unique().tolist()),
    "overall": {
        "gpd_ft_recall":         r_gpd,
        "phasenet_fixed_recall": r_pnx,
        "ensemble_recall":       r_ens,
    },
    "by_magnitude": band_results,
    "geographic_comparison": {
        m: {"kahramanmaras": kah_ref[m], "marmara": mar_ref[m],
            "delta": round(mar_ref[m] - kah_ref[m], 4)}
        for m in kah_ref
    },
    "note": (
        "Detection window: origin ± [-5s, +25s]. Origin assumed at sample 3000 "
        "(project convention). No analyst P-picks in Marmara NPZs — recall is "
        "origin-time-based. Kahramanmaras reference uses fine-tuned model metrics "
        "from final_evaluation_metrics.json."
    ),
}
json_path = ART / "marmara_extended_results.json"
with open(json_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"    Saved → {json_path}")

print("\n" + "=" * 70)
print("PROBLEM C COMPLETE — Marmara Geographic Generalization")
print(f"  Windows evaluated: {N} across {int(df['station'].nunique())} stations")
print(f"  GPD-FT recall:     {r_gpd:.4f}")
print(f"  PhaseNet-Fixed:    {r_pnx:.4f}")
print(f"  Ensemble recall:   {r_ens:.4f}")
print(f"  JSON: artifacts/marmara_extended_results.json")
print(f"  Figure: figures/kahramanmaras_vs_marmara_performance.png")
print("=" * 70)
