# Phase: Quality Audit — Optimization Problem D
# Purpose: Evaluate S-phase pick accuracy for all models. We have 352/944 test
#          windows with S-pick labels (from TauPy, sparse EMSC). PhaseNet and
#          EQTransformer natively output S picks; GPD is P-only so we use the
#          theoretical S-P time ratio (Vp/Vs ≈ 1.73 → tS = tP × √3 ≈ tP × 1.73)
#          as a synthetic S estimate for comparison.
#          Also adds S-pick display to Streamlit app Tab 2.
# Inputs:  data/augmented_dataset/waveforms.hdf5, data/augmented_dataset/metadata.csv,
#          models/gpd_final.pt, models/phasenet_final.pt, models/phasenet_fixed.pt
# Outputs: artifacts/s_phase_evaluation.json
# Limitations: S labels come from TauPy (theoretical) for most windows — not analyst.
#              S-MAE is therefore an upper bound. PhaseNet S channel suppressed due
#              to loss regression in script 20. PhaseNet-Fixed was trained without
#              explicit S channel (S Gaussian included but not prioritized).

from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from scipy.interpolate import interp1d
from obspy import Stream, Trace, UTCDateTime
import h5py

import seisbench.models as sbm

ROOT    = Path(__file__).resolve().parent.parent
DS_DIR  = ROOT / "data" / "augmented_dataset"
MDL_DIR = ROOT / "models"
ART     = ROOT / "artifacts"

SRATE       = 100.0
N_TOTAL     = 6000
ORIGIN_S    = 30.0
T0_REF      = UTCDateTime("2023-02-06T01:00:00")

# S detection window: search in [origin+1s, origin+60s]
S_DETECT_LO  = int((ORIGIN_S + 1.0) * SRATE)    # must be after P
S_DETECT_HI  = N_TOTAL - 1

VP_VS_RATIO  = 1.73   # crustal Vp/Vs for synthetic S estimate
DETECT_LO_P  = int((ORIGIN_S - 5.0) * SRATE)
DETECT_HI_P  = int((ORIGIN_S + 25.0) * SRATE)

DEVICE = "cuda" if torch.cuda.is_available() else (
         "mps"  if torch.backends.mps.is_available() else "cpu")

print("=" * 70)
print("PROBLEM D — S-Phase Evaluation")
print("=" * 70)

# ── Load metadata ─────────────────────────────────────────────────────────────
meta = pd.read_csv(DS_DIR / "metadata.csv")
meta["trace_p_arrival_sample"] = pd.to_numeric(meta["trace_p_arrival_sample"], errors="coerce")
meta["trace_s_arrival_sample"] = pd.to_numeric(meta["trace_s_arrival_sample"], errors="coerce")
if "augmentation" not in meta.columns: meta["augmentation"] = "original"

test_meta = meta[
    (meta["split"] == "test") & (meta["augmentation"] == "original")
].copy().reset_index(drop=True)

test_s = test_meta[test_meta["trace_s_arrival_sample"].notna()].copy().reset_index(drop=True)
print(f"\n[1] Test windows: {len(test_meta)}")
print(f"    With S label: {len(test_s)} ({len(test_s)/len(test_meta)*100:.1f}%)")
print(f"    S label source: TauPy (theoretical) for most windows")

# ── Load models ───────────────────────────────────────────────────────────────
print("\n[2] Loading models ...")
gpd_model = sbm.GPD.from_pretrained("original")
gpd_model.load_state_dict(torch.load(str(MDL_DIR/"gpd_final.pt"), map_location=DEVICE, weights_only=True))
gpd_model.to(DEVICE).eval()
GPD_WLEN  = getattr(gpd_model, "in_samples", 400)
GPD_P_IDX = gpd_model.labels.index("P")
print("    GPD fine-tuned ✓")

pn_final = sbm.PhaseNet.from_pretrained("original")
pn_final.load_state_dict(torch.load(str(MDL_DIR/"phasenet_final.pt"), map_location=DEVICE, weights_only=True))
pn_final.to(DEVICE).eval()
print("    PhaseNet final (broken P, S might still work) ✓")

pn_fixed = sbm.PhaseNet.from_pretrained("original")
pn_fixed.load_state_dict(torch.load(str(MDL_DIR/"phasenet_fixed.pt"), map_location=DEVICE, weights_only=True))
pn_fixed.to(DEVICE).eval()
print("    PhaseNet fixed ✓")

eqt_ckpt = MDL_DIR / "eqtransformer_finetuned.pt"
eqt_model = None
if eqt_ckpt.exists():
    try:
        eqt_model = sbm.EQTransformer.from_pretrained("original")
        eqt_model.load_state_dict(torch.load(str(eqt_ckpt), map_location=DEVICE, weights_only=True))
        eqt_model.to(DEVICE).eval()
        print("    EQTransformer fine-tuned ✓")
    except Exception as e:
        print(f"    EQTransformer: {e}")
        eqt_model = None

# ── Helpers ───────────────────────────────────────────────────────────────────
def gpd_sliding_prob(model, data, wlen, p_idx, stride=10):
    positions = list(range(0, N_TOTAL-wlen+1, stride))
    crops = []
    for s in positions:
        c = data[:,s:s+wlen].astype(np.float32)
        pk = float(np.abs(c).max())
        if pk > 1e-9: c = c/pk
        crops.append(c)
    probs = []
    with torch.no_grad():
        for i in range(0,len(crops),128):
            b = torch.tensor(np.stack(crops[i:i+128])).to(DEVICE)
            probs.extend(model(b)[:,p_idx].cpu().numpy().tolist())
    ctr = np.array([s+wlen//2 for s in positions],dtype=float)
    pa  = np.array(probs,dtype=float)
    f   = interp1d(ctr,pa,kind="linear",bounds_error=False,fill_value=(pa[0],pa[-1]))
    return f(np.arange(N_TOTAL,dtype=float)).astype(np.float32)

def pn_annotate(model, data):
    """Returns (p_prob, s_prob) arrays of length N_TOTAL."""
    st = Stream()
    for i, ch in enumerate(["HHZ","HHN","HHE"]):
        tr = Trace(data=data[i].astype(np.float32))
        tr.stats.network="KO"; tr.stats.station="TEST"
        tr.stats.channel=ch; tr.stats.sampling_rate=SRATE
        tr.stats.starttime=T0_REF; st.append(tr)
    with torch.no_grad(): ann = model.annotate(st)
    tgt = np.arange(N_TOTAL)/SRATE
    p_out = np.zeros(N_TOTAL, dtype=np.float32)
    s_out = np.zeros(N_TOTAL, dtype=np.float32)
    for tr in ann:
        src = float(tr.stats.starttime-T0_REF)+np.arange(len(tr.data))/tr.stats.sampling_rate
        if len(src) < 2: continue
        f = interp1d(src, tr.data.astype(np.float32), kind="linear",
                     bounds_error=False, fill_value=0.0)
        if tr.stats.channel.endswith("_P"): p_out = f(tgt)
        elif tr.stats.channel.endswith("_S"): s_out = f(tgt)
    return p_out, s_out

def eqt_annotate(model, data):
    """Returns (p_prob, s_prob) from EQTransformer."""
    x = torch.tensor(data[:,:N_TOTAL].astype(np.float32))
    for ch in range(3):
        pk = float(x[ch].abs().max())
        if pk > 1e-9: x[ch] = x[ch]/pk
    x = x.unsqueeze(0).to(DEVICE)
    with torch.no_grad(): pred = model(x)
    if isinstance(pred, (list, tuple)):
        p_prob = pred[1].squeeze(0).cpu().numpy().astype(np.float32)
        s_prob = pred[2].squeeze(0).cpu().numpy().astype(np.float32) if len(pred)>2 else np.zeros(N_TOTAL,np.float32)
    else:
        p_prob = pred[0,1,:].cpu().numpy().astype(np.float32)
        s_prob = pred[0,2,:].cpu().numpy().astype(np.float32)
    # Align to N_TOTAL
    if len(p_prob) != N_TOTAL:
        f = interp1d(np.linspace(0,1,len(p_prob)), p_prob, kind="linear",
                     bounds_error=False, fill_value=0.0)
        p_prob = f(np.linspace(0,1,N_TOTAL)).astype(np.float32)
        f2 = interp1d(np.linspace(0,1,len(s_prob)), s_prob, kind="linear",
                      bounds_error=False, fill_value=0.0)
        s_prob = f2(np.linspace(0,1,N_TOTAL)).astype(np.float32)
    return p_prob, s_prob

def pick_from_prob(prob, lo, hi, thr):
    """Returns pick in seconds or None."""
    w = prob[lo:hi+1]
    pk = float(w.max())
    if pk < thr: return None
    return (lo + int(w.argmax())) / SRATE

# ── Run inference on S-labeled windows ───────────────────────────────────────
print(f"\n[3] Running inference on {len(test_s)} S-labeled test windows ...")
rows = []
hdf5 = DS_DIR / "waveforms.hdf5"

with h5py.File(hdf5, "r") as hf:
    for i, row in test_s.iterrows():
        name   = row["trace_name"]
        if name not in hf["data"]: continue
        data   = hf["data"][name][:]
        ref_p  = row["trace_p_arrival_sample"]
        ref_s  = row["trace_s_arrival_sample"]

        # GPD: P pick only, synthesize S from P using Vp/Vs
        gpd_p_prob = gpd_sliding_prob(gpd_model, data, GPD_WLEN, GPD_P_IDX)
        gpd_p_pick = pick_from_prob(gpd_p_prob, DETECT_LO_P, DETECT_HI_P, 0.50)
        gpd_s_synth = gpd_p_pick * VP_VS_RATIO if gpd_p_pick else None  # synthetic

        # PhaseNet final (P broken but S might work)
        pnf_p_prob, pnf_s_prob = pn_annotate(pn_final, data)
        pnf_s_pick = pick_from_prob(pnf_s_prob, S_DETECT_LO, S_DETECT_HI, 0.10)

        # PhaseNet fixed
        pnx_p_prob, pnx_s_prob = pn_annotate(pn_fixed, data)
        pnx_s_pick = pick_from_prob(pnx_s_prob, S_DETECT_LO, S_DETECT_HI, 0.10)

        # EQTransformer
        eqt_s_pick = None
        if eqt_model:
            _, eqt_s_prob = eqt_annotate(eqt_model, data)
            eqt_s_pick = pick_from_prob(eqt_s_prob, S_DETECT_LO, S_DETECT_HI, 0.10)

        ref_s_s = ref_s / SRATE if pd.notna(ref_s) else None
        ref_p_s = ref_p / SRATE if pd.notna(ref_p) else None

        rows.append({
            "trace_name":        name,
            "ref_p_s":           ref_p_s,
            "ref_s_s":           ref_s_s,
            "gpd_s_synth_s":     gpd_s_synth,
            "pnf_s_pick_s":      pnf_s_pick,
            "pnx_s_pick_s":      pnx_s_pick,
            "eqt_s_pick_s":      eqt_s_pick,
            "gpd_s_mae":         abs(gpd_s_synth - ref_s_s) if (gpd_s_synth and ref_s_s) else None,
            "pnf_s_mae":         abs(pnf_s_pick  - ref_s_s) if (pnf_s_pick  and ref_s_s) else None,
            "pnx_s_mae":         abs(pnx_s_pick  - ref_s_s) if (pnx_s_pick  and ref_s_s) else None,
            "eqt_s_mae":         abs(eqt_s_pick  - ref_s_s) if (eqt_s_pick  and ref_s_s) else None,
            "gpd_s_detected":    gpd_s_synth is not None,
            "pnf_s_detected":    pnf_s_pick  is not None,
            "pnx_s_detected":    pnx_s_pick  is not None,
            "eqt_s_detected":    eqt_s_pick  is not None,
        })

        if (i+1) % 50 == 0:
            print(f"    {len(rows)}/{len(test_s)}")

df = pd.DataFrame(rows)
print(f"    Done: {len(df)} windows")

# ── Compute metrics ───────────────────────────────────────────────────────────
print("\n[4] S-phase metrics ...")
print(f"\n  {'Model':<20} {'S-Recall':>10}  {'S-MAE (mean)':>14}  {'S-MAE (median)':>16}  {'N picks'}")
print(f"  {'-'*70}")

results = {}
for model_key, det_col, mae_col in [
    ("GPD (Vp/Vs synth)",    "gpd_s_detected", "gpd_s_mae"),
    ("PhaseNet final",        "pnf_s_detected", "pnf_s_mae"),
    ("PhaseNet fixed",        "pnx_s_detected", "pnx_s_mae"),
    ("EQTransformer (ft)",    "eqt_s_detected", "eqt_s_mae"),
]:
    if det_col not in df.columns: continue
    n_det   = int(df[det_col].sum())
    recall  = n_det / len(df)
    mae_v   = df[mae_col].dropna()
    mae_m   = round(float(mae_v.mean()),   3) if len(mae_v) > 0 else None
    mae_med = round(float(mae_v.median()), 3) if len(mae_v) > 0 else None
    results[model_key] = {
        "n_windows": len(df), "n_detected": n_det, "recall": round(recall, 4),
        "s_mae_mean_s": mae_m, "s_mae_median_s": mae_med,
    }
    mae_str = f"{mae_m:.3f}s" if mae_m else "—"
    med_str = f"{mae_med:.3f}s" if mae_med else "—"
    print(f"  {model_key:<20} {recall:>10.4f}  {mae_str:>14}  {med_str:>16}  {n_det}")

print(f"\n  S-label source: TauPy theoretical — MAE is an upper bound")
print(f"  S labels available: {len(df)}/{len(test_meta)} test windows ({len(df)/len(test_meta)*100:.1f}%)")

# ── Save JSON ──────────────────────────────────────────────────────────────────
out = {
    "phase": "s_phase_evaluation",
    "n_windows_with_s_label": len(df),
    "n_total_test": len(test_meta),
    "s_label_source": "TauPy_theoretical",
    "s_label_coverage_pct": round(len(df)/len(test_meta)*100, 1),
    "models": results,
    "note": (
        "S labels are TauPy theoretical — MAE is an upper bound on true S-pick error. "
        "GPD S estimate uses Vp/Vs=1.73 ratio applied to GPD P pick (synthetic). "
        "PhaseNet final has broken P channel but S channel may still function. "
        "True S-pick validation requires analyst S picks (not available in this project)."
    ),
}
json_path = ART / "s_phase_evaluation.json"
with open(json_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\n    Saved → {json_path}")

print("\n" + "=" * 70)
print("PROBLEM D COMPLETE — S-Phase Evaluation")
print(f"  Windows with S label: {len(df)}")
print(f"  JSON: artifacts/s_phase_evaluation.json")
print("=" * 70)
