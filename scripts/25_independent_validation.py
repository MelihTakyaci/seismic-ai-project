# Phase: Quality Audit — Step 4
# Purpose: Temporal independence validation. Evaluates model performance on
#          windows from the last 8 weeks of the recording period (June–August 2023),
#          which represents a different temporal regime from the peak aftershock
#          period (Feb–May 2023). Also evaluates on the 127 events exclusive to
#          the test split (never seen in training in any form).
# Inputs:  data/augmented_dataset/waveforms.hdf5, data/augmented_dataset/metadata.csv,
#          data/catalog/phase5_catalog.csv, models/gpd_final.pt,
#          models/phasenet_final.pt
# Outputs: artifacts/independent_validation.json
# Limitations: Temporal split is approximate — events in June–Aug 2023 may overlap
#              with training events at different stations. True temporal independence
#              would require a strict cutoff applied before any training split.
#              Station-level generalization is NOT tested here (all stations in train).

from pathlib import Path
import json

import h5py
import numpy as np
import pandas as pd
import torch
from scipy.interpolate import interp1d
from obspy import Stream, Trace, UTCDateTime

import seisbench.models as sbm

ROOT    = Path(__file__).resolve().parent.parent
DS_DIR  = ROOT / "data" / "augmented_dataset"
MDL_DIR = ROOT / "models"
ART     = ROOT / "artifacts"
CAT_F   = ROOT / "data" / "catalog" / "phase5_catalog.csv"

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

print("=" * 70)
print("INDEPENDENT VALIDATION — Temporal + Event-Isolated Splits")
print("=" * 70)

# ── Load data ──────────────────────────────────────────────────────────────
meta = pd.read_csv(DS_DIR / "metadata.csv")
meta["trace_p_arrival_sample"] = pd.to_numeric(meta["trace_p_arrival_sample"], errors="coerce")
meta["source_magnitude"]       = pd.to_numeric(meta["source_magnitude"],       errors="coerce")
if "augmentation" not in meta.columns:
    meta["augmentation"] = "original"
meta["event_id_str"] = meta["event_id"].astype(str)

# Catalog for temporal metadata
cat = pd.read_csv(CAT_F)
cat["event_id_str"]  = cat["event_id"].astype(str)
cat["origin_time"]   = pd.to_datetime(cat["origin_time"], utc=True)

# Test set
test_meta = meta[
    (meta["split"] == "test") & (meta["augmentation"] == "original")
].copy().reset_index(drop=True)
test_meta = test_meta.merge(
    cat[["event_id_str", "origin_time"]].drop_duplicates(),
    on="event_id_str", how="left"
)
print(f"\n[1] Test windows: {len(test_meta)}")
print(f"    With origin_time: {test_meta['origin_time'].notna().sum()}")

# ── Define temporal splits ─────────────────────────────────────────────────
# Peak period: Feb 6 – May 31, 2023 (active aftershock sequence)
# Late period: Jun 1 – Aug 6, 2023 (decaying aftershock sequence)
CUTOFF = pd.Timestamp("2023-06-01", tz="UTC")
early_mask = test_meta["origin_time"] < CUTOFF
late_mask  = test_meta["origin_time"] >= CUTOFF
print(f"    Early period (Feb–May 2023): {early_mask.sum()} windows")
print(f"    Late  period (Jun–Aug 2023): {late_mask.sum()} windows")

# ── Define event-isolated split ────────────────────────────────────────────
train_meta  = meta[meta["split"] == "train"]
train_events= set(train_meta["event_id_str"])
test_events = set(test_meta["event_id_str"])
excl_events = test_events - train_events
excl_mask   = test_meta["event_id_str"].isin(excl_events)
print(f"    Event-exclusive (unseen events): {excl_mask.sum()} windows")

# ── Load models ────────────────────────────────────────────────────────────
print("\n[2] Loading models...")
gpd_model = sbm.GPD.from_pretrained("original")
gpd_ckpt  = next(p for p in [MDL_DIR/"gpd_final.pt", MDL_DIR/"gpd_koeri_finetuned.pt"]
                 if p.exists())
gpd_model.load_state_dict(torch.load(str(gpd_ckpt), map_location=DEVICE, weights_only=True))
gpd_model.to(DEVICE).eval()

pn_model = sbm.PhaseNet.from_pretrained("original")
pn_ckpt  = next(p for p in [MDL_DIR/"phasenet_final.pt", MDL_DIR/"phasenet_koeri_finetuned.pt"]
                if p.exists())
pn_model.load_state_dict(torch.load(str(pn_ckpt), map_location=DEVICE, weights_only=True))
pn_model.to(DEVICE).eval()
print(f"    GPD ({gpd_ckpt.name}) + PhaseNet ({pn_ckpt.name})")

GPD_WLEN  = getattr(gpd_model, "in_samples", 400)
GPD_P_IDX = gpd_model.labels.index("P")

# ── Helpers ────────────────────────────────────────────────────────────────
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

def pn_prob(model, data):
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

def detect_and_pick(prob, thr):
    w = prob[DETECT_LO:DETECT_HI+1]
    pk = float(w.max())
    if pk < thr: return False, None
    return True, (DETECT_LO + int(w.argmax())) / SRATE

# ── Run inference ──────────────────────────────────────────────────────────
print("\n[3] Running inference on test set...")
rows = []
hdf5 = DS_DIR / "waveforms.hdf5"

with h5py.File(hdf5, "r") as hf:
    for i, row in test_meta.iterrows():
        name = row["trace_name"]
        if name not in hf["data"]: continue
        data = hf["data"][name][:]
        mag  = row["source_magnitude"]
        ref_p= row["trace_p_arrival_sample"]
        is_early = bool(early_mask.loc[i]) if i in early_mask.index else None
        is_excl  = bool(excl_mask.loc[i]) if i in excl_mask.index else None

        gpd_p = gpd_prob(gpd_model, data, GPD_WLEN, GPD_P_IDX)
        pn_p  = pn_prob(pn_model, data)
        ens_p = 0.5 * gpd_p + 0.5 * pn_p

        gpd_det, gpd_pk = detect_and_pick(gpd_p, 0.50)
        ens_det, ens_pk = detect_and_pick(ens_p, 0.15)

        gpd_mae = round(abs(gpd_pk - ref_p/SRATE), 3) if (gpd_pk and pd.notna(ref_p)) else None
        ens_mae = round(abs(ens_pk - ref_p/SRATE), 3) if (ens_pk and pd.notna(ref_p)) else None

        rows.append({
            "trace_name": name,
            "magnitude": float(mag) if pd.notna(mag) else None,
            "is_early_period": is_early,
            "is_excl_event":   is_excl,
            "gpd_detected": gpd_det, "gpd_pick_s": gpd_pk, "gpd_mae_s": gpd_mae,
            "ens_detected": ens_det, "ens_pick_s": ens_pk, "ens_mae_s": ens_mae,
        })
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(test_meta)}")

df = pd.DataFrame(rows)
print(f"    Done: {len(df)} windows")

# ── Compute split metrics ──────────────────────────────────────────────────
print("\n[4] Computing metrics across splits...")

def band_metrics(sub, detected_col, mae_col):
    if len(sub) == 0: return {}
    n_det  = int(sub[detected_col].sum())
    recall = round(n_det / len(sub), 4)
    mae_v  = sub[mae_col].dropna()
    p_mae  = round(float(mae_v.mean()), 3) if len(mae_v) > 0 else None
    return {"n": len(sub), "n_detected": n_det, "recall": recall, "p_mae_s": p_mae}

SPLITS = {
    "full_test":     df,
    "early_period":  df[df["is_early_period"] == True],
    "late_period":   df[df["is_early_period"] == False],
    "event_isolated":df[df["is_excl_event"]   == True],
    "event_shared":  df[df["is_excl_event"]   == False],
}

results = {}
for split_name, sub in SPLITS.items():
    results[split_name] = {
        "gpd_ft":  band_metrics(sub, "gpd_detected", "gpd_mae_s"),
        "ensemble":band_metrics(sub, "ens_detected", "ens_mae_s"),
    }

print(f"\n  {'Split':<25} {'N':>5}  {'GPD-FT Recall':>14}  {'Ens Recall':>11}  {'GPD P-MAE':>10}")
print(f"  {'-'*70}")
for sname, sr in results.items():
    gm = sr["gpd_ft"]
    em = sr["ensemble"]
    print(f"  {sname:<25} {gm.get('n',0):>5}  "
          f"{gm.get('recall','—'):>14}  {em.get('recall','—'):>11}  "
          f"{gm.get('p_mae_s','—'):>10}")

# Check temporal degradation
early_r = results["early_period"]["gpd_ft"].get("recall")
late_r  = results["late_period"]["gpd_ft"].get("recall")
excl_r  = results["event_isolated"]["gpd_ft"].get("recall")

print(f"\n  Temporal stability:")
if early_r and late_r:
    print(f"    Early (Feb–May): {early_r:.4f}  Late (Jun–Aug): {late_r:.4f}  "
          f"Δ={late_r-early_r:+.4f}")
    if abs(late_r - early_r) < 0.05:
        print("    ✓ PASS — <5% degradation from early to late period.")
    else:
        print("    ⚠ WARNING — >5% temporal degradation detected.")

print(f"\n  Event-isolation stability:")
if excl_r:
    full_r = results["full_test"]["gpd_ft"].get("recall")
    print(f"    Full test: {full_r:.4f}  Exclusive events only: {excl_r:.4f}  "
          f"Δ={excl_r-full_r:+.4f}")
    if abs(excl_r - full_r) < 0.05:
        print("    ✓ PASS — Model generalises to unseen events.")
    else:
        print("    ⚠ WARNING — Recall drops on unseen events.")

# ── Save JSON ──────────────────────────────────────────────────────────────
out = {
    "phase": "independent_validation",
    "n_test_windows": len(df),
    "temporal_cutoff": str(CUTOFF),
    "splits": {k: {"n": len(v)} for k, v in SPLITS.items()},
    "results": results,
}
with open(ART / "independent_validation.json", "w") as f:
    json.dump(out, f, indent=2)

print("\n" + "=" * 70)
print("INDEPENDENT VALIDATION — COMPLETE")
print(f"  JSON: artifacts/independent_validation.json")
print("=" * 70)
