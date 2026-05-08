# Phase: Fine-tuning
# Purpose: Load all NPZ waveform windows (pilot + Phase 5), compute theoretical P/S
#          arrival samples via TauPy, build a SeisBench WaveformDataset (HDF5 + CSV),
#          and produce a stratified 70/15/15 train/val/test split.
# Inputs:  data/windows/*.npz, data/windows/phase5/*.npz,
#          data/catalog/koeri_pilot_catalog.csv, data/catalog/phase5_catalog.csv,
#          artifacts/station_summary.csv, artifacts/phase5_station_list.csv
# Outputs: data/finetune_dataset/waveforms.hdf5,
#          data/finetune_dataset/metadata.csv,
#          artifacts/finetune_split.json
# Limitations: TauPy uses IASP91 (1-D global velocity model); P/S labels inherit
#              the ~0.5-2 s uncertainty of the 1-D model relative to local Turkey
#              crustal structure. Windows for which TauPy fails get NaN labels (treated
#              as noise by the labeller). Phase 5 dataset has no noise windows; pilot
#              dataset includes 65 noise windows that contribute the noise class.

import json
import random
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from obspy.taup import TauPyModel
from obspy.geodetics import locations2degrees

ROOT    = Path(__file__).resolve().parent.parent
WIN_DIR = ROOT / "data" / "windows"
WIN5    = ROOT / "data" / "windows" / "phase5"
OUT_DIR = ROOT / "data" / "finetune_dataset"
ART     = ROOT / "artifacts"
OUT_DIR.mkdir(parents=True, exist_ok=True)
(ROOT / "models").mkdir(exist_ok=True)

SRATE    = 100.0
ORIGIN_S = 30.0          # origin time position in every 60-s window (samples 0-5999)
ORIGIN_SAMPLE = int(ORIGIN_S * SRATE)   # = 3000

random.seed(42)
np.random.seed(42)

print("=" * 70)
print("FINE-TUNING PREP — Build SeisBench WaveformDataset")
print("=" * 70)

# ── Step 1: Build combined event catalog lookup ────────────────────────────────
print("\n[1] Loading catalogs and station coordinates...")

cat_pilot = pd.read_csv(ROOT / "data" / "catalog" / "koeri_pilot_catalog.csv")
cat_phase5 = pd.read_csv(ROOT / "data" / "catalog" / "phase5_catalog.csv")
cat_all = pd.concat([cat_pilot, cat_phase5], ignore_index=True)
cat_all["magnitude"] = pd.to_numeric(cat_all["magnitude"], errors="coerce")
cat_all["depth_km"]  = pd.to_numeric(cat_all["depth_km"],  errors="coerce").fillna(5.0)
cat_map = {row["event_id"]: row for _, row in cat_all.iterrows()}

sta_pilot  = pd.read_csv(ART / "station_summary.csv")
sta_phase5 = pd.read_csv(ART / "phase5_station_list.csv")
sta_map = {}
for _, r in sta_pilot.iterrows():
    sta_map[f"KO.{r['station']}"] = {"lat": r["latitude"], "lon": r["longitude"]}
for _, r in sta_phase5.iterrows():
    sta_map[f"KO.{r['station']}"] = {"lat": r["latitude"], "lon": r["longitude"]}

print(f"    Catalog events: {len(cat_all)}  |  Station coords: {len(sta_map)}")

# ── Step 2: Discover all NPZ windows ──────────────────────────────────────────
print("\n[2] Discovering NPZ windows...")

npz_paths = (
    [p for p in WIN_DIR.glob("*.npz")]        # pilot: event + noise
    + [p for p in WIN5.glob("*.npz")]          # phase5: event only
)
print(f"    Found {len(npz_paths)} NPZ files total")

# ── Step 3: Load metadata from each NPZ and compute TauPy labels ──────────────
print("\n[3] Computing TauPy P/S arrival samples (IASP91)...")
taup = TauPyModel(model="iasp91")

def compute_p_s_samples(event_id, station_id):
    """Return (p_sample, s_sample) in absolute sample coordinates within 6000-sample window.
    Returns (None, None) if event or station info is missing or TauPy fails."""
    ev  = cat_map.get(str(event_id))
    sta = sta_map.get(str(station_id))
    if ev is None or sta is None:
        return None, None
    try:
        dist_deg = locations2degrees(
            float(ev["latitude"]), float(ev["longitude"]),
            sta["lat"], sta["lon"]
        )
        dep = max(0.0, float(ev["depth_km"]))
        arrs = taup.get_travel_times(dep, dist_deg, phase_list=["P", "p", "S", "s"])
        p_times = [a.time for a in arrs if a.name in ("P", "p")]
        s_times = [a.time for a in arrs if a.name in ("S", "s")]
        p_t = min(p_times) if p_times else None
        s_t = min(s_times) if s_times else None
        # Convert travel time → absolute sample in the 60-second window
        p_samp = int(round((ORIGIN_S + p_t) * SRATE)) if p_t is not None else None
        s_samp = int(round((ORIGIN_S + s_t) * SRATE)) if s_t is not None else None
        # Clamp: must fall within the 6000-sample window
        if p_samp is not None and not (0 <= p_samp < 6000):
            p_samp = None
        if s_samp is not None and not (0 <= s_samp < 6000):
            s_samp = None
        return p_samp, s_samp
    except Exception:
        return None, None

records = []        # will hold one dict per window
n_taup_ok = n_taup_fail = 0

for i, npz_path in enumerate(npz_paths):
    try:
        npz = np.load(str(npz_path), allow_pickle=True)
    except Exception:
        continue

    window_type = str(npz.get("window_type", "event"))
    station     = str(npz["station"])
    event_id    = str(npz["event_id"])
    magnitude   = float(npz["magnitude"]) if npz["magnitude"] != "" else float("nan")

    if window_type == "event" and not np.isnan(magnitude):
        p_samp, s_samp = compute_p_s_samples(event_id, station)
        if p_samp is not None:
            n_taup_ok += 1
        else:
            n_taup_fail += 1
    else:
        # Noise window — no P/S labels
        p_samp, s_samp = None, None

    records.append({
        "npz_path":                 npz_path,
        "trace_name":               npz_path.stem,
        "station":                  station,
        "event_id":                 event_id,
        "magnitude":                magnitude,
        "window_type":              window_type,
        "trace_p_arrival_sample":   p_samp,    # None → NaN in CSV → ignored by labeller
        "trace_s_arrival_sample":   s_samp,
    })

    if (i + 1) % 500 == 0:
        print(f"      {i+1}/{len(npz_paths)} processed...")

print(f"    TauPy: {n_taup_ok} OK  |  {n_taup_fail} failed (NaN labels)")

# ── Step 4: Stratified split 70 / 15 / 15 ─────────────────────────────────────
print("\n[4] Stratified split (70 / 15 / 15)...")

def stratum_label(rec):
    if rec["window_type"] != "event":
        return "noise"
    m = rec["magnitude"]
    if np.isnan(m):
        return "noise"
    if m < 3.0:
        return "M2-3"
    if m < 4.0:
        return "M3-4"
    return "M4+"

strata = defaultdict(list)
for i, rec in enumerate(records):
    strata[stratum_label(rec)].append(i)

train_idx, val_idx, test_idx = [], [], []
for stratum, indices in strata.items():
    random.shuffle(indices)
    n = len(indices)
    n_test = max(1, int(n * 0.15))
    n_val  = max(1, int(n * 0.15))
    test_idx.extend(indices[:n_test])
    val_idx.extend(indices[n_test:n_test + n_val])
    train_idx.extend(indices[n_test + n_val:])

# Assign split labels to records
split_map = {}
for i in train_idx: split_map[i] = "train"
for i in val_idx:   split_map[i] = "dev"
for i in test_idx:  split_map[i] = "test"
for i, rec in enumerate(records):
    rec["split"] = split_map.get(i, "train")

print(f"    Total: {len(records)}  →  train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")

# Class distribution per split
print("\n    Class distribution per split:")
for split in ["train", "dev", "test"]:
    sub = [r for r in records if r["split"] == split]
    for st in ["noise", "M2-3", "M3-4", "M4+"]:
        n = sum(1 for r in sub if stratum_label(r) == st)
        print(f"      {split:5s}  {st:5s}: {n}")

# ── Step 5: Write SeisBench WaveformDataset (HDF5 + CSV) ──────────────────────
print("\n[5] Writing SeisBench WaveformDataset...")

hdf5_path = OUT_DIR / "waveforms.hdf5"
meta_path = OUT_DIR / "metadata.csv"

meta_rows = []
with h5py.File(hdf5_path, "w") as hf:
    grp = hf.create_group("data")
    for rec in records:
        try:
            npz = np.load(str(rec["npz_path"]), allow_pickle=True)
            data = npz["data"].astype(np.float32)  # shape (3, 6000)
        except Exception:
            continue
        grp.create_dataset(rec["trace_name"], data=data, compression="gzip", compression_opts=4)

        meta_rows.append({
            "trace_name":               rec["trace_name"],
            "station":                  rec["station"],
            "event_id":                 rec["event_id"],
            "source_magnitude":         rec["magnitude"] if not np.isnan(rec["magnitude"]) else "",
            "window_type":              rec["window_type"],
            "trace_sampling_rate_hz":   SRATE,
            "trace_npts":               6000,
            # P/S arrival in samples — NaN if unavailable (noise or TauPy failure)
            "trace_p_arrival_sample":   rec["trace_p_arrival_sample"]
                                        if rec["trace_p_arrival_sample"] is not None else float("nan"),
            "trace_s_arrival_sample":   rec["trace_s_arrival_sample"]
                                        if rec["trace_s_arrival_sample"] is not None else float("nan"),
            "split":                    rec["split"],
            "trace_component_order":    "ZNE",   # HHZ, HHN, HHE → Z, N, E
        })

meta_df = pd.DataFrame(meta_rows)
meta_df.to_csv(meta_path, index=False)

hdf5_mb = hdf5_path.stat().st_size / 1e6
print(f"    waveforms.hdf5  : {hdf5_mb:.1f} MB  ({len(meta_rows)} traces)")
print(f"    metadata.csv    : {len(meta_df)} rows")

p_ok = meta_df["trace_p_arrival_sample"].notna().sum()
s_ok = meta_df["trace_s_arrival_sample"].notna().sum()
print(f"    P labels present: {p_ok}/{len(meta_df)}")
print(f"    S labels present: {s_ok}/{len(meta_df)}")

# ── Step 6: Save split indices ─────────────────────────────────────────────────
print("\n[6] Saving artifacts/finetune_split.json...")

split_json = {
    "n_total":    len(records),
    "n_train":    len(train_idx),
    "n_val":      len(val_idx),
    "n_test":     len(test_idx),
    "train_trace_names": [records[i]["trace_name"] for i in train_idx],
    "val_trace_names":   [records[i]["trace_name"] for i in val_idx],
    "test_trace_names":  [records[i]["trace_name"] for i in test_idx],
    "taup_p_ok":   n_taup_ok,
    "taup_p_fail": n_taup_fail,
}
with open(ART / "finetune_split.json", "w") as f:
    json.dump(split_json, f, indent=2)
print(f"    Saved artifacts/finetune_split.json")

print("\n" + "=" * 70)
print("DATASET PREP — COMPLETE")
print(f"  data/finetune_dataset/waveforms.hdf5  ({hdf5_mb:.0f} MB)")
print(f"  data/finetune_dataset/metadata.csv    ({len(meta_df)} traces)")
print(f"  artifacts/finetune_split.json")
print(f"  → Run scripts/13_finetune_phasenet.py next")
print("=" * 70)
