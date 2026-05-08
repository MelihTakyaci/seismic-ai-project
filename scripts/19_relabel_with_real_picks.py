# Phase: Fine-tuning / Re-labelling
# Purpose: Replace TauPy theoretical P-labels with real EMSC analyst picks for matched
#          windows in augmented_dataset/metadata.csv and finetune_dataset/metadata.csv.
#          Propagates the per-window correction to all derived augmented windows.
# Inputs:  artifacts/koeri_real_picks.csv, data/augmented_dataset/metadata.csv,
#          data/finetune_dataset/metadata.csv
# Outputs: data/augmented_dataset/metadata.csv (updated in-place, backup saved),
#          data/finetune_dataset/metadata.csv (updated in-place, backup saved),
#          artifacts/relabel_report.csv
# Limitations: Only P-phase picks are updated (EMSC S picks sparse).
#              Augmented windows derived from shuffle augmentation have shuffled channel
#              order but identical time axis, so same correction applies.
#              Correction is applied additively — windows where new sample < 0 or >= 6000
#              are flagged as invalid and left at original TauPy value.

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT   = Path(__file__).resolve().parent.parent
ART    = ROOT / "artifacts"
AUG_F  = ROOT / "data" / "augmented_dataset" / "metadata.csv"
FIN_F  = ROOT / "data" / "finetune_dataset"   / "metadata.csv"
NPTS   = 6000

print("=" * 70)
print("RE-LABELLING — Replace TauPy P picks with EMSC analyst picks")
print("=" * 70)

# ── Step 1: Load EMSC picks ─────────────────────────────────────────────────
print("\n[1] Loading EMSC picks...")
picks_df = pd.read_csv(ART / "koeri_real_picks.csv")
p_picks  = picks_df[picks_df["phase"] == "P"].copy()
p_picks["p_sample_in_window"] = pd.to_numeric(p_picks["p_sample_in_window"], errors="coerce")
p_picks  = p_picks[p_picks["p_sample_in_window"].notna()]
p_picks["event_id"] = p_picks["event_id"].astype(str)
p_picks["sta_bare"] = p_picks["station"].str.strip()   # EMSC gives bare station name
print(f"    EMSC P picks loaded  : {len(p_picks)}")
print(f"    Unique events        : {p_picks['event_id'].nunique()}")
print(f"    Unique stations      : {p_picks['sta_bare'].nunique()}")

# Build lookup: (event_id, sta_bare) → emsc_p_sample
emsc_lookup = {}
for _, row in p_picks.iterrows():
    key = (row["event_id"], row["sta_bare"])
    emsc_lookup[key] = int(round(row["p_sample_in_window"]))

print(f"    Lookup entries (event+sta pairs): {len(emsc_lookup)}")

# ── Step 2: Load augmented_dataset metadata ─────────────────────────────────
print("\n[2] Loading augmented_dataset metadata...")
meta = pd.read_csv(AUG_F)
meta["trace_p_arrival_sample"] = pd.to_numeric(meta["trace_p_arrival_sample"], errors="coerce")

# Backup
bak_path = AUG_F.with_suffix(".csv.bak_pre_relabel")
shutil.copy(AUG_F, bak_path)
print(f"    Backup saved: {bak_path.name}")
print(f"    Total rows: {len(meta)}")

orig_mask = meta["augmentation"] == "original"
orig_meta = meta[orig_mask].copy()
print(f"    Original windows: {len(orig_meta)}")

# Parse station bare name from "KO.KOZT" → "KOZT"
meta["sta_bare"] = meta["station"].str.split(".").str[-1]
meta["event_id_str"] = meta["event_id"].astype(str)

# ── Step 3: Build augmented trace_name → original trace_name mapping ────────
print("\n[3] Building original → augmented index mapping...")
# Augmented trace names follow pattern: aug_{suffix}_{original_trace_name}
# Suffix can contain underscores (e.g. noise20, scale05, shift, shuffle)
# Extract original_name by stripping "aug_{suffix}_" prefix where augmentation != original
aug_mask = meta["augmentation"] != "original"
aug_meta = meta[aug_mask].copy()

def extract_original_name(aug_name, aug_suffix):
    prefix = f"aug_{aug_suffix}_"
    if aug_name.startswith(prefix):
        return aug_name[len(prefix):]
    return None

aug_meta["original_name"] = aug_meta.apply(
    lambda r: extract_original_name(r["trace_name"], r["augmentation"]), axis=1
)
n_unresolved = aug_meta["original_name"].isna().sum()
if n_unresolved > 0:
    print(f"    [!] {n_unresolved} augmented rows could not resolve original name")

# Map original trace_name → row index in meta (global index)
orig_name_to_idx = {}
for idx, row in meta[orig_mask].iterrows():
    orig_name_to_idx[row["trace_name"]] = idx

# Map original trace_name → list of augmented row indices
orig_to_aug_indices = {}
for idx, row in aug_meta.iterrows():
    oname = row["original_name"]
    if oname is not None:
        orig_to_aug_indices.setdefault(oname, []).append(idx)

print(f"    Original names with augmented children: {len(orig_to_aug_indices)}")

# ── Step 4: Apply corrections ────────────────────────────────────────────────
print("\n[4] Applying EMSC corrections to original windows + propagating...")

n_matched        = 0
n_no_pick        = 0
n_out_of_bounds  = 0
n_small_error    = 0  # |correction| < 5 samples (0.05s)
corrections_list = []
report_rows      = []

for idx, row in meta[orig_mask].iterrows():
    ev_id   = str(row["event_id"])
    sta     = row["sta_bare"]
    taupy_p = row["trace_p_arrival_sample"]
    key     = (ev_id, sta)

    if key not in emsc_lookup:
        n_no_pick += 1
        report_rows.append({
            "trace_name":  row["trace_name"],
            "event_id":    ev_id,
            "station":     row["station"],
            "taupy_p":     taupy_p,
            "emsc_p":      np.nan,
            "correction":  np.nan,
            "status":      "no_emsc_pick",
        })
        continue

    emsc_p = emsc_lookup[key]

    # Validate bounds
    if emsc_p < 0 or emsc_p >= NPTS:
        n_out_of_bounds += 1
        report_rows.append({
            "trace_name":  row["trace_name"],
            "event_id":    ev_id,
            "station":     row["station"],
            "taupy_p":     taupy_p,
            "emsc_p":      emsc_p,
            "correction":  emsc_p - taupy_p if pd.notna(taupy_p) else np.nan,
            "status":      "emsc_out_of_bounds",
        })
        continue

    correction = (emsc_p - taupy_p) if pd.notna(taupy_p) else np.nan
    n_matched += 1
    if pd.notna(correction) and abs(correction) < 5:
        n_small_error += 1
    corrections_list.append(correction)

    # Update original window
    meta.at[idx, "trace_p_arrival_sample"] = float(emsc_p)

    # Propagate to augmented children
    for aug_idx in orig_to_aug_indices.get(row["trace_name"], []):
        aug_old_p = meta.at[aug_idx, "trace_p_arrival_sample"]
        if pd.notna(aug_old_p) and pd.notna(correction):
            new_p = float(aug_old_p) + correction
            if 0 <= new_p < NPTS:
                meta.at[aug_idx, "trace_p_arrival_sample"] = new_p
            # If out of bounds, leave at old value

    report_rows.append({
        "trace_name":  row["trace_name"],
        "event_id":    ev_id,
        "station":     row["station"],
        "taupy_p":     taupy_p,
        "emsc_p":      emsc_p,
        "correction":  correction,
        "status":      "updated",
    })

corrections_arr = np.array([c for c in corrections_list if not np.isnan(c)])

print(f"    Original windows matched  : {n_matched}")
print(f"    No EMSC pick found        : {n_no_pick}")
print(f"    EMSC pick out of bounds   : {n_out_of_bounds}")
print(f"    |correction| < 5 samples  : {n_small_error} ({100*n_small_error/max(n_matched,1):.1f}%)")
if len(corrections_arr) > 0:
    print(f"    Correction (samples):  mean={corrections_arr.mean():.1f}  "
          f"median={np.median(corrections_arr):.1f}  "
          f"std={corrections_arr.std():.1f}  "
          f"max|err|={np.abs(corrections_arr).max():.0f}")
    print(f"    Correction (seconds):  mean={corrections_arr.mean()/100:.3f}s  "
          f"median={np.median(corrections_arr)/100:.3f}s")

# Drop helper columns before saving
meta.drop(columns=["sta_bare", "event_id_str"], inplace=True)

# ── Step 5: Save updated augmented_dataset metadata ─────────────────────────
print("\n[5] Saving updated augmented_dataset/metadata.csv...")
meta.to_csv(AUG_F, index=False)
print(f"    Saved: {AUG_F} ({len(meta)} rows)")

# ── Step 6: Update finetune_dataset metadata ─────────────────────────────────
print("\n[6] Updating finetune_dataset/metadata.csv...")
if FIN_F.exists():
    fin_meta = pd.read_csv(FIN_F)
    fin_meta["trace_p_arrival_sample"] = pd.to_numeric(
        fin_meta["trace_p_arrival_sample"], errors="coerce")

    # Backup
    shutil.copy(FIN_F, FIN_F.with_suffix(".csv.bak_pre_relabel"))

    fin_meta["sta_bare"] = fin_meta["station"].str.split(".").str[-1]
    fin_meta["event_id_str"] = fin_meta["event_id"].astype(str)

    n_fin_updated = 0
    for idx, row in fin_meta.iterrows():
        key = (row["event_id_str"], row["sta_bare"])
        if key in emsc_lookup:
            emsc_p = emsc_lookup[key]
            if 0 <= emsc_p < NPTS:
                fin_meta.at[idx, "trace_p_arrival_sample"] = float(emsc_p)
                n_fin_updated += 1

    fin_meta.drop(columns=["sta_bare", "event_id_str"], inplace=True)
    fin_meta.to_csv(FIN_F, index=False)
    print(f"    Updated {n_fin_updated}/{len(fin_meta)} rows in finetune_dataset/metadata.csv")
else:
    print(f"    [!] {FIN_F} not found — skipping")

# ── Step 7: Save report ──────────────────────────────────────────────────────
print("\n[7] Saving relabel report...")
report_df = pd.DataFrame(report_rows)
report_path = ART / "relabel_report.csv"
report_df.to_csv(report_path, index=False)
print(f"    Saved: {report_path} ({len(report_df)} rows)")

# Distribution of corrections
if len(corrections_arr) > 0:
    bins = [-500, -100, -50, -20, -10, -5, 0, 5, 10, 20, 50, 100, 500]
    hist, _ = np.histogram(corrections_arr, bins=bins)
    print("\n    Correction distribution (samples):")
    for lo, hi, cnt in zip(bins[:-1], bins[1:], hist):
        if cnt > 0:
            bar = "#" * min(cnt // 10, 50)
            print(f"      [{lo:+5d},{hi:+5d}): {cnt:5d}  {bar}")

# ── Step 8: Summary ──────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("RE-LABELLING — COMPLETE")
print(f"  Windows re-labelled   : {n_matched} / {len(orig_meta)} original windows")
print(f"  Augmented propagated  : {sum(len(v) for v in orig_to_aug_indices.values())} augmented windows affected")
print(f"  TauPy median error    : {np.median(corrections_arr)/100:.3f}s (corrected)")
print(f"  Backups               : metadata.csv.bak_pre_relabel in both dataset dirs")
print(f"  Report                : artifacts/relabel_report.csv")
print(f"  → Run scripts/20_final_ensemble_finetune.py next")
print("=" * 70)
