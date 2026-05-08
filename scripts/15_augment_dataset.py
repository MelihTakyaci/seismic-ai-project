# Phase: Fine-tuning
# Purpose: Augment the clean P-labeled training windows from data/finetune_dataset/
#          to produce a larger training set for fine-tuning PhaseNet.
#          Applies 4 augmentation types (8 copies per window) to training windows only.
#          Val and test splits remain identical to finetune_dataset (no data leakage).
# Inputs:  data/finetune_dataset/waveforms.hdf5, data/finetune_dataset/metadata.csv
# Outputs: data/augmented_dataset/waveforms.hdf5, data/augmented_dataset/metadata.csv
# Limitations: Only windows with valid trace_p_arrival_sample are used (same filter
#              as scripts/13_finetune_phasenet.py v2). Augmented windows are labelled
#              with the original TauPy P/S samples adjusted where necessary (time shift).
#              Gaussian noise levels (SNR 20/10/5 dB) are relative to per-channel RMS.
#              Channel shuffle swaps HHN↔HHE only; HHZ is preserved.

import json
import random
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT    = Path(__file__).resolve().parent.parent
IN_DIR  = ROOT / "data" / "finetune_dataset"
OUT_DIR = ROOT / "data" / "augmented_dataset"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SRATE   = 100.0      # Hz
NPTS    = 6000       # samples per window (60 s)

# Random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

print("=" * 70)
print("DATA AUGMENTATION — Expanding clean P-labeled training windows")
print("=" * 70)

# ── Step 1: Load metadata, filter to P-labeled windows ────────────────────────
print("\n[1] Loading metadata...")
meta = pd.read_csv(IN_DIR / "metadata.csv")
meta["trace_p_arrival_sample"] = pd.to_numeric(meta["trace_p_arrival_sample"], errors="coerce")
meta["trace_s_arrival_sample"] = pd.to_numeric(meta["trace_s_arrival_sample"], errors="coerce")

# Only windows with a valid TauPy P label
p_valid = meta["trace_p_arrival_sample"].notna()
meta_labeled = meta[p_valid].copy().reset_index(drop=True)

# Split breakdown
for split in ["train", "dev", "test"]:
    n = (meta_labeled["split"] == split).sum()
    print(f"    {split:5s}: {n} P-labeled windows")

train_meta = meta_labeled[meta_labeled["split"] == "train"].copy()
print(f"\n    Augmenting {len(train_meta)} training windows × 8 copies = "
      f"{len(train_meta) * 8:,} augmented windows")
print(f"    Total training after augmentation: "
      f"{len(train_meta) + len(train_meta) * 8:,}")

# ── Step 2: Augmentation functions ────────────────────────────────────────────
print("\n[2] Defining augmentation functions...")

def add_gaussian_noise(data, snr_db):
    """Add Gaussian noise at a given SNR (dB) relative to per-channel RMS."""
    out = data.copy()
    for ch in range(data.shape[0]):
        rms = float(np.sqrt(np.mean(data[ch] ** 2)))
        if rms < 1e-9:
            continue  # skip silent channels
        noise_rms = rms / (10 ** (snr_db / 20.0))
        noise = np.random.normal(0.0, noise_rms, data.shape[1]).astype(np.float32)
        out[ch] = data[ch] + noise
    return out

def time_shift(data, p_samp, s_samp, max_shift_samples=30):
    """
    Apply a random time shift of ±max_shift_samples to the window.
    The window is shifted by rolling the data array and zero-padding the wrapped edge.
    P/S labels are adjusted accordingly.
    Returns (shifted_data, new_p_samp, new_s_samp).
    new_p_samp / new_s_samp is None if the label falls outside [0, NPTS).
    """
    delta = np.random.randint(-max_shift_samples, max_shift_samples + 1)
    if delta == 0:
        delta = max_shift_samples // 2   # avoid no-op; use half max as fallback

    shifted = np.roll(data, -delta, axis=1).astype(np.float32)

    # Zero out the wrapped-around edge introduced by roll
    if delta > 0:
        shifted[:, -delta:] = 0.0   # end of window
    elif delta < 0:
        shifted[:, :-delta] = 0.0   # start of window

    new_p = int(p_samp) - delta if p_samp is not None and not np.isnan(p_samp) else None
    new_s = int(s_samp) - delta if s_samp is not None and not np.isnan(s_samp) else None

    new_p = new_p if (new_p is not None and 0 <= new_p < NPTS) else None
    new_s = new_s if (new_s is not None and 0 <= new_s < NPTS) else None

    return shifted, new_p, new_s

def amplitude_scale(data, scale):
    """Scale all channels by a constant factor. Labels unchanged."""
    return (data * scale).astype(np.float32)

def channel_shuffle(data):
    """Swap HHN (channel 1) and HHE (channel 2). HHZ (channel 0) is preserved."""
    out = data.copy()
    out[1] = data[2]
    out[2] = data[1]
    return out

# ── Step 3: Write augmented HDF5 + metadata ───────────────────────────────────
print("\n[3] Writing augmented dataset to data/augmented_dataset/...")

out_hdf5 = OUT_DIR / "waveforms.hdf5"
out_meta  = OUT_DIR / "metadata.csv"

# Pre-build augmentation spec for training windows.
# Each entry: (suffix, aug_fn, adjusts_labels)
AUG_SPECS = [
    ("noise20",  lambda d, p, s: (add_gaussian_noise(d, 20), p, s)),
    ("noise10",  lambda d, p, s: (add_gaussian_noise(d, 10), p, s)),
    ("noise5",   lambda d, p, s: (add_gaussian_noise(d,  5), p, s)),
    ("shift",    lambda d, p, s: time_shift(d, p, s, max_shift_samples=30)),
    ("scale05",  lambda d, p, s: (amplitude_scale(d, 0.5),  p, s)),
    ("scale15",  lambda d, p, s: (amplitude_scale(d, 1.5),  p, s)),
    ("scale20",  lambda d, p, s: (amplitude_scale(d, 2.0),  p, s)),
    ("shuffle",  lambda d, p, s: (channel_shuffle(d),        p, s)),
]

meta_rows   = []   # will hold metadata for all output windows
n_aug_ok    = 0
n_aug_skip  = 0

# Build a quick lookup: trace_name → row from original metadata
all_labeled_names = set(meta_labeled["trace_name"])

with h5py.File(IN_DIR / "waveforms.hdf5", "r") as hf_in, \
     h5py.File(out_hdf5, "w") as hf_out:

    grp_out = hf_out.create_group("data")

    # ── Pass 1: copy ALL P-labeled windows (all splits) unchanged ──────────────
    print("    Pass 1: copying original P-labeled windows...")
    n_copied = 0
    for _, row in meta_labeled.iterrows():
        name = row["trace_name"]
        if name not in hf_in["data"]:
            continue
        data = hf_in["data"][name][:]
        grp_out.create_dataset(name, data=data, compression="gzip", compression_opts=4)
        meta_rows.append({
            "trace_name":               name,
            "station":                  row["station"],
            "event_id":                 row["event_id"],
            "source_magnitude":         row["source_magnitude"],
            "window_type":              row["window_type"],
            "trace_sampling_rate_hz":   SRATE,
            "trace_npts":               NPTS,
            "trace_p_arrival_sample":   row["trace_p_arrival_sample"],
            "trace_s_arrival_sample":   row["trace_s_arrival_sample"],
            "split":                    row["split"],
            "trace_component_order":    "ZNE",
            "augmentation":             "original",
        })
        n_copied += 1

    print(f"      Copied {n_copied} original windows.")

    # ── Pass 2: augment training windows only ──────────────────────────────────
    print(f"    Pass 2: augmenting {len(train_meta)} training windows...")
    for idx, (_, row) in enumerate(train_meta.iterrows()):
        name = row["trace_name"]
        if name not in hf_in["data"]:
            continue

        data    = hf_in["data"][name][:].astype(np.float32)
        p_samp  = row["trace_p_arrival_sample"]   # may be NaN but we filtered above
        s_samp  = row["trace_s_arrival_sample"]   # may be NaN — that's fine

        for suffix, aug_fn in AUG_SPECS:
            aug_name = f"aug_{suffix}_{name}"
            try:
                aug_data, new_p, new_s = aug_fn(data, p_samp, s_samp)

                # Skip if the time-shift augmentation produced an out-of-window P
                if suffix == "shift" and new_p is None:
                    n_aug_skip += 1
                    continue

                grp_out.create_dataset(aug_name, data=aug_data,
                                       compression="gzip", compression_opts=4)

                meta_rows.append({
                    "trace_name":               aug_name,
                    "station":                  row["station"],
                    "event_id":                 row["event_id"],
                    "source_magnitude":         row["source_magnitude"],
                    "window_type":              "event",
                    "trace_sampling_rate_hz":   SRATE,
                    "trace_npts":               NPTS,
                    "trace_p_arrival_sample":   new_p if new_p is not None else float("nan"),
                    "trace_s_arrival_sample":   new_s if new_s is not None else float("nan"),
                    "split":                    "train",
                    "trace_component_order":    "ZNE",
                    "augmentation":             suffix,
                })
                n_aug_ok += 1

            except Exception as e:
                n_aug_skip += 1

        if (idx + 1) % 500 == 0:
            print(f"      {idx+1}/{len(train_meta)} training windows processed...")

    print(f"      Augmented: {n_aug_ok} ok  |  {n_aug_skip} skipped")

# ── Step 4: Write metadata CSV ─────────────────────────────────────────────────
print("\n[4] Writing metadata.csv...")
meta_df = pd.DataFrame(meta_rows)
meta_df.to_csv(out_meta, index=False)

# ── Step 5: Summary ────────────────────────────────────────────────────────────
hdf5_mb = out_hdf5.stat().st_size / 1e6

print("\n" + "=" * 70)
print("DATA AUGMENTATION — COMPLETE")
print(f"  Output HDF5           : data/augmented_dataset/waveforms.hdf5  ({hdf5_mb:.0f} MB)")
print(f"  Output metadata       : data/augmented_dataset/metadata.csv    ({len(meta_df)} rows)")
print()
print("  Dataset breakdown:")
for split in ["train", "dev", "test"]:
    orig = ((meta_df["split"] == split) & (meta_df["augmentation"] == "original")).sum()
    aug  = ((meta_df["split"] == split) & (meta_df["augmentation"] != "original")).sum()
    print(f"    {split:5s} : {orig:>5} original  +  {aug:>6} augmented  = {orig+aug:>6} total")

print()
print("  Augmentation type breakdown (training only):")
for suffix, _ in AUG_SPECS:
    n = (meta_df["augmentation"] == suffix).sum()
    print(f"    {suffix:<10}: {n}")

print()
print("  P-label coverage:")
p_ok    = meta_df["trace_p_arrival_sample"].notna().sum()
p_total = len(meta_df)
print(f"    P present: {p_ok}/{p_total}  ({100*p_ok/p_total:.1f}%)")
print("=" * 70)
print("  → Update scripts/13_finetune_phasenet.py to point to")
print("    data/augmented_dataset/ instead of data/finetune_dataset/")
print("=" * 70)
