# Phase: Quality Audit — Optimization Problem 2
# Purpose: Re-train PhaseNet with corrected loss function to fix recall regression.
#          Root cause: original script 20 loss sums over all 6000 timesteps equally,
#          causing noise channel to dominate and suppress P probability.
#          Fix: (1) windowed 3001-sample crops with P at random position [200,2800],
#               (2) Gaussian sigma=10 (narrower label), (3) focal loss (gamma=2),
#               (4) cosine annealing LR, (5) only P-labeled original windows in train.
# Inputs:  data/augmented_dataset/waveforms.hdf5, data/augmented_dataset/metadata.csv
# Outputs: models/phasenet_fixed.pt, artifacts/phasenet_fixed_log.csv
# Limitations: Uses same P labels as script 20 (EMSC-corrected where available,
#              TauPy otherwise). S channel is not fine-tuned here (P only).

import csv
import time
import random
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import seisbench.models as sbm

ROOT    = Path(__file__).resolve().parent.parent
DS_DIR  = ROOT / "data" / "augmented_dataset"
MDL_DIR = ROOT / "models"
ART     = ROOT / "artifacts"
MDL_DIR.mkdir(exist_ok=True)

# ── Hyper-parameters ────────────────────────────────────────────────────────
BATCH_SIZE  = 32
MAX_EPOCHS  = 40
LR_INIT     = 3e-4
LR_MIN      = 1e-6
T_MAX       = 40           # cosine annealing period = full training
GAMMA_FOCAL = 2.0
SIGMA_GAUSS = 10           # narrower Gaussian (was implicitly ~10 before, now enforced)
CROP_LEN    = 3001         # shorter crop → P context only, avoids noise dominance
P_MIN_POS   = 200          # minimum sample position for P within crop
P_MAX_POS   = 2800         # maximum sample position for P within crop
NPTS_FULL   = 6000
NUM_WORKERS = 0
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

DEVICE = "cuda" if torch.cuda.is_available() else (
         "mps"  if torch.backends.mps.is_available() else "cpu")

print("=" * 70)
print("PROBLEM 2 — PhaseNet Fixed Fine-tuning")
print(f"  Device: {DEVICE}")
print(f"  Crop length: {CROP_LEN} samples ({CROP_LEN/100:.1f}s)")
print(f"  Gaussian sigma: {SIGMA_GAUSS} samples ({SIGMA_GAUSS/100*1000:.0f} ms)")
print(f"  Focal loss gamma: {GAMMA_FOCAL}")
print(f"  LR: {LR_INIT} → {LR_MIN} (cosine, T_max={T_MAX})")
print("=" * 70)

# ── Load metadata ────────────────────────────────────────────────────────────
meta = pd.read_csv(DS_DIR / "metadata.csv")
meta["trace_p_arrival_sample"] = pd.to_numeric(meta["trace_p_arrival_sample"], errors="coerce")
if "augmentation" not in meta.columns:
    meta["augmentation"] = "original"

hdf5_path = DS_DIR / "waveforms.hdf5"
n_total   = len(meta)
n_p_label = meta["trace_p_arrival_sample"].notna().sum()
print(f"\n  Metadata rows: {n_total}  |  P-labeled: {n_p_label}")

# ── Gaussian label ────────────────────────────────────────────────────────────
def gaussian_label(length, center, sigma):
    t = np.arange(length, dtype=np.float32)
    g = np.exp(-0.5 * ((t - center) / sigma) ** 2)
    return g.astype(np.float32)

# ── Dataset ───────────────────────────────────────────────────────────────────
class PhaseNetWindowedDataset(Dataset):
    """
    For each P-labeled original trace, extract a CROP_LEN-sample sub-window
    with the P arrival placed at a random position within [P_MIN_POS, P_MAX_POS].
    This prevents positional overfitting and removes noise-channel dominance.

    Val/test: P placed at center of crop (deterministic).
    Only P-labeled windows are included (no noise-only windows in training).
    """
    def __init__(self, hdf5_path, meta_df, split, augmented=False):
        rows = meta_df[meta_df["split"] == split].copy()
        # Always use original-only for val/test; for train, can include augmented
        if not augmented or split != "train":
            rows = rows[rows["augmentation"] == "original"]
        # Only P-labeled windows
        rows = rows[rows["trace_p_arrival_sample"].notna()].reset_index(drop=True)

        is_train = (split == "train")
        print(f"    [{split:5s}] {len(rows)} P-labeled windows  (augmented={augmented and is_train})")
        print(f"      Preloading...", end="", flush=True)

        t0 = time.time()
        self.X    = np.empty((len(rows), 3, CROP_LEN), dtype=np.float32)
        self.Y    = np.empty((len(rows), CROP_LEN),    dtype=np.float32)
        self.p_in_crop = []

        rng = np.random.default_rng(RANDOM_SEED if not is_train else None)

        with h5py.File(str(hdf5_path), "r") as hf:
            for k, (_, row) in enumerate(rows.iterrows()):
                name   = row["trace_name"]
                p_samp = int(row["trace_p_arrival_sample"])

                # Determine P position within crop
                if is_train:
                    p_in_crop = int(rng.integers(P_MIN_POS, P_MAX_POS + 1))
                else:
                    p_in_crop = CROP_LEN // 2

                # Crop start in the full trace
                crop_start = p_samp - p_in_crop
                crop_end   = crop_start + CROP_LEN

                # Clamp to valid range
                if crop_start < 0:
                    crop_start = 0
                    crop_end   = CROP_LEN
                    p_in_crop  = p_samp - crop_start
                if crop_end > NPTS_FULL:
                    crop_end   = NPTS_FULL
                    crop_start = NPTS_FULL - CROP_LEN
                    p_in_crop  = p_samp - crop_start

                # Sanity-clamp p_in_crop
                p_in_crop = int(np.clip(p_in_crop, 0, CROP_LEN - 1))

                full_trace = hf["data"][name][:, :NPTS_FULL].astype(np.float32)
                crop = full_trace[:, crop_start:crop_end]

                # Pad if needed (edge case)
                if crop.shape[1] < CROP_LEN:
                    pad = CROP_LEN - crop.shape[1]
                    crop = np.pad(crop, ((0,0),(0,pad)), constant_values=0)

                # Peak-normalize
                pk = float(np.abs(crop).max())
                if pk > 1e-9:
                    crop = crop / pk

                self.X[k]    = crop
                self.Y[k]    = gaussian_label(CROP_LEN, p_in_crop, SIGMA_GAUSS)
                self.p_in_crop.append(p_in_crop)

        elapsed = time.time() - t0
        print(f" done ({elapsed:.1f}s, {self.X.nbytes/1e6:.0f} MB)")

    def __len__(self):  return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.Y[idx])

print("\n[1] Building datasets ...")
ds_train = PhaseNetWindowedDataset(hdf5_path, meta, "train", augmented=False)
ds_val   = PhaseNetWindowedDataset(hdf5_path, meta, "dev",   augmented=False)

train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, drop_last=True)
val_loader   = DataLoader(ds_val,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS)
print(f"    Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")

# ── Load pretrained PhaseNet ──────────────────────────────────────────────────
print("\n[2] Loading pretrained PhaseNet ...")
try:
    model = sbm.PhaseNet.from_pretrained("original")
    print("    ✓ from_pretrained('original')")
except Exception as e:
    print(f"    Warning: {e}")
    model = sbm.PhaseNet()
model.to(DEVICE)

# Adapt first conv to accept CROP_LEN input if needed (PhaseNet is fully conv, OK)
optimizer = torch.optim.Adam(model.parameters(), lr=LR_INIT)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=T_MAX, eta_min=LR_MIN)

# ── Focal loss on P channel ──────────────────────────────────────────────────
def focal_loss_pn(pred, p_target, gamma=GAMMA_FOCAL):
    """
    pred:     (B, 3, T) — PhaseNet output (P=0, S=1, N=2 channels, softmax)
    p_target: (B, T)    — P Gaussian in [0, 1]

    Focal loss for P channel and complement N channel.
    Focal weight (1-p_t)^gamma downweights easy "noise" timesteps automatically.
    """
    p_pred = pred[:, 0, :].clamp(1e-7, 1 - 1e-7)   # (B, T)
    n_pred = pred[:, 2, :].clamp(1e-7, 1 - 1e-7)   # (B, T)
    n_target = 1.0 - p_target

    # P channel focal loss
    p_bce  = -(p_target * torch.log(p_pred) + (1 - p_target) * torch.log(1 - p_pred))
    p_pt   = torch.where(p_target >= 0.5, p_pred, 1.0 - p_pred)
    p_fl   = ((1 - p_pt) ** gamma) * p_bce

    # N channel focal loss (complement)
    n_bce  = -(n_target * torch.log(n_pred) + (1 - n_target) * torch.log(1 - n_pred))
    n_pt   = torch.where(n_target >= 0.5, n_pred, 1.0 - n_pred)
    n_fl   = ((1 - n_pt) ** gamma) * n_bce

    return (p_fl.mean() + n_fl.mean()) * 0.5

# ── Training loop ────────────────────────────────────────────────────────────
print("\n[3] Training ...")
best_val  = float("inf")
log_rows  = []
ckpt_path = MDL_DIR / "phasenet_fixed.pt"
PATIENCE  = 8
patience  = 0

for epoch in range(1, MAX_EPOCHS + 1):
    t0 = time.time()
    model.train()
    tr_losses = []
    for X, y in train_loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        pred = model(X)
        # pred shape: (B, 3, T_out) — PhaseNet may return different T if crop≠6000
        # Align T dimension
        T_out = pred.shape[2]
        y_trim = y[:, :T_out]
        loss = focal_loss_pn(pred, y_trim)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        tr_losses.append(loss.item())

    model.eval()
    vl_losses = []
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred  = model(X)
            T_out = pred.shape[2]
            y_trim = y[:, :T_out]
            vl_losses.append(focal_loss_pn(pred, y_trim).item())

    tr_loss = float(np.mean(tr_losses))
    vl_loss = float(np.mean(vl_losses))
    lr_now  = optimizer.param_groups[0]["lr"]
    elapsed = time.time() - t0
    scheduler.step()

    saved = ""
    if vl_loss < best_val:
        best_val  = vl_loss
        patience  = 0
        torch.save(model.state_dict(), ckpt_path)
        saved = " ✓ saved"
    else:
        patience += 1
        saved = f" (patience {patience}/{PATIENCE})"

    print(f"  Ep {epoch:3d}/{MAX_EPOCHS}  train={tr_loss:.4f}  val={vl_loss:.4f}  "
          f"lr={lr_now:.2e}  ({elapsed:.0f}s){saved}")
    log_rows.append({"epoch": epoch, "train_loss": round(tr_loss,6),
                     "val_loss": round(vl_loss,6), "lr": lr_now,
                     "elapsed_s": round(elapsed,1)})

    if patience >= PATIENCE:
        print(f"  Early stopping at epoch {epoch}")
        break

# Save log
log_path = ART / "phasenet_fixed_log.csv"
with open(log_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=log_rows[0].keys())
    w.writeheader(); w.writerows(log_rows)

best_ep = min(log_rows, key=lambda r: r["val_loss"])["epoch"]
print(f"\n  Best val_loss={best_val:.4f} at epoch {best_ep}")
print(f"  Checkpoint: models/phasenet_fixed.pt")

# ── Quick recall test on val set ─────────────────────────────────────────────
print("\n[4] Quick recall check on val set ...")
model.eval()
# Load the best checkpoint
model.load_state_dict(torch.load(str(ckpt_path), map_location=DEVICE, weights_only=True))

ORIGIN_S   = 30.0
DETECT_TOL = 5.0
DETECT_POST= 25.0
DETECT_LO  = int((ORIGIN_S - DETECT_TOL) * 100)   # for full 6000-sample traces
DETECT_HI  = int((ORIGIN_S + DETECT_POST) * 100)

# For val set: evaluate on FULL 6000-sample traces (as used in final eval)
from scipy.interpolate import interp1d
from obspy import Stream, Trace, UTCDateTime as UDT

T0_REF = UDT("2023-02-06T01:00:00")

val_meta = meta[(meta["split"] == "dev") & (meta["augmentation"] == "original") &
                (meta["trace_p_arrival_sample"].notna())].copy().reset_index(drop=True)

n_det = 0
with h5py.File(str(hdf5_path), "r") as hf:
    for _, row in val_meta.iterrows():
        data = hf["data"][row["trace_name"]][:, :NPTS_FULL].astype(np.float32)
        # Use PhaseNet annotate() on full trace
        st = Stream()
        for i, ch in enumerate(["HHZ", "HHN", "HHE"]):
            tr = Trace(data=data[i])
            tr.stats.network = "KO"; tr.stats.station = "TEST"
            tr.stats.channel = ch; tr.stats.sampling_rate = 100.0
            tr.stats.starttime = T0_REF
            st.append(tr)
        with torch.no_grad():
            ann = model.annotate(st)
        tr_p = next((t for t in ann if t.stats.channel.endswith("_P")), None)
        if tr_p is None: continue
        tgt = np.arange(NPTS_FULL) / 100.0
        src = float(tr_p.stats.starttime - T0_REF) + np.arange(len(tr_p.data)) / tr_p.stats.sampling_rate
        if len(src) < 2: continue
        f = interp1d(src, tr_p.data.astype(np.float32), kind="linear",
                     bounds_error=False, fill_value=0.0)
        prob = f(tgt)
        if float(prob[DETECT_LO:DETECT_HI+1].max()) >= 0.30:
            n_det += 1

recall_val = n_det / len(val_meta) if len(val_meta) > 0 else 0
print(f"  Val P-labeled windows: {len(val_meta)}")
print(f"  Detected (thr=0.30):   {n_det}")
print(f"  Val recall:            {recall_val:.4f}")

if recall_val >= 0.65:
    print(f"  ✓ TARGET MET (recall ≥ 0.65)")
else:
    print(f"  ⚠ Below target (recall < 0.65) — may need further tuning or threshold adjustment")

print("\n" + "=" * 70)
print("PROBLEM 2 COMPLETE")
print(f"  Model:      models/phasenet_fixed.pt")
print(f"  Log:        artifacts/phasenet_fixed_log.csv")
print(f"  Val recall: {recall_val:.4f}  (threshold 0.30 on full 6000-sample traces)")
print("=" * 70)
