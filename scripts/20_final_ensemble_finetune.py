# Phase: Fine-tuning (Final)
# Purpose: Re-train PhaseNet and GPD on EMSC-corrected labels from augmented_dataset.
#          Produces phasenet_final.pt and gpd_final.pt — the two models used in
#          the final ensemble evaluation (script 21) and Streamlit demo (app.py).
# Inputs:  data/augmented_dataset/waveforms.hdf5, data/augmented_dataset/metadata.csv
#          (must have been updated by script 19)
# Outputs: models/phasenet_final.pt, models/gpd_final.pt,
#          artifacts/phasenet_final_log.csv, artifacts/gpd_final_log.csv
# Limitations: PhaseNet requires time-series output — loss is mean cross-entropy over
#              all time steps; GPD uses per-window 3-class loss.
#              S labels remain TauPy-derived (EMSC S picks were sparse).

import csv
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

import seisbench.models as sbm

ROOT    = Path(__file__).resolve().parent.parent
DS_DIR  = ROOT / "data" / "augmented_dataset"
MDL_DIR = ROOT / "models"
ART     = ROOT / "artifacts"
MDL_DIR.mkdir(exist_ok=True)

# ── Shared hyper-parameters ────────────────────────────────────────────────
BATCH_SIZE  = 32
MAX_EPOCHS  = 30
LR          = 1e-4
PATIENCE    = 5
LR_FACTOR   = 0.5
LR_PATIENCE = 3
NUM_WORKERS = 0
NPTS        = 6000

DEVICE = "cuda" if torch.cuda.is_available() else (
         "mps"  if torch.backends.mps.is_available() else "cpu")

print("=" * 70)
print("FINAL ENSEMBLE FINE-TUNING (EMSC-corrected labels)")
print(f"  Device: {DEVICE}  |  Batch: {BATCH_SIZE}  |  Max epochs: {MAX_EPOCHS}")
print("=" * 70)

# ── Load metadata ──────────────────────────────────────────────────────────
meta = pd.read_csv(DS_DIR / "metadata.csv")
meta["trace_p_arrival_sample"] = pd.to_numeric(meta["trace_p_arrival_sample"], errors="coerce")
meta["trace_s_arrival_sample"] = pd.to_numeric(meta["trace_s_arrival_sample"], errors="coerce")
if "augmentation" not in meta.columns:
    meta["augmentation"] = "original"

hdf5_path = DS_DIR / "waveforms.hdf5"
n_p = meta["trace_p_arrival_sample"].notna().sum()
print(f"\n  Metadata rows: {len(meta)}  |  P-labeled: {n_p}")

# ════════════════════════════════════════════════════════════════════════════
#  PART A — PhaseNet Fine-tuning
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("PART A: PhaseNet Fine-tuning → models/phasenet_final.pt")
print("─" * 70)

# ── Gaussian label builder ──────────────────────────────────────────────────
def gaussian_label(length, center, sigma=10):
    t = np.arange(length, dtype=np.float32)
    g = np.exp(-0.5 * ((t - center) / sigma) ** 2).astype(np.float32)
    return g

class PhaseNetDataset(Dataset):
    """
    Each original P-labeled row → P Gaussian label on full 6000-sample trace.
    Augmented rows inherit same label (with corrected sample).
    Noise windows (no P label) → zero Gaussian label.
    """
    def __init__(self, hdf5_path, meta_df, split):
        rows = meta_df[meta_df["split"] == split].copy()
        if split != "train":
            rows = rows[rows["augmentation"] == "original"]
        rows = rows.reset_index(drop=True)

        # Build specs: (trace_name, p_sample or NaN)
        specs = []
        for _, r in rows.iterrows():
            specs.append((r["trace_name"], r["trace_p_arrival_sample"]))

        print(f"    [{split:5s}] {len(specs)} windows")
        print(f"      Preloading into memory...", end="", flush=True)

        t0 = time.time()
        self.X = np.empty((len(specs), 3, NPTS), dtype=np.float32)
        self.Y = np.zeros((len(specs), NPTS),    dtype=np.float32)

        with h5py.File(str(hdf5_path), "r") as hf:
            for k, (name, p_samp) in enumerate(specs):
                trace = hf["data"][name][:, :NPTS].astype(np.float32)
                peak  = float(np.abs(trace).max())
                if peak > 1e-9:
                    trace = trace / peak
                self.X[k] = trace
                if pd.notna(p_samp):
                    self.Y[k] = gaussian_label(NPTS, float(p_samp))

        elapsed = time.time() - t0
        print(f" done ({elapsed:.1f}s, {self.X.nbytes/1e6:.0f} MB)")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.Y[idx])

print("\n[A1] Building PhaseNet datasets...")
pn_train = PhaseNetDataset(hdf5_path, meta, "train")
pn_val   = PhaseNetDataset(hdf5_path, meta, "dev")
pn_train_loader = DataLoader(pn_train, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, drop_last=True)
pn_val_loader   = DataLoader(pn_val,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS)
print(f"    Train batches: {len(pn_train_loader)}  |  Val batches: {len(pn_val_loader)}")

print("\n[A2] Loading pretrained PhaseNet...")
try:
    pn_model = sbm.PhaseNet.from_pretrained("original")
    print("    from_pretrained('original') — OK")
except Exception as e:
    print(f"    Warning: {e}")
    pn_model = sbm.PhaseNet()
    print("    Fell back to default PhaseNet init")
pn_model.to(DEVICE)

pn_optimizer = torch.optim.Adam(pn_model.parameters(), lr=LR)
pn_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    pn_optimizer, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE)

def pn_loss_fn(pred, target):
    # pred: (B, 3, T) — channel 0=P, channel 1=S, channel 2=Noise
    # target: (B, T) — P Gaussian
    p_pred  = pred[:, 0, :]
    n_pred  = pred[:, 2, :]
    p_tgt   = target
    n_tgt   = 1.0 - target
    loss_p  = torch.mean(-torch.sum(p_tgt  * torch.log(p_pred.clamp(1e-7)), dim=1))
    loss_n  = torch.mean(-torch.sum(n_tgt  * torch.log(n_pred.clamp(1e-7)), dim=1))
    return (loss_p + loss_n) * 0.5

print("\n[A3] Training PhaseNet...")
best_pn_val = float("inf")
pn_patience = 0
pn_log      = []
pn_ckpt     = MDL_DIR / "phasenet_final.pt"

for epoch in range(1, MAX_EPOCHS + 1):
    t0 = time.time()
    pn_model.train()
    train_losses = []
    for X, y in pn_train_loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        pn_optimizer.zero_grad()
        pred = pn_model(X)
        loss = pn_loss_fn(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pn_model.parameters(), max_norm=1.0)
        pn_optimizer.step()
        train_losses.append(loss.item())

    pn_model.eval()
    val_losses = []
    with torch.no_grad():
        for X, y in pn_val_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = pn_model(X)
            val_losses.append(pn_loss_fn(pred, y).item())

    tr_loss  = float(np.mean(train_losses))
    vl_loss  = float(np.mean(val_losses))
    elapsed  = time.time() - t0
    lr_now   = pn_optimizer.param_groups[0]["lr"]
    print(f"  Ep {epoch:3d}/{MAX_EPOCHS}  train={tr_loss:.4f}  val={vl_loss:.4f}  "
          f"lr={lr_now:.2e}  ({elapsed:.0f}s)")

    pn_log.append({"epoch": epoch, "train_loss": round(tr_loss, 6),
                   "val_loss": round(vl_loss, 6), "lr": lr_now,
                   "elapsed_s": round(elapsed, 1)})

    if vl_loss < best_pn_val:
        best_pn_val = vl_loss
        pn_patience = 0
        torch.save(pn_model.state_dict(), pn_ckpt)
        print(f"    ✓ New best val={best_pn_val:.4f} → saved")
    else:
        pn_patience += 1
        print(f"    No improvement ({pn_patience}/{PATIENCE})")
        if pn_patience >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}")
            break
    pn_scheduler.step(vl_loss)

log_path = ART / "phasenet_final_log.csv"
with open(log_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=pn_log[0].keys())
    writer.writeheader(); writer.writerows(pn_log)

best_pn_ep = min(pn_log, key=lambda r: r["val_loss"])["epoch"]
print(f"\n  PhaseNet final: best val={best_pn_val:.4f} at epoch {best_pn_ep}")
print(f"  Checkpoint: models/phasenet_final.pt")

# Free memory before GPD training
del pn_train, pn_val, pn_train_loader, pn_val_loader
import gc; gc.collect()
if DEVICE == "mps":
    torch.mps.empty_cache()
elif DEVICE == "cuda":
    torch.cuda.empty_cache()

# ════════════════════════════════════════════════════════════════════════════
#  PART B — GPD Fine-tuning
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("PART B: GPD Fine-tuning → models/gpd_final.pt")
print("─" * 70)

print("\n[B1] Loading pretrained GPD...")
try:
    gpd_model = sbm.GPD.from_pretrained("original")
    print("    from_pretrained('original') — OK")
except Exception as e:
    print(f"    Warning: {e}")
    gpd_model = sbm.GPD()
    print("    Fell back to default GPD init")
gpd_model.to(DEVICE)

GPD_WLEN   = getattr(gpd_model, "in_samples", 400)
labels_str = gpd_model.labels
P_IDX = labels_str.index("P")
S_IDX = labels_str.index("S")
N_IDX = labels_str.index("N")
NUM_CLS = len(labels_str)
print(f"    in_samples={GPD_WLEN}  labels={labels_str}  P={P_IDX},S={S_IDX},N={N_IDX}")

class GPDWindowDataset(Dataset):
    def __init__(self, hdf5_path, meta_df, split, wlen, p_idx, s_idx, n_idx, num_cls):
        rows = meta_df[meta_df["split"] == split].copy()
        if split != "train":
            rows = rows[rows["augmentation"] == "original"]
        rows = rows[rows["trace_p_arrival_sample"].notna()].reset_index(drop=True)

        half  = wlen // 2
        specs = []

        for _, row in rows.iterrows():
            name   = row["trace_name"]
            p_samp = int(row["trace_p_arrival_sample"])
            s_samp = row["trace_s_arrival_sample"]

            # P window
            ps, pe = p_samp - half, p_samp - half + wlen
            if ps < 0:   ps, pe = 0, wlen
            elif pe > NPTS: pe, ps = NPTS, NPTS - wlen
            if pe <= NPTS:
                specs.append((name, ps, pe, p_idx))

            # Noise window
            if p_samp > wlen + 100:
                specs.append((name, 0, wlen, n_idx))

            # S window
            if not pd.isna(s_samp):
                si = int(s_samp)
                if abs(si - p_samp) > wlen:
                    ss, se = si - half, si - half + wlen
                    if ss < 0:   ss, se = 0, wlen
                    elif se > NPTS: se, ss = NPTS, NPTS - wlen
                    if se <= NPTS:
                        specs.append((name, ss, se, s_idx))

        counts = {p_idx: 0, s_idx: 0, n_idx: 0}
        for _, _, _, lbl in specs:
            counts[lbl] += 1
        n_spec = len(specs)
        print(f"    [{split:5s}] {len(rows)} traces → {n_spec} windows "
              f"(P={counts[p_idx]}, S={counts[s_idx]}, N={counts[n_idx]})")
        print(f"      Preloading...", end="", flush=True)

        t0 = time.time()
        self.X = np.empty((n_spec, 3, wlen), dtype=np.float32)
        self.Y = np.zeros((n_spec, num_cls), dtype=np.float32)
        with h5py.File(str(hdf5_path), "r") as hf:
            for k, (name, start, end, label) in enumerate(specs):
                crop = hf["data"][name][:, start:end].astype(np.float32)
                peak = float(np.abs(crop).max())
                if peak > 1e-9:
                    crop = crop / peak
                self.X[k] = crop
                self.Y[k, label] = 1.0
        elapsed = time.time() - t0
        print(f" done ({elapsed:.1f}s, {self.X.nbytes/1e6:.0f} MB)")

    def __len__(self):  return len(self.X)
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.Y[idx])

print("\n[B2] Building GPD datasets...")
gpd_train = GPDWindowDataset(hdf5_path, meta, "train", GPD_WLEN, P_IDX, S_IDX, N_IDX, NUM_CLS)
gpd_val   = GPDWindowDataset(hdf5_path, meta, "dev",   GPD_WLEN, P_IDX, S_IDX, N_IDX, NUM_CLS)
gpd_train_loader = DataLoader(gpd_train, batch_size=64, shuffle=True,
                               num_workers=NUM_WORKERS, drop_last=True)
gpd_val_loader   = DataLoader(gpd_val,   batch_size=64, shuffle=False,
                               num_workers=NUM_WORKERS)
print(f"    Train batches: {len(gpd_train_loader)}  |  Val batches: {len(gpd_val_loader)}")

gpd_optimizer = torch.optim.Adam(gpd_model.parameters(), lr=LR)
gpd_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    gpd_optimizer, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE)

def soft_cross_entropy(pred, target):
    return torch.mean(-torch.sum(target * torch.log(pred.clamp(1e-7)), dim=1))

print("\n[B3] Training GPD...")
best_gpd_val = float("inf")
gpd_patience = 0
gpd_log      = []
gpd_ckpt     = MDL_DIR / "gpd_final.pt"

for epoch in range(1, MAX_EPOCHS + 1):
    t0 = time.time()
    gpd_model.train()
    train_losses = []
    for X, y in gpd_train_loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        gpd_optimizer.zero_grad()
        pred = gpd_model(X)
        loss = soft_cross_entropy(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gpd_model.parameters(), max_norm=1.0)
        gpd_optimizer.step()
        train_losses.append(loss.item())

    gpd_model.eval()
    val_losses = []
    with torch.no_grad():
        for X, y in gpd_val_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = gpd_model(X)
            val_losses.append(soft_cross_entropy(pred, y).item())

    tr_loss  = float(np.mean(train_losses))
    vl_loss  = float(np.mean(val_losses))
    elapsed  = time.time() - t0
    lr_now   = gpd_optimizer.param_groups[0]["lr"]
    print(f"  Ep {epoch:3d}/{MAX_EPOCHS}  train={tr_loss:.4f}  val={vl_loss:.4f}  "
          f"lr={lr_now:.2e}  ({elapsed:.0f}s)")

    gpd_log.append({"epoch": epoch, "train_loss": round(tr_loss, 6),
                    "val_loss": round(vl_loss, 6), "lr": lr_now,
                    "elapsed_s": round(elapsed, 1)})

    if vl_loss < best_gpd_val:
        best_gpd_val = vl_loss
        gpd_patience = 0
        torch.save(gpd_model.state_dict(), gpd_ckpt)
        print(f"    ✓ New best val={best_gpd_val:.4f} → saved")
    else:
        gpd_patience += 1
        print(f"    No improvement ({gpd_patience}/{PATIENCE})")
        if gpd_patience >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}")
            break
    gpd_scheduler.step(vl_loss)

log_path = ART / "gpd_final_log.csv"
with open(log_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=gpd_log[0].keys())
    writer.writeheader(); writer.writerows(gpd_log)

best_gpd_ep = min(gpd_log, key=lambda r: r["val_loss"])["epoch"]
print(f"\n  GPD final: best val={best_gpd_val:.4f} at epoch {best_gpd_ep}")
print(f"  Checkpoint: models/gpd_final.pt")

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("FINAL ENSEMBLE FINE-TUNING — COMPLETE")
print(f"  PhaseNet best val_loss: {best_pn_val:.4f}  (epoch {best_pn_ep})  → models/phasenet_final.pt")
print(f"  GPD      best val_loss: {best_gpd_val:.4f}  (epoch {best_gpd_ep})  → models/gpd_final.pt")
print(f"  Logs: artifacts/phasenet_final_log.csv, artifacts/gpd_final_log.csv")
print(f"  → Run scripts/21_final_evaluation.py next")
print("=" * 70)
