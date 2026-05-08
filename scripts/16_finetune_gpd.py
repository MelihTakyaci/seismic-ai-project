# Phase: Fine-tuning
# Purpose: Fine-tune pretrained GPD on the KOERI Kahramanmaraş augmented dataset.
#          Creates P-centered (label=P), noise (label=N), and S-centered (label=S)
#          400-sample windows from each 6000-sample HDF5 trace. Early stopping on val loss.
# Inputs:  data/augmented_dataset/waveforms.hdf5, data/augmented_dataset/metadata.csv
# Outputs: models/gpd_koeri_finetuned.pt, artifacts/gpd_finetune_log.csv
# Limitations: GPD labels are "PSN" (P=0,S=1,N=2); output shape (batch,3) not time-series.
#              S windows sparse — most training is P/N balanced pairs.
#              h5py file opened per __getitem__ call; safe only with NUM_WORKERS=0.

from pathlib import Path
import csv
import time

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

BATCH_SIZE  = 64
MAX_EPOCHS  = 30
LR          = 1e-4
PATIENCE    = 5
LR_FACTOR   = 0.5
LR_PATIENCE = 3
NUM_WORKERS = 0
NPTS        = 6000    # samples per trace in HDF5

DEVICE = "cuda" if torch.cuda.is_available() else (
         "mps"  if torch.backends.mps.is_available() else "cpu")

print("=" * 70)
print("GPD FINE-TUNING — KOERI Kahramanmaraş dataset")
print(f"  Device: {DEVICE}  |  Batch: {BATCH_SIZE}  |  Max epochs: {MAX_EPOCHS}")
print("=" * 70)

# ── Load pretrained GPD ────────────────────────────────────────────────────────
print("\n[1] Loading pretrained GPD...")
try:
    model = sbm.GPD.from_pretrained("original")
    print("    from_pretrained('original') — OK")
except Exception as e:
    print(f"    Warning: {e}")
    model = sbm.GPD()
    print("    Fell back to default GPD initialization")

model.to(DEVICE)
WINDOWLEN  = getattr(model, "in_samples", 400)
labels_str = model.labels          # "PSN" — a string, not a list
P_IDX = labels_str.index("P")     # 0
S_IDX = labels_str.index("S")     # 1
N_IDX = labels_str.index("N")     # 2
NUM_CLS = len(labels_str)          # 3
print(f"    Parameters  : {sum(p.numel() for p in model.parameters()):,}")
print(f"    Labels      : {labels_str}  (P={P_IDX}, S={S_IDX}, N={N_IDX})")
print(f"    in_samples  : {WINDOWLEN}")

# ── Custom Dataset ─────────────────────────────────────────────────────────────
class GPDWindowDataset(Dataset):
    """
    For every P-labeled HDF5 trace generate:
      · P-window  : WINDOWLEN samples centred on p_arrival_sample  → label P
      · N-window  : WINDOWLEN samples from [0, WINDOWLEN] (pre-P)  → label N
      · S-window  : WINDOWLEN samples centred on s_arrival_sample   → label S
                    (only when S is available and non-overlapping with P)

    All windows are preloaded into numpy arrays at construction time to avoid
    repeated HDF5 file opens during training (opens are ~1ms each; 91k opens
    per epoch would add ~90s of pure I/O overhead).
    """
    def __init__(self, hdf5_path, meta_df, split, wlen,
                 p_idx, s_idx, n_idx, num_cls):
        meta = meta_df[meta_df["split"] == split].copy()
        if split != "train":
            meta = meta[meta["augmentation"] == "original"]
        meta = meta[meta["trace_p_arrival_sample"].notna()].reset_index(drop=True)

        # Build crop specs first (no I/O yet)
        specs = []   # (trace_name, start, end, label_idx)
        half  = wlen // 2

        for _, row in meta.iterrows():
            name   = row["trace_name"]
            p_samp = int(row["trace_p_arrival_sample"])
            s_samp = row["trace_s_arrival_sample"]

            # P-centred window
            p_start = p_samp - half
            p_end   = p_start + wlen
            if p_start < 0:
                p_start, p_end = 0, wlen
            elif p_end > NPTS:
                p_end, p_start = NPTS, NPTS - wlen
            if p_end <= NPTS:
                specs.append((name, p_start, p_end, p_idx))

            # Noise window (first wlen samples — well before P)
            if p_samp > wlen + 100:
                specs.append((name, 0, wlen, n_idx))

            # S-centred window (optional)
            if not pd.isna(s_samp):
                s_samp_i = int(s_samp)
                if abs(s_samp_i - p_samp) > wlen:
                    s_start = s_samp_i - half
                    s_end   = s_start + wlen
                    if s_start < 0:
                        s_start, s_end = 0, wlen
                    elif s_end > NPTS:
                        s_end, s_start = NPTS, NPTS - wlen
                    if s_end <= NPTS:
                        specs.append((name, s_start, s_end, s_idx))

        counts = {p_idx: 0, s_idx: 0, n_idx: 0}
        for _, _, _, lbl in specs:
            counts[lbl] += 1

        n_spec = len(specs)
        print(f"    [{split:5s}] {len(meta)} traces  →  {n_spec} windows "
              f"(P={counts[p_idx]}, S={counts[s_idx]}, N={counts[n_idx]})")
        print(f"      Preloading {n_spec} crops into memory ...", end="", flush=True)

        # ── Preload all crops (single HDF5 open per dataset) ──────────────────
        t_load = time.time()
        self.X = np.empty((n_spec, 3, wlen), dtype=np.float32)
        self.Y = np.zeros((n_spec, num_cls),  dtype=np.float32)

        with h5py.File(str(hdf5_path), "r") as hf:
            for k, (name, start, end, label) in enumerate(specs):
                crop = hf["data"][name][:, start:end].astype(np.float32)
                peak = float(np.abs(crop).max())
                if peak > 1e-9:
                    crop = crop / peak
                self.X[k] = crop
                self.Y[k, label] = 1.0

        elapsed = time.time() - t_load
        mem_mb  = self.X.nbytes / 1e6
        print(f" done ({elapsed:.1f}s, {mem_mb:.0f} MB)")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.Y[idx])

# ── Build datasets ─────────────────────────────────────────────────────────────
print("\n[2] Building datasets from augmented_dataset...")
meta = pd.read_csv(DS_DIR / "metadata.csv")
meta["trace_p_arrival_sample"] = pd.to_numeric(meta["trace_p_arrival_sample"], errors="coerce")
meta["trace_s_arrival_sample"] = pd.to_numeric(meta["trace_s_arrival_sample"], errors="coerce")
if "augmentation" not in meta.columns:
    meta["augmentation"] = "original"

hdf5_path = DS_DIR / "waveforms.hdf5"

train_ds = GPDWindowDataset(hdf5_path, meta, "train", WINDOWLEN,
                             P_IDX, S_IDX, N_IDX, NUM_CLS)
val_ds   = GPDWindowDataset(hdf5_path, meta, "dev",   WINDOWLEN,
                             P_IDX, S_IDX, N_IDX, NUM_CLS)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, drop_last=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS)
print(f"    Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")

# ── Optimiser + loss ───────────────────────────────────────────────────────────
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE)

def soft_cross_entropy(pred, target):
    """Cross-entropy for soft (one-hot) targets.
    pred, target: (batch, n_classes) — pred is softmax output.
    """
    return torch.mean(-torch.sum(target * torch.log(pred.clamp(min=1e-7)), dim=1))

# ── Training loop ──────────────────────────────────────────────────────────────
print("\n[3] Training...")

best_val_loss  = float("inf")
patience_count = 0
log_rows       = []
checkpoint_path = MDL_DIR / "gpd_koeri_finetuned.pt"

for epoch in range(1, MAX_EPOCHS + 1):
    t0 = time.time()

    # Train
    model.train()
    train_losses = []
    for X, y in train_loader:
        X = X.to(DEVICE)
        y = y.to(DEVICE)
        optimizer.zero_grad()
        pred = model(X)          # (B, 3) softmax probs
        loss = soft_cross_entropy(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_losses.append(loss.item())

    train_loss = float(np.mean(train_losses))

    # Validate
    model.eval()
    val_losses = []
    with torch.no_grad():
        for X, y in val_loader:
            X = X.to(DEVICE)
            y = y.to(DEVICE)
            pred = model(X)
            val_losses.append(soft_cross_entropy(pred, y).item())

    val_loss = float(np.mean(val_losses))
    elapsed  = time.time() - t0
    lr_now   = optimizer.param_groups[0]["lr"]

    print(f"  Epoch {epoch:3d}/{MAX_EPOCHS}  "
          f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
          f"lr={lr_now:.2e}  ({elapsed:.0f}s)")

    log_rows.append({
        "epoch":      epoch,
        "train_loss": round(train_loss, 6),
        "val_loss":   round(val_loss, 6),
        "lr":         lr_now,
        "elapsed_s":  round(elapsed, 1),
    })

    if val_loss < best_val_loss:
        best_val_loss  = val_loss
        patience_count = 0
        torch.save(model.state_dict(), checkpoint_path)
        print(f"    ✓ New best val_loss={best_val_loss:.4f}  → saved checkpoint")
    else:
        patience_count += 1
        print(f"    No improvement ({patience_count}/{PATIENCE})")
        if patience_count >= PATIENCE:
            print(f"  Early stopping at epoch {epoch} (patience exhausted)")
            break

    scheduler.step(val_loss)

# ── Save training log ──────────────────────────────────────────────────────────
log_path = ART / "gpd_finetune_log.csv"
with open(log_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
    writer.writeheader()
    writer.writerows(log_rows)

best_ep = min(log_rows, key=lambda r: r["val_loss"])["epoch"]
print(f"\n[4] Training log → {log_path}")
print(f"    Best val_loss: {best_val_loss:.4f}  at epoch {best_ep}")
print("\n" + "=" * 70)
print("GPD FINE-TUNING — COMPLETE")
print(f"  Best checkpoint : models/gpd_koeri_finetuned.pt")
print(f"  Training log    : artifacts/gpd_finetune_log.csv")
print(f"  → Run scripts/17_ensemble_evaluate.py next")
print("=" * 70)
