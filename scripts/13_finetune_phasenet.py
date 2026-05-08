# Phase: Fine-tuning
# Purpose: Fine-tune pretrained PhaseNet on the KOERI Kahramanmaraş dataset using
#          SeisBench generators. Applies early stopping on validation loss with
#          patience=5. Saves best checkpoint and per-epoch training log.
# Inputs:  data/finetune_dataset/waveforms.hdf5, data/finetune_dataset/metadata.csv
# Outputs: models/phasenet_koeri_finetuned.pt, artifacts/finetune_log.csv
# Limitations: Fine-tuning on ~4-5k windows is a small sample by deep learning standards;
#              overfitting is possible. Early stopping and lr scheduling mitigate this.
#              Training is CPU-bound (~30-90 min on Apple Silicon, faster on CUDA GPU).
#              PhaseNet input is 3001 samples; windows are randomly cropped from 6000-sample
#              NPZ files around the TauPy P label during training.

from pathlib import Path
import csv
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import seisbench.models as sbm
import seisbench.data as sbd
import seisbench.generate as sbg

ROOT    = Path(__file__).resolve().parent.parent
DS_DIR  = ROOT / "data" / "augmented_dataset"
MDL_DIR = ROOT / "models"
ART     = ROOT / "artifacts"
MDL_DIR.mkdir(exist_ok=True)

# ── Hyperparameters ────────────────────────────────────────────────────────────
BATCH_SIZE   = 32
MAX_EPOCHS   = 30
LR           = 1e-4
PATIENCE     = 5          # early stopping patience (epochs without val improvement)
LR_FACTOR    = 0.5        # ReduceLROnPlateau factor
LR_PATIENCE  = 3          # epochs before LR reduction
SIGMA        = 20         # Gaussian label width in samples (= 0.2 s at 100 Hz)
WINDOWLEN    = 3001       # PhaseNet input length
NPTS         = 6000       # full window length in source HDF5
SAMPLES_BEFORE = 500      # samples before P pick in validation windows
NUM_WORKERS  = 0          # DataLoader workers (0 = main thread; safe on all OS)

DEVICE = "cuda" if torch.cuda.is_available() else (
         "mps"  if torch.backends.mps.is_available() else "cpu")

print("=" * 70)
print("PHASE FINE-TUNING — PhaseNet on KOERI Kahramanmaraş data")
print(f"  Device: {DEVICE}  |  Batch: {BATCH_SIZE}  |  Max epochs: {MAX_EPOCHS}")
print(f"  LR: {LR}  |  Early stopping patience: {PATIENCE}")
print("=" * 70)

# ── Step 1: Load dataset — filter to windows with valid P labels only ──────────
print("\n[1] Loading SeisBench WaveformDataset (P-labeled windows only)...")
dataset = sbd.WaveformDataset(DS_DIR, sampling_rate=100)

# Keep only rows where trace_p_arrival_sample is not NaN.
# Windows without a TauPy P label are treated as pure noise by
# ProbabilisticLabeller, corrupting the model with mislabeled event windows.
_meta  = dataset.metadata
_valid = _meta["trace_p_arrival_sample"].notna()
print(f"    Total windows: {len(_meta)}  |  With P label: {int(_valid.sum())}  "
      f"|  Dropped (NaN P): {int((~_valid).sum())}")
dataset.filter(_valid, inplace=True)

train_data = dataset.train()
val_data   = dataset.dev()
print(f"    Train: {len(train_data)}  |  Val: {len(val_data)}")

# ── Step 2: Load pretrained PhaseNet ──────────────────────────────────────────
print("\n[2] Loading pretrained PhaseNet (STEAD weights)...")
model = sbm.PhaseNet.from_pretrained("original")
model.to(DEVICE)
print(f"    Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"    Output labels: {model.labels}")   # "NPS"

# ── Step 3: Build seisbench generators ────────────────────────────────────────
print("\n[3] Building seisbench generators with augmentations...")

# Shared augmentation list (applied to every sample in order)
def make_augmentations(training: bool):
    augs = []

    if training:
        # Training: centre the 6000-sample extraction on P (samples_before=NPTS//2=3000).
        # P sits at position 3000 in the 6000-sample window.  RandomWindow then crops
        # to 3001 samples with a random start in [0, 2999], placing P at position
        # 3000-start ∈ [1, 3000].  This forces the model to learn waveform features
        # rather than memorising that P is always at sample 500 (positional overfitting).
        augs.append(sbg.WindowAroundSample(
            ["trace_p_arrival_sample", "trace_s_arrival_sample"],
            samples_before=NPTS // 2,   # 3000 — P centred in 6000-sample window
            selection="random",
            windowlen=NPTS,
            strategy="pad",
        ))
        augs.append(sbg.RandomWindow(
            windowlen=WINDOWLEN,        # 3001 — random crop; P at diverse positions
            strategy="pad",
        ))
    else:
        # Validation: P at fixed sample SAMPLES_BEFORE=500 for consistent labelling.
        augs.append(sbg.WindowAroundSample(
            ["trace_p_arrival_sample", "trace_s_arrival_sample"],
            samples_before=SAMPLES_BEFORE,   # 500
            selection="first",
            windowlen=WINDOWLEN,             # 3001 — exact model input
            strategy="pad",
        ))

    # Normalisation: remove linear trend + peak-amplitude normalise per trace
    augs.append(sbg.Normalize(
        detrend_axis=-1,
        amp_norm_axis=-1,
        amp_norm_type="peak",
    ))

    augs.append(sbg.ChangeDtype(np.float32))

    # Create Gaussian P/S labels at the pick positions; Noise = 1 - P - S
    # SeisBench 0.7 identifies the noise channel by lowercase "n"; PhaseNet
    # output order is [Noise=0, P=1, S=2], so model_labels must be "nPS".
    augs.append(sbg.ProbabilisticLabeller(
        label_columns={
            "trace_p_arrival_sample": "P",
            "trace_s_arrival_sample": "S",
        },
        sigma=SIGMA,
        model_labels="nPS",   # "n"=Noise ch0, "P"=P ch1, "S"=S ch2
        dim=0,
    ))

    return augs

train_gen = sbg.GenericGenerator(train_data)
train_gen.add_augmentations(make_augmentations(training=True))

val_gen = sbg.GenericGenerator(val_data)
val_gen.add_augmentations(make_augmentations(training=False))

train_loader = DataLoader(train_gen, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=NUM_WORKERS,
                          drop_last=True)
val_loader   = DataLoader(val_gen,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=NUM_WORKERS)

print(f"    Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")

# ── Step 4: Optimiser and schedulers ──────────────────────────────────────────
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE
)

def soft_cross_entropy(pred, target):
    """Cross-entropy loss for soft (probabilistic) targets.
    pred, target: (batch, channels, time) — pred is softmax output in [0,1].
    """
    return torch.mean(-torch.sum(target * torch.log(pred.clamp(min=1e-7)), dim=1))

# ── Step 5: Training loop ──────────────────────────────────────────────────────
print("\n[4] Training...")

best_val_loss  = float("inf")
patience_count = 0
log_rows       = []

checkpoint_path = MDL_DIR / "phasenet_koeri_finetuned.pt"

for epoch in range(1, MAX_EPOCHS + 1):
    t0 = time.time()

    # ── Train ──
    model.train()
    train_losses = []
    for batch in train_loader:
        # batch["X"] = waveform tensor (batch, channels, time) — tensor, not list
        # batch["y"] = label tensor   (batch, 3, time)
        X = batch["X"].float().to(DEVICE)     # (B, 3, 3001)
        y = batch["y"].float().to(DEVICE)     # (B, 3, 3001)

        optimizer.zero_grad()
        pred = model(X)                  # (B, 3, 3001) — softmax probabilities
        loss = soft_cross_entropy(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_losses.append(loss.item())

    train_loss = float(np.mean(train_losses))

    # ── Validate ──
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in val_loader:
            X = batch["X"].float().to(DEVICE)
            y = batch["y"].float().to(DEVICE)
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

    # Save best checkpoint
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_count = 0
        torch.save(model.state_dict(), checkpoint_path)
        print(f"    ✓ New best val_loss={best_val_loss:.4f}  → saved checkpoint")
    else:
        patience_count += 1
        print(f"    No improvement ({patience_count}/{PATIENCE})")
        if patience_count >= PATIENCE:
            print(f"  Early stopping at epoch {epoch} (patience={PATIENCE} exhausted)")
            break

    scheduler.step(val_loss)

# ── Step 6: Save training log ─────────────────────────────────────────────────
log_path = ART / "finetune_log.csv"
with open(log_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
    writer.writeheader()
    writer.writerows(log_rows)

print(f"\n[5] Saved training log → {log_path}")
print(f"    Best val_loss: {best_val_loss:.4f}  at epoch "
      f"{min(log_rows, key=lambda r: r['val_loss'])['epoch']}")

print("\n" + "=" * 70)
print("FINE-TUNING — COMPLETE")
print(f"  Best checkpoint : models/phasenet_koeri_finetuned.pt")
print(f"  Training log    : artifacts/finetune_log.csv")
print(f"  → Run scripts/14_evaluate_finetuned.py next")
print("=" * 70)
