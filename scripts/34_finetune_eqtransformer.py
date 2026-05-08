# Phase: Quality Audit — Optimization Problem B
# Purpose: Fine-tune EQTransformer pick heads (pick_lstms, pick_attentions,
#          pick_decoders, pick_convs) on KOERI data to overcome domain mismatch.
#          Encoder (res_cnn_stack + bi_lstm_stack) frozen — only 103,684 params trained.
#          Uses peak-normalized per-channel input matching EQT's native normalization.
# Inputs:  data/augmented_dataset/waveforms.hdf5, data/augmented_dataset/metadata.csv
# Outputs: models/eqtransformer_finetuned.pt, artifacts/eqt_finetune_log.csv
# Limitations: EQT in_samples=6000 so full-window training is used (no cropping).
#              S labels are sparse (15% of windows) — S-head fine-tuning is noisy.
#              Batch size 16 to fit MPS memory.

import csv
import time
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

BATCH_SIZE  = 16
MAX_EPOCHS  = 20
LR          = 5e-5
PATIENCE    = 3
SIGMA_P     = 10     # Gaussian sigma for P label
SIGMA_S     = 10     # Gaussian sigma for S label
NPTS        = 6000
NUM_WORKERS = 0

DEVICE = "cuda" if torch.cuda.is_available() else (
         "mps"  if torch.backends.mps.is_available() else "cpu")

print("=" * 70)
print("PROBLEM B — EQTransformer Fine-tuning (pick heads only)")
print(f"  Device: {DEVICE}  |  Batch: {BATCH_SIZE}  |  LR: {LR}")
print("=" * 70)

# ── Load metadata ─────────────────────────────────────────────────────────────
meta = pd.read_csv(DS_DIR / "metadata.csv")
meta["trace_p_arrival_sample"] = pd.to_numeric(meta["trace_p_arrival_sample"], errors="coerce")
meta["trace_s_arrival_sample"] = pd.to_numeric(meta["trace_s_arrival_sample"], errors="coerce")
if "augmentation" not in meta.columns:
    meta["augmentation"] = "original"

hdf5_path = DS_DIR / "waveforms.hdf5"
n_p = meta["trace_p_arrival_sample"].notna().sum()
print(f"\n  Metadata rows: {len(meta)}  |  P-labeled: {n_p}")
print(f"  S-labeled: {meta['trace_s_arrival_sample'].notna().sum()}")

# ── Gaussian label ─────────────────────────────────────────────────────────────
def gaussian_label(length, center, sigma, dtype=np.float32):
    t = np.arange(length, dtype=np.float64)
    g = np.exp(-0.5 * ((t - center) / sigma) ** 2)
    return g.astype(dtype)

# ── Dataset (EQT-compatible normalization: per-channel peak-norm) ──────────────
class EQTDataset(Dataset):
    """
    EQTransformer expects (3, 6000) input, per-channel peak-normalized.
    Output: (3, 6000) target — channels: [Det, P, S]
    Detection label: 1 everywhere the P Gaussian is > 0.01 (broad window)
    P label: Gaussian sigma=10 at P sample
    S label: Gaussian sigma=10 at S sample (zeros if no S label)
    """
    def __init__(self, hdf5_path, meta_df, split, augmented=False):
        rows = meta_df[meta_df["split"] == split].copy()
        if not augmented or split != "train":
            rows = rows[rows["augmentation"] == "original"]
        rows = rows[rows["trace_p_arrival_sample"].notna()].reset_index(drop=True)

        is_train = (split == "train")
        print(f"    [{split:5s}] {len(rows)} P-labeled windows")
        print(f"      Preloading...", end="", flush=True)

        t0 = time.time()
        self.X = np.empty((len(rows), 3, NPTS), dtype=np.float32)
        self.Y = np.zeros((len(rows), 3, NPTS), dtype=np.float32)   # Det, P, S

        with h5py.File(str(hdf5_path), "r") as hf:
            for k, (_, row) in enumerate(rows.iterrows()):
                trace = hf["data"][row["trace_name"]][:, :NPTS].astype(np.float32)

                # Per-channel peak normalization (EQT native)
                for ch in range(3):
                    pk = float(np.abs(trace[ch]).max())
                    if pk > 1e-9:
                        trace[ch] = trace[ch] / pk
                self.X[k] = trace

                p_samp = float(row["trace_p_arrival_sample"])
                s_samp = row["trace_s_arrival_sample"]

                # P Gaussian
                p_gauss = gaussian_label(NPTS, p_samp, SIGMA_P)
                self.Y[k, 1] = p_gauss

                # Detection: broad 1 in P window ±50 samples
                det = gaussian_label(NPTS, p_samp, sigma=50)
                self.Y[k, 0] = np.clip(det * 3, 0, 1)   # broad, clipped to 1

                # S Gaussian (sparse)
                if pd.notna(s_samp):
                    self.Y[k, 2] = gaussian_label(NPTS, float(s_samp), SIGMA_S)

        elapsed = time.time() - t0
        print(f" done ({elapsed:.1f}s, {self.X.nbytes/1e6:.0f} MB)")

    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.Y[idx])

print("\n[1] Building datasets ...")
ds_train = EQTDataset(hdf5_path, meta, "train",  augmented=False)
ds_val   = EQTDataset(hdf5_path, meta, "dev",    augmented=False)

train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, drop_last=True)
val_loader   = DataLoader(ds_val,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS)
print(f"    Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")

# ── Load EQT and freeze encoder ───────────────────────────────────────────────
print("\n[2] Loading EQTransformer and freezing encoder ...")
try:
    model = sbm.EQTransformer.from_pretrained("original")
    print("    ✓ from_pretrained('original')")
except Exception as e:
    print(f"    Warning: {e}")
    model = sbm.EQTransformer()

# Freeze all except pick heads
PICK_HEAD_NAMES = {"pick_lstms", "pick_attentions", "pick_decoders", "pick_convs"}
frozen_params = 0
trained_params = 0
for name, mod in model.named_children():
    if name in PICK_HEAD_NAMES:
        for p in mod.parameters():
            p.requires_grad = True
        trained_params += sum(p.numel() for p in mod.parameters())
        print(f"    ✓ Trainable: {name}")
    else:
        for p in mod.parameters():
            p.requires_grad = False
        frozen_params += sum(p.numel() for p in mod.parameters())
        print(f"    ✗ Frozen:    {name}")

model.to(DEVICE)
print(f"\n    Trainable params: {trained_params:,}  |  Frozen: {frozen_params:,}")
print(f"    Fraction trained: {trained_params/(trained_params+frozen_params)*100:.1f}%")

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=2)

# ── Loss: BCELoss for each of the 3 output channels ───────────────────────────
def eqt_loss(pred, target):
    """
    pred:   (B, 3, T) — EQT raw output (no activation applied yet by SeisBench)
    target: (B, 3, T) — Gaussian labels for Det, P, S
    """
    # EQT output is sigmoid-activated — clamp for numerical safety
    p = pred.clamp(1e-7, 1 - 1e-7)
    bce = -(target * torch.log(p) + (1 - target) * torch.log(1 - p))
    # Weight P channel 2x vs Detection and S (P is the primary target)
    weights = torch.tensor([1.0, 2.0, 1.0], device=pred.device).view(1, 3, 1)
    return (bce * weights).mean()

# ── Training loop ─────────────────────────────────────────────────────────────
print("\n[3] Training ...")
best_val  = float("inf")
log_rows  = []
ckpt_path = MDL_DIR / "eqtransformer_finetuned.pt"
patience  = 0

for epoch in range(1, MAX_EPOCHS + 1):
    t0 = time.time()
    model.train()
    tr_losses = []
    for X, y in train_loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        pred = model(X)
        # EQT returns tuple (det, p, s) or concatenated — handle both
        if isinstance(pred, (list, tuple)):
            pred_cat = torch.stack(list(pred), dim=1)  # (B, 3, T)
        else:
            pred_cat = pred
        loss = eqt_loss(pred_cat, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0)
        optimizer.step()
        tr_losses.append(loss.item())

    model.eval()
    vl_losses = []
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = model(X)
            if isinstance(pred, (list, tuple)):
                pred_cat = torch.stack(list(pred), dim=1)
            else:
                pred_cat = pred
            vl_losses.append(eqt_loss(pred_cat, y).item())

    tr_loss  = float(np.mean(tr_losses))
    vl_loss  = float(np.mean(vl_losses))
    elapsed  = time.time() - t0
    lr_now   = optimizer.param_groups[0]["lr"]
    scheduler.step(vl_loss)

    saved = ""
    if vl_loss < best_val:
        best_val = vl_loss; patience = 0
        torch.save(model.state_dict(), ckpt_path)
        saved = " ✓ saved"
    else:
        patience += 1
        saved = f" (patience {patience}/{PATIENCE})"

    print(f"  Ep {epoch:3d}/{MAX_EPOCHS}  train={tr_loss:.4f}  val={vl_loss:.4f}  "
          f"lr={lr_now:.2e}  ({elapsed:.0f}s){saved}")
    log_rows.append({"epoch": epoch, "train_loss": round(tr_loss, 6),
                     "val_loss": round(vl_loss, 6), "lr": lr_now,
                     "elapsed_s": round(elapsed, 1)})

    if patience >= PATIENCE:
        print(f"  Early stopping at epoch {epoch}")
        break

log_path = ART / "eqt_finetune_log.csv"
with open(log_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=log_rows[0].keys())
    w.writeheader(); w.writerows(log_rows)
best_ep = min(log_rows, key=lambda r: r["val_loss"])["epoch"]
print(f"\n  Best val_loss={best_val:.4f} at epoch {best_ep}")

# ── Quick recall test ─────────────────────────────────────────────────────────
print("\n[4] Quick recall check on val set (thr=0.50) ...")
model.eval()
model.load_state_dict(torch.load(str(ckpt_path), map_location=DEVICE, weights_only=True))

ORIGIN_S    = 30.0
DETECT_TOL  = 5.0
DETECT_POST = 25.0
DETECT_LO   = int((ORIGIN_S - DETECT_TOL) * 100)
DETECT_HI   = int((ORIGIN_S + DETECT_POST) * 100)

val_meta = meta[(meta["split"] == "dev") & (meta["augmentation"] == "original") &
                (meta["trace_p_arrival_sample"].notna())].copy().reset_index(drop=True)

n_det_50 = 0
n_det_30 = 0
n_det_10 = 0
with h5py.File(str(hdf5_path), "r") as hf:
    for _, row in val_meta.iterrows():
        data = hf["data"][row["trace_name"]][:, :NPTS].astype(np.float32)
        for ch in range(3):
            pk = float(np.abs(data[ch]).max())
            if pk > 1e-9: data[ch] = data[ch] / pk
        x = torch.tensor(data).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred = model(x)
        if isinstance(pred, (list, tuple)):
            p_prob = pred[1].squeeze(0).cpu().numpy()   # P channel output
        else:
            p_prob = pred[0, 1, :].cpu().numpy()

        peak = float(p_prob[DETECT_LO:DETECT_HI+1].max())
        if peak >= 0.50: n_det_50 += 1
        if peak >= 0.30: n_det_30 += 1
        if peak >= 0.10: n_det_10 += 1

N = len(val_meta)
print(f"  Val P-labeled windows: {N}")
print(f"  thr=0.50: {n_det_50}/{N} = {n_det_50/N:.4f}")
print(f"  thr=0.30: {n_det_30}/{N} = {n_det_30/N:.4f}")
print(f"  thr=0.10: {n_det_10}/{N} = {n_det_10/N:.4f}")

best_recall = max(n_det_50, n_det_30, n_det_10) / N
target_met  = best_recall >= 0.70

print("\n" + "=" * 70)
print("PROBLEM B COMPLETE — EQTransformer Fine-tuning")
print(f"  Best val_loss: {best_val:.4f}  (epoch {best_ep})")
print(f"  Val recall: thr=0.50: {n_det_50/N:.4f}  thr=0.30: {n_det_30/N:.4f}  thr=0.10: {n_det_10/N:.4f}")
print(f"  Target (recall ≥ 0.70): {'✓ MET' if target_met else f'⚠ best={best_recall:.4f}'}")
print(f"  Checkpoint: models/eqtransformer_finetuned.pt")
print(f"  Log:        artifacts/eqt_finetune_log.csv")
print("=" * 70)
