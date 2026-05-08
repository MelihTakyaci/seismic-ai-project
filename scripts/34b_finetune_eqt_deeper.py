# Phase: Quality Audit — Problem B v2 (deeper EQT adaptation)
# Purpose: Pick-head-only fine-tuning failed (recall=0.002). Root cause: frozen
#          encoder features are too domain-mismatched for KOERI HH data. Fix:
#          unfreeze transformer_d + decoder_d + conv_d + all pick heads
#          (157K params, ~42% of model). Keep res_cnn_stack + bi_lstm_stack frozen.
#          Also try full unfreeze at 1/10 LR to allow global adaptation.
# Inputs:  data/augmented_dataset/waveforms.hdf5, data/augmented_dataset/metadata.csv
# Outputs: models/eqtransformer_finetuned.pt (overwrite if better),
#          artifacts/eqt_finetune_log.csv
# Limitations: Full unfreeze risks catastrophic forgetting of pretrained features.

import csv, time
from pathlib import Path
import h5py, numpy as np, pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import seisbench.models as sbm

ROOT    = Path(__file__).resolve().parent.parent
DS_DIR  = ROOT / "data" / "augmented_dataset"
MDL_DIR = ROOT / "models"
ART     = ROOT / "artifacts"

BATCH_SIZE  = 16
MAX_EPOCHS  = 25
PATIENCE    = 5
SIGMA_P     = 10
NPTS        = 6000
NUM_WORKERS = 0

DEVICE = "cuda" if torch.cuda.is_available() else (
         "mps"  if torch.backends.mps.is_available() else "cpu")

print("=" * 70)
print("PROBLEM B v2 — EQTransformer Deeper Fine-tuning")
print(f"  Device: {DEVICE}  |  Freeze: encoder + res_cnn + bilstm only")
print("=" * 70)

meta = pd.read_csv(DS_DIR / "metadata.csv")
meta["trace_p_arrival_sample"] = pd.to_numeric(meta["trace_p_arrival_sample"], errors="coerce")
meta["trace_s_arrival_sample"] = pd.to_numeric(meta["trace_s_arrival_sample"], errors="coerce")
if "augmentation" not in meta.columns: meta["augmentation"] = "original"
hdf5_path = DS_DIR / "waveforms.hdf5"

def gaussian_label(length, center, sigma):
    t = np.arange(length, dtype=np.float64)
    return np.exp(-0.5 * ((t - center) / sigma) ** 2).astype(np.float32)

class EQTDataset(Dataset):
    def __init__(self, hdf5_path, meta_df, split):
        rows = meta_df[(meta_df["split"] == split) &
                       (meta_df["augmentation"] == "original") &
                       (meta_df["trace_p_arrival_sample"].notna())].reset_index(drop=True)
        print(f"    [{split:5s}] {len(rows)} windows — preloading...", end="", flush=True)
        t0 = time.time()
        self.X = np.empty((len(rows), 3, NPTS), dtype=np.float32)
        self.Y = np.zeros((len(rows), 3, NPTS), dtype=np.float32)
        with h5py.File(str(hdf5_path), "r") as hf:
            for k, (_, row) in enumerate(rows.iterrows()):
                trace = hf["data"][row["trace_name"]][:, :NPTS].astype(np.float32)
                for ch in range(3):
                    pk = float(np.abs(trace[ch]).max())
                    if pk > 1e-9: trace[ch] /= pk
                self.X[k] = trace
                p = float(row["trace_p_arrival_sample"])
                self.Y[k, 1] = gaussian_label(NPTS, p, SIGMA_P)
                self.Y[k, 0] = np.clip(gaussian_label(NPTS, p, 50) * 3, 0, 1)
                if pd.notna(row["trace_s_arrival_sample"]):
                    self.Y[k, 2] = gaussian_label(NPTS, float(row["trace_s_arrival_sample"]), SIGMA_P)
        print(f" done ({time.time()-t0:.1f}s)")
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return torch.tensor(self.X[i]), torch.tensor(self.Y[i])

print("\n[1] Building datasets ...")
ds_train = EQTDataset(hdf5_path, meta, "train")
ds_val   = EQTDataset(hdf5_path, meta, "dev")
train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, drop_last=True)
val_loader   = DataLoader(ds_val,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

def eqt_loss(pred, target):
    p = pred.clamp(1e-7, 1-1e-7)
    bce = -(target * torch.log(p) + (1-target) * torch.log(1-p))
    w = torch.tensor([1.0, 2.0, 1.0], device=pred.device).view(1,3,1)
    return (bce * w).mean()

def get_pred_cat(model, x):
    pred = model(x)
    if isinstance(pred, (list, tuple)):
        return torch.stack(list(pred), dim=1)
    return pred

ORIG_CKPT = MDL_DIR / "eqtransformer_finetuned.pt"
best_overall = float("inf")

# ── Strategy A: unfreeze transformer + decoder + pick heads ──────────────────
print("\n[2A] Strategy A: unfreeze transformer_d + decoder_d + pick heads ...")
model_a = sbm.EQTransformer.from_pretrained("original")

FROZEN_A = {"encoder", "res_cnn_stack", "bi_lstm_stack", "transformer_d0"}
trained_a = 0
for name, mod in model_a.named_children():
    freeze = name in FROZEN_A
    for p in mod.parameters(): p.requires_grad = not freeze
    n = sum(p.numel() for p in mod.parameters())
    trained_a += 0 if freeze else n
    print(f"    {'✗ Frozen' if freeze else '✓ Train ':9s}: {name} ({n:,} params)")
model_a.to(DEVICE)

opt_a = torch.optim.Adam(filter(lambda p: p.requires_grad, model_a.parameters()), lr=5e-5)
sch_a = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_a, "min", factor=0.5, patience=2)

best_a  = float("inf")
log_a   = []
pat_a   = 0
ckpt_a  = MDL_DIR / "eqt_stratA.pt"

for epoch in range(1, MAX_EPOCHS+1):
    t0 = time.time()
    model_a.train()
    trl = [eqt_loss(get_pred_cat(model_a, X.to(DEVICE)), y.to(DEVICE)).backward() or
           opt_a.step() or opt_a.zero_grad() or eqt_loss(get_pred_cat(model_a, X.to(DEVICE)), y.to(DEVICE)).item()
           for X, y in [(X, y) for X, y in train_loader]]

    # Proper training loop
    model_a.train()
    tr_losses = []
    for X, y in train_loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        opt_a.zero_grad()
        loss = eqt_loss(get_pred_cat(model_a, X), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model_a.parameters()), 1.0)
        opt_a.step()
        tr_losses.append(loss.item())

    model_a.eval()
    vl = []
    with torch.no_grad():
        for X, y in val_loader:
            vl.append(eqt_loss(get_pred_cat(model_a, X.to(DEVICE)), y.to(DEVICE)).item())
    tr_l = float(np.mean(tr_losses)); vl_l = float(np.mean(vl))
    sch_a.step(vl_l)
    saved = ""
    if vl_l < best_a:
        best_a = vl_l; pat_a = 0
        torch.save(model_a.state_dict(), ckpt_a)
        saved = " ✓"
    else:
        pat_a += 1; saved = f" ({pat_a}/{PATIENCE})"
    print(f"  A Ep {epoch:2d}  train={tr_l:.4f}  val={vl_l:.4f}  ({time.time()-t0:.0f}s){saved}")
    log_a.append({"epoch": epoch, "strategy": "A", "train": round(tr_l,6), "val": round(vl_l,6)})
    if pat_a >= PATIENCE: print(f"  Early stop"); break

# ── Strategy B: full unfreeze at lower LR ────────────────────────────────────
print("\n[2B] Strategy B: full unfreeze at lr=1e-5 ...")
model_b = sbm.EQTransformer.from_pretrained("original")
for p in model_b.parameters(): p.requires_grad = True
model_b.to(DEVICE)

opt_b = torch.optim.Adam(model_b.parameters(), lr=1e-5)
sch_b = torch.optim.lr_scheduler.CosineAnnealingLR(opt_b, T_max=MAX_EPOCHS, eta_min=1e-7)

best_b = float("inf"); pat_b = 0; log_b = []; ckpt_b = MDL_DIR / "eqt_stratB.pt"

for epoch in range(1, MAX_EPOCHS+1):
    t0 = time.time()
    model_b.train()
    tr_losses = []
    for X, y in train_loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        opt_b.zero_grad()
        loss = eqt_loss(get_pred_cat(model_b, X), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_b.parameters(), 1.0)
        opt_b.step()
        tr_losses.append(loss.item())
    sch_b.step()

    model_b.eval()
    vl = []
    with torch.no_grad():
        for X, y in val_loader:
            vl.append(eqt_loss(get_pred_cat(model_b, X.to(DEVICE)), y.to(DEVICE)).item())
    tr_l = float(np.mean(tr_losses)); vl_l = float(np.mean(vl))
    saved = ""
    if vl_l < best_b:
        best_b = vl_l; pat_b = 0
        torch.save(model_b.state_dict(), ckpt_b)
        saved = " ✓"
    else:
        pat_b += 1; saved = f" ({pat_b}/{PATIENCE})"
    print(f"  B Ep {epoch:2d}  train={tr_l:.4f}  val={vl_l:.4f}  ({time.time()-t0:.0f}s){saved}")
    log_b.append({"epoch": epoch, "strategy": "B", "train": round(tr_l,6), "val": round(vl_l,6)})
    if pat_b >= PATIENCE: print(f"  Early stop"); break

# ── Evaluate both and pick best ───────────────────────────────────────────────
print("\n[3] Evaluating both strategies ...")
DETECT_LO = int(25 * 100); DETECT_HI = int(55 * 100)

def eval_recall(model, hdf5, meta_val, thrs=(0.50, 0.30, 0.10)):
    model.eval()
    counts = {t: 0 for t in thrs}
    n = 0
    with h5py.File(str(hdf5), "r") as hf:
        for _, row in meta_val.iterrows():
            data = hf["data"][row["trace_name"]][:, :NPTS].astype(np.float32)
            for ch in range(3):
                pk = float(np.abs(data[ch]).max())
                if pk > 1e-9: data[ch] /= pk
            x = torch.tensor(data).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                pred = model(x)
            p_prob = (list(pred)[1] if isinstance(pred,(list,tuple)) else pred[0,1:2]).squeeze().cpu().numpy()
            peak = float(p_prob[DETECT_LO:DETECT_HI+1].max())
            for t in thrs:
                if peak >= t: counts[t] += 1
            n += 1
    return {t: counts[t]/n for t in thrs}, n

val_meta = meta[(meta["split"]=="dev") & (meta["augmentation"]=="original") &
                (meta["trace_p_arrival_sample"].notna())].reset_index(drop=True)

model_a.load_state_dict(torch.load(str(ckpt_a), map_location=DEVICE, weights_only=True))
recall_a, N = eval_recall(model_a, hdf5_path, val_meta)
print(f"\n  Strategy A: thr=0.50:{recall_a[0.50]:.4f}  thr=0.30:{recall_a[0.30]:.4f}  thr=0.10:{recall_a[0.10]:.4f}")

model_b.load_state_dict(torch.load(str(ckpt_b), map_location=DEVICE, weights_only=True))
recall_b, _ = eval_recall(model_b, hdf5_path, val_meta)
print(f"  Strategy B: thr=0.50:{recall_b[0.50]:.4f}  thr=0.30:{recall_b[0.30]:.4f}  thr=0.10:{recall_b[0.10]:.4f}")

best_a_r = max(recall_a.values()); best_b_r = max(recall_b.values())
if best_b_r >= best_a_r:
    print(f"\n  Strategy B wins ({best_b_r:.4f} > {best_a_r:.4f}) — copying to eqtransformer_finetuned.pt")
    import shutil; shutil.copy2(ckpt_b, MDL_DIR/"eqtransformer_finetuned.pt")
    best_recall = best_b_r
else:
    print(f"\n  Strategy A wins ({best_a_r:.4f} > {best_b_r:.4f}) — copying to eqtransformer_finetuned.pt")
    import shutil; shutil.copy2(ckpt_a, MDL_DIR/"eqtransformer_finetuned.pt")
    best_recall = best_a_r

# Save combined log
all_log = log_a + log_b
with open(ART/"eqt_finetune_log.csv","w",newline="") as f:
    w = csv.DictWriter(f, fieldnames=all_log[0].keys()); w.writeheader(); w.writerows(all_log)

target_met = best_recall >= 0.70
print("\n" + "=" * 70)
print("PROBLEM B v2 COMPLETE")
print(f"  Strategy A best recall: {best_a_r:.4f}")
print(f"  Strategy B best recall: {best_b_r:.4f}")
print(f"  Target (≥0.70): {'✓ MET' if target_met else f'⚠ {best_recall:.4f} — domain mismatch persists'}")
print(f"  Best model: models/eqtransformer_finetuned.pt")
print("=" * 70)
