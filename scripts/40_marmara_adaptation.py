# Phase: Quality Audit — Optimization 3
# Purpose: Domain-adaptive fine-tuning of GPD on mixed Kahramanmaras + Marmara
#          data. Marmara windows (5,838) have no P-pick labels — use GPD-FT
#          pseudo-labels (99.5% recall on Marmara). Training: 80% Kahramanmaras
#          + 20% Marmara pseudo-labeled. Evaluates recall improvement on Marmara.
# Inputs:  data/augmented_dataset/ (Kahramanmaras labeled),
#          data/windows/marmara/*.npz (Marmara, pseudo-labeled via GPD-FT),
#          models/gpd_final.pt
# Outputs: models/gpd_marmara_adapted.pt, artifacts/marmara_adaptation_results.json
# Limitations: Pseudo-labels from GPD-FT have ~0.5% error on Marmara.
#              Mixed training may slightly degrade Kahramanmaras recall.

from pathlib import Path
import json, time
import numpy as np
import pandas as pd
import torch
import h5py
from scipy.interpolate import interp1d
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import seisbench.models as sbm

ROOT     = Path(__file__).resolve().parent.parent
DS_DIR   = ROOT / "data" / "augmented_dataset"
MAR_DIR  = ROOT / "data" / "windows" / "marmara"
MDL_DIR  = ROOT / "models"
ART      = ROOT / "artifacts"

SRATE     = 100.0
N_TOTAL   = 6000
ORIGIN_S  = 30.0
DETECT_LO = int((ORIGIN_S - 5.0) * SRATE)
DETECT_HI = int((ORIGIN_S + 25.0) * SRATE)
GPD_WLEN  = 400
SIGMA_P   = 10
BATCH     = 32
MAX_EPOCHS = 15
PATIENCE  = 4

DEVICE = "cuda" if torch.cuda.is_available() else (
         "mps"  if torch.backends.mps.is_available() else "cpu")

print("=" * 70)
print("OPTIMIZATION 3 — Marmara Domain Adaptation (Mixed Training)")
print(f"  Device: {DEVICE}")
print("=" * 70)

# ── Step 1: Inventory Marmara windows ─────────────────────────────────────────
mar_files = sorted(MAR_DIR.glob("marmara_*.npz"))
print(f"\n[1] Marmara windows available: {len(mar_files)}")

# ── Generate pseudo-labels via GPD-FT ─────────────────────────────────────────
print("\n[2] Loading GPD-FT for pseudo-labeling ...")
gpd_model = sbm.GPD.from_pretrained("original")
gpd_model.load_state_dict(torch.load(str(MDL_DIR/"gpd_final.pt"), map_location=DEVICE, weights_only=True))
gpd_model.to(DEVICE).eval()
GPD_P_IDX = gpd_model.labels.index("P")
print("    GPD-FT ✓")

def gpd_sliding_prob(model, data, wlen, p_idx, stride=10):
    positions = list(range(0, N_TOTAL - wlen + 1, stride))
    crops = []
    for s in positions:
        c = data[:, s:s+wlen].astype(np.float32)
        pk = float(np.abs(c).max())
        if pk > 1e-9: c = c / pk
        crops.append(c)
    probs = []
    with torch.no_grad():
        for i in range(0, len(crops), 128):
            b = torch.tensor(np.stack(crops[i:i+128])).to(DEVICE)
            probs.extend(model(b)[:, p_idx].cpu().numpy().tolist())
    ctr = np.array([s + wlen // 2 for s in positions], dtype=float)
    pa  = np.array(probs, dtype=float)
    f   = interp1d(ctr, pa, kind="linear", bounds_error=False, fill_value=(pa[0], pa[-1]))
    return f(np.arange(N_TOTAL, dtype=float)).astype(np.float32)

# Pseudo-label a sample of Marmara windows (use 1,000 for training mix)
N_MARMARA_TRAIN = min(1000, int(len(mar_files) * 0.80))
N_MARMARA_TEST  = len(mar_files) - N_MARMARA_TRAIN
np.random.seed(42)
idx_all    = np.random.permutation(len(mar_files))
train_idx  = idx_all[:N_MARMARA_TRAIN]
test_idx   = idx_all[N_MARMARA_TRAIN:]

print(f"\n    Pseudo-labeling {N_MARMARA_TRAIN} Marmara training windows ...")
mar_X  = np.empty((N_MARMARA_TRAIN, 3, N_TOTAL), dtype=np.float32)
mar_P  = np.empty(N_MARMARA_TRAIN, dtype=np.float32)
valid  = []

for k, i in enumerate(train_idx):
    fp   = mar_files[i]
    d    = np.load(fp, allow_pickle=True)
    data = d["data"].astype(np.float32)
    prob = gpd_sliding_prob(gpd_model, data, GPD_WLEN, GPD_P_IDX)
    w    = prob[DETECT_LO:DETECT_HI+1]
    pk   = float(w.max())
    if pk < 0.50: continue      # skip low-confidence pseudo-labels
    p_samp = DETECT_LO + int(w.argmax())
    mar_X[len(valid)] = data
    mar_P[len(valid)] = p_samp
    valid.append(k)
    if (k+1) % 200 == 0:
        print(f"    {k+1}/{N_MARMARA_TRAIN}  valid={len(valid)}")

mar_X = mar_X[:len(valid)]
mar_P = mar_P[:len(valid)]
print(f"    Pseudo-labeled: {len(valid)}/{N_MARMARA_TRAIN} (high-confidence picks)")

# ── Datasets ──────────────────────────────────────────────────────────────────
class KahDataset(Dataset):
    """Kahramanmaras train windows from HDF5."""
    def __init__(self, hdf5_path, meta_df, n_max=4000):
        rows = meta_df[(meta_df["split"]=="train") &
                       (meta_df["augmentation"]=="original") &
                       (meta_df["trace_p_arrival_sample"].notna())].reset_index(drop=True)
        if len(rows) > n_max: rows = rows.sample(n_max, random_state=42).reset_index(drop=True)
        print(f"    Kahramanmaras train: {len(rows)} windows — loading...", end="", flush=True)
        t0 = time.time()
        self.X = np.empty((len(rows), 3, N_TOTAL), dtype=np.float32)
        self.P = np.empty(len(rows), dtype=np.float32)
        with h5py.File(str(hdf5_path), "r") as hf:
            for k, (_, row) in enumerate(rows.iterrows()):
                if row["trace_name"] not in hf["data"]: continue
                trace = hf["data"][row["trace_name"]][:, :N_TOTAL].astype(np.float32)
                for ch in range(3):
                    pk = float(np.abs(trace[ch]).max())
                    if pk > 1e-9: trace[ch] /= pk
                self.X[k] = trace
                self.P[k]  = float(row["trace_p_arrival_sample"])
        print(f" done ({time.time()-t0:.1f}s)")
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        t = np.arange(N_TOTAL, dtype=np.float64)
        label = np.exp(-0.5 * ((t - self.P[i]) / SIGMA_P)**2).astype(np.float32)
        return torch.tensor(self.X[i]), torch.tensor(label)

class MarDataset(Dataset):
    """Marmara pseudo-labeled windows."""
    def __init__(self, X, P):
        self.X = X; self.P = P
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        t = np.arange(N_TOTAL, dtype=np.float64)
        label = np.exp(-0.5 * ((t - self.P[i]) / SIGMA_P)**2).astype(np.float32)
        return torch.tensor(self.X[i]), torch.tensor(label)

meta = pd.read_csv(DS_DIR / "metadata.csv")
meta["trace_p_arrival_sample"] = pd.to_numeric(meta["trace_p_arrival_sample"], errors="coerce")
if "augmentation" not in meta.columns: meta["augmentation"] = "original"

print("\n[3] Building mixed training dataset ...")
kah_ds = KahDataset(DS_DIR / "waveforms.hdf5", meta)
mar_ds = MarDataset(mar_X, mar_P)
mixed_ds = ConcatDataset([kah_ds, mar_ds])
loader   = DataLoader(mixed_ds, batch_size=BATCH, shuffle=True, num_workers=0, drop_last=True)
print(f"    Mixed dataset: {len(kah_ds)} Kahramanmaras + {len(mar_ds)} Marmara = {len(mixed_ds)} total")

# ── Fine-tune GPD ─────────────────────────────────────────────────────────────
print("\n[4] Fine-tuning GPD (mixed dataset) ...")
model = sbm.GPD.from_pretrained("original")
model.load_state_dict(torch.load(str(MDL_DIR/"gpd_final.pt"), map_location=DEVICE, weights_only=True))
model.to(DEVICE)

for p in model.parameters(): p.requires_grad = True

opt = torch.optim.Adam(model.parameters(), lr=1e-5)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS, eta_min=1e-7)

def gpd_loss(pred, label):
    """GPD outputs class probabilities: [noise, P] or similar. Use BCE on P channel."""
    p_idx = GPD_P_IDX
    p_pred = pred[:, p_idx].clamp(1e-7, 1-1e-7)
    # Label: peak value in the detection window
    peak_label = label[:, DETECT_LO:DETECT_HI+1].max(dim=1).values
    return torch.nn.functional.binary_cross_entropy(p_pred, peak_label)

# Actually GPD takes 400-sample crops and outputs class probs per crop.
# For training compatibility, use the same crop-based approach as script 16.
# Re-implement with random crops around P pick.
class GPDCropDataset(Dataset):
    """Random 400-sample crop placed around P pick."""
    def __init__(self, base_ds, wlen=400, crops_per=4):
        self.base = base_ds; self.wlen = wlen; self.crops_per = crops_per
    def __len__(self): return len(self.base) * self.crops_per
    def __getitem__(self, idx):
        base_idx = idx // self.crops_per
        X, label = self.base[base_idx]
        X_np = X.numpy(); lab_np = label.numpy()
        p_samp = int(np.argmax(lab_np))
        # Random jitter ±100 around P
        jitter = np.random.randint(-100, 101)
        start  = max(0, min(N_TOTAL - self.wlen, p_samp - self.wlen//2 + jitter))
        crop   = X_np[:, start:start+self.wlen].astype(np.float32)
        pk     = float(np.abs(crop).max())
        if pk > 1e-9: crop /= pk
        # P falls in crop?
        p_in_crop = p_samp - start
        y = 1.0 if 0 <= p_in_crop < self.wlen else 0.0
        return torch.tensor(crop), torch.tensor(y, dtype=torch.float32)

crop_ds  = GPDCropDataset(mixed_ds)
crop_loader = DataLoader(crop_ds, batch_size=128, shuffle=True, num_workers=0, drop_last=True)

best_loss = float("inf"); pat = 0
for epoch in range(1, MAX_EPOCHS+1):
    t0 = time.time()
    model.train()
    losses = []
    for X_crop, y_crop in crop_loader:
        X_crop, y_crop = X_crop.to(DEVICE), y_crop.to(DEVICE)
        opt.zero_grad()
        pred = model(X_crop)          # (B, n_classes)
        p_pred = pred[:, GPD_P_IDX].clamp(1e-7, 1-1e-7)
        loss = torch.nn.functional.binary_cross_entropy(p_pred, y_crop)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    sch.step()
    tr_l = float(np.mean(losses))
    saved = ""
    if tr_l < best_loss:
        best_loss = tr_l; pat = 0
        torch.save(model.state_dict(), MDL_DIR/"gpd_marmara_adapted.pt")
        saved = " ✓"
    else:
        pat += 1; saved = f" ({pat}/{PATIENCE})"
    print(f"  Ep {epoch:2d}  loss={tr_l:.4f}  ({time.time()-t0:.0f}s){saved}")
    if pat >= PATIENCE: print("  Early stop"); break

# ── Step 3: Evaluate on Marmara test set ─────────────────────────────────────
print("\n[5] Evaluating Marmara recall: original GPD-FT vs adapted ...")

def eval_marmara_recall(model, test_files):
    model.eval()
    n_det = 0; n_tot = len(test_files)
    for fp in test_files:
        d    = np.load(fp, allow_pickle=True)
        data = d["data"].astype(np.float32)
        prob = gpd_sliding_prob(model, data, GPD_WLEN, GPD_P_IDX)
        if prob[DETECT_LO:DETECT_HI+1].max() >= 0.50:
            n_det += 1
    return n_det / n_tot if n_tot > 0 else 0.0

test_files = [mar_files[i] for i in test_idx]
print(f"    Marmara test set: {len(test_files)} windows")

# Original GPD-FT recall
gpd_model.eval()
recall_orig = eval_marmara_recall(gpd_model, test_files)
print(f"    GPD-FT (original):  {recall_orig:.4f}")

# Adapted recall
model.load_state_dict(torch.load(str(MDL_DIR/"gpd_marmara_adapted.pt"), map_location=DEVICE, weights_only=True))
recall_adapt = eval_marmara_recall(model, test_files)
print(f"    GPD (Marmara-adapted): {recall_adapt:.4f}  (Δ={recall_adapt-recall_orig:+.4f})")

# ── Save JSON ─────────────────────────────────────────────────────────────────
out = {
    "phase": "marmara_domain_adaptation",
    "n_marmara_train_pseudo": len(valid),
    "n_marmara_test": len(test_files),
    "n_kahramanmaras_train": len(kah_ds),
    "pseudo_label_confidence_threshold": 0.50,
    "recall": {
        "gpd_ft_original":  round(recall_orig,  4),
        "gpd_ft_adapted":   round(recall_adapt, 4),
        "delta":            round(recall_adapt - recall_orig, 4),
    },
    "note": (
        "Pseudo-labels generated by GPD-FT (99.5% recall on Marmara). "
        "Mixed training: 80% Kahramanmaras + 20% Marmara. "
        "Model: GPD initialized from gpd_final.pt, fine-tuned at lr=1e-5."
    ),
}
with open(ART / "marmara_adaptation_results.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"\n    Saved → {ART}/marmara_adaptation_results.json")

print("\n" + "=" * 70)
print("OPTIMIZATION 3 COMPLETE — Marmara Domain Adaptation")
print(f"  GPD-FT original recall:  {recall_orig:.4f}")
print(f"  GPD adapted recall:      {recall_adapt:.4f}  (Δ={recall_adapt-recall_orig:+.4f})")
print(f"  Model: models/gpd_marmara_adapted.pt")
print("=" * 70)
