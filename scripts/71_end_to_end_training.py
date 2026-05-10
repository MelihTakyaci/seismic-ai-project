# Phase: 10
# Purpose: End-to-end deep learning training — full GPD unfreeze + EQTransformer
# Inputs: data/augmented_dataset/{waveforms.hdf5, metadata.csv} or national dataset
# Outputs: models/gpd_national.pt, models/eqt_national.pt, artifacts/national_training_log.json
# Limitations: Events exist at only 2 stations; cross-station validation used

"""
End-to-End Training — National Scale

Two training tracks run sequentially:
  Track A: GPD full unfreeze — all conv layers + fc layers, P-arrival-centered crops
  Track B: EQTransformer — detection on full 6000-sample traces

Both use:
  - ALL data (original + augmented noise from 10 stations)
  - Cross-station event split: train events from one station, test on another
  - Class-weighted loss to handle 1:38 event:noise imbalance
  - Heavy augmentation + gradient accumulation
"""

import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import seisbench.models as sbm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from seismic_engine.config import SRATE

NATIONAL_HDF5 = ROOT / "data" / "national_dataset" / "waveforms_national.hdf5"
NATIONAL_META = ROOT / "data" / "national_dataset" / "metadata_national.csv"
FALLBACK_HDF5 = ROOT / "data" / "augmented_dataset" / "waveforms.hdf5"
FALLBACK_META = ROOT / "data" / "augmented_dataset" / "metadata.csv"

MODELS_DIR = ROOT / "models"
ARTIFACTS_DIR = ROOT / "artifacts"

BATCH_SIZE = 32
ACCUMULATION_STEPS = 8
MAX_EPOCHS = 60
LR_INIT = 3e-4
LR_MIN = 1e-7
WARMUP_EPOCHS = 3
PATIENCE = 12
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0

AUG_NOISE_STD = 0.05
AUG_AMP_SCALE_RANGE = (0.5, 2.0)
AUG_CHANNEL_DROPOUT_P = 0.1


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SeismicDatasetGPD(Dataset):
    """P-arrival-centered dataset for GPD (400-sample windows)."""

    def __init__(self, hdf5_path: Path, meta: pd.DataFrame,
                 augment: bool = False, window_len: int = 400):
        self.hdf5_path = str(hdf5_path)
        self.meta = meta.reset_index(drop=True)
        self.augment = augment
        self.window_len = window_len
        self._hf = None

    def _open(self):
        if self._hf is None:
            self._hf = h5py.File(self.hdf5_path, "r")

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        self._open()
        row = self.meta.iloc[idx]
        data = self._hf["data"][row["trace_name"]][:].astype(np.float32)
        is_event = 1 if row["window_type"] == "event" else 0

        n_samples = data.shape[1]

        if is_event and pd.notna(row.get("trace_p_arrival_sample")):
            p_sample = int(row["trace_p_arrival_sample"])
            half = self.window_len // 2
            if self.augment:
                jitter = np.random.randint(-half // 2, half // 2)
                center = p_sample + jitter
            else:
                center = p_sample
            start = max(0, center - half)
            start = min(start, max(0, n_samples - self.window_len))
            end = start + self.window_len
            if end > n_samples:
                start = max(0, n_samples - self.window_len)
                end = start + self.window_len
            data = data[:, start:end]
        else:
            if n_samples > self.window_len:
                if self.augment:
                    start = np.random.randint(0, n_samples - self.window_len)
                else:
                    start = (n_samples - self.window_len) // 2
                data = data[:, start:start + self.window_len]

        if data.shape[1] < self.window_len:
            pad = np.zeros((3, self.window_len), dtype=np.float32)
            pad[:, :data.shape[1]] = data
            data = pad

        if self.augment:
            data = self._augment(data)

        peak = np.abs(data).max()
        if peak > 1e-9:
            data = data / peak

        return torch.tensor(data, dtype=torch.float32), torch.tensor(is_event, dtype=torch.long)

    def _augment(self, data: np.ndarray) -> np.ndarray:
        if np.random.random() < 0.5:
            noise = np.random.randn(*data.shape).astype(np.float32) * AUG_NOISE_STD
            data = data + noise * (np.abs(data).max() + 1e-9)
        if np.random.random() < 0.5:
            data = data * np.random.uniform(*AUG_AMP_SCALE_RANGE)
        if np.random.random() < AUG_CHANNEL_DROPOUT_P:
            data[np.random.randint(0, 3)] = 0.0
        if np.random.random() < 0.3:
            data = -data
        return data

    def close(self):
        if self._hf is not None:
            self._hf.close()
            self._hf = None


class SeismicDatasetEQT(Dataset):
    """Full-trace dataset for EQTransformer."""

    def __init__(self, hdf5_path: Path, meta: pd.DataFrame,
                 augment: bool = False, crop_len: int = 6000):
        self.hdf5_path = str(hdf5_path)
        self.meta = meta.reset_index(drop=True)
        self.augment = augment
        self.crop_len = crop_len
        self._hf = None

    def _open(self):
        if self._hf is None:
            self._hf = h5py.File(self.hdf5_path, "r")

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        self._open()
        row = self.meta.iloc[idx]
        data = self._hf["data"][row["trace_name"]][:].astype(np.float32)
        is_event = 1 if row["window_type"] == "event" else 0

        n = data.shape[1]
        if n > self.crop_len:
            if self.augment:
                start = np.random.randint(0, n - self.crop_len)
            else:
                start = (n - self.crop_len) // 2
            data = data[:, start:start + self.crop_len]
        elif n < self.crop_len:
            pad = np.zeros((3, self.crop_len), dtype=np.float32)
            pad[:, :n] = data
            data = pad

        if self.augment:
            if np.random.random() < 0.5:
                noise = np.random.randn(*data.shape).astype(np.float32) * AUG_NOISE_STD
                data = data + noise * (np.abs(data).max() + 1e-9)
            if np.random.random() < 0.5:
                data = data * np.random.uniform(0.5, 2.0)
            if np.random.random() < 0.3:
                data = -data

        peak = np.abs(data).max()
        if peak > 1e-9:
            data = data / peak

        return torch.tensor(data, dtype=torch.float32), torch.tensor(is_event, dtype=torch.long)

    def close(self):
        if self._hf is not None:
            self._hf.close()
            self._hf = None


def _compute_loss_and_preds(output, target, device):
    """Handle GPD (batch,3 softmax) and EQT (tuple of 3 curves) outputs.
    No class weighting — the WeightedRandomSampler handles imbalance."""
    if isinstance(output, tuple):
        det_curve = output[0]
        det_prob = det_curve.max(dim=1).values
        loss = F.binary_cross_entropy(det_prob, target.float())
        preds = (det_prob > 0.5).long()
    elif output.dim() == 2 and output.shape[1] == 3:
        eq_prob = output[:, 0]
        loss = F.binary_cross_entropy(eq_prob, target.float())
        preds = (eq_prob > 0.5).long()
    elif output.dim() == 2 and output.shape[1] == 2:
        loss = F.cross_entropy(output, target)
        preds = output.argmax(dim=1)
    else:
        loss = F.binary_cross_entropy(output.squeeze(), target.float())
        preds = (output.squeeze() > 0.5).long()
    return loss, preds


def train_epoch(model, loader, optimizer, device,
                accumulation_steps, grad_clip=1.0):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    optimizer.zero_grad()

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss, preds = _compute_loss_and_preds(output, target, device)
        (loss / accumulation_steps).backward()

        total_loss += loss.item() * data.size(0)
        correct += (preds == target).sum().item()
        total += data.size(0)

        if (batch_idx + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()

    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    optimizer.zero_grad()

    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    all_probs = []

    for data, target in loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss, preds = _compute_loss_and_preds(output, target, device)

        if isinstance(output, tuple):
            probs = output[0].max(dim=1).values
        elif output.dim() == 2 and output.shape[1] == 3:
            probs = output[:, 0]
        elif output.dim() == 2 and output.shape[1] == 2:
            probs = F.softmax(output, dim=1)[:, 1]
        else:
            probs = output.squeeze()

        total_loss += loss.item() * data.size(0)
        correct += (preds == target).sum().item()
        total += data.size(0)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_targets.extend(target.cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    tp = ((all_preds == 1) & (all_targets == 1)).sum()
    fn = ((all_preds == 0) & (all_targets == 1)).sum()
    fp = ((all_preds == 1) & (all_targets == 0)).sum()
    tn = ((all_preds == 0) & (all_targets == 0)).sum()

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
        "recall": float(recall),
        "precision": float(precision),
        "tnr": float(tnr),
        "f1": float(f1),
        "tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn),
    }


def make_balanced_sampler(meta: pd.DataFrame) -> WeightedRandomSampler:
    """Oversample events to balance training batches."""
    labels = (meta["window_type"] == "event").astype(int).values
    class_counts = np.bincount(labels)
    weights_per_class = 1.0 / class_counts
    sample_weights = weights_per_class[labels]
    return WeightedRandomSampler(sample_weights, num_samples=len(meta), replacement=True)


def split_data(meta: pd.DataFrame):
    """Cross-station split: train events from one station, test from another.
    Noise distributed across both splits based on station."""
    event_stations = sorted(meta[meta["window_type"] == "event"]["station"].unique())
    all_stations = sorted(meta["station"].unique())

    if len(event_stations) >= 2:
        test_event_station = event_stations[0]
        train_event_stations = set(event_stations[1:])
        noise_only_stations = set(all_stations) - set(event_stations)
        n_noise_test = max(1, len(noise_only_stations) // 3)
        test_noise_stations = set(sorted(noise_only_stations)[:n_noise_test])
        train_noise_stations = noise_only_stations - test_noise_stations

        test_stations = {test_event_station} | test_noise_stations
        train_stations = train_event_stations | train_noise_stations
    else:
        station_list = sorted(all_stations)
        n_test = max(1, len(station_list) // 5)
        test_stations = set(station_list[:n_test])
        train_stations = set(station_list[n_test:])

    train_meta = meta[meta["station"].isin(train_stations)]
    test_meta = meta[meta["station"].isin(test_stations)]

    return train_meta, test_meta, sorted(train_stations), sorted(test_stations)


def train_gpd_e2e(meta: pd.DataFrame, hdf5_path: Path, device: torch.device) -> Dict:
    """Train GPD end-to-end with P-arrival-centered crops."""
    print("\n" + "=" * 70)
    print("TRACK A: GPD END-TO-END (P-ARRIVAL-CENTERED)")
    print("=" * 70)

    train_meta, test_meta, train_stations, test_stations = split_data(meta)

    print(f"  Train stations ({len(train_stations)}): {train_stations}")
    print(f"  Test stations ({len(test_stations)}): {test_stations}")
    train_ev = (train_meta["window_type"] == "event").sum()
    train_ns = (train_meta["window_type"] == "noise").sum()
    test_ev = (test_meta["window_type"] == "event").sum()
    test_ns = (test_meta["window_type"] == "noise").sum()
    print(f"  Train: {len(train_meta)} traces ({train_ev} events, {train_ns} noise)")
    print(f"  Test: {len(test_meta)} traces ({test_ev} events, {test_ns} noise)")

    train_ds = SeismicDatasetGPD(hdf5_path, train_meta, augment=True, window_len=400)
    test_ds = SeismicDatasetGPD(hdf5_path, test_meta, augment=False, window_len=400)

    sampler = make_balanced_sampler(train_meta)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=0, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"  Balanced sampler: events oversampled ~{len(train_meta)//max(train_ev,1)}x")

    model = sbm.GPD.from_pretrained("original")
    for param in model.parameters():
        param.requires_grad = True

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  GPD unfrozen: {n_params:,} trainable parameters")
    model.to(device)

    conv_params = []
    fc_params = []
    for name, param in model.named_parameters():
        if "conv" in name or "bn" in name.lower()[:2]:
            conv_params.append(param)
        else:
            fc_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": conv_params, "lr": LR_INIT * 0.1},
        {"params": fc_params, "lr": LR_INIT},
    ], weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=LR_MIN
    )

    best_f1 = 0
    best_epoch = 0
    best_metrics = {}
    no_improve = 0
    history = []

    for epoch in range(MAX_EPOCHS):
        t0 = time.time()
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, device,
            ACCUMULATION_STEPS, GRAD_CLIP
        )
        val_metrics = evaluate(model, test_loader, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"  Epoch {epoch+1:3d}/{MAX_EPOCHS}: "
              f"loss={train_loss:.4f} acc={train_acc:.3f} | "
              f"val_R={val_metrics['recall']:.3f} val_TNR={val_metrics['tnr']:.3f} "
              f"val_F1={val_metrics['f1']:.3f} "
              f"[TP={val_metrics['tp']} FN={val_metrics['fn']} "
              f"FP={val_metrics['fp']} TN={val_metrics['tn']}] ({elapsed:.1f}s)")

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_epoch = epoch + 1
            best_metrics = val_metrics.copy()
            no_improve = 0
            torch.save(model.state_dict(), str(MODELS_DIR / "gpd_national.pt"))
            print(f"    -> New best F1={best_f1:.4f}, saved")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    train_ds.close()
    test_ds.close()

    print(f"\n  GPD Best: epoch={best_epoch}, F1={best_f1:.4f}")
    print(f"    recall={best_metrics.get('recall',0):.4f} "
          f"TNR={best_metrics.get('tnr',0):.4f} "
          f"precision={best_metrics.get('precision',0):.4f}")

    return {
        "model": "gpd_e2e",
        "best_epoch": best_epoch,
        "best_f1": best_f1,
        "best_metrics": best_metrics,
        "train_stations": train_stations,
        "test_stations": test_stations,
        "n_train": len(train_meta),
        "n_test": len(test_meta),
        "n_train_events": int(train_ev),
        "n_test_events": int(test_ev),
        "history": history,
    }


def train_eqt(meta: pd.DataFrame, hdf5_path: Path, device: torch.device) -> Dict:
    """Train EQTransformer for detection on full traces."""
    print("\n" + "=" * 70)
    print("TRACK B: EQTRANSFORMER TRAINING")
    print("=" * 70)

    train_meta, test_meta, train_stations, test_stations = split_data(meta)

    train_ev = (train_meta["window_type"] == "event").sum()
    test_ev = (test_meta["window_type"] == "event").sum()
    print(f"  Train: {len(train_meta)} ({train_ev} events), Test: {len(test_meta)} ({test_ev} events)")

    model = sbm.EQTransformer.from_pretrained("original")
    eqt_samples = getattr(model, "in_samples", 6000)
    print(f"  EQT in_samples: {eqt_samples}")

    train_ds = SeismicDatasetEQT(hdf5_path, train_meta, augment=True, crop_len=eqt_samples)
    test_ds = SeismicDatasetEQT(hdf5_path, test_meta, augment=False, crop_len=eqt_samples)

    sampler = make_balanced_sampler(train_meta)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE // 2, sampler=sampler,
                              num_workers=0, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE // 2, shuffle=False, num_workers=0)

    for param in model.parameters():
        param.requires_grad = True

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  EQTransformer unfrozen: {n_params:,} trainable parameters")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_INIT * 0.5,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=LR_MIN
    )

    best_f1 = 0
    best_epoch = 0
    best_metrics = {}
    no_improve = 0
    history = []

    for epoch in range(MAX_EPOCHS):
        t0 = time.time()
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, device,
            ACCUMULATION_STEPS, GRAD_CLIP
        )
        val_metrics = evaluate(model, test_loader, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"  Epoch {epoch+1:3d}/{MAX_EPOCHS}: "
              f"loss={train_loss:.4f} acc={train_acc:.3f} | "
              f"val_R={val_metrics['recall']:.3f} val_TNR={val_metrics['tnr']:.3f} "
              f"val_F1={val_metrics['f1']:.3f} "
              f"[TP={val_metrics['tp']} FN={val_metrics['fn']} "
              f"FP={val_metrics['fp']} TN={val_metrics['tn']}] ({elapsed:.1f}s)")

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_epoch = epoch + 1
            best_metrics = val_metrics.copy()
            no_improve = 0
            torch.save(model.state_dict(), str(MODELS_DIR / "eqt_national.pt"))
            print(f"    -> New best F1={best_f1:.4f}, saved")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    train_ds.close()
    test_ds.close()

    print(f"\n  EQT Best: epoch={best_epoch}, F1={best_f1:.4f}")
    print(f"    recall={best_metrics.get('recall',0):.4f} "
          f"TNR={best_metrics.get('tnr',0):.4f}")

    return {
        "model": "eqt_e2e",
        "best_epoch": best_epoch,
        "best_f1": best_f1,
        "best_metrics": best_metrics,
        "train_stations": train_stations,
        "test_stations": test_stations,
        "n_train": len(train_meta),
        "n_test": len(test_meta),
        "n_train_events": int(train_ev),
        "n_test_events": int(test_ev),
        "history": history,
    }


def main():
    print("=" * 70)
    print("END-TO-END DEEP LEARNING TRAINING")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)

    device = get_device()
    print(f"Device: {device}")

    if NATIONAL_HDF5.exists() and NATIONAL_META.exists():
        hdf5_path = NATIONAL_HDF5
        meta = pd.read_csv(NATIONAL_META)
        print(f"Using NATIONAL dataset: {len(meta)} traces")
    elif FALLBACK_HDF5.exists():
        hdf5_path = FALLBACK_HDF5
        meta = pd.read_csv(FALLBACK_META)
        print(f"Using FULL augmented dataset: {len(meta)} traces (10 stations)")
    else:
        print("[ERROR] No dataset found!")
        return

    n_ev = (meta["window_type"] == "event").sum()
    n_ns = (meta["window_type"] == "noise").sum()
    print(f"  Events: {n_ev}")
    print(f"  Noise: {n_ns}")
    print(f"  Stations: {sorted(meta['station'].unique())}")
    print(f"  Event:Noise ratio: 1:{n_ns/max(n_ev,1):.0f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    results = {}

    try:
        gpd_result = train_gpd_e2e(meta, hdf5_path, device)
        results["gpd"] = gpd_result
    except Exception as e:
        print(f"\n[ERROR] GPD training failed: {e}")
        import traceback
        traceback.print_exc()
        results["gpd"] = {"error": str(e)}

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    try:
        eqt_result = train_eqt(meta, hdf5_path, device)
        results["eqt"] = eqt_result
    except Exception as e:
        print(f"\n[ERROR] EQTransformer training failed: {e}")
        import traceback
        traceback.print_exc()
        results["eqt"] = {"error": str(e)}

    log_path = ARTIFACTS_DIR / "national_training_log.json"
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nTraining log saved: {log_path}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE — SUMMARY")
    print("=" * 70)
    for name, res in results.items():
        if "error" in res:
            print(f"  {name}: FAILED — {res['error']}")
        else:
            m = res.get("best_metrics", {})
            print(f"  {name}: F1={res['best_f1']:.4f} (epoch {res['best_epoch']}) "
                  f"recall={m.get('recall',0):.3f} TNR={m.get('tnr',0):.3f}")


if __name__ == "__main__":
    main()
