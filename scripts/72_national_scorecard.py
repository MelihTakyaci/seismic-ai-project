# Phase: 10
# Purpose: Strict generalization evaluation on UNSEEN stations
# Inputs: models/gpd_national.pt, models/eqt_national.pt, data/augmented_dataset/
# Outputs: artifacts/national_scorecard.json
# Limitations: Requires trained national models from scripts/71

"""
National Scorecard — Cross-Station Generalization Proof

Evaluates trained models STRICTLY on the station they never saw events from
during training. Training used KO.KOZT events → test on KO.KMRS events.
Noise from holdout stations (KO.ANTB, KO.DARE, KO.KRBG) also included.

Metrics:
  - Overall recall, precision, TNR, F1
  - Per-station recall and TNR
  - Per-magnitude-band recall
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np
import pandas as pd
import torch
import seisbench.models as sbm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

NATIONAL_HDF5 = ROOT / "data" / "national_dataset" / "waveforms_national.hdf5"
NATIONAL_META = ROOT / "data" / "national_dataset" / "metadata_national.csv"
FALLBACK_HDF5 = ROOT / "data" / "augmented_dataset" / "waveforms.hdf5"
FALLBACK_META = ROOT / "data" / "augmented_dataset" / "metadata.csv"
MODELS_DIR = ROOT / "models"


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_test_split(meta: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the same split as training: test station = first event station."""
    event_stations = sorted(meta[meta["window_type"] == "event"]["station"].unique())
    all_stations = sorted(meta["station"].unique())

    if len(event_stations) >= 2:
        test_event_station = event_stations[0]
        noise_only_stations = set(all_stations) - set(event_stations)
        n_noise_test = max(1, len(noise_only_stations) // 3)
        test_noise_stations = set(sorted(noise_only_stations)[:n_noise_test])
        test_stations = {test_event_station} | test_noise_stations
    else:
        station_list = sorted(all_stations)
        n_test = max(1, len(station_list) // 5)
        test_stations = set(station_list[:n_test])

    return meta[meta["station"].isin(test_stations)]


def evaluate_gpd(meta: pd.DataFrame, hdf5_path: Path, device: torch.device) -> Dict:
    """Evaluate GPD with P-arrival-centered crops (matching training)."""
    model_path = MODELS_DIR / "gpd_national.pt"
    if not model_path.exists():
        return {"error": "gpd_national.pt not found"}

    model = sbm.GPD.from_pretrained("original")
    model.load_state_dict(torch.load(str(model_path), map_location=device, weights_only=True))
    model.to(device).eval()
    window_len = 400

    hf = h5py.File(str(hdf5_path), "r")
    results = []

    for idx, row in meta.iterrows():
        data = hf["data"][row["trace_name"]][:].astype(np.float32)
        is_event = row["window_type"] == "event"
        n_samples = data.shape[1]

        if is_event and pd.notna(row.get("trace_p_arrival_sample")):
            p_sample = int(row["trace_p_arrival_sample"])
            half = window_len // 2
            start = max(0, p_sample - half)
            start = min(start, max(0, n_samples - window_len))
            crop = data[:, start:start + window_len]
        else:
            if n_samples > window_len:
                start = (n_samples - window_len) // 2
                crop = data[:, start:start + window_len]
            else:
                crop = np.zeros((3, window_len), dtype=np.float32)
                crop[:, :n_samples] = data

        if crop.shape[1] < window_len:
            pad = np.zeros((3, window_len), dtype=np.float32)
            pad[:, :crop.shape[1]] = crop
            crop = pad

        peak = np.abs(crop).max()
        if peak > 1e-9:
            crop = crop / peak

        x = torch.tensor(crop[np.newaxis], dtype=torch.float32).to(device)
        with torch.no_grad():
            out = model(x)

        eq_prob = float(out[0, 0].cpu().numpy())
        pred_event = eq_prob > 0.5

        results.append({
            "trace_name": row["trace_name"],
            "station": row["station"],
            "is_event_gt": is_event,
            "is_event_pred": pred_event,
            "eq_prob": eq_prob,
            "magnitude": row.get("source_magnitude"),
        })

    hf.close()
    return _compute_scorecard(results, "gpd_national")


def evaluate_eqt(meta: pd.DataFrame, hdf5_path: Path, device: torch.device) -> Dict:
    """Evaluate EQTransformer on full traces."""
    model_path = MODELS_DIR / "eqt_national.pt"
    if not model_path.exists():
        return {"error": "eqt_national.pt not found"}

    model = sbm.EQTransformer.from_pretrained("original")
    model.load_state_dict(torch.load(str(model_path), map_location=device, weights_only=True))
    model.to(device).eval()
    crop_len = getattr(model, "in_samples", 6000)

    hf = h5py.File(str(hdf5_path), "r")
    results = []

    for idx, row in meta.iterrows():
        data = hf["data"][row["trace_name"]][:].astype(np.float32)
        is_event = row["window_type"] == "event"

        n = data.shape[1]
        if n > crop_len:
            start = (n - crop_len) // 2
            crop = data[:, start:start + crop_len]
        else:
            crop = np.zeros((3, crop_len), dtype=np.float32)
            crop[:, :n] = data

        peak = np.abs(crop).max()
        if peak > 1e-9:
            crop = crop / peak

        x = torch.tensor(crop[np.newaxis], dtype=torch.float32).to(device)
        with torch.no_grad():
            out = model(x)

        if isinstance(out, tuple):
            det_curve = out[0][0].cpu().numpy()
            det_prob = float(det_curve.max())
        elif isinstance(out, torch.Tensor):
            probs = out[0].cpu().numpy()
            det_prob = float(probs[0].max()) if probs.ndim > 1 else float(probs[0])
        else:
            det_prob = 0.5

        pred_event = det_prob > 0.5

        results.append({
            "trace_name": row["trace_name"],
            "station": row["station"],
            "is_event_gt": is_event,
            "is_event_pred": pred_event,
            "eq_prob": det_prob,
            "magnitude": row.get("source_magnitude"),
        })

    hf.close()
    return _compute_scorecard(results, "eqt_national")


def _compute_scorecard(results: List[Dict], model_name: str) -> Dict:
    """Compute comprehensive scorecard from prediction results."""
    df = pd.DataFrame(results)

    events = df[df["is_event_gt"]]
    noise = df[~df["is_event_gt"]]

    tp = events["is_event_pred"].sum()
    fn = (~events["is_event_pred"]).sum()
    fp = noise["is_event_pred"].sum()
    tn = (~noise["is_event_pred"]).sum()

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    scorecard = {
        "model": model_name,
        "overall": {
            "n_events": int(len(events)),
            "n_noise": int(len(noise)),
            "TP": int(tp), "FN": int(fn), "FP": int(fp), "TN": int(tn),
            "recall": round(float(recall), 4),
            "precision": round(float(precision), 4),
            "tnr": round(float(tnr), 4),
            "f1": round(float(f1), 4),
        },
    }

    station_metrics = {}
    for station in sorted(df["station"].unique()):
        sdf = df[df["station"] == station]
        s_ev = sdf[sdf["is_event_gt"]]
        s_ns = sdf[~sdf["is_event_gt"]]
        entry = {"n_events": int(len(s_ev)), "n_noise": int(len(s_ns))}
        if len(s_ev) > 0:
            entry["recall"] = round(float(s_ev["is_event_pred"].sum() / len(s_ev)), 4)
        if len(s_ns) > 0:
            entry["tnr"] = round(float((~s_ns["is_event_pred"]).sum() / len(s_ns)), 4)
        station_metrics[station] = entry
    scorecard["per_station"] = station_metrics

    mag_metrics = {}
    for lo, hi, label in [(0.5, 1.5, "M0.5-1.5"), (1.5, 2.0, "M1.5-2"),
                          (2.0, 3.0, "M2-3"), (3.0, 4.0, "M3-4"), (4.0, 10.0, "M4+")]:
        band = events[events["magnitude"].notna() &
                      (events["magnitude"] >= lo) & (events["magnitude"] < hi)]
        if len(band) > 0:
            band_recall = band["is_event_pred"].sum() / len(band)
            mag_metrics[label] = {
                "n": int(len(band)),
                "detected": int(band["is_event_pred"].sum()),
                "recall": round(float(band_recall), 4),
            }
    scorecard["per_magnitude"] = mag_metrics

    return scorecard


def main():
    print("=" * 70)
    print("NATIONAL GENERALIZATION SCORECARD")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)

    device = get_device()
    print(f"Device: {device}")

    if NATIONAL_HDF5.exists():
        hdf5_path = NATIONAL_HDF5
        meta = pd.read_csv(NATIONAL_META)
    else:
        hdf5_path = FALLBACK_HDF5
        meta = pd.read_csv(FALLBACK_META)

    holdout_meta = get_test_split(meta)

    print(f"\nEvaluation set (UNSEEN during training): {len(holdout_meta)} traces")
    print(f"  Events: {(holdout_meta['window_type']=='event').sum()}")
    print(f"  Noise: {(holdout_meta['window_type']=='noise').sum()}")
    print(f"  Stations: {sorted(holdout_meta['station'].unique())}")

    all_results = {}

    print("\n--- GPD National (P-arrival centered) ---")
    gpd_scores = evaluate_gpd(holdout_meta, hdf5_path, device)
    all_results["gpd_national"] = gpd_scores
    if "error" not in gpd_scores:
        o = gpd_scores["overall"]
        print(f"  Recall={o['recall']:.4f}, Precision={o['precision']:.4f}, "
              f"TNR={o['tnr']:.4f}, F1={o['f1']:.4f}")
    else:
        print(f"  {gpd_scores['error']}")

    print("\n--- EQTransformer National ---")
    eqt_scores = evaluate_eqt(holdout_meta, hdf5_path, device)
    all_results["eqt_national"] = eqt_scores
    if "error" not in eqt_scores:
        o = eqt_scores["overall"]
        print(f"  Recall={o['recall']:.4f}, Precision={o['precision']:.4f}, "
              f"TNR={o['tnr']:.4f}, F1={o['f1']:.4f}")
    else:
        print(f"  {eqt_scores['error']}")

    out_path = ROOT / "artifacts" / "national_scorecard.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nScorecard saved: {out_path}")

    print("\n" + "=" * 70)
    print("DETAILED GENERALIZATION REPORT")
    print("=" * 70)

    for model_name, scores in all_results.items():
        if "error" in scores:
            print(f"\n{model_name}: {scores['error']}")
            continue

        print(f"\n{'='*50}")
        print(f"Model: {model_name}")
        print(f"{'='*50}")

        o = scores["overall"]
        print(f"\n  Overall: recall={o['recall']:.4f} precision={o['precision']:.4f} "
              f"TNR={o['tnr']:.4f} F1={o['f1']:.4f}")
        print(f"  Confusion: TP={o['TP']} FN={o['FN']} FP={o['FP']} TN={o['TN']}")

        if "per_station" in scores:
            print(f"\n  Per Station:")
            for st, sv in sorted(scores["per_station"].items()):
                parts = [f"ev={sv['n_events']}", f"ns={sv['n_noise']}"]
                if "recall" in sv:
                    parts.append(f"recall={sv['recall']:.3f}")
                if "tnr" in sv:
                    parts.append(f"tnr={sv['tnr']:.3f}")
                print(f"    {st:14s} {', '.join(parts)}")

        if "per_magnitude" in scores:
            print(f"\n  Per Magnitude Band:")
            for band, bv in scores["per_magnitude"].items():
                print(f"    {band:10s}: {bv['detected']}/{bv['n']} "
                      f"(recall={bv['recall']:.3f})")

    print("\n" + "=" * 70)
    print("GENERALIZATION VERDICT")
    print("=" * 70)

    for model_name, scores in all_results.items():
        if "error" in scores:
            continue
        o = scores["overall"]
        if o["recall"] >= 0.80 and o["tnr"] >= 0.80:
            verdict = "STRONG GENERALIZATION"
        elif o["recall"] >= 0.60 and o["tnr"] >= 0.60:
            verdict = "MODERATE GENERALIZATION"
        elif o["recall"] >= 0.40:
            verdict = "WEAK GENERALIZATION — needs more data"
        else:
            verdict = "POOR — model does not generalize"
        print(f"  {model_name}: {verdict} (R={o['recall']:.3f}, TNR={o['tnr']:.3f})")


if __name__ == "__main__":
    main()
