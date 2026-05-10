# Phase: 7
# Purpose: Retrain with station normalizer fitted on NOISE ONLY (station baseline), then retrain SVM
# Inputs: data/augmented_dataset/{waveforms.hdf5, metadata.csv}
# Outputs: models/station_normalizer.json, models/trace_classifier.joblib
# Limitations: Normalizer learns noise floor per station; events deviate from this baseline

import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import h5py
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import recall_score, classification_report
import joblib

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dataclasses import replace
from seismic_engine.config import Paths, GPDConfig
from seismic_engine.models.gpd_embedder import GPDEmbedder
from seismic_engine.inference.trace_classifier import TraceClassifier
from seismic_engine.inference.station_normalizer import StationNormalizer


def main():
    paths = Paths()
    meta = pd.read_csv(paths.metadata)
    hf = h5py.File(str(paths.hdf5), "r")

    cfg = replace(GPDConfig(), sliding_stride=200)
    embedder = GPDEmbedder(device="cpu")

    # --- Step 1: Fit normalizer on NOISE ONLY ---
    print("=" * 60)
    print("STEP 1: Fitting station normalizer on NOISE traces only")
    print("=" * 60)

    noise_orig = meta[
        (meta["window_type"] == "noise") & (meta["augmentation"] == "original")
    ]
    print(f"Noise traces for normalizer: {len(noise_orig)}")

    noise_features_by_station = defaultdict(list)
    for idx, row in noise_orig.iterrows():
        data = hf["data"][row["trace_name"]][:]
        embeddings, _, _ = embedder.extract_sliding(data, stride=cfg.sliding_stride)
        if len(embeddings) == 0:
            continue
        features = TraceClassifier.compute_trace_features(embeddings)
        noise_features_by_station[row["station"]].append(features)

    noise_arrays = {st: np.array(f) for st, f in noise_features_by_station.items()}
    normalizer = StationNormalizer()
    stats = normalizer.fit(noise_arrays)
    print(f"Normalizer fitted: {stats}")
    normalizer.save(ROOT / "models" / "station_normalizer.json")

    # --- Step 2: Extract ALL features with normalization ---
    print("\n" + "=" * 60)
    print("STEP 2: Extracting normalized features for training")
    print("=" * 60)

    # Use all original traces + background noise subset
    orig = meta[meta["augmentation"] == "original"].copy()
    bg_noise = meta[(meta["window_type"] == "noise") & (meta["augmentation"] == "background")]
    bg_sample = bg_noise.sample(n=min(400, len(bg_noise)), random_state=42)

    all_train = pd.concat([orig, bg_sample], ignore_index=True).sample(
        frac=1, random_state=42
    ).reset_index(drop=True)

    print(f"Training set: {len(all_train)} traces")
    print(f"  Events: {(all_train['window_type'] == 'event').sum()}")
    print(f"  Noise: {(all_train['window_type'] == 'noise').sum()}")

    features_list = []
    labels = []
    stations = []

    for idx, row in all_train.iterrows():
        data = hf["data"][row["trace_name"]][:]
        embeddings, _, _ = embedder.extract_sliding(data, stride=cfg.sliding_stride)
        if len(embeddings) == 0:
            continue

        features = TraceClassifier.compute_trace_features(embeddings)
        features_norm = normalizer.transform(features, station=row["station"])

        features_list.append(features_norm)
        labels.append(1 if row["window_type"] == "event" else 0)
        stations.append(row["station"])

        if len(features_list) % 200 == 0:
            print(f"  [{len(features_list)}/{len(all_train)}] extracted...")

    hf.close()

    X = np.array(features_list)
    y = np.array(labels)
    print(f"\nFeature matrix: {X.shape}")

    # --- Step 3: Train SVM ---
    print("\n" + "=" * 60)
    print("STEP 3: Training SVM on normalized features")
    print("=" * 60)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_pca = min(100, X_scaled.shape[0] - 1, X_scaled.shape[1])
    pca = PCA(n_components=n_pca)
    X_pca = pca.fit_transform(X_scaled)
    print(f"PCA: {n_pca} components, variance: {pca.explained_variance_ratio_.sum():.4f}")

    svm = SVC(
        kernel="rbf", C=20.0, gamma="scale",
        class_weight="balanced", probability=True,
    )
    svm.fit(X_pca, y)

    y_proba = svm.predict_proba(X_pca)[:, list(svm.classes_).index(1)]

    # --- Step 4: Threshold sweep ---
    print("\n--- Threshold Sweep ---")
    best_thr = 0.44
    best_f1 = 0
    for thr in [0.30, 0.35, 0.40, 0.44, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        y_t = (y_proba >= thr).astype(int)
        r = recall_score(y, y_t, pos_label=1)
        t = recall_score(y, y_t, pos_label=0)
        f1 = 2 * r * t / (r + t) if (r + t) > 0 else 0
        marker = ""
        if t >= 0.90 and f1 > best_f1:
            best_f1 = f1
            best_thr = thr
            marker = " <-- best @ 90% TNR"
        print(f"  threshold={thr:.2f}: recall={r:.4f}, TNR={t:.4f}, F1={f1:.4f}{marker}")

    print(f"\nSelected threshold: {best_thr}")

    y_final = (y_proba >= best_thr).astype(int)
    print(f"Final: recall={recall_score(y, y_final, pos_label=1):.4f}, "
          f"TNR={recall_score(y, y_final, pos_label=0):.4f}")

    # --- Step 5: Save ---
    out_path = ROOT / "models" / "trace_classifier.joblib"
    joblib.dump({
        "scaler": scaler,
        "pca": pca,
        "svm": svm,
        "svm_C": 20.0,
        "n_pca": n_pca,
        "detection_threshold": best_thr,
        "sliding_stride": cfg.sliding_stride,
    }, str(out_path))
    print(f"\nSaved: {out_path} (stride={cfg.sliding_stride})")

    # --- Step 6: Per-station breakdown ---
    print("\n--- Per-Station (at threshold={:.2f}) ---".format(best_thr))
    per_station = defaultdict(lambda: {"tp": 0, "fn": 0, "fp": 0, "tn": 0})
    for i, (label, prob, station) in enumerate(zip(y, y_proba, stations)):
        pred = 1 if prob >= best_thr else 0
        if label == 1 and pred == 1:
            per_station[station]["tp"] += 1
        elif label == 1 and pred == 0:
            per_station[station]["fn"] += 1
        elif label == 0 and pred == 1:
            per_station[station]["fp"] += 1
        else:
            per_station[station]["tn"] += 1

    for st_name, c in sorted(per_station.items()):
        st_r = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) > 0 else float("nan")
        st_t = c["tn"] / (c["tn"] + c["fp"]) if (c["tn"] + c["fp"]) > 0 else float("nan")
        total = c["tp"] + c["fn"] + c["fp"] + c["tn"]
        print(f"  {st_name:14s}: recall={st_r:.3f} TNR={st_t:.3f} (n={total})")


if __name__ == "__main__":
    main()
