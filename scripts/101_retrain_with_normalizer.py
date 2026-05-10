# Phase: 7
# Purpose: Retrain SVM pipeline with per-station z-score normalization applied before StandardScaler
# Inputs: data/augmented_dataset/{waveforms.hdf5, metadata.csv}, models/station_normalizer.json
# Outputs: models/trace_classifier.joblib (updated with normalization-aware pipeline)
# Limitations: Same 2-station training set; normalizer must be fitted first

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

    normalizer = StationNormalizer()
    normalizer.load(ROOT / "models" / "station_normalizer.json")

    cfg = replace(GPDConfig(), sliding_stride=200)
    embedder = GPDEmbedder(device="cpu")

    orig = meta[meta["augmentation"] == "original"].copy()
    events = orig[orig["window_type"] == "event"]
    noise = orig[orig["window_type"] == "noise"]

    bg_noise = meta[(meta["window_type"] == "noise") & (meta["augmentation"] == "background")]
    bg_sample = bg_noise.sample(n=min(300, len(bg_noise)), random_state=42)

    all_train = pd.concat([events, noise, bg_sample], ignore_index=True).sample(
        frac=1, random_state=42
    ).reset_index(drop=True)

    print(f"Training set: {len(all_train)} traces")
    print(f"  Events: {(all_train['window_type'] == 'event').sum()}")
    print(f"  Noise: {(all_train['window_type'] == 'noise').sum()}")
    print()

    features_list = []
    labels = []
    stations = []

    for idx, row in all_train.iterrows():
        trace_name = row["trace_name"]
        station = row["station"]
        is_event = row["window_type"] == "event"

        data = hf["data"][trace_name][:]
        embeddings, centers, p_probs = embedder.extract_sliding(data, stride=cfg.sliding_stride)

        if len(embeddings) == 0:
            continue

        features = TraceClassifier.compute_trace_features(embeddings)
        features_norm = normalizer.transform(features, station=station)

        features_list.append(features_norm)
        labels.append(1 if is_event else 0)
        stations.append(station)

        if (len(features_list)) % 100 == 0:
            print(f"  [{len(features_list)}/{len(all_train)}] extracted...")

    hf.close()

    X = np.array(features_list)
    y = np.array(labels)
    print(f"\nFeature matrix: {X.shape}")
    print(f"Labels: {(y == 1).sum()} events, {(y == 0).sum()} noise")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_pca = min(100, X_scaled.shape[0] - 1, X_scaled.shape[1])
    pca = PCA(n_components=n_pca)
    X_pca = pca.fit_transform(X_scaled)
    print(f"PCA: {n_pca} components, variance explained: {pca.explained_variance_ratio_.sum():.4f}")

    svm = SVC(
        kernel="rbf", C=20.0, gamma="scale",
        class_weight="balanced", probability=True,
    )
    svm.fit(X_pca, y)

    y_pred = svm.predict(X_pca)
    y_proba = svm.predict_proba(X_pca)[:, list(svm.classes_).index(1)]

    print(f"\n--- Training Metrics ---")
    print(classification_report(y, y_pred, target_names=["noise", "event"]))

    recall = recall_score(y, y_pred, pos_label=1)
    tnr = recall_score(y, y_pred, pos_label=0)
    print(f"Recall: {recall:.4f}, TNR: {tnr:.4f}")

    thresholds = [0.30, 0.35, 0.40, 0.44, 0.50, 0.55, 0.60]
    print(f"\n--- Threshold Sweep ---")
    for thr in thresholds:
        y_t = (y_proba >= thr).astype(int)
        r = recall_score(y, y_t, pos_label=1)
        t = recall_score(y, y_t, pos_label=0)
        print(f"  threshold={thr:.2f}: recall={r:.4f}, TNR={t:.4f}, F1={2*r*t/(r+t) if (r+t)>0 else 0:.4f}")

    best_thr = 0.44
    print(f"\nUsing threshold: {best_thr}")

    out_path = ROOT / "models" / "trace_classifier.joblib"
    joblib.dump({
        "scaler": scaler,
        "pca": pca,
        "svm": svm,
        "svm_C": 20.0,
        "n_pca": n_pca,
        "detection_threshold": best_thr,
    }, str(out_path))
    print(f"Saved retrained model: {out_path}")

    per_station = defaultdict(lambda: {"tp": 0, "fn": 0, "fp": 0, "tn": 0})
    for i, (label, pred, station) in enumerate(zip(y, y_pred, stations)):
        if label == 1 and pred == 1:
            per_station[station]["tp"] += 1
        elif label == 1 and pred == 0:
            per_station[station]["fn"] += 1
        elif label == 0 and pred == 1:
            per_station[station]["fp"] += 1
        else:
            per_station[station]["tn"] += 1

    print(f"\n--- Per-Station Results ---")
    for st_name, counts in sorted(per_station.items()):
        st_recall = counts["tp"] / (counts["tp"] + counts["fn"]) if (counts["tp"] + counts["fn"]) > 0 else 0
        st_tnr = counts["tn"] / (counts["tn"] + counts["fp"]) if (counts["tn"] + counts["fp"]) > 0 else 0
        print(f"  {st_name}: recall={st_recall:.4f}, TNR={st_tnr:.4f} "
              f"(TP={counts['tp']}, FN={counts['fn']}, FP={counts['fp']}, TN={counts['tn']})")


if __name__ == "__main__":
    main()
