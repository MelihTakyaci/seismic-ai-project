# Phase: 7
# Purpose: Fit per-station z-score normalizer from augmented dataset to reduce station identity leakage
# Inputs: data/augmented_dataset/{waveforms.hdf5, metadata.csv}, models/trace_classifier.joblib
# Outputs: models/station_normalizer.json
# Limitations: Fitted on training data (same 2 stations); generalizes to unseen stations via global fallback

import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import h5py

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from seismic_engine.config import Paths, GPDConfig, SRATE
from seismic_engine.models.gpd_embedder import GPDEmbedder
from seismic_engine.inference.trace_classifier import TraceClassifier
from seismic_engine.inference.station_normalizer import StationNormalizer
from dataclasses import replace


def main():
    paths = Paths()
    meta = pd.read_csv(paths.metadata)
    hf = h5py.File(str(paths.hdf5), "r")

    orig = meta[meta["augmentation"] == "original"].copy()
    print(f"Fitting station normalizer on {len(orig)} original traces")
    print(f"Stations: {sorted(orig['station'].unique())}")

    cfg = replace(GPDConfig(), sliding_stride=200)
    embedder = GPDEmbedder(device="cpu")

    features_by_station = defaultdict(list)

    for idx, row in orig.iterrows():
        trace_name = row["trace_name"]
        station = row["station"]
        data = hf["data"][trace_name][:]

        embeddings, centers, p_probs = embedder.extract_sliding(data, stride=cfg.sliding_stride)
        if len(embeddings) == 0:
            continue

        features = TraceClassifier.compute_trace_features(embeddings)
        features_by_station[station].append(features)

        if (idx + 1) % 100 == 0:
            print(f"  [{idx+1}/{len(orig)}] processed...")

    hf.close()

    features_arrays = {
        station: np.array(feats) for station, feats in features_by_station.items()
    }

    normalizer = StationNormalizer()
    stats = normalizer.fit(features_arrays)
    print(f"\nNormalizer fitted: {stats}")

    out_path = ROOT / "models" / "station_normalizer.json"
    normalizer.save(out_path)
    print(f"Saved to: {out_path}")

    for station, feats in features_arrays.items():
        normed = normalizer.transform(feats, station=station)
        print(f"  {station}: raw mean={feats.mean():.4f}, std={feats.std():.4f} "
              f"-> normed mean={normed.mean():.4f}, std={normed.std():.4f}")


if __name__ == "__main__":
    main()
