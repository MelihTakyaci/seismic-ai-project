"""
Per-station Z-score normalization for GPD bn5 features.

Strips station-specific baseline noise from the 805-dim feature vector
before PCA/SVM, reducing station identity leakage from 84.8% toward chance.
"""

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np


class StationNormalizer:
    """Maintains per-station mean/std for bn5 feature normalization."""

    def __init__(self):
        self.station_stats: Dict[str, Dict[str, np.ndarray]] = {}
        self._global_mean: Optional[np.ndarray] = None
        self._global_std: Optional[np.ndarray] = None
        self._fitted = False

    def fit(self, features_by_station: Dict[str, np.ndarray]) -> Dict:
        """
        Compute per-station and global statistics.

        features_by_station: {station_name: (N_traces, 805) array}
        """
        all_features = []

        for station, feats in features_by_station.items():
            if len(feats) < 2:
                continue
            self.station_stats[station] = {
                "mean": feats.mean(axis=0),
                "std": feats.std(axis=0) + 1e-8,
                "n": len(feats),
            }
            all_features.append(feats)

        if all_features:
            combined = np.vstack(all_features)
            self._global_mean = combined.mean(axis=0)
            self._global_std = combined.std(axis=0) + 1e-8

        self._fitted = True

        return {
            "n_stations": len(self.station_stats),
            "stations": list(self.station_stats.keys()),
            "total_traces": sum(s["n"] for s in self.station_stats.values()),
        }

    def transform(self, features: np.ndarray, station: Optional[str] = None) -> np.ndarray:
        """
        Apply per-station z-score normalization.

        If station is known and has stats, use station-specific normalization.
        Otherwise fall back to global normalization.
        """
        if not self._fitted:
            return features

        is_1d = features.ndim == 1
        if is_1d:
            features = features.reshape(1, -1)

        if station and station in self.station_stats:
            mean = self.station_stats[station]["mean"]
            std = self.station_stats[station]["std"]
        elif self._global_mean is not None:
            mean = self._global_mean
            std = self._global_std
        else:
            return features.squeeze() if is_1d else features

        normalized = (features - mean) / std

        return normalized.squeeze() if is_1d else normalized

    def save(self, path: Path) -> None:
        data = {
            "station_stats": {
                k: {"mean": v["mean"].tolist(), "std": v["std"].tolist(), "n": v["n"]}
                for k, v in self.station_stats.items()
            },
            "global_mean": self._global_mean.tolist() if self._global_mean is not None else None,
            "global_std": self._global_std.tolist() if self._global_std is not None else None,
        }
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"  Saved station normalizer: {path}")

    def load(self, path: Path) -> None:
        with open(path, "r") as f:
            data = json.load(f)

        for k, v in data["station_stats"].items():
            self.station_stats[k] = {
                "mean": np.array(v["mean"], dtype=np.float32),
                "std": np.array(v["std"], dtype=np.float32),
                "n": v["n"],
            }

        if data["global_mean"] is not None:
            self._global_mean = np.array(data["global_mean"], dtype=np.float32)
            self._global_std = np.array(data["global_std"], dtype=np.float32)

        self._fitted = True
        print(f"  Loaded station normalizer: {path} ({len(self.station_stats)} stations)")
