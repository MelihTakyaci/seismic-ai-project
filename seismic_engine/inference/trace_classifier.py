"""
Trace-level event/noise classifier using aggregated GPD embeddings.

Architecture:
  1. GPD sliding windows → per-window 200-dim bn5 embeddings
  2. Aggregate across windows: mean, std, max, skew → 800 base features
  3. Temporal dynamics: max_change, mean_change, change_std, max_dist, n_outlier → 5 features
  4. StandardScaler → PCA(75) → SVM RBF(C=20) → binary event/noise decision
  5. If event: use GPD P-probs for P-arrival location
  6. If noise: suppress output (zero P-probs)

This eliminates the max-amplification problem that plagues window-level classification.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import skew as scipy_skew
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import joblib

from ..config import DEVICE, NPTS, SRATE, GPDConfig
from ..models.gpd_embedder import GPDEmbedder
from .station_normalizer import StationNormalizer


class TraceClassifier:
    """Trace-level event/noise gate using aggregated embedding features."""

    def __init__(
        self,
        embedder: GPDEmbedder,
        cfg: GPDConfig = GPDConfig(),
        svm_C: float = 20.0,
        n_pca: int = 75,
        detection_threshold: float = 0.69,
    ):
        self.embedder = embedder
        self.cfg = cfg
        self.svm_C = svm_C
        self.n_pca = n_pca
        self.detection_threshold = detection_threshold

        self.scaler: Optional[StandardScaler] = None
        self.pca: Optional[PCA] = None
        self.svm: Optional[SVC] = None
        self.station_normalizer: Optional[StationNormalizer] = None
        self._fitted = False

    @staticmethod
    def compute_trace_features(embeddings: np.ndarray) -> np.ndarray:
        """
        Compute 805-dim trace feature vector from per-window embeddings.

        embeddings: (N_windows, 200) GPD bn5 embeddings
        Returns: (805,) feature vector
        """
        base = np.concatenate([
            embeddings.mean(axis=0),
            embeddings.std(axis=0),
            embeddings.max(axis=0),
            scipy_skew(embeddings, axis=0),
        ])

        if len(embeddings) > 1:
            diffs = np.linalg.norm(np.diff(embeddings, axis=0), axis=1)
            max_change = diffs.max()
            mean_change = diffs.mean()
            change_std = diffs.std()
        else:
            max_change = mean_change = change_std = 0.0

        mean_emb = embeddings.mean(axis=0)
        dists = np.linalg.norm(embeddings - mean_emb, axis=1)
        max_dist = dists.max()
        dist_threshold = dists.mean() + 2 * dists.std() if dists.std() > 0 else dists.max() + 1
        n_outlier = int((dists > dist_threshold).sum())

        temporal = np.array([
            max_change, mean_change, change_std, max_dist, n_outlier,
        ], dtype=np.float32)

        return np.concatenate([base, temporal])

    def fit(
        self,
        event_traces: list,
        noise_traces: list,
    ) -> Dict:
        """
        Train the trace-level classifier.

        event_traces: list of dicts with 'embeddings' key (N_win, 200)
        noise_traces: list of dicts with 'embeddings' key (N_win, 200)
        """
        X_event = np.array([
            self.compute_trace_features(t["embeddings"]) for t in event_traces
        ])
        X_noise = np.array([
            self.compute_trace_features(t["embeddings"]) for t in noise_traces
        ])

        X = np.vstack([X_event, X_noise])
        y = np.concatenate([np.ones(len(X_event)), np.zeros(len(X_noise))]).astype(int)

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        n_components = min(self.n_pca, X_scaled.shape[0] - 1, X_scaled.shape[1])
        self.pca = PCA(n_components=n_components)
        X_pca = self.pca.fit_transform(X_scaled)

        self.svm = SVC(
            kernel="rbf", C=self.svm_C, gamma="scale",
            class_weight="balanced", probability=True,
        )
        self.svm.fit(X_pca, y)
        self._fitted = True

        y_pred = self.svm.predict(X_pca)
        from sklearn.metrics import recall_score
        train_recall = recall_score(y, y_pred, pos_label=1)
        train_tnr = recall_score(y, y_pred, pos_label=0)

        metrics = {
            "n_event": len(X_event),
            "n_noise": len(X_noise),
            "feature_dim": int(X.shape[1]),
            "pca_components": n_components,
            "variance_explained": round(float(self.pca.explained_variance_ratio_.sum()), 4),
            "svm_C": self.svm_C,
            "train_recall": round(float(train_recall), 4),
            "train_tnr": round(float(train_tnr), 4),
        }
        print(f"  TraceClassifier: {len(X)} traces, PCA({n_components}), "
              f"recall={train_recall:.1%}, TNR={train_tnr:.1%}")
        return metrics

    def predict(
        self, data: np.ndarray, n_samples: Optional[int] = None,
        station: Optional[str] = None,
    ) -> Dict:
        """
        Two-stage hybrid inference on a single trace.

        Stage 1: Trace-level SVM classifies entire trace as event/noise
        Stage 2: If event, use GPD P-probs for P-arrival location

        Args:
            station: Optional station code (e.g. "KO.KMRS") for per-station
                     z-score normalization that strips station identity leakage.
        """
        n = data.shape[1] if n_samples is None else n_samples

        embeddings, centers, p_probs = self.embedder.extract_sliding(
            data, stride=self.cfg.sliding_stride,
        )

        if len(p_probs) == 0:
            return self._empty_result(n)

        features = self.compute_trace_features(embeddings)

        if self.station_normalizer is not None:
            features = self.station_normalizer.transform(features, station=station)

        features_scaled = self.scaler.transform(features.reshape(1, -1))
        features_pca = self.pca.transform(features_scaled)

        event_prob = float(self.svm.predict_proba(features_pca)[0, list(self.svm.classes_).index(1)])
        is_event = event_prob >= self.detection_threshold

        if is_event:
            interp_fn = interp1d(
                centers, p_probs.astype(float), kind="linear",
                bounds_error=False,
                fill_value=(float(p_probs[0]), float(p_probs[-1])),
            )
            p_prob_full = interp_fn(np.arange(n, dtype=float)).astype(np.float32)
        else:
            p_prob_full = np.zeros(n, dtype=np.float32)

        return {
            "p_prob": p_prob_full,
            "trace_event_prob": event_prob,
            "is_event": is_event,
            "gpd_max_prob": float(p_probs.max()),
            "n_total_windows": len(p_probs),
        }

    def predict_proba(
        self, data: np.ndarray, n_samples: Optional[int] = None,
    ) -> np.ndarray:
        result = self.predict(data, n_samples)
        return result["p_prob"]

    def _empty_result(self, n: int) -> Dict:
        return {
            "p_prob": np.zeros(n, dtype=np.float32),
            "trace_event_prob": 0.0,
            "is_event": False,
            "gpd_max_prob": 0.0,
            "n_total_windows": 0,
        }

    def save(self, path: Path) -> None:
        joblib.dump({
            "scaler": self.scaler,
            "pca": self.pca,
            "svm": self.svm,
            "svm_C": self.svm_C,
            "n_pca": self.n_pca,
            "detection_threshold": self.detection_threshold,
        }, str(path))
        print(f"  Saved trace classifier: {path}")

    def load(self, path: Path) -> None:
        from dataclasses import replace as _replace
        data = joblib.load(str(path))
        self.scaler = data["scaler"]
        self.pca = data["pca"]
        self.svm = data["svm"]
        self.svm_C = data.get("svm_C", 20.0)
        self.n_pca = data.get("n_pca", 75)
        self.detection_threshold = data.get("detection_threshold", 0.69)

        saved_stride = data.get("sliding_stride")
        if saved_stride and saved_stride != self.cfg.sliding_stride:
            self.cfg = _replace(self.cfg, sliding_stride=saved_stride)

        self._fitted = True

        normalizer_path = path.parent / "station_normalizer.json"
        if normalizer_path.exists():
            self.station_normalizer = StationNormalizer()
            self.station_normalizer.load(normalizer_path)
        print(f"  Loaded trace classifier: {path}")
