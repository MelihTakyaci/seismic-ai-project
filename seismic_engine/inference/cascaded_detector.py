"""
Cascaded Detector: GPD+SVM trigger → PhaseNet precise phase picker.

Architecture:
  Stage 1 (Trigger): GPD sliding windows + SVM → binary event/noise decision
  Stage 2 (Picker):  PhaseNet U-Net on detected events → sample-level P/S picks

This replaces the crude GPD argmax (MAE ~16s) with PhaseNet's specialized
U-Net architecture designed for sample-level phase picking (target MAE ~0.1s).
"""

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import seisbench.models as sbm

from ..config import DEVICE, NPTS, SRATE, GPDConfig
from ..models.gpd_embedder import GPDEmbedder
from .trace_classifier import TraceClassifier


class CascadedDetector:
    """
    Two-stage cascade: GPD+SVM detection → PhaseNet phase picking.

    Usage:
        detector = CascadedDetector()
        result = detector.predict(data_3ch, station="KO.KMRS")
    """

    def __init__(
        self,
        classifier: Optional[TraceClassifier] = None,
        phasenet_model: Optional[object] = None,
        device: str = "cpu",
    ):
        self._device = device
        self._classifier = classifier
        self._phasenet = phasenet_model
        self._pn_ready = False

    def load(self, model_dir: Path) -> None:
        """Load both models from the models/ directory."""
        if self._classifier is None:
            embedder = GPDEmbedder(device=self._device)
            self._classifier = TraceClassifier(embedder, cfg=GPDConfig(), svm_C=20.0, n_pca=100)
            self._classifier.load(model_dir / "trace_classifier.joblib")

        if self._phasenet is None:
            self._phasenet = sbm.PhaseNet.from_pretrained("instance")
            self._phasenet.to(self._device).eval()
            self._pn_ready = True

    @property
    def is_ready(self) -> bool:
        return self._classifier is not None and self._classifier._fitted and self._pn_ready

    def predict(
        self,
        data: np.ndarray,
        station: Optional[str] = None,
    ) -> Dict:
        """
        Full cascade: detect then pick.

        Args:
            data: (3, N) waveform array, 100 Hz sampling rate
            station: station code for z-score normalization

        Returns:
            Dict with detection result + PhaseNet P/S picks if event detected.
        """
        if not self.is_ready:
            raise RuntimeError("CascadedDetector not loaded. Call .load() first.")

        detection = self._classifier.predict(data, station=station)

        result = {
            "is_event": detection["is_event"],
            "trace_event_prob": detection["trace_event_prob"],
            "gpd_max_prob": detection["gpd_max_prob"],
            "n_total_windows": detection["n_total_windows"],
            "p_prob_gpd": detection["p_prob"],
            "phasenet_p_sample": None,
            "phasenet_p_seconds": None,
            "phasenet_p_confidence": None,
            "phasenet_s_sample": None,
            "phasenet_s_seconds": None,
            "phasenet_s_confidence": None,
            "phasenet_p_prob": None,
            "phasenet_s_prob": None,
            "picker_used": "gpd_argmax",
        }

        if detection["is_event"]:
            pn_result = self._run_phasenet(data)
            result.update(pn_result)
            result["picker_used"] = "phasenet"

        return result

    def _run_phasenet(self, data: np.ndarray) -> Dict:
        """
        Run PhaseNet on a single 3-component trace.

        PhaseNet expects (batch, 3, N) input normalized per-trace.
        Output: (batch, 3, N) with channels [P, S, Noise].
        """
        n_samples = data.shape[1]

        trace = data.astype(np.float32).copy()
        for ch in range(3):
            ch_data = trace[ch]
            peak = np.abs(ch_data).max()
            if peak > 1e-9:
                trace[ch] = ch_data / peak

        pn_in_samples = getattr(self._phasenet, "in_samples", 3001)

        if n_samples >= pn_in_samples:
            picks = self._phasenet_single_pass(trace, n_samples, pn_in_samples)
        else:
            padded = np.zeros((3, pn_in_samples), dtype=np.float32)
            padded[:, :n_samples] = trace
            picks = self._phasenet_single_pass(padded, pn_in_samples, pn_in_samples)

        return picks

    def _phasenet_single_pass(
        self, trace: np.ndarray, n_samples: int, pn_in_samples: int
    ) -> Dict:
        """Sliding window PhaseNet with overlap-add for traces longer than model input."""
        stride = pn_in_samples // 2
        positions = list(range(0, max(1, n_samples - pn_in_samples + 1), stride))
        if not positions:
            positions = [0]

        p_prob_full = np.zeros(n_samples, dtype=np.float32)
        s_prob_full = np.zeros(n_samples, dtype=np.float32)
        count = np.zeros(n_samples, dtype=np.float32)

        for start in positions:
            end = start + pn_in_samples
            if end > n_samples:
                start = max(0, n_samples - pn_in_samples)
                end = start + pn_in_samples

            segment = trace[:, start:end]
            if segment.shape[1] < pn_in_samples:
                pad_seg = np.zeros((3, pn_in_samples), dtype=np.float32)
                pad_seg[:, :segment.shape[1]] = segment
                segment = pad_seg

            x = torch.tensor(segment[np.newaxis], dtype=torch.float32).to(self._device)
            with torch.no_grad():
                out = self._phasenet(x)

            if isinstance(out, torch.Tensor):
                probs = out[0].cpu().numpy()
            else:
                probs = out[0].cpu().numpy()

            seg_len = min(pn_in_samples, n_samples - start)
            p_prob_full[start:start + seg_len] += probs[0, :seg_len]
            s_prob_full[start:start + seg_len] += probs[1, :seg_len]
            count[start:start + seg_len] += 1.0

        mask = count > 0
        p_prob_full[mask] /= count[mask]
        s_prob_full[mask] /= count[mask]

        p_peak_idx = int(np.argmax(p_prob_full))
        p_confidence = float(p_prob_full[p_peak_idx])
        s_peak_idx = int(np.argmax(s_prob_full))
        s_confidence = float(s_prob_full[s_peak_idx])

        result = {
            "phasenet_p_prob": p_prob_full,
            "phasenet_s_prob": s_prob_full,
        }

        if p_confidence > 0.1:
            result["phasenet_p_sample"] = p_peak_idx
            result["phasenet_p_seconds"] = round(p_peak_idx / SRATE, 4)
            result["phasenet_p_confidence"] = round(p_confidence, 4)

        if s_confidence > 0.1:
            result["phasenet_s_sample"] = s_peak_idx
            result["phasenet_s_seconds"] = round(s_peak_idx / SRATE, 4)
            result["phasenet_s_confidence"] = round(s_confidence, 4)

        return result
