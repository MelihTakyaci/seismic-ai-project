# Methodology Notes — Seismic AI Pipeline
# Updated: Phase 3 (2026-04-24)
# Note: This file is indexed by the Phase 6 RAG layer.

---

## M1 — Detection Criterion Definition

All detection results in artifacts/detection_results.json use the following criterion:

**An event window is "detected" by a model if the model produces a P-wave pick
in the interval [origin_s − 5s, origin_s + 25s]** within the 60-second window,
where `origin_s = 30.0 s` is the position of the catalog origin time.

Rationale: P-wave travel time from aftershock to station ranges from ~0s (co-located)
to ~25s (station at ~150 km hypocentral distance at 6 km/s). The −5s allowance covers
catalog origin time uncertainty. This criterion was revised during Phase 3 after initial
inspection revealed that the first attempted criterion (±5s symmetric) incorrectly
rejected valid detections at distal stations.

For STA/LTA: "detected" means first trigger time falls in this interval.
For DL models: "detected" means peak P-probability pick falls in this interval.

**False positive definition (noise windows):** Any model trigger or P pick at any time
in a noise window. Note: noise window labels are unreliable (see limitations.md L2).

---

## M2 — Preprocessing Pipeline

Applied uniformly to all 226 pilot NPZ windows (161 event, 65 noise):

1. Load raw miniSEED via ObsPy
2. Merge traces; zero-fill gaps ≤ 5% of window length
3. Detrend (linear polynomial fit, removed per trace)
4. Bandpass filter: 1–45 Hz, 4th order Butterworth, zero-phase (ObsPy `filter`)
5. Resample to 100.0 Hz (SeisBench standard input rate)
6. Per-trace z-score normalization: divide by standard deviation if std > 0
7. Trim/pad to exactly 6000 samples (60.0 s at 100 Hz)
8. Save as NPZ: array shape (3, 6000) — [HHZ, HHN, HHE]

Windows discarded if: any of HHZ, HHN, HHE missing, OR gap fraction > 5%.

---

## M3 — STA/LTA Baseline

- Algorithm: recursive STA/LTA (ObsPy `recursive_sta_lta`)
- Applied to HHZ component only
- STA window: 0.5 s (50 samples at 100 Hz)
- LTA window: 10.0 s (1000 samples at 100 Hz)
- Detection threshold: 3.0 (standard operational value)
- De-trigger threshold: 1.5 (50% of trigger threshold)
- No P/S phase picking (STA/LTA provides only trigger times)

---

## M4 — SeisBench Model Configurations (Zero-Shot)

All three models loaded with pre-trained weights (no fine-tuning on Turkish data).

| Model | Weights | Input shape | Window | Outputs |
|-------|---------|-------------|--------|---------|
| PhaseNet | `original` (STEAD) | (3, 3001) | 30.01 s | P prob, S prob, noise prob |
| EQTransformer | `original` (STEAD) | (3, 6000) | 60 s | detection prob, P prob, S prob |
| GPD | `original` (STEAD) | (1, 400) | 4 s | P/S/noise class prob |

All models invoked via `model.annotate(obspy_stream)`, which handles internal windowing
and striding. P and S picks extracted as time of peak probability exceeding threshold 0.3.

Input streams reconstructed from NPZ arrays with `starttime = UTCDateTime("2023-02-06T01:00:00")`
as a synthetic reference; all pick times are converted to seconds-in-window coordinates.

---

## M5 — EQTransformer Zero-Shot Anomaly (Phase 3 Finding)

EQTransformer yielded detection rates of 11–21% across all magnitude bands (M 2–4+),
substantially below PhaseNet (59–77%) and GPD (62–77%) on the same data.

This is not consistent with EQTransformer's reported performance on STEAD benchmarks,
where it equals or exceeds PhaseNet. Probable causes:

1. **Domain mismatch:** EQTransformer was pre-trained on STEAD (primarily North American
   events). The 2023 Kahramanmaraş sequence has waveform character (attenuation, coda
   length, S/P amplitude ratio) differing from the STEAD distribution.

2. **Annotation channel mapping:** The EQTransformer internal pick probability may peak
   at probabilities just below 0.3 for many Turkish events. The threshold 0.3 is a
   conservative choice that may be too high for zero-shot cross-domain inference.

3. **Window alignment sensitivity:** EQTransformer's transformer architecture is
   known to be more sensitive to exact trace normalization and input alignment than
   the convolutional PhaseNet.

**Recommended action for Phase 5:** Lower EQTransformer P/S threshold to 0.1 or run
a threshold sweep (0.05–0.5) to find the operating point that maximises F1. Fine-tuning
on a Turkish labeled subset is the definitive solution.

---

## M7 — Phase Picking MAE Qualification (Phase 4 Finding)

The P-pick MAE values reported in artifacts/evaluation_metrics.json (PhaseNet: 9.6 s,
EQTransformer: 14.4 s, GPD: 11.1 s) are inflated by two systematic factors:

1. **Out-of-window theoretical arrivals.** The 2023 Kahramanmaraş aftershock zone spans
   ~300 km along strike. For aftershocks at the far NE segment (38.0–38.5°N) recorded
   at KO.KOZT (37.48°N), epicentral distances reach ~150 km, giving IASP91 P travel
   times of ~22 s (theoretical_P_in_window ≈ 52 s). If the model picks at t=38 s, the
   error is 14 s — not because the pick is wrong, but because the distant aftershock P
   arrives near the window edge and the theoretical reference may not match the local
   Turkish crustal structure.

2. **Non-detected event picks included.** The MAE computation includes all events where
   the model produced any P pick, regardless of whether that pick was within the
   detection window [origin−5 s, origin+25 s]. Picks from events that were NOT detected
   (offset > 25 s from origin) contribute disproportionately large errors.

**Qualified MAE interpretation:** The reported P-MAE values are upper bounds on picking
error. A more defensible estimate would restrict to (a) detected events only and (b)
events where theoretical P arrival is within the 60-second window. This is noted as a
future refinement for Phase 5. The current values should NOT be compared directly to
published PhaseNet or EQTransformer benchmark results on STEAD.

---

## M8 — Phase 5 Scale-Up Results and Comparison with Phase 4

**Source:** artifacts/evaluation_metrics.json (Phase 5), artifacts/phase4_backup/evaluation_metrics.json

Phase 5 scale-up (12 KO stations, 6 months) produced 4,835 preprocessed event windows
from 5,187 successfully downloaded miniSEED files. Zero-shot recall results:

| Model         | Phase 4 Recall | Phase 5 Recall | Direction |
|---------------|----------------|----------------|-----------|
| STA/LTA       | 0.839          | 0.484          | ↓         |
| PhaseNet      | 0.665          | 0.319          | ↓         |
| EQTransformer | 0.174          | 0.051          | ↓         |
| GPD           | 0.702          | 0.593          | ↓         |

All four models show lower recall in Phase 5 than Phase 4. This is not explained by
model behaviour alone — two structural factors explain most of the decline:

1. **Detection window truncation (see limitations.md L10):** Phase 5 includes far
   stations at 161–241 km. For these stations, P arrivals fall outside the detection
   criterion [origin−5s, origin+25s] even when the model picks them correctly.

2. **Non-uniform data coverage:** 96.2% of download attempts failed due to FDSN
   parallel client initialization collision (see limitations.md L9). The 5,187
   successful downloads are not uniformly distributed across the 11,338 events or
   6 months. This may introduce selection bias in the evaluation sample.

GPD retains the highest recall in Phase 5 (0.593), consistent with its Phase 4
performance. STA/LTA drops substantially (0.839 → 0.484), suggesting that the
automatic trigger threshold 3.0 is not optimal for mixed near/far station data.
EQTransformer remains below 0.10 — domain mismatch persists across both phases.

---

## M6 — Noise Window Cleaning Strategy (Phase 3)

Because noise window labels are contaminated (see limitations.md L2), false positive
rates computed in Phase 4 will be qualified as "upper bound" values. Phase 4 will also
apply a post-hoc STA/LTA filter: any noise window with peak STA/LTA > 3.0 will be
flagged as "likely_contaminated" and excluded from the primary false positive computation,
with rates reported separately for clean vs. all noise windows.
