# Project Limitations — Seismic AI Pipeline
# Updated: Phase 5 (2026-04-24)
# Note: This file is indexed by the Phase 6 RAG layer. All claims reference
# specific pipeline artifacts, not external assertions.

---

## L1 — Catalog Completeness Threshold (Critical)

**Source:** data/catalog/koeri_pilot_catalog.csv, artifacts/data_access_log.md

The pilot event catalog was obtained from the EMSC FDSN event service. EMSC has a
regional detection threshold of approximately M ≈ 2.0 for the Kahramanmaraş region
during the 2023 aftershock sequence. As a result:

- 0 events with M < 2.0 appear in data/catalog/koeri_pilot_catalog.csv
- The catalog contains 2,318 events with M ≥ 2.0 for the 2-week pilot window

**Impact on primary metric:** The project's primary evaluation goal is recall on M < 2.0
events. Because no M < 2.0 events are in the current pilot catalog, recall on M < 2.0
CANNOT be directly computed from this pilot catalog. The evaluation in Phase 4 will
compute recall on M 2.0–3.0 events as a proxy, which is a weaker test of the micro-
seismic detection objective.

**Mitigation path:** Accessing the KOERI internal catalog (web interface only, not
automated) could provide M ≥ 1.0 events. This is deferred to Phase 5 (scale-up).
Phase 4 evaluation metrics are explicitly stratified by magnitude band and will clearly
report this gap.

---

## L2 — Noise Window Contamination (Critical — Discovered Phase 2)

**Source:** figures/pilot_noise_examples.png, artifacts/waveform_inventory.csv

Visual inspection of pilot_noise_examples.png reveals that "noise" windows contain
clear seismic arrivals in all 5 plotted examples. Noise windows were selected by
requiring a gap of >300 seconds with no catalogued event (EMSC catalog, M ≥ 2.0).
However, because the EMSC catalog is incomplete below M ≈ 2.0, the inter-event "quiet
periods" used for noise window placement still contain undetected smaller aftershocks.

**Impact:** The noise label applied to these windows is incorrect for many windows.
A false positive model output on a "noise" window may be a true positive detection of
an undetected sub-threshold aftershock. This inflates measured false positive rates.

**Quantification:** Not yet quantifiable (requires sub-threshold catalog, which is
unavailable). Assumed to affect a significant fraction of the 65 noise NPZ windows.

**Mitigation in Phase 3:** Before using noise windows in evaluation, apply a STA/LTA
scan (scripts/05_run_stalta.py) on noise windows and discard any with peak STA/LTA > 3.
This removes the most obviously contaminated windows and produces a cleaner noise class.

---

## L3 — No P/S Phase Picks in Catalog

**Source:** data/catalog/koeri_pilot_catalog.csv (p_pick_time, s_pick_time columns are empty)

The EMSC FDSN event service provides origin times and magnitudes only. P and S phase
pick times are not included in the EMSC FDSN response. All rows in koeri_pilot_catalog.csv
have empty p_pick_time and s_pick_time fields.

**Impact:** Phase picking mean absolute error (MAE) — a planned secondary metric — cannot
be computed against catalog picks. Alternative: compare model P/S picks against each other
or use the known origin time + station distance to estimate expected P arrival.

**Mitigation:** Theoretical P/S pick times can be computed using a 1D velocity model
(e.g., IASP91 or a Turkey-specific model) and origin time + hypocentral distance. This
approach introduces additional uncertainty (~0.2–0.5 s) from the velocity model, which
must be stated when reporting MAE.

---

## L4 — Network Code Correction (IJ → KO)

**Source:** artifacts/data_access_log.md (Section 2, Correction 1)

The project planning document (artifacts/planning.md) specified IJ network stations.
Phase 1 testing revealed IJ network returns HTTP 204 (no data) for all bounding boxes
in Turkey during the pilot period. All waveform data uses the KO network (Kandilli
Observatory). This does not affect scientific validity but deviates from the plan.

---

## L5 — No Fine-Tuning in Pilot Phase

**Source:** CLAUDE.md (Phase 3 definition)

Phase 3 runs PhaseNet, EQTransformer, and GPD in zero-shot (pre-trained weights, no
fine-tuning). Pre-training datasets for these models are primarily North American and
global earthquake catalogs. Turkish waveform characteristics (crust, attenuation,
instrument response) may differ from training distribution, causing systematic biases
in phase picks.

**Impact:** Zero-shot performance on Kahramanmaraş data may underestimate what is
achievable after Turkish-data fine-tuning. Phase 5 scale-up will consider fine-tuning
if zero-shot recall on M 2.0–3.0 events is below 0.5.

---

## L6 — Data Volume in Pilot

**Source:** artifacts/waveform_inventory.csv

The pilot dataset contains:
- 161 event NPZ windows (M ≥ 2.0), from 2 stations (KO.KOZT, KO.KMRS)
- 65 noise NPZ windows (label reliability compromised per L2)
- 226 total NPZ windows, 15.4 MB

This is a small dataset. Statistical conclusions drawn from Phase 4 evaluation apply
only to this pilot subset. Confidence intervals on precision/recall will be wide.
Phase 5 scale-up (5–8 stations, ~3 months) is required before drawing broader
conclusions about model performance.

---

## L8 — EQTransformer Zero-Shot Anomaly

**Source:** artifacts/detection_results.json, artifacts/methodology_notes.md (M5)

EQTransformer (zero-shot, original weights) achieved only 11–21% detection rate on
the Kahramanmaraş pilot dataset — substantially below PhaseNet (59–77%) and GPD
(62–77%) on the same windows. This is inconsistent with STEAD benchmark performance.
The probable cause is domain mismatch between STEAD (training) and Kahramanmaraş
waveforms (inference). Results involving EQTransformer in Phase 4 evaluation should
be interpreted as a lower bound on achievable performance; fine-tuning is recommended
before drawing conclusions about EQTransformer's suitability for Turkish waveforms.

---

## L9 — Phase 5 FDSN Parallel Client Initialization Failure (Critical)

**Source:** artifacts/phase5_gap_report.csv, data/catalog/phase5_download_progress.json

Phase 5 download (scripts/10_phase5_download.py) attempted 136,056 waveform requests
(11,338 EMSC events × 12 KO stations) using 8 parallel workers with thread-local
ObsPy FDSN clients. Of these, 130,869 (96.2%) failed with the error:
`"The current client does not have a dataselect service."`

Root cause: Thread-local Client initialization at `https://eida.koeri.boun.edu.tr`
competed under simultaneous connection load. When a thread's Client constructor
request to the FDSNWS service description endpoint is rate-limited or returned
incomplete JSON, ObsPy creates a client object without a dataselect service.
Subsequent `get_waveforms()` calls on that client then raise this error immediately
(not a network error — an ObsPy internal state error). The error is permanent for
that thread's client for the rest of the batch.

**Actual Phase 5 dataset:**
- Total downloads attempted: 136,056
- Successful downloads: 5,187 (3.8% success rate)
- Data volume: 0.15 GB
- Per-station successes: 83–493 windows (vs. ~11,338 attempted per station)
- Stations with 0 OK downloads (before replacement): KO.GAZ (0), KO.KOZT (0), KO.KMRS (1)
- Station replacements triggered: KO.GAZ → KO.GULA, KO.KMRS → KO.DYBB, KO.KOZT → KO.YESY

**Impact on Phase 5 analysis:** Phase 5 analysis is conducted on 5,187 event windows
(vs. 161 in pilot). This is a ~32x increase in data volume but is far below the
intended ~136k windows. The spatial and temporal coverage is non-uniform: each station
contributes ~400 windows, but these are not evenly distributed across events or months.

**Mitigation path:** Re-running with a single-client sequential or low-concurrency
(2 workers) approach would resolve the initialization collision. The server
demonstrably has data (proven by the 5,187 successes). This is deferred as it would
require ~19 hours of sequential download time. Phase 5 conclusions are limited to the
5,187 available windows.

---

## L10 — Detection Window Truncation at Far Stations (Phase 5)

**Source:** artifacts/phase5_station_list.csv, scripts/11_phase5_analyze.py

The 60-second event window (30 s before origin + 30 s after origin) was designed for
the Phase 2 pilot stations at epicentral distances ≤ 70 km (P travel time ≤ ~12 s).
Phase 5 includes 6 "far" stations at distances 161–241 km, where P travel times
(at ~6 km/s) reach 27–40 s after origin.

**Consequences:**
1. For stations beyond ~150 km, P may arrive near or after the 60-second window end.
   A P wave at 241 km arrives at ~40 s after origin = 70 s from window start, which
   falls outside the 60-second window entirely.
2. The detection criterion [origin_s − 5 s, origin_s + 25 s] (= [25 s, 55 s] in
   window coordinates) further restricts detection. For a station at 180 km,
   P arrives at ~30 s after origin = 60 s in the window, just barely inside,
   but outside the detection criterion's 25-second post-origin cutoff.

**Impact on Phase 5 recall:** Far-station event windows are systematically missed
by all models regardless of waveform quality, because the P arrival itself is
truncated. This inflates the FN count and deflates recall relative to what the
models could achieve with correctly windowed data.

**Evidence:** Recall is broadly flat across magnitude bands (M 2–3 ≈ M 3–4 ≈ M≥4
for all models), inconsistent with SNR-driven performance (higher-M events should be
detected more easily). The flat profile suggests a distance-dependent structural miss
rate rather than signal amplitude limitation.

**Recommended fix:** For Phase 5 re-analysis, use windows of 120 s (30 s before +
90 s after origin) and extend DETECT_POST to 50 s to cover all distances ≤ 300 km.
This is a future refinement; current Phase 5 metrics are reported as-is.

---

## L7 — Geographic Generalization

**Source:** artifacts/planning.md (Section 4, Scope Boundaries)

The pilot model evaluation is conducted on Kahramanmaraş region data (SE Turkey,
~37–38°N). The deployment target is Western Marmara (~40–41°N). These regions have
different crustal velocity structures, attenuation characteristics, and station
coverages. Model performance on Western Marmara data has not yet been evaluated.
Phase 5 will address this explicitly using KO.RKY + KO.KRBG stations.
