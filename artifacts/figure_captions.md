# Figure Captions — Seismic AI Pipeline
# Phase 5 (2026-04-25)
# All figures reference computed metrics from this project's own artifacts.


## Phase 5 Dataset Note

Phase 5 analysis uses 11280 preprocessed event windows from 12 KO HH stations over 6 months (2023-02-06 to 2023-08-06). Download success rate was 3.8% (5,187 of 136,056 attempted requests) due to FDSN parallel client initialization collision — see limitations.md L9. No noise windows included (noise class omitted in Phase 5 per design).

---

## figures/pilot_event_examples.png

Representative event waveforms from the Kahramanmaraş 2023 aftershock pilot dataset.
Five events shown, spanning M 2.7–5.0, recorded at KO.KOZT and KO.KMRS (KO network,
HH broadband channels, 100 Hz). Each panel shows one 60-second window after
preprocessing (linear detrend, bandpass 1–45 Hz, z-score normalization). Three
components shown: HHZ (blue), HHN (orange), HHE (green). Red dashed line marks catalog
origin time at t=30 s. P and S arrivals are clearly visible after t=30 s.
Source: artifacts/waveform_inventory.csv.

## figures/pilot_noise_examples.png

"Noise" windows from the pilot dataset, selected from inter-event gaps > 300 s in the
EMSC catalog. Visual inspection reveals clear seismic arrivals in all displayed windows,
confirming contamination by sub-threshold aftershocks undetected by the EMSC catalog.
This is documented as Limitation L2 in artifacts/limitations.md. These windows should
be treated as having uncertain class labels.

## figures/waveform_with_picks_examples.png

Five event waveforms (M 5.0–6.0) with P and S pick estimates overlaid from three
zero-shot SeisBench models. Red dashed/dotted: PhaseNet P/S; blue dashed/dotted:
EQTransformer P/S; green dashed: GPD P. Black solid: catalog origin time (t=30 s).
Picks plotted on HHZ (vertical) only. PhaseNet and GPD produce picks in the physically
expected window (after origin); EQTransformer produces fewer picks due to zero-shot
domain mismatch (see methodology_notes.md M5 and limitations.md L8).

## figures/stalta_vs_dl_comparison.png

Detection rate (%) by magnitude band for four detectors (STA/LTA, PhaseNet,
EQTransformer, GPD) run in zero-shot mode on the Kahramanmaraş pilot dataset
(KO.KOZT + KO.KMRS, 2023-02-06 to 2023-02-20). Detection criterion: P pick or
STA/LTA trigger in [origin − 5 s, origin + 25 s]. Dashed line at 50%.
STA/LTA achieves highest raw detection rate but cannot discriminate events from noise.
PhaseNet and GPD reach 77% detection on M≥4 events in zero-shot mode.
Source: artifacts/detection_results.json.

## figures/confusion_matrix.png

Recall summary bar chart for each detector (STA/LTA, PhaseNet, EQTransformer, GPD) evaluated on 11280 event windows from Phase 5 (12 KO stations, 6 months). No noise windows collected in Phase 5 (see limitations.md L9). GPD achieves highest recall (0.593), followed by STA/LTA (0.484), PhaseNet (0.319), and EQTransformer (0.051). Recall values are lower than Phase 4 pilot — partly attributable to far-station detection window mismatch (see limitations.md L10).
Source: artifacts/evaluation_metrics.json.

## figures/precision_recall_by_magnitude.png

Recall by magnitude band (M 2–3, M 3–4, M≥4) for all four detectors in Phase 5. No precision metric computed (no noise windows). Event counts per band: M 2–3 n=9254, M 3–4 n=1757, M≥4 n=269. Recall is broadly flat across magnitude bands for all models, suggesting that distance-dependent detection window truncation (see limitations.md L10) rather than SNR dominates the miss rate.
Source: artifacts/evaluation_metrics.json.

## figures/station_map.png

Map of 12 primary and 3 alternate KO (Kandilli Observatory) HH broadband stations selected for Phase 5 scale-up. Stations stratified by epicentral distance from M7.8 main shock (37.166°N, 37.032°E): near <50 km (2 stations), mid 50–150 km (4 stations), far 150–300 km (6 stations). Orange star marks M7.8 main shock epicenter. Three primary stations (KO.GAZ, KO.KMRS, KO.KOZT) had zero successful downloads and were replaced by alternates. Source: artifacts/phase5_station_list.csv.

## figures/event_map.png

Spatial distribution of 11338 EMSC-catalogued aftershocks in the Kahramanmaraş
region from 2023-02-06 to 2023-08-06 (6-month Phase 5 window).
Symbol size and color scale with magnitude (colorbar). Blue triangles mark the 12 primary Phase 5 stations.
The aftershock cloud spans roughly 36.5–38.5°N, 35.5–38.5°E, concentrated along the
East Anatolian Fault zone. Omori decay is visible in temporal density (not shown here).
Source: data/catalog/phase5_catalog.csv.
