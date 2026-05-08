# Seismic AI — Key Results

**Project:** Automatic Micro-Seismic Detection & TBDY-2018 Decision Support
**Team:** Bertuğ Taş, Kadir Emir Yücel, Melih Takyaci, Emre Özdemir, Efendi Nasiboğlu
**Institution:** Dokuz Eylül University, Department of Computer Science
**Event:** III. National Basic Sciences Youth Symposium and Science Fair, 12–13 May 2026

> All metrics are computed on real KOERI waveform data. Phase picks referenced to EMSC analyst picks (not TauPy theoretical). Scientific framing: **automatic seismic event detection**, not earthquake prediction.

---

## 1. Detection Model Comparison (M ≥ 2.0)

Test set: **944 event windows** from 12 KO/IJ network stations, 6-month Kahramanmaraş aftershock sequence.
Labels: EMSC analyst P-phase picks (re-labelled via script 19, TauPy fallback for unmatched).

| Model | Recall (All) | Recall M2–3 | Recall M3–4 | Recall M≥4 | P-MAE (s) |
|-------|:------------:|:-----------:|:-----------:|:----------:|:---------:|
| STA/LTA (baseline) | 0.762 | 0.733 | 0.874 | 1.000 | 3.78 |
| PhaseNet zero-shot | 0.694 | 0.671 | 0.797 | 0.815 | 2.23 |
| EQTransformer zero-shot | 0.176 | 0.176 | 0.168 | 0.222 | 1.50 |
| GPD zero-shot | 0.892 | 0.876 | 0.965 | 0.963 | 4.12 |
| PhaseNet (fine-tuned†) | 0.086 | 0.090 | 0.063 | 0.074 | 7.16 |
| GPD (fine-tuned) | **1.000** | **1.000** | **1.000** | **1.000** | 4.78 |
| PhaseNet Fixed (focal loss) | 0.910 | — | — | — | — |
| **Ensemble (GPD+PhaseNet)** | **1.000** | **1.000** | **1.000** | **1.000** | **4.81** |

† PhaseNet fine-tuned (script 20) degraded due to loss function imbalance (noise→1 collapse). PhaseNet Fixed (script 30, focal loss) is the corrected variant; used in the ensemble.

---

## 2. Sub-Threshold Detection (M < 2.0)

Test set: **1,036 event windows** from ISC + EMSC + AFAD combined catalog (44,214 deduplicated events).
Stations: KO.KOZT and KO.KMRS. Detection window: [origin − 5 s, origin + 25 s].

| Magnitude Band | N | GPD (fine-tuned) | PhaseNet Fixed | Ensemble |
|----------------|---|:----------------:|:--------------:|:--------:|
| M 0.5 – 1.0 | 353 | 0.9887 | 0.9433 | **1.0000** |
| M 1.0 – 1.5 | 343 | 0.9913 | 0.9504 | **1.0000** |
| M 1.5 – 2.0 | 340 | 0.9912 | 0.9706 | **0.9971** |

---

## 3. Geographic Generalization (Marmara Region)

Evaluation on **5,838 event windows** from 28 KO network stations in Western Marmara.
Models trained exclusively on Kahramanmaraş data — zero Marmara events in training set.

| Model | Marmara Recall | Kahramanmaraş Recall | Δ |
|-------|:--------------:|:--------------------:|:-:|
| GPD (fine-tuned) | 0.9954 | 1.0000 | −0.005 |
| PhaseNet Fixed | 0.7859 | 0.9100 | −0.124 |
| **Ensemble** | **0.9991** | **1.0000** | **−0.001** |

By magnitude band (Marmara):

| Band | N | GPD-FT | Ensemble |
|------|---|:------:|:--------:|
| M 0.0 – 1.5 | 101 | 1.0000 | 1.0000 |
| M 1.5 – 2.0 | 3,638 | 0.9940 | 0.9992 |
| M 2.0 – 3.0 | 1,812 | 0.9974 | 0.9990 |
| M 3.0+ | 287 | — | — |

---

## 4. Phase Picking Accuracy

P-phase picking MAE referenced to EMSC analyst picks.

| Subset | N | GPD-FT Coarse | After AIC Refinement | After Best-of-3 |
|--------|---|:-------------:|:--------------------:|:---------------:|
| EMSC-matched (full) | 740 | 0.700 s (median) | 0.650 s | 0.400 s |
| Conditional (coarse < 1.0 s) | 445 | 0.280 s | — | — |
| All test windows | 944 | 4.779 s (mean) | — | — |

> Target P-MAE < 0.3 s: **met** on EMSC-matched subset with coarse error < 1 s (445/740 windows). Not met at population level due to ~295/740 windows where coarse pick error > 1 s.

S-phase (PhaseNet Fixed vs EMSC analyst S-picks, 163 matched windows):
- S-MAE median: **1.740 s** (S-optimized model) vs 3.220 s (unoptimized PhaseNet Fixed)

---

## 5. Zero-Leakage Validation

Split: 1,761 events → 1,232 train / 264 val / 265 test (70/15/15 by event ID, zero window-level leakage).

| Model | Clean Split Recall | Windowed Split Recall | Δ |
|-------|:-----------------:|:--------------------:|:-:|
| GPD (fine-tuned) | 0.9989 | 1.0000 | −0.001 |
| Ensemble | 1.0000 | 1.000 | 0.000 |

**Generalization confirmed** — negligible degradation from windowed to clean split.

---

## 6. PSHA Module

### 6a. Gutenberg-Richter Analysis (Kahramanmaraş Region)

| Parameter | Value |
|-----------|-------|
| Catalog | Combined (ISC + EMSC + AFAD), 2023-02-06 to 2023-08-05 |
| N events | 42,078 |
| Observation period | 0.493 years |
| Mc (MAXC + 0.2 correction) | 2.0 |
| b-value (MLE, Aki 1965) | 0.758 ± 0.007 |
| a-value | 5.987 |
| Note | Aftershock-dominated; annual rates inflated ~50× vs background |

### 6b. 3-Zone Regional Calibration (ISC 1990–2023)

| Zone | b-value | σ | N events | Mc |
|------|:-------:|:-:|:--------:|:--:|
| NAF (North Anatolian Fault) | 1.158 | ±0.016 | 11,987 | 3.15 |
| EAF (East Anatolian Fault) | 0.996 | ±0.016 | 7,558 | 3.15 |
| Central Anatolia | 1.038 | ±0.009 | 26,709 | 3.15 |

- Total ISC catalog: 46,254 events (M≥3.0, Turkey, 1990–2023), aftershock period excluded
- Spatial grid: 338 cells (0.5°×0.5°), Omori-Utsu 10× correction on 43 aftershock-zone cells

---

## 7. Earthquake Impact Simulator

Calibration against historical events (Boore-Atkinson 2008 GMPE, HAZUS MH-MR5 fragility):

| Event | M | Actual Fatalities | Predicted | Ratio |
|-------|---|:-----------------:|:---------:|:-----:|
| 1999 Kocaeli | 7.6 | 17,480 | 17,672 | **1.01×** ✓ |
| 2020 İzmir | 6.9 | 114 | 86 | 0.75× |
| 2023 Kahramanmaraş | 7.8 | ~50,000 | — | — |

> Impact simulator uses province-centroid spatial averaging — underestimates near-fault exposure for Istanbul scenarios (see `artifacts/limitations.md` L-11).

---

## 8. Vs30 Grid

- Method: TBDY-2018 anchor values (42 Turkish cities), IDW interpolation (1/d²)
- Coverage: 1,825 cells at 0.25°×0.25° resolution across Turkey
- Soil classification: ZB / ZC / ZD / ZE per TBDY-2018 NEHRP equivalent
- File: `artifacts/vs30_grid.csv`

---

## Reproducibility

All metrics above are computed by scripts in `scripts/` and saved to `artifacts/`. No results are hard-coded in the application layer — `app.py` reads directly from the JSON/CSV artifacts at runtime.

Key artifact files:
- `artifacts/final_evaluation_metrics.json` — Table 1 (M≥2.0 detection)
- `artifacts/small_event_evaluation.json` — Table 2 (M<2.0 detection)
- `artifacts/marmara_extended_results.json` — Table 3 (geographic generalization)
- `artifacts/phase_picking_refined.json` — Table 4 (P-MAE refinement)
- `artifacts/clean_event_split.json` — Table 5 (zero-leakage validation)
- `artifacts/gutenberg_richter_params.json` — Table 6a
- `artifacts/zone_calibration.json` — Table 6b
- `artifacts/hazard_calculator_validation.json` — Table 7 (impact calibration)
- `artifacts/vs30_grid.csv` — Table 8
