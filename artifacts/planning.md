# Seismic AI Project — Planning Document
# Phase 0
# Authors: Bertuğ Taş, Kadir Emir Yücel, Melih Takyaci, Emre Özdemir, Efendi Nasiboğlu
# Institution: Dokuz Eylül University, Department of Computer Science, İzmir, Turkey
# Event: III. National Basic Sciences Youth Symposium and Science Fair, 12–13 May 2026
# Date: 2026-04-24

---

## 1. Project Restatement

This project develops a two-layer pipeline for **automatic seismic event detection** in
the Western Marmara region of Turkey, with a focus on sub-threshold micro-seismic events
(M < 2.0) that fall below the completeness threshold of the national catalog
(approximately Mc ≈ 2.7).

**What this project does:**
- Fine-tunes and compares three pre-trained deep learning seismic phase detection models
  (PhaseNet, EQTransformer, GPD) via the SeisBench framework on real KOERI/IJ network
  waveform data from the 2023 Kahramanmaraş aftershock sequence.
- Evaluates each model's ability to produce reliable preliminary P/S phase picks on
  catalogued micro-seismic events, using recall on M < 2.0 events as the primary metric.
- Builds a grounded LLM question-answering layer (Phase 6 only) over verified internal
  project artifacts. The LLM does not fabricate results; it answers only from indexed
  internal files.

**What this project does NOT do:**
- Does NOT predict or forecast future earthquakes.
- Does NOT claim to replace AFAD or KOERI operational detection systems.
- Does NOT use unverified third-party data mirrors.
- Does NOT apply to regions outside Western Marmara without re-validation.

**Core scientific question:**
Can a pre-trained deep learning seismic model detect catalogued micro-seismic events
(M < 2.0) in KOERI Turkey data, and produce reliable preliminary P/S phase picks?

---

## 2. Source-by-Source Data Understanding

### 2.1 STEAD Dataset
- **Role:** Supplementary training/pre-training reference only. Not the primary waveform source.
- **Usage:** Local earthquake + noise subset ONLY. The full dataset (85 GB) is EXCLUDED.
- **URL:** https://github.com/smousavi05/STEAD
- **Access method:** Direct GitHub download of subset HDF5 files.
- **Limitation:** STEAD is a global catalog. Turkish-specific waveform characteristics
  may differ from STEAD statistics. We use STEAD only to verify SeisBench model behavior
  before applying to KOERI data.

### 2.2 KOERI Earthquake Catalog
- **Role:** Ground truth event labels — origin times, latitude, longitude, depth, magnitude.
  Used to define waveform windows and evaluate model picks against catalog picks.
- **Primary URL (web search):** https://www.koeri.boun.edu.tr/sismo/zeqdb/indexeng.asp
- **Primary URL (data download):** https://www.koeri.boun.edu.tr/sismo/2/deprem-verileri/sayisal-veriler/
- **Access method:** FDSN event service (ObsPy `Client("KOERI").get_events()`),
  or direct CSV/QuakeML download from the web interface.
- **Geographic filter:** Western Marmara bounding box (see Section 4).
- **Temporal filter:** Pilot: 2 weeks from 2023 Kahramanmaraş aftershock sequence onset.

### 2.3 KOERI-EIDA Waveform Data (miniSEED via FDSN)
- **Role:** Primary continuous waveform source for all model training and evaluation.
- **Dataselect URL:** https://eida.koeri.boun.edu.tr/fdsnws/dataselect/1/
- **Station metadata URL:** https://eida.koeri.boun.edu.tr/fdsnws/station/1/
- **Availability URL:** https://eida.koeri.boun.edu.tr/fdsnws/availability/1/
- **Web interface:** https://eida.koeri.boun.edu.tr/webinterface/
- **FDSN registry:** https://www.fdsn.org/datacenters/detail/KOERI/
- **Access method:** ObsPy `Client("EIDA")` or direct FDSN HTTP requests.
  Targeted window downloads only — never full continuous pulls in pilot.
- **Format:** miniSEED (3-component: HHZ, HHN, HHE or BHZ/BHN/BHE).
- **Authentication:** Some KOERI channels require EIDA token. We will test open access
  first and document any authentication requirements in data_access_log.md.

### 2.4 KOERI / IJ Network Station Metadata
- **Role:** Station selection, coordinate lookup, channel code identification for
  Western Marmara pilot stations.
- **IJ network URL:** https://www.fdsn.org/networks/detail/IJ/
- **KOERI station list URL:** https://www.koeri.boun.edu.tr/sismo/2/sismik-ag-listeleri/
- **Access method:** ObsPy `Client("EIDA").get_stations()` with network="IJ".
- **Usage in pilot:** Select exactly 2 stations within the Western Marmara bounding box
  with good SNR history and 3-component availability.

### 2.5 AFAD Networks (Fallback Only)
- **Role:** Fallback if KOERI waveform coverage is insufficient for the target area.
- **TU network:** https://www.fdsn.org/networks/detail/TU/
- **TK network:** https://www.fdsn.org/networks/detail/TK/
- **Trigger for use:** Only if KOERI/IJ stations do not provide 3-component data with
  sufficient SNR in the Western Marmara bounding box.
- **Tradeoff if used:** Different instrument responses, different metadata conventions,
  potential gap in pick-catalog alignment methodology.

### 2.6 Explicitly Excluded Sources
- Temporary aftershock networks (YA, YB) — not included in pilot or main phase.
- Any unverified third-party mirrors of KOERI or AFAD data.
- Full STEAD dataset (85 GB) — too large, and Turkey-specific tuning is the goal.

---

## 3. Source Accessibility and Access Method

| Source | Access Type | Authentication Required | ObsPy Compatible |
|--------|-------------|------------------------|------------------|
| KOERI Catalog | FDSN Event API + Web CSV | None (public) | Yes |
| KOERI-EIDA Waveforms | FDSN Dataselect API | Some channels need EIDA token | Yes |
| KOERI-EIDA Station Metadata | FDSN Station API | None | Yes |
| IJ Network Info | FDSN Registry (static) | None | Yes |
| STEAD Subset | GitHub direct download | None | No (HDF5 custom) |
| AFAD TU/TK (fallback) | FDSN Event + Dataselect | Varies | Yes |

Authentication risk: EIDA token registration may introduce a delay. Phase 1 will
test open-access channels first and document the outcome.

---

## 4. Scope Boundaries

### Geographic Scope (Fixed)
- **Region:** Western Marmara, Turkey
- **Bounding box (approximate):**
  - Latitude: 39.5°N – 41.5°N
  - Longitude: 26.0°E – 29.5°E
  - Covers: Istanbul – Tekirdağ – Bursa – Çanakkale corridor
- **Why Western Marmara:** High seismic hazard, dense IJ network coverage, TBDY-2018 soil
  classification data available, strategic relevance for Istanbul risk assessment.

### Temporal Scope (Fixed)
- **Primary dataset:** 2023 Kahramanmaraş aftershock sequence (starting February 6, 2023)
- **Pilot window:** 2 weeks of data from 2 IJ stations (to be confirmed in Phase 1)
- **Main phase window:** ~3 months of data, 5–8 stations (Phase 5 only, after pilot)

### Magnitude Scope
- **Primary target:** M < 2.0 micro-seismic events (below national catalog completeness)
- **Evaluation bands:** M < 1.0, 1.0–2.0, 2.0–3.0, M > 3.0

### What Is Explicitly Outside Scope (Current Phase)
- Eastern Marmara or other Turkish seismic zones
- Fine-tuning models beyond pilot validation (Phase 3 baseline only)
- LLM integration (Phase 6 only)
- AFAD as primary source
- Any event outside the 2023 Kahramanmaraş aftershock sequence for training data

---

## 5. Pilot Acquisition Plan

### Pilot Constraints
- Exactly **2 IJ network stations** with 3-component coverage in Western Marmara
- Exactly **2 weeks of event-targeted waveform windows**
- Catalog events from KOERI for the selected 2-week window
- Window strategy: 60-second windows per event (30 s before to 30 s after origin time),
  plus paired noise windows of equal length and count from quiet periods

### What Will NOT Be Acquired in Pilot
- Waveforms from more than 2 stations
- More than 2 weeks of waveform data
- Full continuous miniSEED streams (targeted windows only)
- Any data from AFAD networks (unless KOERI fails completely)
- STEAD data (only used if SeisBench baseline needs verification)

### Data Volume Estimate for Pilot
- Approx. 50–200 catalogued events in 2 weeks at 2 stations
- At 60 s window, 3 components, 100 Hz, 4 bytes/sample:
  - Per event per station: 60 × 3 × 100 × 4 = 72,000 bytes ≈ 70 KB
  - 200 events × 2 stations × 2 (event + noise) = ~56 MB
- Well within the 20 GB hard limit. This is approximately 0.003% of the hard limit.

---

## 6. Processing Strategy

### Preprocessing (Phase 2)
1. Load raw miniSEED via ObsPy
2. Merge traces, fill gaps with zeros if < 5% of window
3. Detrend (linear)
4. Bandpass filter: 1–45 Hz (4th order Butterworth, zero-phase)
5. Resample to 100 Hz (SeisBench standard)
6. Normalize: per-trace z-score normalization
7. Output: fixed-length 60-second 3-component arrays (6000 samples per component)
8. Store as NPZ files with metadata (event ID, station, origin time, catalog P/S picks)

### Phase Picking Reference
- Catalog P and S picks from KOERI will be used as ground truth references.
- If catalog picks are unavailable for a specific event, that event will be labeled
  with origin time only and excluded from pick-accuracy evaluation (but included in
  detection evaluation).

---

## 7. Repository Structure

```
seismic-ai-project/
├── data/
│   ├── raw/              # Raw miniSEED files (temporary, not committed)
│   ├── windows/          # Preprocessed 60-s NPZ windows (event + noise)
│   ├── catalog/          # KOERI catalog CSV files
│   └── metadata/         # Station XML / inventory files
├── notebooks/            # Exploratory notebooks (not primary pipeline)
├── scripts/
│   ├── 01_check_access.py
│   ├── 02_fetch_catalog.py
│   ├── 03_download_waveforms.py
│   ├── 04_preprocess.py
│   ├── 05_run_stalta.py
│   ├── 06_run_seisbench.py
│   ├── 07_evaluate.py
│   └── 08_plot_artifacts.py
├── artifacts/
│   ├── planning.md              ← THIS FILE
│   ├── data_access_log.md
│   ├── station_summary.csv
│   ├── event_catalog.csv
│   ├── waveform_inventory.csv
│   ├── detection_results.json
│   ├── evaluation_metrics.json
│   ├── methodology_notes.md
│   ├── figure_captions.md
│   └── limitations.md
├── figures/
├── llm/
│   ├── index/
│   └── interface.py
├── requirements.txt
├── README.md
└── ASSUMPTIONS.md
```

All artifact filenames are fixed. The Phase 6 RAG layer indexes them by filename.

---

## 8. Initial Methods

### Layer 1 — Seismic Detection Pipeline

**Classical baseline (Phase 3):**
- STA/LTA detector with short-term window 0.5 s, long-term window 10 s, threshold 3.0
- Applied to the vertical (Z) component only
- Purpose: establish a non-ML lower bound for detection performance

**Deep learning models (Phase 3):**
- **PhaseNet** (Zhu & Beroza, 2019): U-Net style, simultaneous P/S picking
- **EQTransformer** (Mousavi et al., 2020): Transformer-based, detection + P/S picking
- **GPD** (Ross et al., 2018): CNN-based P/S/noise classification
- All loaded via SeisBench (`seisbench.models`) with pre-trained weights
- Initial run: zero-shot inference (no fine-tuning) on pilot data
- If zero-shot recall on M < 2.0 is below 0.5, consider fine-tuning in Phase 5

**SeisBench framework version:** to be confirmed in Phase 1 environment setup.

### Layer 2 — Grounded LLM Q&A (Phase 6 only)
- RAG architecture over indexed artifacts/ directory
- LLM answers cite specific artifact file and passage
- LLM refuses questions outside the internal artifact scope
- Stack: LlamaIndex or LangChain (to be confirmed), Claude API (grounded, not generative)
- No LLM implementation until Phase 6. Not relevant to Phases 0–5.

---

## 9. Evaluation Plan

### Primary Metric
- **Recall on M < 2.0 events** — the fraction of catalogued micro-seismic events that
  the model detects within ±5 s of the catalog origin time.

### Secondary Metrics
- Precision: fraction of model detections that correspond to a catalogued event
- F1 score: harmonic mean of precision and recall
- False positives per hour: operational stability metric
- Phase picking MAE: mean absolute error between model P-pick and catalog P-pick (seconds)
  Target: approaching 0.1 s
- Phase picking MAE for S separately

### Stratification
Results will be stratified by magnitude band:
- M < 1.0
- 1.0 ≤ M < 2.0
- 2.0 ≤ M < 3.0
- M ≥ 3.0

### Model Comparison
| Model | Detection | P-pick | S-pick | Notes |
|-------|-----------|--------|--------|-------|
| STA/LTA | P/R/F1 | N/A | N/A | Classical baseline |
| PhaseNet | P/R/F1 | MAE | MAE | Pre-trained, zero-shot |
| EQTransformer | P/R/F1 | MAE | MAE | Pre-trained, zero-shot |
| GPD | P/R/F1 | MAE | MAE | Pre-trained, zero-shot |

### Success Criteria
- At least one DL model achieves recall ≥ 0.5 on M < 2.0 events in zero-shot mode
- Phase picking MAE approaches 0.1 s on M ≥ 2.0 events (catalog picks available)
- All metrics are computed from this project's own artifacts — no borrowed numbers

---

## 10. Future LLM Integration (Phase 6)

The LLM layer will be implemented only after the detection pipeline is fully evaluated
and all artifacts are produced. Key design constraints:

- The LLM answers ONLY from indexed internal artifacts (event_catalog.csv,
  detection_results.json, evaluation_metrics.json, methodology_notes.md,
  limitations.md, figure_captions.md, TBDY-2018 provisions, soil data).
- Every answer must cite the specific artifact file and passage.
- The LLM refuses questions it cannot answer from internal artifacts.
- The interface is a Streamlit app with 3 tabs:
  - Tab 1: Interactive event map (Marmara)
  - Tab 2: Waveform viewer with P/S picks
  - Tab 3: LLM Q&A over internal artifacts only

Language model choice, embedding model, and vector store will be decided in Phase 6.
The LLM does not perform seismic analysis — it only retrieves and presents results
that were already computed by the detection pipeline.

---

## 11. Risks and Controls

### Scientific Risks

| Risk | Likelihood | Impact | Control |
|------|-----------|--------|---------|
| Low recall on M < 2.0 in zero-shot mode | High | High | Document clearly; fine-tune in Phase 5 if needed |
| Catalog incompleteness (M < 2.0 events missing) | Medium | High | Treat as lower bound on recall; note in limitations.md |
| Waveform SNR too low for micro-events at 2 pilot stations | Medium | Medium | Select stations with documented low-noise floors |
| P/S picks unavailable in KOERI catalog for small events | High | Medium | Exclude from pick MAE; include in detection evaluation only |
| STA/LTA baseline performs comparably to DL (null result) | Low | Low | Still a publishable negative result; document thoroughly |
| Instrument response removal artifacts | Low | Medium | Apply only if needed; document in methodology_notes.md |

### Engineering Risks

| Risk | Likelihood | Impact | Control |
|------|-----------|--------|---------|
| EIDA authentication required for waveform access | Medium | High | Test open access in Phase 1; register EIDA token as fallback |
| FDSN endpoint downtime | Low | Medium | Cache all fetched data locally; retry with timeout |
| ObsPy / SeisBench version conflicts | Medium | Medium | Pin all dependencies in requirements.txt |
| miniSEED gaps in waveform windows | Medium | Low | Document gap fraction; discard windows with > 5% gaps |
| KOERI catalog coverage gaps for pilot period | Low | Medium | Extend temporal window or switch to a week with higher activity |
| Storage overflow on shared machine | Low | Low | Pilot data volume is ~56 MB; monitor disk usage in scripts |
| SeisBench model hub download failure | Low | Medium | Cache model weights locally after first download |

---

## 12. Recommended First Action

**Phase 1 — Data Access Validation:**

1. Install required packages (ObsPy, SeisBench, pandas, matplotlib) in a clean
   Python 3.10+ environment. Pin versions in requirements.txt immediately.
2. Write and run `scripts/01_check_access.py` to test all FDSN endpoints
   (KOERI catalog, EIDA waveform, EIDA station metadata).
3. Write and run `scripts/02_fetch_catalog.py` to download KOERI catalog for
   Western Marmara over the pilot period.
4. Identify 2 candidate IJ network stations with 3-component coverage and low noise.
5. Document every result — success, failure, and fallback — in `artifacts/data_access_log.md`.

**Do not download any waveforms until Phase 1 access validation is complete.**
**Do not begin preprocessing, modeling, or evaluation until Phase 2.**
