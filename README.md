# Seismic AI — Automatic Micro-Seismic Detection & TBDY-2018 Decision Support System

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![SeisBench](https://img.shields.io/badge/SeisBench-latest-green)](https://github.com/seisbench/seisbench)
[![Streamlit](https://img.shields.io/badge/Streamlit-live-red)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

**III. National Basic Sciences Youth Symposium and Science Fair**
Dokuz Eylül University, Department of Computer Science, İzmir, Turkey
12–13 May 2026

**Team:** Bertuğ Taş, Kadir Emir Yücel, Melih Takyaci, Emre Özdemir, Efendi Nasiboğlu

---

## Scientific Framing

This project performs **automatic seismic event detection** and **probabilistic seismic hazard analysis** — **NOT earthquake prediction**. All outputs are based on past seismicity data and standard statistical methods (Gutenberg-Richter, FEMA HAZUS, Boore-Atkinson 2008 GMPE).

---

## Project Overview

A two-layer architecture for micro-seismic pattern detection and construction decision support:

### Layer 1 — Seismic Detection Pipeline

- Real KOERI waveform data (KO/IJ network, miniSEED via FDSN)
- 2023 Kahramanmaraş aftershock sequence (11,338 events, M≥2.0)
- 44,214 sub-threshold events (M 0.5–2.0) from ISC + EMSC + AFAD catalogs
- Models: **PhaseNet**, **EQTransformer**, **GPD** via SeisBench + classical STA/LTA baseline
- Fine-tuning on 39,456 augmented KOERI windows with EMSC analyst phase picks
- Domain adaptation to Western Marmara region via pseudo-labeling

### Layer 2 — Decision Support & Analysis

- **TBDY-2018** (Turkish Building Earthquake Code) rule engine with soil classification
- **PSHA** — 3-zone Gutenberg-Richter analysis (NAF / EAF / Central Anatolia, ISC 1990–2023)
- **Vs30 grid** — TBDY-2018 spectral values for 81 Turkish provinces
- **Earthquake impact simulator** — FEMA HAZUS adapted for Turkey, calibrated against 1999 Kocaeli and 2023 Kahramanmaraş
- **RAG Q&A system** — grounded LLM answers over internal project artifacts only

---

## Key Results

| Model | Recall (M≥2.0) | Recall (M<2.0) | Marmara | P-MAE |
|-------|---------------|---------------|---------|-------|
| STA/LTA (baseline) | 0.762 | — | — | N/A |
| PhaseNet zero-shot | 0.694 | — | — | 2.23s |
| EQTransformer zero-shot | 0.176 | — | — | 1.50s |
| GPD zero-shot | 0.892 | — | — | 4.12s |
| EQTransformer fine-tuned | 0.642 | — | — | — |
| PhaseNet Fixed (focal loss) | 0.910 | 0.943–0.971 | 0.786 | — |
| GPD fine-tuned | 1.000 | 0.989–0.991 | 0.995 | 4.78s |
| **Ensemble (GPD+PhaseNet)** | **1.000** | **0.997–1.000** | **0.999** | **4.81s** |

> Phase picking references: EMSC analyst picks (not TauPy theoretical). P-MAE median on EMSC-matched windows.

---

## Live Demo

The Streamlit app has 4 tabs:

| Tab | Content |
|-----|---------|
| 🗺️ **Olay Haritası** | Interactive map of 11,338 detected events + PSHA hazard overlay (P(M≥6, 50yr)) |
| 〰️ **Dalga Formu Görüntüleyici** | Waveform viewer with 60-frame animated P/S detection simulation |
| 🏗️ **TBDY-2018 Risk Danışmanı** | Soil class (Vs30) + DTS + PSHA probability + construction code provisions |
| 🔴 **Etki Senaryosu** | ShakeMap + casualty estimate + economic loss (province-level, HAZUS) |

### Run locally

```bash
git clone https://github.com/bertugtas/seismic-ai-project
cd seismic-ai-project
pip install -r requirements.txt
streamlit run app.py
```

> **Note:** Large data files (miniSEED waveforms, HDF5 datasets, model weights) are not included in this repository. See [Data & Models](#data--models) section below.

---

## Repository Structure

```
seismic-ai-project/
├── app.py                          # Streamlit app (4 tabs)
├── scripts/
│   ├── 01_check_access.py          # FDSN endpoint validation
│   ├── 02_fetch_catalog.py         # KOERI/EMSC catalog download
│   ├── 03_download_waveforms.py    # Pilot waveform download
│   ├── 04_preprocess.py            # Detrend, bandpass, normalize
│   ├── 05_run_stalta.py            # STA/LTA baseline
│   ├── 06_run_seisbench.py         # Zero-shot PhaseNet/EQT/GPD
│   ├── 07_evaluate.py              # Precision, recall, F1, MAE
│   ├── 08_plot_artifacts.py        # All figures
│   ├── 10_phase5_download.py       # Scale-up download (12 stations, 6 months)
│   ├── 11_phase5_analyze.py        # Batch preprocessing
│   ├── 12_prepare_finetune_dataset.py  # HDF5 + augmentation
│   ├── 16_finetune_gpd.py          # GPD fine-tuning (SeisBench)
│   ├── 17_ensemble_evaluation.py   # PhaseNet+GPD ensemble
│   ├── 18_fetch_koeri_picks.py     # EMSC analyst phase picks
│   ├── 19_relabel_picks.py         # Re-label with EMSC picks
│   ├── 20_final_finetune.py        # Final training (EMSC labels)
│   ├── 21_final_evaluation.py      # Full evaluation suite
│   ├── 28–31_*.py                  # Sub-threshold M<2.0 evaluation
│   ├── 32_clean_event_split.py     # Zero-leakage train/val/test split
│   ├── 33–36_*.py                  # Optimization round 2
│   ├── 37_regional_risk_assessment.py  # TBDY-2018 rule engine
│   ├── 38–41_*.py                  # Final optimization round
│   ├── 42_gutenberg_richter.py     # G-R analysis (b-value, Mc)
│   ├── 43_hazard_calculator.py     # PSHA probability calculator
│   ├── 44_fetch_vs30.py            # TBDY-2018 Vs30 grid
│   ├── 45_regional_psha_calibration.py  # 3-zone ISC calibration
│   ├── 46_download_impact_data.py  # Impact scenario datasets
│   └── 47_impact_calculator.py     # FEMA HAZUS impact engine
├── artifacts/
│   ├── planning.md
│   ├── final_evaluation_metrics.json
│   ├── gutenberg_richter_params.json
│   ├── spatial_hazard_grid_v2.csv  # 338 cells, ISC 1990-2023
│   ├── vs30_grid.csv               # 1825 cells, TBDY-2018 IDW
│   ├── zone_calibration.json       # 3-zone b-values
│   └── ...
├── figures/                        # All publication-ready figures
├── llm/
│   ├── index/chunks.json           # RAG index (15 chunks)
│   └── interface.py                # (used by app.py Tab 3)
├── data/
│   ├── catalog/                    # CSV catalogs (tracked)
│   └── impact/                     # Impact scenario data (tracked)
├── requirements.txt
├── ASSUMPTIONS.md
└── README.md
```

---

## Data & Models

Large binary files are **not** tracked in git. To reproduce results:

| File | Size | How to get |
|------|------|------------|
| `data/windows/*.npz` | ~2 GB | Run `scripts/10_phase5_download.py` + `11_phase5_analyze.py` |
| `data/augmented_dataset/waveforms.hdf5` | ~800 MB | Run `scripts/12_prepare_finetune_dataset.py` |
| `models/gpd_final.pt` | ~2 MB | Run `scripts/20_final_finetune.py` |
| `models/phasenet_fixed.pt` | ~7 MB | Run `scripts/30_phasenet_fixed.py` |

Data sources: KOERI-EIDA FDSN (`eida.koeri.boun.edu.tr`), EMSC (`fdsn.eu`), ISC (`isc.ac.uk`)

---

## Installation

```bash
# Python 3.12 required
pip install -r requirements.txt
```

Key dependencies:
- `seisbench` — SeisBench model hub (PhaseNet, EQTransformer, GPD)
- `obspy` — seismological data processing
- `streamlit` — web application
- `torch` — PyTorch for fine-tuning
- `plotly` — interactive figures
- `anthropic` — LLM Q&A layer (Tab 3)

---

## Methodology

### Detection Pipeline
1. **Catalog query** — EMSC FDSN for M≥2.0 events, ISC for M<2.0
2. **Waveform download** — miniSEED via KOERI-EIDA, targeted windows (±30s around origin)
3. **Preprocessing** — detrend, bandpass 1–45 Hz, resample 100 Hz, normalize per channel
4. **Baseline** — STA/LTA (short=0.5s, long=10s)
5. **Deep learning** — SeisBench `annotate()` for zero-shot; custom sliding window for fine-tuned
6. **Fine-tuning** — EMSC analyst picks as labels; KOERI windows + augmentation (jitter, flip, noise)
7. **Ensemble** — average P-probability from GPD-FT + PhaseNet-Fixed; threshold 0.15

### PSHA Module
- Gutenberg-Richter: MLE b-value (Aki 1965), MAXC Mc with +0.2 correction
- 3 seismotectonic zones: NAF (b=1.16), EAF (b=1.00), Central Anatolia (b=1.04)
- Poisson model: P(t) = 1 − exp(−λt), 50-year exposure
- Omori-Utsu correction (10×) for aftershock-dominated Kahramanmaraş cells

### Impact Simulator
- Ground motion: Boore-Atkinson 2008 GMPE (empirically calibrated for Turkey)
- MMI conversion: Worden et al. 2012
- Building damage: HAZUS-MH MR5 lognormal fragility curves (masonry/RC/steel)
- Casualties: HAZUS indoor model with day/night occupancy
- **Validation**: 1999 Kocaeli M7.6 → predicted 17,672 vs actual 17,480 (ratio: 1.01×)

---

## Scientific Limitations

- Detection models are trained on Kahramanmaraş aftershocks — slight performance degradation expected on other regions
- P-MAE target <0.3s not met at population level (met within EMSC-matched subset: 0.28s median)
- EQTransformer domain gap persists (recall 0.642 despite full fine-tuning)
- Impact simulator uses province-level resolution — underestimates Istanbul near-fault exposure
- PSHA catalog is aftershock-dominated; Omori correction is an approximation

See [`artifacts/limitations.md`](artifacts/limitations.md) for the full list.

---

## References

- Woollam et al. (2022). SeisBench — A Toolbox for Machine Learning in Seismology. *SRL*
- Zhu & Beroza (2019). PhaseNet. *GJI*
- Mousavi et al. (2020). EQTransformer. *Nature Communications*
- Boore & Atkinson (2008). BA08 GMPE. *Earthquake Spectra*
- AFAD (2018). TBDY-2018 Turkish Building Earthquake Code
- Worden et al. (2012). MMI conversion. *BSSA*
- FEMA (2012). HAZUS-MH MR5 Technical Manual

---

## Citation

If you use this work, please cite:

```bibtex
@misc{tas2026seismicai,
  title        = {Seismic AI: Automatic Micro-Seismic Detection and TBDY-2018
                  Decision Support via Deep Learning},
  author       = {Ta{\c{s}}, Bertu{\u{g}} and Y{\"u}cel, Kadir Emir and
                  Takyac{\i}, Melih and {\"O}zdemir, Emre and Nasibo{\u{g}}lu, Efendi},
  year         = {2026},
  howpublished = {III. National Basic Sciences Youth Symposium and Science Fair,
                  Dokuz Eyl{\"u}l University, \.{I}zmir, Turkey, 12--13 May 2026},
  url          = {https://github.com/BertugTas/seismic-ai-project}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

Data from KOERI, EMSC, ISC are subject to their respective terms of use.
TBDY-2018 provisions are reproduced for educational purposes only.
