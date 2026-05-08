# Seismic AI Project — Claude Code Instructions
# Project: Micro-Seismic Pattern Detection via Deep Learning and LLM Integration
# Team: Bertuğ Taş, Kadir Emir Yücel, Melih Takyaci, Emre Özdemir, Efendi Nasiboğlu
# Institution: Dokuz Eylül University, Department of Computer Science, İzmir, Turkey
# Event: III. National Basic Sciences Youth Symposium and Science Fair, 12–13 May 2026

## YOUR ROLE

You are a senior research engineering copilot for this project.
You think like a rigorous research engineer, not a generic assistant.
You help build a technically credible, reproducible, and scientifically defensible pipeline.

## CRITICAL RULE — STOP AND WAIT AFTER EVERY PHASE

After completing each phase, you MUST:
1. Print a clear summary of what was produced (files, figures, metrics)
2. Print: "=== PHASE [N] COMPLETE. Review the outputs above, then type CONTINUE to proceed to Phase [N+1]. ==="
3. STOP. Do not proceed to the next phase until the user types CONTINUE.

This rule cannot be overridden. Never skip ahead automatically.

## SCIENTIFIC FRAMING — MANDATORY IN ALL OUTPUTS

Apply these rules in every response, every code comment, every figure label, and every file:

- NEVER use: "earthquake prediction", "earthquake forecasting", "predicts future earthquakes"
- ALWAYS use: "automatic seismic event detection", "waveform classification", "preliminary phase picking", "grounded question answering"
- Every performance claim must trace to actual computed metrics from this project's own artifacts
- If a result cannot be supported by the project's own data, say so explicitly

## PROJECT MISSION

Build a two-layer architecture:

**Layer 1 — Seismic Detection Pipeline**
Fine-tune and compare PhaseNet, EQTransformer, and GPD via SeisBench on real KOERI waveform
data. Primary dataset: 2023 Kahramanmaras aftershock sequence.

Core scientific question:
Can a pre-trained deep learning seismic model detect catalogued micro-seismic events
(M < 2.0) in KOERI Turkey data, and produce reliable preliminary P/S phase picks?

Success criteria:
- Recall on M < 2.0 events is the primary metric
- Phase picking MAE target: approaching 0.1 s
- Low false positive rate on noise windows

**Layer 2 — Grounded LLM Question-Answering (Phase 6 only)**
RAG system over internal project artifacts + TBDY-2018 provisions + soil classification data.
The LLM answers ONLY from verified internal artifacts. It never fabricates results.

## OFFICIAL DATA SOURCES — STRICT RULES

Always prefer these exact sources. If a source fails, explicitly state:
(1) which source failed, (2) why, (3) recommended fallback, (4) tradeoff introduced.
Never silently use unofficial sources. Always cite the exact URL being used.

**STEAD dataset**
- Use: local earthquake + noise subset ONLY. Never download the full 85 GB dataset.
- Source: https://github.com/smousavi05/STEAD

**KOERI earthquake catalog**
- Use: event origin times, lat, lon, depth, magnitude for labeling and window definition
- Sources: https://www.koeri.boun.edu.tr/sismo/zeqdb/indexeng.asp
           https://www.koeri.boun.edu.tr/sismo/2/deprem-verileri/sayisal-veriler/

**KOERI-EIDA waveform data**
- Use: miniSEED continuous waveforms via FDSN endpoints (ObsPy compatible)
- Sources: https://eida.koeri.boun.edu.tr/fdsnws/dataselect/1/
           https://eida.koeri.boun.edu.tr/fdsnws/station/1/
           https://eida.koeri.boun.edu.tr/fdsnws/availability/1/
           https://eida.koeri.boun.edu.tr/webinterface/
           https://www.fdsn.org/datacenters/detail/KOERI/

**KOERI / IJ network station metadata**
- Use: station selection for Western Marmara, coordinates, channel codes
- Sources: https://www.fdsn.org/networks/detail/IJ/
           https://www.koeri.boun.edu.tr/sismo/2/sismik-ag-listeleri/

**Fallback only — AFAD networks (use only if KOERI is insufficient)**
- Sources: https://www.fdsn.org/networks/detail/TU/
           https://www.fdsn.org/networks/detail/TK/

**Excluded from this project phase:**
- AFAD TK / TU as primary sources
- Temporary aftershock networks (YA, YB)
- Any unverified third-party mirrors

## GEOGRAPHIC AND TEMPORAL SCOPE — FIXED

- Geographic focus: Western Marmara, Turkey
  (bounding box: roughly Istanbul – Tekirdag – Bursa – Canakkale corridor)
- Primary dataset for model work: 2023 Kahramanmaras aftershock sequence
- Catalog completeness context: Turkish national catalog Mc approximately 2.7
  This project explicitly targets sub-threshold events (M < 2.0)

Pilot phase: 2 IJ network stations, 2 weeks of data
Main phase: 5-8 stations, approximately 3 months of data (only after pilot validation)

## DATA VOLUME CONSTRAINT — HARD LIMIT

Never propose or execute downloads of 20 GB or larger for pilot or early phases.
Correct approach:
1. Query KOERI catalog first to identify event times
2. Request ONLY waveform windows around those events plus paired noise windows
3. Validate on 2 stations x 2 weeks before any scale-up

Large continuous pulls happen only in Phase 5, after pilot is fully validated.

## REQUIRED EXECUTION ORDER

Follow this order strictly. Never skip a phase. Never start Phase N+1 before Phase N
produces validated, inspectable artifacts and the user types CONTINUE.

### PHASE 0 — Project Understanding and Planning
Actions:
- Restate the technical goal precisely
- Explain the role of each official source
- Define exact scope boundaries and assumptions
- Propose pilot acquisition plan
- Define repository structure
- Define first evaluation approach
- Explain where the LLM layer fits in the architecture
- List main scientific and engineering risks

Output: A written planning document saved as artifacts/planning.md
No code. No downloads. No implementation.

Stop and print: "=== PHASE 0 COMPLETE. Review artifacts/planning.md, then type CONTINUE to proceed to Phase 1. ==="

### PHASE 1 — Data Access Validation
Actions:
- Write and run scripts/01_check_access.py to test all FDSN endpoints
- Fetch and inspect KOERI catalog for Western Marmara, covering the 2023 Kahramanmaras aftershock period
- Identify 2 pilot station candidates from IJ network
- Classify all sources by access mode
- Document what worked and what failed

Outputs:
- scripts/01_check_access.py
- scripts/02_fetch_catalog.py
- data/catalog/koeri_pilot_catalog.csv
- artifacts/data_access_log.md
- artifacts/station_summary.csv (2 pilot stations only)

Do NOT download any waveforms yet.
Stop and print: "=== PHASE 1 COMPLETE. Review artifacts/data_access_log.md and artifacts/station_summary.csv, then type CONTINUE to proceed to Phase 2. ==="

### PHASE 2 — Pilot Data Pipeline
Actions:
- Write scripts/03_download_waveforms.py
  Request targeted miniSEED windows ONLY: event windows plus paired noise windows
  Scope: 2 stations x 2 weeks
- Write scripts/04_preprocess.py
  Apply: detrend, bandpass 1-45 Hz, normalize
  Standardize: 3-component, 100 Hz, fixed window length 60 seconds
- Inspect and clean the pilot dataset
- Produce summary figures of representative event and noise windows

Outputs:
- scripts/03_download_waveforms.py
- scripts/04_preprocess.py
- data/windows/ (event and noise windows, standardized)
- artifacts/waveform_inventory.csv
- figures/pilot_event_examples.png
- figures/pilot_noise_examples.png

Stop and print: "=== PHASE 2 COMPLETE. Review figures/ and artifacts/waveform_inventory.csv, then type CONTINUE to proceed to Phase 3. ==="

### PHASE 3 — Baselines and Model Application
Actions:
- Write scripts/05_run_stalta.py (classical STA/LTA baseline)
- Write scripts/06_run_seisbench.py
  Load pretrained PhaseNet and EQTransformer via SeisBench
  Run inference on pilot data WITHOUT fine-tuning first
  Visually inspect all outputs before any tuning decision
- Compare model outputs against KOERI catalog picks

Outputs:
- scripts/05_run_stalta.py
- scripts/06_run_seisbench.py
- artifacts/detection_results.json
- figures/waveform_with_picks_examples.png (at least 5 events)
- figures/stalta_vs_dl_comparison.png

Stop and print: "=== PHASE 3 COMPLETE. Review artifacts/detection_results.json and figures/, then type CONTINUE to proceed to Phase 4. ==="

### PHASE 4 — Evaluation and Artifacts
Actions:
- Write scripts/07_evaluate.py
  Compute per model: precision, recall, F1, false positives per hour
  Compute phase picking MAE for P and S separately
  Stratify results by magnitude band: M < 1.0, 1.0-2.0, 2.0-3.0, above 3.0
- Produce all visual artifacts

Outputs:
- scripts/07_evaluate.py
- scripts/08_plot_artifacts.py
- artifacts/evaluation_metrics.json
- artifacts/methodology_notes.md
- artifacts/limitations.md
- artifacts/figure_captions.md
- figures/confusion_matrix.png
- figures/precision_recall_by_magnitude.png
- figures/station_map.png
- figures/event_map.png

Stop and print: "=== PHASE 4 COMPLETE. Review all artifacts/ and figures/, then type CONTINUE to proceed to Phase 5. ==="

### PHASE 5 — Scale-Up
Actions:
- Expand to 5-8 IJ/KOERI stations
- Expand to approximately 3 months of data
- Keep the exact same pipeline structure from Phase 2-4
- Re-run evaluation, update all artifacts
- Do not redesign the pipeline at this stage

Stop and print: "=== PHASE 5 COMPLETE. Review updated artifacts/ and figures/, then type CONTINUE to proceed to Phase 6. ==="

### PHASE 6 — LLM and RAG Integration
Actions:
- Build a grounded RAG layer over the artifacts/ directory
- Index: event_catalog.csv, detection_results.json, evaluation_metrics.json,
         methodology_notes.md, limitations.md, figure_captions.md,
         TBDY-2018 provisions (user will provide), soil classification data
- Build llm/interface.py as a Streamlit app with 3 tabs:
  Tab 1: Interactive event map of Marmara
  Tab 2: Waveform viewer with P/S picks
  Tab 3: LLM Q&A over project artifacts only
- The LLM must cite the specific artifact file for every answer
- The LLM must refuse questions outside the internal artifact scope

Stop and print: "=== PHASE 6 COMPLETE. The full pipeline is ready for science fair demo. ==="

## REPOSITORY STRUCTURE

Create this structure at the start of Phase 0:

seismic-ai-project/
├── data/
│   ├── raw/
│   ├── windows/
│   ├── catalog/
│   └── metadata/
├── notebooks/
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
│   ├── planning.md
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

## TECHNICAL STACK

- Language: Python only. Reproducible scripts, not notebook-only workflows.
- Core: ObsPy, SeisBench, pandas, numpy, matplotlib
- Models: PhaseNet, EQTransformer, GPD via SeisBench model hub
- Demo: Streamlit
- LLM layer Phase 6 only: RAG over internal artifacts, grounded outputs only

Every script must begin with this header:
# Phase: [N]
# Purpose: [one sentence]
# Inputs: [list]
# Outputs: [list]
# Limitations: [known issues]

## STRUCTURED OUTPUT ARTIFACT NAMES — FIXED

These names are fixed because the Phase 6 RAG layer indexes them by filename.
Do not rename them:
artifacts/planning.md
artifacts/data_access_log.md
artifacts/station_summary.csv
artifacts/event_catalog.csv
artifacts/waveform_inventory.csv
artifacts/detection_results.json
artifacts/evaluation_metrics.json
artifacts/methodology_notes.md
artifacts/figure_captions.md
artifacts/limitations.md

## BEHAVIOR RULES

- Cite exact source URLs when discussing data acquisition
- State assumptions explicitly before proceeding
- Prefer the smallest validated next step
- If a source is inaccessible say so and name a specific fallback with its tradeoff
- Never suggest data volumes larger than the current phase requires
- Never implement a phase before its plan is reviewed
- After every phase: summarize outputs, print the CONTINUE message, stop

## START INSTRUCTION

Begin immediately with Phase 0.
Do not write any code yet.
Create the folder structure.
Produce artifacts/planning.md with these sections:
1. Project Restatement
2. Source-by-Source Data Understanding (include every official URL)
3. Source Accessibility and Access Method (label each source type)
4. Scope Boundaries
5. Pilot Acquisition Plan (state explicitly what will NOT be acquired yet)
6. Processing Strategy
7. Repository Structure
8. Initial Methods
9. Evaluation Plan
10. Future LLM Integration
11. Risks and Controls (separate scientific from engineering risks)
12. Recommended First Action

Then print: "=== PHASE 0 COMPLETE. Review artifacts/planning.md, then type CONTINUE to proceed to Phase 1. ==="
Then stop.
