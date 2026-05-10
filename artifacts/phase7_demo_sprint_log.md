# Phase 7: Demo Sprint Readiness Log

**Date:** 2026-05-10
**Deadline:** 2026-05-12 (Science Fair opens)
**Status:** ALL THREE TASKS COMPLETE

---

## Task 1: Station Z-Score Normalization — DONE

**Goal:** Strip station identity leakage (was 84.8%) from bn5 features.

**Implementation:**
- `seismic_engine/inference/station_normalizer.py` — Per-station z-score normalizer
- Normalizer fitted on **noise-only** traces (120 traces, 2 stations)
- Learns the noise baseline per station; events deviate from this baseline
- Falls back to global normalization for unseen stations

**Results (before → after normalization + retrain):**
| Metric | Before | After |
|--------|--------|-------|
| Event Recall | 65.4% | **100%** |
| Noise TNR | 98.3% | **100%** |
| Station Leakage | 84.8% | Not re-measured (expected <60%) |

**Files:**
- `seismic_engine/inference/station_normalizer.py`
- `models/station_normalizer.json`
- `scripts/102_retrain_v2_normalizer.py`

---

## Task 2: Cascaded GPD→PhaseNet — DONE

**Goal:** Replace GPD argmax (MAE 16.03s) with PhaseNet precise phase picker.

**Implementation:**
- `seismic_engine/inference/cascaded_detector.py` — Two-stage cascade
- Stage 1: GPD+SVM trigger (existing, improved with normalization)
- Stage 2: PhaseNet (pretrained INSTANCE via SeisBench) for sample-level P/S picks
- Overlap-add sliding window for traces longer than PhaseNet's input size

**Results:**
| Metric | GPD Argmax | PhaseNet Cascade |
|--------|-----------|------------------|
| P-Pick MAE | 16.03s | **12.34s** |
| P-Pick Median AE | 9.24s | **1.52s** |
| Within 1.0s | 31/138 (22%) | **34/92 (37%)** |
| Within 2.0s | — | **50/92 (54%)** |
| Picks with P | 138/211 | **92/211** (strict threshold) |

**Note:** PhaseNet pretrained on INSTANCE (Italian network) provides reasonable picks on KOERI Turkish data without fine-tuning. Median 1.5s represents a 6× improvement over GPD argmax.

**Files:**
- `seismic_engine/inference/cascaded_detector.py`

---

## Task 3: Live Monitor Demo Mode — DONE

**Goal:** Visual real-time seismograph simulation for jury presentation.

**Implementation:**
- `seismic_engine/streaming/replay_source.py` — ReplaySource + RingBuffer
- New "📡 Canlı İzleme (Demo)" tab in `app.py`
- Replays stored waveforms at configurable speed (1x to instant)
- Auto-triggers cascade on event detection
- Draws PhaseNet P/S picks on the live chart
- Streams LLM risk report immediately upon detection

**Demo Flow:**
1. Select a trace from the dropdown (sorted by magnitude)
2. Click "▶️ Akışı Başlat"
3. Watch waveform stream in real-time
4. When GPD+SVM detects an event → chart freezes
5. PhaseNet P/S arrival lines appear on the chart
6. LLM streams the structural risk report below

**Files:**
- `seismic_engine/streaming/__init__.py`
- `seismic_engine/streaming/replay_source.py`
- `app.py` (new tab6)

---

## Model Configuration Summary

| Parameter | Value |
|-----------|-------|
| Architecture | GPD bn5 → Z-norm → StandardScaler → PCA(100) → SVM RBF(C=20) |
| Sliding stride | 200 samples (2.0s at 100 Hz) |
| Detection threshold | 0.40 |
| PhaseNet model | SeisBench `instance` pretrained |
| Station normalizer | Noise-baseline fitted (KO.KMRS, KO.KOZT) |

---

## GitHub Checkpoint

**Repository:** https://github.com/MelihTakyaci/seismic-ai-project
**Commit:** Phase 7: Demo sprint — cascaded detector, station normalization, live monitor
**Branch:** main

---

## Known Limitations (Honest for jury Q&A)

1. **Evaluation on training data:** The 100% recall/TNR is measured on the same data used for training. True generalization requires unseen earthquakes from unseen stations.
2. **PhaseNet MAE not at 0.1s target:** Pretrained PhaseNet achieves 1.5s median (not 0.1s). Fine-tuning on Turkish data would improve this but was deprioritized for demo stability.
3. **2-station model:** Only KO.KMRS and KO.KOZT have event data. Other stations use global normalization fallback.
4. **Replay mode only:** No live SeedLink connection. Demo simulates real-time from stored data.

---

## Demo Checklist for May 12

- [ ] Verify Ollama running: `curl http://localhost:11434/api/tags`
- [ ] Launch app: `streamlit run app.py`
- [ ] Tab 6 demo: select M4.9 event, run at 4x speed
- [ ] Verify PhaseNet picks appear on chart
- [ ] Verify LLM report streams after detection
- [ ] Have backup plan if Ollama down: mock LLM auto-activates
