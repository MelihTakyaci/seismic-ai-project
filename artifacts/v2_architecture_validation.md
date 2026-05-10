# Phase 6: V2.0 Architecture Validation Report

**Date:** 2026-05-10
**Auditor Role:** Adversarial engineering reviewer
**Mandate:** Attack proposed V2.0 upgrades. Give definitive verdicts.

---

## Executive Summary

Three proposed V2.0 upgrades were benchmarked and stress-tested against real system constraints. Two receive qualified SUFFICIENT verdicts with mandatory conditions. One is FLAWED and requires a different approach entirely.

| Proposal | Verdict | Risk Level |
|----------|---------|------------|
| Cascaded GPD→PhaseNet | **SUFFICIENT (conditional)** | Medium |
| SeedLink Ring Buffer | **SUFFICIENT (conditional)** | Low |
| End-to-End Fine-Tuning at Scale | **FLAWED — REQUIRES ALTERNATIVE** | High |

---

## Validation 1: Cascaded GPD→PhaseNet for 0.1s Phase MAE

### Current Problem
GPD argmax phase picking gives **MAE = 16.03s** (target: 0.1s). This is not a bug — GPD is a 400-sample window classifier that outputs P(event|window), not a sample-level phase picker. Taking argmax of window-level probabilities across a 60s trace is architecturally wrong for phase picking.

### Proposed Solution
Two-stage cascade:
1. **Stage 1 (Trigger):** GPD+SVM classifies trace as event/noise (current system, working)
2. **Stage 2 (Pick):** PhaseNet runs ONLY on detected events, outputs sample-level P/S picks

### Benchmark Data

| Model | Params | Inference (CPU) | Architecture |
|-------|--------|-----------------|--------------|
| GPD (trigger) | 1,741,003 | 0.23 ms/window | Window classifier (400 samples) |
| PhaseNet (picker) | 268,443 | 0.58 ms/trace | U-Net, sample-level output |
| EQTransformer (alternative) | 376,935 | 9.77 ms/trace | Transformer, detection+picking |

**Combined cascade latency:** 0.23ms × ~30 windows + 0.58ms = **~7.5 ms/trace** (CPU)
**Single-model alternative (EQTransformer):** 9.77 ms/trace (CPU)

### Critical Analysis

**Strengths:**
- PhaseNet is specifically designed for sample-level phase picking (U-Net architecture)
- SeisBench PhaseNet pretrained on INSTANCE/STEAD achieves ~0.1s MAE on standard benchmarks
- Combined latency (7.5ms) is FASTER than EQTransformer alone (9.77ms)
- GPD trigger already works — we keep existing detection and add picking

**Weaknesses:**
- PhaseNet pretrained model may not generalize to KOERI data without fine-tuning (different noise characteristics, station responses)
- Two models = two failure modes, two sets of hyperparameters
- If GPD misses an event (current recall = 65.4%), PhaseNet never sees it
- PhaseNet expects 3-component input at specific normalization — must match our preprocessing exactly

**Killer Question: Why not just replace GPD entirely with EQTransformer?**

EQTransformer does BOTH detection AND phase picking in a single forward pass:
- 9.77ms/trace is still real-time capable (102 traces/second)
- Eliminates the cascade complexity
- Published MAE: ~0.03s P-pick, ~0.04s S-pick on STEAD
- BUT: 17× slower than GPD alone for detection-only tasks

### Verdict: **SUFFICIENT (conditional)**

The cascade is computationally valid and architecturally sound. However:

**Mandatory Conditions:**
1. PhaseNet MUST be validated on KOERI data before claiming 0.1s MAE — publish the actual measured MAE on our 138 detected events, not literature numbers
2. Keep EQTransformer as a benchmark comparator — if single-model EQTransformer achieves comparable MAE with simpler deployment, prefer it
3. The cascade does NOT fix the 65.4% recall problem — it only improves picks on ALREADY-DETECTED events. Detection recall is the harder unsolved problem.

**Recommended Implementation Order:**
1. Run pretrained PhaseNet on our 138 TP events → measure actual MAE
2. Run pretrained EQTransformer on same 138 events → measure actual MAE
3. If PhaseNet MAE < 1.0s without fine-tuning → cascade is validated
4. If both need fine-tuning → prefer EQTransformer (simpler single-model path)

---

## Validation 2: SeedLink Ring Buffer for Streaming

### Current Problem
Current system is batch-mode: upload a miniSEED file → classify → return result. No continuous monitoring capability. For a real observatory deployment, we need persistent streaming from KOERI stations.

### Proposed Solution
SeedLink client maintaining a circular buffer (ring buffer) of recent waveforms per station. GPD processes windows as they fill.

### Benchmark Data

| Metric | Value | Implication |
|--------|-------|-------------|
| GPD batch=100 latency | 351 ms | Can process 100 stations' windows in one GPU/MPS call |
| GPD single window | 0.23 ms | Negligible per-window cost |
| 1 Hz trigger rate | 1000 ms budget | 351ms batch leaves 649ms headroom |
| Memory (GPD+SVM) | 385 MB | Fits comfortably in any modern system |
| MPS available | Yes (PyTorch 2.6.0) | Apple Silicon acceleration for demo laptop |

**Throughput calculation:**
- 100 stations × 1 window/second = 100 windows/second
- GPD batch inference: 351ms for 100 windows
- Duty cycle: 35.1% → leaves 64.9% headroom for I/O, SVM, logging
- **Verdict: 100 stations at 1 Hz is feasible on a single laptop with MPS**

### Critical Analysis

**Strengths:**
- ObsPy has mature SeedLink client (`obspy.clients.seedlink`)
- Ring buffer is standard pattern — numpy circular buffer, O(1) insert
- KOERI EIDA endpoint (`eida.koeri.boun.edu.tr`) supports SeedLink protocol
- Batch processing amortizes GPU kernel launch overhead
- 385MB memory footprint means this runs on the demo laptop

**Weaknesses:**
- SeedLink connections DROP. KOERI EIDA has documented outages. Need reconnection logic with exponential backoff.
- Gap handling is non-trivial: if a station drops for 30s, the ring buffer has a hole. Do you zero-pad? Skip? Flag the window as unreliable?
- State management across restarts: which events have been processed? Need a persistent event log.
- Network latency from KOERI EIDA to your demo laptop is variable (50-500ms). Buffer must absorb jitter.
- The REAL bottleneck is not compute — it's the SeedLink data availability. Many KOERI stations have 30-60 minute latency to EIDA.

**Killer Question: Does SeedLink even matter for the science fair demo?**

The demo is 12-13 May 2026. You're showing a poster and a laptop demo. SeedLink adds:
- Network dependency (what if conference WiFi blocks port 18000?)
- Failure mode visible to judges ("why is it showing no data?")
- Complexity with zero scientific contribution to the detection paper

A pre-loaded replay buffer (simulating real-time from stored miniSEED) gives the SAME visual demo with ZERO network risk.

### Verdict: **SUFFICIENT (conditional)**

The architecture is sound and computationally proven. However:

**Mandatory Conditions:**
1. For the science fair demo (12-13 May): use a **replay buffer** from pre-downloaded data, NOT live SeedLink. Eliminates network failure risk entirely.
2. SeedLink implementation is a V2.0 post-demo feature. Build the ring buffer interface now, but feed it from files during demo.
3. Must implement: reconnection with backoff, gap flagging (not gap filling), event deduplication, and a health dashboard showing buffer fill state.

**Recommended Architecture:**
```
[DataSource Interface]
    ├── ReplaySource (demo mode: reads miniSEED files at real-time pace)
    └── SeedLinkSource (production mode: connects to EIDA)
         ↓
[Ring Buffer] (per-station, 120s circular numpy array)
         ↓
[Trigger Loop] (1 Hz: extract window → GPD batch → SVM → emit events)
         ↓
[Event Log] (SQLite or append-only JSON)
```

---

## Validation 3: Massive End-to-End Fine-Tuning

### Current Problem
- Station identity leakage: 84.8% station classification from bn5 features
- Dual-station inconsistency: 30.7% of same-earthquake pairs get different classifications
- Only ~80 independent earthquakes from 2 stations in a single aftershock sequence
- Model may have memorized station noise floors rather than learning seismic physics

### Proposed Solution
Fine-tune GPD (or PhaseNet/EQTransformer) end-to-end on a massive Turkish seismic dataset:
- Expand from 2 → 50+ stations
- Expand from ~80 → 5000+ earthquakes
- Train on multi-region data to force generalization
- Use SSL pretraining (wav2vec-style) on unlabeled continuous data

### Benchmark Data

| Component | Detail |
|-----------|--------|
| GPD total params | 1,741,003 |
| GPD fc1 (dense layer) | 1,280,200 params (73.5% of model) |
| GPD conv layers | ~460K params (26.5%) |
| Current training set | ~80 earthquakes, 2 stations, 1 sequence |
| Minimum for generalization | ~5000 earthquakes, 20+ stations, 3+ regions |
| STEAD Turkey subset | ~15,000 traces (but quality/metadata issues) |
| KOERI catalog 2020-2024 | ~50,000 M≥1.5 events available |

### Critical Analysis

**The Fundamental Problem is NOT Model Architecture — It's Data:**

The current system's failures (station leakage, inconsistency) are **data problems**, not architecture problems:
- 2 stations → model learns "KO.KMRS noise floor" vs "KO.KOZT noise floor"
- 1 aftershock sequence → model learns temporal patterns specific to Feb-Jun 2023
- ~80 earthquakes → severe overfitting regardless of architecture

Fine-tuning GPD end-to-end on the SAME 80 earthquakes will make this WORSE (deeper memorization). The proposal only works if you SIMULTANEOUSLY solve the data problem.

**Catastrophic Forgetting Risk:**

GPD was pretrained on STEAD (~1.2M traces, global dataset). Its conv layers encode universal seismic features. Fine-tuning end-to-end on 80 Turkish traces will:
- Destroy the universal feature representations
- Overfit to Turkish station characteristics
- Lose ability to detect event types not in the fine-tuning set

This is not hypothetical — it's the standard catastrophic forgetting phenomenon in transfer learning with small fine-tuning sets.

**SSL Pretraining Reality Check:**

Self-supervised pretraining (wav2vec-style on continuous waveforms) is scientifically interesting but:
- Requires engineering a custom pretraining loop (2-4 weeks of engineering time)
- Needs 1000+ hours of continuous data (~360 GB at 100 Hz, 3-component)
- Violates the project's 20 GB data limit for early phases
- No published evidence that SSL improves micro-seismic detection specifically
- This is a research project unto itself — it's a PhD topic, not a science fair add-on

**What Actually Fixes Station Leakage:**

The 84.8% station classification accuracy means bn5 features encode station identity. Three approaches:

1. **More stations (data fix):** 20+ stations forces the model to find features that generalize across station noise floors. This WORKS but requires significant data acquisition.

2. **Station-invariant training (adversarial):** Add a gradient reversal layer that penalizes station-predictive features. Elegant but complex to implement correctly.

3. **Feature normalization (cheap fix):** Apply per-station z-score normalization to bn5 features before SVM. Removes station-specific bias without touching GPD. Can be implemented in 20 lines of code.

### Verdict: **FLAWED — REQUIRES ALTERNATIVE**

End-to-end fine-tuning on the current dataset will make the system WORSE. The proposal conflates two separate problems:
1. **Data insufficiency** (solvable with more data)
2. **Architecture limitation** (solvable with per-station normalization + cascade picker)

**Recommended Alternative — Staged Approach:**

**Stage A (Immediate, 1 day):**
- Per-station feature normalization in SVM pipeline
- Re-evaluate dual-station consistency after normalization
- Expected improvement: station leakage drops from 84.8% to <60%

**Stage B (Short-term, 1 week):**
- Expand to 8-10 stations (already planned for Phase 5 scale-up)
- Retrain SVM only (freeze GPD, no fine-tuning)
- Re-evaluate: if recall improves and consistency improves → done

**Stage C (Medium-term, post-demo):**
- If Stage B insufficient: fine-tune ONLY GPD's fc1 layer (73.5% of params) with learning rate 1e-5
- Keep conv layers frozen (preserves universal seismic features)
- Requires minimum 500 earthquakes from 10+ stations

**Stage D (Long-term, if pursuing publication):**
- Full end-to-end training on 5000+ events
- Multi-region Turkish data (Marmara + Eastern Anatolia + Aegean)
- Adversarial station-invariance objective
- This is a 6-month research effort, not a demo feature

---

## Cross-Cutting Findings

### The Real V2.0 Priority Stack

Based on this validation, the highest-impact improvements ordered by effort:

| Priority | Action | Impact | Effort |
|----------|--------|--------|--------|
| 1 | Per-station feature normalization | Fixes consistency | 2 hours |
| 2 | PhaseNet cascade for phase picking | Fixes 16s→<1s MAE | 1 day |
| 3 | Replay buffer for demo streaming | Visual demo impact | 1 day |
| 4 | Expand to 8-10 stations + retrain SVM | Fixes generalization | 1 week |
| 5 | SeedLink production client | Real monitoring | 2 weeks |
| 6 | End-to-end fine-tuning | Marginal if 1-4 done | 2+ months |

### What to Tell the Judges

The current system has known limitations that are **scientifically honest to state**:
- "Our detection recall of 65.4% on M<2.0 events demonstrates the challenge of sub-catalog detection"
- "Phase picking requires a dedicated architecture (PhaseNet cascade), which is our immediate next step"
- "Station generalization requires multi-station training data — our pilot proves the pipeline works, scale-up proves it generalizes"

Do NOT claim the system "predicts earthquakes" or achieves performance it hasn't demonstrated.

---

## Final Recommendation

For the 12-13 May 2026 demo:
1. Implement per-station normalization (2 hours)
2. Add PhaseNet phase picking on detected events (measure real MAE)
3. Build replay-buffer demo mode (pre-loaded data, simulated real-time)
4. Present honest metrics with confidence intervals

The system's scientific value is not in achieving perfect numbers — it's in demonstrating that a reproducible, end-to-end pipeline from raw waveforms to grounded risk reports is achievable with open-source tools and limited compute.

---

*Report generated as part of Phase 6 Architecture Validation. All benchmark numbers measured on this project's hardware (Apple Silicon, PyTorch 2.6.0, MPS available).*
