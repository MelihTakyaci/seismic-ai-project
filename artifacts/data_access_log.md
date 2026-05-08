# Data Access Log — Phase 1
# Date: 2026-04-24
# Authors: Bertuğ Taş, Kadir Emir Yücel, Melih Takyaci, Emre Özdemir, Efendi Nasiboğlu

---

## 1. FDSN Endpoint Reachability

| Endpoint | URL | HTTP Status | Result |
|----------|-----|-------------|--------|
| KOERI-EIDA dataselect | https://eida.koeri.boun.edu.tr/fdsnws/dataselect/1/ | 200 | **PASS** |
| KOERI-EIDA station | https://eida.koeri.boun.edu.tr/fdsnws/station/1/ | 200 | **PASS** |
| KOERI-EIDA event | https://eida.koeri.boun.edu.tr/fdsnws/event/1/ | 404 | **FAIL — no event service** |
| KOERI-EIDA availability | https://eida.koeri.boun.edu.tr/fdsnws/availability/1/ | 200 | **PASS** |
| EMSC FDSN event service | ObsPy Client("EMSC") | 200 | **PASS** |
| STEAD GitHub | https://github.com/smousavi05/STEAD | 200 | **PASS** |
| KOERI web catalog | http://www.koeri.boun.edu.tr/scripts/lst0.asp | 200 | PASS (HTML only, not machine-readable) |

---

## 2. Key Corrections Discovered in Phase 1

### Correction 1: Network Code — IJ → KO

**Finding:** The planning document specified "IJ network stations." Phase 1 testing
revealed that the IJ (Istanbul University Seismic Network) returns HTTP 204 (no data)
for all station queries in both the Western Marmara and Kahramanmaraş bounding boxes
during the pilot window.

**Correct network:** KO (Kandilli Observatory Regional Seismic Network), which is also
served by the KOERI-EIDA FDSN node and covers all of Turkey.

- IJ query result: HTTP 204 (no content) for any bounding box in Turkey during pilot period
- KO query result: 62 stations in Western Marmara, 6 stations in Kahramanmaraş

**Action taken:** All subsequent work uses network="KO".

---

### Correction 2: Event Catalog Service — KOERI EIDA → EMSC

**Finding:** KOERI's EIDA node (eida.koeri.boun.edu.tr) does NOT expose an FDSN event
service. HTTP query to /fdsnws/event/1/ returns HTTP 404.

**Root cause:** KOERI EIDA is a waveform and station metadata node. Event catalog access
for Turkey is provided by EMSC (European-Mediterranean Seismological Centre), which
maintains a FDSN-compliant event service covering the Turkish catalog.

- IRIS FDSN event service: returns HTTP 204 (no data) for Turkey region in this period
- USGS FDSN event service: returns HTTP 204 (no data) for Turkey region in this period
- EMSC FDSN event service: returns correct events — **confirmed working**

**Tradeoff:** EMSC catalog has a detection threshold of approximately M ≥ 2.0 for the
Western Marmara region and M ≥ 1.5 for the Kahramanmaraş sequence. Sub-threshold
events (M < 2.0 in Marmara, M < 1.5 in Kahramanmaraş) are not in the EMSC catalog.
This is a known limitation documented in artifacts/limitations.md.

**Action taken:** All event catalog fetches use ObsPy Client("EMSC").

---

### Correction 3: Pilot Bounding Box — Western Marmara → Kahramanmaraş

**Finding:** The planning document specifies "Primary dataset: 2023 Kahramanmaraş
aftershock sequence" AND "Geographic focus: Western Marmara." These refer to different
regions of Turkey.

Event count comparison for the 2-week pilot window (2023-02-06 to 2023-02-20):

| Region | Bounding Box | Events (M ≥ 1.0, EMSC) | Events M < 2.0 | Conclusion |
|--------|-------------|------------------------|----------------|------------|
| Western Marmara | 39.5–41.5°N, 26.0–29.5°E | 7 | 0 | Insufficient for pilot |
| Kahramanmaraş | 36.5–38.5°N, 35.5–38.5°E | 2,318 | ~0 in EMSC (M≥2 threshold) | Abundant for pipeline testing |

**Interpretation:** The project architecture is:
- **Training/validation data:** Kahramanmaraş aftershock sequence (2,318+ catalogued events
  M ≥ 2.0 from EMSC in 2 weeks; many more sub-threshold events in the local KOERI catalog)
- **Deployment target:** Western Marmara (seismic hazard relevance for Istanbul corridor)

**Action taken:** Pilot waveform windows will be fetched from the Kahramanmaraş region
(KO.KOZT + KO.KMRS stations). Western Marmara (KO.RKY + KO.KRBG) remains the main
phase deployment target (Phase 5).

---

## 3. Waveform Access Validation

**Test:** Fetch a 2-second window from KO.BOTS HN channels (open-access probe).

```
Client("https://eida.koeri.boun.edu.tr").get_waveforms(
    network="KO", station="BOTS", location="*", channel="HN*",
    starttime=UTCDateTime("2023-02-06T01:17:00"),
    endtime=UTCDateTime("2023-02-06T01:17:02")
)
→ 3 traces: KO.BOTS..HNE, KO.BOTS..HNN, KO.BOTS..HNZ
→ Result: OPEN ACCESS CONFIRMED (no authentication required)
```

**Authentication status:** Open-access waveforms are available for KO network without
an EIDA token. No token registration required for the pilot.

---

## 4. Channel Type Survey — Western Marmara (KO Network)

25 out of 62 KO stations in Western Marmara have HH broadband channels (preferred for
M < 2.0 detection). The remaining 37 have HN strong-motion channels only.

Pilot selection for Western Marmara (main phase):
- **KO.RKY** (40.687°N, 27.178°E) — HHE, HHN, HHZ — 100 Hz
- **KO.KRBG** (40.393°N, 27.298°E) — HHE, HHN, HHZ, HNE, HNN, HNZ — 100 Hz

Channel type rationale: HH broadband sensors have sensitivity across 0.01–40 Hz and
detect weak ground motion. HN strong-motion accelerometers clip at low amplitudes and
are poorly suited for M < 2.0 detection. All pilot work uses HH channels exclusively.

---

## 5. Channel Type Survey — Kahramanmaraş (KO Network)

6 KO stations in the Kahramanmaraş bounding box:

| Station | Lat | Lon | Channels | Type |
|---------|-----|-----|----------|------|
| KO.KOZT | 37.480 | 35.826 | HHE,HHN,HHZ | HH broadband |
| KO.KMRS | 37.509 | 36.900 | HHE,HHN,HHZ | HH broadband |
| KO.SARI | 38.407 | 36.418 | HHE,HHN,HHZ | HH broadband |
| KO.CEYT | 37.011 | 35.748 | HHE,HHN,HHZ | HH broadband |
| KO.GAZ  | 37.172 | 37.210 | HHE,HHN,HHZ | HH broadband |
| KO.KHMN | 37.392 | 37.157 | HNE,HNN,HNZ | HN strong-motion |

Pilot selection for Kahramanmaraş:
- **KO.KOZT** (37.480°N, 35.826°E) — HH broadband, ~70 km SW of epicenter
- **KO.KMRS** (37.509°N, 36.900°E) — HH broadband, ~15 km NNW of epicenter

---

## 6. EMSC Catalog Summary — Kahramanmaraş Pilot Window

- Period: 2023-02-06 to 2023-02-20 (2 weeks)
- Region: 36.5–38.5°N, 35.5–38.5°E
- Total events (M ≥ 1.0): 2,318
- M 2.0–3.0: 1,057 events
- M ≥ 3.0: 1,261 events
- M < 2.0: 0 (EMSC catalog threshold; sub-M2 events not included)
- P/S picks in EMSC catalog: 0 (EMSC provides origin times only, not phase picks)
- Saved to: data/catalog/koeri_pilot_catalog.csv

**Limitation on picks:** EMSC does not provide P and S phase picks in the FDSN event
service response. For phase picking evaluation (MAE metric), we must:
(a) use model-derived picks as reference for relative comparisons, or
(b) access KOERI's internal pick database via web interface (manual, not automated).

This is documented as a known limitation in artifacts/limitations.md.

---

## 7. Source Classification Summary

| Source | Type | Auth Required | Automatable | Coverage |
|--------|------|--------------|-------------|----------|
| KOERI-EIDA waveform (dataselect) | FDSN REST | None | Yes (ObsPy) | KO network, Turkey |
| KOERI-EIDA station metadata | FDSN REST | None | Yes (ObsPy) | KO, IJ, TL networks |
| EMSC event catalog | FDSN REST | None | Yes (ObsPy) | Turkey + Europe, M≥~1.5 |
| KOERI web catalog | HTML scraping | None | No (manual only) | Turkey, M≥~1.0 |
| STEAD dataset | GitHub download | None | Yes (requests/curl) | Global, HDF5 format |
| AFAD networks (TU/TK) | FDSN REST | Varies | Partial | Turkey, not used in pilot |

---

## 8. Open Issues for Phase 2

1. **P/S pick availability**: No automated access to KOERI phase picks via FDSN.
   Phase picking MAE evaluation must use alternative reference (see limitations.md).

2. **Sub-M2 catalog**: EMSC does not include M < 2.0 events. This directly constrains
   the primary evaluation metric (recall on M < 2.0). Will investigate KOERI web
   catalog access options or note as a fundamental limitation.

3. **Kahramanmaraş waveform density**: With 2,318 events in 2 weeks, event windows will
   overlap significantly (aftershock coda). Preprocessing must handle overlapping windows.

4. **AFAD overlap**: AFAD operated temporary networks during the Kahramanmaraş response.
   We confirm AFAD is not used in pilot (per project rules). KO network only.
