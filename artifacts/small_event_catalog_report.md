# Small Event Catalog Report — Step 1A

**Region**: Lat 36.5–38.5, Lon 35.5–38.5
**Period**: 2023-02-06 → 2023-08-06
**Magnitude range**: M 0.5–2.0

## Source Query Results

| Source | Events found | Status |
|--------|-------------|--------|
| EMSC | 1208 | ✓ Success |
| INGV | 0 | ✗ Failed / No data |
| ISC | 29897 | ✓ Success |
| ISC_HTTP | 29956 | ✓ Success |
| AFAD | 10000 | ✓ Success |
| KOERI_WEB | 0 | ✗ Failed / No data |
| USGS | 0 | ✗ Failed / No data |

**Total (before dedup)**: 71061
**After deduplication**: 44214

## Magnitude Distribution (deduplicated)

- M 0.5–1.0: **3297 events**
- M 1.0–1.5: **17953 events**
- M 1.5–2.0: **22961 events**

## Recommended Next Step

If ≥ 200 events found: proceed to Step 1B waveform download.
If < 200 events found: use all available; note limitation explicitly.
If 0 events found: document as hard limitation; AFAD data request required.

## Known Limitations

- KOERI has no FDSN event endpoint; web scraping is fragile
- EMSC catalog completeness threshold for Turkey: Mc ≈ 2.0–2.5
- ISC catalog may be incomplete for 2023 data (retrospective assembly)
- AFAD API access may be throttled or require authentication
- Sub-threshold completeness (M < 1.5) requires local seismic network records