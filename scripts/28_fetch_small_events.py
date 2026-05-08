# Phase: 5 Quality Audit — Optimization Step 1A
# Purpose: Fetch M 0.5–2.0 events in Kahramanmaras region (Feb–Aug 2023)
#          from multiple catalog sources to enable sub-threshold recall evaluation.
# Inputs:  None (network queries only)
# Outputs: data/catalog/small_events_catalog.csv, artifacts/small_event_catalog_report.md
# Limitations: KOERI has no FDSN event endpoint — web scraping attempted as fallback.
#              AFAD API may require authentication for fine-grained queries.
#              ISC catalog is retrospective and may lag recent events.

from pathlib import Path
import json
import time
import traceback

import requests
import pandas as pd
from obspy import UTCDateTime
from obspy.clients.fdsn import Client

ROOT    = Path(__file__).resolve().parent.parent
CAT_DIR = ROOT / "data" / "catalog"
ART     = ROOT / "artifacts"
CAT_DIR.mkdir(parents=True, exist_ok=True)

# ── Region / time bounds ────────────────────────────────────────────────────
MINLAT  = 36.5
MAXLAT  = 38.5
MINLON  = 35.5
MAXLON  = 38.5
MINMAG  = 0.5
MAXMAG  = 2.0
TSTART  = UTCDateTime("2023-02-06")
TEND    = UTCDateTime("2023-08-06")
TSTART_STR = "2023-02-06"
TEND_STR   = "2023-08-06"

print("=" * 70)
print("STEP 1A — Fetching M 0.5–2.0 Events: Kahramanmaras 2023")
print(f"  Region: Lat {MINLAT}–{MAXLAT}, Lon {MINLON}–{MAXLON}")
print(f"  Period: {TSTART_STR} → {TEND_STR}")
print(f"  Magnitude: {MINMAG}–{MAXMAG}")
print("=" * 70)

results = {}   # source → list of dicts

# ── SOURCE 1: EMSC FDSN ─────────────────────────────────────────────────────
print("\n[1] EMSC FDSN (seismicportal.eu) ...")
try:
    client_emsc = Client("EMSC")
    cat_emsc = client_emsc.get_events(
        starttime=TSTART, endtime=TEND,
        minlatitude=MINLAT, maxlatitude=MAXLAT,
        minlongitude=MINLON, maxlongitude=MAXLON,
        minmagnitude=MINMAG, maxmagnitude=MAXMAG,
    )
    rows_emsc = []
    for ev in cat_emsc:
        o   = ev.preferred_origin() or ev.origins[0]
        mag = ev.preferred_magnitude() or (ev.magnitudes[0] if ev.magnitudes else None)
        rows_emsc.append({
            "event_id":    str(ev.resource_id),
            "origin_time": str(o.time),
            "latitude":    float(o.latitude),
            "longitude":   float(o.longitude),
            "depth_km":    float(o.depth / 1000) if o.depth is not None else None,
            "magnitude":   float(mag.mag) if mag else None,
            "mag_type":    str(mag.magnitude_type) if mag else None,
            "source":      "EMSC",
        })
    results["EMSC"] = rows_emsc
    print(f"    ✓ EMSC returned {len(rows_emsc)} events (M {MINMAG}–{MAXMAG})")
except Exception as e:
    results["EMSC"] = []
    print(f"    ✗ EMSC failed: {e}")

# ── SOURCE 2: INGV FDSN ─────────────────────────────────────────────────────
print("\n[2] INGV FDSN ...")
try:
    client_ingv = Client("INGV")
    cat_ingv = client_ingv.get_events(
        starttime=TSTART, endtime=TEND,
        minlatitude=MINLAT, maxlatitude=MAXLAT,
        minlongitude=MINLON, maxlongitude=MAXLON,
        minmagnitude=MINMAG, maxmagnitude=MAXMAG,
    )
    rows_ingv = []
    for ev in cat_ingv:
        o   = ev.preferred_origin() or ev.origins[0]
        mag = ev.preferred_magnitude() or (ev.magnitudes[0] if ev.magnitudes else None)
        rows_ingv.append({
            "event_id":    str(ev.resource_id),
            "origin_time": str(o.time),
            "latitude":    float(o.latitude),
            "longitude":   float(o.longitude),
            "depth_km":    float(o.depth / 1000) if o.depth is not None else None,
            "magnitude":   float(mag.mag) if mag else None,
            "mag_type":    str(mag.magnitude_type) if mag else None,
            "source":      "INGV",
        })
    results["INGV"] = rows_ingv
    print(f"    ✓ INGV returned {len(rows_ingv)} events (M {MINMAG}–{MAXMAG})")
except Exception as e:
    results["INGV"] = []
    print(f"    ✗ INGV failed: {e}")

# ── SOURCE 3: ISC FDSN ──────────────────────────────────────────────────────
print("\n[3] ISC FDSN (isc.ac.uk) ...")
try:
    client_isc = Client("ISC")
    cat_isc = client_isc.get_events(
        starttime=TSTART, endtime=TEND,
        minlatitude=MINLAT, maxlatitude=MAXLAT,
        minlongitude=MINLON, maxlongitude=MAXLON,
        minmagnitude=MINMAG, maxmagnitude=MAXMAG,
    )
    rows_isc = []
    for ev in cat_isc:
        o   = ev.preferred_origin() or ev.origins[0]
        mag = ev.preferred_magnitude() or (ev.magnitudes[0] if ev.magnitudes else None)
        rows_isc.append({
            "event_id":    str(ev.resource_id),
            "origin_time": str(o.time),
            "latitude":    float(o.latitude),
            "longitude":   float(o.longitude),
            "depth_km":    float(o.depth / 1000) if o.depth is not None else None,
            "magnitude":   float(mag.mag) if mag else None,
            "mag_type":    str(mag.magnitude_type) if mag else None,
            "source":      "ISC",
        })
    results["ISC"] = rows_isc
    print(f"    ✓ ISC returned {len(rows_isc)} events (M {MINMAG}–{MAXMAG})")
except Exception as e:
    results["ISC"] = []
    print(f"    ✗ ISC failed: {e}")

# ── SOURCE 4: ISC REST API (direct HTTP) ────────────────────────────────────
print("\n[4] ISC REST API (direct HTTP query) ...")
try:
    isc_url = (
        "http://www.isc.ac.uk/fdsnws/event/1/query"
        f"?starttime={TSTART_STR}T00:00:00&endtime={TEND_STR}T23:59:59"
        f"&minlat={MINLAT}&maxlat={MAXLAT}&minlon={MINLON}&maxlon={MAXLON}"
        f"&minmag={MINMAG}&maxmag={MAXMAG}&format=text&nodata=404"
    )
    print(f"    URL: {isc_url}")
    resp = requests.get(isc_url, timeout=120)
    if resp.status_code == 200:
        lines = [l for l in resp.text.strip().split("\n") if not l.startswith("#") and l.strip()]
        rows_isc_http = []
        for line in lines[1:]:  # skip header
            parts = line.split("|")
            if len(parts) < 13: continue
            try:
                rows_isc_http.append({
                    "event_id":    parts[0].strip(),
                    "origin_time": parts[1].strip(),
                    "latitude":    float(parts[2]),
                    "longitude":   float(parts[3]),
                    "depth_km":    float(parts[4]) if parts[4].strip() else None,
                    "magnitude":   float(parts[10]) if parts[10].strip() else None,
                    "mag_type":    parts[9].strip(),
                    "source":      "ISC_HTTP",
                })
            except Exception:
                continue
        results["ISC_HTTP"] = rows_isc_http
        print(f"    ✓ ISC HTTP returned {len(rows_isc_http)} events")
    elif resp.status_code == 404:
        results["ISC_HTTP"] = []
        print(f"    ✗ ISC HTTP: 404 no data found")
    else:
        results["ISC_HTTP"] = []
        print(f"    ✗ ISC HTTP: status {resp.status_code}")
except Exception as e:
    results["ISC_HTTP"] = []
    print(f"    ✗ ISC HTTP failed: {e}")

# ── SOURCE 5: AFAD REST API ─────────────────────────────────────────────────
print("\n[5] AFAD API (deprem.afad.gov.tr) ...")
try:
    # AFAD v2 filter endpoint
    afad_url = "https://deprem.afad.gov.tr/apiv2/event/filter"
    params = {
        "start":    TSTART_STR + " 00:00:00",
        "end":      TEND_STR   + " 23:59:59",
        "minlat":   str(MINLAT),
        "maxlat":   str(MAXLAT),
        "minlon":   str(MINLON),
        "maxlon":   str(MAXLON),
        "minmag":   str(MINMAG),
        "maxmag":   str(MAXMAG),
        "orderby":  "timedesc",
        "limit":    "10000",
        "skip":     "0",
    }
    headers = {"Content-Type": "application/json"}
    print(f"    URL: {afad_url}")
    resp_afad = requests.get(afad_url, params=params, headers=headers, timeout=60)
    if resp_afad.status_code == 200:
        data_afad = resp_afad.json()
        # AFAD returns list directly or under a key
        if isinstance(data_afad, list):
            evlist = data_afad
        elif isinstance(data_afad, dict):
            evlist = data_afad.get("eventList", data_afad.get("data", data_afad.get("result", [])))
        else:
            evlist = []
        rows_afad = []
        for ev in evlist:
            try:
                mag_val = ev.get("magnitude") or ev.get("mag") or ev.get("ml")
                if mag_val is None: continue
                mag_val = float(mag_val)
                if not (MINMAG <= mag_val <= MAXMAG): continue
                rows_afad.append({
                    "event_id":    str(ev.get("eventID") or ev.get("id") or ""),
                    "origin_time": str(ev.get("date") or ev.get("origintime") or ev.get("time") or ""),
                    "latitude":    float(ev.get("latitude") or ev.get("lat") or 0),
                    "longitude":   float(ev.get("longitude") or ev.get("lon") or 0),
                    "depth_km":    float(ev.get("depth") or 0),
                    "magnitude":   mag_val,
                    "mag_type":    str(ev.get("magnitudeType") or ev.get("magtype") or "ML"),
                    "source":      "AFAD",
                })
            except Exception:
                continue
        results["AFAD"] = rows_afad
        print(f"    ✓ AFAD returned {len(rows_afad)} events (M {MINMAG}–{MAXMAG})")
    else:
        results["AFAD"] = []
        print(f"    ✗ AFAD: status {resp_afad.status_code}  body={resp_afad.text[:200]}")
except Exception as e:
    results["AFAD"] = []
    print(f"    ✗ AFAD failed: {e}")

# ── SOURCE 6: KOERI web scraping (zeqdb) ────────────────────────────────────
print("\n[6] KOERI web catalog (zeqdb) — HTTP scrape ...")
try:
    # KOERI zeqdb uses a POST or GET form; try the text query endpoint
    koeri_url = "http://www.koeri.boun.edu.tr/sismo/zeqdb/submitRecherche.asp"
    # Try known KOERI FDSN-style REST endpoint variants
    koeri_rest_candidates = [
        (
            "http://www.koeri.boun.edu.tr/sismo/zeqdb/indexeng.asp",
            {}
        ),
        (
            "http://www.koeri.boun.edu.tr/sismo/2/deprem-verileri/sayisal-veriler/",
            {}
        ),
    ]
    # The zeqdb is a web form; try direct HTTP GET with query params
    koeri_query = (
        "http://www.koeri.boun.edu.tr/sismo/zeqdb/submitRecherche.asp"
        f"?Filtre_Debut_Date_Texte={TSTART_STR.replace('-','/')}"
        f"&Filtre_Fin_Date_Texte={TEND_STR.replace('-','/')}"
        f"&Filtre_Magnitude_Min={MINMAG}&Filtre_Magnitude_Max={MAXMAG}"
        f"&Filtre_Latitude_Min={MINLAT}&Filtre_Latitude_Max={MAXLAT}"
        f"&Filtre_Longitude_Min={MINLON}&Filtre_Longitude_Max={MAXLON}"
        f"&Recherche=Recherche"
    )
    print(f"    Trying: {koeri_query[:80]}...")
    resp_k = requests.get(koeri_query, timeout=30)
    if resp_k.status_code == 200 and len(resp_k.text) > 500:
        # Try to parse as HTML table (rough check)
        text = resp_k.text
        n_rows = text.count("<tr") + text.count("<TR")
        print(f"    Response {resp_k.status_code}, ~{n_rows} HTML rows — saving raw for inspection")
        raw_path = CAT_DIR / "koeri_raw_response.html"
        raw_path.write_text(text, encoding="utf-8", errors="replace")
        print(f"    Raw HTML saved → {raw_path}")

        # Try to extract rows with BeautifulSoup if available
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(text, "html.parser")
            rows_k = []
            for tr in soup.find_all("tr")[1:]:
                tds = [td.get_text(strip=True) for td in tr.find_all("td")]
                if len(tds) >= 6:
                    try:
                        mag = float(tds[4]) if tds[4] else None
                        if mag and MINMAG <= mag <= MAXMAG:
                            rows_k.append({
                                "event_id":    tds[0] if tds else "",
                                "origin_time": f"{tds[1]} {tds[2]}" if len(tds) > 2 else tds[1],
                                "latitude":    float(tds[3]) if tds[3] else None,
                                "longitude":   float(tds[4]) if len(tds) > 4 else None,
                                "depth_km":    float(tds[5]) if len(tds) > 5 and tds[5] else None,
                                "magnitude":   mag,
                                "mag_type":    "ML",
                                "source":      "KOERI_WEB",
                            })
                    except Exception:
                        pass
            results["KOERI_WEB"] = rows_k
            print(f"    ✓ KOERI web parsed {len(rows_k)} candidate rows")
        except ImportError:
            results["KOERI_WEB"] = []
            print("    ✗ BeautifulSoup not available — skipping HTML parse")
    else:
        results["KOERI_WEB"] = []
        print(f"    ✗ KOERI web: status {resp_k.status_code}")
except Exception as e:
    results["KOERI_WEB"] = []
    print(f"    ✗ KOERI web failed: {e}")

# ── SOURCE 7: USGS ComCat (FDSN) ────────────────────────────────────────────
print("\n[7] USGS ComCat (FDSN) ...")
try:
    client_usgs = Client("USGS")
    cat_usgs = client_usgs.get_events(
        starttime=TSTART, endtime=TEND,
        minlatitude=MINLAT, maxlatitude=MAXLAT,
        minlongitude=MINLON, maxlongitude=MAXLON,
        minmagnitude=MINMAG, maxmagnitude=MAXMAG,
    )
    rows_usgs = []
    for ev in cat_usgs:
        o   = ev.preferred_origin() or ev.origins[0]
        mag = ev.preferred_magnitude() or (ev.magnitudes[0] if ev.magnitudes else None)
        rows_usgs.append({
            "event_id":    str(ev.resource_id),
            "origin_time": str(o.time),
            "latitude":    float(o.latitude),
            "longitude":   float(o.longitude),
            "depth_km":    float(o.depth / 1000) if o.depth is not None else None,
            "magnitude":   float(mag.mag) if mag else None,
            "mag_type":    str(mag.magnitude_type) if mag else None,
            "source":      "USGS",
        })
    results["USGS"] = rows_usgs
    print(f"    ✓ USGS returned {len(rows_usgs)} events (M {MINMAG}–{MAXMAG})")
except Exception as e:
    results["USGS"] = []
    print(f"    ✗ USGS failed: {e}")

# ── Consolidate and deduplicate ─────────────────────────────────────────────
print("\n[8] Consolidating results ...")
all_rows = []
for src, rows in results.items():
    all_rows.extend(rows)

print(f"\n  Source summary:")
print(f"  {'Source':<15} {'Events':>8}")
print(f"  {'-'*25}")
total = 0
for src, rows in results.items():
    n = len(rows)
    total += n
    status = "✓" if n > 0 else "✗"
    print(f"  {status} {src:<13} {n:>8}")
print(f"  {'TOTAL':<15} {total:>8}")

if all_rows:
    df = pd.DataFrame(all_rows)
    df["magnitude"] = pd.to_numeric(df["magnitude"], errors="coerce")
    df = df.dropna(subset=["magnitude"])
    df = df[df["magnitude"].between(MINMAG, MAXMAG)]

    # Deduplicate by origin_time proximity (within 5s) and location proximity (0.1 deg)
    # Simple approach: sort by source priority, keep first
    SOURCE_PRIORITY = ["AFAD", "ISC_HTTP", "ISC", "EMSC", "INGV", "USGS", "KOERI_WEB"]
    df["src_rank"] = df["source"].map(
        {s: i for i, s in enumerate(SOURCE_PRIORITY)}
    ).fillna(99)
    df["origin_time_dt"] = pd.to_datetime(df["origin_time"], utc=True, errors="coerce")
    df = df.sort_values(["src_rank", "origin_time_dt"]).reset_index(drop=True)

    # Group by rounded lat/lon/time to deduplicate
    df["lat_r"] = df["latitude"].round(2)
    df["lon_r"] = df["longitude"].round(2)
    df["time_r"] = df["origin_time_dt"].dt.floor("10s")
    df_dedup = df.drop_duplicates(subset=["lat_r", "lon_r", "time_r"]).copy()
    df_dedup = df_dedup.drop(columns=["src_rank", "lat_r", "lon_r", "time_r"])

    out_path = CAT_DIR / "small_events_catalog.csv"
    df_dedup.to_csv(out_path, index=False)
    print(f"\n  After deduplication: {len(df_dedup)} unique events")

    # Magnitude distribution
    print(f"\n  Magnitude distribution:")
    bins = [0.5, 1.0, 1.5, 2.0]
    labels = ["0.5–1.0", "1.0–1.5", "1.5–2.0"]
    df_dedup["mag_band"] = pd.cut(df_dedup["magnitude"], bins=bins, labels=labels, right=True)
    for band in labels:
        n = (df_dedup["mag_band"] == band).sum()
        print(f"    M {band}: {n} events")

    print(f"\n  ✓ Saved → {out_path}")
else:
    df_dedup = pd.DataFrame()
    print("\n  ✗ No events found from any source.")

# ── Write report ────────────────────────────────────────────────────────────
report_lines = [
    "# Small Event Catalog Report — Step 1A",
    "",
    f"**Region**: Lat {MINLAT}–{MAXLAT}, Lon {MINLON}–{MAXLON}",
    f"**Period**: {TSTART_STR} → {TEND_STR}",
    f"**Magnitude range**: M {MINMAG}–{MAXMAG}",
    "",
    "## Source Query Results",
    "",
    "| Source | Events found | Status |",
    "|--------|-------------|--------|",
]
for src, rows in results.items():
    n = len(rows)
    status = "✓ Success" if n > 0 else "✗ Failed / No data"
    report_lines.append(f"| {src} | {n} | {status} |")

report_lines += [
    "",
    f"**Total (before dedup)**: {total}",
    f"**After deduplication**: {len(df_dedup)}",
    "",
    "## Magnitude Distribution (deduplicated)",
    "",
]
if not df_dedup.empty and "mag_band" in df_dedup.columns:
    for band in ["0.5–1.0", "1.0–1.5", "1.5–2.0"]:
        n = (df_dedup["mag_band"] == band).sum()
        report_lines.append(f"- M {band}: **{n} events**")

report_lines += [
    "",
    "## Recommended Next Step",
    "",
    "If ≥ 200 events found: proceed to Step 1B waveform download.",
    "If < 200 events found: use all available; note limitation explicitly.",
    "If 0 events found: document as hard limitation; AFAD data request required.",
    "",
    "## Known Limitations",
    "",
    "- KOERI has no FDSN event endpoint; web scraping is fragile",
    "- EMSC catalog completeness threshold for Turkey: Mc ≈ 2.0–2.5",
    "- ISC catalog may be incomplete for 2023 data (retrospective assembly)",
    "- AFAD API access may be throttled or require authentication",
    "- Sub-threshold completeness (M < 1.5) requires local seismic network records",
]

report_path = ART / "small_event_catalog_report.md"
(ART).mkdir(parents=True, exist_ok=True)
report_path.write_text("\n".join(report_lines), encoding="utf-8")
print(f"\n  Report → {report_path}")

print("\n" + "=" * 70)
print("STEP 1A COMPLETE")
print(f"  Events found: {len(df_dedup)}")
if len(df_dedup) >= 200:
    print(f"  ✓ Sufficient events for Step 1B waveform download")
elif len(df_dedup) > 0:
    print(f"  ⚠ Fewer than 200 events — proceed with what is available")
else:
    print(f"  ✗ No M < 2.0 events found — AFAD direct data request may be needed")
print("=" * 70)
