# Phase: 1
# Purpose: Test all FDSN endpoints and data sources for accessibility before any download
# Inputs: None (network requests only)
# Outputs: prints access status for each endpoint; results fed into data_access_log.md
# Limitations: Results depend on current network conditions and KOERI/EIDA uptime;
#              some channels may require EIDA token (tested as open-access first)

import sys
import traceback
from datetime import datetime, timezone

import requests
from obspy.clients.fdsn import Client
from obspy import UTCDateTime

# ── Configuration ─────────────────────────────────────────────────────────────
PILOT_START = UTCDateTime("2023-02-06T00:00:00")  # Kahramanmaraş main shock date
PILOT_END   = UTCDateTime("2023-02-20T00:00:00")  # 2-week pilot window

# Western Marmara bounding box
LAT_MIN, LAT_MAX = 39.5, 41.5
LON_MIN, LON_MAX = 26.0, 29.5

FDSN_ENDPOINTS = {
    "KOERI":  "https://eida.koeri.boun.edu.tr",
    "IRIS":   "https://service.iris.edu",         # fallback cross-check
}

HTTP_ENDPOINTS = {
    "KOERI_dataselect":  "https://eida.koeri.boun.edu.tr/fdsnws/dataselect/1/",
    "KOERI_station":     "https://eida.koeri.boun.edu.tr/fdsnws/station/1/",
    "KOERI_event":       "https://eida.koeri.boun.edu.tr/fdsnws/event/1/",
    "KOERI_availability":"https://eida.koeri.boun.edu.tr/fdsnws/availability/1/",
}

results = {}

def check(label, fn):
    """Run fn(), record OK or FAIL with message."""
    try:
        msg = fn()
        results[label] = ("OK", msg)
        print(f"  [OK]   {label}: {msg}")
    except Exception as e:
        results[label] = ("FAIL", str(e)[:200])
        print(f"  [FAIL] {label}: {str(e)[:200]}")


print("=" * 70)
print("PHASE 1 — FDSN Endpoint Access Check")
print(f"Run time (UTC): {datetime.now(timezone.utc).isoformat()}")
print("=" * 70)

# ── 1. Raw HTTP reachability ───────────────────────────────────────────────────
print("\n[1] Raw HTTP reachability (WSDL endpoints)")
for name, url in HTTP_ENDPOINTS.items():
    def _http(u=url):
        r = requests.get(u + "?", timeout=15)
        return f"HTTP {r.status_code}"
    check(name, _http)

# ── 2. ObsPy FDSN Client — station metadata ───────────────────────────────────
print("\n[2] ObsPy FDSN Client — IJ network station metadata")
def _stations():
    c = Client("https://eida.koeri.boun.edu.tr")
    inv = c.get_stations(
        network="IJ",
        latitude=(LAT_MIN + LAT_MAX) / 2,
        longitude=(LON_MIN + LON_MAX) / 2,
        maxradius=3.0,
        starttime=PILOT_START,
        endtime=PILOT_END,
        level="channel",
    )
    n = sum(1 for net in inv for sta in net for _ in sta)
    return f"{n} channels found in IJ network near Western Marmara"
check("KOERI_fdsn_stations_IJ_near_marmara", _stations)

# Broader search if narrow radius returns nothing
def _stations_broad():
    c = Client("https://eida.koeri.boun.edu.tr")
    inv = c.get_stations(
        network="IJ",
        minlatitude=LAT_MIN,
        maxlatitude=LAT_MAX,
        minlongitude=LON_MIN,
        maxlongitude=LON_MAX,
        starttime=PILOT_START,
        endtime=PILOT_END,
        level="channel",
    )
    names = [f"{net.code}.{sta.code}" for net in inv for sta in net]
    return f"{len(names)} stations: {', '.join(names[:10])}"
check("KOERI_fdsn_stations_IJ_bbox", _stations_broad)

# ── 3. ObsPy FDSN Client — event catalog ──────────────────────────────────────
print("\n[3] ObsPy FDSN Client — KOERI event catalog (pilot window)")
def _events():
    c = Client("https://eida.koeri.boun.edu.tr")
    cat = c.get_events(
        starttime=PILOT_START,
        endtime=PILOT_END,
        minlatitude=LAT_MIN,
        maxlatitude=LAT_MAX,
        minlongitude=LON_MIN,
        maxlongitude=LON_MAX,
        minmagnitude=0.0,
    )
    return f"{len(cat)} events retrieved"
check("KOERI_fdsn_events_marmara_pilot", _events)

# ── 4. Waveform availability probe (1 sample window, no full download) ─────────
print("\n[4] Waveform availability probe (tiny 1-second window, no bulk download)")
def _waveform_probe():
    c = Client("https://eida.koeri.boun.edu.tr")
    # Probe: fetch 1 second from the dataselect endpoint to confirm open access
    st = c.get_waveforms(
        network="IJ",
        station="*",
        location="*",
        channel="HH*",
        starttime=PILOT_START,
        endtime=PILOT_START + 1,
    )
    ids = list({tr.id for tr in st})
    return f"Open-access waveform OK — {len(ids)} trace IDs: {ids[:5]}"
check("KOERI_fdsn_waveform_open_access_probe", _waveform_probe)

# ── 5. STEAD GitHub reachability ───────────────────────────────────────────────
print("\n[5] STEAD dataset GitHub reachability")
def _stead():
    r = requests.get("https://github.com/smousavi05/STEAD", timeout=15)
    return f"HTTP {r.status_code}"
check("STEAD_github_reachable", _stead)

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("ACCESS CHECK SUMMARY")
print("=" * 70)
ok_count   = sum(1 for s, _ in results.values() if s == "OK")
fail_count = sum(1 for s, _ in results.values() if s == "FAIL")
print(f"  PASSED: {ok_count}   FAILED: {fail_count}   TOTAL: {len(results)}")
print()
for label, (status, msg) in results.items():
    tag = "✓" if status == "OK" else "✗"
    print(f"  {tag} {label}")
    if status == "FAIL":
        print(f"      → {msg}")

print()
print("Copy these results into artifacts/data_access_log.md (run 02_fetch_catalog.py next).")
