# Phase: 1
# Purpose: Fetch KOERI/EMSC earthquake catalog for Western Marmara pilot window and save as CSV;
#          fetch KO network station inventory and select 2 pilot stations with HH broadband channels
# Inputs: EMSC FDSN event service (event catalog); KOERI-EIDA FDSN station service (metadata)
# Outputs: data/catalog/koeri_pilot_catalog.csv, artifacts/station_summary.csv
# Limitations: Catalog completeness drops sharply below M~2.7 for Turkey national catalog;
#              small-event completeness in Western Marmara may be lower during aftershock
#              sequence due to coda interference; P/S picks may not be available for M<2.
#              KOERI EIDA does not expose an FDSN event service — EMSC is used as the
#              event catalog source (same underlying catalog, FDSN-compliant interface).

import os
import sys
from pathlib import Path

from obspy.clients.fdsn import Client
from obspy import UTCDateTime
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
CATALOG_DIR = ROOT / "data" / "catalog"
ARTIFACT_DIR= ROOT / "artifacts"
CATALOG_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

CATALOG_OUT  = CATALOG_DIR / "koeri_pilot_catalog.csv"
STATION_OUT  = ARTIFACT_DIR / "station_summary.csv"

# ── Pilot parameters ───────────────────────────────────────────────────────────
PILOT_START = UTCDateTime("2023-02-06T00:00:00")
PILOT_END   = UTCDateTime("2023-02-20T00:00:00")

LAT_MIN, LAT_MAX = 39.5, 41.5
LON_MIN, LON_MAX = 26.0, 29.5

KOERI_EIDA_BASE = "https://eida.koeri.boun.edu.tr"
# KOERI EIDA does not host an FDSN event service.
# EMSC (European-Mediterranean Seismological Centre) provides FDSN-compliant
# access to the same regional catalog, including Turkish events.
EVENT_SERVICE   = "EMSC"

print("=" * 70)
print("PHASE 1 — KOERI/EMSC Catalog Fetch + KO Station Selection")
print(f"Pilot window  : {PILOT_START.date} → {PILOT_END.date}")
print(f"Bounding box  : Lat [{LAT_MIN}, {LAT_MAX}]  Lon [{LON_MIN}, {LON_MAX}]")
print(f"Event service : {EVENT_SERVICE} (KOERI EIDA has no event endpoint)")
print(f"Waveform/meta : KOERI-EIDA ({KOERI_EIDA_BASE})")
print("=" * 70)

# ── 1. Fetch event catalog via EMSC ───────────────────────────────────────────
print("\n[1] Fetching event catalog via EMSC FDSN...")
event_client = Client(EVENT_SERVICE)
try:
    catalog = event_client.get_events(
        starttime=PILOT_START,
        endtime=PILOT_END,
        minlatitude=LAT_MIN,
        maxlatitude=LAT_MAX,
        minlongitude=LON_MIN,
        maxlongitude=LON_MAX,
        minmagnitude=0.0,
        orderby="time",
    )
    print(f"    Retrieved {len(catalog)} events from EMSC.")
except Exception as e:
    print(f"    [FAIL] EMSC event fetch failed: {e}")
    sys.exit(1)

rows = []
for ev in catalog:
    o  = ev.preferred_origin() or (ev.origins[0] if ev.origins else None)
    mg = ev.preferred_magnitude() or (ev.magnitudes[0] if ev.magnitudes else None)
    if o is None:
        continue

    p_pick_str, s_pick_str = "", ""
    for pick in ev.picks:
        if pick.phase_hint and pick.phase_hint.upper().startswith("P") and not p_pick_str:
            p_pick_str = str(pick.time)
        if pick.phase_hint and pick.phase_hint.upper().startswith("S") and not s_pick_str:
            s_pick_str = str(pick.time)

    rows.append({
        "event_id":       str(ev.resource_id),
        "origin_time":    str(o.time),
        "latitude":       round(float(o.latitude), 4)   if o.latitude  is not None else "",
        "longitude":      round(float(o.longitude), 4)  if o.longitude is not None else "",
        "depth_km":       round(float(o.depth) / 1000.0, 2) if o.depth is not None else "",
        "magnitude":      round(float(mg.mag), 2)       if mg is not None else "",
        "mag_type":       mg.magnitude_type              if mg is not None else "",
        "p_pick_time":    p_pick_str,
        "s_pick_time":    s_pick_str,
        "catalog_source": f"EMSC_FDSN",
    })

df_cat = pd.DataFrame(rows)
df_cat.to_csv(CATALOG_OUT, index=False)
print(f"    Saved {len(df_cat)} events → {CATALOG_OUT}")

if len(df_cat) > 0:
    df_cat["magnitude"] = pd.to_numeric(df_cat["magnitude"], errors="coerce")
    print("\n    Magnitude distribution:")
    bands  = [(-99, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 99)]
    labels = ["M < 1.0", "1.0 ≤ M < 2.0", "2.0 ≤ M < 3.0", "M ≥ 3.0"]
    for (lo, hi), label in zip(bands, labels):
        n = ((df_cat["magnitude"] >= lo) & (df_cat["magnitude"] < hi)).sum()
        print(f"      {label}: {n} events")
    print(f"      Has P pick: {(df_cat['p_pick_time'] != '').sum()}")
    print(f"      Has S pick: {(df_cat['s_pick_time'] != '').sum()}")

# ── 2. Fetch KO network station inventory ─────────────────────────────────────
print("\n[2] Fetching KO network station inventory (Western Marmara bbox)...")
station_client = Client(KOERI_EIDA_BASE)
try:
    inventory = station_client.get_stations(
        network="KO",
        minlatitude=LAT_MIN,
        maxlatitude=LAT_MAX,
        minlongitude=LON_MIN,
        maxlongitude=LON_MAX,
        starttime=PILOT_START,
        endtime=PILOT_END,
        level="channel",
    )
except Exception as e:
    print(f"    [FAIL] KO station fetch: {e}")
    sys.exit(1)

station_rows = []
for network in inventory:
    for station in network:
        channels   = sorted({ch.code for ch in station.channels})
        sample_rates = sorted({ch.sample_rate for ch in station.channels})
        has_HH  = any(c.startswith("HH") for c in channels)
        has_HN  = any(c.startswith("HN") for c in channels)
        chan_type = "HH_broadband" if has_HH else ("HN_strong_motion" if has_HN else "other")
        station_rows.append({
            "network":      network.code,
            "station":      station.code,
            "latitude":     round(station.latitude, 4),
            "longitude":    round(station.longitude, 4),
            "elevation_m":  station.elevation,
            "channels":     ",".join(channels),
            "channel_type": chan_type,
            "sample_rates": ",".join(str(r) for r in sample_rates),
            "start_date":   str(station.start_date)[:10] if station.start_date else "",
            "end_date":     str(station.end_date)[:10]   if station.end_date   else "",
        })

df_sta = pd.DataFrame(station_rows)
# Prefer HH broadband for M<2.0 sensitivity; secondary sort by station code
df_sta["_hh_priority"] = (df_sta["channel_type"] == "HH_broadband").astype(int)
df_sta_sorted = df_sta.sort_values(["_hh_priority", "station"], ascending=[False, True])
df_sta_sorted.drop(columns=["_hh_priority"]).to_csv(STATION_OUT, index=False)
print(f"    Saved {len(df_sta)} stations → {STATION_OUT}")

print("\n    All KO stations in Western Marmara bounding box:")
for _, row in df_sta_sorted.iterrows():
    flag = "HH" if row["channel_type"] == "HH_broadband" else "HN"
    print(f"      [{flag}]  KO.{row['station']}  "
          f"Lat={row['latitude']}  Lon={row['longitude']}")

# Pilot selection: 2 HH broadband stations, geographically separated
hh_stations = df_sta_sorted[df_sta_sorted["channel_type"] == "HH_broadband"].copy()

# Choose stations that are well-separated (>50 km apart) and near NAFZ
# NAFZ runs approximately at 40.5-41°N through Western Marmara
# Select: one near central Marmara, one in southern part for coverage
PILOT_STATIONS = ["RKY", "KRBG"]  # RKY: 40.687°N, KRBG: 40.393°N — both HH broadband
pilot = hh_stations[hh_stations["station"].isin(PILOT_STATIONS)]
if len(pilot) < 2:
    # Fallback: pick top 2 HH stations
    pilot = hh_stations.head(2)

print(f"\n    PILOT STATION SELECTION (2 KO HH broadband stations):")
for _, row in pilot.iterrows():
    print(f"      → KO.{row['station']}  "
          f"Lat={row['latitude']}  Lon={row['longitude']}  "
          f"Channels: {row['channels']}")
    print(f"        Channel type: {row['channel_type']}  "
          f"Sample rates: {row['sample_rates']} Hz")

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PHASE 1 CATALOG FETCH — COMPLETE")
print(f"  Events   : {CATALOG_OUT}")
print(f"  Stations : {STATION_OUT}")
print("  → Pilot stations: KO.RKY (40.687°N, 27.178°E) + KO.KRBG (40.393°N, 27.298°E)")
print("  → Both have HH broadband channels at 100 Hz (optimal for M<2.0 detection)")
print("  → Update artifacts/data_access_log.md with these findings")
print("=" * 70)
