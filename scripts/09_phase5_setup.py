# Phase: 5
# Purpose: Select 12 KO HH broadband stations by distance bin; fetch 6-month EMSC catalog
#          for the Kahramanmaras aftershock sequence (2023-02-06 to 2023-08-06)
# Inputs: KOERI-EIDA FDSN station service, EMSC FDSN event service
# Outputs: data/catalog/phase5_catalog.csv, artifacts/phase5_station_list.csv
# Limitations: EMSC catalog threshold ~M2.0; sub-threshold events not included;
#              catalog retrieved in monthly chunks to avoid EMSC pagination limits;
#              6-month event count drops with Omori decay — later months have fewer events.

import time
import json
from pathlib import Path

import pandas as pd
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
from obspy.geodetics import locations2degrees, degrees2kilometers

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
CAT_DIR  = ROOT / "data" / "catalog"
ART_DIR  = ROOT / "artifacts"
CAT_DIR.mkdir(parents=True, exist_ok=True)
ART_DIR.mkdir(exist_ok=True)

CATALOG_OUT = CAT_DIR  / "phase5_catalog.csv"
STA_OUT     = ART_DIR  / "phase5_station_list.csv"

# ── Parameters ─────────────────────────────────────────────────────────────────
EPI_LAT, EPI_LON = 37.166, 37.032   # M7.8 main shock epicentre
PHASE5_START = UTCDateTime("2023-02-06T00:00:00")
PHASE5_END   = UTCDateTime("2023-08-06T00:00:00")
# Bounding box covering the full aftershock zone + station search radius
EV_LAT_MIN, EV_LAT_MAX = 36.0, 39.0
EV_LON_MIN, EV_LON_MAX = 35.0, 39.5
MIN_MAG = 1.5            # EMSC threshold — effective yield ~M2.0

# Target 12 stations: 2 near + 4 mid + 6 far
STATION_PLAN = {
    "near":  {"range": (0,   50),  "target": 2, "codes": ["GAZ",  "KMRS"]},
    "mid":   {"range": (50,  150), "target": 4, "codes": ["KOZT", "CEYT", "TAHT", "SARI"]},
    "far":   {"range": (150, 300), "target": 6, "codes": ["URFA", "DARE", "KARA", "BNN", "MERS", "SVRC"]},
}
# Alternates used if a primary station has >30% gap fraction
ALTERNATES = ["GULA", "DYBB", "YESY"]

KOERI_BASE = "https://eida.koeri.boun.edu.tr"

print("=" * 70)
print("PHASE 5 SETUP — Station Selection + 6-Month Catalog Fetch")
print(f"Epicentre: {EPI_LAT}°N, {EPI_LON}°E")
print(f"Catalog  : {PHASE5_START.date} → {PHASE5_END.date}")
print("=" * 70)

# ── 1. Station selection ───────────────────────────────────────────────────────
print("\n[1] Fetching KO HH station inventory for 300-km radius...")
sta_client = Client(KOERI_BASE)
inv = sta_client.get_stations(
    network="KO",
    minlatitude=34.5, maxlatitude=39.8,
    minlongitude=33.0, maxlongitude=41.0,
    starttime=PHASE5_START,
    endtime=PHASE5_END,
    level="channel",
)

sta_rows = []
for net in inv:
    for sta in net:
        chs = {ch.code for ch in sta.channels}
        if not any(c.startswith("HH") for c in chs):
            continue
        dist_deg = locations2degrees(EPI_LAT, EPI_LON, sta.latitude, sta.longitude)
        dist_km  = degrees2kilometers(dist_deg)
        if dist_km > 300:
            continue
        dist_bin = ("near" if dist_km < 50 else
                    "mid"  if dist_km < 150 else "far")
        codes = sorted({c for c in chs if c.startswith("HH")})
        is_primary  = sta.code in (STATION_PLAN["near"]["codes"] +
                                   STATION_PLAN["mid"]["codes"]  +
                                   STATION_PLAN["far"]["codes"])
        is_alt = sta.code in ALTERNATES
        sta_rows.append({
            "network":   net.code,
            "station":   sta.code,
            "latitude":  round(sta.latitude, 4),
            "longitude": round(sta.longitude, 4),
            "dist_km":   round(dist_km, 1),
            "dist_bin":  dist_bin,
            "hh_channels": ",".join(codes),
            "role":      "primary" if is_primary else ("alternate" if is_alt else "reserve"),
        })

sta_df = pd.DataFrame(sta_rows).sort_values("dist_km").reset_index(drop=True)
sta_df.to_csv(STA_OUT, index=False)
print(f"    Saved {len(sta_df)} KO HH stations within 300 km → {STA_OUT}")

primary = sta_df[sta_df["role"] == "primary"]
print(f"\n    12 PRIMARY pilot stations (distance-stratified):")
for _, r in primary.iterrows():
    print(f"      [{r['dist_bin']:4s}]  KO.{r['station']:6s}  "
          f"{r['latitude']:.3f}N  {r['longitude']:.3f}E  "
          f"{r['dist_km']:.0f} km  {r['hh_channels']}")

alternates = sta_df[sta_df["role"] == "alternate"]
print(f"\n    Alternates (used if primary station >30% gap fraction):")
for _, r in alternates.iterrows():
    print(f"      [alt]   KO.{r['station']:6s}  {r['dist_km']:.0f} km")

# ── 2. Catalog fetch — monthly chunks ─────────────────────────────────────────
print(f"\n[2] Fetching EMSC catalog in monthly chunks (avoids pagination limits)...")
ev_client = Client("EMSC")

# Monthly boundaries
months = []
t = PHASE5_START
while t < PHASE5_END:
    t_next = min(t + 31 * 86400, PHASE5_END)
    months.append((t, t_next))
    t = t_next

all_rows = []
for t_start, t_end in months:
    label = f"{t_start.date} → {t_end.date}"
    try:
        cat = ev_client.get_events(
            starttime=t_start, endtime=t_end,
            minlatitude=EV_LAT_MIN, maxlatitude=EV_LAT_MAX,
            minlongitude=EV_LON_MIN, maxlongitude=EV_LON_MAX,
            minmagnitude=MIN_MAG,
            orderby="time",
        )
        for ev in cat:
            o  = ev.preferred_origin() or (ev.origins[0] if ev.origins else None)
            mg = ev.preferred_magnitude() or (ev.magnitudes[0] if ev.magnitudes else None)
            if o is None:
                continue
            all_rows.append({
                "event_id":      str(ev.resource_id),
                "origin_time":   str(o.time),
                "latitude":      round(float(o.latitude),  4) if o.latitude  else "",
                "longitude":     round(float(o.longitude), 4) if o.longitude else "",
                "depth_km":      round(float(o.depth)/1000, 2) if o.depth else "",
                "magnitude":     round(float(mg.mag), 2) if mg else "",
                "mag_type":      mg.magnitude_type if mg else "",
                "catalog_source":"EMSC_FDSN",
            })
        print(f"    {label}: {len(cat)} events")
        time.sleep(1)
    except Exception as e:
        print(f"    {label}: FAIL — {e}")

cat_df = pd.DataFrame(all_rows)
cat_df["magnitude"] = pd.to_numeric(cat_df["magnitude"], errors="coerce")
cat_df.to_csv(CATALOG_OUT, index=False)
print(f"\n    Total events fetched: {len(cat_df)}")
print(f"    Saved → {CATALOG_OUT}")

print("\n    Magnitude distribution:")
for lo, hi, label in [(2,3,"M 2.0-3.0"),(3,4,"M 3.0-4.0"),(4,99,"M>=4.0")]:
    n = ((cat_df.magnitude>=lo)&(cat_df.magnitude<hi)).sum()
    print(f"      {label}: {n} events")

# ── Volume estimate ────────────────────────────────────────────────────────────
n_ev = len(cat_df)
n_sta = len(primary)
est_mb = n_ev * n_sta * 30 / 1024   # KB → MB (30 KB per compressed miniSEED window)
est_gb = est_mb / 1024               # MB → GB
print(f"\n    Download volume estimate (all events, {n_sta} stations):")
print(f"      {n_ev} events × {n_sta} stations × ~30 KB = {est_mb:.0f} MB ({est_gb:.1f} GB)")
if est_gb > 8.0:
    max_events = int(8 * 1024 * 1024 / (n_sta * 30))
    print(f"      WARNING: Exceeds 8 GB limit. Will cap download at {max_events} events.")
    print(f"      Strategy: keep all M>=4 + stratified sample from M2-4.")
else:
    print(f"      Within 8 GB limit. Downloading all {n_ev} events.")

print("\n" + "=" * 70)
print("PHASE 5 SETUP — COMPLETE")
print(f"  Station list: {STA_OUT}")
print(f"  Catalog     : {CATALOG_OUT}")
print("  → Run scripts/10_phase5_download.py next")
print("=" * 70)
