# Phase: 2
# Purpose: Download targeted 60-second miniSEED windows for pilot events and paired noise windows
# Inputs: data/catalog/koeri_pilot_catalog.csv, KOERI-EIDA FDSN waveform service
# Outputs: data/raw/event_*.mseed, data/raw/noise_*.mseed, artifacts/waveform_inventory.csv
# Limitations: EMSC catalog contains only M>=2.0 events — no sub-threshold micro-seismic events;
#              noise windows are selected from inter-event gaps >300s, which may still contain
#              undetected micro-aftershocks; some windows may fail if KOERI-EIDA has gaps.

import time
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from obspy.clients.fdsn import Client
from obspy import UTCDateTime

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
RAW_DIR    = ROOT / "data" / "raw"
CATALOG    = ROOT / "data" / "catalog" / "koeri_pilot_catalog.csv"
ARTIFACT   = ROOT / "artifacts" / "waveform_inventory.csv"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ── Parameters ─────────────────────────────────────────────────────────────────
PILOT_STATIONS = [
    ("KO", "KOZT", "", "HH*"),   # 37.480°N, 35.826°E — ~70 km SW of M7.8 epicenter
    ("KO", "KMRS", "", "HH*"),   # 37.509°N, 36.900°E — ~15 km NNW of epicenter
]
WIN_BEFORE  = 30.0   # seconds before origin time
WIN_AFTER   = 30.0   # seconds after origin time
NOISE_OFFSET= 300.0  # noise window starts this many seconds before event window
MIN_GAP     = 300.0  # minimum inter-event gap required to accept a noise window (seconds)
PILOT_N     = 100    # total events to download (stratified sample)
FDSN_BASE   = "https://eida.koeri.boun.edu.tr"
RETRY_WAIT  = 5      # seconds between retries

# ── Load and sample catalog ────────────────────────────────────────────────────
print("=" * 70)
print("PHASE 2 — Waveform Download: Targeted Event + Noise Windows")
print(f"Stations   : {[f'{n}.{s}' for n,s,_,_ in PILOT_STATIONS]}")
print(f"Window     : {WIN_BEFORE}s before – {WIN_AFTER}s after origin time")
print(f"Pilot size : {PILOT_N} events (stratified sample)")
print("=" * 70)

random.seed(42)
df = pd.read_csv(CATALOG)
df["magnitude"] = pd.to_numeric(df["magnitude"], errors="coerce")
df["origin_time"] = pd.to_datetime(df["origin_time"], utc=True)
df = df.sort_values("origin_time").reset_index(drop=True)

# Compute inter-event time (gap before each event)
df["gap_before_s"] = df["origin_time"].diff().dt.total_seconds().fillna(9999)

# Stratified sample: aim for roughly 40 / 35 / 25 across M bands
bands = [
    (2.0, 3.0, 40),
    (3.0, 4.0, 35),
    (4.0, 99., 25),
]
sampled_ids = set()
sample_rows = []
for lo, hi, n in bands:
    sub = df[(df["magnitude"] >= lo) & (df["magnitude"] < hi)]
    chosen = sub.sample(min(n, len(sub)), random_state=42)
    sample_rows.append(chosen)
    sampled_ids.update(chosen.index.tolist())

pilot_df = pd.concat(sample_rows).sort_values("origin_time").reset_index(drop=True)

# Recompute gap for sampled events (against full catalog)
def gap_before(row, full_df):
    """Seconds between this event's window start and the previous catalogued event."""
    t_start = row["origin_time"] - pd.Timedelta(seconds=WIN_BEFORE)
    earlier = full_df[full_df["origin_time"] < t_start]
    if len(earlier) == 0:
        return 9999.0
    prev_t = earlier["origin_time"].max()
    return (t_start - prev_t).total_seconds()

print(f"\n[1] Stratified sample: {len(pilot_df)} events")
for lo, hi, _ in bands:
    n = ((pilot_df["magnitude"] >= lo) & (pilot_df["magnitude"] < hi)).sum()
    print(f"    M {lo:.1f}-{hi:.1f}: {n} events")

# ── Connect to FDSN ────────────────────────────────────────────────────────────
client = Client(FDSN_BASE)

# ── Download loop ──────────────────────────────────────────────────────────────
inventory_rows = []
n_event_ok  = 0
n_event_fail = 0
n_noise_ok  = 0
n_noise_fail = 0

print(f"\n[2] Downloading waveform windows...")
print(f"    (This will produce ~{PILOT_N * len(PILOT_STATIONS) * 2 * 72 // 1024} MB of data)")

for idx, row in pilot_df.iterrows():
    ev_t    = UTCDateTime(row["origin_time"].isoformat().replace("+00:00", "Z"))
    ev_id   = str(row["event_id"]).replace("/", "_").replace(":", "_")[-30:]
    ev_mag  = row["magnitude"]

    t_start = ev_t - WIN_BEFORE
    t_end   = ev_t + WIN_AFTER

    # Noise window: placed NOISE_OFFSET seconds before event window start
    n_start = t_start - NOISE_OFFSET - (WIN_BEFORE + WIN_AFTER)
    n_end   = n_start + WIN_BEFORE + WIN_AFTER

    # Check gap: ensure no catalogued event within MIN_GAP of noise window
    noise_ok = gap_before({"origin_time": pd.Timestamp(n_start.datetime, tz="UTC")}, df) >= MIN_GAP

    for (network, station, location, channel) in PILOT_STATIONS:
        sta_id = f"{network}.{station}"

        # ── Event window ────────────────────────────────────────────────────
        ev_file = RAW_DIR / f"event_{sta_id}_{ev_id}.mseed"
        ev_status = "skip_exists"
        if not ev_file.exists():
            try:
                st = client.get_waveforms(network, station, location, channel,
                                          t_start, t_end)
                st.write(str(ev_file), format="MSEED")
                ev_status = "ok"
                n_event_ok += 1
            except Exception as e:
                ev_status = f"fail: {str(e)[:60]}"
                n_event_fail += 1
                time.sleep(RETRY_WAIT)
        else:
            ev_status = "exists"
            n_event_ok += 1

        inventory_rows.append({
            "window_type": "event",
            "station":     sta_id,
            "event_id":    row["event_id"],
            "magnitude":   ev_mag,
            "t_start":     str(t_start),
            "t_end":       str(t_end),
            "file":        str(ev_file.name),
            "status":      ev_status,
        })

        # ── Noise window ────────────────────────────────────────────────────
        if noise_ok:
            ns_file = RAW_DIR / f"noise_{sta_id}_{ev_id}.mseed"
            ns_status = "skip_exists"
            if not ns_file.exists():
                try:
                    st = client.get_waveforms(network, station, location, channel,
                                              n_start, n_end)
                    st.write(str(ns_file), format="MSEED")
                    ns_status = "ok"
                    n_noise_ok += 1
                except Exception as e:
                    ns_status = f"fail: {str(e)[:60]}"
                    n_noise_fail += 1
                    time.sleep(RETRY_WAIT)
            else:
                ns_status = "exists"
                n_noise_ok += 1

            inventory_rows.append({
                "window_type": "noise",
                "station":     sta_id,
                "event_id":    row["event_id"],
                "magnitude":   ev_mag,
                "t_start":     str(n_start),
                "t_end":       str(n_end),
                "file":        str(ns_file.name),
                "status":      ns_status,
            })
        else:
            inventory_rows.append({
                "window_type": "noise",
                "station":     sta_id,
                "event_id":    row["event_id"],
                "magnitude":   ev_mag,
                "t_start":     str(n_start),
                "t_end":       str(n_end),
                "file":        "",
                "status":      "skip_gap_too_small",
            })

    # Progress every 10 events
    if (idx + 1) % 10 == 0:
        print(f"    Progress: {idx+1}/{len(pilot_df)} events  "
              f"ev_ok={n_event_ok} ev_fail={n_event_fail}  "
              f"ns_ok={n_noise_ok} ns_fail={n_noise_fail}")

# ── Save inventory ─────────────────────────────────────────────────────────────
inv_df = pd.DataFrame(inventory_rows)
inv_df.to_csv(ARTIFACT, index=False)

# ── Summary ────────────────────────────────────────────────────────────────────
raw_files = list(RAW_DIR.glob("*.mseed"))
total_mb = sum(f.stat().st_size for f in raw_files) / 1e6

print("\n" + "=" * 70)
print("PHASE 2 DOWNLOAD — COMPLETE")
print(f"  Event windows downloaded  : {n_event_ok}")
print(f"  Event windows failed      : {n_event_fail}")
print(f"  Noise windows downloaded  : {n_noise_ok}")
print(f"  Noise windows failed/skip : {n_noise_fail}")
print(f"  Total .mseed files        : {len(raw_files)}")
print(f"  Total raw data size       : {total_mb:.1f} MB")
print(f"  Inventory saved           : {ARTIFACT}")
print("=" * 70)
print("  → Run scripts/04_preprocess.py next")
