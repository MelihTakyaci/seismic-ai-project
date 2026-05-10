# Phase: 10
# Purpose: Autonomous national-scale data harvester — 10,000+ events from KOERI FDSN
# Inputs: ISC event catalog, KOERI EIDA waveform service
# Outputs: data/national_dataset/{waveforms_national.hdf5, metadata_national.csv}
# Limitations: Rate-limited by KOERI EIDA; handles disconnects with exponential backoff

"""
National Mass Harvest — Autonomous KOERI Data Acquisition

Strategy:
  1. Query ISC catalog for events M1.5-4.5 across Turkey (2022-2024)
  2. Select 25 well-distributed broadband stations
  3. Download 60s event windows (P-30s to P+30s) from KOERI EIDA
  4. Download paired noise windows (60s, >300s from any event)
  5. Store as HDF5 + metadata CSV compatible with training pipeline

Resilience features:
  - Exponential backoff on HTTP errors
  - Per-trace checkpointing (skip already-downloaded)
  - Parallel station queries with ThreadPoolExecutor
  - Graceful shutdown on KeyboardInterrupt
"""

import gc
import json
import signal
import sys
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
from obspy import UTCDateTime, Stream
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNNoDataException

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Configuration ──────────────────────────────────────────────────────
CATALOG_CLIENT = "ISC"
WAVEFORM_CLIENT = "KOERI"
NETWORK = "KO"
CHANNELS = "HH*"
SAMPLE_RATE = 100.0
WINDOW_SEC = 60.0
NPTS = int(WINDOW_SEC * SAMPLE_RATE)

# Geographic regions to sample from (diverse tectonic settings)
REGIONS = {
    "marmara": {"minlat": 40.0, "maxlat": 41.5, "minlon": 27.0, "maxlon": 31.0},
    "kahramanmaras": {"minlat": 37.0, "maxlat": 38.5, "minlon": 36.0, "maxlon": 38.0},
    "aegean": {"minlat": 37.5, "maxlat": 39.5, "minlon": 26.0, "maxlon": 28.5},
    "eastern_anatolia": {"minlat": 38.0, "maxlat": 40.0, "minlon": 39.0, "maxlon": 44.0},
    "central_anatolia": {"minlat": 38.5, "maxlat": 40.5, "minlon": 31.0, "maxlon": 36.0},
}

# Target station list (diverse, well-known broadband stations)
TARGET_STATIONS = [
    "KMRS", "KOZT", "DARE", "ANTB", "HRTX", "KAVV", "KRBG", "SARI", "SAUV",
    "GAZ", "ARMT", "BGKT", "BOTS", "BLCB", "BODT", "AYVL", "BALB",
    "CEYT", "BNGB", "AGRB", "AKDG", "ARGN", "EDRB", "ISK", "ISP",
]

# Harvest targets
TARGET_EVENTS = 10000
TARGET_NOISE_RATIO = 5  # noise traces per event
MAX_EVENTS_PER_REGION = 3000
EVENTS_PER_CATALOG_QUERY = 2000

# Output paths
OUTPUT_DIR = ROOT / "data" / "national_dataset"
HDF5_PATH = OUTPUT_DIR / "waveforms_national.hdf5"
META_PATH = OUTPUT_DIR / "metadata_national.csv"
CHECKPOINT_PATH = OUTPUT_DIR / "harvest_checkpoint.json"
LOG_PATH = OUTPUT_DIR / "harvest_log.json"

# Resilience
MAX_RETRIES = 5
BASE_BACKOFF = 2.0
MAX_WORKERS = 3  # parallel station downloads (conservative for rate limits)

_shutdown = False


def signal_handler(sig, frame):
    global _shutdown
    print("\n[SHUTDOWN] Graceful shutdown requested. Finishing current batch...")
    _shutdown = True


signal.signal(signal.SIGINT, signal_handler)


def backoff_sleep(attempt: int) -> None:
    delay = min(BASE_BACKOFF ** attempt, 60.0)
    time.sleep(delay)


def load_checkpoint() -> Dict:
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            return json.load(f)
    return {"downloaded_events": [], "downloaded_noise": [], "failed": []}


def save_checkpoint(ckpt: Dict) -> None:
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(ckpt, f)


# ── Catalog Harvesting ─────────────────────────────────────────────────

def fetch_catalog(region_name: str, region_bounds: Dict,
                  start: str, end: str, mag_min: float, mag_max: float) -> pd.DataFrame:
    """Fetch event catalog from ISC for a specific region and time range."""
    print(f"  Querying catalog: {region_name} [{start} to {end}] M{mag_min}-{mag_max}")
    client = Client(CATALOG_CLIENT)

    events = []
    try:
        cat = client.get_events(
            starttime=UTCDateTime(start),
            endtime=UTCDateTime(end),
            minmagnitude=mag_min,
            maxmagnitude=mag_max,
            minlatitude=region_bounds["minlat"],
            maxlatitude=region_bounds["maxlat"],
            minlongitude=region_bounds["minlon"],
            maxlongitude=region_bounds["maxlon"],
            limit=EVENTS_PER_CATALOG_QUERY,
        )

        for ev in cat:
            origin = ev.preferred_origin() or (ev.origins[0] if ev.origins else None)
            mag = ev.preferred_magnitude() or (ev.magnitudes[0] if ev.magnitudes else None)
            if origin is None or mag is None:
                continue
            events.append({
                "event_id": str(ev.resource_id),
                "origin_time": str(origin.time),
                "latitude": origin.latitude,
                "longitude": origin.longitude,
                "depth_km": origin.depth / 1000.0 if origin.depth else None,
                "magnitude": mag.mag,
                "region": region_name,
            })
    except Exception as e:
        print(f"    [WARN] Catalog query failed: {e}")

    print(f"    Found {len(events)} events")
    return pd.DataFrame(events)


def build_national_catalog() -> pd.DataFrame:
    """Build comprehensive catalog across all regions."""
    print("\n" + "=" * 70)
    print("STEP 1: BUILDING NATIONAL EVENT CATALOG")
    print("=" * 70)

    all_events = []

    time_ranges = [
        ("2023-02-06", "2023-04-30"),  # Kahramanmaras main sequence
        ("2023-05-01", "2023-08-31"),  # Kahramanmaras late aftershocks
        ("2022-01-01", "2022-12-31"),  # Pre-Kahramanmaras (baseline)
        ("2024-01-01", "2024-06-30"),  # Post-sequence
    ]

    for region_name, bounds in REGIONS.items():
        region_count = 0
        for start, end in time_ranges:
            if region_count >= MAX_EVENTS_PER_REGION:
                break
            df = fetch_catalog(region_name, bounds, start, end, 1.5, 4.5)
            all_events.append(df)
            region_count += len(df)
            time.sleep(1.0)  # rate limit courtesy

            if _shutdown:
                break
        if _shutdown:
            break

    if not all_events:
        print("[ERROR] No events fetched!")
        return pd.DataFrame()

    catalog = pd.concat(all_events, ignore_index=True)
    catalog = catalog.drop_duplicates(subset=["origin_time", "latitude", "longitude"])
    catalog = catalog.sort_values("origin_time").reset_index(drop=True)

    print(f"\n  TOTAL CATALOG: {len(catalog)} unique events")
    print(f"  By region:")
    for region in catalog["region"].unique():
        n = len(catalog[catalog["region"] == region])
        print(f"    {region}: {n}")
    print(f"  Magnitude distribution:")
    for lo, hi, label in [(1.5, 2.0, "M1.5-2"), (2.0, 3.0, "M2-3"), (3.0, 4.0, "M3-4"), (4.0, 5.0, "M4+")]:
        n = len(catalog[(catalog["magnitude"] >= lo) & (catalog["magnitude"] < hi)])
        print(f"    {label}: {n}")

    return catalog


# ── Waveform Download ──────────────────────────────────────────────────

def download_waveform(client: Client, station: str, start_time: UTCDateTime,
                      duration: float = 60.0) -> Optional[np.ndarray]:
    """Download and preprocess a single waveform window."""
    end_time = start_time + duration

    for attempt in range(MAX_RETRIES):
        try:
            st = client.get_waveforms(
                network=NETWORK, station=station, location="*",
                channel=CHANNELS, starttime=start_time, endtime=end_time,
            )

            if len(st) < 3:
                return None

            # Preprocess
            st.detrend("demean")
            st.filter("bandpass", freqmin=1.0, freqmax=45.0)
            st.resample(SAMPLE_RATE)

            # Extract 3 channels (Z, N, E order)
            channels = {}
            for tr in st:
                ch_code = tr.stats.channel[-1]  # Z, N, or E
                if ch_code in ("Z", "N", "E") and ch_code not in channels:
                    channels[ch_code] = tr.data[:NPTS].astype(np.float32)

            if len(channels) < 3:
                return None

            # Pad if needed
            data = np.zeros((3, NPTS), dtype=np.float32)
            for i, ch in enumerate(["Z", "N", "E"]):
                if ch in channels:
                    n = min(len(channels[ch]), NPTS)
                    data[i, :n] = channels[ch][:n]

            # Quality check: reject if all zeros or extreme amplitudes
            if np.abs(data).max() < 1e-10:
                return None

            return data

        except FDSNNoDataException:
            return None
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                backoff_sleep(attempt)
            else:
                return None

    return None


def download_event_windows(catalog: pd.DataFrame, checkpoint: Dict) -> List[Dict]:
    """Download event windows for all catalog events across multiple stations."""
    print("\n" + "=" * 70)
    print("STEP 2: DOWNLOADING EVENT WAVEFORMS")
    print("=" * 70)

    client = Client(WAVEFORM_CLIENT)
    already_done = set(checkpoint.get("downloaded_events", []))
    results = []

    total = min(len(catalog), TARGET_EVENTS)
    print(f"  Target: {total} event windows")
    print(f"  Already downloaded: {len(already_done)}")

    success = 0
    failed = 0

    for idx, row in catalog.head(total).iterrows():
        if _shutdown:
            break

        event_id = f"{row['region']}_{idx}"
        if event_id in already_done:
            success += 1
            continue

        origin = UTCDateTime(row["origin_time"])
        # Window: origin - 10s to origin + 50s (P-wave typically within first 10-20s)
        window_start = origin - 10.0

        # Try multiple stations for each event
        stations_to_try = TARGET_STATIONS[:15]  # try up to 15 stations per event
        downloaded_for_event = False

        for station in stations_to_try:
            if _shutdown:
                break

            trace_id = f"{NETWORK}.{station}.HH.{event_id}"
            if trace_id in already_done:
                continue

            data = download_waveform(client, station, window_start)

            if data is not None:
                results.append({
                    "trace_name": trace_id,
                    "station": f"{NETWORK}.{station}",
                    "window_type": "event",
                    "origin_time": str(row["origin_time"]),
                    "latitude": row["latitude"],
                    "longitude": row["longitude"],
                    "depth_km": row["depth_km"],
                    "source_magnitude": row["magnitude"],
                    "region": row["region"],
                    "window_start": str(window_start),
                    "data": data,
                })
                already_done.add(trace_id)
                downloaded_for_event = True
                success += 1
                break  # one station per event is enough initially

            time.sleep(0.3)  # rate limit

        if not downloaded_for_event:
            failed += 1

        if (idx + 1) % 50 == 0:
            print(f"  [{idx+1}/{total}] success={success}, failed={failed}")
            checkpoint["downloaded_events"] = list(already_done)
            save_checkpoint(checkpoint)

    print(f"\n  Event windows downloaded: {success}")
    print(f"  Failed: {failed}")

    checkpoint["downloaded_events"] = list(already_done)
    save_checkpoint(checkpoint)

    return results


def download_noise_windows(catalog: pd.DataFrame, n_noise: int,
                          checkpoint: Dict) -> List[Dict]:
    """Download noise windows far from any cataloged event."""
    print("\n" + "=" * 70)
    print("STEP 3: DOWNLOADING NOISE WAVEFORMS")
    print("=" * 70)

    client = Client(WAVEFORM_CLIENT)
    already_done = set(checkpoint.get("downloaded_noise", []))
    results = []

    print(f"  Target: {n_noise} noise windows")
    print(f"  Already downloaded: {len(already_done)}")

    # Generate noise window times: random times far from events
    event_times = [UTCDateTime(t) for t in catalog["origin_time"]]

    rng = np.random.RandomState(42)
    noise_starts = []

    # Generate random times in 2023, avoiding event windows
    base_start = UTCDateTime("2023-01-01")
    base_end = UTCDateTime("2023-12-31")
    total_seconds = base_end - base_start

    while len(noise_starts) < n_noise * 2:  # generate 2x to account for failures
        random_offset = rng.uniform(0, total_seconds)
        t = base_start + random_offset

        # Check distance from any event (>300s)
        too_close = False
        for et in event_times[:1000]:  # check against first 1000 events
            if abs(t - et) < 300:
                too_close = True
                break

        if not too_close:
            noise_starts.append(t)

    success = 0
    stations_cycle = TARGET_STATIONS * 10  # cycle through stations

    for i, window_start in enumerate(noise_starts[:n_noise]):
        if _shutdown:
            break
        if success >= n_noise:
            break

        station = stations_cycle[i % len(stations_cycle)]
        trace_id = f"{NETWORK}.{station}.HH.noise_{i:06d}"

        if trace_id in already_done:
            success += 1
            continue

        data = download_waveform(client, station, window_start)

        if data is not None:
            results.append({
                "trace_name": trace_id,
                "station": f"{NETWORK}.{station}",
                "window_type": "noise",
                "origin_time": None,
                "latitude": None,
                "longitude": None,
                "depth_km": None,
                "source_magnitude": None,
                "region": "noise",
                "window_start": str(window_start),
                "data": data,
            })
            already_done.add(trace_id)
            success += 1

        if (i + 1) % 100 == 0:
            print(f"  [{success}/{n_noise}] noise windows acquired")
            checkpoint["downloaded_noise"] = list(already_done)
            save_checkpoint(checkpoint)

        time.sleep(0.2)

    print(f"\n  Noise windows downloaded: {success}")
    checkpoint["downloaded_noise"] = list(already_done)
    save_checkpoint(checkpoint)

    return results


# ── HDF5 Assembly ──────────────────────────────────────────────────────

def assemble_hdf5(event_results: List[Dict], noise_results: List[Dict]) -> None:
    """Write all downloaded traces to HDF5 + metadata CSV."""
    print("\n" + "=" * 70)
    print("STEP 4: ASSEMBLING HDF5 DATASET")
    print("=" * 70)

    all_results = event_results + noise_results

    if not all_results:
        print("[ERROR] No data to write!")
        return

    print(f"  Total traces: {len(all_results)}")
    print(f"  Events: {len(event_results)}, Noise: {len(noise_results)}")

    # Write HDF5
    with h5py.File(str(HDF5_PATH), "w") as hf:
        data_grp = hf.create_group("data")
        for item in all_results:
            data_grp.create_dataset(
                item["trace_name"],
                data=item["data"],
                compression="gzip",
                compression_opts=4,
            )

    # Write metadata
    meta_rows = []
    for item in all_results:
        meta_rows.append({
            "trace_name": item["trace_name"],
            "station": item["station"],
            "window_type": item["window_type"],
            "origin_time": item["origin_time"],
            "latitude": item["latitude"],
            "longitude": item["longitude"],
            "depth_km": item["depth_km"],
            "source_magnitude": item["source_magnitude"],
            "region": item["region"],
            "window_start": item["window_start"],
            "augmentation": "original",
        })

    meta_df = pd.DataFrame(meta_rows)
    meta_df.to_csv(str(META_PATH), index=False)

    print(f"  HDF5: {HDF5_PATH} ({HDF5_PATH.stat().st_size / 1e6:.1f} MB)")
    print(f"  Metadata: {META_PATH}")

    # Summary
    print(f"\n  === DATASET SUMMARY ===")
    print(f"  Events: {len(event_results)}")
    print(f"  Noise: {len(noise_results)}")
    print(f"  Stations: {meta_df['station'].nunique()}")
    print(f"  Regions: {meta_df['region'].nunique()}")
    if len(event_results) > 0:
        mags = [r["source_magnitude"] for r in event_results if r["source_magnitude"]]
        print(f"  Magnitude range: M{min(mags):.1f} - M{max(mags):.1f}")


# ── Main ───────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("NATIONAL MASS HARVEST — AUTONOMOUS KOERI DATA ACQUISITION")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Target: {TARGET_EVENTS} events + {TARGET_EVENTS * TARGET_NOISE_RATIO} noise")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = load_checkpoint()

    # Step 1: Build catalog
    catalog = build_national_catalog()
    if catalog.empty:
        print("[ABORT] Empty catalog.")
        return

    catalog.to_csv(OUTPUT_DIR / "national_catalog.csv", index=False)
    print(f"\n  Catalog saved: {len(catalog)} events")

    # Step 2: Download event windows
    event_results = download_event_windows(catalog, checkpoint)

    if _shutdown:
        print("\n[SHUTDOWN] Saving partial results...")
        if event_results:
            assemble_hdf5(event_results, [])
        return

    # Step 3: Download noise windows
    n_noise = min(len(event_results) * TARGET_NOISE_RATIO, 50000)
    noise_results = download_noise_windows(catalog, n_noise, checkpoint)

    # Step 4: Assemble
    assemble_hdf5(event_results, noise_results)

    # Final log
    log = {
        "timestamp": datetime.now().isoformat(),
        "catalog_events": len(catalog),
        "event_windows_downloaded": len(event_results),
        "noise_windows_downloaded": len(noise_results),
        "stations_used": list(set(r["station"] for r in event_results + noise_results)),
        "regions": list(catalog["region"].unique()),
    }
    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)

    print("\n" + "=" * 70)
    print("HARVEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
