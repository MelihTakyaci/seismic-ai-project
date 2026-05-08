# Phase: Fine-tuning / Evaluation
# Purpose: Fetch real analyst-reviewed P/S phase picks from KOERI or EMSC FDSN event
#          service (includearrivals=true). Match picks to our augmented_dataset windows
#          by event_id + station. Report coverage vs TauPy theoretical picks.
#          If real picks are denser: re-labelling recommendation is printed.
# Inputs:  data/augmented_dataset/metadata.csv, data/catalog/phase5_catalog.csv
#          (FDSN endpoint queried live)
# Outputs: artifacts/koeri_real_picks.csv
# Limitations: KOERI EIDA may not expose /fdsnws/event/ with phase arrivals.
#              EMSC provides picks for M≥2.5 events only; sub-threshold events missing.
#              Matching uses event_id→origin-time lookup with 10 s tolerance.
#              Station code matching uses bare station name (no network qualifier).

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNException

ROOT    = Path(__file__).resolve().parent.parent
ART     = ROOT / "artifacts"
CAT_F   = ROOT / "data" / "catalog" / "phase5_catalog.csv"
META_F  = ROOT / "data" / "augmented_dataset" / "metadata.csv"

WIN_BEFORE = 30.0   # seconds before origin in our 6000-sample windows
SRATE      = 100.0

# Time range of Phase 5 data
T_START = UTCDateTime("2023-02-06")
T_END   = UTCDateTime("2023-08-06")

MATCH_TOL_S  = 10.0   # max seconds difference between catalog and FDSN origin times

print("=" * 70)
print("PHASE PICK FETCH — KOERI / EMSC real analyst picks")
print("=" * 70)

# ── Step 1: Load our catalog and metadata ──────────────────────────────────────
print("\n[1] Loading internal catalog and metadata...")

cat_df = pd.read_csv(CAT_F)
cat_df["origin_time"] = pd.to_datetime(cat_df["origin_time"], utc=True)
print(f"    Catalog events: {len(cat_df)}")

meta = pd.read_csv(META_F)
meta["trace_p_arrival_sample"] = pd.to_numeric(meta["trace_p_arrival_sample"], errors="coerce")
orig_meta = meta[meta.get("augmentation", "original") == "original"].copy() if "augmentation" in meta.columns else meta.copy()
print(f"    Augmented dataset (original windows): {len(orig_meta)}")
print(f"    Unique event IDs: {orig_meta['event_id'].nunique()}")

# Build event_id → origin_time lookup
event_lookup = {}
for _, row in cat_df.iterrows():
    event_lookup[str(row["event_id"])] = row["origin_time"]

# ── Step 2: Try FDSN endpoints for phase arrivals ─────────────────────────────
print("\n[2] Querying FDSN event service for phase arrivals...")

ENDPOINTS = [
    ("KOERI-EIDA", "https://eida.koeri.boun.edu.tr"),
    ("EMSC",       "EMSC"),
    ("IRIS",       "IRIS"),
]

catalog = None
source_used = None

for name, base_url in ENDPOINTS:
    print(f"\n    Trying {name} ({base_url})...")
    try:
        client = Client(base_url, timeout=60)
        catalog = client.get_events(
            starttime=T_START,
            endtime=T_END,
            minmagnitude=2.0,
            maxmagnitude=9.0,
            minlatitude=36.0, maxlatitude=39.0,
            minlongitude=35.0, maxlongitude=40.0,
            includearrivals=True,
        )
        n_ev  = len(catalog)
        n_arr = sum(len(e.picks) for e in catalog)
        print(f"    ✓ {name}: {n_ev} events, {n_arr} picks")
        source_used = name
        break

    except FDSNException as e:
        print(f"    ✗ {name} FDSNException: {str(e)[:120]}")
    except Exception as e:
        print(f"    ✗ {name} Error: {str(e)[:120]}")
        time.sleep(2)

if catalog is None:
    print("\n  [!] All FDSN endpoints failed for includearrivals=True.")
    print("      Trying EMSC without arrivals (origin-times only as fallback)...")
    try:
        client  = Client("EMSC", timeout=60)
        catalog = client.get_events(
            starttime=T_START, endtime=T_END,
            minmagnitude=2.0,
            minlatitude=36.0, maxlatitude=39.0,
            minlongitude=35.0, maxlongitude=40.0,
            includearrivals=False,
        )
        print(f"      EMSC (no arrivals): {len(catalog)} events")
        source_used = "EMSC-no-arrivals"
    except Exception as e:
        print(f"      EMSC also failed: {e}")
        print("\n  Cannot fetch picks. Writing empty CSV and exiting.")
        pd.DataFrame(columns=[
            "event_id", "station", "phase", "pick_time_utc",
            "p_sample_in_window", "source"
        ]).to_csv(ART / "koeri_real_picks.csv", index=False)
        sys.exit(0)

# ── Step 3: Extract P/S picks from catalog ────────────────────────────────────
print(f"\n[3] Extracting picks (source: {source_used})...")

if source_used == "EMSC-no-arrivals":
    print("    No phase picks available — writing empty CSV.")
    pick_rows = []
else:
    # Build a lookup: ObsPy event resource_id → our catalog event_id
    def find_our_event(fdsn_origin_time, tol_s=MATCH_TOL_S):
        """Match an FDSN event origin time to our catalog by time proximity."""
        t_fdsn = pd.Timestamp(fdsn_origin_time.datetime, tz="UTC")
        for ev_id, t_cat in event_lookup.items():
            diff = abs((t_fdsn - t_cat).total_seconds())
            if diff <= tol_s:
                return ev_id
        return None

    pick_rows = []
    n_matched_events = 0
    n_unmatched      = 0

    for event in catalog:
        origin = event.preferred_origin() or (event.origins[0] if event.origins else None)
        if origin is None:
            continue

        our_ev_id = find_our_event(origin.time)
        if our_ev_id is None:
            n_unmatched += 1
            continue
        n_matched_events += 1

        # Build pick-id → pick dict for fast lookup
        pick_by_id = {str(p.resource_id): p for p in event.picks}

        # Walk arrivals to get phase label (more reliable than pick.phase_hint)
        arrivals_seen = set()
        for arr in (origin.arrivals or []):
            if arr.phase not in ("P", "S", "Pg", "Sg", "Pn", "Sn"):
                continue
            phase = "P" if arr.phase.startswith("P") else "S"
            pid   = str(arr.pick_id)
            if pid in arrivals_seen:
                continue
            arrivals_seen.add(pid)

            pick = pick_by_id.get(pid)
            if pick is None:
                continue

            sta  = pick.waveform_id.station_code if pick.waveform_id else None
            if sta is None:
                continue

            pick_t = pick.time

            # P-sample position in our 6000-sample window
            # Window start = event_origin_time - WIN_BEFORE
            origin_t_pd = pd.Timestamp(origin.time.datetime, tz="UTC")
            ev_data = cat_df[cat_df["event_id"] == our_ev_id]
            if len(ev_data) == 0:
                continue
            origin_actual = ev_data.iloc[0]["origin_time"]
            win_start_t   = origin_actual - pd.Timedelta(seconds=WIN_BEFORE)
            pick_t_pd     = pd.Timestamp(pick_t.datetime, tz="UTC")
            delta_s       = (pick_t_pd - win_start_t).total_seconds()
            p_sample      = int(round(delta_s * SRATE))

            pick_rows.append({
                "event_id":            our_ev_id,
                "station":             sta,
                "phase":               phase,
                "pick_time_utc":       str(pick_t),
                "p_sample_in_window":  p_sample if 0 <= p_sample < 6000 else np.nan,
                "source":              source_used,
            })

    print(f"    FDSN events processed   : {len(catalog)}")
    print(f"    Matched to our catalog  : {n_matched_events}")
    print(f"    Unmatched (time mismatch): {n_unmatched}")
    print(f"    Total picks extracted   : {len(pick_rows)}")

# ── Step 4: Match picks to our windows + compare with TauPy ───────────────────
print("\n[4] Matching picks to augmented_dataset windows...")

picks_df = pd.DataFrame(pick_rows) if pick_rows else pd.DataFrame(columns=[
    "event_id", "station", "phase", "pick_time_utc",
    "p_sample_in_window", "source"
])
picks_df.to_csv(ART / "koeri_real_picks.csv", index=False)
print(f"    Written: artifacts/koeri_real_picks.csv ({len(picks_df)} rows)")

if len(picks_df) > 0:
    p_picks  = picks_df[picks_df["phase"] == "P"]
    s_picks  = picks_df[picks_df["phase"] == "S"]
    print(f"    P picks: {len(p_picks)}  |  S picks: {len(s_picks)}")
    print(f"    Unique events with picks: {picks_df['event_id'].nunique()}")
    print(f"    Unique stations with picks: {picks_df['station'].nunique()}")

    # Match to our windows
    # Metadata station format: "KO.KOZT" (network.station)
    # EMSC picks station format: "KOZT" (bare station code)
    orig_meta_with_ev = orig_meta[orig_meta["event_id"].notna()].copy()
    orig_meta_with_ev["event_id"]  = orig_meta_with_ev["event_id"].astype(str)
    orig_meta_with_ev["sta_bare"]  = orig_meta_with_ev["station"].str.split(".").str[-1]
    picks_df["event_id"]           = picks_df["event_id"].astype(str)

    merged = orig_meta_with_ev.merge(
        p_picks[["event_id", "station", "p_sample_in_window", "source"]],
        left_on=["event_id", "sta_bare"],
        right_on=["event_id", "station"],
        how="left",
        suffixes=("_taupy", "_real"),
    )

    has_real  = merged["p_sample_in_window"].notna().sum()
    has_taupy = merged["trace_p_arrival_sample"].notna().sum()
    print(f"\n    Windows in augmented_dataset : {len(merged)}")
    print(f"    With TauPy P label           : {has_taupy}")
    print(f"    With real analyst P pick     : {has_real}")
    print(f"    Coverage improvement         : {has_taupy} → {has_real} "
          f"({'+' if has_real>=has_taupy else ''}{has_real-has_taupy:+d})")

    # MAE between TauPy and real picks (where both exist)
    both = merged[
        merged["p_sample_in_window"].notna() &
        merged["trace_p_arrival_sample"].notna()
    ].copy()
    if len(both) > 0:
        mae_samp = (both["p_sample_in_window"] - both["trace_p_arrival_sample"]).abs()
        mae_s    = mae_samp / SRATE
        print(f"\n    TauPy vs real pick MAE: {mae_s.mean():.3f}s  "
              f"(median {mae_s.median():.3f}s, max {mae_s.max():.3f}s)")
        print(f"    → {'TauPy is adequate as labels (MAE < 0.5s)' if mae_s.median() < 0.5 else 'Real picks significantly differ from TauPy — re-labelling recommended'}")

# ── Step 5: Summary ────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("KOERI REAL PICKS — COMPLETE")
print(f"  Source used : {source_used}")
print(f"  Output      : artifacts/koeri_real_picks.csv ({len(picks_df)} rows)")
if len(picks_df) == 0:
    print("  [!] No real picks retrieved.")
    print("      Possible causes:")
    print("      1. KOERI has no /fdsnws/event/ endpoint (Phase 1 finding)")
    print("      2. EMSC does not return arrivals for this bbox/period")
    print("      3. Network timeout — retry with a shorter time window")
    print("  Recommendation: TauPy theoretical picks remain the best available")
    print("  reference for this dataset. Consider AFAD web service as fallback.")
print("=" * 70)
