# Phase: 5
# Purpose: Download event-targeted waveform windows for 12 KO stations over 6-month catalog.
#          Single persistent FDSN client (fixes L9 race condition), sequential with
#          exponential backoff retry (3 attempts: 5 s, 15 s, 30 s), fully resumable.
# Inputs:  data/catalog/phase5_catalog.csv, artifacts/phase5_station_list.csv,
#          data/catalog/phase5_download_progress.json (if exists — resumes from here)
# Outputs: data/raw/phase5/*.mseed, data/catalog/phase5_download_progress.json,
#          artifacts/phase5_waveform_inventory.csv
# Limitations: Sequential download avoids the thread-local client race condition (L9)
#              but is slower: ~1–2 s/request → estimate 4–8 h for ~7 k remaining pairs.
#              Progress is saved every SAVE_EVERY successful downloads and on interrupt.
#              Rate limiting: SLEEP_BETWEEN_REQUESTS s between each request.

import json
import time
import signal
import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from obspy.clients.fdsn import Client
from obspy import UTCDateTime

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent.parent
RAW_DIR       = ROOT / "data" / "raw" / "phase5"
CAT_FILE      = ROOT / "data" / "catalog" / "phase5_catalog.csv"
STA_FILE      = ROOT / "artifacts" / "phase5_station_list.csv"
PROGRESS_FILE = ROOT / "data" / "catalog" / "phase5_download_progress.json"
INV_FILE      = ROOT / "artifacts" / "phase5_waveform_inventory.csv"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ── Parameters ─────────────────────────────────────────────────────────────────
WIN_BEFORE              = 30.0   # seconds before origin
WIN_AFTER               = 30.0   # seconds after origin
RETRY_WAITS             = [5, 15, 30]   # seconds between retry attempts (3 total)
SLEEP_BETWEEN_REQUESTS  = 0.4    # polite delay between requests (s)
SAVE_EVERY              = 50     # save progress file every N successful downloads
FDSN_BASE               = "https://eida.koeri.boun.edu.tr"

print("=" * 70)
print("PHASE 5 — Sequential Waveform Download (Single Client, Resumable)")
print(f"  FDSN     : {FDSN_BASE}")
print(f"  Window   : -{WIN_BEFORE}s / +{WIN_AFTER}s around origin")
print(f"  Retry    : {len(RETRY_WAITS)} attempts, waits {RETRY_WAITS} s")
print(f"  Rate     : {SLEEP_BETWEEN_REQUESTS} s between requests")
print("=" * 70)

# ── Load catalog ───────────────────────────────────────────────────────────────
print("\n[1] Loading catalog and station list...")
cat_df = pd.read_csv(CAT_FILE)
cat_df["magnitude"]   = pd.to_numeric(cat_df["magnitude"],   errors="coerce")
cat_df["origin_time"] = pd.to_datetime(cat_df["origin_time"], utc=True)
cat_df = cat_df.dropna(subset=["latitude", "longitude", "magnitude"]).reset_index(drop=True)

sta_df       = pd.read_csv(STA_FILE)
primary_stas = sta_df[sta_df["role"] == "primary"]["station"].tolist()
active_stas  = [f"KO.{s}" for s in primary_stas]

print(f"    Catalog : {len(cat_df)} events")
print(f"    Stations: {active_stas}")

# ── Load / initialise progress file ───────────────────────────────────────────
print("\n[2] Loading progress file...")
if PROGRESS_FILE.exists():
    with open(PROGRESS_FILE) as f:
        progress = json.load(f)
    n_ok   = sum(1 for v in progress["completed"].values() if v in ("ok", "exists"))
    n_fail = sum(1 for v in progress["completed"].values() if v.startswith("fail"))
    print(f"    Existing: {len(progress['completed'])} entries — {n_ok} ok, {n_fail} fail")
    print(f"    Will retry all failed entries with single client.")
else:
    progress = {
        "started_at":  datetime.now(timezone.utc).isoformat(),
        "completed":   {},
        "gap_counts":  {s: {"ok": 0, "gap": 0} for s in active_stas},
        "replaced_stations": {},
    }
    print("    No progress file found — starting fresh.")

def save_progress():
    progress["last_updated"] = datetime.now(timezone.utc).isoformat()
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f)

# ── Graceful shutdown on Ctrl-C ────────────────────────────────────────────────
_shutdown = False

def _handle_sigint(sig, frame):
    global _shutdown
    print("\n  [!] Interrupt received — saving progress and exiting after current request.")
    _shutdown = True

signal.signal(signal.SIGINT, _handle_sigint)

# ── Single persistent FDSN client ─────────────────────────────────────────────
print("\n[3] Initialising single FDSN client...")
try:
    client = Client(FDSN_BASE)
    print(f"    Client ready: {FDSN_BASE}")
except Exception as e:
    print(f"    FATAL: Cannot initialise FDSN client: {e}")
    sys.exit(1)

# ── Download function with retry ───────────────────────────────────────────────
def download_one(client, row, sta_id):
    """
    Download one (event, station) pair.
    Returns (status, nbytes, is_gap).
    Retries up to len(RETRY_WAITS) times with exponential backoff.
    """
    ev_t    = UTCDateTime(row["origin_time"].isoformat().replace("+00:00", "Z"))
    ev_id   = str(row["event_id"]).replace("/", "_").replace(":", "_")[-28:]
    t_start = ev_t - WIN_BEFORE
    t_end   = ev_t + WIN_AFTER
    net, sta = sta_id.split(".")
    fname   = RAW_DIR / f"ev_{sta_id.replace('.','_')}_{ev_id}.mseed"

    if fname.exists():
        return "exists", fname.stat().st_size, False

    last_err = ""
    for attempt, wait in enumerate(RETRY_WAITS):
        try:
            st = client.get_waveforms(net, sta, "*", "HH*", t_start, t_end)
            st.merge(fill_value=0)
            z_traces = [tr for tr in st if tr.stats.channel.endswith("Z")]
            gap_frac = float(np.mean(z_traces[0].data == 0)) if z_traces else 0.0
            st.write(str(fname), format="MSEED")
            return "ok", fname.stat().st_size, (gap_frac > 0.05)
        except Exception as e:
            last_err = str(e)[:100]
            if attempt < len(RETRY_WAITS) - 1:
                time.sleep(wait)
            # else: fall through to return fail

    return f"fail:{last_err}", 0, True

# ── Build pending task list ────────────────────────────────────────────────────
# Include: pairs not in progress OR pairs that previously failed (worth retrying
# with the fixed single-client approach).
print("\n[4] Building task list (pending + previously failed)...")

pending = []
already_ok = 0
for _, row in cat_df.iterrows():
    for sta_id in active_stas:
        key    = f"{row['event_id']}::{sta_id}"
        status = progress["completed"].get(key, None)
        if status in ("ok", "exists"):
            already_ok += 1
        else:
            pending.append((row, sta_id, key))

print(f"    Already successful : {already_ok}")
print(f"    Pending (new+retry): {len(pending)}")
est_h = len(pending) * (SLEEP_BETWEEN_REQUESTS + 1.2) / 3600
print(f"    Estimated time     : {est_h:.1f} h  (at ~{SLEEP_BETWEEN_REQUESTS+1.2:.1f} s/req)")

# ── Main download loop ─────────────────────────────────────────────────────────
print(f"\n[5] Downloading...")

n_ok_session  = 0
n_fail_session = 0
n_since_save  = 0
t0            = time.time()

for i, (row, sta_id, key) in enumerate(pending):
    if _shutdown:
        break

    status, nbytes, is_gap = download_one(client, row, sta_id)

    progress["completed"][key] = status

    gc = progress["gap_counts"].setdefault(sta_id, {"ok": 0, "gap": 0})
    if status in ("ok", "exists"):
        n_ok_session  += 1
        n_since_save  += 1
        gc["ok" if not is_gap else "gap"] = gc.get("ok" if not is_gap else "gap", 0) + 1
    else:
        n_fail_session += 1
        gc["gap"] = gc.get("gap", 0) + 1

    # Save progress periodically
    if n_since_save >= SAVE_EVERY:
        save_progress()
        n_since_save = 0

    # Progress report every 200 downloads
    if (i + 1) % 200 == 0 or (i + 1) == len(pending):
        elapsed    = time.time() - t0
        rate       = (i + 1) / elapsed if elapsed > 0 else 0
        remaining  = (len(pending) - i - 1) / rate / 3600 if rate > 0 else 0
        total_ok   = already_ok + n_ok_session
        print(f"  [{i+1:>6}/{len(pending)}]  ok={n_ok_session:>5}  fail={n_fail_session:>5}  "
              f"total_ok={total_ok:>6}  {rate:.1f} req/s  ETA ~{remaining:.1f}h")

    time.sleep(SLEEP_BETWEEN_REQUESTS)

# Final save
save_progress()

# ── Build inventory CSV ────────────────────────────────────────────────────────
print("\n[6] Writing inventory CSV...")
inv_rows = []
for key, status in progress["completed"].items():
    ev_id_raw, sta_id = key.split("::", 1)
    ev_row = cat_df[cat_df["event_id"] == ev_id_raw]
    mag    = float(ev_row["magnitude"].iloc[0]) if len(ev_row) > 0 else float("nan")
    ev_id_short = ev_id_raw.replace("/","_").replace(":","_")[-28:]
    fname  = f"ev_{sta_id.replace('.','_')}_{ev_id_short}.mseed"
    inv_rows.append({
        "event_id":  ev_id_raw,
        "station":   sta_id,
        "magnitude": mag,
        "file":      fname,
        "status":    "ok" if status in ("ok", "exists") else "fail",
    })

inv_df = pd.DataFrame(inv_rows)
inv_df.to_csv(INV_FILE, index=False)

# ── Summary ────────────────────────────────────────────────────────────────────
all_mseed = list(RAW_DIR.glob("*.mseed"))
final_gb  = sum(f.stat().st_size for f in all_mseed) / 1e9
total_ok  = already_ok + n_ok_session

print("\n" + "=" * 70)
print("PHASE 5 DOWNLOAD — COMPLETE")
print(f"  This session : {n_ok_session} ok  |  {n_fail_session} fail")
print(f"  Total ok     : {total_ok}  (including previous runs)")
print(f"  miniSEED files on disk : {len(all_mseed)}")
print(f"  Total raw data size    : {final_gb:.2f} GB")
print(f"  Progress file          : {PROGRESS_FILE}")
print(f"  Inventory              : {INV_FILE}")
print("=" * 70)
print("  → Re-run scripts/11_phase5_analyze.py to preprocess new downloads")
print("  → Then re-run scripts/12_prepare_finetune_dataset.py to rebuild HDF5")
print("=" * 70)
