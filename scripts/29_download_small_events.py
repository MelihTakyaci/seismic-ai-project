# Phase: Quality Audit — Optimization Step 1B
# Purpose: Download waveform windows for M 0.5–2.0 events from KOERI EIDA,
#          using same 2 pilot stations (KO.KOZT, KO.KMRS) and same preprocessing
#          pipeline as the main dataset. Enables sub-threshold recall evaluation.
# Inputs:  data/catalog/small_events_catalog.csv
# Outputs: data/windows/small_events/*.npz, artifacts/small_event_inventory.csv
# Limitations: EIDA may not have waveforms for all sub-threshold events.
#              Success rate expected 30-60%. M < 1.0 events may have very low SNR.
#              P pick times are not available — detection window uses origin time.

from pathlib import Path
import random
import traceback

import numpy as np
import pandas as pd
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from scipy.signal import butter, filtfilt, resample_poly
from math import gcd

ROOT    = Path(__file__).resolve().parent.parent
CAT_DIR = ROOT / "data" / "catalog"
WIN_DIR = ROOT / "data" / "windows" / "small_events"
ART     = ROOT / "artifacts"
WIN_DIR.mkdir(parents=True, exist_ok=True)

# ── Parameters ──────────────────────────────────────────────────────────────
STATIONS     = ["KO.KOZT", "KO.KMRS"]
CHANNELS     = "HH*"
SRATE_TARGET = 100.0
WIN_LEN_S    = 60.0
ORIGIN_OFFSET= 30.0       # seconds before origin to start window
N_SAMPLES    = int(SRATE_TARGET * WIN_LEN_S)  # 6000
BP_LO, BP_HI = 1.0, 45.0
RANDOM_SEED  = 42
N_PER_BAND   = 200        # 200 per magnitude band = 600 total
REQUIRED_CH  = ("HHZ", "HHN", "HHE")
MAX_GAP_FRAC = 0.05

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def bandpass(data, lo, hi, fs):
    nyq  = fs / 2.0
    b, a = butter(4, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, data)

def safe_resample(trace, target_fs):
    """Resample an ObsPy Trace to target_fs using scipy resample_poly."""
    src_fs = trace.stats.sampling_rate
    if abs(src_fs - target_fs) < 0.01:
        return trace.data.astype(np.float64)
    g = gcd(int(round(target_fs)), int(round(src_fs)))
    up   = int(round(target_fs)) // g
    down = int(round(src_fs))    // g
    return resample_poly(trace.data.astype(np.float64), up, down)

print("=" * 70)
print("STEP 1B — Download M 0.5–2.0 Waveforms")
print(f"  Stations: {STATIONS}")
print(f"  Target: {N_PER_BAND} per band × 3 bands × {len(STATIONS)} stations")
print("=" * 70)

# ── Load and stratified-sample catalog ─────────────────────────────────────
df = pd.read_csv(CAT_DIR / "small_events_catalog.csv")
df["magnitude"] = pd.to_numeric(df["magnitude"], errors="coerce")
df = df.dropna(subset=["magnitude"])

# Prefer ISC events (best quality retrospective picks)
df_isc = df[df["source"].isin(["ISC", "ISC_HTTP"])].copy()
df_isc["origin_time_dt"] = pd.to_datetime(df_isc["origin_time"], utc=True, errors="coerce")
df_isc = df_isc.dropna(subset=["origin_time_dt"]).reset_index(drop=True)

BANDS = [
    ("sub1",  0.5,  1.0),
    ("m1to15", 1.0, 1.5),
    ("m15to2", 1.5, 2.0),
]

sampled = []
for band_name, lo, hi in BANDS:
    sub = df_isc[(df_isc["magnitude"] > lo) & (df_isc["magnitude"] <= hi)]
    n_take = min(N_PER_BAND, len(sub))
    chosen = sub.sample(n=n_take, random_state=RANDOM_SEED).copy()
    chosen["mag_band"] = f"M{lo}-{hi}"
    sampled.append(chosen)
    print(f"  Band M{lo}–{hi}: {len(sub)} available → sampling {n_take}")

sample_df = pd.concat(sampled, ignore_index=True)
print(f"\n  Total events to try: {len(sample_df)}")
print(f"  × {len(STATIONS)} stations = up to {len(sample_df) * len(STATIONS)} downloads")

# ── EIDA client ─────────────────────────────────────────────────────────────
print("\n[2] Connecting to KOERI EIDA ...")
try:
    client = Client("http://eida.koeri.boun.edu.tr")
    print("    ✓ EIDA client ready")
except Exception as e:
    print(f"    ✗ EIDA failed: {e}. Trying ORFEUS ...")
    client = Client("ORFEUS")
    print("    ✓ ORFEUS fallback")

# ── Download loop ────────────────────────────────────────────────────────────
print("\n[3] Downloading and preprocessing ...")
inventory = []
n_ok  = 0
n_fail= 0
n_done= 0
total_attempts = len(sample_df) * len(STATIONS)

for _, ev in sample_df.iterrows():
    t_origin = UTCDateTime(float(ev["origin_time_dt"].timestamp()))
    t_start  = t_origin - ORIGIN_OFFSET
    t_end    = t_origin + (WIN_LEN_S - ORIGIN_OFFSET)
    mag      = float(ev["magnitude"])
    ev_id    = str(ev["event_id"])
    mag_band = str(ev["mag_band"])

    for sta_str in STATIONS:
        net, sta = sta_str.split(".")
        n_done += 1
        safe_evid = ev_id.replace("/", "_").replace(":", "_")[:40]
        fname = f"small_{sta_str}_{safe_evid}.npz"
        out_path = WIN_DIR / fname

        if out_path.exists():
            n_ok += 1
            inventory.append({
                "filename":   fname,
                "station":    sta_str,
                "event_id":   ev_id,
                "magnitude":  mag,
                "mag_band":   mag_band,
                "t_start":    str(t_start),
                "t_origin":   str(t_origin),
                "status":     "cached",
            })
            continue

        try:
            st = client.get_waveforms(
                network=net, station=sta, location="*", channel=CHANNELS,
                starttime=t_start, endtime=t_end + 5,
            )
            st.merge(method=1, fill_value=0)
            st.detrend("demean")
            st.detrend("linear")

            # Check all 3 components present
            present_ch = {tr.stats.channel for tr in st}
            missing = [c for c in REQUIRED_CH if c not in present_ch]
            if missing:
                raise ValueError(f"Missing channels: {missing}")

            # Process each component
            channels_out = []
            for ch in REQUIRED_CH:
                tr = st.select(channel=ch)[0].copy()
                # Check gap fraction
                expected_n = int(WIN_LEN_S * tr.stats.sampling_rate)
                gap_frac = 1.0 - min(len(tr.data), expected_n) / max(expected_n, 1)
                if gap_frac > MAX_GAP_FRAC:
                    raise ValueError(f"Gap fraction {gap_frac:.2f} > {MAX_GAP_FRAC}")

                # Bandpass
                data = bandpass(tr.data.astype(np.float64), BP_LO, BP_HI, tr.stats.sampling_rate)
                # Resample to 100 Hz
                data = safe_resample(tr, SRATE_TARGET)

                # Trim/pad to exactly 6000 samples
                if len(data) >= N_SAMPLES:
                    data = data[:N_SAMPLES]
                else:
                    data = np.pad(data, (0, N_SAMPLES - len(data)))

                # Z-score normalize
                std = float(np.std(data))
                if std > 1e-9:
                    data = data / std
                channels_out.append(data.astype(np.float32))

            waveform = np.stack(channels_out, axis=0)  # (3, 6000)

            np.savez(
                out_path,
                data=waveform,
                window_type=np.bytes_("small_event"),
                station=np.bytes_(sta_str),
                event_id=np.bytes_(ev_id),
                magnitude=np.float32(mag),
                t_start=np.bytes_(str(t_start)),
                t_origin=np.bytes_(str(t_origin)),
                mag_band=np.bytes_(mag_band),
            )
            n_ok += 1
            inventory.append({
                "filename":   fname,
                "station":    sta_str,
                "event_id":   ev_id,
                "magnitude":  mag,
                "mag_band":   mag_band,
                "t_start":    str(t_start),
                "t_origin":   str(t_origin),
                "status":     "ok",
            })

        except Exception as e:
            n_fail += 1
            inventory.append({
                "filename":   fname,
                "station":    sta_str,
                "event_id":   ev_id,
                "magnitude":  mag,
                "mag_band":   mag_band,
                "t_start":    str(t_start),
                "t_origin":   str(t_origin),
                "status":     f"fail: {str(e)[:80]}",
            })

        if n_done % 50 == 0:
            print(f"    {n_done}/{total_attempts}  OK={n_ok}  FAIL={n_fail}  "
                  f"rate={n_ok/n_done*100:.0f}%")

print(f"\n  Download complete:")
print(f"    Total attempts: {total_attempts}")
print(f"    Successful:     {n_ok} ({n_ok/total_attempts*100:.1f}%)")
print(f"    Failed:         {n_fail}")

# ── Save inventory ───────────────────────────────────────────────────────────
inv_df = pd.DataFrame(inventory)
inv_path = ART / "small_event_inventory.csv"
inv_df.to_csv(inv_path, index=False)
print(f"\n  Inventory → {inv_path}")

# Report by band
ok_inv = inv_df[inv_df["status"].str.startswith("ok") | (inv_df["status"] == "cached")]
print(f"\n  Success by magnitude band:")
print(f"  {'Band':<12} {'Attempts':>10} {'OK':>6} {'Rate':>7}")
print(f"  {'-'*38}")
for mag_band in ok_inv["mag_band"].unique() if len(ok_inv) > 0 else []:
    n_att = (inv_df["mag_band"] == mag_band).sum()
    n_ok_band = (ok_inv["mag_band"] == mag_band).sum()
    rate = n_ok_band / n_att * 100 if n_att > 0 else 0
    print(f"  {mag_band:<12} {n_att:>10} {n_ok_band:>6} {rate:>6.1f}%")

print("\n" + "=" * 70)
print("STEP 1B COMPLETE")
print(f"  Windows saved: {n_ok}")
print(f"  Directory: data/windows/small_events/")
print(f"  Inventory: artifacts/small_event_inventory.csv")
if n_ok >= 200:
    print(f"  ✓ Sufficient windows for Step 1C evaluation")
else:
    print(f"  ⚠ Fewer than 200 windows — proceeding with {n_ok} available")
print("=" * 70)
