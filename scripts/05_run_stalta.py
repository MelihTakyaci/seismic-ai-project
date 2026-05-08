# Phase: 3
# Purpose: Apply STA/LTA classical detector to all pilot NPZ windows as a non-ML baseline
# Inputs: data/windows/*.npz, artifacts/waveform_inventory.csv
# Outputs: artifacts/stalta_results.json (detection results for all windows)
# Limitations: STA/LTA detects only on HHZ (vertical component); does not provide P/S picks;
#              dense aftershock coda degrades LTA stability; threshold 3.0 is a standard
#              starting value and may need tuning for Turkey waveforms.

import json
from pathlib import Path

import numpy as np
import pandas as pd
from obspy.signal.trigger import recursive_sta_lta, trigger_onset

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
WIN_DIR  = ROOT / "data" / "windows"
OUT_JSON = ROOT / "artifacts" / "stalta_results.json"

# ── STA/LTA parameters ─────────────────────────────────────────────────────────
SRATE       = 100.0    # Hz
STA_S       = 0.5      # short-term window (seconds)
LTA_S       = 10.0     # long-term window (seconds)
THRESHOLD   = 3.0      # detection threshold
N_STA       = int(STA_S  * SRATE)   # 50 samples
N_LTA       = int(LTA_S  * SRATE)   # 1000 samples
WIN_LEN_S   = 60.0
ORIGIN_S    = 30.0     # origin time position in event window (seconds)
DETECT_TOL  = 5.0      # seconds BEFORE origin to allow (timing uncertainty in catalog)
DETECT_POST = 25.0     # seconds AFTER origin to allow (max P travel time for ~150 km distance)

print("=" * 70)
print("PHASE 3 — STA/LTA Baseline Detector")
print(f"  STA={STA_S}s ({N_STA} samples)  LTA={LTA_S}s ({N_LTA} samples)  threshold={THRESHOLD}")
print(f"  Detection window: [{ORIGIN_S-DETECT_TOL:.0f}s, {ORIGIN_S+DETECT_POST:.0f}s] "
      f"(origin−{DETECT_TOL:.0f}s to origin+{DETECT_POST:.0f}s; covers P travel times ≤150 km)")
print("=" * 70)

# ── Load inventory ─────────────────────────────────────────────────────────────
inv = pd.read_csv(ROOT / "artifacts" / "waveform_inventory.csv")
ok_inv = inv[inv["status"] == "ok"].copy()
print(f"\n[0] Valid NPZ windows: {len(ok_inv)}")

results = []

for _, row in ok_inv.iterrows():
    npz_stem = row["npz"] if pd.notna(row["npz"]) and row["npz"] != "" else ""
    if not npz_stem:
        continue
    npz_path = WIN_DIR / npz_stem
    if not npz_path.exists():
        continue

    npz = np.load(str(npz_path), allow_pickle=True)
    data = npz["data"]  # shape (3, 6000): [HHZ, HHN, HHE]
    z_trace = data[0].astype(np.float64)

    # Compute STA/LTA characteristic function on HHZ
    cft = recursive_sta_lta(z_trace, N_STA, N_LTA)

    # Find all triggers
    triggers = trigger_onset(cft, THRESHOLD, THRESHOLD * 0.5)
    trigger_times_s = [t[0] / SRATE for t in triggers]
    peak_value = float(cft.max())

    # For event windows: detected if any trigger within ±DETECT_TOL of ORIGIN_S
    w_type = str(row["window_type"])
    detected = False
    first_trigger_s = None
    offset_s = None

    if len(trigger_times_s) > 0:
        first_trigger_s = trigger_times_s[0]
        if w_type == "event":
            for t in trigger_times_s:
                if (ORIGIN_S - DETECT_TOL) <= t <= (ORIGIN_S + DETECT_POST):
                    detected = True
                    first_trigger_s = t
                    offset_s = round(t - ORIGIN_S, 3)
                    break
        elif w_type == "noise":
            detected = True  # any trigger in noise window = false positive

    results.append({
        "npz":             npz_stem,
        "window_type":     w_type,
        "station":         str(row["station"]),
        "event_id":        str(row["event_id"]),
        "magnitude":       float(row["magnitude"]),
        "detected":        detected,
        "first_trigger_s": round(first_trigger_s, 3) if first_trigger_s is not None else None,
        "offset_from_origin_s": offset_s,
        "peak_stalta":     round(peak_value, 3),
        "n_triggers":      len(trigger_times_s),
        "trigger_times_s": [round(t, 3) for t in trigger_times_s[:10]],
    })

# ── Write JSON ─────────────────────────────────────────────────────────────────
output = {
    "model": "STA/LTA",
    "parameters": {
        "sta_s": STA_S, "lta_s": LTA_S, "threshold": THRESHOLD,
        "component": "HHZ", "detect_tolerance_s": DETECT_TOL,
    },
    "results": results,
}
with open(OUT_JSON, "w") as f:
    json.dump(output, f, indent=2)
print(f"\n[1] STA/LTA results saved → {OUT_JSON}")

# ── Quick summary ──────────────────────────────────────────────────────────────
ev_rows = [r for r in results if r["window_type"] == "event"]
ns_rows = [r for r in results if r["window_type"] == "noise"]

ev_det    = sum(1 for r in ev_rows if r["detected"])
ns_det    = sum(1 for r in ns_rows if r["detected"])
ev_total  = len(ev_rows)
ns_total  = len(ns_rows)

print("\nSTA/LTA Preliminary Detection Rates (Phase 3 — visual reference only):")
print(f"  Event windows : {ev_det}/{ev_total} detected "
      f"({100*ev_det/max(ev_total,1):.1f}%)")
print(f"  Noise windows : {ns_det}/{ns_total} triggered "
      f"({100*ns_det/max(ns_total,1):.1f}%) "
      f"[many may be contaminated — see limitations.md L2]")

# By magnitude band
print("\n  Event detection by magnitude band:")
for lo, hi, label in [(2,3,"M 2-3"), (3,4,"M 3-4"), (4,99,"M>=4")]:
    band = [r for r in ev_rows if lo <= r["magnitude"] < hi]
    det  = sum(1 for r in band if r["detected"])
    print(f"    {label}: {det}/{len(band)} ({100*det/max(len(band),1):.1f}%)")

print("\n  → Run scripts/06_run_seisbench.py next")
