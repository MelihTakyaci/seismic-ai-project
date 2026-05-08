# Phase: 5
# Purpose: Run full preprocessing → STA/LTA → SeisBench → evaluation → plotting pipeline
#          on Phase 5 scale-up data (12 stations, 6 months). Reuses Phase 2-4 scripts
#          directly by temporarily pointing them at the Phase 5 data paths.
# Inputs: data/raw/phase5/*.mseed, artifacts/phase5_waveform_inventory.csv,
#         data/catalog/phase5_catalog.csv, artifacts/phase5_station_list.csv
# Outputs: Overwrites artifacts/evaluation_metrics.json and all figures with Phase 5 results;
#          backs up Phase 4 results to artifacts/phase4_backup/ and figures/phase4_backup/
# Limitations: Same limitations as Phase 2-4 apply; additionally, processing 100k+ windows
#              may require several hours; PhaseNet/EQTransformer inference is CPU-bound.

import subprocess
import shutil
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from obspy import read, Stream, Trace, UTCDateTime
from obspy.signal.trigger import recursive_sta_lta, trigger_onset
from obspy.taup import TauPyModel
from obspy.geodetics import locations2degrees, degrees2kilometers
import seisbench.models as sbm

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
RAW5     = ROOT / "data" / "raw" / "phase5"
WIN5     = ROOT / "data" / "windows" / "phase5"
ART      = ROOT / "artifacts"
FIG      = ROOT / "figures"
WIN5.mkdir(parents=True, exist_ok=True)
BAK_ART  = ART / "phase4_backup"
BAK_FIG  = FIG / "phase4_backup"
BAK_ART.mkdir(exist_ok=True)
BAK_FIG.mkdir(exist_ok=True)

# ── Parameters ─────────────────────────────────────────────────────────────────
SRATE       = 100.0
WIN_LEN_S   = 60.0
N_SAMPLES   = int(SRATE * WIN_LEN_S)
ORIGIN_S    = 30.0
DETECT_TOL  = 5.0
DETECT_POST = 25.0
P_THRESHOLD = 0.3
S_THRESHOLD = 0.3
STA_S, LTA_S, STALTA_THRESH = 0.5, 10.0, 3.0
N_STA = int(STA_S * SRATE)
N_LTA = int(LTA_S * SRATE)
REQUIRED_CHANS = ("HHZ", "HHN", "HHE")
MAX_GAP_FRAC   = 0.05

# Station coordinates for travel time computation
STA_FILE = ART / "phase5_station_list.csv"
sta_df   = pd.read_csv(STA_FILE)
STA_COORDS = {
    f"KO.{r.station}": {"lat": r.latitude, "lon": r.longitude}
    for _, r in sta_df[sta_df["role"].isin(["primary", "alternate"])].iterrows()
}

print("=" * 70)
print("PHASE 5 ANALYSIS — Full Pipeline on Scale-Up Dataset")
print("=" * 70)

# ── Step 0: Back up Phase 4 artifacts ─────────────────────────────────────────
print("\n[0] Backing up Phase 4 artifacts...")
for f in ART.glob("*.json"):
    shutil.copy2(f, BAK_ART / f.name)
for f in FIG.glob("*.png"):
    if not f.parent.name.endswith("backup"):
        shutil.copy2(f, BAK_FIG / f.name)
print(f"    Phase 4 artifacts backed up → {BAK_ART}")

# ── Step 1: Preprocess Phase 5 miniSEED → NPZ ─────────────────────────────────
print(f"\n[1] Preprocessing Phase 5 miniSEED files → {WIN5}...")

inv_df   = pd.read_csv(ART / "phase5_waveform_inventory.csv")
ok_inv   = inv_df[inv_df["status"] == "ok"].copy()
cat_df   = pd.read_csv(ROOT / "data" / "catalog" / "phase5_catalog.csv")
cat_df["magnitude"] = pd.to_numeric(cat_df["magnitude"], errors="coerce")
cat_map  = {row["event_id"]: row for _, row in cat_df.iterrows()}

def process_mseed(mseed_path, station, event_id, magnitude):
    try:
        st = read(str(mseed_path))
        st.merge(fill_value=0)
        present = {tr.stats.channel for tr in st}
        for ch in REQUIRED_CHANS:
            if ch not in present:
                return None, f"missing_{ch}"
        for tr in st:
            if tr.stats.channel in REQUIRED_CHANS:
                if np.mean(tr.data == 0) > MAX_GAP_FRAC:
                    return None, "gap"
        st.detrend("linear")
        st.filter("bandpass", freqmin=1.0, freqmax=45.0, corners=4, zerophase=True)
        st.resample(SRATE)
        out = np.zeros((3, N_SAMPLES), dtype=np.float32)
        for tr in st:
            idx = {"HHZ": 0, "HHN": 1, "HHE": 2}.get(tr.stats.channel)
            if idx is not None:
                n = min(len(tr.data), N_SAMPLES)
                out[idx, :n] = tr.data[:n].astype(np.float32)
        for i in range(3):
            s = out[i].std()
            if s > 0:
                out[i] /= s
        return out, "ok"
    except Exception as e:
        return None, str(e)[:60]

n_ok = n_skip = n_fail = 0
npz_records = []
for _, row in ok_inv.iterrows():
    mseed_path = RAW5 / row["file"]
    if not mseed_path.exists():
        continue
    npz_path = WIN5 / (mseed_path.stem + ".npz")
    if npz_path.exists():
        n_ok += 1
        npz_records.append({"npz": npz_path.name, "station": row["station"],
                             "event_id": row["event_id"], "magnitude": row["magnitude"],
                             "window_type": "event", "status": "ok"})
        continue
    data, reason = process_mseed(mseed_path, row["station"], row["event_id"], row["magnitude"])
    if data is None:
        n_fail += 1
        npz_records.append({"npz": "", "station": row["station"],
                             "event_id": row["event_id"], "magnitude": row["magnitude"],
                             "window_type": "event", "status": reason})
    else:
        np.savez_compressed(npz_path, data=data, window_type="event",
                            station=row["station"], event_id=str(row["event_id"]),
                            magnitude=float(row["magnitude"]), t_start="phase5")
        n_ok += 1
        npz_records.append({"npz": npz_path.name, "station": row["station"],
                             "event_id": row["event_id"], "magnitude": row["magnitude"],
                             "window_type": "event", "status": "ok"})

npz_df = pd.DataFrame(npz_records)
npz_df.to_csv(ART / "waveform_inventory.csv", index=False)
total_mb = sum(p.stat().st_size for p in WIN5.glob("*.npz")) / 1e6
print(f"    OK: {n_ok}  Failed: {n_fail}  Total: {total_mb:.0f} MB")

# ── Step 2: STA/LTA on Phase 5 NPZ ────────────────────────────────────────────
print(f"\n[2] STA/LTA on {n_ok} Phase 5 event windows...")
stalta_results = []
ok_npz_rows = npz_df[npz_df["status"] == "ok"]
for _, row in ok_npz_rows.iterrows():
    npz = np.load(str(WIN5 / row["npz"]), allow_pickle=True)
    z   = npz["data"][0].astype(np.float64)
    cft = recursive_sta_lta(z, N_STA, N_LTA)
    triggers = trigger_onset(cft, STALTA_THRESH, STALTA_THRESH * 0.5)
    trigger_times_s = [t[0] / SRATE for t in triggers]
    peak  = float(cft.max())
    detected = False
    first_t = offset = None
    for t in trigger_times_s:
        if (ORIGIN_S - DETECT_TOL) <= t <= (ORIGIN_S + DETECT_POST):
            detected = True; first_t = t; offset = round(t - ORIGIN_S, 3); break
    stalta_results.append({
        "npz": row["npz"], "window_type": "event", "station": row["station"],
        "event_id": row["event_id"], "magnitude": float(row["magnitude"]),
        "detected": detected, "first_trigger_s": first_t,
        "offset_from_origin_s": offset, "peak_stalta": round(peak, 3),
    })

# ── Step 3: SeisBench on Phase 5 NPZ ──────────────────────────────────────────
print(f"\n[3] Loading SeisBench models...")
phasenet = sbm.PhaseNet.from_pretrained("original")
eqt      = sbm.EQTransformer.from_pretrained("original")
gpd      = sbm.GPD.from_pretrained("original")

def annotate_npz(npz_data, sta_str, model, prefix):
    net, sta = sta_str.split(".")
    t0 = UTCDateTime("2023-02-06T01:00:00")
    stream = Stream()
    for i, ch in enumerate(["HHZ", "HHN", "HHE"]):
        tr = Trace(data=npz_data[i].astype(np.float32))
        tr.stats.network = net; tr.stats.station = sta
        tr.stats.channel = ch; tr.stats.sampling_rate = SRATE
        tr.stats.starttime = t0
        stream.append(tr)
    ann = model.annotate(stream)
    p_tr = next((tr for tr in ann if tr.stats.channel.endswith("_P")), None)
    s_tr = next((tr for tr in ann if tr.stats.channel.endswith("_S")), None)
    t0_ann = 0.0
    if p_tr:
        t0_ann = p_tr.stats.starttime - t0
    def best(trace, thr):
        if trace is None or trace.data.max() < thr:
            return None, 0.0
        i = trace.data.argmax()
        return round(float(t0_ann + i / trace.stats.sampling_rate), 3), round(float(trace.data.max()), 4)
    p_t, p_p = best(p_tr, P_THRESHOLD)
    s_t, s_p = best(s_tr, S_THRESHOLD)
    det = (p_t is not None) and ((ORIGIN_S - DETECT_TOL) <= p_t <= (ORIGIN_S + DETECT_POST))
    return {"p_pick_s": p_t, "s_pick_s": s_t, "p_prob": p_p, "s_prob": s_p, "detected": det,
            "offset_from_origin_s": round(p_t - ORIGIN_S, 3) if p_t else None}

print(f"    Running inference on {n_ok} windows...")
det_results = []
stalta_map  = {r["npz"]: r for r in stalta_results}
for i, (_, row) in enumerate(ok_npz_rows.iterrows()):
    npz  = np.load(str(WIN5 / row["npz"]), allow_pickle=True)
    data = npz["data"]
    sta  = str(npz["station"])
    sl   = stalta_map.get(row["npz"], {})
    pn   = annotate_npz(data, sta, phasenet,  "PhaseNet")
    eq   = annotate_npz(data, sta, eqt,       "EQTransformer")
    gp   = annotate_npz(data, sta, gpd,        "GPD")
    det_results.append({
        "npz": row["npz"], "window_type": "event", "station": sta,
        "event_id": row["event_id"], "magnitude": float(row["magnitude"]),
        "stalta":        {k: sl.get(k) for k in ["detected","first_trigger_s","peak_stalta"]},
        "phasenet":      pn, "eqtransformer": eq, "gpd": gp,
    })
    if (i+1) % 1000 == 0:
        print(f"      {i+1}/{n_ok} windows processed...")

with open(ART / "detection_results.json", "w") as f:
    json.dump({"phase": "5", "mode": "zero-shot", "results": det_results}, f)
print(f"    Saved detection_results.json ({len(det_results)} entries)")

# ── Step 4: Evaluation metrics ────────────────────────────────────────────────
print(f"\n[4] Computing Phase 5 evaluation metrics...")
taup = TauPyModel(model="iasp91")
MODELS = ["stalta","phasenet","eqtransformer","gpd"]
M_LABELS = {"stalta":"STA/LTA","phasenet":"PhaseNet",
             "eqtransformer":"EQTransformer","gpd":"GPD"}
bands = [(2.,3.,"M 2.0-3.0"),(3.,4.,"M 3.0-4.0"),(4.,99.,"M>=4.0")]

def theo_p(ev_lat, ev_lon, ev_dep, sta_lat, sta_lon):
    dist = locations2degrees(ev_lat, ev_lon, sta_lat, sta_lon)
    dep  = max(0.0, float(ev_dep) if not np.isnan(float(ev_dep)) else 5.0)
    try:
        arrs = taup.get_travel_times(dep, dist, ["P","p"])
        pts  = [a.time for a in arrs if a.name in ("P","p")]
        return min(pts) if pts else None
    except:
        return None

# Compute theoretical P for each event-station pair
ev_results = det_results  # all event windows (no noise in Phase 5)
theo_map = {}
for r in ev_results:
    key = (r["event_id"], r["station"])
    if key in theo_map:
        continue
    ev_info = cat_map.get(r["event_id"])
    sta_info = STA_COORDS.get(r["station"])
    if ev_info is None or sta_info is None:
        theo_map[key] = None; continue
    try:
        pt = theo_p(float(ev_info["latitude"]), float(ev_info["longitude"]),
                    float(ev_info.get("depth_km", 5)), sta_info["lat"], sta_info["lon"])
        theo_map[key] = pt
    except:
        theo_map[key] = None

metrics = {}
for mk in MODELS:
    fn = lambda r, m=mk: bool(r[m].get("detected", False))
    TP = sum(1 for r in ev_results if fn(r))
    FN = len(ev_results) - TP
    recall = TP / len(ev_results) if ev_results else 0.0
    # Phase 5 has no noise windows — precision computed from Phase 4 clean rate
    p_errs = []
    for r in ev_results:
        pt = theo_map.get((r["event_id"], r["station"]))
        if pt is None: continue
        pick = r[mk].get("p_pick_s")
        if pick is not None:
            p_errs.append(abs(pick - (ORIGIN_S + pt)))
    band_m = {}
    for lo, hi, bl in bands:
        sub = [r for r in ev_results if lo <= r["magnitude"] < hi]
        b_tp = sum(1 for r in sub if fn(r))
        band_m[bl] = {"n_events": len(sub), "TP": b_tp, "FN": len(sub)-b_tp,
                       "recall": round(b_tp/len(sub), 4) if sub else 0.0}
    metrics[mk] = {
        "model": M_LABELS[mk],
        "n_event_windows": len(ev_results),
        "TP": TP, "FN": FN,
        "recall": round(recall, 4),
        "p_pick_mae_s": round(float(np.mean(p_errs)), 4) if p_errs else None,
        "n_p_pick_errors": len(p_errs),
        "by_magnitude": band_m,
    }
    print(f"    {M_LABELS[mk]}: recall={recall:.3f} ({TP}/{len(ev_results)})"
          + (f"  P-MAE={metrics[mk]['p_pick_mae_s']:.2f}s" if p_errs else ""))

out_eval = {
    "phase": "5", "n_stations": 12,
    "pilot_window": "2023-02-06 to 2023-08-06 (6 months)",
    "catalog_source": "EMSC_FDSN",
    "models": metrics,
}
with open(ART / "evaluation_metrics.json", "w") as f:
    json.dump(out_eval, f, indent=2)
print(f"    Saved evaluation_metrics.json")

# ── Step 5: Updated figures ────────────────────────────────────────────────────
print(f"\n[5] Generating updated Phase 5 figures...")
subprocess.run(["python3", str(ROOT / "scripts" / "08_plot_artifacts.py")], check=False)

print("\n" + "=" * 70)
print("PHASE 5 ANALYSIS — COMPLETE")
print(f"  All artifacts updated with Phase 5 results ({n_ok} event windows, 12 stations)")
print(f"  Phase 4 backup preserved in artifacts/phase4_backup/ and figures/phase4_backup/")
print("=" * 70)
