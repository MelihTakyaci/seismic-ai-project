# Phase: 4
# Purpose: Compute precision, recall, F1, false positives per hour, and phase picking MAE
#          for STA/LTA, PhaseNet, EQTransformer, and GPD on the pilot dataset
# Inputs: artifacts/detection_results.json, data/catalog/koeri_pilot_catalog.csv,
#         artifacts/station_summary.csv
# Outputs: artifacts/evaluation_metrics.json
# Limitations: No catalog P/S picks available (EMSC); picking MAE uses theoretical
#              P arrival from IASP91 velocity model as reference (uncertainty ~0.3-0.5 s);
#              noise window labels contaminated by sub-threshold aftershocks (see L2),
#              so false positive rates reported as upper bounds; pilot covers M>=2.0 only.

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from obspy.taup import TauPyModel
from obspy.geodetics import degrees2kilometers, locations2degrees

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
DET_JSON   = ROOT / "artifacts" / "detection_results.json"
CATALOG    = ROOT / "data" / "catalog" / "koeri_pilot_catalog.csv"
STA_CSV    = ROOT / "artifacts" / "station_summary.csv"
OUT_JSON   = ROOT / "artifacts" / "evaluation_metrics.json"

# ── Parameters ─────────────────────────────────────────────────────────────────
PILOT_STATIONS = {
    "KO.KOZT": {"lat": 37.4795, "lon": 35.8264},
    "KO.KMRS": {"lat": 37.5087, "lon": 36.9000},
}
WIN_LEN_S       = 60.0
ORIGIN_S        = 30.0
NOISE_STALTA_THRESH = 3.0   # STA/LTA peak above this → likely contaminated noise

MODELS = ["stalta", "phasenet", "eqtransformer", "gpd"]
MODEL_LABELS = {
    "stalta":        "STA/LTA",
    "phasenet":      "PhaseNet",
    "eqtransformer": "EQTransformer",
    "gpd":           "GPD",
}

print("=" * 70)
print("PHASE 4 — Evaluation: Precision / Recall / F1 / FP-per-hour / Pick MAE")
print("=" * 70)

# ── Load data ──────────────────────────────────────────────────────────────────
with open(DET_JSON) as f:
    det_data = json.load(f)
results = det_data["results"]

cat_df = pd.read_csv(CATALOG)
cat_df["event_id_short"] = cat_df["event_id"].apply(lambda x: str(x))
cat_map = {row["event_id"]: row for _, row in cat_df.iterrows()}

# ── Separate event / noise windows ────────────────────────────────────────────
ev_results = [r for r in results if r["window_type"] == "event"]
ns_results = [r for r in results if r["window_type"] == "noise"]

# Identify clean vs contaminated noise windows (STA/LTA peak > threshold)
clean_ns = [r for r in ns_results
            if r["stalta"].get("peak_stalta", 0) <= NOISE_STALTA_THRESH]
contam_ns = [r for r in ns_results
             if r["stalta"].get("peak_stalta", 0) > NOISE_STALTA_THRESH]
print(f"\n[0] Noise window audit:")
print(f"    Total noise windows   : {len(ns_results)}")
print(f"    Likely clean (STA/LTA peak ≤ {NOISE_STALTA_THRESH}): {len(clean_ns)}")
print(f"    Likely contaminated    : {len(contam_ns)}")
print(f"    [FP rates computed on clean noise only; all-noise rates also reported]")

# ── TauP model for theoretical P/S picks ───────────────────────────────────────
print("\n[1] Computing theoretical P/S arrival times via IASP91...")
taup = TauPyModel(model="iasp91")

def theoretical_arrivals(ev_lat, ev_lon, ev_depth_km, sta_lat, sta_lon):
    """Return theoretical P and S arrival times (seconds after origin) via IASP91."""
    dist_deg = locations2degrees(ev_lat, ev_lon, sta_lat, sta_lon)
    depth = max(0.0, float(ev_depth_km) if not math.isnan(float(ev_depth_km)) else 5.0)
    try:
        arrs = taup.get_travel_times(
            source_depth_in_km=depth,
            distance_in_degree=dist_deg,
            phase_list=["P", "p", "S", "s"],
        )
        p_times = [a.time for a in arrs if a.name in ("P", "p")]
        s_times = [a.time for a in arrs if a.name in ("S", "s")]
        p_tt = min(p_times) if p_times else None
        s_tt = min(s_times) if s_times else None
        return p_tt, s_tt, dist_deg
    except Exception:
        return None, None, dist_deg

# Pre-compute theoretical arrivals for each event-station pair
theo_map = {}   # key: (event_id, station) → (p_tt_s, s_tt_s, dist_deg)
for r in ev_results:
    key = (r["event_id"], r["station"])
    if key in theo_map:
        continue
    ev_info = cat_map.get(r["event_id"])
    if ev_info is None:
        theo_map[key] = (None, None, None)
        continue
    sta_info = PILOT_STATIONS.get(r["station"])
    if sta_info is None:
        theo_map[key] = (None, None, None)
        continue
    try:
        lat = float(ev_info["latitude"])
        lon = float(ev_info["longitude"])
        dep = float(ev_info["depth_km"]) if str(ev_info["depth_km"]) not in ("", "nan") else 5.0
    except Exception:
        theo_map[key] = (None, None, None)
        continue
    p_tt, s_tt, dist = theoretical_arrivals(lat, lon, dep, sta_info["lat"], sta_info["lon"])
    theo_map[key] = (p_tt, s_tt, dist)

n_theo = sum(1 for v in theo_map.values() if v[0] is not None)
print(f"    Theoretical arrivals computed: {n_theo}/{len(theo_map)} event-station pairs")

# ── Per-model evaluation ───────────────────────────────────────────────────────
print("\n[2] Computing detection metrics per model...")

# Magnitude bands — note M<2 is absent from pilot catalog
bands = [
    (2.0,  3.0, "M 2.0-3.0"),
    (3.0,  4.0, "M 3.0-4.0"),
    (4.0, 99.0, "M>=4.0"),
]

noise_duration_hours = len(ns_results) * WIN_LEN_S / 3600.0
clean_noise_hours    = len(clean_ns)   * WIN_LEN_S / 3600.0

metrics = {}

for model_key in MODELS:
    label = MODEL_LABELS[model_key]

    def is_detected(r, mk=model_key):
        return bool(r[mk].get("detected", False))

    # ── Overall ──────────────────────────────────────────────────────────────
    TP = sum(1 for r in ev_results if is_detected(r))
    FN = len(ev_results) - TP

    FP_all   = sum(1 for r in ns_results if is_detected(r))
    FP_clean = sum(1 for r in clean_ns   if is_detected(r))
    TN_all   = len(ns_results) - FP_all
    TN_clean = len(clean_ns)   - FP_clean

    precision_all   = TP / (TP + FP_all)   if (TP + FP_all) > 0   else 0.0
    precision_clean = TP / (TP + FP_clean) if (TP + FP_clean) > 0 else 0.0
    recall          = TP / (TP + FN)        if (TP + FN) > 0        else 0.0

    f1_all   = (2 * precision_all   * recall / (precision_all   + recall)
                if (precision_all   + recall) > 0 else 0.0)
    f1_clean = (2 * precision_clean * recall / (precision_clean + recall)
                if (precision_clean + recall) > 0 else 0.0)

    fp_per_hour_all   = FP_all   / noise_duration_hours if noise_duration_hours > 0 else 0.0
    fp_per_hour_clean = FP_clean / clean_noise_hours    if clean_noise_hours    > 0 else 0.0

    # ── By magnitude band ────────────────────────────────────────────────────
    band_metrics = {}
    for lo, hi, blabel in bands:
        band_ev = [r for r in ev_results if lo <= r["magnitude"] < hi]
        if not band_ev:
            band_metrics[blabel] = None
            continue
        b_tp = sum(1 for r in band_ev if is_detected(r))
        b_fn = len(band_ev) - b_tp
        b_recall = b_tp / len(band_ev)
        b_precision = precision_all  # use overall precision (no per-band noise split)
        b_f1 = (2 * b_precision * b_recall / (b_precision + b_recall)
                if (b_precision + b_recall) > 0 else 0.0)
        band_metrics[blabel] = {
            "n_events": len(band_ev),
            "TP": b_tp, "FN": b_fn,
            "recall":    round(b_recall, 4),
            "precision": round(b_precision, 4),
            "f1":        round(b_f1, 4),
        }

    # ── Phase picking MAE ────────────────────────────────────────────────────
    p_errors, s_errors = [], []
    if model_key in ("phasenet", "eqtransformer", "gpd"):
        for r in ev_results:
            key = (r["event_id"], r["station"])
            p_tt, s_tt, _ = theo_map.get(key, (None, None, None))
            if p_tt is None:
                continue
            # P pick: model pick time relative to window start, minus (ORIGIN_S + p_tt)
            mp = r[model_key]
            p_pick = mp.get("p_pick_s")
            s_pick = mp.get("s_pick_s")
            theoretical_p_in_window = ORIGIN_S + p_tt

            if p_pick is not None:
                p_errors.append(abs(p_pick - theoretical_p_in_window))
            if s_pick is not None and s_tt is not None:
                theoretical_s_in_window = ORIGIN_S + s_tt
                s_errors.append(abs(s_pick - theoretical_s_in_window))

    p_mae = round(float(np.mean(p_errors)), 4) if p_errors else None
    s_mae = round(float(np.mean(s_errors)), 4) if s_errors else None
    p_med = round(float(np.median(p_errors)), 4) if p_errors else None

    metrics[model_key] = {
        "model":         label,
        "n_event_windows": len(ev_results),
        "n_noise_windows": len(ns_results),
        "n_clean_noise_windows": len(clean_ns),
        "TP": TP, "FN": FN,
        "FP_all_noise": FP_all, "FP_clean_noise": FP_clean,
        "TN_all_noise": TN_all, "TN_clean_noise": TN_clean,
        "recall":             round(recall,          4),
        "precision_all":      round(precision_all,   4),
        "precision_clean":    round(precision_clean, 4),
        "f1_all":             round(f1_all,          4),
        "f1_clean":           round(f1_clean,        4),
        "fp_per_hour_all":    round(fp_per_hour_all,   2),
        "fp_per_hour_clean":  round(fp_per_hour_clean, 2),
        "p_pick_mae_s":       p_mae,
        "p_pick_median_ae_s": p_med,
        "s_pick_mae_s":       s_mae,
        "n_p_pick_errors":    len(p_errors),
        "n_s_pick_errors":    len(s_errors),
        "by_magnitude":       band_metrics,
    }

    print(f"\n  {label}:")
    print(f"    Recall            : {recall:.3f}  ({TP}/{len(ev_results)})")
    print(f"    Precision (clean) : {precision_clean:.3f}  FP/hr (clean): {fp_per_hour_clean:.1f}")
    print(f"    F1 (clean)        : {f1_clean:.3f}")
    if p_mae is not None:
        print(f"    P-pick MAE        : {p_mae:.3f} s  (n={len(p_errors)}, vs IASP91 theoretical)")
    if s_mae is not None:
        print(f"    S-pick MAE        : {s_mae:.3f} s  (n={len(s_errors)})")

# ── Save evaluation_metrics.json ───────────────────────────────────────────────
output = {
    "phase": "4",
    "catalog_source": "EMSC_FDSN",
    "pilot_stations": list(PILOT_STATIONS.keys()),
    "pilot_window": "2023-02-06 to 2023-02-20",
    "magnitude_range_in_catalog": "M 2.0 – 6.0+ (no M<2.0 due to EMSC threshold)",
    "pick_mae_reference": "IASP91 theoretical arrival (uncertainty ~0.3-0.5 s)",
    "noise_window_note": (
        f"{len(contam_ns)}/{len(ns_results)} noise windows likely contaminated "
        f"(STA/LTA peak > {NOISE_STALTA_THRESH}); "
        "precision and FP rates use clean-noise subset as primary metric"
    ),
    "models": metrics,
}
with open(OUT_JSON, "w") as f:
    json.dump(output, f, indent=2)
print(f"\n[3] Saved → {OUT_JSON}")

# ── Print comparison table ────────────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"{'Model':<16} {'Recall':>8} {'Prec(c)':>8} {'F1(c)':>8} "
      f"{'FP/hr(c)':>10} {'P-MAE(s)':>10}")
print("─" * 70)
for mk in MODELS:
    m = metrics[mk]
    pm = f"{m['p_pick_mae_s']:.3f}" if m['p_pick_mae_s'] is not None else "  N/A "
    print(f"{m['model']:<16} {m['recall']:>8.3f} {m['precision_clean']:>8.3f} "
          f"{m['f1_clean']:>8.3f} {m['fp_per_hour_clean']:>10.1f} {pm:>10}")
print("─" * 70)
print("  (c) = computed on clean noise windows only")
print("  P-MAE = vs IASP91 theoretical arrival; reference uncertainty ≈0.3-0.5 s")
print(f"  Primary metric (recall on M<2.0): NOT COMPUTABLE — no M<2.0 in pilot catalog")
print(f"  Best proxy: recall on M 2.0-3.0 (see by_magnitude in evaluation_metrics.json)")
print("\n  → Run scripts/08_plot_artifacts.py next")
