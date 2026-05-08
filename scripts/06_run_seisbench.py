# Phase: 3
# Purpose: Run pretrained PhaseNet, EQTransformer, and GPD on pilot NPZ windows via SeisBench;
#          merge with STA/LTA results; produce detection_results.json and comparison figures
# Inputs: data/windows/*.npz, artifacts/stalta_results.json
# Outputs: artifacts/detection_results.json, figures/waveform_with_picks_examples.png,
#          figures/stalta_vs_dl_comparison.png
# Limitations: All models run in zero-shot mode (no fine-tuning on Turkish data);
#              pre-training datasets are primarily North American/global catalogs;
#              GPD operates on 4s sliding windows and is evaluated as a P/S classifier only.

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from obspy import Stream, Trace, UTCDateTime
import seisbench.models as sbm

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
WIN_DIR   = ROOT / "data" / "windows"
STALTA_J  = ROOT / "artifacts" / "stalta_results.json"
OUT_JSON  = ROOT / "artifacts" / "detection_results.json"
FIG_DIR   = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ── Parameters ─────────────────────────────────────────────────────────────────
SRATE       = 100.0
WIN_LEN_S   = 60.0
ORIGIN_S    = 30.0          # origin time position in window (seconds)
DETECT_TOL  = 5.0           # seconds before origin to allow (catalog timing uncertainty)
DETECT_POST = 25.0          # seconds after origin to allow (covers P travel time ≤150 km at ~6 km/s)
P_THRESHOLD = 0.3           # probability threshold for P pick
S_THRESHOLD = 0.3           # probability threshold for S pick
CHANNEL_MAP = {0: "HHZ", 1: "HHN", 2: "HHE"}
T_AXIS      = np.linspace(0, WIN_LEN_S, int(WIN_LEN_S * SRATE))

print("=" * 70)
print("PHASE 3 — SeisBench: PhaseNet + EQTransformer + GPD (zero-shot)")
print(f"  P threshold: {P_THRESHOLD}   S threshold: {S_THRESHOLD}")
print(f"  Detection window: [{ORIGIN_S-DETECT_TOL:.0f}s, {ORIGIN_S+DETECT_POST:.0f}s] "
      f"(origin−{DETECT_TOL:.0f}s to origin+{DETECT_POST:.0f}s)")
print("=" * 70)

# ── Load models ────────────────────────────────────────────────────────────────
print("\n[1] Loading pretrained models...")
phasenet = sbm.PhaseNet.from_pretrained("original")
eqt      = sbm.EQTransformer.from_pretrained("original")
gpd      = sbm.GPD.from_pretrained("original")
print("    PhaseNet, EQTransformer, GPD — loaded (zero-shot weights)")

# ── Load STA/LTA results ───────────────────────────────────────────────────────
with open(STALTA_J) as f:
    stalta_data = json.load(f)
stalta_map = {r["npz"]: r for r in stalta_data["results"]}


def make_stream(data_arr, sta_str, t0=UTCDateTime("2023-02-06T01:00:00")):
    """Build an ObsPy Stream from (3, N) numpy array."""
    net, sta = sta_str.split(".")
    st = Stream()
    for i, ch in CHANNEL_MAP.items():
        tr = Trace(data=data_arr[i].astype(np.float32))
        tr.stats.network       = net
        tr.stats.station       = sta
        tr.stats.channel       = ch
        tr.stats.sampling_rate = SRATE
        tr.stats.starttime     = t0
        st.append(tr)
    return st


def pick_from_annotation(ann_stream, model_prefix, threshold_p, threshold_s, origin_s, tol):
    """
    Extract best P and S picks from SeisBench annotation stream.
    Returns dict with p_pick_s, s_pick_s, p_prob, s_prob, detected.
    """
    p_trace = s_trace = None
    for tr in ann_stream:
        if f"{model_prefix}_P" in tr.stats.channel or tr.stats.channel.endswith("_P"):
            p_trace = tr
        if f"{model_prefix}_S" in tr.stats.channel or tr.stats.channel.endswith("_S"):
            s_trace = tr

    t0_ann = 0.0
    if p_trace is not None:
        t0_ann = p_trace.stats.starttime - UTCDateTime("2023-02-06T01:00:00")

    def best_pick(trace, threshold):
        if trace is None:
            return None, 0.0
        data = trace.data
        if data.max() < threshold:
            return None, float(data.max())
        peak_idx = data.argmax()
        peak_t   = t0_ann + peak_idx / trace.stats.sampling_rate
        return round(float(peak_t), 3), round(float(data.max()), 4)

    p_t, p_prob = best_pick(p_trace, threshold_p)
    s_t, s_prob = best_pick(s_trace, threshold_s)

    # Detected if a P pick falls in [origin_s - tol, origin_s + DETECT_POST]
    # P wave always arrives after origin; DETECT_POST covers travel times up to ~150 km
    detected = (p_t is not None) and ((origin_s - tol) <= p_t <= (origin_s + DETECT_POST))

    return {
        "p_pick_s": p_t,
        "s_pick_s": s_t,
        "p_prob":   p_prob,
        "s_prob":   s_prob,
        "detected": detected,
        "offset_from_origin_s": round(p_t - origin_s, 3) if p_t is not None else None,
    }


def gpd_detect(data_arr, origin_s, tol, win_s=4.0, stride_s=0.5):
    """
    Run GPD on 4s sliding windows across the 60s trace.
    Returns best P pick time and probability.
    """
    win_n    = int(win_s   * SRATE)
    stride_n = int(stride_s * SRATE)
    total_n  = data_arr.shape[1]

    p_probs = []
    p_times = []
    for start in range(0, total_n - win_n + 1, stride_n):
        segment = data_arr[:, start: start + win_n]
        t_center = (start + win_n // 2) / SRATE
        # GPD input shape: (batch, channels, samples) — 1D conv on vertical component
        # Use Z component (index 0)
        x = torch_tensor_from_np(segment[:1])  # (1, 1, 400)
        try:
            pred = gpd.classify(
                Stream([Trace(data=segment[0].astype(np.float32),
                              header={"sampling_rate": SRATE,
                                      "network":"KO","station":"X","channel":"HHZ",
                                      "starttime": UTCDateTime("2023-02-06T01:00:00") + start/SRATE})])
            )
            # classify returns SeisBench Picks — check if any P pick
            for pick in pred.picks:
                if pick.phase == "P":
                    pt = float(pick.peak_time - UTCDateTime("2023-02-06T01:00:00"))
                    p_probs.append(float(pick.peak_value))
                    p_times.append(pt)
        except Exception:
            pass

    if not p_probs:
        return {"p_pick_s": None, "p_prob": 0.0, "detected": False, "offset_from_origin_s": None}

    # Best P pick closest to origin_s
    best_idx = min(range(len(p_times)), key=lambda i: abs(p_times[i] - origin_s))
    p_t    = p_times[best_idx]
    p_prob = p_probs[best_idx]
    detected = abs(p_t - origin_s) <= tol
    return {
        "p_pick_s": round(p_t, 3),
        "p_prob":   round(p_prob, 4),
        "detected": detected,
        "offset_from_origin_s": round(p_t - origin_s, 3),
    }


# ── Load inventory ─────────────────────────────────────────────────────────────
inv = pd.read_csv(ROOT / "artifacts" / "waveform_inventory.csv")
ok_inv = inv[inv["status"] == "ok"].copy()
print(f"\n[2] Processing {len(ok_inv)} valid NPZ windows...")

results = []
n_done  = 0

for _, row in ok_inv.iterrows():
    npz_stem = row.get("npz", "")
    if not npz_stem or pd.isna(npz_stem):
        continue
    npz_path = WIN_DIR / npz_stem
    if not npz_path.exists():
        continue

    npz  = np.load(str(npz_path), allow_pickle=True)
    data = npz["data"]   # (3, 6000)
    sta  = str(npz["station"])
    wtype = str(row["window_type"])
    mag   = float(row["magnitude"])
    t0    = UTCDateTime("2023-02-06T01:00:00")

    # STA/LTA result for this window
    sl = stalta_map.get(npz_stem, {})

    # Build stream for annotation
    st = make_stream(data, sta, t0)

    # ── PhaseNet ────────────────────────────────────────────────────────────
    try:
        ann_pn = phasenet.annotate(st)
        pn_res = pick_from_annotation(ann_pn, "PhaseNet", P_THRESHOLD, S_THRESHOLD, ORIGIN_S, DETECT_TOL)
    except Exception as e:
        pn_res = {"p_pick_s": None, "s_pick_s": None, "p_prob": 0.0, "s_prob": 0.0,
                  "detected": False, "offset_from_origin_s": None, "error": str(e)[:80]}

    # ── EQTransformer ────────────────────────────────────────────────────────
    try:
        ann_eq = eqt.annotate(st)
        eq_res = pick_from_annotation(ann_eq, "EQTransformer", P_THRESHOLD, S_THRESHOLD, ORIGIN_S, DETECT_TOL)
    except Exception as e:
        eq_res = {"p_pick_s": None, "s_pick_s": None, "p_prob": 0.0, "s_prob": 0.0,
                  "detected": False, "offset_from_origin_s": None, "error": str(e)[:80]}

    # ── GPD (sliding window classify) ───────────────────────────────────────
    try:
        # GPD classify works on a Stream directly with annotate
        ann_gpd = gpd.annotate(st)
        gpd_res = pick_from_annotation(ann_gpd, "GPD", P_THRESHOLD, S_THRESHOLD, ORIGIN_S, DETECT_TOL)
    except Exception as e:
        gpd_res = {"p_pick_s": None, "s_pick_s": None, "p_prob": 0.0, "s_prob": 0.0,
                   "detected": False, "offset_from_origin_s": None, "error": str(e)[:80]}

    results.append({
        "npz":         npz_stem,
        "window_type": wtype,
        "station":     sta,
        "event_id":    str(row["event_id"]),
        "magnitude":   mag,
        "stalta":      {k: sl.get(k) for k in ["detected","first_trigger_s","peak_stalta","offset_from_origin_s"]},
        "phasenet":    pn_res,
        "eqtransformer": eq_res,
        "gpd":         gpd_res,
    })

    n_done += 1
    if n_done % 20 == 0:
        print(f"    {n_done}/{len(ok_inv)} windows processed...")

# ── Save detection_results.json ────────────────────────────────────────────────
output = {
    "phase": "3",
    "mode":  "zero-shot (no fine-tuning)",
    "models": ["STA/LTA", "PhaseNet", "EQTransformer", "GPD"],
    "parameters": {
        "p_threshold": P_THRESHOLD,
        "s_threshold": S_THRESHOLD,
        "detect_tolerance_s": DETECT_TOL,
        "origin_s": ORIGIN_S,
    },
    "results": results,
}
with open(OUT_JSON, "w") as f:
    json.dump(output, f, indent=2)
print(f"\n[3] Saved detection_results.json ({len(results)} entries)")

# ── Summary table ──────────────────────────────────────────────────────────────
ev_r  = [r for r in results if r["window_type"] == "event"]
ns_r  = [r for r in results if r["window_type"] == "noise"]
models_info = [
    ("STA/LTA",       lambda r: r["stalta"].get("detected", False)),
    ("PhaseNet",      lambda r: r["phasenet"]["detected"]),
    ("EQTransformer", lambda r: r["eqtransformer"]["detected"]),
    ("GPD",           lambda r: r["gpd"]["detected"]),
]

print("\n" + "─" * 60)
print(f"{'Model':<18} {'Events det.':<15} {'Noise trig.':<15}")
print("─" * 60)
for name, fn in models_info:
    ev_det = sum(1 for r in ev_r if fn(r))
    ns_det = sum(1 for r in ns_r if fn(r))
    print(f"{name:<18} {ev_det}/{len(ev_r)} ({100*ev_det/max(len(ev_r),1):.0f}%)      "
          f"{ns_det}/{len(ns_r)} ({100*ns_det/max(len(ns_r),1):.0f}%)")
print("─" * 60)
print("  [Note: noise window labels unreliable — see limitations.md L2]")

# ── Figure 1: Waveform + picks for 5 events ────────────────────────────────────
print("\n[4] Generating waveform_with_picks_examples.png...")

ev_npzs = [p for p in WIN_DIR.glob("event_*.npz")]
# Prefer M>=3 events for visibility
scored = []
for p in ev_npzs:
    n = np.load(str(p), allow_pickle=True)
    scored.append((float(n.get("magnitude", 0)), p))
scored.sort(key=lambda x: -x[0])
example_npzs = [p for _, p in scored[:5]]

# Find corresponding detection results
res_map = {r["npz"]: r for r in results}

fig, axes = plt.subplots(5, 3, figsize=(16, 14), squeeze=False)
fig.suptitle("Event Waveforms with P/S Pick Estimates — PhaseNet / EQTransformer / GPD (zero-shot)\n"
             "Kahramanmaraş Pilot, KO Network, HH Broadband",
             fontsize=11, fontweight="bold")

pick_colors = {"PhaseNet_P": "#e74c3c", "PhaseNet_S": "#e74c3c",
               "EQT_P":      "#2980b9", "EQT_S":      "#2980b9",
               "GPD_P":      "#27ae60"}
chan_colors  = ["#1f77b4", "#ff7f0e", "#2ca02c"]
t_ax = np.linspace(0, WIN_LEN_S, 6000)

for row_i, npz_path in enumerate(example_npzs):
    npz  = np.load(str(npz_path), allow_pickle=True)
    data = npz["data"]
    mag  = float(npz.get("magnitude", 0))
    sta  = str(npz.get("station", ""))
    r    = res_map.get(npz_path.name, {})

    for ci, chan in enumerate(["HHZ", "HHN", "HHE"]):
        ax = axes[row_i][ci]
        ax.plot(t_ax, data[ci], lw=0.5, color=chan_colors[ci], alpha=0.85)
        ax.axvline(ORIGIN_S, color="black", lw=1.2, ls="-", label="Origin" if ci == 0 else "")

        if ci == 0:  # Only annotate picks on Z component
            pn  = r.get("phasenet", {})
            eq  = r.get("eqtransformer", {})
            gp  = r.get("gpd", {})

            if pn.get("p_pick_s") is not None:
                ax.axvline(pn["p_pick_s"], color="#e74c3c", lw=1.2, ls="--",
                           label=f"PN P({pn['p_prob']:.2f})")
            if pn.get("s_pick_s") is not None:
                ax.axvline(pn["s_pick_s"], color="#e74c3c", lw=1.2, ls=":",
                           label=f"PN S({pn['s_prob']:.2f})")
            if eq.get("p_pick_s") is not None:
                ax.axvline(eq["p_pick_s"], color="#2980b9", lw=1.2, ls="--",
                           label=f"EQT P({eq['p_prob']:.2f})")
            if eq.get("s_pick_s") is not None:
                ax.axvline(eq["s_pick_s"], color="#2980b9", lw=1.2, ls=":",
                           label=f"EQT S({eq['s_prob']:.2f})")
            if gp.get("p_pick_s") is not None:
                ax.axvline(gp["p_pick_s"], color="#27ae60", lw=1.2, ls="--",
                           label=f"GPD P({gp['p_prob']:.2f})")

            ax.legend(fontsize=6, loc="upper right", framealpha=0.7)

        ax.set_title(f"{sta} {chan}  M{mag:.1f}", fontsize=8)
        ax.set_xlim(0, WIN_LEN_S)
        if row_i == 4:
            ax.set_xlabel("Time (s)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_ylabel("Norm. ampl.", fontsize=7)

plt.tight_layout()
plt.savefig(FIG_DIR / "waveform_with_picks_examples.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"    Saved → figures/waveform_with_picks_examples.png")

# ── Figure 2: STA/LTA vs DL detection rate comparison ─────────────────────────
print("\n[5] Generating stalta_vs_dl_comparison.png...")

bands  = [(2, 3, "M 2–3"), (3, 4, "M 3–4"), (4, 99, "M≥4")]
model_names = ["STA/LTA", "PhaseNet", "EQTransformer", "GPD"]
fns = [
    lambda r: r["stalta"].get("detected", False),
    lambda r: r["phasenet"]["detected"],
    lambda r: r["eqtransformer"]["detected"],
    lambda r: r["gpd"]["detected"],
]
colors = ["#95a5a6", "#e74c3c", "#2980b9", "#27ae60"]

fig, axes = plt.subplots(1, len(bands), figsize=(14, 5), sharey=True)
fig.suptitle("Detection Rate by Magnitude Band — Zero-Shot Baseline Comparison\n"
             "Kahramanmaraş Pilot, KO.KOZT + KO.KMRS, Phase 3",
             fontsize=11, fontweight="bold")

x = np.arange(len(model_names))
bar_width = 0.6

for bi, (lo, hi, label) in enumerate(bands):
    ax = axes[bi]
    band_ev = [r for r in ev_r if lo <= r["magnitude"] < hi]
    if not band_ev:
        ax.set_title(f"{label}\n(n=0)")
        continue

    rates = []
    for fn in fns:
        det = sum(1 for r in band_ev if fn(r))
        rates.append(100 * det / len(band_ev))

    bars = ax.bar(x, rates, width=bar_width, color=colors, alpha=0.85, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=20, ha="right", fontsize=9)
    ax.set_title(f"{label}\n(n={len(band_ev)})", fontsize=10)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Detection rate (%)" if bi == 0 else "", fontsize=9)
    ax.axhline(50, color="gray", ls="--", lw=0.8, alpha=0.5)

    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{rate:.0f}%", ha="center", va="bottom", fontsize=8)

    ax.tick_params(labelsize=8)

plt.tight_layout()
plt.savefig(FIG_DIR / "stalta_vs_dl_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"    Saved → figures/stalta_vs_dl_comparison.png")

# ── Final summary ──────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PHASE 3 — COMPLETE")
print(f"  detection_results.json : {OUT_JSON}")
print(f"  Figures                : figures/waveform_with_picks_examples.png")
print(f"                           figures/stalta_vs_dl_comparison.png")
print("=" * 70)
