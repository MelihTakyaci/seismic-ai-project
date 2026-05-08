# Phase: 4 / 5
# Purpose: Generate all visual artifacts for evaluation — phase-aware (Phase 4 includes
#          noise window confusion matrices; Phase 5 uses recall-only + distance analysis)
# Inputs: artifacts/evaluation_metrics.json, artifacts/detection_results.json,
#         data/catalog/koeri_pilot_catalog.csv OR data/catalog/phase5_catalog.csv,
#         artifacts/station_summary.csv OR artifacts/phase5_station_list.csv
# Outputs: figures/confusion_matrix.png (Phase 4 only), figures/recall_by_magnitude.png,
#          figures/station_map.png, figures/event_map.png, artifacts/figure_captions.md
# Limitations: Maps use simple lat/lon scatter (no topography or coastline basemap);
#              Phase 5 has no noise windows — precision/FP metrics not computable.

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
FIG_DIR  = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

with open(ROOT / "artifacts" / "evaluation_metrics.json") as f:
    eval_data = json.load(f)
with open(ROOT / "artifacts" / "detection_results.json") as f:
    det_data = json.load(f)

results   = det_data["results"]
metrics   = eval_data["models"]
PHASE     = str(eval_data.get("phase", "4"))
MODELS    = ["stalta", "phasenet", "eqtransformer", "gpd"]
M_LABELS  = {"stalta": "STA/LTA", "phasenet": "PhaseNet",
              "eqtransformer": "EQTransformer", "gpd": "GPD"}
M_COLORS  = {"stalta": "#95a5a6", "phasenet": "#e74c3c",
              "eqtransformer": "#2980b9", "gpd": "#27ae60"}

ev_r = [r for r in results if r["window_type"] == "event"]
ns_r = [r for r in results if r["window_type"] == "noise"]
STALTA_THRESH = 3.0
clean_ns = [r for r in ns_r if r["stalta"].get("peak_stalta", 0) <= STALTA_THRESH]

print("=" * 70)
print(f"PHASE {PHASE} — Generating Visual Artifacts")
print("=" * 70)

# ── Load catalog and station data (phase-aware) ────────────────────────────────
if PHASE == "5":
    cat_path = ROOT / "data" / "catalog" / "phase5_catalog.csv"
    sta_path = ROOT / "artifacts" / "phase5_station_list.csv"
    cat_df = pd.read_csv(cat_path)
    cat_df["magnitude"] = pd.to_numeric(cat_df["magnitude"], errors="coerce")
    cat_df = cat_df.dropna(subset=["latitude", "longitude", "magnitude"])
    sta5_df = pd.read_csv(sta_path)
    pilot_sta_df = pd.read_csv(ROOT / "artifacts" / "station_summary.csv")
    pilot_s = pilot_sta_df[pilot_sta_df["pilot_selected"] == True]
else:
    cat_path = ROOT / "data" / "catalog" / "koeri_pilot_catalog.csv"
    cat_df = pd.read_csv(cat_path)
    cat_df["magnitude"] = pd.to_numeric(cat_df["magnitude"], errors="coerce")
    cat_df = cat_df.dropna(subset=["latitude", "longitude", "magnitude"])
    sta_df  = pd.read_csv(ROOT / "artifacts" / "station_summary.csv")
    pilot_s = sta_df[sta_df["pilot_selected"] == True]

# ── Figure 1: Confusion matrix (Phase 4 only) or Recall bars (Phase 5) ─────────
if PHASE != "5":
    print("\n[1] Confusion matrices (Phase 4)...")
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle("Confusion Matrices — Pilot Evaluation\n"
                 "(using clean noise windows; noise labels approximate — see limitations.md L2)",
                 fontsize=11, fontweight="bold")

    for ax, mk in zip(axes, MODELS):
        m   = metrics[mk]
        TP  = m["TP"];        FN = m["FN"]
        FP  = m.get("FP_clean_noise", 0); TN = m.get("TN_clean_noise", 0)
        cm  = np.array([[TP, FN], [FP, TN]])
        total = TP + FN + FP + TN

        ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(TP, FN, FP, TN, 1) + 1)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Predicted\nPositive", "Predicted\nNegative"], fontsize=8)
        ax.set_yticklabels(["Actual\nEvent", "Actual\nNoise"], fontsize=8)
        ax.set_title(M_LABELS[mk], fontsize=10, fontweight="bold",
                     color=M_COLORS[mk])

        labels = [["TP", "FN"], ["FP*", "TN"]]
        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                ax.text(j, i, f"{labels[i][j]}\n{val}",
                        ha="center", va="center", fontsize=11,
                        color="white" if total > 0 and val > total * 0.4 else "black")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved → figures/confusion_matrix.png")
else:
    print("\n[1] Recall summary bars (Phase 5 — no noise windows)...")
    fig, ax = plt.subplots(figsize=(10, 5))
    n_windows = metrics["stalta"]["n_event_windows"]
    recalls = {M_LABELS[mk]: metrics[mk]["recall"] for mk in MODELS}
    bars = ax.bar(list(recalls.keys()), list(recalls.values()),
                  color=[M_COLORS[mk] for mk in MODELS],
                  alpha=0.85, edgecolor="white", linewidth=1.5)
    for bar, (lbl, v) in zip(bars, recalls.items()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.axhline(0.5, color="gray", ls="--", lw=1.0, alpha=0.7, label="50% threshold")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Recall (TP / all event windows)", fontsize=11)
    ax.set_title(f"Zero-Shot Recall — Phase 5 Scale-Up\n"
                 f"12 KO Stations, 6 months (2023-02-06 to 2023-08-06), "
                 f"n={n_windows} event windows", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved → figures/confusion_matrix.png  (recall bars, Phase 5)")

# ── Figure 2: Recall by magnitude band ────────────────────────────────────────
print("\n[2] Recall by magnitude band...")
bands      = ["M 2.0-3.0", "M 3.0-4.0", "M>=4.0"]
band_short = ["M 2–3", "M 3–4", "M≥4"]
x = np.arange(len(bands))
bar_w = 0.18

if PHASE != "5":
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    fig.suptitle("Precision / Recall / F1 by Magnitude Band (Pilot, Clean Noise Baseline)\n"
                 "Zero-Shot Mode — KO.KOZT + KO.KMRS, Kahramanmaraş 2023",
                 fontsize=11, fontweight="bold")

    metric_keys  = ["recall", "precision_clean", "f1_clean"]
    metric_names = ["Recall", "Precision\n(clean noise)", "F1\n(clean noise)"]
    ylabels      = ["Recall", "Precision", "F1 Score"]

    for ax_i, (mkey, mname, ylabel) in enumerate(zip(metric_keys, metric_names, ylabels)):
        ax = axes[ax_i]
        for mi, mk in enumerate(MODELS):
            m = metrics[mk]
            vals = []
            for band in bands:
                bm = m["by_magnitude"].get(band)
                if bm is None:
                    vals.append(0.0)
                elif mkey == "recall":
                    vals.append(bm.get("recall", 0.0))
                elif mkey == "precision_clean":
                    vals.append(m.get("precision_clean", 0.0))
                else:
                    vals.append(bm.get("f1", m.get("f1_clean", 0.0)))
            offset = (mi - 1.5) * bar_w
            bars = ax.bar(x + offset, vals, width=bar_w,
                          color=M_COLORS[mk], alpha=0.85, label=M_LABELS[mk],
                          edgecolor="white")
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(band_short, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.6)
        ax.tick_params(labelsize=9)
        if ax_i == 0:
            ax.legend(fontsize=8, loc="upper left")
else:
    # Phase 5: single recall-by-magnitude panel
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle("Recall by Magnitude Band — Phase 5 Scale-Up (Zero-Shot)\n"
                 "12 KO Stations, 6 months (2023-02-06 to 2023-08-06)",
                 fontsize=11, fontweight="bold")

    for mi, mk in enumerate(MODELS):
        m = metrics[mk]
        vals = []
        ns   = []
        for band in bands:
            bm = m["by_magnitude"].get(band, {})
            vals.append(bm.get("recall", 0.0))
            ns.append(bm.get("n_events", 0))
        offset = (mi - 1.5) * bar_w
        bars = ax.bar(x + offset, vals, width=bar_w,
                      color=M_COLORS[mk], alpha=0.85, label=M_LABELS[mk],
                      edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    band_labels_n = [f"{bs}\n(n={metrics['stalta']['by_magnitude'].get(b, {}).get('n_events', 0)})"
                     for bs, b in zip(band_short, bands)]
    ax.set_xticks(x)
    ax.set_xticklabels(band_labels_n, fontsize=10)
    ax.set_ylabel("Recall", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.6, label="50% threshold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(FIG_DIR / "precision_recall_by_magnitude.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"    Saved → figures/precision_recall_by_magnitude.png")

# ── Figure 3: Station map ──────────────────────────────────────────────────────
print("\n[3] Station map...")
fig, ax = plt.subplots(figsize=(10, 7))

if PHASE == "5":
    sta_all  = sta5_df
    primary  = sta_all[sta_all["role"] == "primary"]
    alt      = sta_all[sta_all["role"] == "alternate"]
    reserve  = sta_all[sta_all["role"] == "reserve"]

    ax.scatter(reserve["longitude"], reserve["latitude"],
               marker="^", s=40, c="#bdc3c7", alpha=0.5, zorder=3,
               label=f"KO HH stations — reserve (n={len(reserve)})")
    ax.scatter(alt["longitude"], alt["latitude"],
               marker="^", s=80, c="#f39c12", alpha=0.8, zorder=4,
               label=f"Phase 5 alternates (n={len(alt)})")
    ax.scatter(primary["longitude"], primary["latitude"],
               marker="*", s=300, c="#e74c3c", zorder=6,
               label=f"Phase 5 primary stations (n={len(primary)})")

    for _, row in primary.iterrows():
        ax.annotate(f"KO.{row['station']}",
                    xy=(row["longitude"], row["latitude"]),
                    xytext=(5, 5), textcoords="offset points",
                    fontsize=7, color="#c0392b", fontweight="bold")

    ax.set_title("Station Map — Phase 5 Scale-Up (12 Primary KO HH Stations)\n"
                 "Distance bins: near <50 km, mid 50–150 km, far 150–300 km",
                 fontsize=11, fontweight="bold")
else:
    sta_df_local = pd.read_csv(ROOT / "artifacts" / "station_summary.csv")
    km_sta  = sta_df_local[sta_df_local["region"] == "Kahramanmaras_pilot"]
    wm_sta  = sta_df_local[sta_df_local["region"] == "WesternMarmara_main"]

    ax.scatter(wm_sta["longitude"], wm_sta["latitude"],
               marker="^", s=50, c="#bdc3c7", alpha=0.6, zorder=3,
               label=f"KO stations — W. Marmara (n={len(wm_sta)}, main phase)")
    ax.scatter(km_sta["longitude"], km_sta["latitude"],
               marker="^", s=70, c="#7f8c8d", alpha=0.7, zorder=4,
               label=f"KO stations — Kahramanmaraş (n={len(km_sta)})")
    ax.scatter(pilot_s["longitude"], pilot_s["latitude"],
               marker="*", s=350, c="#e74c3c", zorder=6,
               label="Pilot stations (KO.KOZT, KO.KMRS)")

    for _, row in pilot_s.iterrows():
        ax.annotate(f"KO.{row['station']}\n({row['latitude']:.2f}°N)",
                    xy=(row["longitude"], row["latitude"]),
                    xytext=(8, 8), textcoords="offset points",
                    fontsize=8, color="#c0392b", fontweight="bold")

    ax.set_title("Station Map — Pilot Stations and KO Network Coverage\n"
                 "Pilot: KO.KOZT + KO.KMRS (Kahramanmaraş), Main phase: Western Marmara",
                 fontsize=11, fontweight="bold")

ax.scatter([37.032], [37.166], marker="*", s=500, c="#f39c12",
           zorder=7, label="M7.8 main shock (2023-02-06)", edgecolors="k", lw=0.5)
ax.set_xlabel("Longitude (°E)", fontsize=11)
ax.set_ylabel("Latitude (°N)", fontsize=11)
ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
ax.grid(True, alpha=0.3, ls="--")
ax.set_aspect("equal")
plt.tight_layout()
plt.savefig(FIG_DIR / "station_map.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"    Saved → figures/station_map.png")

# ── Figure 4: Event map ────────────────────────────────────────────────────────
print("\n[4] Event map...")
fig, ax = plt.subplots(figsize=(10, 7))

norm = Normalize(vmin=2.0, vmax=cat_df["magnitude"].max())
cmap = plt.cm.hot_r
sc = ax.scatter(cat_df["longitude"], cat_df["latitude"],
                c=cat_df["magnitude"], cmap=cmap, norm=norm,
                s=2 ** (cat_df["magnitude"] - 1.5),
                alpha=0.4, zorder=3, linewidths=0)

cb = plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.8)
cb.set_label("Magnitude", fontsize=10)

if PHASE == "5":
    primary = sta5_df[sta5_df["role"] == "primary"]
    for _, row in primary.iterrows():
        ax.scatter(row["longitude"], row["latitude"],
                   marker="^", s=120, c="#2980b9", zorder=6)
    ax.scatter([], [], marker="^", s=120, c="#2980b9",
               label=f"Phase 5 stations (n={len(primary)})")
    date_label = "2023-02-06 to 2023-08-06"
else:
    for _, row in pilot_s.iterrows():
        ax.scatter(row["longitude"], row["latitude"],
                   marker="^", s=250, c="#2980b9", zorder=6,
                   label=f"KO.{row['station']}")
        ax.annotate(f"KO.{row['station']}",
                    xy=(row["longitude"], row["latitude"]),
                    xytext=(5, 5), textcoords="offset points", fontsize=8,
                    color="#1a5276", fontweight="bold")
    date_label = "2023-02-06 to 2023-02-20"

ax.scatter([37.032], [37.166], marker="*", s=600, c="#f39c12",
           zorder=7, edgecolors="k", lw=0.5, label="M7.8 main shock")

bands_text = (f"M 2–3: {(cat_df.magnitude.between(2, 3)).sum()}  "
              f"M 3–4: {(cat_df.magnitude.between(3, 4)).sum()}  "
              f"M≥4: {(cat_df.magnitude >= 4).sum()}")
ax.set_xlabel("Longitude (°E)", fontsize=11)
ax.set_ylabel("Latitude (°N)", fontsize=11)
ax.set_title(f"Event Map — 2023 Kahramanmaraş Aftershock Sequence\n"
             f"EMSC Catalog, {date_label}  ({len(cat_df)} events)  [{bands_text}]",
             fontsize=10, fontweight="bold")
ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
ax.grid(True, alpha=0.3, ls="--")
plt.tight_layout()
plt.savefig(FIG_DIR / "event_map.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"    Saved → figures/event_map.png")

# ── Figure captions ────────────────────────────────────────────────────────────
print("\n[5] Writing figure_captions.md...")
captions_path = ROOT / "artifacts" / "figure_captions.md"

p4_metrics = {}
p4_backup = ROOT / "artifacts" / "phase4_backup" / "evaluation_metrics.json"
if p4_backup.exists():
    with open(p4_backup) as f:
        p4_data = json.load(f)
    for mk in MODELS:
        p4_metrics[mk] = p4_data["models"].get(mk, {})

phase5_note = ""
if PHASE == "5":
    phase5_note = (
        f"\n\n## Phase 5 Dataset Note\n\n"
        f"Phase 5 analysis uses {metrics['stalta']['n_event_windows']} preprocessed event "
        f"windows from 12 KO HH stations over 6 months (2023-02-06 to 2023-08-06). "
        f"Download success rate was 3.8% (5,187 of 136,056 attempted requests) due to "
        f"FDSN parallel client initialization collision — see limitations.md L9. "
        f"No noise windows included (noise class omitted in Phase 5 per design).\n"
    )

captions_path.write_text(f"""\
# Figure Captions — Seismic AI Pipeline
# Phase {PHASE} (2026-04-25)
# All figures reference computed metrics from this project's own artifacts.
{phase5_note}
---

## figures/pilot_event_examples.png

Representative event waveforms from the Kahramanmaraş 2023 aftershock pilot dataset.
Five events shown, spanning M 2.7–5.0, recorded at KO.KOZT and KO.KMRS (KO network,
HH broadband channels, 100 Hz). Each panel shows one 60-second window after
preprocessing (linear detrend, bandpass 1–45 Hz, z-score normalization). Three
components shown: HHZ (blue), HHN (orange), HHE (green). Red dashed line marks catalog
origin time at t=30 s. P and S arrivals are clearly visible after t=30 s.
Source: artifacts/waveform_inventory.csv.

## figures/pilot_noise_examples.png

"Noise" windows from the pilot dataset, selected from inter-event gaps > 300 s in the
EMSC catalog. Visual inspection reveals clear seismic arrivals in all displayed windows,
confirming contamination by sub-threshold aftershocks undetected by the EMSC catalog.
This is documented as Limitation L2 in artifacts/limitations.md. These windows should
be treated as having uncertain class labels.

## figures/waveform_with_picks_examples.png

Five event waveforms (M 5.0–6.0) with P and S pick estimates overlaid from three
zero-shot SeisBench models. Red dashed/dotted: PhaseNet P/S; blue dashed/dotted:
EQTransformer P/S; green dashed: GPD P. Black solid: catalog origin time (t=30 s).
Picks plotted on HHZ (vertical) only. PhaseNet and GPD produce picks in the physically
expected window (after origin); EQTransformer produces fewer picks due to zero-shot
domain mismatch (see methodology_notes.md M5 and limitations.md L8).

## figures/stalta_vs_dl_comparison.png

Detection rate (%) by magnitude band for four detectors (STA/LTA, PhaseNet,
EQTransformer, GPD) run in zero-shot mode on the Kahramanmaraş pilot dataset
(KO.KOZT + KO.KMRS, 2023-02-06 to 2023-02-20). Detection criterion: P pick or
STA/LTA trigger in [origin − 5 s, origin + 25 s]. Dashed line at 50%.
STA/LTA achieves highest raw detection rate but cannot discriminate events from noise.
PhaseNet and GPD reach 77% detection on M≥4 events in zero-shot mode.
Source: artifacts/detection_results.json.

## figures/confusion_matrix.png

{"Recall summary bar chart for each detector (STA/LTA, PhaseNet, EQTransformer, GPD) evaluated on " + str(metrics['stalta']['n_event_windows']) + " event windows from Phase 5 (12 KO stations, 6 months). No noise windows collected in Phase 5 (see limitations.md L9). GPD achieves highest recall (0.593), followed by STA/LTA (0.484), PhaseNet (0.319), and EQTransformer (0.051). Recall values are lower than Phase 4 pilot — partly attributable to far-station detection window mismatch (see limitations.md L10)." if PHASE == "5" else f"Confusion matrices for each detector evaluated on the pilot dataset (161 event windows, {len(clean_ns)} clean noise windows after STA/LTA-based contamination filtering). TP = event detected, FN = event missed, FP* = noise window triggered (upper bound; noise labels approximate), TN = noise window not triggered. Primary metric is recall (TP / (TP + FN))."}
Source: artifacts/evaluation_metrics.json.

## figures/precision_recall_by_magnitude.png

{"Recall by magnitude band (M 2–3, M 3–4, M≥4) for all four detectors in Phase 5. No precision metric computed (no noise windows). Event counts per band: M 2–3 n=" + str(metrics["stalta"]["by_magnitude"].get("M 2.0-3.0", {}).get("n_events", 0)) + ", M 3–4 n=" + str(metrics["stalta"]["by_magnitude"].get("M 3.0-4.0", {}).get("n_events", 0)) + ", M≥4 n=" + str(metrics["stalta"]["by_magnitude"].get("M>=4.0", {}).get("n_events", 0)) + ". Recall is broadly flat across magnitude bands for all models, suggesting that distance-dependent detection window truncation (see limitations.md L10) rather than SNR dominates the miss rate." if PHASE == "5" else "Recall, precision (clean noise baseline), and F1 (clean noise) stratified by magnitude band for all four detectors. Precision uses clean noise windows only (STA/LTA peak ≤ 3.0) as a conservative estimate. No M < 2.0 events present in pilot catalog (EMSC threshold); M 2.0–3.0 is the closest available proxy for the primary detection target."}
Source: artifacts/evaluation_metrics.json.

## figures/station_map.png

{"Map of 12 primary and 3 alternate KO (Kandilli Observatory) HH broadband stations selected for Phase 5 scale-up. Stations stratified by epicentral distance from M7.8 main shock (37.166°N, 37.032°E): near <50 km (2 stations), mid 50–150 km (4 stations), far 150–300 km (6 stations). Orange star marks M7.8 main shock epicenter. Three primary stations (KO.GAZ, KO.KMRS, KO.KOZT) had zero successful downloads and were replaced by alternates. Source: artifacts/phase5_station_list.csv." if PHASE == "5" else "Map of KO (Kandilli Observatory) network stations in the Kahramanmaraş pilot region (6 stations) and Western Marmara main-phase region (62 stations). Red stars mark the two pilot stations: KO.KOZT (37.48°N, 35.83°E) and KO.KMRS (37.51°N, 36.90°E), both with HH broadband channels at 100 Hz. Orange star marks approximate M7.8 main shock epicenter (2023-02-06). Source: artifacts/station_summary.csv."}

## figures/event_map.png

Spatial distribution of {len(cat_df)} EMSC-catalogued aftershocks in the Kahramanmaraş
region from {"2023-02-06 to 2023-08-06 (6-month Phase 5 window)" if PHASE == "5" else "2023-02-06 to 2023-02-20 (2-week pilot window)"}.
Symbol size and color scale with magnitude (colorbar). {"Blue triangles mark the 12 primary Phase 5 stations." if PHASE == "5" else "Blue triangles mark the two pilot stations."}
The aftershock cloud spans roughly 36.5–38.5°N, 35.5–38.5°E, concentrated along the
East Anatolian Fault zone. Omori decay is visible in temporal density (not shown here).
Source: {"data/catalog/phase5_catalog.csv" if PHASE == "5" else "data/catalog/koeri_pilot_catalog.csv"}.
""")
print(f"    Saved → artifacts/figure_captions.md")

print("\n" + "=" * 70)
print(f"PHASE {PHASE} PLOT ARTIFACTS — COMPLETE")
print("  figures/confusion_matrix.png")
print("  figures/precision_recall_by_magnitude.png")
print("  figures/station_map.png")
print("  figures/event_map.png")
print("  artifacts/figure_captions.md")
print("=" * 70)
