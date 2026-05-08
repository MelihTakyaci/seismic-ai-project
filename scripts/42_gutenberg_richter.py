# Phase: PSHA Step 1
# Purpose: Gutenberg-Richter b-value estimation and long-term seismic hazard assessment.
#          Uses past seismicity statistics to estimate exceedance probabilities.
#          This is sismik tehlike analizi (seismic hazard assessment),
#          NOT deprem tahmini (earthquake prediction).
# Inputs:  data/catalog/phase5_catalog.csv (11,338 events M>=2.0),
#          data/catalog/small_events_catalog.csv (44,214 events M<2.0)
# Outputs: artifacts/gutenberg_richter_params.json,
#          artifacts/spatial_hazard_grid.csv,
#          figures/gutenberg_richter_fit.png,
#          figures/b_value_map.png
# Limitations: Catalog covers ~6-month aftershock-dominated period (2023).
#              Annual rates are extrapolated; may overestimate background seismicity.
#              Spatial cells with N<20 events have high uncertainty on b-value.
#              Poisson assumption may not hold during active aftershock sequences.

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

ROOT    = Path(__file__).resolve().parent.parent
CAT_DIR = ROOT / "data" / "catalog"
ART     = ROOT / "artifacts"
FIG     = ROOT / "figures"

print("=" * 70)
print("PSHA STEP 1 — Gutenberg-Richter Analysis")
print("  Olasılıksal Sismik Tehlike Analizi (OSTA)")
print("  Long-term statistical hazard — NOT earthquake prediction")
print("=" * 70)

# ── Step 1: Load and merge catalogs ───────────────────────────────────────
print("\n[1] Loading catalogs ...")

p5  = pd.read_csv(CAT_DIR / "phase5_catalog.csv")
sm  = pd.read_csv(CAT_DIR / "small_events_catalog.csv")

# Harmonise columns
p5 = p5[["origin_time", "latitude", "longitude", "depth_km", "magnitude"]].copy()
sm = sm[["origin_time", "latitude", "longitude", "depth_km", "magnitude"]].copy()

cat = pd.concat([p5, sm], ignore_index=True)
cat["origin_time"] = pd.to_datetime(cat["origin_time"], utc=True, errors="coerce")
cat = cat.dropna(subset=["magnitude", "latitude", "longitude", "origin_time"])
cat["magnitude"] = pd.to_numeric(cat["magnitude"], errors="coerce")
cat = cat.dropna(subset=["magnitude"])
cat = cat.sort_values("origin_time").reset_index(drop=True)

# Observation period
t_start = cat["origin_time"].min()
t_end   = cat["origin_time"].max()
T_years = (t_end - t_start).days / 365.25
if T_years < 0.1:
    T_years = 0.5   # safety fallback

print(f"    Phase5 catalog:       {len(p5):,} events (M≥2.0)")
print(f"    Small events catalog: {len(sm):,} events (M<2.0)")
print(f"    Combined (unique):    {len(cat):,} events")
print(f"    Date range:  {t_start.date()} → {t_end.date()}  ({T_years:.2f} years)")
print(f"    Magnitude range: {cat.magnitude.min():.1f} – {cat.magnitude.max():.1f}")

# ── Step 2: Geographic bounding box ───────────────────────────────────────
lat_lo, lat_hi = 36.0, 39.0
lon_lo, lon_hi = 35.0, 39.5

region = cat[
    (cat.latitude  >= lat_lo) & (cat.latitude  <= lat_hi) &
    (cat.longitude >= lon_lo) & (cat.longitude <= lon_hi)
].copy()
print(f"\n[2] Kahramanmaraş region ({lat_lo}-{lat_hi}°N, {lon_lo}-{lon_hi}°E): {len(region):,} events")

# ── Step 3: Global G-R analysis ────────────────────────────────────────────
print("\n[3] Global Gutenberg-Richter analysis ...")

BIN_W = 0.1
mag_bins = np.arange(0.0, 8.05, BIN_W)
bin_centers = mag_bins[:-1] + BIN_W / 2

mag_arr = region["magnitude"].values

# Frequency counts per bin
counts, _ = np.histogram(mag_arr, bins=mag_bins)

# Cumulative counts N(≥M)
cum_counts = np.cumsum(counts[::-1])[::-1]

# Magnitude of Completeness (Mc) — Maximum Curvature method
# Peak of non-cumulative frequency; +0.2 standard correction (Woessner & Wiemer 2005)
maxc_idx = np.argmax(counts)
Mc_maxc  = bin_centers[maxc_idx] + 0.2   # standard MAXC correction
Mc_maxc  = round(Mc_maxc, 1)
# Enforce Mc >= 2.0: EMSC catalog (the reliable seismicity source) starts at M2.0.
# Using Mc < 2.0 would mix AFAD sub-threshold events with EMSC rates — systematically
# biasing b-value low. Mc=max(MAXC, 2.0) gives the more reliable estimate.
Mc = max(Mc_maxc, 2.0)
print(f"    Mc (MAXC raw):      {Mc_maxc:.1f}  → enforced Mc: {Mc:.1f} (EMSC catalog floor)")

# Events above Mc
mask_mc = mag_arr >= Mc
N_mc    = mask_mc.sum()
mean_M  = mag_arr[mask_mc].mean()

# Annual rate for M >= Mc
N_annual_mc = N_mc / T_years

# b-value via Maximum Likelihood (Aki 1965)
# b = log10(e) / (mean_M - Mc + delta_M/2)
delta_M = BIN_W
b_value = np.log10(np.e) / (mean_M - Mc + delta_M / 2)

# Uncertainty (Shi & Bolt 1982)
b_std = 2.30 * b_value**2 * np.sqrt(np.var(mag_arr[mask_mc]) / N_mc)

# a-value
a_value = np.log10(N_annual_mc) + b_value * Mc

print(f"    Mc (MAXC method):   {Mc:.1f}")
print(f"    N events ≥ Mc:      {N_mc:,}")
print(f"    Mean magnitude:     {mean_M:.3f}")
print(f"    b-value (MLE):      {b_value:.4f} ± {b_std:.4f}")
print(f"    a-value:            {a_value:.4f}")

# Validate b-value is realistic
if 0.6 <= b_value <= 1.5:
    print(f"    ✓ b-value in expected range [0.6, 1.5]")
else:
    print(f"    ⚠ b-value {b_value:.3f} outside typical [0.8, 1.2] — check catalog completeness")

# Annual rates and return periods for magnitude thresholds
thresholds = [3.0, 4.0, 5.0, 6.0, 7.0]
rates = {}
for M_thr in thresholds:
    lam = 10 ** (a_value - b_value * M_thr)
    T_ret = 1 / lam if lam > 0 else np.inf
    rates[f"M{M_thr:.0f}"] = {
        "magnitude_threshold": M_thr,
        "annual_rate":         round(lam, 6),
        "return_period_yr":    round(T_ret, 1),
        "prob_10yr":           round(1 - np.exp(-lam * 10), 4),
        "prob_50yr":           round(1 - np.exp(-lam * 50), 4),
        "prob_100yr":          round(1 - np.exp(-lam * 100), 4),
    }
    print(f"    M≥{M_thr:.0f}: λ={lam:.4f}/yr  T_ret={T_ret:.0f}yr  "
          f"P(50yr)={1-np.exp(-lam*50):.3f}")

# ── Step 4: Spatial grid ───────────────────────────────────────────────────
print("\n[4] Computing spatial G-R grid (0.5°×0.5° cells) ...")

GRID_DX = 0.5
MIN_EVENTS = 20   # minimum events per cell for reliable b-value

lat_edges = np.arange(lat_lo, lat_hi + GRID_DX, GRID_DX)
lon_edges = np.arange(lon_lo, lon_hi + GRID_DX, GRID_DX)

grid_rows = []
for i in range(len(lat_edges) - 1):
    for j in range(len(lon_edges) - 1):
        clat = (lat_edges[i] + lat_edges[i + 1]) / 2
        clon = (lon_edges[j] + lon_edges[j + 1]) / 2

        cell_mask = (
            (region.latitude  >= lat_edges[i]) & (region.latitude  < lat_edges[i + 1]) &
            (region.longitude >= lon_edges[j]) & (region.longitude < lon_edges[j + 1])
        )
        cell_mags = region.loc[cell_mask, "magnitude"].values
        N_cell    = len(cell_mags)

        if N_cell < MIN_EVENTS:
            continue

        # Cell Mc (use global Mc if too few events for local estimate)
        cell_counts, _ = np.histogram(cell_mags, bins=mag_bins)
        cell_maxc_idx  = np.argmax(cell_counts)
        cell_Mc = bin_centers[cell_maxc_idx] + 0.1
        cell_Mc = round(cell_Mc, 1)

        cell_mask_mc = cell_mags >= cell_Mc
        if cell_mask_mc.sum() < 10:
            cell_Mc = Mc   # fallback to global
            cell_mask_mc = cell_mags >= cell_Mc
        if cell_mask_mc.sum() < 5:
            continue

        cell_mean_M   = cell_mags[cell_mask_mc].mean()
        cell_N_annual = cell_mask_mc.sum() / T_years
        cell_b = np.log10(np.e) / (cell_mean_M - cell_Mc + delta_M / 2)
        cell_a = np.log10(cell_N_annual) + cell_b * cell_Mc

        lam_M4 = 10 ** (cell_a - cell_b * 4.0)
        lam_M5 = 10 ** (cell_a - cell_b * 5.0)
        lam_M6 = 10 ** (cell_a - cell_b * 6.0)
        P50_M5 = 1 - np.exp(-lam_M5 * 50)
        P50_M6 = 1 - np.exp(-lam_M6 * 50)

        grid_rows.append({
            "lat_center":   round(clat, 3),
            "lon_center":   round(clon, 3),
            "n_events":     N_cell,
            "Mc":           cell_Mc,
            "b_value":      round(cell_b, 4),
            "a_value":      round(cell_a, 4),
            "lambda_M4":    round(lam_M4, 6),
            "lambda_M5":    round(lam_M5, 6),
            "lambda_M6":    round(lam_M6, 6),
            "P50yr_M5":     round(P50_M5, 4),
            "P50yr_M6":     round(P50_M6, 4),
        })

grid_df = pd.DataFrame(grid_rows)
print(f"    Grid cells with ≥{MIN_EVENTS} events: {len(grid_df)}")
if len(grid_df) > 0:
    print(f"    b-value range in grid: {grid_df.b_value.min():.3f} – {grid_df.b_value.max():.3f}")
    print(f"    P(M≥5, 50yr) range:    {grid_df.P50yr_M5.min():.3f} – {grid_df.P50yr_M5.max():.3f}")

# ── Step 5: Save artifacts ─────────────────────────────────────────────────
print("\n[5] Saving artifacts ...")

gr_params = {
    "description": (
        "Gutenberg-Richter parameters for Kahramanmaraş aftershock region. "
        "Long-term statistical seismic hazard (sismik tehlike analizi). "
        "NOT earthquake prediction (deprem tahmini)."
    ),
    "catalog_info": {
        "n_total_events":  len(region),
        "date_start":      str(t_start.date()),
        "date_end":        str(t_end.date()),
        "T_years":         round(T_years, 3),
        "lat_bounds":      [lat_lo, lat_hi],
        "lon_bounds":      [lon_lo, lon_hi],
    },
    "gr_fit": {
        "Mc":       Mc,
        "b_value":  round(b_value, 4),
        "b_std":    round(b_std, 4),
        "a_value":  round(a_value, 4),
        "N_above_Mc": int(N_mc),
        "method":   "Maximum Likelihood (Aki 1965); Mc via MAXC (Woessner & Wiemer 2005)",
    },
    "annual_rates_and_probabilities": rates,
}

with open(ART / "gutenberg_richter_params.json", "w") as f:
    json.dump(gr_params, f, indent=2)
print(f"    Saved → artifacts/gutenberg_richter_params.json")

grid_df.to_csv(ART / "spatial_hazard_grid.csv", index=False)
print(f"    Saved → artifacts/spatial_hazard_grid.csv ({len(grid_df)} cells)")

# ── Step 6: G-R fit figure ─────────────────────────────────────────────────
print("\n[6] Producing figures ...")

fig_gr, axes = plt.subplots(1, 2, figsize=(13, 5))
fig_gr.patch.set_facecolor("#0f1525")

ax1 = axes[0]
ax1.set_facecolor("#0f1525")

# Non-cumulative (histogram bars)
mask_plot = cum_counts > 0
mag_plot  = bin_centers[mask_plot]
cum_plot  = cum_counts[mask_plot] / T_years   # annual rates

ax1.bar(bin_centers, counts / T_years,
        width=BIN_W * 0.9, color="#00d4ff", alpha=0.4, label="Frekans (kümülatif değil)")
ax1.semilogy()

# Cumulative (N ≥ M)
ax1.scatter(mag_plot, cum_plot,
            color="#00ff88", s=25, zorder=5, label="Kümülatif N(≥M)")

# G-R fit line
M_fit = np.linspace(Mc - 0.5, mag_arr.max(), 200)
N_fit = 10 ** (a_value - b_value * M_fit)
ax1.plot(M_fit, N_fit,
         color="#ff6b35", linewidth=2, linestyle="--",
         label=f"G-R eğrisi: b={b_value:.3f}±{b_std:.3f}")

# Mc line
ax1.axvline(x=Mc, color="yellow", linewidth=1.5, linestyle=":",
            label=f"Mc={Mc:.1f} (MAXC)")

ax1.set_xlabel("Magnitüd", color="#e8ecf0", fontsize=11)
ax1.set_ylabel("Yıllık olay sayısı N(≥M)", color="#e8ecf0", fontsize=11)
ax1.set_title("Gutenberg-Richter Magnitüd-Frekans İlişkisi\n"
              "Olasılıksal Sismik Tehlike Analizi — Deprem Tahmini Değildir",
              color="#e8ecf0", fontsize=10, pad=8)
ax1.tick_params(colors="#e8ecf0")
ax1.spines[:].set_color("rgba(0,212,255,0.2)" if False else "#22334a")
for sp in ax1.spines.values(): sp.set_edgecolor("#22334a")
ax1.legend(fontsize=8, facecolor="#1a2540", labelcolor="#e8ecf0", framealpha=0.8)
ax1.set_xlim(0.0, 8.0)
ax1.grid(True, alpha=0.15, color="#00d4ff")

# Poisson probability panel
ax2 = axes[1]
ax2.set_facecolor("#0f1525")
M_range = np.linspace(2.5, 7.5, 200)
for t_exp, color, lbl in [(10, "#00d4ff", "10 yıl"), (50, "#ff6b35", "50 yıl"), (100, "#ff2200", "100 yıl")]:
    lam_arr = 10 ** (a_value - b_value * M_range)
    P_arr   = 1 - np.exp(-lam_arr * t_exp)
    ax2.plot(M_range, P_arr, color=color, linewidth=2, label=f"P({lbl})")

ax2.axhline(y=0.10, color="white", linewidth=0.8, linestyle=":", alpha=0.4)
ax2.axhline(y=0.50, color="white", linewidth=0.8, linestyle=":", alpha=0.4)
ax2.text(7.4, 0.11, "10%", color="white", fontsize=8, alpha=0.5, ha="right")
ax2.text(7.4, 0.51, "50%", color="white", fontsize=8, alpha=0.5, ha="right")

ax2.set_xlabel("Magnitüd eşiği (≥M)", color="#e8ecf0", fontsize=11)
ax2.set_ylabel("Aşım olasılığı P(t)", color="#e8ecf0", fontsize=11)
ax2.set_title("Poisson Aşım Olasılığı — Uzun Vadeli Tehlike\n"
              "(Deprem Tahmini Değildir — İstatistiksel Ortalama)",
              color="#e8ecf0", fontsize=10, pad=8)
ax2.tick_params(colors="#e8ecf0")
for sp in ax2.spines.values(): sp.set_edgecolor("#22334a")
ax2.legend(fontsize=9, facecolor="#1a2540", labelcolor="#e8ecf0", framealpha=0.8)
ax2.set_xlim(2.5, 7.5)
ax2.set_ylim(0, 1.05)
ax2.grid(True, alpha=0.15, color="#00d4ff")

fig_gr.tight_layout(pad=2.5)
fig_gr.savefig(FIG / "gutenberg_richter_fit.png", dpi=150,
               bbox_inches="tight", facecolor="#0f1525")
plt.close(fig_gr)
print(f"    Saved → figures/gutenberg_richter_fit.png")

# ── Step 7: b-value spatial map ────────────────────────────────────────────
if len(grid_df) >= 3:
    fig_bmap, ax_b = plt.subplots(figsize=(9, 7))
    fig_bmap.patch.set_facecolor("#0f1525")
    ax_b.set_facecolor("#0f1525")

    sc = ax_b.scatter(
        grid_df.lon_center, grid_df.lat_center,
        c=grid_df.b_value,
        cmap="RdYlGn_r",
        vmin=0.6, vmax=1.4,
        s=grid_df.n_events.clip(upper=500) * 1.5,
        alpha=0.85, edgecolors="white", linewidths=0.4, zorder=5,
    )

    # All catalog events as background scatter
    ax_b.scatter(region.longitude, region.latitude,
                 c="#00d4ff", s=0.3, alpha=0.08, zorder=2)

    cb = fig_bmap.colorbar(sc, ax=ax_b, pad=0.01, fraction=0.03)
    cb.set_label("b-değeri", color="#e8ecf0", fontsize=10)
    cb.ax.yaxis.set_tick_params(color="#e8ecf0")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="#e8ecf0")
    cb.outline.set_edgecolor("#22334a")

    ax_b.set_xlabel("Boylam (°E)", color="#e8ecf0", fontsize=10)
    ax_b.set_ylabel("Enlem (°N)", color="#e8ecf0", fontsize=10)
    ax_b.set_title("Uzamsal b-Değeri Haritası — Kahramanmaraş Bölgesi\n"
                   "Daire boyutu = olay sayısı  |  Renk = b-değeri (Gutenberg-Richter)",
                   color="#e8ecf0", fontsize=10, pad=8)
    ax_b.tick_params(colors="#e8ecf0")
    for sp in ax_b.spines.values(): sp.set_edgecolor("#22334a")
    ax_b.set_xlim(lon_lo, lon_hi)
    ax_b.set_ylim(lat_lo, lat_hi)
    ax_b.grid(True, alpha=0.15, color="#00d4ff")

    fig_bmap.tight_layout()
    fig_bmap.savefig(FIG / "b_value_map.png", dpi=150,
                     bbox_inches="tight", facecolor="#0f1525")
    plt.close(fig_bmap)
    print(f"    Saved → figures/b_value_map.png")
else:
    print(f"    ⚠ Too few grid cells ({len(grid_df)}) for b-value map — skipping")
    # Create placeholder
    fig_ph, ax_ph = plt.subplots(figsize=(8, 5))
    fig_ph.patch.set_facecolor("#0f1525")
    ax_ph.set_facecolor("#0f1525")
    ax_ph.text(0.5, 0.5, f"Insufficient grid cells (N={len(grid_df)}) for spatial b-value map.\n"
               "Global b-value: {:.3f}".format(b_value),
               color="#e8ecf0", ha="center", va="center", transform=ax_ph.transAxes, fontsize=12)
    ax_ph.axis("off")
    fig_ph.savefig(FIG / "b_value_map.png", dpi=150,
                   bbox_inches="tight", facecolor="#0f1525")
    plt.close(fig_ph)
    print(f"    Saved → figures/b_value_map.png (placeholder)")

# ── Summary ────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PSHA STEP 1 COMPLETE — Gutenberg-Richter Analysis")
print(f"  Catalog:      {len(region):,} events  |  T={T_years:.2f} years")
print(f"  Mc:           {Mc:.1f}")
print(f"  b-value:      {b_value:.4f} ± {b_std:.4f}")
print(f"  a-value:      {a_value:.4f}")
print(f"  Grid cells:   {len(grid_df)}")
print()
for M_thr in [4.0, 5.0, 6.0]:
    r = rates[f"M{M_thr:.0f}"]
    print(f"  M≥{M_thr:.0f}:  λ={r['annual_rate']:.4f}/yr  "
          f"T={r['return_period_yr']:.0f}yr  "
          f"P(10yr)={r['prob_10yr']:.3f}  "
          f"P(50yr)={r['prob_50yr']:.3f}  "
          f"P(100yr)={r['prob_100yr']:.3f}")
print()
print("  ⚠  SCIENTIFIC NOTE: These are long-term statistical estimates based")
print("     on Poisson seismicity model. NOT earthquake prediction.")
print("     Catalog is aftershock-dominated — rates likely overestimate")
print("     long-term background seismicity.")
print("=" * 70)
