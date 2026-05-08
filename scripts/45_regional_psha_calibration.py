# Phase: Quality Audit — Improvement 3
# Purpose: 3-zone PSHA calibration from ISC background catalog 1990–2023.
#          Computes zone-specific b-values for NAF (North Anatolian Fault),
#          EAF (East Anatolian Fault), and Central Anatolia. Excludes the
#          2023-02-06 to 2023-08-06 aftershock contamination period.
#          Rebuilds spatial_hazard_grid_v2.csv with calibrated rates.
# Inputs:  ISC FDSN API (http://www.isc.ac.uk/fdsnws/event/1/query)
#          artifacts/spatial_hazard_grid.csv (v1 reference)
# Outputs: artifacts/spatial_hazard_grid_v2.csv, artifacts/zone_calibration.json,
#          figures/b_value_zone_map.png
# Limitations: ISC catalog completeness Mc≈3.0 pre-2000, Mc≈2.8 post-2000.
#              Aftershock exclusion window chosen conservatively (6 months).
#              Zone boundaries are simplified rectangles.

from pathlib import Path
import json, time, io
import numpy as np
import pandas as pd
import urllib.request
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parent.parent
ART  = ROOT / "artifacts"
FIG  = ROOT / "figures"
CACHE = ROOT / "data" / "catalog" / "isc_background_catalog.csv"

# ── Zone definitions ─────────────────────────────────────────────────────────
# Three seismotectonic zones as rectangles. Events may overlap zone boundaries;
# each event is assigned to its primary zone by priority: NAF > EAF > CA.
ZONES = {
    "NAF": {
        "name":   "Kuzey Anadolu Fayı (KAF) Zonu",
        "lat_lo": 39.0, "lat_hi": 42.0,
        "lon_lo": 26.0, "lon_hi": 36.0,
        "expected_b": (0.9, 1.1),
        "note": "NAF seismic belt — high b from distributed aftershocks",
        "color": "#e74c3c",
    },
    "EAF": {
        "name":   "Doğu Anadolu Fayı (DAF) Zonu",
        "lat_lo": 36.0, "lat_hi": 39.0,
        "lon_lo": 36.0, "lon_hi": 44.0,
        "expected_b": (0.75, 0.95),
        "note": "EAF seismic belt — lower b reflects compressional stress",
        "color": "#e67e22",
    },
    "CA": {
        "name":   "Orta Anadolu Zonu",
        "lat_lo": 36.0, "lat_hi": 42.0,
        "lon_lo": 26.0, "lon_hi": 44.0,  # catch-all (after NAF/EAF assigned)
        "expected_b": (0.95, 1.15),
        "note": "Central Anatolia — moderate seismicity, higher b",
        "color": "#27ae60",
    },
}

def classify_zone(lat, lon):
    if (ZONES["NAF"]["lat_lo"] <= lat <= ZONES["NAF"]["lat_hi"] and
            ZONES["NAF"]["lon_lo"] <= lon <= ZONES["NAF"]["lon_hi"]):
        return "NAF"
    if (ZONES["EAF"]["lat_lo"] <= lat <= ZONES["EAF"]["lat_hi"] and
            ZONES["EAF"]["lon_lo"] <= lon <= ZONES["EAF"]["lon_hi"]):
        return "EAF"
    return "CA"

# ── Fetch ISC catalog in year-chunks ────────────────────────────────────────
AFTERSHOCK_START = pd.Timestamp("2023-02-06", tz="UTC")
AFTERSHOCK_END   = pd.Timestamp("2023-08-06", tz="UTC")

def fetch_isc_chunk(start, end, min_mag=3.0, timeout=60):
    url = (
        f"http://www.isc.ac.uk/fdsnws/event/1/query?"
        f"format=text&minlatitude=36&maxlatitude=42"
        f"&minlongitude=26&maxlongitude=44"
        f"&starttime={start}&endtime={end}"
        f"&minmagnitude={min_mag}"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "SeismicAI/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read().decode("utf-8", errors="replace")
    lines = [l for l in raw.splitlines() if l and not l.startswith("#")]
    if not lines:
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO("\n".join(lines)), sep="|",
                     names=["EventID","Time","Latitude","Longitude",
                             "Depth_km","Author","Catalog","Contributor",
                             "ContribID","MagType","Magnitude","MagAuthor",
                             "LocName","EventType"])
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce", utc=True)
    df["Magnitude"] = pd.to_numeric(df["Magnitude"], errors="coerce")
    df = df.dropna(subset=["Time","Magnitude","Latitude","Longitude"])
    return df

print("=" * 70)
print("IMPROVEMENT 3 — 3-Zone PSHA Regional Calibration")
print("=" * 70)

if CACHE.exists():
    print(f"\n[1] Loading cached ISC catalog: {CACHE}")
    cat = pd.read_csv(CACHE, parse_dates=["Time"])
    cat["Time"] = pd.to_datetime(cat["Time"], format="mixed", utc=True)
    print(f"    Cached events: {len(cat)}")
else:
    print("\n[1] Fetching ISC background catalog (M≥3.0, Turkey, 1990-2023) ...")
    print("    Excluding aftershock period: 2023-02-06 → 2023-08-06")

    chunks = []
    # Fetch in 5-year blocks
    periods = [
        ("1990-01-01", "1994-12-31"),
        ("1995-01-01", "1999-12-31"),
        ("2000-01-01", "2004-12-31"),
        ("2005-01-01", "2009-12-31"),
        ("2010-01-01", "2014-12-31"),
        ("2015-01-01", "2019-12-31"),
        ("2020-01-01", "2023-01-31"),   # stops before aftershock period
        ("2023-08-07", "2023-12-31"),   # resumes after aftershock exclusion
    ]
    for start, end in periods:
        print(f"    Fetching {start} → {end} ...", end="", flush=True)
        t0 = time.time()
        try:
            chunk = fetch_isc_chunk(start, end)
            chunks.append(chunk)
            print(f" {len(chunk)} events ({time.time()-t0:.0f}s)")
        except Exception as e:
            print(f" FAILED: {e}")
        time.sleep(0.5)

    cat = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    cat.to_csv(CACHE, index=False)   # save raw before filtering
    print(f"\n    Total events fetched (raw): {len(cat)}")
    print(f"    Saved cache → {CACHE}")

# Remove aftershock contamination window
if len(cat):
    cat["Time"] = pd.to_datetime(cat["Time"], format="mixed", utc=True)
    mask = ~((cat["Time"] >= AFTERSHOCK_START) & (cat["Time"] <= AFTERSHOCK_END))
    n_before = len(cat)
    cat = cat[mask].reset_index(drop=True)
    print(f"    After aftershock exclusion: {len(cat)} events "
          f"(removed {n_before - len(cat)} from {AFTERSHOCK_START.date()} to {AFTERSHOCK_END.date()})")

# ── Assign zones ──────────────────────────────────────────────────────────────
print("\n[2] Assigning seismotectonic zones ...")
cat["zone"] = cat.apply(lambda r: classify_zone(r["Latitude"], r["Longitude"]), axis=1)

for zone_id, zinfo in ZONES.items():
    n = (cat["zone"] == zone_id).sum()
    print(f"    {zone_id} ({zinfo['name'][:30]}...): {n} events")

# ── Compute zone-specific b-values (MLE, Aki 1965) ───────────────────────────
print("\n[3] Computing zone-specific b-values (MLE, Mc from MAXC+0.2 correction) ...")

def compute_gr_mle(magnitudes, Mc_floor=2.5, delta_M=0.1):
    """Maximum-Likelihood b-value (Aki 1965) with MAXC Mc detection."""
    mag = np.array(magnitudes)
    mag = mag[np.isfinite(mag)]
    if len(mag) < 50:
        return None, None, None, None

    # MAXC magnitude of completeness
    bins   = np.arange(mag.min(), mag.max() + delta_M, delta_M)
    counts, edges = np.histogram(mag, bins=bins)
    bin_centers   = edges[:-1] + delta_M / 2
    Mc_maxc = float(bin_centers[np.argmax(counts)] + 0.2)
    Mc      = max(Mc_maxc, Mc_floor)

    # MLE b-value above Mc
    mag_above = mag[mag >= Mc]
    if len(mag_above) < 30:
        return None, None, None, None

    b     = np.log10(np.e) / (mag_above.mean() - Mc + delta_M / 2)
    sigma = b / np.sqrt(len(mag_above))

    # Activity rate λ(M≥Mc) per year (using total observation window)
    n_yr  = (cat["Time"].max() - cat["Time"].min()).days / 365.25
    lam_Mc = len(mag_above) / n_yr if n_yr > 0 else np.nan

    return round(b, 4), round(sigma, 4), round(Mc, 2), round(lam_Mc, 3)

zone_results = {}
for zone_id in ["NAF", "EAF", "CA"]:
    z_mag = cat[cat["zone"] == zone_id]["Magnitude"].dropna()
    b, sigma, Mc, lam = compute_gr_mle(z_mag)
    expected = ZONES[zone_id]["expected_b"]
    in_range = bool((b is not None) and (expected[0] <= b <= expected[1]))
    flag = "✓" if in_range else "⚠"
    print(f"    {zone_id}: b={b} ± {sigma}  Mc={Mc}  λ(Mc)={lam}/yr  "
          f"[expected {expected[0]}–{expected[1]}]  {flag}")
    zone_results[zone_id] = {
        "name":     ZONES[zone_id]["name"],
        "b_value":  b,
        "b_sigma":  sigma,
        "Mc":       Mc,
        "lambda_Mc_per_yr": lam,
        "n_events": int((cat["zone"] == zone_id).sum()),
        "expected_range": list(expected),
        "in_expected_range": in_range,
        "note": ZONES[zone_id]["note"],
    }

# ── Rebuild spatial hazard grid v2 ─────────────────────────────────────────────
print("\n[4] Rebuilding spatial_hazard_grid_v2.csv ...")

# Load existing v1 grid as spatial backbone
v1 = pd.read_csv(ART / "spatial_hazard_grid.csv")
print(f"    v1 grid cells: {len(v1)}")

# Also build additional cells from ISC catalog (broader Turkey coverage)
CELL_DEG = 0.5
lat_range = np.arange(36.0, 42.0, CELL_DEG)
lon_range = np.arange(26.0, 44.0, CELL_DEG)

N_TOTAL = (cat["Time"].max() - cat["Time"].min()).days / 365.25
print(f"    Observation window: {N_TOTAL:.2f} years")

rows_v2 = []
n_omori_corrected = 0

for lat in lat_range:
    for lon in lon_range:
        lat_c = lat + CELL_DEG / 2
        lon_c = lon + CELL_DEG / 2
        zone_id = classify_zone(lat_c, lon_c)

        mask = (
            (cat["Latitude"]  >= lat) & (cat["Latitude"]  < lat + CELL_DEG) &
            (cat["Longitude"] >= lon) & (cat["Longitude"] < lon + CELL_DEG)
        )
        cell_events = cat[mask]

        if len(cell_events) < 5:
            continue

        # Zone b-value (fallback global if zone fails)
        zr = zone_results.get(zone_id, {})
        b_cell = zr.get("b_value") or 0.85
        Mc_cell = zr.get("Mc") or 2.8

        mag_above = cell_events["Magnitude"][cell_events["Magnitude"] >= Mc_cell]
        if len(mag_above) < 5:
            continue

        # Local b-value if sufficient events (>30), else zone b
        if len(mag_above) >= 30:
            b_local, _, _, _ = compute_gr_mle(mag_above.values, Mc_floor=Mc_cell)
            if b_local and 0.5 <= b_local <= 1.8:
                b_cell = b_local

        a_cell = np.log10(len(mag_above) / N_TOTAL) + b_cell * Mc_cell

        def lambda_M(m):
            return 10 ** (a_cell - b_cell * m)

        lam5 = lambda_M(5.0)
        lam6 = lambda_M(6.0)
        lam7 = lambda_M(7.0)

        # Check if cell is in aftershock zone (Kahramanmaras region)
        in_aftershock_zone = (36.0 <= lat_c <= 39.0) and (35.0 <= lon_c <= 39.5)
        omori_factor = 1.0
        if in_aftershock_zone:
            # Catalog still dominated by aftershocks despite exclusion window
            # Use conservative 10× factor (more conservative than v1's 50× global)
            omori_factor = 10.0
            lam5 /= omori_factor
            lam6 /= omori_factor
            lam7 /= omori_factor
            n_omori_corrected += 1

        # Poisson probability over 50 years
        P50_M5 = 1 - np.exp(-lam5 * 50)
        P50_M6 = 1 - np.exp(-lam6 * 50)
        P50_M7 = 1 - np.exp(-lam7 * 50)

        rows_v2.append({
            "lat_center":    round(lat_c, 3),
            "lon_center":    round(lon_c, 3),
            "zone":          zone_id,
            "n_events":      len(cell_events),
            "n_above_Mc":    len(mag_above),
            "Mc":            round(Mc_cell, 2),
            "b_value":       round(b_cell, 4),
            "a_value":       round(a_cell, 4),
            "lambda_M5":     round(lam5, 6),
            "lambda_M6":     round(lam6, 6),
            "lambda_M7":     round(lam7, 6),
            "P50yr_M5":      round(min(P50_M5, 1.0), 4),
            "P50yr_M6":      round(min(P50_M6, 1.0), 4),
            "P50yr_M7":      round(min(P50_M7, 1.0), 4),
            "omori_factor":  omori_factor,
            "source":        "ISC 1990-2023 (excl. 2023-02-06/08-06)",
        })

grid_v2 = pd.DataFrame(rows_v2)
out_v2 = ART / "spatial_hazard_grid_v2.csv"
grid_v2.to_csv(out_v2, index=False)
print(f"    v2 grid cells: {len(grid_v2)}  (Omori-corrected: {n_omori_corrected})")
print(f"    b-value range: {grid_v2['b_value'].min():.3f}–{grid_v2['b_value'].max():.3f}")
print(f"    Saved → {out_v2}")

# ── Per-zone summary statistics ───────────────────────────────────────────────
print("\n    Zone summary (grid v2):")
for z in ["NAF", "EAF", "CA"]:
    sub = grid_v2[grid_v2["zone"] == z]
    if len(sub):
        print(f"      {z}: {len(sub)} cells  "
              f"b={sub['b_value'].mean():.3f}±{sub['b_value'].std():.3f}  "
              f"P(M6,50yr)={sub['P50yr_M6'].mean():.3f}")

# ── Figure: b-value zone map ──────────────────────────────────────────────────
print("\n[5] Generating b-value zone map ...")
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("3-Zone PSHA Calibration — ISC Turkey 1990–2023\n"
             "(Aftershock period 2023-02-06 to 2023-08-06 excluded)",
             fontsize=13, fontweight="bold")

# Left: b-value map
ax1 = axes[0]
ax1.set_title("b-value per grid cell (v2 calibration)", fontsize=11)
sc = ax1.scatter(
    grid_v2["lon_center"], grid_v2["lat_center"],
    c=grid_v2["b_value"], cmap="RdYlGn",
    vmin=0.5, vmax=1.5, s=80, alpha=0.85, zorder=3
)
plt.colorbar(sc, ax=ax1, label="b-value")

# Zone rectangles
zone_colors = {"NAF": "#e74c3c", "EAF": "#e67e22", "CA": "#27ae60"}
zone_patches = []
for zid, zinfo in ZONES.items():
    if zid == "CA":
        continue  # CA is the background — skip rectangle
    rect = mpatches.FancyArrowPatch(
        posA=(zinfo["lon_lo"], zinfo["lat_lo"]),
        posB=(zinfo["lon_hi"], zinfo["lat_hi"]),
        arrowstyle="-", linestyle="--",
        color=zone_colors[zid], linewidth=2, zorder=5
    )
    ax1.add_patch(mpatches.Rectangle(
        (zinfo["lon_lo"], zinfo["lat_lo"]),
        zinfo["lon_hi"] - zinfo["lon_lo"],
        zinfo["lat_hi"] - zinfo["lat_lo"],
        fill=False, edgecolor=zone_colors[zid],
        linewidth=2.5, linestyle="--", zorder=5
    ))
    zone_patches.append(mpatches.Patch(color=zone_colors[zid], label=f"{zid}"))

ax1.legend(handles=zone_patches, loc="upper right", fontsize=9)
ax1.set_xlabel("Boylam (°E)"); ax1.set_ylabel("Enlem (°N)")
ax1.set_xlim(25.5, 44.5); ax1.set_ylim(35.5, 42.5)
ax1.grid(alpha=0.3)

# Right: Zone b-value distributions
ax2 = axes[1]
ax2.set_title("b-value dağılımı (zone karşılaştırması)", fontsize=11)
for zid in ["NAF", "EAF", "CA"]:
    sub = grid_v2[grid_v2["zone"] == zid]["b_value"]
    if len(sub):
        ax2.hist(sub, bins=15, alpha=0.65, label=f"{zid} (μ={sub.mean():.3f})",
                 color=zone_colors[zid], edgecolor="white")
        lo, hi = ZONES[zid]["expected_b"]
        ax2.axvline(lo, color=zone_colors[zid], linestyle=":", alpha=0.7)
        ax2.axvline(hi, color=zone_colors[zid], linestyle=":", alpha=0.7)

ax2.set_xlabel("b-değeri"); ax2.set_ylabel("Hücre sayısı")
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

plt.tight_layout()
out_fig = FIG / "b_value_zone_map.png"
plt.savefig(out_fig, dpi=150, bbox_inches="tight")
plt.close()
print(f"    Saved → {out_fig}")

# ── Save zone calibration JSON ─────────────────────────────────────────────────
T_obs_start = str(cat["Time"].min())[:10] if len(cat) else "unknown"
T_obs_end   = str(cat["Time"].max())[:10] if len(cat) else "unknown"

calibration_out = {
    "version": "v2",
    "method": "3-zone seismotectonic PSHA calibration",
    "catalog": {
        "source":  "ISC FDSN (www.isc.ac.uk/fdsnws/event/1/)",
        "period":  f"{T_obs_start} to {T_obs_end}",
        "excluded": "2023-02-06 to 2023-08-06 (Kahramanmaras aftershock sequence)",
        "min_magnitude": 3.0,
        "bbox": "36–42°N, 26–44°E",
        "n_events_total": len(cat),
    },
    "zones": zone_results,
    "grid_v2": {
        "path":   "artifacts/spatial_hazard_grid_v2.csv",
        "n_cells": len(grid_v2),
        "cell_size_deg": CELL_DEG,
        "omori_corrected_cells": n_omori_corrected,
        "omori_factor_aftershock_zone": 10.0,
        "b_value_mean": round(float(grid_v2["b_value"].mean()), 4),
        "b_value_std":  round(float(grid_v2["b_value"].std()), 4),
    },
    "references": [
        "Aki, K. (1965). Maximum likelihood estimate of b in the formula log N = a – bM.",
        "Woessner, J. & Wiemer, S. (2005). Assessing the quality of earthquake catalogues.",
        "ISC Bulletin: http://www.isc.ac.uk/iscbulletin/",
    ]
}
out_json = ART / "zone_calibration.json"
with open(out_json, "w") as f:
    json.dump(calibration_out, f, indent=2, ensure_ascii=False)
print(f"    Saved → {out_json}")

print("\n" + "=" * 70)
print("IMPROVEMENT 3 COMPLETE — 3-Zone PSHA Regional Calibration")
print(f"  ISC catalog: {len(cat)} events, {N_TOTAL:.1f} years")
for zid in ["NAF", "EAF", "CA"]:
    zr = zone_results[zid]
    flag = "✓" if zr["in_expected_range"] else "⚠"
    print(f"  {zid}: b={zr['b_value']} ± {zr['b_sigma']}  {flag}")
print(f"  Grid v2: {len(grid_v2)} cells → artifacts/spatial_hazard_grid_v2.csv")
print(f"  Figure: figures/b_value_zone_map.png")
print("=" * 70)
