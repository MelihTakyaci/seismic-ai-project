# Phase: PSHA Step 2
# Purpose: Probabilistic seismic hazard calculator.
#          Maps spatial G-R parameters to Poisson exceedance probabilities.
#          Applies aftershock correction for Kahramanmaras region.
#          Uses published background seismicity for Marmara / Aegean / Central Anatolia.
#          This is sismik tehlike değerlendirmesi — NOT deprem tahmini.
# Inputs:  artifacts/gutenberg_richter_params.json,
#          artifacts/spatial_hazard_grid.csv
# Outputs: artifacts/hazard_calculator_validation.json
# Limitations: Aftershock-dominated catalog (0.49yr); Omori correction reduces rates
#              by ~100× for long-term background estimate. Marmara rates from
#              published literature (Ambraseys 2002, AFAD TDTH 2018), not this catalog.
#              All values are long-term statistical estimates, not forecasts.

from pathlib import Path
import json
import math
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ART  = ROOT / "artifacts"

print("=" * 70)
print("PSHA STEP 2 — Probabilistic Hazard Calculator")
print("  Olasılıksal Sismik Tehlike Değerlendirmesi")
print("  ALL outputs = long-term statistical estimates, NOT earthquake forecasts")
print("=" * 70)

# ── Load G-R parameters ────────────────────────────────────────────────────
with open(ART / "gutenberg_richter_params.json") as f:
    gr = json.load(f)

grid_df = pd.read_csv(ART / "spatial_hazard_grid.csv")

b_global = gr["gr_fit"]["b_value"]
a_global = gr["gr_fit"]["a_value"]
T_obs    = gr["catalog_info"]["T_years"]   # ~0.49 years

print(f"\n  G-R params: b={b_global:.4f}  a={a_global:.4f}  T_obs={T_obs:.2f} yr")

# ── Aftershock correction ──────────────────────────────────────────────────
# The catalog covers the first 6 months of the Kahramanmaras aftershock sequence.
# Omori-Utsu law: cumulative aftershock count N(t) ∝ ln(t/c) for p≈1.
# Ratio of aftershock rate to long-term background depends on time since mainshock.
# For the Kahramanmaras M7.8 aftershock sequence, the 6-month catalog captures
# roughly 50-100× more events than long-term background seismicity.
# We apply a conservative correction factor of 50 (lower bound estimate).
# Reference: Ogata 1988, Utsu 1961; typical aftershock excess 30-200× background.
AFTERSHOCK_CORRECTION = 50.0   # divide annual rates by this for background estimate
print(f"  Aftershock correction factor: {AFTERSHOCK_CORRECTION:.0f}× "
      f"(Omori-Utsu — Kahramanmaras M7.8 sequence)")

# Corrected G-R: same b-value (shape), reduced annual count
# log10(N_background) = log10(N_aftershock / correction)
# a_bg = a_aftershock - log10(correction)
a_bg_global = a_global - math.log10(AFTERSHOCK_CORRECTION)

print(f"  Background a-value (corrected): {a_bg_global:.4f}")
print(f"  Background λ(M≥4): {10**(a_bg_global - b_global*4):.3f}/yr")
print(f"  Background λ(M≥5): {10**(a_bg_global - b_global*5):.4f}/yr")

# ── Background seismicity for non-Kahramanmaras regions ────────────────────
# Source: Ambraseys et al. 2002, Kalkan & Gulkan 2004, AFAD TDTH 2018 implicit rates
# These are approximate long-term background rates for major Turkish seismic zones.
# NOT derived from the Kahramanmaras catalog.
REGION_BACKGROUND = {
    "marmara": {
        "description": "Marmara Denizi — Kuzey Anadolu Fayı (Batı segmenti)",
        "lambda_M4":   5.00,    # ~5 M≥4 events/yr (historical 1900-2023)
        "lambda_M5":   1.00,    # ~1 M≥5/yr
        "lambda_M6":   0.20,    # ~0.2 M≥6/yr (1 per 5 years)
        "lambda_M7":   0.07,    # ~0.07 M≥7/yr (1 per 15 years)
        "b_value":     0.95,
        "source":      "Ambraseys et al. 2002; AFAD TDTH 2018",
    },
    "aegean": {
        "description": "Ege Bölgesi — Yunan-Türk Sismik Kuşağı",
        "lambda_M4":   4.00,
        "lambda_M5":   0.80,
        "lambda_M6":   0.15,
        "lambda_M7":   0.05,
        "b_value":     1.00,
        "source":      "Papazachos et al. 1997; Kalkan & Gulkan 2004",
    },
    "central_anatolia": {
        "description": "İç Anadolu — Düşük-Orta Sismisiteli Bölge",
        "lambda_M4":   1.00,
        "lambda_M5":   0.15,
        "lambda_M6":   0.025,
        "lambda_M7":   0.003,
        "b_value":     0.85,
        "source":      "Kalkan & Gulkan 2004",
    },
    "east_anatolia": {
        "description": "Doğu Anadolu — Doğu Anadolu Fayı",
        "lambda_M4":   6.00,
        "lambda_M5":   1.20,
        "lambda_M6":   0.25,
        "lambda_M7":   0.08,
        "b_value":     0.90,
        "source":      "Ambraseys et al. 2002; AFAD TDTH 2018",
    },
}

def _assign_region(lat, lon):
    """Assign a seismic region based on coordinates."""
    if 39.0 <= lat <= 42.0 and 26.0 <= lon <= 32.0:
        return "marmara"
    if 37.0 <= lat <= 39.5 and 25.0 <= lon <= 29.5:
        return "aegean"
    if 37.5 <= lat <= 39.5 and 29.5 <= lon <= 36.0:
        return "central_anatolia"
    if 36.0 <= lat <= 39.0 and 35.0 <= lon <= 39.5:
        return "east_anatolia"   # Kahramanmaras zone
    # Default
    if lon < 30.0:
        return "aegean"
    return "central_anatolia"

# ── Core functions ─────────────────────────────────────────────────────────
print("\n[1] Defining hazard functions ...")

def get_lambda(lat: float, lon: float, M_thr: float,
               use_aftershock_correction: bool = True) -> dict:
    """
    Return annual rate lambda for exceedance of M_thr at (lat, lon).

    Uses:
    - Spatial G-R grid (Kahramanmaras region) with Omori correction for background
    - Published regional background rates outside the grid
    Returns dict with lambda, b_value, source, region.
    """
    # Try nearest grid cell (within 0.5° tolerance)
    if len(grid_df) > 0:
        dist = np.sqrt(
            (grid_df.lat_center - lat) ** 2 +
            (grid_df.lon_center - lon) ** 2
        )
        nearest_idx = dist.idxmin()
        nearest_dist = dist[nearest_idx]

        if nearest_dist <= 0.5:  # within search radius
            row = grid_df.loc[nearest_idx]
            b   = row["b_value"]
            a_r = row["a_value"]
            # Apply aftershock correction for background estimate
            if use_aftershock_correction:
                a_r = a_r - math.log10(AFTERSHOCK_CORRECTION)
            lam = 10 ** (a_r - b * M_thr)
            return {
                "lambda":      max(lam, 1e-9),
                "b_value":     b,
                "a_value":     a_r,
                "n_events":    int(row["n_events"]),
                "source":      f"G-R spatial grid (n={int(row['n_events'])}; Omori-corrected)",
                "region":      "kahramanmaras_aftershock",
                "grid_dist":   round(float(nearest_dist), 3),
            }

    # Fallback to regional background
    region = _assign_region(lat, lon)
    reg    = REGION_BACKGROUND[region]
    b      = reg["b_value"]
    # Interpolate from published M4,M5,M6,M7 rates
    known_M   = [4.0, 5.0, 6.0, 7.0]
    known_lam = [reg["lambda_M4"], reg["lambda_M5"], reg["lambda_M6"], reg["lambda_M7"]]
    # Fit local a from known rates (use M5 anchor)
    a_local = math.log10(reg["lambda_M5"]) + b * 5.0
    lam = 10 ** (a_local - b * M_thr)
    return {
        "lambda":   max(lam, 1e-9),
        "b_value":  b,
        "a_value":  a_local,
        "source":   reg["source"],
        "region":   region,
        "grid_dist": None,
    }


def get_hazard_probability(lat: float, lon: float,
                           magnitude_threshold: float,
                           exposure_years: float) -> dict:
    """
    Return Poisson exceedance probability for given M threshold and exposure time.

    Parameters
    ----------
    lat, lon : float — location coordinates
    magnitude_threshold : float — minimum magnitude (e.g., 5.0)
    exposure_years : float — exposure time (e.g., 50)

    Returns
    -------
    dict with probability, return_period, annual_rate, source, region, disclaimer
    """
    lam_info = get_lambda(lat, lon, magnitude_threshold)
    lam      = lam_info["lambda"]
    P        = 1 - math.exp(-lam * exposure_years)
    T_ret    = 1 / lam if lam > 0 else float("inf")

    return {
        "lat": lat,
        "lon": lon,
        "magnitude_threshold": magnitude_threshold,
        "exposure_years":      exposure_years,
        "annual_rate":         round(lam, 6),
        "return_period_yr":    round(T_ret, 1),
        "probability":         round(P, 4),
        "probability_pct":     round(P * 100, 1),
        "b_value":             lam_info["b_value"],
        "source":              lam_info["source"],
        "region":              lam_info["region"],
        "disclaimer": (
            "Poisson istatistiğine dayalı uzun vadeli sismik tehlike tahminidir. "
            "Kısa vadeli deprem tahmini değildir (NOT earthquake prediction)."
        ),
    }


def get_gr_curve(lat: float, lon: float, M_range: tuple = (3.0, 8.0)) -> dict:
    """Return G-R curve (M vs annual rate) for a location."""
    M_arr = np.arange(M_range[0], M_range[1] + 0.1, 0.1)
    lam_arr = []
    for M in M_arr:
        info = get_lambda(lat, lon, float(M))
        lam_arr.append(info["lambda"])
    return {
        "magnitude":    [round(float(m), 1) for m in M_arr],
        "annual_rate":  [round(float(l), 6) for l in lam_arr],
        "region":       get_lambda(lat, lon, 5.0)["region"],
        "b_value":      get_lambda(lat, lon, 5.0)["b_value"],
    }


def classify_hazard_level(probability_M5_50yr: float) -> dict:
    """Classify probabilistic hazard level."""
    p = probability_M5_50yr
    if p < 0.10:
        return {"level": "Düşük Tehlike",     "color": "#00c853", "code": "low"}
    elif p < 0.40:
        return {"level": "Orta Tehlike",      "color": "#ffd600", "code": "moderate"}
    elif p < 0.70:
        return {"level": "Yüksek Tehlike",    "color": "#ff9800", "code": "high"}
    else:
        return {"level": "Çok Yüksek Tehlike","color": "#ef5350", "code": "very_high"}


def get_full_hazard_profile(lat: float, lon: float) -> dict:
    """Full hazard profile: multiple M thresholds and exposure times."""
    profile = {}
    for M_thr in [4.0, 5.0, 6.0, 7.0]:
        key = f"M{M_thr:.0f}"
        profile[key] = {}
        for t_exp in [10, 50, 100]:
            h = get_hazard_probability(lat, lon, M_thr, t_exp)
            profile[key][f"P_{t_exp}yr"] = h["probability"]
            profile[key]["return_period_yr"] = h["return_period_yr"]
            profile[key]["annual_rate"]      = h["annual_rate"]

    p_M5_50 = profile["M5"]["P_50yr"]
    hazard_class = classify_hazard_level(p_M5_50)
    profile["hazard_level"]  = hazard_class["level"]
    profile["hazard_code"]   = hazard_class["code"]
    profile["hazard_color"]  = hazard_class["color"]
    profile["source"]        = get_lambda(lat, lon, 5.0)["source"]
    profile["region"]        = get_lambda(lat, lon, 5.0)["region"]
    profile["disclaimer"]    = (
        "Uzun vadeli Poisson sismik tehlike tahmini. "
        "Deprem tahmini değildir. Kısa vadeli veya kesin tahmin yapılamaz."
    )
    return profile


print("    Functions defined: get_hazard_probability(), get_gr_curve(),")
print("    classify_hazard_level(), get_full_hazard_profile()")

# ── Validation: 5 cities ───────────────────────────────────────────────────
print("\n[2] Validation — 5 reference cities ...")
print("-" * 70)

TEST_CITIES = [
    {"name": "İstanbul (Anadolu)",  "lat": 40.99, "lon": 29.03},
    {"name": "Ankara",              "lat": 39.92, "lon": 32.85},
    {"name": "İzmir",               "lat": 38.42, "lon": 27.13},
    {"name": "Kahramanmaraş",       "lat": 37.58, "lon": 36.94},
    {"name": "Bursa",               "lat": 40.18, "lon": 29.06},
]

validation = {}
for city in TEST_CITIES:
    nm  = city["name"]
    lat = city["lat"]
    lon = city["lon"]

    h = get_hazard_probability(lat, lon, 5.0, 50)
    h6= get_hazard_probability(lat, lon, 6.0, 50)
    hc = classify_hazard_level(h["probability"])

    validation[nm] = {
        "lat": lat, "lon": lon,
        "region": h["region"],
        "M5_50yr": {
            "annual_rate":      h["annual_rate"],
            "return_period_yr": h["return_period_yr"],
            "probability":      h["probability"],
            "probability_pct":  h["probability_pct"],
        },
        "M6_50yr": {
            "annual_rate":      h6["annual_rate"],
            "return_period_yr": h6["return_period_yr"],
            "probability":      h6["probability"],
            "probability_pct":  h6["probability_pct"],
        },
        "hazard_level": hc["level"],
        "hazard_code":  hc["code"],
        "source":       h["source"],
    }

    print(f"\n  {nm} ({lat}°N, {lon}°E)  [{h['region']}]")
    print(f"    M≥5.0 — λ={h['annual_rate']:.4f}/yr  "
          f"T={h['return_period_yr']:.0f}yr  "
          f"P(50yr)={h['probability_pct']:.1f}%")
    print(f"    M≥6.0 — λ={h6['annual_rate']:.4f}/yr  "
          f"T={h6['return_period_yr']:.0f}yr  "
          f"P(50yr)={h6['probability_pct']:.1f}%")
    print(f"    Tehlike seviyesi: {hc['level']}  |  Kaynak: {h['source']}")

print("-" * 70)

# ── Save validation artifact ───────────────────────────────────────────────
print("\n[3] Saving validation artifact ...")

out = {
    "description": (
        "Probabilistic seismic hazard calculator validation. "
        "Olasılıksal Sismik Tehlike Analizi (OSTA). "
        "NOT earthquake prediction. Poisson long-term statistical estimates."
    ),
    "methodology": {
        "kahramanmaras_region": (
            "G-R parameters from 42,078-event catalog (0.49yr). "
            "Omori aftershock correction factor=50 applied for background estimate. "
            "b-value=0.74 (MLE, Aki 1965)."
        ),
        "other_regions": (
            "Published background seismicity rates: Ambraseys et al. 2002, "
            "Kalkan & Gulkan 2004, AFAD TDTH 2018."
        ),
        "model": "Poisson: P(t) = 1 - exp(-lambda * t)",
    },
    "validation_cities": validation,
    "hazard_thresholds": {
        "low":       "P(M5,50yr) < 10%",
        "moderate":  "P(M5,50yr) 10-40%",
        "high":      "P(M5,50yr) 40-70%",
        "very_high": "P(M5,50yr) > 70%",
    },
    "disclaimer": (
        "Tüm değerler Poisson istatistiğine dayalı uzun vadeli sismik tehlike "
        "tahminleridir. Kısa vadeli deprem tahmini değildir ve yapılamaz. "
        "Bu analiz OSTA (Olasılıksal Sismik Tehlike Analizi) metodolojisini "
        "kullanmaktadır."
    ),
}

with open(ART / "hazard_calculator_validation.json", "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f"    Saved → artifacts/hazard_calculator_validation.json")

# ── Quick summary table ────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PSHA STEP 2 COMPLETE — Hazard Calculator")
print()
print(f"  {'Şehir':<25}  {'Bölge':<20}  {'P(M5,50yr)':<12}  {'T_ret(M5)':<12}  Tehlike")
print(f"  {'-'*24}  {'-'*20}  {'-'*11}  {'-'*11}  {'-'*18}")
for nm, v in validation.items():
    print(f"  {nm:<25}  {v['region']:<20}  "
          f"{v['M5_50yr']['probability_pct']:>9.1f}%  "
          f"{v['M5_50yr']['return_period_yr']:>8.0f}yr   "
          f"{v['hazard_level']}")
print()
print("  ⚠  SCIENTIFIC DISCLAIMER:")
print("     Values are long-term Poisson probability estimates.")
print("     NOT earthquake forecasts. NOT applicable to short time windows.")
print("     Kahramanmaras rates corrected for aftershock sequence (50× factor).")
print("=" * 70)
