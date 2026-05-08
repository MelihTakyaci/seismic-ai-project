# Phase: 6 — TBDY-2018 Integration
# Purpose: Rule-based regional seismic risk assessment module aligned with
#          TBDY-2018 (Türkiye Bina Deprem Yönetmeliği). No LLM required —
#          all recommendations are deterministic, article-referenced lookups.
# Inputs:  AFAD TDTH API (Ss/S1 spectral values), artifacts/detection_results.json,
#          artifacts/marmara_extended_results.json
# Outputs: Structured JSON risk report per (lat, lon, building_type) query
# Limitations: Soil class derived from Vs30 approximation when AFAD API unavailable.
#              Fay yakınlık (fault proximity) check uses simplified distance rules.
#              TDTH API may be rate-limited or unavailable without VPN.

from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent

# ── TBDY-2018 Lookup Tables ────────────────────────────────────────────────────

# Zemin büyütme katsayıları — Tablo 2.1 & 2.2 (simplified: mid-range Ss)
SOIL_AMPLIFICATION = {
    "ZA": {"FS": 0.8,  "F1": 0.8,  "vs30_lo": 1500, "vs30_hi": 9999},
    "ZB": {"FS": 0.9,  "F1": 0.8,  "vs30_lo": 760,  "vs30_hi": 1500},
    "ZC": {"FS": 1.3,  "F1": 1.5,  "vs30_lo": 360,  "vs30_hi": 760},
    "ZD": {"FS": 1.6,  "F1": 2.4,  "vs30_lo": 180,  "vs30_hi": 360},
    "ZE": {"FS": 2.4,  "F1": 3.5,  "vs30_lo": 0,    "vs30_hi": 180},
}

# SDS → DTS mapping (Tablo 3.2)
# BKS-2 binalar bir alt sınıfa alınır
def get_dts(sds: float, bks: int) -> int:
    if sds >= 0.75:
        base = 1
    elif sds >= 0.50:
        base = 2
    elif sds >= 0.25:
        base = 3
    else:
        base = 4
    if bks == 2:
        base = max(1, base - 1)
    return base

# Yapı yükseklik sınırları (kat) — Bölüm 4, Tablo 4.1 (simplified)
HEIGHT_LIMITS = {
    # (bina_türü, dts, zemin_grubu) → max kat sayısı (None = sınırsız/özel tasarım)
    ("betonarme", 1, "AB"): 23,   # ~70 m
    ("betonarme", 1, "CD"): 18,   # ~56 m
    ("betonarme", 1, "E"):  7,
    ("betonarme", 2, "AB"): 30,   # ~91 m
    ("betonarme", 2, "CD"): 23,
    ("betonarme", 2, "E"):  10,
    ("betonarme", 3, "AB"): None,
    ("betonarme", 3, "CD"): None,
    ("betonarme", 3, "E"):  None,
    ("betonarme", 4, "AB"): None,
    ("betonarme", 4, "CD"): None,
    ("betonarme", 4, "E"):  None,
    ("çelik",     1, "AB"): None,   # özel tasarım
    ("çelik",     1, "CD"): 35,
    ("çelik",     1, "E"):  18,
    ("çelik",     2, "AB"): None,
    ("çelik",     2, "CD"): None,
    ("çelik",     2, "E"):  None,
    ("çelik",     3, "AB"): None,
    ("çelik",     3, "CD"): None,
    ("çelik",     3, "E"):  None,
    ("çelik",     4, "AB"): None,
    ("çelik",     4, "CD"): None,
    ("çelik",     4, "E"):  None,
    ("yığma",     1, "AB"): 2,
    ("yığma",     1, "CD"): 2,
    ("yığma",     1, "E"):  0,    # yasak
    ("yığma",     2, "AB"): 3,
    ("yığma",     2, "CD"): 2,
    ("yığma",     2, "E"):  2,
    ("yığma",     3, "AB"): 4,
    ("yığma",     3, "CD"): 3,
    ("yığma",     3, "E"):  2,
    ("yığma",     4, "AB"): 4,
    ("yığma",     4, "CD"): 4,
    ("yığma",     4, "E"):  3,
}

def _soil_group(soil_class: str) -> str:
    if soil_class in ("ZA", "ZB"): return "AB"
    if soil_class in ("ZC", "ZD"): return "CD"
    return "E"

# Marmara bölgesi Ss/S1 yaklaşım tablosu (AFAD TDTH fallback)
MARMARA_SPECTRAL = [
    # (lat_lo, lat_hi, lon_lo, lon_hi, Ss, S1, notes)
    (40.8, 41.2, 28.5, 29.2, 1.20, 0.50, "İstanbul Anadolu yakası"),
    (40.8, 41.2, 28.0, 28.5, 1.10, 0.48, "İstanbul Avrupa yakası"),
    (40.5, 40.8, 27.0, 28.0, 0.90, 0.40, "Tekirdağ"),
    (40.0, 40.5, 29.0, 30.0, 1.05, 0.45, "Bursa kuzey"),
    (39.5, 40.5, 26.0, 27.0, 0.75, 0.32, "Çanakkale"),
    (40.3, 40.8, 27.8, 28.5, 0.95, 0.42, "Silivri–Çorlu"),
]

def _marmara_fallback_spectral(lat: float, lon: float):
    for la, lh, lo, loh, ss, s1, note in MARMARA_SPECTRAL:
        if la <= lat <= lh and lo <= lon <= loh:
            return ss, s1, note
    # Generic Marmara estimate
    return 0.90, 0.40, "Marmara bölgesi genel tahmini"

# ── Function 1: get_soil_class ─────────────────────────────────────────────────

def _load_vs30_grid():
    """Load precomputed TBDY-2018 Vs30 grid (artifacts/vs30_grid.csv)."""
    try:
        from pathlib import Path
        import pandas as pd
        grid_path = Path(__file__).resolve().parent.parent / "artifacts" / "vs30_grid.csv"
        if grid_path.exists():
            return pd.read_csv(grid_path)
    except Exception:
        pass
    return None

_VS30_GRID = _load_vs30_grid()


def _lookup_vs30_grid(lat: float, lon: float):
    """Find nearest cell in Vs30 grid. Returns (soil_class, Vs30_mps, Ss_g, S1_g) or None."""
    if _VS30_GRID is None:
        return None
    df = _VS30_GRID
    dist = (df["lat"] - lat)**2 + (df["lon"] - lon)**2
    idx  = dist.idxmin()
    if dist[idx] > 0.5:   # >~55 km — outside grid coverage
        return None
    row = df.loc[idx]
    return row["soil_class"], int(row["Vs30_mps"]), float(row["Ss_g"]), float(row["S1_g"])


def get_soil_class(lat: float, lon: float, vs30: Optional[float] = None):
    """
    Returns soil class (ZA-ZE), Ss, S1, and SDS for given coordinates.

    Priority:
    1. If vs30 provided → direct lookup
    2. TBDY-2018 precomputed grid (artifacts/vs30_grid.csv, 42-anchor IDW)
    3. Marmara regional fallback table
    Assumes ZD (most common urban Turkish soil) if all else fails.
    """
    region_note = ""

    # Direct vs30 lookup
    if vs30 is not None:
        for sc, props in SOIL_AMPLIFICATION.items():
            if props["vs30_lo"] <= vs30 < props["vs30_hi"]:
                soil_class = sc
                break
        else:
            soil_class = "ZE"
        ss, s1, region_note = _marmara_fallback_spectral(lat, lon)
        vs30_display = int(vs30)
        source = "Vs30 girişi"

    else:
        # Try precomputed TBDY-2018 grid
        grid_result = _lookup_vs30_grid(lat, lon)
        if grid_result is not None:
            soil_class, vs30_display, ss, s1 = grid_result
            source = f"TBDY-2018 IDW (Vs30 ≈ {vs30_display} m/s)"
        else:
            # Marmara regional fallback
            ss, s1, region_note = _marmara_fallback_spectral(lat, lon)
            soil_class = "ZD"
            vs30_display = 270
            source = "Bölgesel tahmin (TBDY-2018 ızgara dışı)"

    amp = SOIL_AMPLIFICATION[soil_class]
    sds = amp["FS"] * ss
    sd1 = amp["F1"] * s1
    return {
        "soil_class": soil_class,
        "Vs30_mps":   vs30_display,
        "Ss": round(ss, 3), "S1": round(s1, 3),
        "SDS": round(sds, 3), "SD1": round(sd1, 3),
        "source": source,
        "region_note": region_note,
    }

# ── Function 2: get_local_seismicity ──────────────────────────────────────────

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))

def get_local_seismicity(lat: float, lon: float, radius_km: float = 50):
    """
    Query project detection_results.json for events within radius_km.
    Returns event statistics from our own detected catalog.
    """
    det_path = ROOT / "artifacts" / "detection_results.json"
    nearby = []
    if det_path.exists():
        with open(det_path) as f:
            det = json.load(f)
        events = det if isinstance(det, list) else det.get("events", [])
        for ev in events:
            ev_lat = ev.get("latitude") or ev.get("lat")
            ev_lon = ev.get("longitude") or ev.get("lon")
            if ev_lat is None or ev_lon is None:
                continue
            dist = haversine_km(lat, lon, float(ev_lat), float(ev_lon))
            if dist <= radius_km:
                nearby.append({
                    "distance_km": round(dist, 1),
                    "magnitude":   ev.get("magnitude", ev.get("mag")),
                    "depth_km":    ev.get("depth_km", ev.get("depth")),
                    "time":        ev.get("time", ev.get("origin_time")),
                    "detected":    ev.get("ensemble_detected", True),
                })
    n = len(nearby)
    mags = [e["magnitude"] for e in nearby if e.get("magnitude") is not None]
    return {
        "n_events_nearby": n,
        "radius_km": radius_km,
        "max_magnitude": round(max(mags), 1) if mags else None,
        "mean_magnitude": round(sum(mags)/len(mags), 2) if mags else None,
        "activity_level": (
            "Yüksek" if n > 50 else
            "Orta"   if n > 10 else
            "Düşük"  if n > 0  else "Veri yok"
        ),
        "source": str(det_path.name),
    }

# ── Function 3: get_tbdy_recommendation ───────────────────────────────────────

BKS_MAP = {
    "konut":    (1, "BKS 1 — Konut / Ofis"),
    "ofis":     (1, "BKS 1 — Konut / Ofis"),
    "ticari":   (1, "BKS 1 — Ticari Yapı"),
    "sanayi":   (1, "BKS 1 — Sanayi Yapısı"),
    "okul":     (2, "BKS 2 — Okul (kritik)"),
    "hastane":  (2, "BKS 2 — Hastane (kritik)"),
    "kamu":     (2, "BKS 2 — Kamu Binası (kritik)"),
}

RISK_COLOR = {
    (1, "E"): "kırmızı",
    (1, "CD"): "turuncu",
    (1, "AB"): "sarı",
    (2, "E"): "kırmızı",
    (2, "CD"): "turuncu",
    (2, "AB"): "sarı",
    (3, "CD"): "sarı",
    (3, "AB"): "yeşil",
    (3, "E"): "sarı",
    (4, "AB"): "yeşil",
    (4, "CD"): "yeşil",
    (4, "E"): "sarı",
}

def get_tbdy_recommendation(
    soil_class: str, sds: float, building_type: str,
    floors: int = 5, material: str = "betonarme",
    hazard_probability_M6_50yr: Optional[float] = None,
):
    """
    Returns TBDY-2018 provisions for the given combination.
    All references are to TBDY-2018 article numbers.
    If P(M≥6, 50yr) > 0.30, overrides to DTS-1 regardless of spectral values.
    """
    building_type = building_type.lower()
    bks_num, bks_label = BKS_MAP.get(building_type, (1, "BKS 1 — Genel"))
    dts = get_dts(sds, bks_num)
    sg  = _soil_group(soil_class)

    # PSHA-based DTS override (Step 5)
    psha_override = False
    psha_note = None
    if hazard_probability_M6_50yr is not None and hazard_probability_M6_50yr > 0.30:
        if dts > 1:
            dts = 1
            psha_override = True
            psha_note = (
                f"OSTA kaydı: P(M≥6.0, 50yr)={hazard_probability_M6_50yr:.1%} > %30. "
                "Yüksek tehlike olasılığı nedeniyle DTS-1 uygulandı. "
                "(Kaynak: Olasılıksal Sismik Tehlike Analizi — Deprem tahmini değildir.)"
            )

    max_floors = HEIGHT_LIMITS.get((material.lower(), dts, sg))
    risk_color = RISK_COLOR.get((dts, sg), "sarı")

    # Build provision list with article references
    provisions = []
    warnings = []

    # PSHA override warning
    if psha_override and psha_note:
        warnings.append(f"⚠ OSTA Override: {psha_note}")
        provisions.insert(0, {
            "madde": "OSTA — Olasılıksal Sismik Tehlike Analizi",
            "metin": psha_note,
        })

    # Performance level requirement
    if bks_num == 1:
        provisions.append({
            "madde": "Bölüm 3, Tablo 3.4",
            "metin": "DD-2 (475-yıl) depreminde Kontrollü Hasar (KH) performansı hedeflenir."
        })
    elif bks_num == 2:
        provisions.append({
            "madde": "Bölüm 3, Tablo 3.4",
            "metin": "DD-2 (475-yıl) depreminde Sınırlı Hasar (SH) performansı zorunludur. "
                     "BKS-2 binaları bir alt DTS sınıfına alınır."
        })

    # DTS-specific rules
    if dts == 1:
        provisions.append({
            "madde": "Bölüm 3, Tablo 3.2",
            "metin": f"DTS 1 bölgesi (SDS={sds:.2f}g ≥ 0.75g). En yüksek deprem tasarım gereklilikleri uygulanır."
        })
    elif dts == 2:
        provisions.append({
            "madde": "Bölüm 3, Tablo 3.2",
            "metin": f"DTS 2 bölgesi (0.50g ≤ SDS={sds:.2f}g < 0.75g). Yüksek deprem gereklilikleri."
        })
    elif dts == 3:
        provisions.append({
            "madde": "Bölüm 3, Tablo 3.2",
            "metin": f"DTS 3 bölgesi (SDS={sds:.2f}g). Orta deprem gereklilikleri."
        })
    else:
        provisions.append({
            "madde": "Bölüm 3, Tablo 3.2",
            "metin": f"DTS 4 bölgesi (SDS={sds:.2f}g < 0.25g). Düşük deprem gereklilikleri."
        })

    # Soil-specific provisions
    if soil_class == "ZE":
        provisions.append({
            "madde": "Bölüm 16, Madde 16.5–16.7",
            "metin": "ZE zemin: Sahaya özgü zemin davranış analizi zorunludur. "
                     "Kapsamlı jeoteknik araştırma yapılmalıdır."
        })
        if dts in (1, 2):
            warnings.append("UYARI: DTS 1/2 + ZE zemin kombinasyonu en riskli kategori. "
                            "Her yapı için bağımsız zemin mühendisi raporu gereklidir. "
                            "(Bölüm 16, Madde 16.6)")
    elif soil_class == "ZD":
        provisions.append({
            "madde": "Bölüm 16, Tablo 16.1",
            "metin": "ZD zemin (Vs30=180–360 m/s): Standart zemin araştırması yeterlidir. "
                     "Temel tasarımında zemin büyütme katsayısı FS=1.6 kullanılır."
        })

    # Height limit check
    if max_floors == 0:
        warnings.append(
            f"YASAK: Yığma yapı {soil_class} zeminlerde DTS {dts}'de yapılamaz. "
            f"(Bölüm 4, Tablo 4.1)"
        )
    elif max_floors is not None and floors > max_floors:
        warnings.append(
            f"UYARI: {floors} katlı {material} yapı bu kombinasyonda (DTS {dts}, {soil_class}) "
            f"izin verilen maksimum kat sayısını ({max_floors}) aşıyor. "
            f"(Bölüm 4, Tablo 4.1)"
        )
    elif max_floors is None:
        provisions.append({
            "madde": "Bölüm 4, Tablo 4.1",
            "metin": f"{material.title()} yapı bu bölgede özel deprem tasarımı ile "
                     f"yüksek binalara izin verilebilir. Bölüm 13 hükümleri geçerlidir."
        })
    else:
        provisions.append({
            "madde": "Bölüm 4, Tablo 4.1",
            "metin": f"{floors} katlı yapı izin verilen maksimum kat sayısı içindedir "
                     f"(maks. {max_floors} kat, DTS {dts}, {soil_class})."
        })

    # Material recommendation
    if material.lower() == "yığma" and dts in (1, 2):
        warnings.append(
            f"UYARI: Yığma yapı DTS {dts}'de {soil_class} zeminlerde "
            f"{'yasaktır' if soil_class == 'ZE' else 'max 2 kat ile sınırlıdır'}. "
            f"Betonarme veya çelik tercih edin. (Bölüm 4, Tablo 4.1)"
        )

    if soil_class in ("ZA", "ZB") and dts in (1, 2):
        provisions.append({
            "madde": "Bölüm 4",
            "metin": "Çelik yapılar ZA–ZB zeminlerde tercih edilir — daha yüksek sünek davranış."
        })

    return {
        "dts": dts,
        "dts_label": f"DTS {dts}",
        "bks": bks_num,
        "bks_label": bks_label,
        "risk_color": risk_color,
        "max_floors": max_floors,
        "provisions": provisions,
        "warnings": warnings,
        "performans_hedefi": (
            "Sınırlı Hasar (SH) — DD-2'de" if bks_num == 2
            else "Kontrollü Hasar (KH) — DD-2'de"
        ),
        "psha_override": psha_override,
        "hazard_probability_M6_50yr": hazard_probability_M6_50yr,
    }

# ── Function 4: generate_risk_report ──────────────────────────────────────────

def generate_risk_report(
    lat: float, lon: float,
    building_type: str = "konut",
    floors: int = 5,
    material: str = "betonarme",
    vs30: Optional[float] = None,
    radius_km: float = 50,
) -> dict:
    """
    Full TBDY-2018 risk report for a given site and building.
    Returns structured JSON — deterministic, no LLM.
    """
    soil_info   = get_soil_class(lat, lon, vs30=vs30)
    seismicity  = get_local_seismicity(lat, lon, radius_km=radius_km)

    # Try PSHA integration (Step 5 — OSTA override)
    _psha_prob_M6 = None
    try:
        import importlib as _il, sys as _sys
        _sys.path.insert(0, str(ROOT / "scripts"))
        _haz = _il.import_module("43_hazard_calculator")
        _psha_prob_M6 = _haz.get_hazard_probability(lat, lon, 6.0, 50)["probability"]
    except Exception:
        pass

    recommend   = get_tbdy_recommendation(
        soil_info["soil_class"], soil_info["SDS"],
        building_type, floors, material,
        hazard_probability_M6_50yr=_psha_prob_M6,
    )

    return {
        "site": {"lat": lat, "lon": lon},
        "building": {
            "type": building_type, "floors": floors, "material": material
        },
        "soil": soil_info,
        "seismicity": seismicity,
        "tbdy2018": recommend,
        "summary": {
            "risk_color": recommend["risk_color"],
            "dts": recommend["dts_label"],
            "soil_class": soil_info["soil_class"],
            "sds": soil_info["SDS"],
            "n_warnings": len(recommend["warnings"]),
            "verdict": (
                "Yüksek Risk — Uzman incelemesi zorunlu" if recommend["risk_color"] == "kırmızı" else
                "Orta Risk — Dikkatli tasarım gerekli"   if recommend["risk_color"] in ("turuncu", "sarı") else
                "Düşük Risk — Standart tasarım yeterli"
            )
        }
    }


if __name__ == "__main__":
    # Quick test: İstanbul Anadolu yakası, hastane, 8 kat, betonarme
    report = generate_risk_report(
        lat=40.95, lon=29.10,
        building_type="hastane",
        floors=8,
        material="betonarme",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
