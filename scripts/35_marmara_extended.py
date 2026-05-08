# Phase: Quality Audit — Optimization Problem C
# Purpose: Extended Marmara generalization test using real Western Marmara waveforms.
#          Queries all available KO-network stations in the Marmara bounding box,
#          downloads M≥1.5 event windows for Feb–Aug 2023 using ISC catalog,
#          and evaluates GPD-FT, PhaseNet-Fixed, and Ensemble.
#          Key question: does recall drop significantly from Kahramanmaras
#          (training region) to Western Marmara (deployment target region)?
# Inputs:  data/catalog/small_events_catalog.csv (ISC), models/gpd_final.pt,
#          models/phasenet_fixed.pt
# Outputs: data/windows/marmara/*.npz, artifacts/marmara_extended_results.json,
#          figures/kahramanmaras_vs_marmara_performance.png
# Limitations: Western Marmara seismicity rate is lower than Kahramanmaras 2023
#              aftershock sequence. M≥1.5 events may be sparse. KO station
#              coverage varies; not all stations in bbox return data.

from pathlib import Path
import json, random
import numpy as np, pandas as pd
import torch
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, resample_poly
from math import gcd
from obspy import UTCDateTime, Stream, Trace
from obspy.clients.fdsn import Client
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seisbench.models as sbm

ROOT    = Path(__file__).resolve().parent.parent
WIN_DIR = ROOT / "data" / "windows" / "marmara"
ART     = ROOT / "artifacts"
MDL_DIR = ROOT / "models"
FIG_DIR = ROOT / "figures"
WIN_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

# ── Marmara bounding box ─────────────────────────────────────────────────────
MINLAT, MAXLAT = 39.5, 41.5
MINLON, MAXLON = 26.0, 30.5
MINMAG         = 1.5
TSTART_STR     = "2023-02-06"
TEND_STR       = "2023-08-06"
TSTART         = UTCDateTime(TSTART_STR)
TEND           = UTCDateTime(TEND_STR)

SRATE_TARGET = 100.0
WIN_LEN_S    = 60.0
ORIGIN_OFFSET= 30.0
N_SAMPLES    = int(SRATE_TARGET * WIN_LEN_S)
BP_LO, BP_HI = 1.0, 45.0
MAX_GAP_FRAC = 0.05
REQUIRED_CH  = ("HHZ", "HHN", "HHE")
RANDOM_SEED  = 42

DETECT_TOL  = 5.0
DETECT_POST = 25.0
DETECT_LO   = int((ORIGIN_OFFSET - DETECT_TOL) * SRATE_TARGET)
DETECT_HI   = int((ORIGIN_OFFSET + DETECT_POST) * SRATE_TARGET)
T0_REF      = UTCDateTime("2023-02-06T01:00:00")

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DEVICE = "cuda" if torch.cuda.is_available() else (
         "mps"  if torch.backends.mps.is_available() else "cpu")

print("=" * 70)
print("PROBLEM C — Marmara Extended Generalization Test")
print(f"  Region: Lat {MINLAT}–{MAXLAT}, Lon {MINLON}–{MAXLON}")
print(f"  Period: {TSTART_STR} → {TEND_STR}  |  Mag: M≥{MINMAG}")
print("=" * 70)

def bandpass(data, lo, hi, fs):
    nyq = fs / 2.0
    b, a = butter(4, [lo/nyq, hi/nyq], btype="band")
    return filtfilt(b, a, data)

def safe_resample(data_arr, src_fs, tgt_fs):
    if abs(src_fs - tgt_fs) < 0.01: return data_arr.astype(np.float64)
    g = gcd(int(round(tgt_fs)), int(round(src_fs)))
    return resample_poly(data_arr.astype(np.float64), int(round(tgt_fs))//g, int(round(src_fs))//g)

# ── Step 1: Fetch Marmara stations ─────────────────────────────────────────
print("\n[1] Querying KO network stations in Marmara box ...")
client = None
for base_url in ["http://eida.koeri.boun.edu.tr", "ORFEUS"]:
    try:
        client = Client(base_url)
        inv = client.get_stations(
            network="KO", station="*", channel="HH*",
            minlatitude=MINLAT, maxlatitude=MAXLAT,
            minlongitude=MINLON, maxlongitude=MAXLON,
            starttime=TSTART, endtime=TEND, level="station")
        print(f"    ✓ Connected via {base_url}")
        break
    except Exception as e:
        print(f"    ✗ {base_url}: {e}")
        client = None

stations = []
if client and inv:
    for net in inv:
        for sta in net:
            stations.append({
                "network": net.code,
                "station": sta.code,
                "net_sta": f"{net.code}.{sta.code}",
                "latitude": sta.latitude,
                "longitude": sta.longitude,
                "elevation": sta.elevation,
            })
    print(f"    Found {len(stations)} KO HH stations in Marmara box:")
    for s in stations:
        print(f"      {s['net_sta']:<12}  Lat={s['latitude']:.3f}  Lon={s['longitude']:.3f}")
else:
    # Fallback: known Marmara KO stations
    stations = [
        {"network":"KO","station":"RKY","net_sta":"KO.RKY","latitude":40.374,"longitude":27.958,"elevation":143},
        {"network":"KO","station":"KRBG","net_sta":"KO.KRBG","latitude":40.699,"longitude":27.292,"elevation":30},
        {"network":"KO","station":"TRTX","net_sta":"KO.TRTX","latitude":40.780,"longitude":27.690,"elevation":204},
        {"network":"KO","station":"BOZC","net_sta":"KO.BOZC","latitude":40.016,"longitude":27.966,"elevation":60},
        {"network":"KO","station":"KANT","net_sta":"KO.KANT","latitude":40.689,"longitude":29.810,"elevation":158},
        {"network":"KO","station":"MALT","net_sta":"KO.MALT","latitude":40.100,"longitude":26.370,"elevation":200},
    ]
    print(f"    EIDA query failed — using {len(stations)} known Marmara stations as fallback")
    for s in stations:
        print(f"      {s['net_sta']}")

# ── Step 2: Fetch Marmara catalog ───────────────────────────────────────────
print(f"\n[2] Fetching M≥{MINMAG} events from ISC for Marmara region ...")
import requests
isc_url = (
    "http://www.isc.ac.uk/fdsnws/event/1/query"
    f"?starttime={TSTART_STR}T00:00:00&endtime={TEND_STR}T23:59:59"
    f"&minlat={MINLAT}&maxlat={MAXLAT}&minlon={MINLON}&maxlon={MAXLON}"
    f"&minmag={MINMAG}&format=text&nodata=404"
)
marmara_events = []
try:
    resp = requests.get(isc_url, timeout=120)
    if resp.status_code == 200:
        lines = [l for l in resp.text.strip().split("\n") if not l.startswith("#") and l.strip()]
        for line in lines[1:]:
            parts = line.split("|")
            if len(parts) < 11: continue
            try:
                marmara_events.append({
                    "event_id":    parts[0].strip(),
                    "origin_time": parts[1].strip(),
                    "latitude":    float(parts[2]),
                    "longitude":   float(parts[3]),
                    "depth_km":    float(parts[4]) if parts[4].strip() else 10.0,
                    "magnitude":   float(parts[10]) if parts[10].strip() else None,
                })
            except Exception: continue
    print(f"    ISC returned {len(marmara_events)} events")
except Exception as e:
    print(f"    ✗ ISC HTTP failed: {e}")

# Also try EMSC for Marmara
try:
    from obspy.clients.fdsn import Client as FDSNClient
    emsc_c = FDSNClient("EMSC")
    cat_m = emsc_c.get_events(
        starttime=TSTART, endtime=TEND,
        minlatitude=MINLAT, maxlatitude=MAXLAT,
        minlongitude=MINLON, maxlongitude=MAXLON,
        minmagnitude=MINMAG)
    for ev in cat_m:
        o = ev.preferred_origin() or ev.origins[0]
        m = ev.preferred_magnitude() or (ev.magnitudes[0] if ev.magnitudes else None)
        marmara_events.append({
            "event_id":    str(ev.resource_id),
            "origin_time": str(o.time),
            "latitude":    float(o.latitude),
            "longitude":   float(o.longitude),
            "depth_km":    float(o.depth/1000) if o.depth else 10.0,
            "magnitude":   float(m.mag) if m else None,
        })
    print(f"    EMSC added {len(cat_m)} events")
except Exception as e:
    print(f"    EMSC: {e}")

# Deduplicate
ev_df = pd.DataFrame(marmara_events).dropna(subset=["magnitude"])
ev_df["origin_time_dt"] = pd.to_datetime(ev_df["origin_time"], utc=True, errors="coerce")
ev_df = ev_df.dropna(subset=["origin_time_dt"])
ev_df["lat_r"] = ev_df["latitude"].round(2)
ev_df["lon_r"] = ev_df["longitude"].round(2)
ev_df["time_r"] = ev_df["origin_time_dt"].dt.floor("10s")
ev_df = ev_df.drop_duplicates(subset=["lat_r","lon_r","time_r"]).reset_index(drop=True)
print(f"    Total deduplicated: {len(ev_df)} Marmara events (M≥{MINMAG})")

if len(ev_df) == 0:
    print("  ✗ No Marmara events found — cannot proceed with download")
    print("  Exiting.")
    import sys; sys.exit(0)

# ── Step 3: Download waveforms ───────────────────────────────────────────────
print(f"\n[3] Downloading waveforms ({len(ev_df)} events × {len(stations)} stations) ...")
if client is None:
    try: client = Client("http://eida.koeri.boun.edu.tr")
    except: client = Client("ORFEUS")

inventory_rows = []
n_ok = 0; n_fail = 0

for i_ev, ev in ev_df.iterrows():
    t_origin = UTCDateTime(float(ev["origin_time_dt"].timestamp()))
    t_start  = t_origin - ORIGIN_OFFSET
    t_end    = t_origin + (WIN_LEN_S - ORIGIN_OFFSET)
    mag = float(ev["magnitude"])
    ev_id = str(ev["event_id"])

    for sta in stations:
        net = sta["network"]; sname = sta["station"]
        safe_id = ev_id.replace("/","_").replace(":","_")[:35]
        fname = f"marmara_{sta['net_sta']}_{safe_id}.npz"
        out_path = WIN_DIR / fname

        if out_path.exists():
            n_ok += 1
            inventory_rows.append({"filename":fname,"station":sta["net_sta"],
                "event_id":ev_id,"magnitude":mag,"t_origin":str(t_origin),"status":"cached"})
            continue
        try:
            st = client.get_waveforms(network=net, station=sname, location="*",
                                      channel="HH*", starttime=t_start, endtime=t_end+5)
            st.merge(method=1, fill_value=0)
            st.detrend("demean"); st.detrend("linear")
            present = {tr.stats.channel for tr in st}
            missing = [c for c in REQUIRED_CH if c not in present]
            if missing: raise ValueError(f"Missing: {missing}")
            channels_out = []
            for ch in REQUIRED_CH:
                tr = st.select(channel=ch)[0].copy()
                data_ch = bandpass(tr.data.astype(np.float64), BP_LO, BP_HI, tr.stats.sampling_rate)
                data_ch = safe_resample(data_ch, tr.stats.sampling_rate, SRATE_TARGET)
                if len(data_ch) >= N_SAMPLES: data_ch = data_ch[:N_SAMPLES]
                else: data_ch = np.pad(data_ch, (0, N_SAMPLES-len(data_ch)))
                std = float(np.std(data_ch))
                if std > 1e-9: data_ch = data_ch / std
                channels_out.append(data_ch.astype(np.float32))
            waveform = np.stack(channels_out, axis=0)
            np.savez(out_path, data=waveform,
                     station=np.bytes_(sta["net_sta"]),
                     event_id=np.bytes_(ev_id), magnitude=np.float32(mag),
                     t_origin=np.bytes_(str(t_origin)))
            n_ok += 1
            inventory_rows.append({"filename":fname,"station":sta["net_sta"],
                "event_id":ev_id,"magnitude":mag,"t_origin":str(t_origin),"status":"ok"})
        except Exception as e:
            n_fail += 1
            inventory_rows.append({"filename":fname,"station":sta["net_sta"],
                "event_id":ev_id,"magnitude":mag,"t_origin":str(t_origin),
                "status":f"fail:{str(e)[:60]}"})
    if (i_ev+1) % 20 == 0:
        print(f"    {i_ev+1}/{len(ev_df)} events  OK={n_ok}  FAIL={n_fail}")

inv_df = pd.DataFrame(inventory_rows)
print(f"\n    Download complete: {n_ok} OK, {n_fail} fail")
if n_ok == 0:
    print("  ✗ No Marmara windows downloaded — cannot evaluate.")
    import sys; sys.exit(0)

# ── Step 4: Load models ──────────────────────────────────────────────────────
print("\n[4] Loading models ...")
gpd_model = sbm.GPD.from_pretrained("original")
gpd_model.load_state_dict(torch.load(str(MDL_DIR/"gpd_final.pt"), map_location=DEVICE, weights_only=True))
gpd_model.to(DEVICE).eval()
GPD_WLEN = getattr(gpd_model,"in_samples",400)
GPD_P_IDX = gpd_model.labels.index("P")

pn_fixed = sbm.PhaseNet.from_pretrained("original")
pn_fixed.load_state_dict(torch.load(str(MDL_DIR/"phasenet_fixed.pt"), map_location=DEVICE, weights_only=True))
pn_fixed.to(DEVICE).eval()
print("    GPD-FT + PhaseNet-Fixed loaded")

def gpd_prob(model, data):
    wlen = GPD_WLEN; p_idx = GPD_P_IDX; stride = 10
    positions = list(range(0, N_SAMPLES-wlen+1, stride))
    crops = []
    for s in positions:
        c = data[:,s:s+wlen].astype(np.float32)
        pk = float(np.abs(c).max())
        if pk > 1e-9: c = c/pk
        crops.append(c)
    probs = []
    with torch.no_grad():
        for i in range(0,len(crops),128):
            b = torch.tensor(np.stack(crops[i:i+128])).to(DEVICE)
            probs.extend(model(b)[:,p_idx].cpu().numpy().tolist())
    ctr = np.array([s+wlen//2 for s in positions],dtype=float)
    pa = np.array(probs,dtype=float)
    f = interp1d(ctr,pa,kind="linear",bounds_error=False,fill_value=(pa[0],pa[-1]))
    return f(np.arange(N_SAMPLES,dtype=float)).astype(np.float32)

def pn_prob(model, data):
    st = Stream()
    for i,ch in enumerate(["HHZ","HHN","HHE"]):
        tr = Trace(data=data[i].astype(np.float32))
        tr.stats.network="KO"; tr.stats.station="TEST"
        tr.stats.channel=ch; tr.stats.sampling_rate=SRATE_TARGET
        tr.stats.starttime=T0_REF; st.append(tr)
    with torch.no_grad(): ann = model.annotate(st)
    tr_p = next((t for t in ann if t.stats.channel.endswith("_P")),None)
    if tr_p is None: return np.zeros(N_SAMPLES,dtype=np.float32)
    tgt = np.arange(N_SAMPLES)/SRATE_TARGET
    src = float(tr_p.stats.starttime-T0_REF)+np.arange(len(tr_p.data))/tr_p.stats.sampling_rate
    if len(src)<2: return np.zeros(N_SAMPLES,dtype=np.float32)
    f = interp1d(src,tr_p.data.astype(np.float32),kind="linear",bounds_error=False,fill_value=0.0)
    return f(tgt).astype(np.float32)

def detect(prob, thr):
    return float(prob[DETECT_LO:DETECT_HI+1].max()) >= thr

# ── Step 5: Inference ────────────────────────────────────────────────────────
ok_inv = inv_df[inv_df["status"].isin(["ok","cached"])].reset_index(drop=True)
print(f"\n[5] Running inference on {len(ok_inv)} Marmara windows ...")
rows = []
for i, row in ok_inv.iterrows():
    p = WIN_DIR / row["filename"]
    if not p.exists(): continue
    try:
        d = np.load(p); data = d["data"].astype(np.float32)
        gp = gpd_prob(gpd_model, data)
        pp = pn_prob(pn_fixed, data)
        ep = 0.5*gp + 0.5*pp
        rows.append({
            "station": row["station"], "magnitude": float(row["magnitude"]),
            "gpd_det": detect(gp,0.50), "pn_fixed_det": detect(pp,0.10), "ens_det": detect(ep,0.15),
        })
    except Exception: pass
    if (i+1)%30==0: print(f"    {i+1}/{len(ok_inv)}")

df = pd.DataFrame(rows)
print(f"    Done: {len(df)} windows")

# ── Step 6: Compute and compare ──────────────────────────────────────────────
print("\n[6] Results ...")
gpd_r  = float(df["gpd_det"].mean())     if len(df)>0 else 0
pnf_r  = float(df["pn_fixed_det"].mean())if len(df)>0 else 0
ens_r  = float(df["ens_det"].mean())     if len(df)>0 else 0

# Kahramanmaras results (from quality_audit_final)
KAHR_GPD = 1.000; KAHR_ENS = 1.000; KAHR_PN_FIXED = None

print(f"\n  {'Region':<20} {'N':>5}  {'GPD-FT':>8}  {'PN-Fixed':>10}  {'Ensemble':>10}")
print(f"  {'-'*57}")
print(f"  {'Kahramanmaras (train)':<20} {'944':>5}  {KAHR_GPD:>8.4f}  {'N/A':>10}  {KAHR_ENS:>10.4f}")
print(f"  {'Marmara (deployment)':<20} {len(df):>5}  {gpd_r:>8.4f}  {pnf_r:>10.4f}  {ens_r:>10.4f}")

delta_gpd = gpd_r - KAHR_GPD; delta_ens = ens_r - KAHR_ENS
print(f"\n  Transfer gap:")
print(f"    GPD-FT:   Kahramanmaras {KAHR_GPD:.4f} → Marmara {gpd_r:.4f}  Δ={delta_gpd:+.4f}")
print(f"    Ensemble: Kahramanmaras {KAHR_ENS:.4f} → Marmara {ens_r:.4f}  Δ={delta_ens:+.4f}")

if abs(delta_gpd) < 0.10:
    print("  ✓ Transfer gap < 10% — model generalizes across regions")
elif abs(delta_gpd) < 0.20:
    print("  ⚠ Transfer gap 10–20% — moderate domain shift")
else:
    print("  ✗ Transfer gap > 20% — significant domain shift to Marmara")

# ── Figure ───────────────────────────────────────────────────────────────────
print("\n[7] Generating figure ...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor("#0a0e1a")
for ax in axes:
    ax.set_facecolor("#0f1525")
    ax.tick_params(colors="#a0aabb")
    for sp in ax.spines.values(): sp.set_edgecolor("#2a3450")

ax = axes[0]
models_bar = ["GPD-FT", "PN-Fixed", "Ensemble"]
kahr_vals  = [KAHR_GPD, 0.94, KAHR_ENS]   # approximate PN-Fixed from small event eval
marm_vals  = [gpd_r, pnf_r, ens_r]
x = np.arange(len(models_bar)); w = 0.35
ax.bar(x-w/2, kahr_vals, w, label="Kahramanmaraş (train)", color="#00ff88", alpha=0.85)
ax.bar(x+w/2, marm_vals, w, label="Western Marmara (deployment)", color="#ffaa44", alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(models_bar, color="#a0aabb")
ax.set_ylabel("Recall", color="#a0aabb"); ax.set_ylim(0,1.15)
ax.set_title("Recall: Training Region vs Deployment Region", color="#e8ecf0", fontsize=10)
ax.legend(fontsize=8, labelcolor="#e8ecf0", facecolor="#1a2035", edgecolor="#2a3450")

ax2 = axes[1]
per_sta = df.groupby("station")[["gpd_det","ens_det"]].mean().reset_index()
x2 = np.arange(len(per_sta))
ax2.bar(x2-0.2, per_sta["gpd_det"], 0.35, label="GPD-FT", color="#00ff88", alpha=0.85)
ax2.bar(x2+0.2, per_sta["ens_det"], 0.35, label="Ensemble", color="#ffaa44", alpha=0.85)
ax2.set_xticks(x2); ax2.set_xticklabels(per_sta["station"].str.replace("KO.",""),
    rotation=30, ha="right", fontsize=8, color="#a0aabb")
ax2.set_ylabel("Recall", color="#a0aabb"); ax2.set_ylim(0,1.15)
ax2.set_title("Per-Station Recall (Marmara)", color="#e8ecf0", fontsize=10)
ax2.legend(fontsize=8, labelcolor="#e8ecf0", facecolor="#1a2035", edgecolor="#2a3450")

plt.suptitle("Problem C — Kahramanmaraş vs Western Marmara Generalization",
             color="#e8ecf0", fontsize=11, y=1.02)
plt.tight_layout()
fp = FIG_DIR/"kahramanmaras_vs_marmara_performance.png"
plt.savefig(fp, dpi=150, bbox_inches="tight", facecolor="#0a0e1a"); plt.close()
print(f"    Saved → {fp}")

# ── Save JSON ────────────────────────────────────────────────────────────────
out = {
    "phase": "marmara_extended",
    "marmara_bbox": {"minlat":MINLAT,"maxlat":MAXLAT,"minlon":MINLON,"maxlon":MAXLON},
    "n_stations": len(stations),
    "n_events": len(ev_df),
    "n_windows": len(df),
    "kahramanmaras_reference": {"gpd_ft":KAHR_GPD,"ensemble":KAHR_ENS},
    "marmara_results": {"gpd_ft":round(gpd_r,4),"pn_fixed":round(pnf_r,4),"ensemble":round(ens_r,4)},
    "transfer_gap": {"gpd_ft":round(delta_gpd,4),"ensemble":round(delta_ens,4)},
}
jp = ART/"marmara_extended_results.json"
with open(jp,"w") as f: json.dump(out,f,indent=2)
print(f"    Saved → {jp}")

print("\n"+"="*70)
print("PROBLEM C COMPLETE — Marmara Extended Test")
print(f"  Events: {len(ev_df)}  |  Windows: {len(df)}")
print(f"  GPD-FT:   Marmara={gpd_r:.4f}  Δ={delta_gpd:+.4f} vs Kahramanmaras")
print(f"  Ensemble: Marmara={ens_r:.4f}  Δ={delta_ens:+.4f} vs Kahramanmaras")
print("="*70)
