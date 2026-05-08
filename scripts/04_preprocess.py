# Phase: 2
# Purpose: Preprocess raw miniSEED windows into standardized 3-component NPZ arrays
# Inputs: data/raw/*.mseed (from 03_download_waveforms.py), artifacts/waveform_inventory.csv
# Outputs: data/windows/*.npz (standardized), artifacts/waveform_inventory.csv (updated),
#          figures/pilot_event_examples.png, figures/pilot_noise_examples.png
# Limitations: Windows with >5% gap fraction are discarded; single-component windows are
#              discarded (require all 3 of HHZ, HHN, HHE); normalization is per-trace z-score
#              which may amplify noise in very quiet windows.

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from obspy import read, Stream
from obspy.core.inventory import read_inventory

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
RAW_DIR    = ROOT / "data" / "raw"
WIN_DIR    = ROOT / "data" / "windows"
FIG_DIR    = ROOT / "figures"
ARTIFACT   = ROOT / "artifacts" / "waveform_inventory.csv"
WIN_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Processing parameters ──────────────────────────────────────────────────────
TARGET_SRATE    = 100.0       # Hz — SeisBench standard
BANDPASS_FREQS  = (1.0, 45.0) # Hz — standard for regional seismology
WIN_LEN_S       = 60.0        # seconds
N_SAMPLES       = int(TARGET_SRATE * WIN_LEN_S)  # 6000
MAX_GAP_FRAC    = 0.05        # discard window if >5% samples are zero-gap-fill
REQUIRED_CHANS  = ("HHZ", "HHN", "HHE")

# ── Load inventory ─────────────────────────────────────────────────────────────
print("=" * 70)
print("PHASE 2 — Preprocessing: miniSEED → NPZ (standardized windows)")
print(f"  Filter     : {BANDPASS_FREQS[0]}–{BANDPASS_FREQS[1]} Hz bandpass (4th order Butterworth, zero-phase)")
print(f"  Resample   : {TARGET_SRATE} Hz")
print(f"  Window     : {WIN_LEN_S} s ({N_SAMPLES} samples per trace)")
print(f"  Normalize  : per-trace z-score")
print(f"  Discard if : any required channel missing, or >{MAX_GAP_FRAC*100:.0f}% gap")
print("=" * 70)

inv_df = pd.read_csv(ARTIFACT)
# Only process rows with a valid downloaded file
proc_df = inv_df[inv_df["status"].isin(["ok", "exists"])].copy()
print(f"\n[0] Valid miniSEED files in inventory: {len(proc_df)}")

def load_and_check(mseed_path: Path):
    """
    Load a miniSEED file and return a 3-component obspy Stream or None if invalid.
    Checks for: file existence, required channels, gap fraction.
    """
    if not mseed_path.exists():
        return None, "file_not_found"
    try:
        st = read(str(mseed_path))
    except Exception as e:
        return None, f"read_error: {e}"

    # Merge gaps with zero-fill
    st.merge(fill_value=0)

    # Check all required channels present
    present = {tr.stats.channel for tr in st}
    for ch in REQUIRED_CHANS:
        if ch not in present:
            return None, f"missing_channel_{ch}"

    # Check gap fraction (zero-fill)
    for tr in st:
        if tr.stats.channel in REQUIRED_CHANS:
            zero_frac = np.mean(tr.data == 0)
            if zero_frac > MAX_GAP_FRAC:
                return None, f"gap_fraction_{zero_frac:.2f}"

    return st, "ok"


def process_stream(st: Stream, target_npts: int) -> np.ndarray:
    """
    Apply full preprocessing pipeline to a 3-component stream.
    Returns numpy array of shape (3, target_npts): [HHZ, HHN, HHE].
    """
    st = st.copy()
    for tr in st:
        tr.detrend("linear")

    # Bandpass filter (zero-phase, 4th order Butterworth)
    st.filter("bandpass",
              freqmin=BANDPASS_FREQS[0],
              freqmax=BANDPASS_FREQS[1],
              corners=4,
              zerophase=True)

    # Resample to target sample rate
    st.resample(TARGET_SRATE)

    # Build 3-component array [Z, N, E]
    out = np.zeros((3, target_npts), dtype=np.float32)
    ch_map = {"HHZ": 0, "HHN": 1, "HHE": 2}
    for tr in st:
        if tr.stats.channel in ch_map:
            idx = ch_map[tr.stats.channel]
            data = tr.data.astype(np.float32)
            # Trim or pad to exactly target_npts
            n = min(len(data), target_npts)
            out[idx, :n] = data[:n]

    # Per-trace z-score normalization
    for i in range(3):
        std = out[i].std()
        if std > 0:
            out[i] = out[i] / std

    return out


# ── Process all windows ────────────────────────────────────────────────────────
n_ok       = 0
n_discard  = 0
discard_reasons = {}
npz_records = []

# Group by (file, window_type) to process each mseed once
print("\n[1] Processing miniSEED windows...")
for _, row in proc_df.iterrows():
    mseed_path = RAW_DIR / row["file"]
    npz_fname  = WIN_DIR / (mseed_path.stem + ".npz")

    # Skip if already processed
    if npz_fname.exists():
        n_ok += 1
        npz_records.append({"file": row["file"], "npz": npz_fname.name,
                             "window_type": row["window_type"],
                             "station": row["station"],
                             "event_id": row["event_id"],
                             "magnitude": row["magnitude"],
                             "t_start": row["t_start"],
                             "status": "ok"})
        continue

    st, reason = load_and_check(mseed_path)
    if st is None:
        n_discard += 1
        discard_reasons[reason] = discard_reasons.get(reason, 0) + 1
        npz_records.append({"file": row["file"], "npz": "",
                             "window_type": row["window_type"],
                             "station": row["station"],
                             "event_id": row["event_id"],
                             "magnitude": row["magnitude"],
                             "t_start": row["t_start"],
                             "status": reason})
        continue

    try:
        data = process_stream(st, N_SAMPLES)
        np.savez_compressed(npz_fname,
                            data=data,
                            window_type=row["window_type"],
                            station=row["station"],
                            event_id=str(row["event_id"]),
                            magnitude=float(row["magnitude"]),
                            t_start=str(row["t_start"]))
        n_ok += 1
        npz_records.append({"file": row["file"], "npz": npz_fname.name,
                             "window_type": row["window_type"],
                             "station": row["station"],
                             "event_id": row["event_id"],
                             "magnitude": row["magnitude"],
                             "t_start": row["t_start"],
                             "status": "ok"})
    except Exception as e:
        n_discard += 1
        reason = f"process_error: {str(e)[:60]}"
        discard_reasons[reason] = discard_reasons.get(reason, 0) + 1
        npz_records.append({"file": row["file"], "npz": "",
                             "window_type": row["window_type"],
                             "station": row["station"],
                             "event_id": row["event_id"],
                             "magnitude": row["magnitude"],
                             "t_start": row["t_start"],
                             "status": reason})

print(f"    Processed OK : {n_ok}")
print(f"    Discarded    : {n_discard}")
if discard_reasons:
    for r, c in sorted(discard_reasons.items(), key=lambda x: -x[1]):
        print(f"      {r}: {c}")

# ── Update inventory with NPZ info ─────────────────────────────────────────────
npz_df = pd.DataFrame(npz_records)
npz_df.to_csv(ROOT / "artifacts" / "waveform_inventory.csv", index=False)
print(f"\n[2] Updated waveform_inventory.csv  ({len(npz_df)} rows)")

# ── Produce example figures ────────────────────────────────────────────────────
print("\n[3] Generating example figures...")

# Load up to 5 event NPZ and 5 noise NPZ for plotting
event_npzs = [p for p in WIN_DIR.glob("event_*.npz")][:5]
noise_npzs = [p for p in WIN_DIR.glob("noise_*.npz")][:5]

t_axis = np.linspace(0, WIN_LEN_S, N_SAMPLES)
chan_labels = ["HHZ", "HHN", "HHE"]
chan_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

def make_waveform_figure(npz_paths, title_prefix, out_file, wtype):
    n_rows = len(npz_paths)
    if n_rows == 0:
        print(f"    No {wtype} NPZ files found — skipping figure.")
        return

    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 3 * n_rows),
                              squeeze=False)
    fig.suptitle(f"{title_prefix} — Preprocessed Waveforms (Kahramanmaraş Pilot)",
                 fontsize=13, fontweight="bold")

    for row_i, npz_path in enumerate(npz_paths):
        npz = np.load(npz_path, allow_pickle=True)
        data = npz["data"]           # shape (3, 6000)
        mag  = float(npz.get("magnitude", 0))
        sta  = str(npz.get("station", ""))
        t0   = str(npz.get("t_start", ""))[:19]

        for ci in range(3):
            ax = axes[row_i][ci]
            ax.plot(t_axis, data[ci], lw=0.6, color=chan_colors[ci])
            ax.axvline(30.0, color="red", lw=1.0, ls="--", alpha=0.7,
                       label="Origin time" if wtype == "event" else "")
            ax.set_xlim(0, WIN_LEN_S)
            ax.set_title(f"{sta}  {chan_labels[ci]}"
                         + (f"  M{mag:.1f}" if wtype == "event" else ""),
                         fontsize=9)
            if row_i == n_rows - 1:
                ax.set_xlabel("Time (s)")
            if ci == 0:
                ax.set_ylabel(t0, fontsize=7, rotation=0, labelpad=60, ha="right")
            ax.tick_params(labelsize=7)

    plt.tight_layout()
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved → {out_file}")

make_waveform_figure(event_npzs,
                     "Event Windows",
                     FIG_DIR / "pilot_event_examples.png",
                     "event")
make_waveform_figure(noise_npzs,
                     "Noise Windows",
                     FIG_DIR / "pilot_noise_examples.png",
                     "noise")

# ── Final summary ──────────────────────────────────────────────────────────────
npz_all = list(WIN_DIR.glob("*.npz"))
event_n = len([p for p in npz_all if p.name.startswith("event_")])
noise_n = len([p for p in npz_all if p.name.startswith("noise_")])
total_mb = sum(p.stat().st_size for p in npz_all) / 1e6

print("\n" + "=" * 70)
print("PHASE 2 PREPROCESSING — COMPLETE")
print(f"  Event NPZ windows : {event_n}")
print(f"  Noise NPZ windows : {noise_n}")
print(f"  Total NPZ size    : {total_mb:.1f} MB")
print(f"  Inventory updated : artifacts/waveform_inventory.csv")
print(f"  Figures           : figures/pilot_event_examples.png")
print(f"                    : figures/pilot_noise_examples.png")
print("=" * 70)
print("  → Review figures and waveform_inventory.csv")
print("  → Type CONTINUE to proceed to Phase 3")
