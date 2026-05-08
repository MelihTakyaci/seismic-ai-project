# Phase: Quality Audit — Optimization 2
# Purpose: Improve S-pick MAE using real EMSC S picks (28,276 picks available in
#          koeri_real_picks.csv). Steps: (1) match EMSC S picks to test windows,
#          (2) re-evaluate MAE vs real reference, (3) fine-tune PhaseNet-Fixed
#          specifically for S-phase if matched S windows >= 200.
# Inputs:  data/augmented_dataset/waveforms.hdf5, data/augmented_dataset/metadata.csv,
#          artifacts/koeri_real_picks.csv, models/phasenet_fixed.pt
# Outputs: artifacts/s_phase_improved.json, figures/s_phase_improvement.png,
#          models/phasenet_s_optimized.pt (if S-label count sufficient)
# Limitations: EMSC S picks are analyst picks but may lag the true first S arrival
#              by 0.1–0.5s for distant events. Fine-tuning uses sigma=15 (wider)
#              to account for S-pick variability.

from pathlib import Path
import json, time, csv
import numpy as np
import pandas as pd
import torch
import h5py
from scipy.interpolate import interp1d
from obspy import Stream, Trace, UTCDateTime
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seisbench.models as sbm
from torch.utils.data import Dataset, DataLoader

ROOT   = Path(__file__).resolve().parent.parent
DS_DIR = ROOT / "data" / "augmented_dataset"
MDL_DIR = ROOT / "models"
ART    = ROOT / "artifacts"
FIG    = ROOT / "figures"

SRATE      = 100.0
N_TOTAL    = 6000
ORIGIN_S   = 30.0
S_DETECT_LO = int((ORIGIN_S + 1.0) * SRATE)
S_DETECT_HI = N_TOTAL - 1
T0_REF      = UTCDateTime("2023-02-06T01:00:00")

DEVICE = "cuda" if torch.cuda.is_available() else (
         "mps"  if torch.backends.mps.is_available() else "cpu")

print("=" * 70)
print("OPTIMIZATION 2 — S-Phase Improvement via EMSC S Picks")
print(f"  Device: {DEVICE}")
print("=" * 70)

# ── Step 1: Match EMSC S picks to test windows ────────────────────────────────
print("\n[1] Loading EMSC S picks ...")
picks_df = pd.read_csv(ART / "koeri_real_picks.csv")
s_picks  = picks_df[picks_df["phase"] == "S"].copy()
print(f"    EMSC S picks total: {len(s_picks)}")

def _bare_eid(eid):
    return str(eid).split("/")[-1]

s_picks["event_bare"]   = s_picks["event_id"].apply(_bare_eid)
s_picks["station_bare"] = s_picks["station"].str.replace(r"^KO\.", "", regex=True)

# Build S-pick lookup: (event_bare, station_bare) → sample
s_lookup = {}
for _, r in s_picks.iterrows():
    key = (r["event_bare"], r["station_bare"])
    s_lookup[key] = float(r["p_sample_in_window"])   # column name reused for S sample

print(f"    Unique (event, station) S-pick pairs: {len(s_lookup)}")

# Load test metadata
meta = pd.read_csv(DS_DIR / "metadata.csv")
meta["trace_p_arrival_sample"] = pd.to_numeric(meta["trace_p_arrival_sample"], errors="coerce")
meta["trace_s_arrival_sample"] = pd.to_numeric(meta["trace_s_arrival_sample"], errors="coerce")
if "augmentation" not in meta.columns: meta["augmentation"] = "original"

test_meta = meta[
    (meta["split"] == "test") & (meta["augmentation"] == "original")
].copy().reset_index(drop=True)

# Match S picks
def _parse_trace(row):
    name = str(row.get("trace_name", ""))
    stat, eid = "", ""
    if "_KO_" in name:
        parts = name.split("_KO_")
        if len(parts) > 1:
            stat = parts[1].split("_")[0]
    if "emsc_event_" in name:
        eid = name.split("emsc_event_")[-1].split(".")[0]
    return eid, stat

def match_emsc_s(row):
    eid, stat = _parse_trace(row)
    return s_lookup.get((eid, stat))

test_meta["emsc_s_sample"] = test_meta.apply(match_emsc_s, axis=1)
test_s_emsc = test_meta[test_meta["emsc_s_sample"].notna()].copy().reset_index(drop=True)
test_s_taupy = test_meta[test_meta["trace_s_arrival_sample"].notna()].copy().reset_index(drop=True)

print(f"\n    Test windows with TauPy S label:  {len(test_s_taupy)}")
print(f"    Test windows with EMSC S pick:    {len(test_s_emsc)}")

# Compare TauPy vs EMSC S for windows that have both
both_s = test_meta[
    test_meta["emsc_s_sample"].notna() & test_meta["trace_s_arrival_sample"].notna()
].copy()
if len(both_s) > 0:
    diff = (both_s["emsc_s_sample"] - both_s["trace_s_arrival_sample"]) / SRATE
    print(f"\n    TauPy vs EMSC S offset: mean={diff.mean():.3f}s  median={diff.median():.3f}s  "
          f"std={diff.std():.3f}s  N={len(both_s)}")

# ── Load PhaseNet-Fixed ────────────────────────────────────────────────────────
print("\n[2] Loading PhaseNet-Fixed ...")
pn_fixed = sbm.PhaseNet.from_pretrained("original")
pn_fixed.load_state_dict(torch.load(str(MDL_DIR/"phasenet_fixed.pt"), map_location=DEVICE, weights_only=True))
pn_fixed.to(DEVICE).eval()
print("    PhaseNet-Fixed ✓")

def pn_s_pick(model, data, thr=0.10):
    st = Stream()
    for i, ch in enumerate(["HHZ", "HHN", "HHE"]):
        tr = Trace(data=data[i].astype(np.float32))
        tr.stats.network = "KO"; tr.stats.station = "TEST"
        tr.stats.channel = ch;  tr.stats.sampling_rate = SRATE
        tr.stats.starttime = T0_REF; st.append(tr)
    with torch.no_grad(): ann = model.annotate(st)
    tgt = np.arange(N_TOTAL) / SRATE
    s_out = np.zeros(N_TOTAL, dtype=np.float32)
    for tr in ann:
        if not tr.stats.channel.endswith("_S"): continue
        src = float(tr.stats.starttime - T0_REF) + np.arange(len(tr.data)) / tr.stats.sampling_rate
        if len(src) < 2: continue
        f = interp1d(src, tr.data.astype(np.float32), kind="linear",
                     bounds_error=False, fill_value=0.0)
        s_out = f(tgt)
    w = s_out[S_DETECT_LO:S_DETECT_HI+1]
    pk = float(w.max())
    if pk < thr: return None
    return (S_DETECT_LO + int(w.argmax())) / SRATE

# ── Evaluate PhaseNet-Fixed S-MAE: TauPy reference vs EMSC reference ─────────
print("\n[3] Evaluating S-MAE on both reference sets ...")
hdf5 = DS_DIR / "waveforms.hdf5"

# Evaluate on EMSC-referenced S windows
rows_emsc = []
with h5py.File(hdf5, "r") as hf:
    for i, row in test_s_emsc.iterrows():
        name = row["trace_name"]
        if name not in hf["data"]: continue
        data = hf["data"][name][:]
        ref_s_emsc  = float(row["emsc_s_sample"]) / SRATE
        ref_s_taupy = float(row["trace_s_arrival_sample"]) / SRATE if pd.notna(row["trace_s_arrival_sample"]) else None
        pick = pn_s_pick(pn_fixed, data, thr=0.10)
        rows_emsc.append({
            "trace_name":   name,
            "ref_s_emsc":   ref_s_emsc,
            "ref_s_taupy":  ref_s_taupy,
            "pnx_s_pick":   pick,
            "mae_vs_emsc":  abs(pick - ref_s_emsc) if pick else None,
            "mae_vs_taupy": abs(pick - ref_s_taupy) if (pick and ref_s_taupy) else None,
        })
        if (i+1) % 50 == 0:
            print(f"    {len(rows_emsc)}/{len(test_s_emsc)}")

df_emsc = pd.DataFrame(rows_emsc)
print(f"    Evaluated {len(df_emsc)} windows")

mae_emsc  = df_emsc["mae_vs_emsc"].dropna()
mae_taupy = df_emsc["mae_vs_taupy"].dropna()
print(f"\n  PhaseNet-Fixed S-MAE vs EMSC reference:  mean={mae_emsc.mean():.3f}s  median={mae_emsc.median():.3f}s  N={len(mae_emsc)}")
print(f"  PhaseNet-Fixed S-MAE vs TauPy reference: mean={mae_taupy.mean():.3f}s  median={mae_taupy.median():.3f}s  N={len(mae_taupy)}")

# ── Step 3: Fine-tune PhaseNet-Fixed for S-phase (if N >= 200) ────────────────
SIGMA_S = 15   # wider sigma for S

class SDataset(Dataset):
    def __init__(self, hdf5_path, meta_df):
        rows = meta_df[meta_df["emsc_s_sample"].notna() &
                       meta_df["trace_p_arrival_sample"].notna()].copy()
        if "split" in rows.columns:
            rows = rows[rows["split"] == "train"].copy()
        print(f"    S-training windows: {len(rows)}")
        self.X = np.empty((len(rows), 3, N_TOTAL), dtype=np.float32)
        self.Y = np.zeros((len(rows), 3, N_TOTAL), dtype=np.float32)
        with h5py.File(str(hdf5_path), "r") as hf:
            for k, (_, row) in enumerate(rows.iterrows()):
                if row["trace_name"] not in hf["data"]: continue
                trace = hf["data"][row["trace_name"]][:, :N_TOTAL].astype(np.float32)
                for ch in range(3):
                    pk = float(np.abs(trace[ch]).max())
                    if pk > 1e-9: trace[ch] /= pk
                self.X[k] = trace
                s_samp = float(row["emsc_s_sample"])
                t = np.arange(N_TOTAL, dtype=np.float64)
                self.Y[k, 2] = np.exp(-0.5 * ((t - s_samp) / SIGMA_S) ** 2).astype(np.float32)
                # Keep P label
                p_samp = float(row["trace_p_arrival_sample"])
                self.Y[k, 0] = np.exp(-0.5 * ((t - p_samp) / 10.0) ** 2).astype(np.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return torch.tensor(self.X[i]), torch.tensor(self.Y[i])

# Only train if enough S-labeled training windows
all_meta_train = meta.copy()
all_meta_train["emsc_s_sample"] = all_meta_train.apply(match_emsc_s, axis=1)
s_train_count = all_meta_train[
    (all_meta_train["split"] == "train") &
    (all_meta_train["augmentation"] == "original") &
    (all_meta_train["emsc_s_sample"].notna())
].shape[0]
print(f"\n    S-labeled training windows available: {s_train_count}")

if s_train_count >= 200:
    print("\n[4] Fine-tuning PhaseNet-Fixed for S-phase ...")
    ds_s = SDataset(hdf5, all_meta_train)
    if len(ds_s) >= 50:
        loader = DataLoader(ds_s, batch_size=16, shuffle=True, num_workers=0)
        model_s = sbm.PhaseNet.from_pretrained("original")
        model_s.load_state_dict(torch.load(str(MDL_DIR/"phasenet_fixed.pt"), map_location=DEVICE, weights_only=True))
        model_s.to(DEVICE)

        # Only train S-relevant layers (keep P learned features)
        for name_, param in model_s.named_parameters():
            param.requires_grad = True   # full fine-tune at low LR

        opt = torch.optim.Adam(model_s.parameters(), lr=1e-5)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=15, eta_min=1e-7)

        def s_loss(pred, y):
            # Focus loss on S channel (channel 2), weight 3x
            s_pred = pred[:, 2, :].clamp(1e-7, 1-1e-7)
            s_tgt  = y[:, 2, :]
            p_pred = pred[:, 0, :].clamp(1e-7, 1-1e-7)
            p_tgt  = y[:, 0, :]
            bce_s  = -(s_tgt * torch.log(s_pred) + (1-s_tgt) * torch.log(1-s_pred))
            bce_p  = -(p_tgt * torch.log(p_pred) + (1-p_tgt) * torch.log(1-p_pred))
            return (3.0 * bce_s.mean() + bce_p.mean()) / 4.0

        best_loss = float("inf"); pat = 0; PATIENCE = 5
        log = []
        for epoch in range(1, 16):
            t0 = time.time()
            model_s.train()
            losses = []
            for X, y in loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                opt.zero_grad()
                pred = model_s(X)
                if isinstance(pred, (list, tuple)):
                    pred = torch.stack(list(pred), dim=1)
                loss = s_loss(pred, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model_s.parameters(), 1.0)
                opt.step()
                losses.append(loss.item())
            sch.step()
            tr_l = float(np.mean(losses))
            log.append({"epoch": epoch, "train_loss": round(tr_l, 6)})
            saved = ""
            if tr_l < best_loss:
                best_loss = tr_l; pat = 0
                torch.save(model_s.state_dict(), MDL_DIR/"phasenet_s_optimized.pt")
                saved = " ✓"
            else:
                pat += 1; saved = f" ({pat}/{PATIENCE})"
            print(f"  Ep {epoch:2d}  loss={tr_l:.4f}  ({time.time()-t0:.0f}s){saved}")
            if pat >= PATIENCE: print("  Early stop"); break

        # Re-evaluate with S-optimized model
        print("\n  Re-evaluating with phasenet_s_optimized.pt ...")
        model_s.eval()
        model_s.load_state_dict(torch.load(str(MDL_DIR/"phasenet_s_optimized.pt"), map_location=DEVICE, weights_only=True))

        rows_opt = []
        with h5py.File(hdf5, "r") as hf:
            for _, row in test_s_emsc.iterrows():
                name = row["trace_name"]
                if name not in hf["data"]: continue
                data = hf["data"][name][:]
                ref_s = float(row["emsc_s_sample"]) / SRATE
                pick  = pn_s_pick(model_s, data, thr=0.10)
                rows_opt.append({"mae_opt": abs(pick - ref_s) if pick else None})

        df_opt = pd.DataFrame(rows_opt)
        mae_opt = df_opt["mae_opt"].dropna()
        print(f"  S-optimized MAE: mean={mae_opt.mean():.3f}s  median={mae_opt.median():.3f}s")
        target_met = mae_opt.median() < 2.0
        s_opt_results = {
            "trained": True,
            "n_train": s_train_count,
            "mae_mean": round(float(mae_opt.mean()), 3),
            "mae_median": round(float(mae_opt.median()), 3),
            "target_lt_2s": target_met,
        }
    else:
        print("    Not enough windows for stable fine-tuning")
        s_opt_results = {"trained": False, "reason": "insufficient windows"}
else:
    print(f"\n[4] Skipping fine-tuning (only {s_train_count} S-labeled train windows, need ≥200)")
    s_opt_results = {"trained": False, "reason": f"only {s_train_count} S-train windows"}

# ── Figure ─────────────────────────────────────────────────────────────────────
print("\n[5] Producing figure ...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("S-Phase MAE: TauPy Reference vs EMSC Analyst Reference", fontsize=12)

clip = 15.0; bins = np.linspace(0, clip, 50)
ax1 = axes[0]
if len(mae_taupy) > 0:
    ax1.hist(np.clip(mae_taupy.values, 0, clip), bins=bins, alpha=0.7,
             color="#8892a4", label=f"vs TauPy (med={mae_taupy.median():.2f}s)")
ax1.hist(np.clip(mae_emsc.values, 0, clip), bins=bins, alpha=0.7,
         color="#00d4ff", label=f"vs EMSC  (med={mae_emsc.median():.2f}s)")
ax1.set_xlabel("S-MAE (s)"); ax1.set_ylabel("Windows")
ax1.set_title("PhaseNet-Fixed S-MAE: reference comparison")
ax1.legend(); ax1.set_xlim(0, clip)

ax2 = axes[1]
labels = ["TauPy ref.", "EMSC ref."]
medians = [mae_taupy.median() if len(mae_taupy) > 0 else 0, mae_emsc.median()]
colors  = ["#8892a4", "#00d4ff"]
if s_opt_results.get("trained"):
    labels.append("S-optimized")
    medians.append(s_opt_results["mae_median"])
    colors.append("#ff6b35")
bars = ax2.bar(labels, medians, color=colors, alpha=0.85)
ax2.axhline(2.0, color="red", ls="--", lw=1.2, label="Target 2.0s")
ax2.set_ylabel("Median S-MAE (s)"); ax2.set_title("Median S-MAE comparison")
ax2.legend(); ax2.set_ylim(0, max(medians) * 1.2 + 1)
for bar, v in zip(bars, medians):
    ax2.text(bar.get_x() + bar.get_width()/2, v + 0.05, f"{v:.2f}s", ha="center", fontsize=9)

plt.tight_layout()
fig_path = FIG / "s_phase_improvement.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"    Saved → {fig_path}")

# ── Save JSON ─────────────────────────────────────────────────────────────────
out = {
    "phase": "s_phase_improvement",
    "n_emsc_s_picks_total": len(s_picks),
    "n_test_with_emsc_s": len(test_s_emsc),
    "n_test_with_taupy_s": len(test_s_taupy),
    "phasenet_fixed_s_mae": {
        "vs_taupy": {
            "n": len(mae_taupy),
            "mean": round(float(mae_taupy.mean()), 3) if len(mae_taupy)>0 else None,
            "median": round(float(mae_taupy.median()), 3) if len(mae_taupy)>0 else None,
        },
        "vs_emsc": {
            "n": len(mae_emsc),
            "mean": round(float(mae_emsc.mean()), 3) if len(mae_emsc)>0 else None,
            "median": round(float(mae_emsc.median()), 3) if len(mae_emsc)>0 else None,
        },
    },
    "s_optimized_model": s_opt_results,
    "target_lt_2s_median": s_opt_results.get("target_lt_2s", mae_emsc.median() < 2.0),
}
with open(ART / "s_phase_improved.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"    Saved → {ART}/s_phase_improved.json")

print("\n" + "=" * 70)
print("OPTIMIZATION 2 COMPLETE — S-Phase Improvement")
print(f"  EMSC S picks matched: {len(test_s_emsc)}")
print(f"  S-MAE vs EMSC: median={mae_emsc.median():.3f}s  (vs TauPy: {mae_taupy.median():.3f}s)" if len(mae_emsc)>0 else "  No EMSC S matches")
print(f"  S-optimized model: {'trained ✓' if s_opt_results.get('trained') else 'skipped'}")
print("=" * 70)
