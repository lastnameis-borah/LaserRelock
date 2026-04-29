"""
Build a kNN consensus index from historical mode_search/search_*.csv data.

Why not a parametric MLP?
  At grid cells near mode-hop boundaries, the (I, V) -> freq map is
  multi-valued across scans (the boundary drifts). A smooth MLP averages
  modes together and produces meaningless predictions. A non-parametric
  kNN consensus preserves the multi-modality: queries near boundaries
  show high IQR (flagged unstable), queries inside an MHF region show
  tightly clustered neighbors (median is reliable).

Run: python -m ml_mode_finder.train
"""
import glob
import os

import numpy as np
import pandas as pd

SEARCH_DIR = "/home/artiq/LaserRelock/mode_search"
CHECKPOINT_DIR = "/home/artiq/LaserRelock/ml_mode_finder/checkpoints"
INDEX_PATH = os.path.join(CHECKPOINT_DIR, "scan_index.npz")

# Hardware-defined operating ranges (used for normalized distances)
I_MIN, I_MAX = 89.0, 92.0
V_MIN, V_MAX = 20.0, 60.0


def build(min_size_bytes=10_000):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(SEARCH_DIR, "search_*.csv")))
    files = [f for f in files if os.path.getsize(f) > min_size_bytes]
    if not files:
        raise SystemExit(f"No usable scans in {SEARCH_DIR}")

    parts = []
    scan_ids = []
    scan_names = []
    scan_timestamps = []
    for sid, f in enumerate(files):
        df = pd.read_csv(f)
        df = df[df["frequency_THz"] > 0]  # drop bad samples
        # Reject obvious wavemeter glitches (full operating freq range is ~10 GHz around 434.83)
        df = df[(df["frequency_THz"] > 434.5) & (df["frequency_THz"] < 435.0)]
        arr = df[["current_mA", "piezo_V", "frequency_THz"]].to_numpy(dtype=np.float64)
        parts.append(arr)
        scan_ids.append(np.full(len(arr), sid, dtype=np.int32))
        scan_names.append(os.path.basename(f))
        # Parse timestamp from filename: search_YYYYMMDD_HHMMSS.csv
        stem = os.path.basename(f).replace("search_", "").replace(".csv", "")
        scan_timestamps.append(stem)

    data = np.concatenate(parts, axis=0)
    sids = np.concatenate(scan_ids, axis=0)

    np.savez(
        INDEX_PATH,
        current_mA=data[:, 0],
        piezo_V=data[:, 1],
        frequency_THz=data[:, 2],
        scan_id=sids,
        scan_names=np.array(scan_names),
        scan_timestamps=np.array(scan_timestamps),
        I_MIN=I_MIN, I_MAX=I_MAX,
        V_MIN=V_MIN, V_MAX=V_MAX,
    )
    print(f"Built index: {len(data):,} points from {len(files)} scans")
    print(f"  Current range observed: {data[:,0].min():.2f} – {data[:,0].max():.2f} mA")
    print(f"  Piezo   range observed: {data[:,1].min():.2f} – {data[:,1].max():.2f} V")
    print(f"  Freq    range observed: {data[:,2].min():.6f} – {data[:,2].max():.6f} THz")
    print(f"Saved: {INDEX_PATH}")


if __name__ == "__main__":
    build()
