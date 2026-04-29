"""Analyze stability of mode structure across historical mode_search scans.

Answers: are (current, piezo) → frequency maps stable over time, or do they
reshuffle? Determines whether a surrogate model trained on past scans is useful.
"""
import glob
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SEARCH_DIR = "/home/artiq/LaserRelock/mode_search"
FREQ_TARGET = 434.829040

csv_files = sorted(glob.glob(os.path.join(SEARCH_DIR, "search_*.csv")))
csv_files = [f for f in csv_files if os.path.getsize(f) > 10000]  # drop partial

print(f"Found {len(csv_files)} full scans\n")

scans = []
for f in csv_files:
    df = pd.read_csv(f)
    tag = os.path.basename(f).replace("search_", "").replace(".csv", "")
    scans.append((tag, df))

# ─── 1. Pairwise frequency agreement ───
print("=" * 70)
print("PAIRWISE FREQUENCY AGREEMENT (median |Δfreq| across (current, piezo) grid)")
print("=" * 70)
print(f"{'scan A':20s} {'scan B':20s} {'median Δf (GHz)':>16s} {'frac <0.1 GHz':>14s}")

# Pivot each scan to a 2D grid for comparison
grids = {}
for tag, df in scans:
    g = df.pivot_table(index="current_mA", columns="piezo_V",
                       values="frequency_THz", aggfunc="first")
    grids[tag] = g

tags = list(grids.keys())
for i in range(len(tags)):
    for j in range(i + 1, len(tags)):
        a, b = grids[tags[i]], grids[tags[j]]
        common = a.index.intersection(b.index)
        common_p = a.columns.intersection(b.columns)
        diff = (a.loc[common, common_p] - b.loc[common, common_p]).values * 1e3  # THz→GHz
        diff_flat = diff[~np.isnan(diff)]
        if len(diff_flat) == 0:
            continue
        med = np.median(np.abs(diff_flat))
        frac_close = (np.abs(diff_flat) < 0.1).mean()
        print(f"{tags[i]:20s} {tags[j]:20s} {med:16.3f} {frac_close:14.2%}")

# ─── 2. Best operating point drift ───
print("\n" + "=" * 70)
print("BEST OPERATING POINT ACROSS SCANS")
print("=" * 70)
print(f"{'scan':20s} {'current (mA)':>14s} {'piezo (V)':>12s} {'MHF width':>11s}")
for tag, _ in scans:
    jpath = os.path.join(SEARCH_DIR, f"search_{tag}.json")
    if os.path.exists(jpath):
        with open(jpath) as fp:
            j = json.load(fp)
        b = j.get("best")
        if b:
            print(f"{tag:20s} {b['current_mA']:14.2f} {b['piezo_V']:12.2f} "
                  f"{b['mhf_width_V']:11.2f}")

# ─── 3. In-window region overlap ───
print("\n" + "=" * 70)
print("IN-WINDOW REGION OVERLAP (Jaccard)")
print("=" * 70)
masks = {}
for tag, df in scans:
    m = df.pivot_table(index="current_mA", columns="piezo_V",
                       values="in_window", aggfunc="first").fillna(0).astype(bool)
    masks[tag] = m

print(f"{'scan A':20s} {'scan B':20s} {'|A∩B|/|A∪B|':>14s}")
for i in range(len(tags)):
    for j in range(i + 1, len(tags)):
        a, b = masks[tags[i]], masks[tags[j]]
        common_i = a.index.intersection(b.index)
        common_c = a.columns.intersection(b.columns)
        av = a.loc[common_i, common_c].values
        bv = b.loc[common_i, common_c].values
        inter = (av & bv).sum()
        union = (av | bv).sum()
        jac = inter / union if union else float("nan")
        print(f"{tags[i]:20s} {tags[j]:20s} {jac:14.2%}")

# ─── 4. Plot difference maps ───
n = len(scans)
ref_tag, ref_df = scans[0]
ref_grid = grids[ref_tag]

fig, axes = plt.subplots(2, max(n - 1, 1), figsize=(4 * max(n - 1, 1), 8),
                         squeeze=False)
for k, (tag, _) in enumerate(scans[1:]):
    diff = (grids[tag] - ref_grid) * 1e3  # GHz
    ax = axes[0, k]
    im = ax.imshow(diff.values, aspect="auto", origin="lower",
                   extent=[diff.columns.min(), diff.columns.max(),
                           diff.index.min(), diff.index.max()],
                   cmap="RdBu_r", vmin=-5, vmax=5)
    ax.set_title(f"Δfreq vs {ref_tag}\n{tag}", fontsize=9)
    ax.set_xlabel("piezo (V)")
    ax.set_ylabel("current (mA)")
    plt.colorbar(im, ax=ax, label="GHz")

    ax2 = axes[1, k]
    combo = masks[ref_tag].astype(int).values + 2 * masks[tag].astype(int).values
    ax2.imshow(combo, aspect="auto", origin="lower",
               extent=[masks[tag].columns.min(), masks[tag].columns.max(),
                       masks[tag].index.min(), masks[tag].index.max()],
               cmap="viridis", vmin=0, vmax=3)
    ax2.set_title(f"in-window: ref={ref_tag[-6:]}, this={tag[-6:]}\n"
                  "0=neither 1=refonly 2=thisonly 3=both", fontsize=8)
    ax2.set_xlabel("piezo (V)")
    ax2.set_ylabel("current (mA)")

plt.tight_layout()
out = "/home/artiq/LaserRelock/mode_stability_analysis.png"
plt.savefig(out, dpi=100)
print(f"\nSaved diff plot to {out}")
