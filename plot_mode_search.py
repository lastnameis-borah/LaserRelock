"""
Plot Mode Search Results
========================
Plots the scan CSV from mode_search/ as a 2D heatmap of frequency offset
vs (current, piezo), with mode-hop-free regions and candidates overlaid.

Usage:
  python plot_mode_search.py                        # plots latest
  python plot_mode_search.py search_20260331_212645 # plots specific run
"""

import sys
import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

SEARCH_DIR = "/home/artiq/LaserRelock/mode_search"
FREQ_TARGET = 434.829040  # THz
FREQ_WINDOW = 0.000002    # THz
MODE_HOP_THRESHOLD = 0.001  # 1 GHz — same as mode_search.py


def find_search_files(name=None):
    """Find CSV + JSON for a search run."""
    if name is None:
        # Use latest
        latest = os.path.join(SEARCH_DIR, "latest.json")
        if not os.path.exists(latest):
            sys.exit("No latest.json found. Run mode_search.py first.")
        real = os.path.realpath(latest)
        base = os.path.splitext(os.path.basename(real))[0]
    else:
        base = name if not name.endswith(".csv") else name[:-4]

    csv_path = os.path.join(SEARCH_DIR, base + ".csv")
    json_path = os.path.join(SEARCH_DIR, base + ".json")

    if not os.path.exists(csv_path):
        sys.exit(f"CSV not found: {csv_path}")

    return csv_path, json_path, base


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else None
    csv_path, json_path, base = find_search_files(name)

    # Load data
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} points from {os.path.basename(csv_path)}")

    # Load JSON results if available
    results = None
    if os.path.exists(json_path):
        with open(json_path) as f:
            results = json.load(f)

    currents = df["current_mA"].unique()
    currents.sort()
    n_currents = len(currents)
    print(f"Currents: {currents[0]:.1f} – {currents[-1]:.1f} mA ({n_currents} steps)")

    # ── Figure: 4 panels ──
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.25)
    fig.suptitle(f"Mode Search: {base}", fontsize=14, fontweight="bold")

    # ── Panel 1: Freq offset vs piezo, colored by current ──
    ax1 = fig.add_subplot(gs[0, 0])
    cmap = plt.cm.viridis
    for i, curr in enumerate(currents):
        sub = df[df["current_mA"] == curr].sort_values("piezo_V")
        color = cmap(i / max(n_currents - 1, 1))
        ax1.plot(sub["piezo_V"], sub["freq_offset_MHz"], "-", color=color,
                 linewidth=0.6, alpha=0.7)

    ax1.axhspan(-FREQ_WINDOW * 1e6, FREQ_WINDOW * 1e6, color="green", alpha=0.15,
                label=f"Target +/-{FREQ_WINDOW*1e6:.0f} MHz")
    ax1.axhline(0, color="green", linestyle="--", alpha=0.5)
    ax1.set_xlabel("Piezo Voltage (V)")
    ax1.set_ylabel("Freq Offset from Target (MHz)")
    ax1.set_title("Frequency vs Piezo (each line = current)")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Color bar for current
    sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=plt.Normalize(currents[0], currents[-1]))
    sm.set_array([])
    cbar1 = fig.colorbar(sm, ax=ax1, pad=0.02)
    cbar1.set_label("Current (mA)")

    # ── Panel 2: 2D heatmap ──
    ax2 = fig.add_subplot(gs[0, 1])

    piezo_vals = np.sort(df["piezo_V"].unique())
    current_vals = np.sort(df["current_mA"].unique())

    # Build 2D grid
    freq_grid = np.full((len(current_vals), len(piezo_vals)), np.nan)
    c_idx = {c: i for i, c in enumerate(current_vals)}
    p_idx = {p: i for i, p in enumerate(piezo_vals)}
    for _, row in df.iterrows():
        ci = c_idx.get(row["current_mA"])
        pi = p_idx.get(row["piezo_V"])
        if ci is not None and pi is not None:
            freq_grid[ci, pi] = row["freq_offset_MHz"]

    # Symmetric color scale centered on 0
    vmax = min(np.nanmax(np.abs(freq_grid)), 500)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    im = ax2.pcolormesh(piezo_vals, current_vals, freq_grid,
                        cmap="RdBu_r", norm=norm, shading="nearest")
    cbar2 = fig.colorbar(im, ax=ax2, pad=0.02)
    cbar2.set_label("Freq Offset (MHz)")

    # Overlay in-window points
    in_win = df[df["in_window"] == 1]
    if len(in_win) > 0:
        ax2.scatter(in_win["piezo_V"], in_win["current_mA"],
                    c="lime", s=3, marker="s", alpha=0.6, label="In window")

    # Overlay best point
    if results and results.get("best"):
        best = results["best"]
        ax2.plot(best["piezo_V"], best["current_mA"], "r*", markersize=18,
                 markeredgecolor="k", markeredgewidth=0.8, label="Best", zorder=5)

    ax2.set_xlabel("Piezo Voltage (V)")
    ax2.set_ylabel("Current (mA)")
    ax2.set_title("Frequency Offset Heatmap")
    ax2.legend(loc="upper right", fontsize=8)

    # ── Panel 3: MHF regions ──
    ax3 = fig.add_subplot(gs[1, 0])

    for i, curr in enumerate(currents):
        sub = df[df["current_mA"] == curr].sort_values("piezo_V")
        if len(sub) < 2:
            continue

        piezos = sub["piezo_V"].values
        freqs = sub["frequency_THz"].values

        # Detect mode hops
        jumps = np.abs(np.diff(freqs))
        hop_indices = np.where(jumps > MODE_HOP_THRESHOLD)[0]

        # Build regions
        starts = np.concatenate([[0], hop_indices + 1])
        ends = np.concatenate([hop_indices, [len(piezos) - 1]])

        color = cmap(i / max(n_currents - 1, 1))
        for s, e in zip(starts, ends):
            if e - s < 2:
                continue
            width = piezos[e] - piezos[s]
            # Check if region contains target
            f_min = freqs[s:e+1].min()
            f_max = freqs[s:e+1].max()
            contains_target = (f_min <= FREQ_TARGET + FREQ_WINDOW and
                               f_max >= FREQ_TARGET - FREQ_WINDOW)
            alpha = 0.7 if contains_target else 0.15
            ax3.barh(curr, width, left=piezos[s],
                     height=(currents[1] - currents[0]) * 0.8 if len(currents) > 1 else 0.08,
                     color=color, alpha=alpha)

    # Overlay candidates
    if results and results.get("all_candidates"):
        for c in results["all_candidates"]:
            ax3.plot(c["piezo_V"], c["current_mA"], "r*", markersize=10, zorder=5)

    ax3.set_xlabel("Piezo Voltage (V)")
    ax3.set_ylabel("Current (mA)")
    ax3.set_title("Mode-Hop-Free Regions (bright = contains target)")
    ax3.grid(True, alpha=0.3)

    # ── Panel 4: Candidate summary ──
    ax4 = fig.add_subplot(gs[1, 1])

    if results and results.get("all_candidates"):
        cands = results["all_candidates"]
        c_currents = [c["current_mA"] for c in cands]
        c_piezos = [c["piezo_V"] for c in cands]
        c_widths = [c["mhf_width_V"] for c in cands]

        sc = ax4.scatter(c_piezos, c_currents, c=c_widths, s=80,
                         cmap="plasma", edgecolors="k", linewidths=0.5, zorder=3)
        cbar4 = fig.colorbar(sc, ax=ax4, pad=0.02)
        cbar4.set_label("MHF Width (V)")

        if results.get("best"):
            best = results["best"]
            ax4.plot(best["piezo_V"], best["current_mA"], "r*", markersize=22,
                     markeredgecolor="k", markeredgewidth=1, zorder=5,
                     label=f"Best: {best['current_mA']:.1f} mA, "
                           f"{best['piezo_V']:.1f} V, "
                           f"MHF={best['mhf_width_V']:.1f} V")
            ax4.legend(loc="upper left", fontsize=9)

        ax4.set_xlabel("Piezo Voltage (V)")
        ax4.set_ylabel("Current (mA)")
        ax4.set_title(f"Candidates ({len(cands)} found) — color = MHF width")
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, "No JSON results found", transform=ax4.transAxes,
                 ha="center", va="center", fontsize=14)
        ax4.set_title("Candidates")

    fig.subplots_adjust(left=0.06, right=0.94, top=0.93, bottom=0.06)
    plt.show()


if __name__ == "__main__":
    main()
