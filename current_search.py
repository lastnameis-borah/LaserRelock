"""
Current Search — 2D scan with outer piezo loop, inner current loop
====================================================================
Reverse of mode_search.py: outer loop steps through piezo voltages,
inner loop sweeps current at each piezo. Maps mode structure along
the current axis at every piezo.

Output:
  - current_search/search_YYYYMMDD_HHMMSS.csv
  - current_search/search_YYYYMMDD_HHMMSS.json
  - current_search/search_YYYYMMDD_HHMMSS.png

Usage: python current_search.py
"""

import time
import sys
import csv
import json
import os
import matplotlib.pyplot as plt
from datetime import datetime, timezone

import wlmData
import wlmConst
from toptica.lasersdk.dlcpro.v2_0_3 import DLCpro, NetworkConnection

# ─── Configuration ───
DLL_PATH = "/usr/lib/libwlmData.so"
CHANNEL = 3

TOPTICA_IP = "172.29.13.247"
TOPTICA_PORT = 1998
LASER_NAME = "testbed689"

# Target frequency
FREQ_TARGET = 434.829040      # THz
FREQ_WINDOW = 0.000003        # THz — ±3 MHz
FREQ_MIN = FREQ_TARGET - FREQ_WINDOW
FREQ_MAX = FREQ_TARGET + FREQ_WINDOW

# ── Search ranges ──
PIEZO_SCAN_MIN = 20.0         # V   (outer loop)
PIEZO_SCAN_MAX = 60.0         # V
PIEZO_SCAN_STEP = 0.1         # V

CURRENT_SCAN_MIN = 89.0       # mA  (inner loop)
CURRENT_SCAN_MAX = 92.0       # mA
CURRENT_SCAN_STEP = 0.1       # mA

# Timing
SETTLE_TIME = 0.2             # s — wait after setting current

# Mode-hop-free detection
MODE_HOP_THRESHOLD = 0.001    # 1 GHz
MIN_MHF_WIDTH_MA = 0.2        # mA — ignore MHF regions narrower than this

# Output directory
OUTPUT_DIR = "/home/artiq/LaserRelock/current_search"


def load_wavemeter():
    try:
        wlmData.LoadDLL(DLL_PATH)
    except:
        sys.exit(f"Error: Couldn't find DLL on path {DLL_PATH}.")
    if wlmData.dll.GetWLMCount(0) == 0:
        sys.exit("Error: No running wlmServer instance found.")
    ver = [wlmData.dll.GetWLMVersion(i) for i in range(4)]
    print(f"WLM Version: [{ver[0]}.{ver[1]}.{ver[2]}.{ver[3]}]")


def get_frequency():
    freq = wlmData.dll.GetFrequencyNum(CHANNEL, 0.0)
    if freq <= 0:
        return None
    return freq


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(OUTPUT_DIR, f"search_{timestamp}.csv")
    json_path = os.path.join(OUTPUT_DIR, f"search_{timestamp}.json")
    plot_path = os.path.join(OUTPUT_DIR, f"search_{timestamp}.png")

    load_wavemeter()

    print(f"Connecting to {LASER_NAME} at {TOPTICA_IP}...")
    toptica_conn = NetworkConnection(TOPTICA_IP, TOPTICA_PORT)
    dlc = DLCpro(toptica_conn)

    try:
        dlc.open()
        print(f"Connected to {LASER_NAME} (SN: {dlc.serial_number.get()})")

        lock = dlc.laser1.dl.lock
        if lock.lock_enabled.get():
            print("Disabling lock for search...")
            lock.lock_enabled.set(False)
            time.sleep(0.5)

        n_piezos = int((PIEZO_SCAN_MAX - PIEZO_SCAN_MIN)
                       / PIEZO_SCAN_STEP) + 1
        n_currents = int((CURRENT_SCAN_MAX - CURRENT_SCAN_MIN)
                         / CURRENT_SCAN_STEP) + 1
        est_time = n_piezos * n_currents * SETTLE_TIME / 60

        print(f"\n{'='*70}")
        print(f"REVERSE MODE SEARCH (outer=piezo, inner=current)")
        print(f"  Piezo:   {PIEZO_SCAN_MIN:.1f} – {PIEZO_SCAN_MAX:.1f} V "
              f"(step {PIEZO_SCAN_STEP} V)")
        print(f"  Current: {CURRENT_SCAN_MIN:.1f} – {CURRENT_SCAN_MAX:.1f} mA "
              f"(step {CURRENT_SCAN_STEP} mA)")
        print(f"  Target:  {FREQ_TARGET:.6f} THz (±{FREQ_WINDOW*1e6:.0f} MHz)")
        print(f"  Points:  {n_piezos} x {n_currents} = {n_piezos * n_currents}")
        print(f"  Est. time: ~{est_time:.0f} min")
        print(f"{'='*70}\n")

        # ── CSV log ──
        csvfile = open(csv_path, "w", newline="")
        writer = csv.writer(csvfile)
        writer.writerow(["timestamp_utc", "piezo_V", "current_mA",
                         "frequency_THz", "freq_offset_MHz", "in_window"])

        # ── Live plot ──
        plt.ion()
        fig, (ax_scan, ax_mhf) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        fig.canvas.manager.set_window_title(
            "Current Search — Ch%d + %s" % (CHANNEL, LASER_NAME))
        fig.subplots_adjust(left=0.10, right=0.95, top=0.93, bottom=0.08,
                            hspace=0.25)

        ax_scan.set_ylabel("Freq Offset (MHz)")
        ax_scan.set_title("Frequency vs Current (each line = different piezo)")
        ax_scan.axhspan(-2, 2, color='green', alpha=0.15, label='Target ±2 MHz')
        ax_scan.axhline(0, color='green', linestyle='--', alpha=0.5)
        ax_scan.set_xlim(CURRENT_SCAN_MIN, CURRENT_SCAN_MAX)
        ax_scan.set_ylim(-8000, 8000)
        ax_scan.grid(True, alpha=0.3)
        ax_scan.legend(loc='upper right', fontsize=8)

        ax_mhf.set_xlabel("Current (mA)")
        ax_mhf.set_ylabel("Piezo (V)")
        ax_mhf.set_title("Mode-Hop-Free Regions (red dots = target frequency)")
        ax_mhf.set_xlim(CURRENT_SCAN_MIN, CURRENT_SCAN_MAX)
        ax_mhf.grid(True, alpha=0.3)

        fig.canvas.draw()
        fig.canvas.flush_events()

        cmap = plt.cm.viridis
        piezo_idx = 0
        live_line, = ax_scan.plot([], [], '-', linewidth=1, alpha=0.8)

        all_candidates = []

        # ── Sweep ──
        piezo = PIEZO_SCAN_MIN
        while piezo <= PIEZO_SCAN_MAX + 1e-6:
            color = cmap(piezo_idx / max(n_piezos - 1, 1))
            piezo_idx += 1

            print(f"\n── Piezo = {piezo:.2f} V ({piezo_idx}/{n_piezos}) ──")
            dlc.laser1.dl.pc.voltage_set.set(piezo)
            time.sleep(SETTLE_TIME * 3)

            live_line.set_color(color)
            live_line.set_data([], [])

            scan_data = []
            live_currents = []
            live_offsets = []

            current = CURRENT_SCAN_MIN
            while current <= CURRENT_SCAN_MAX + 1e-6:
                dlc.laser1.dl.cc.current_set.set(current)
                time.sleep(SETTLE_TIME)

                freq = get_frequency()
                if freq is not None:
                    offset_mhz = (freq - FREQ_TARGET) * 1e6
                    scan_data.append((current, freq, offset_mhz))

                    now_str = datetime.now(timezone.utc).strftime(
                        "%Y-%m-%d %H:%M:%S.%f")[:-3]
                    in_win = 1 if FREQ_MIN <= freq <= FREQ_MAX else 0
                    writer.writerow([now_str, f"{piezo:.4f}", f"{current:.4f}",
                                     f"{freq:.6f}", f"{offset_mhz:.4f}", in_win])

                    live_currents.append(current)
                    live_offsets.append(offset_mhz)
                    live_line.set_data(live_currents, live_offsets)

                    if len(scan_data) % 5 == 0:
                        fig.canvas.draw_idle()
                        fig.canvas.flush_events()

                    marker = "  <<<" if FREQ_MIN <= freq <= FREQ_MAX else ""
                    if len(scan_data) % 1 == 0 or marker:
                        print(f"  Current {current:6.2f} mA -> {freq:.6f} THz "
                              f"({offset_mhz:+8.1f} MHz){marker}")

                current = round(current + CURRENT_SCAN_STEP, 4)

            # Permanent faded line
            if live_currents:
                ax_scan.plot(live_currents, live_offsets, '-', color=color,
                            linewidth=0.8, alpha=0.5,
                            label=f'{piezo:.1f} V' if piezo_idx <= 8 else '')
                if piezo_idx <= 8:
                    ax_scan.legend(loc='upper right', fontsize=7, ncol=2)

            fig.canvas.draw_idle()
            fig.canvas.flush_events()

            if not scan_data:
                print(f"  No signal, skipping.")
                piezo = round(piezo + PIEZO_SCAN_STEP, 4)
                continue

            # ── Find MHF regions along current axis ──
            regions = []
            region_start = 0
            for i in range(1, len(scan_data)):
                freq_jump = abs(scan_data[i][1] - scan_data[i-1][1])
                if freq_jump > MODE_HOP_THRESHOLD:
                    if i - region_start >= 2:
                        regions.append((region_start, i - 1))
                    region_start = i
            if len(scan_data) - region_start >= 2:
                regions.append((region_start, len(scan_data) - 1))

            print(f"  Found {len(regions)} mode-hop-free region(s)")

            for r_start, r_end in regions:
                c_start = scan_data[r_start][0]
                c_end = scan_data[r_end][0]
                f_min = min(d[1] for d in scan_data[r_start:r_end+1])
                f_max = max(d[1] for d in scan_data[r_start:r_end+1])
                width = c_end - c_start

                print(f"    Region: current {c_start:.2f}-{c_end:.2f} mA "
                      f"(width {width:.2f} mA), "
                      f"freq {f_min:.6f}-{f_max:.6f} THz")

                if width < MIN_MHF_WIDTH_MA:
                    print(f"    (skipped — narrower than {MIN_MHF_WIDTH_MA} mA)")
                    continue

                # Draw a bar for every MHF region
                ax_mhf.barh(piezo, width, left=c_start,
                           height=PIEZO_SCAN_STEP * 0.8,
                           color=color, alpha=0.5)

                # Does this region span the target frequency?
                if f_min <= FREQ_TARGET <= f_max:
                    region_currents = [d[0] for d in scan_data[r_start:r_end+1]]
                    region_freqs = [d[1] for d in scan_data[r_start:r_end+1]]
                    target_current = None
                    for k in range(len(region_freqs) - 1):
                        f0, f1 = region_freqs[k], region_freqs[k+1]
                        if (f0 - FREQ_TARGET) * (f1 - FREQ_TARGET) <= 0:
                            c0, c1 = region_currents[k], region_currents[k+1]
                            if f1 != f0:
                                target_current = c0 + (FREQ_TARGET - f0) * (c1 - c0) / (f1 - f0)
                            else:
                                target_current = (c0 + c1) / 2.0
                            break
                    if target_current is None:
                        target_current = region_currents[
                            min(range(len(region_freqs)),
                                key=lambda k: abs(region_freqs[k] - FREQ_TARGET))
                        ]

                    print(f"    >>> TARGET FOUND! Current={target_current:.3f} mA, "
                          f"MHF width={width:.2f} mA")

                    candidate = {
                        "piezo_V": round(piezo, 4),
                        "current_mA": round(target_current, 4),
                        "frequency_THz": round(FREQ_TARGET, 6),
                        "freq_offset_MHz": 0.0,
                        "mhf_width_mA": round(width, 4),
                        "mhf_current_min_mA": round(c_start, 4),
                        "mhf_current_max_mA": round(c_end, 4),
                    }
                    all_candidates.append(candidate)

                    ax_mhf.plot(target_current, piezo, 'ro',
                               markersize=7, zorder=5)

                fig.canvas.draw_idle()
                fig.canvas.flush_events()

            piezo = round(piezo + PIEZO_SCAN_STEP, 4)

        csvfile.close()

        # Pick best = widest MHF region
        best_result = None
        if all_candidates:
            best_result = max(all_candidates, key=lambda c: c["mhf_width_mA"])

        fig.canvas.draw()
        fig.canvas.flush_events()

        # Save plot
        fig.savefig(plot_path, dpi=150)
        print(f"\nPlot saved: {plot_path}")

        # Save JSON
        results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": {
                "freq_target_THz": FREQ_TARGET,
                "freq_window_THz": FREQ_WINDOW,
                "piezo_range_V": [PIEZO_SCAN_MIN, PIEZO_SCAN_MAX],
                "piezo_step_V": PIEZO_SCAN_STEP,
                "current_range_mA": [CURRENT_SCAN_MIN, CURRENT_SCAN_MAX],
                "current_step_mA": CURRENT_SCAN_STEP,
            },
            "best": best_result,
            "all_candidates": all_candidates,
        }

        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved: {json_path}")
        print(f"Scan CSV saved: {csv_path}")

        if best_result:
            print(f"\n{'='*70}")
            print(f"BEST OPERATING POINT:")
            print(f"  Piezo:     {best_result['piezo_V']:.2f} V")
            print(f"  Current:   {best_result['current_mA']:.3f} mA")
            print(f"  MHF width: {best_result['mhf_width_mA']:.2f} mA "
                  f"(current {best_result['mhf_current_min_mA']:.2f}"
                  f"-{best_result['mhf_current_max_mA']:.2f} mA)")
            print(f"\n  {len(all_candidates)} total candidate(s) found.")
            print(f"{'='*70}")
        else:
            print(f"\nNo target crossing found.")

        plt.ioff()
        plt.show()

    except KeyboardInterrupt:
        print("\nSearch interrupted.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            csvfile.close()
        except:
            pass
        try:
            dlc.close()
        except:
            pass
        print("Done.")


if __name__ == "__main__":
    main()
