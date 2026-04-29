"""
ML-driven lock: jump to predicted (current, piezo), then fine local piezo
sweep to land on target frequency, then engage lock.

Pipeline:
  1. Load surrogate, pick lowest-current / lowest-piezo MHF region.
  2. Set current and piezo on the laser.
  3. Read freq. If in window → engage lock.
  4. Else: probe step + walk piezo in the direction toward target,
           bounded by MHF region. Stop when freq stays in window.
  5. Engage lock; verify it holds.

Run: python -m ml_mode_finder.lock
"""
import csv
import os
import sys
import time
from collections import deque
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import wlmData
from toptica.lasersdk.dlcpro.v2_0_3 import DLCpro, NetworkConnection

from ml_mode_finder.find_mode import (
    find, FREQ_TARGET, FREQ_WINDOW,
)

# Hardware
DLL_PATH = "/usr/lib/libwlmData.so"
CHANNEL = 3
TOPTICA_IP = "172.29.13.247"
TOPTICA_PORT = 1998
LASER_NAME = "testbed689"

# Lock window
FREQ_MIN = FREQ_TARGET - FREQ_WINDOW
FREQ_MAX = FREQ_TARGET + FREQ_WINDOW

# Timing (matches auto_relock_v2)
SETTLE_AFTER_SET = 0.3      # s — after writing current/piezo
STEP_DELAY = 0.2            # s — after each piezo step before reading freq
STABLE_DELAY = 0.2          # s — between stability-check reads
LOCK_SETTLE = 2.0           # s — after engaging lock

# Local sweep (v2-style)
STEP_SIZE_V = 0.1           # V — fixed step size
STABLE_READINGS = 5         # consecutive in-window reads before engage
# Sweep is bounded by [mhf_piezo_min_V, mhf_piezo_max_V] only — no fixed cap.

# Continuous monitor / auto-relock
MONITOR_INTERVAL_S = 0.5            # plot/log/check cadence
MONITOR_RELOCK_ATTEMPTS = 3         # retries at the same candidate before stepping to next
PIEZO_DEADBAND_V = 0.1              # V — re-centering ignored within this of target
RECENTER_INTERVAL = 5               # ticks between recenter nudges
RECENTER_CURRENT_STEP = 0.01        # mA — small nudges to push piezo back
RECENTER_MAX_CURRENT_OFFSET = 0.5   # mA — total deviation cap from initial current
RECENTER_MAX_FAILURES = 3           # auto-disable after this many freq-driving nudges
RECENTER_DIRECTION = -1             # +1 or -1: sign of current->piezo coupling
PLOT_HISTORY_S = 600                # show last N seconds of data on plots

# Output log
LOG_DIR = "/home/artiq/LaserRelock/relock_log"
LOG_PATH = os.path.join(
    LOG_DIR,
    f"ml_lock_ch{CHANNEL}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
)


def get_frequency():
    f = wlmData.dll.GetFrequencyNum(CHANNEL, 0.0)
    return None if f <= 0 else f


def read_freq_avg(n=3, delay=0.05):
    """Read n freqs and return mean of valid samples, or None."""
    vals = []
    for _ in range(n):
        f = get_frequency()
        if f is not None:
            vals.append(f)
        time.sleep(delay)
    if not vals:
        return None
    return sum(vals) / len(vals)


def local_piezo_sweep(dlc, mhf_min, mhf_max):
    """
    auto_relock_v2-style: read freq, walk piezo in fixed direction
    in STEP_SIZE_V steps until freq has crossed the target. Then
    stop and verify STABLE_READINGS in-window samples without moving.

    Direction assumes positive df/dV (decreasing piezo lowers freq),
    matching v2's hard-coded behavior.

    Bounded by [mhf_min, mhf_max] so we don't sweep into a neighboring mode.
    Returns (success, final_piezo).
    """
    f = get_frequency()
    piezo = dlc.laser1.dl.pc.voltage_act.get()
    if f is None:
        print("  [SWEEP] No freq reading.")
        return False, piezo

    err_mhz = (f - FREQ_TARGET) * 1e6
    print(f"  [SWEEP] Start V={piezo:.3f}, f={f:.6f} ({err_mhz:+.2f} MHz)")

    if FREQ_MIN <= f <= FREQ_MAX:
        direction = None
    elif f > FREQ_MAX:
        direction = "ABOVE"
    else:
        direction = "BELOW"

    if direction is not None:
        step = -STEP_SIZE_V if direction == "ABOVE" else STEP_SIZE_V
        print(f"  [SWEEP] Freq {direction}; step={step*1000:+.1f} mV")
        while True:
            # Stop once freq has crossed the target midpoint
            if (f <= FREQ_TARGET if direction == "ABOVE"
                    else f >= FREQ_TARGET):
                break
            new_piezo = piezo + step
            if not (mhf_min <= new_piezo <= mhf_max):
                print(f"  [SWEEP] Hit MHF bound at V={new_piezo:.3f} "
                      f"(region {mhf_min:.2f}-{mhf_max:.2f}).")
                return False, piezo
            dlc.laser1.dl.pc.voltage_set.set(new_piezo)
            piezo = new_piezo
            time.sleep(STEP_DELAY)
            f_new = get_frequency()
            if f_new is not None:
                f = f_new
            print(f"    V={piezo:.3f}V -> f={f:.6f} "
                  f"({(f-FREQ_TARGET)*1e6:+.2f} MHz)")

    # Verify stability without moving
    print(f"  [SWEEP] Verifying stability ({STABLE_READINGS} reads)...")
    stable = 0
    for _ in range(STABLE_READINGS * 2):
        time.sleep(STABLE_DELAY)
        fv = get_frequency()
        if fv is not None and FREQ_MIN <= fv <= FREQ_MAX:
            stable += 1
            if stable >= STABLE_READINGS:
                print(f"  [SWEEP] Stable in window. V={piezo:.3f}, f={fv:.6f}")
                return True, piezo
        else:
            stable = 0

    print(f"  [SWEEP] Could not stabilize.")
    return False, piezo


def attempt_lock(dlc, current_mA, piezo_V, mhf_min, mhf_max):
    """Set (I, V), v2-style local sweep, engage lock, verify."""
    lock = dlc.laser1.dl.lock
    if lock.lock_enabled.get():
        print("  Disabling existing lock.")
        lock.lock_enabled.set(False)
        time.sleep(0.3)

    print(f"  Setting I={current_mA:.2f} mA, V={piezo_V:.2f} V")
    dlc.laser1.dl.cc.current_set.set(float(current_mA))
    dlc.laser1.dl.pc.voltage_set.set(float(piezo_V))
    time.sleep(SETTLE_AFTER_SET)

    success, final_piezo = local_piezo_sweep(dlc, mhf_min, mhf_max)
    if not success:
        return False

    print(f"  Engaging lock at V={final_piezo:.3f}...")
    lock.lock_enabled.set(True)
    time.sleep(LOCK_SETTLE)

    f = get_frequency()
    state = lock.state_txt.get()
    if f is not None and FREQ_MIN <= f <= FREQ_MAX:
        print(f"  ✓ Locked. f={f:.6f} THz | lock_state={state}")
        return True
    print(f"  ✗ Lock failed. f={f} | lock_state={state}")
    lock.lock_enabled.set(False)
    return False


def monitor(dlc, csvfile, writer, best, cands, t_lock_acquired):
    """
    Live monitor + auto-relock loop. Plots freq/piezo/current vs time,
    logs to CSV, and re-runs attempt_lock if the laser drifts out of
    window. Falls through to alt ML candidates if the primary keeps
    failing. Includes v3-style piezo recentering via small current
    nudges to extend lock duration.

    Run until the user closes the plot window or hits Ctrl-C.
    """
    times = deque()
    freq_offsets_mhz = deque()
    piezo_vals = deque()
    current_vals = deque()

    initial_piezo = dlc.laser1.dl.pc.voltage_act.get()
    initial_current = dlc.laser1.dl.cc.current_act.get()
    print(f"\n[MONITOR] Initial piezo target: {initial_piezo:.3f} V "
          f"(initial current: {initial_current:.3f} mA)")

    state = {
        "best": dict(best),
        "cands": cands,
        "alt_idx": 0,
        "relock_attempts": 0,
        "recenter_counter": 0,
        "recenter_failures": 0,
        "recenter_enabled": True,
        "initial_piezo": initial_piezo,
        "initial_current": initial_current,
        "tick": 0,
    }

    fig, (ax_f, ax_p, ax_c) = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    fig.canvas.manager.set_window_title(
        f"ML Lock Monitor — Ch{CHANNEL} {LASER_NAME}"
    )
    fig.subplots_adjust(left=0.09, right=0.96, top=0.94, bottom=0.07, hspace=0.3)

    f_line, = ax_f.plot([], [], 'b-', linewidth=1.0)
    ax_f.set_ylabel("Freq offset (MHz)")
    ax_f.set_title(f"Channel {CHANNEL} — freq offset from {FREQ_TARGET:.6f} THz")
    ax_f.axhspan(-FREQ_WINDOW * 1e6, FREQ_WINDOW * 1e6,
                 color='green', alpha=0.15, label='Target window')
    ax_f.axhline(0, color='green', linestyle='--', alpha=0.5)
    ax_f.grid(True, alpha=0.3)
    ax_f.legend(loc='upper right', fontsize=8)

    p_line, = ax_p.plot([], [], 'm-', linewidth=1.0)
    ax_p.axhline(initial_piezo, color='green', linestyle='--', alpha=0.7,
                 label=f'Recenter target ({initial_piezo:.2f} V)')
    ax_p.set_ylabel("Piezo (V)")
    ax_p.set_title("Piezo voltage")
    ax_p.grid(True, alpha=0.3)
    ax_p.legend(loc='upper right', fontsize=8)

    c_line, = ax_c.plot([], [], 'c-', linewidth=1.0)
    ax_c.set_ylabel("Current (mA)")
    ax_c.set_xlabel("Time since lock (s)")
    ax_c.set_title("Diode current")
    ax_c.grid(True, alpha=0.3)

    def trim_history():
        # Drop oldest samples beyond PLOT_HISTORY_S
        if not times:
            return
        cutoff = times[-1] - PLOT_HISTORY_S
        while times and times[0] < cutoff:
            times.popleft()
            freq_offsets_mhz.popleft()
            if piezo_vals:
                piezo_vals.popleft()
            if current_vals:
                current_vals.popleft()

    def do_relock():
        """Try to relock; advance through candidates as needed."""
        s = state
        s["relock_attempts"] += 1
        cur_best = s["best"]
        print(f"\n[RELOCK] attempt {s['relock_attempts']}/"
              f"{MONITOR_RELOCK_ATTEMPTS} on (I={cur_best['current_mA']:.2f} mA, "
              f"V={cur_best['piezo_V']:.2f} V)")
        ok = attempt_lock(dlc,
                          cur_best["current_mA"], cur_best["piezo_V"],
                          cur_best["mhf_piezo_min_V"],
                          cur_best["mhf_piezo_max_V"])
        csvfile.flush()
        if ok:
            s["relock_attempts"] = 0
            s["initial_piezo"] = dlc.laser1.dl.pc.voltage_act.get()
            s["initial_current"] = dlc.laser1.dl.cc.current_act.get()
            s["recenter_failures"] = 0
            s["recenter_enabled"] = True
            print(f"[RELOCK] ✓ Recovered. New piezo target: "
                  f"{s['initial_piezo']:.3f} V")
            return True

        if s["relock_attempts"] >= MONITOR_RELOCK_ATTEMPTS:
            # Step to next ML candidate
            s["alt_idx"] += 1
            if s["alt_idx"] < len(s["cands"]):
                s["best"] = s["cands"][s["alt_idx"]]
                s["relock_attempts"] = 0
                print(f"[RELOCK] Stepping to next ML candidate "
                      f"(I={s['best']['current_mA']:.2f} mA, "
                      f"V={s['best']['piezo_V']:.2f} V)")
            else:
                print("[RELOCK] No more candidates. "
                      "Re-train surrogate or run mode_search.")
        return False

    def update(_frame):
        s = state
        s["tick"] += 1
        elapsed = time.monotonic() - t_lock_acquired
        f = get_frequency()
        try:
            piezo = dlc.laser1.dl.pc.voltage_act.get()
            current_ma = dlc.laser1.dl.cc.current_act.get()
            temp_c = dlc.laser1.dl.tc.temp_act.get()
            lock_state = dlc.laser1.dl.lock.state_txt.get()
            lock_on = dlc.laser1.dl.lock.lock_enabled.get()
        except Exception as e:
            print(f"[MONITOR] Toptica read error: {e}")
            return f_line, p_line, c_line

        if f is None:
            return f_line, p_line, c_line

        in_win = FREQ_MIN <= f <= FREQ_MAX
        offset_mhz = (f - FREQ_TARGET) * 1e6

        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        writer.writerow([ts, CHANNEL, f"{f:.6f}",
                         f"{piezo:.4f}", f"{current_ma:.4f}",
                         f"{temp_c:.4f}" if temp_c is not None else "",
                         lock_state, int(in_win)])
        if s["tick"] % 20 == 0:
            csvfile.flush()

        times.append(elapsed)
        freq_offsets_mhz.append(offset_mhz)
        piezo_vals.append(piezo)
        current_vals.append(current_ma)
        trim_history()

        print(f"t={elapsed:7.1f}s | f={f:.6f} ({offset_mhz:+7.2f} MHz) | "
              f"V={piezo:.3f} | I={current_ma:.3f} mA | T={temp_c:.3f} C | "
              f"lock={lock_state}", end="")

        # ── Out of window or lock disengaged: relock ──
        if not in_win or not lock_on:
            print(" | OUT — relock")
            do_relock()
        else:
            # Reset relock counter on healthy in-window read
            if s["relock_attempts"] > 0:
                s["relock_attempts"] = 0

            # ── Piezo recentering (v3 style) ──
            if s["recenter_enabled"]:
                s["recenter_counter"] += 1
                err_v = piezo - s["initial_piezo"]
                if (s["recenter_counter"] >= RECENTER_INTERVAL
                        and abs(err_v) > PIEZO_DEADBAND_V):
                    s["recenter_counter"] = 0
                    cur_offset = abs(current_ma - s["initial_current"])
                    if cur_offset >= RECENTER_MAX_CURRENT_OFFSET:
                        print(f" | recenter SKIP (cur offset {cur_offset:.3f})",
                              end="")
                    else:
                        nudge = (RECENTER_DIRECTION * RECENTER_CURRENT_STEP
                                 * (1 if err_v > 0 else -1))
                        new_cur = current_ma + nudge
                        dlc.laser1.dl.cc.current_set.set(new_cur)
                        time.sleep(STEP_DELAY)
                        f_chk = get_frequency()
                        if f_chk is not None and not (FREQ_MIN <= f_chk <= FREQ_MAX):
                            dlc.laser1.dl.cc.current_set.set(current_ma)
                            s["recenter_failures"] += 1
                            print(f" | recenter REVERTED "
                                  f"(f→{f_chk:.6f})", end="")
                            if s["recenter_failures"] >= RECENTER_MAX_FAILURES:
                                s["recenter_enabled"] = False
                                print(" | RECENTERING DISABLED", end="")
                        else:
                            s["recenter_failures"] = 0
                            print(f" | recenter {err_v:+.3f}V→cur {new_cur:.3f}",
                                  end="")
            print()

        # ── Plot updates ──
        if times:
            t_list = list(times)
            f_line.set_data(t_list, list(freq_offsets_mhz))
            ax_f.relim(); ax_f.autoscale_view()
            p_line.set_data(t_list, list(piezo_vals))
            ax_p.relim(); ax_p.autoscale_view()
            c_line.set_data(t_list, list(current_vals))
            ax_c.relim(); ax_c.autoscale_view()

        return f_line, p_line, c_line

    ani = animation.FuncAnimation(
        fig, update, interval=int(MONITOR_INTERVAL_S * 1000),
        blit=False, cache_frame_data=False,
    )
    print(f"\n[MONITOR] Live monitor started "
          f"(interval={MONITOR_INTERVAL_S}s). Close plot or Ctrl-C to stop.\n")
    plt.show()
    # Keep `ani` referenced so it isn't garbage-collected
    return ani


def main():
    t_start = time.monotonic()

    # Wavemeter
    try:
        wlmData.LoadDLL(DLL_PATH)
    except Exception:
        sys.exit(f"Couldn't load wavemeter DLL at {DLL_PATH}")
    if wlmData.dll.GetWLMCount(0) == 0:
        sys.exit("No wlmServer instance found")

    # Pick (I, V) via ML
    print("Loading surrogate and selecting operating point...")
    best, cands = find()
    if best is None:
        sys.exit("ML model found no target-crossing MHF region")
    print(f"Selected (lowest current → lowest piezo):")
    print(f"  I = {best['current_mA']:.2f} mA")
    print(f"  V = {best['piezo_V']:.2f} V "
          f"(MHF {best['mhf_piezo_min_V']:.2f}-{best['mhf_piezo_max_V']:.2f}, "
          f"width {best['mhf_width_V']:.1f} V)")
    print(f"  Total candidates: {len(cands)}")

    # Hardware
    print(f"\nConnecting to {LASER_NAME} at {TOPTICA_IP}...")
    dlc = DLCpro(NetworkConnection(TOPTICA_IP, TOPTICA_PORT))

    os.makedirs(LOG_DIR, exist_ok=True)
    csvfile = open(LOG_PATH, "w", newline="")
    writer = csv.writer(csvfile)
    writer.writerow(["timestamp_utc", "channel", "frequency_THz", "piezo_V",
                     "current_mA", "temp_C", "lock_state", "in_window"])
    print(f"Logging to: {LOG_PATH}")

    try:
        dlc.open()
        print(f"Connected (SN: {dlc.serial_number.get()})")

        ok = attempt_lock(dlc,
                          best["current_mA"], best["piezo_V"],
                          best["mhf_piezo_min_V"], best["mhf_piezo_max_V"])
        csvfile.flush()

        locked_best = None
        if ok:
            locked_best = best
        else:
            print(f"\nFirst attempt failed. Trying remaining "
                  f"{len(cands) - 1} candidates...")
            for ai, alt in enumerate(cands[1:], start=1):
                print(f"\nAlt: I={alt['current_mA']:.2f} mA, "
                      f"V={alt['piezo_V']:.2f} V")
                if attempt_lock(dlc,
                                alt["current_mA"], alt["piezo_V"],
                                alt["mhf_piezo_min_V"],
                                alt["mhf_piezo_max_V"]):
                    locked_best = alt
                    cands = [alt] + cands[:ai] + cands[ai + 1:]
                    break
                csvfile.flush()

        if locked_best is None:
            elapsed = time.monotonic() - t_start
            print(f"\nFATAL: All candidates failed after {elapsed:.2f} s. "
                  f"Re-train surrogate or run a fresh mode_search.")
            return

        elapsed = time.monotonic() - t_start
        t_lock_acquired = time.monotonic()
        print(f"\nLock acquired. Time to lock: {elapsed:.2f} s")
        print(f"Entering continuous monitor (auto-relocks on unlock).")
        monitor(dlc, csvfile, writer, locked_best, cands, t_lock_acquired)

    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error: {e}")
    finally:
        try:
            csvfile.close()
        except Exception:
            pass
        try:
            dlc.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
