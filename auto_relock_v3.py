import time
import sys
import csv
import wlmData
import wlmConst
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
from datetime import datetime, timezone
from toptica.lasersdk.dlcpro.v2_0_3 import DLCpro, NetworkConnection

# ─── Configuration ───
DLL_PATH = "/usr/lib/libwlmData.so"
CHANNEL = 3

# Toptica DLC Pro settings
TOPTICA_IP = "172.29.13.247"
TOPTICA_PORT = 1998
LASER_NAME = "testbed689"
OUTPUT_FILE = "/home/artiq/LaserRelock/relock_log/relock_log_v3_ch%d_%s.csv" % (CHANNEL, datetime.now().strftime("%Y%m%d_%H%M%S"))

# Target Frequency Window (10 MHz)
FREQ_MIN = 434.829037
FREQ_MAX = 434.829041

# Piezo safety params
VOLTAGE_MIN = 20.0  # V
VOLTAGE_MAX = 60.0  # V
STEP_SIZE_V = 0.01  # 10mV
DELAY = 0.2         # s

# Relock stability params
STABLE_READINGS = 5
SETTLE_AFTER_LOCK = 5.0
MAX_RELOCK_ATTEMPTS = 5

# Mode recovery params
CURRENT_SCAN_RANGE = 4.0
CURRENT_STEP = 0.1
CURRENT_SETTLE = 1.0
MAX_MODE_RECOVERIES = 10

# ─── Piezo re-centering params ───
PIEZO_DEADBAND = 0.1         # V — don't adjust if piezo is within this of target
RECENTER_CURRENT_STEP = 0.01  # mA — very small nudges (was 0.05, caused mode hops)
RECENTER_INTERVAL = 5        # re-center every N update cycles
RECENTER_MAX_CURRENT_OFFSET = 0.5  # mA — SAFETY: max total current change from initial
RECENTER_MAX_FAILURES = 3    # Auto-disable re-centering after this many failed nudges
# Direction: +1 means increasing current pushes piezo UP
#            -1 means increasing current pushes piezo DOWN
# If piezo drifts the wrong way, flip this sign.
RECENTER_DIRECTION = 1


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


def human_intervention(voltage):
    if voltage < VOLTAGE_MIN or voltage > VOLTAGE_MAX:
        print(f"\n[CRITICAL] Proposed voltage {voltage:.3f}V is out of safe bounds ({VOLTAGE_MIN}V - {VOLTAGE_MAX}V)!")
        print("Human intervention required. Stopping auto-relocking.")
        sys.exit(1)


def scan_current_for_mode(dlc):
    """Scan diode current to find the correct laser mode."""
    lock = dlc.laser1.dl.lock

    if lock.lock_enabled.get():
        lock.lock_enabled.set(False)
        time.sleep(DELAY)

    operating_current = dlc.laser1.dl.cc.current_act.get()
    print(f"\n{'='*60}")
    print(f"[MODE RECOVERY] Starting diode current scan")
    print(f"  Operating current: {operating_current:.2f} mA")
    print(f"  Scan range: ±{CURRENT_SCAN_RANGE/2:.1f} mA in {CURRENT_STEP:.1f} mA steps")
    print(f"  Target freq: {FREQ_MIN:.6f} – {FREQ_MAX:.6f} THz")
    print(f"{'='*60}")

    half_range = CURRENT_SCAN_RANGE / 2.0
    current_min = operating_current - half_range
    current_max = operating_current + half_range

    steps = int(half_range / CURRENT_STEP)
    for i in range(1, steps + 1):
        for direction in [+1, -1]:
            test_current = operating_current + direction * i * CURRENT_STEP

            if test_current < current_min or test_current > current_max:
                continue

            print(f"  Trying current: {test_current:.2f} mA ...", end=" ")
            dlc.laser1.dl.cc.current_set.set(test_current)
            time.sleep(CURRENT_SETTLE)

            freq = get_frequency()
            if freq is None:
                print("no signal")
                continue

            print(f"freq = {freq:.6f} THz", end=" ")

            if FREQ_MIN <= freq <= FREQ_MAX:
                print("✓ IN WINDOW!")
                print(f"\n[MODE RECOVERY] Found good mode at {test_current:.2f} mA")
                print(f"  Frequency: {freq:.6f} THz")
                return True
            else:
                diff_ghz = (freq - (FREQ_MIN + FREQ_MAX) / 2) * 1000
                print(f"({diff_ghz:+.3f} GHz off)")
                
    print(f"\n[MODE RECOVERY] No good mode found. Restoring current to {operating_current:.2f} mA")
    dlc.laser1.dl.cc.current_set.set(operating_current)
    time.sleep(CURRENT_SETTLE)
    return False


def try_piezo_relock(dlc, target_freq):
    """Adjust piezo voltage to bring frequency to target."""
    lock = dlc.laser1.dl.lock
    freq = get_frequency()
    if freq is None:
        return False

    if lock.lock_enabled.get():
        print("  Unlocking laser...")
        lock.lock_enabled.set(False)
        time.sleep(DELAY)

    current_voltage = dlc.laser1.dl.pc.voltage_act.get()

    if freq > FREQ_MAX:
        direction = "ABOVE"
    elif freq < FREQ_MIN:
        direction = "BELOW"
    else:
        direction = None

    if direction:
        print(f"  Frequency {direction} window. Adjusting piezo...")
        while (freq > target_freq if direction == "ABOVE" else freq < target_freq):
            step = -STEP_SIZE_V if direction == "ABOVE" else STEP_SIZE_V
            new_voltage = current_voltage + step
            human_intervention(new_voltage)
            dlc.laser1.dl.pc.voltage_set.set(new_voltage)
            current_voltage = new_voltage
            print(f"  -> Piezo: {current_voltage:.3f}V")
            time.sleep(DELAY)
            freq_new = get_frequency()
            if freq_new is not None:
                freq = freq_new
            print(f"  -> Freq: {freq:.6f} THz")

    print(f"  Verifying stability...")
    stable_count = 0
    for _ in range(STABLE_READINGS * 2):
        time.sleep(DELAY)
        f = get_frequency()
        if f is not None and FREQ_MIN <= f <= FREQ_MAX:
            stable_count += 1
            if stable_count >= STABLE_READINGS:
                break
        else:
            stable_count = 0

    if stable_count < STABLE_READINGS:
        print(f"  Could not stabilize frequency.")
        return False

    print(f"  Stable! Re-locking...")
    lock.lock_enabled.set(True)
    time.sleep(SETTLE_AFTER_LOCK)

    f_after = get_frequency()
    lock_state = lock.state_txt.get()
    if f_after is not None and FREQ_MIN <= f_after <= FREQ_MAX:
        print(f"  ✓ Lock held! Freq: {f_after:.6f} THz | Lock: {lock_state}")
        return True
    else:
        freq_str = f"{f_after:.6f} THz" if f_after else "N/A"
        print(f"  ✗ Lock caused freq jump! Freq: {freq_str} | Lock: {lock_state}")
        return False


# ─── Data buffers ───
times = deque()
freq_offsets = deque()
piezo_vals = deque()
current_vals = deque()
freq_base = None
t_start = None

# ─── Set up the plot ───
fig, (ax_freq, ax_piezo, ax_curr) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
fig.canvas.manager.set_window_title("Auto Relock v3 — Ch%d + %s" % (CHANNEL, LASER_NAME))
fig.subplots_adjust(left=0.10, right=0.95, top=0.95, bottom=0.07, hspace=0.3)

freq_line, = ax_freq.plot([], [], 'b-o', markersize=3, linewidth=1.2)
ax_freq.set_ylabel("MHz")
ax_freq.set_title("Channel %d — Real-Time Frequency" % CHANNEL)
ax_freq.grid(True, alpha=0.3)
base_label = ax_freq.text(-0.01, 1.02, "", transform=ax_freq.transAxes, fontsize=10,
                          ha='left', va='bottom', fontweight='bold')
freq_band = None

piezo_line, = ax_piezo.plot([], [], 'm-o', markersize=3, linewidth=1.2)
ax_piezo.set_ylabel("Voltage (V)")
ax_piezo.set_title("%s — Piezo Voltage" % LASER_NAME)
ax_piezo.grid(True, alpha=0.3)
piezo_target_line = None  # Will show initial piezo target

curr_line, = ax_curr.plot([], [], 'c-o', markersize=3, linewidth=1.2)
ax_curr.set_xlabel("Time (s elapsed)")
ax_curr.set_ylabel("Current (mA)")
ax_curr.set_title("%s — Laser Current" % LASER_NAME)
ax_curr.grid(True, alpha=0.3)


def main():
    global freq_base, t_start, freq_band, piezo_target_line

    relock_attempts = 0
    mode_recoveries = 0
    initial_piezo = None      # Recorded when laser first locks
    initial_current = None    # Recorded alongside initial piezo
    recenter_counter = 0      # Count updates to pace re-centering
    recenter_failures = 0     # Track consecutive failed nudges
    recenter_enabled = True   # Can be disabled if too many failures

    load_wavemeter()

    print(f"Connecting to {LASER_NAME} at {TOPTICA_IP}...")
    toptica_conn = NetworkConnection(TOPTICA_IP, TOPTICA_PORT)
    dlc = DLCpro(toptica_conn)

    try:
        dlc.open()
        print(f"Connected to {LASER_NAME} (SN: {dlc.serial_number.get()})")

        operating_current = dlc.laser1.dl.cc.current_act.get()
        print(f"Operating current: {operating_current:.2f} mA")

        target_freq = (FREQ_MIN + FREQ_MAX) / 2.0
        print(f"Tracking Channel {CHANNEL} for window: {FREQ_MIN:.6f} to {FREQ_MAX:.6f} THz")
        print(f"Lockpoint target: {target_freq:.6f} THz")

        t_start = datetime.now(timezone.utc)

        csvfile = open(OUTPUT_FILE, "w", newline="")
        writer = csv.writer(csvfile)
        writer.writerow(["timestamp_utc", "channel", "frequency_THz", "piezo_V", "current_mA", "temp_C"])
        log_count = 0
        print(f"Logging to: {OUTPUT_FILE}")

        def update(frame):
            nonlocal relock_attempts, mode_recoveries, log_count
            nonlocal initial_piezo, initial_current, recenter_counter
            nonlocal recenter_failures, recenter_enabled
            global freq_base, freq_band, piezo_target_line

            now = datetime.now(timezone.utc)
            elapsed = (now - t_start).total_seconds()

            # ── Read wavemeter frequency ──
            freq = get_frequency()

            if freq is not None:
                if freq_base is None:
                    freq_base = int(freq * 10000) / 10000.0
                    base_label.set_text("+ %.4f THz" % freq_base)
                    lo = (FREQ_MIN - freq_base) * 1e6
                    hi = (FREQ_MAX - freq_base) * 1e6
                    freq_band = ax_freq.axhspan(lo, hi, color='green', alpha=0.15, label='Target window')
                    ax_freq.legend(loc='upper right', fontsize=8)

                offset_ghz = (freq - freq_base) * 1e6
                times.append(elapsed)
                freq_offsets.append(offset_ghz)

            # ── Read Toptica piezo voltage ──
            try:
                piezo = dlc.laser1.dl.pc.voltage_act.get()
            except Exception as e:
                piezo = None
                print(f"Toptica read error: {e}")

            if piezo is not None:
                piezo_vals.append(piezo)

            # ── Relock logic ──
            lock = dlc.laser1.dl.lock

            if freq is not None:
                current_voltage = dlc.laser1.dl.pc.voltage_act.get()
                lock_state = lock.state_txt.get()
                print(f"Freq: {freq:.6f} THz | Piezo: {current_voltage:.3f} V | Lock: {lock_state}", end="")

                # Log to CSV
                now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                try:
                    current_mA = dlc.laser1.dl.cc.current_act.get()
                    temp_C = dlc.laser1.dl.tc.temp_act.get()
                except Exception:
                    current_mA = None
                    temp_C = None
                writer.writerow([now_str, CHANNEL, f"{freq:.6f}",
                                 f"{current_voltage:.4f}" if current_voltage else "",
                                 f"{current_mA:.4f}" if current_mA is not None else "",
                                 f"{temp_C:.4f}" if temp_C is not None else ""])
                csvfile.flush()
                log_count += 1

                if current_mA is not None:
                    current_vals.append(current_mA)

                if freq > FREQ_MAX or freq < FREQ_MIN:
                    print()  # newline after status
                    # ── Stage 1: Piezo-only relock ──
                    if relock_attempts < MAX_RELOCK_ATTEMPTS:
                        relock_attempts += 1
                        print(f"\n[RELOCK] Attempt {relock_attempts}/{MAX_RELOCK_ATTEMPTS}")

                        if try_piezo_relock(dlc, target_freq):
                            relock_attempts = 0
                            mode_recoveries = 0
                            # Record initial piezo after successful lock
                            initial_piezo = dlc.laser1.dl.pc.voltage_act.get()
                            initial_current = dlc.laser1.dl.cc.current_act.get()
                            recenter_failures = 0
                            recenter_enabled = True
                            print(f"  [RECENTER] Initial piezo target set: {initial_piezo:.3f} V")
                            # Draw target line on plot
                            if piezo_target_line is not None:
                                piezo_target_line.remove()
                            piezo_target_line = ax_piezo.axhline(y=initial_piezo, color='green',
                                                                  linestyle='--', linewidth=1, alpha=0.7,
                                                                  label=f'Piezo target ({initial_piezo:.2f}V)')
                            ax_piezo.legend(loc='upper right', fontsize=8)

                    # ── Stage 2: Mode recovery via current scan ──
                    else:
                        mode_recoveries += 1
                        print(f"\n[MODE RECOVERY] Piezo relock failed {MAX_RELOCK_ATTEMPTS} times.")
                        print(f"Mode recovery attempt {mode_recoveries}/{MAX_MODE_RECOVERIES}")

                        if mode_recoveries > MAX_MODE_RECOVERIES:
                            print(f"\n{'='*60}")
                            print(f"[CRITICAL] {MAX_MODE_RECOVERIES} mode recovery attempts failed.")
                            print(f"Human intervention required.")
                            print(f"{'='*60}")
                            lock.lock_enabled.set(False)
                            plt.close('all')
                            input("Fix the laser manually, then press Enter to exit...")
                            sys.exit(1)

                        if scan_current_for_mode(dlc):
                            print("Mode found! Attempting piezo relock...")
                            if try_piezo_relock(dlc, target_freq):
                                print("✓ Successfully recovered from mode hop!")
                                relock_attempts = 0
                                mode_recoveries = 0
                                initial_piezo = dlc.laser1.dl.pc.voltage_act.get()
                                initial_current = dlc.laser1.dl.cc.current_act.get()
                                recenter_failures = 0
                                recenter_enabled = True
                                print(f"  [RECENTER] Initial piezo target updated: {initial_piezo:.3f} V")
                            else:
                                print("Piezo relock failed after mode recovery.")
                                relock_attempts = 0
                        else:
                            print("Mode recovery scan failed to find target mode.")
                            relock_attempts = 0

                else:
                    # Frequency in window — all good
                    relock_attempts = 0
                    mode_recoveries = 0

                    # ── Record initial piezo on first successful lock ──
                    if initial_piezo is None and lock.lock_enabled.get():
                        initial_piezo = current_voltage
                        initial_current = current_mA
                        print(f" | [RECENTER] Initial piezo target: {initial_piezo:.3f} V", end="")
                        piezo_target_line = ax_piezo.axhline(y=initial_piezo, color='green',
                                                              linestyle='--', linewidth=1, alpha=0.7,
                                                              label=f'Piezo target ({initial_piezo:.2f}V)')
                        ax_piezo.legend(loc='upper right', fontsize=8)

                    # ── Piezo re-centering (while locked and in window) ──
                    if initial_piezo is not None and lock.lock_enabled.get() and recenter_enabled:
                        recenter_counter += 1
                        piezo_error = current_voltage - initial_piezo

                        if recenter_counter >= RECENTER_INTERVAL and abs(piezo_error) > PIEZO_DEADBAND:
                            recenter_counter = 0

                            # SAFETY: check current deviation from initial
                            current_offset = abs(current_mA - initial_current) if initial_current else 0
                            if current_offset >= RECENTER_MAX_CURRENT_OFFSET:
                                print(f" | [RECENTER] SAFETY LIMIT: current {current_mA:.3f} mA is {current_offset:.3f} mA from initial — skipping", end="")
                            else:
                                # Nudge current to push piezo back toward initial position
                                if piezo_error > 0:
                                    nudge = RECENTER_DIRECTION * RECENTER_CURRENT_STEP
                                else:
                                    nudge = -RECENTER_DIRECTION * RECENTER_CURRENT_STEP

                                new_current = current_mA + nudge
                                dlc.laser1.dl.cc.current_set.set(new_current)
                                print(f" | [RECENTER] Piezo {piezo_error:+.3f}V off → current {current_mA:.3f} → {new_current:.3f} mA", end="")

                                # SAFETY: verify frequency is still in window after nudge
                                time.sleep(DELAY)
                                freq_check = get_frequency()
                                if freq_check is not None and (freq_check < FREQ_MIN or freq_check > FREQ_MAX):
                                    # Revert the current change!
                                    dlc.laser1.dl.cc.current_set.set(current_mA)
                                    recenter_failures += 1
                                    print(f" | REVERTED (freq went to {freq_check:.6f})", end="")
                                    if recenter_failures >= RECENTER_MAX_FAILURES:
                                        recenter_enabled = False
                                        print(f" | RE-CENTERING DISABLED after {RECENTER_MAX_FAILURES} failures", end="")
                                else:
                                    recenter_failures = 0  # Reset on success

                    print()  # newline after status

            # ── Update plots ──
            if len(times) > 0:
                freq_line.set_data(list(times), list(freq_offsets))
                ax_freq.relim()
                ax_freq.autoscale_view()

            if len(piezo_vals) > 0:
                t_list = list(times)
                p_list = list(piezo_vals)
                min_len = min(len(t_list), len(p_list))
                piezo_line.set_data(t_list[-min_len:], p_list[-min_len:])
                ax_piezo.relim()
                ax_piezo.autoscale_view()

            if len(current_vals) > 0:
                t_list = list(times)
                c_list = list(current_vals)
                min_len = min(len(t_list), len(c_list))
                curr_line.set_data(t_list[-min_len:], c_list[-min_len:])
                ax_curr.relim()
                ax_curr.autoscale_view()

            return freq_line, piezo_line, curr_line

        ani = animation.FuncAnimation(fig, update, interval=int(DELAY * 1000), blit=False, cache_frame_data=False)
        plt.show()

    except KeyboardInterrupt:
        print("\nManually stopped monitoring.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        try:
            csvfile.close()
            print(f"\n{log_count} readings saved to '{OUTPUT_FILE}'.")
        except:
            pass
        try:
            dlc.close()
        except:
            pass
        print("Exiting...")

if __name__ == "__main__":
    main()
