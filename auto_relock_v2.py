import time
import sys
import wlmData
import wlmConst
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
from datetime import datetime, timezone
from toptica.lasersdk.dlcpro.v2_0_3 import DLCpro, NetworkConnection

# ─── Configuration ───
DLL_PATH = "/usr/lib/libwlmData.so"
CHANNEL = 8

# Toptica DLC Pro settings
TOPTICA_IP = "172.29.13.247"
TOPTICA_PORT = 1998
LASER_NAME = "testbed689"

# Target Frequency Window (10 MHz)
FREQ_MIN = 434.829035
FREQ_MAX = 434.829045

# Piezo safety params
VOLTAGE_MIN = 20.0  # V
VOLTAGE_MAX = 50.0  # V
STEP_SIZE_V = 0.01  # 10mV
DELAY = 0.2         # s

# Relock stability params
STABLE_READINGS = 3         # Consecutive in-window readings before re-locking
SETTLE_AFTER_LOCK = 3.0     # Seconds to wait after engaging lock
MAX_RELOCK_ATTEMPTS = 5     # Max piezo-only attempts before mode recovery

# Mode recovery params (diode current scan)
CURRENT_SCAN_RANGE = 2.0    # mA total range to scan (±1.0 mA from operating point)
CURRENT_STEP = 0.1          # mA per step
CURRENT_SETTLE = 0.5        # Seconds to wait after each current step
MAX_MODE_RECOVERIES = 10     # Max mode recovery attempts before giving up


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
        print("Human intervention required. Stopping automatic adjustments.")
        input("Press Enter to exit...")
        sys.exit(1)


def scan_current_for_mode(dlc):
    """Scan diode current to find the correct laser mode.

    Scans outward from the current operating point in alternating
    directions (up, down, up further, down further, ...) to find a
    current where the frequency falls within the target window.

    Returns True if a good mode was found, False otherwise.
    """
    lock = dlc.laser1.dl.lock

    # Make sure lock is off during current scan
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

    # Calculate scan bounds
    half_range = CURRENT_SCAN_RANGE / 2.0
    current_min = operating_current - half_range
    current_max = operating_current + half_range

    # Scan outward from operating point: +0.1, -0.1, +0.2, -0.2, ...
    steps = int(half_range / CURRENT_STEP)
    for i in range(1, steps + 1):
        for direction in [+1, -1]:
            test_current = operating_current + direction * i * CURRENT_STEP

            # Safety check 
            if test_current < current_min or test_current > current_max:
                continue

            print(f"  Trying current: {test_current:.2f} mA ...", end=" ")
            dlc.laser1.dl.cc.current_set.set(test_current)
            time.sleep(CURRENT_SETTLE)

            # Check frequency
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

    # Nothing found — restore original current
    print(f"\n[MODE RECOVERY] No good mode found. Restoring current to {operating_current:.2f} mA")
    dlc.laser1.dl.cc.current_set.set(operating_current)
    time.sleep(CURRENT_SETTLE)
    return False


def try_piezo_relock(dlc, target_freq):
    """Adjust piezo voltage to bring frequency to target.

    Returns True if frequency is brought within window and lock holds.
    """
    lock = dlc.laser1.dl.lock
    freq = get_frequency()
    if freq is None:
        return False

    # Unlock if needed
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
        # Already in window, just try to lock
        direction = None

    # Adjust piezo if needed
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

    # Verify stability before re-locking
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

    # Re-lock
    print(f"  Stable! Re-locking...")
    lock.lock_enabled.set(True)
    time.sleep(SETTLE_AFTER_LOCK)

    # Verify lock held
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
freq_base = None
t_start = None

# ─── Set up the plot ───
fig, (ax_freq, ax_piezo) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
fig.canvas.manager.set_window_title("Auto Relock v2 — Ch%d + %s" % (CHANNEL, LASER_NAME))
fig.subplots_adjust(left=0.10, right=0.95, top=0.93, bottom=0.08, hspace=0.25)

# Frequency subplot (top)
freq_line, = ax_freq.plot([], [], 'b-o', markersize=3, linewidth=1.2)
ax_freq.set_ylabel("GHz")
ax_freq.set_title("Channel %d — Real-Time Frequency" % CHANNEL)
ax_freq.grid(True, alpha=0.3)
base_label = ax_freq.text(-0.01, 1.02, "", transform=ax_freq.transAxes, fontsize=10,
                          ha='left', va='bottom', fontweight='bold')
freq_band = None

# Piezo voltage subplot (bottom)
piezo_line, = ax_piezo.plot([], [], 'm-o', markersize=3, linewidth=1.2)
ax_piezo.set_xlabel("Time (s elapsed)")
ax_piezo.set_ylabel("Voltage (V)")
ax_piezo.set_title("%s — Piezo Voltage" % LASER_NAME)
ax_piezo.grid(True, alpha=0.3)


def main():
    global freq_base, t_start, freq_band

    relock_attempts = 0
    mode_recoveries = 0

    load_wavemeter()

    print(f"Connecting to {LASER_NAME} at {TOPTICA_IP}...")
    toptica_conn = NetworkConnection(TOPTICA_IP, TOPTICA_PORT)
    dlc = DLCpro(toptica_conn)

    try:
        dlc.open()
        print(f"Connected to {LASER_NAME} (SN: {dlc.serial_number.get()})")

        # Record initial operating current for reference
        operating_current = dlc.laser1.dl.cc.current_act.get()
        print(f"Operating current: {operating_current:.2f} mA")

        target_freq = (FREQ_MIN + FREQ_MAX) / 2.0
        print(f"Tracking Channel {CHANNEL} for window: {FREQ_MIN:.6f} to {FREQ_MAX:.6f} THz")
        print(f"Lockpoint target: {target_freq:.6f} THz")

        t_start = datetime.now(timezone.utc)

        def update(frame):
            nonlocal relock_attempts, mode_recoveries
            global freq_base, freq_band

            now = datetime.now(timezone.utc)
            elapsed = (now - t_start).total_seconds()

            # ── Read wavemeter frequency ──
            freq = get_frequency()

            if freq is not None:
                if freq_base is None:
                    freq_base = int(freq * 1000) / 1000.0
                    base_label.set_text("+ %.3f THz" % freq_base)
                    lo = (FREQ_MIN - freq_base) * 1000.0
                    hi = (FREQ_MAX - freq_base) * 1000.0
                    freq_band = ax_freq.axhspan(lo, hi, color='green', alpha=0.15, label='Target window')
                    ax_freq.legend(loc='upper right', fontsize=8)

                offset_ghz = (freq - freq_base) * 1000.0
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
                print(f"Freq: {freq:.6f} THz | Piezo: {current_voltage:.3f} V | Lock: {lock_state}")

                if freq > FREQ_MAX or freq < FREQ_MIN:
                    # ── Stage 1: Piezo-only relock ──
                    if relock_attempts < MAX_RELOCK_ATTEMPTS:
                        relock_attempts += 1
                        print(f"\n[RELOCK] Attempt {relock_attempts}/{MAX_RELOCK_ATTEMPTS}")

                        if try_piezo_relock(dlc, target_freq):
                            relock_attempts = 0
                            mode_recoveries = 0

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
                            # Found a good mode — try piezo relock
                            print("Mode found! Attempting piezo relock...")
                            if try_piezo_relock(dlc, target_freq):
                                print("✓ Successfully recovered from mode hop!")
                                relock_attempts = 0
                                mode_recoveries = 0
                            else:
                                print("Piezo relock failed after mode recovery.")
                                relock_attempts = 0  # Reset piezo counter, try again
                        else:
                            print("Mode recovery scan failed to find target mode.")
                            relock_attempts = 0  # Reset to try piezo again before next scan

                else:
                    # Frequency in window — all good
                    relock_attempts = 0
                    mode_recoveries = 0

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

            return freq_line, piezo_line

        ani = animation.FuncAnimation(fig, update, interval=int(DELAY * 1000), blit=False, cache_frame_data=False)
        plt.show()

    except KeyboardInterrupt:
        print("\nManually stopped monitoring.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        try:
            dlc.close()
        except:
            pass
        print("Exiting...")

if __name__ == "__main__":
    main()
