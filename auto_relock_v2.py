import time
import sys
import csv
import socket
import struct
import numpy as np
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
OUTPUT_FILE = "relock_log_v2_ch%d_%s.csv" % (CHANNEL, datetime.now().strftime("%Y%m%d_%H%M%S"))

# Target Frequency Window (10 MHz)
FREQ_MIN = 434.829030
FREQ_MAX = 434.829050

# Piezo safety params
VOLTAGE_MIN = 20.0  # V
VOLTAGE_MAX = 60.0  # V
STEP_SIZE_V = 0.01  # 10mV
DELAY = 0.2         # s

# Relock stability params
STABLE_READINGS = 5         # Consecutive in-window readings before re-locking
SETTLE_AFTER_LOCK = 5.0     # Seconds to wait after engaging lock
MAX_RELOCK_ATTEMPTS = 5     # Max piezo-only attempts before mode recovery

# Mode recovery params
CURRENT_SCAN_RANGE = 4.0    # mA range to scan (±2.0 mA from operating point)
CURRENT_STEP = 0.1          # mA per step
CURRENT_SETTLE = 1.0        # Seconds to wait after each current step
MAX_MODE_RECOVERIES = 10    # Max mode recovery attempts before giving up

# Camera server (Thorlabs DCC1545M via camera_server.py)
CAMERA_IP = "172.29.13.140"
CAMERA_PORT = 5556
ROI_X1, ROI_Y1 = 395, 430
ROI_X2, ROI_Y2 = 595, 630


def recv_exact(sock, n):
    data = b''
    while len(data) < n:
        chunk = sock.recv(min(n - len(data), 65536))
        if not chunk:
            raise ConnectionError("Camera server disconnected")
        data += chunk
    return data


def get_roi_intensity(sock):
    """Request a frame and return ROI mean, or None on failure."""
    try:
        sock.sendall(b'FRAME')
        header = recv_exact(sock, 8)
        h, w = struct.unpack('>II', header)
        if h == 0 or w == 0:
            return None
        data = recv_exact(sock, h * w)
        frame = np.frombuffer(data, dtype=np.uint8).reshape(h, w)
        roi = frame[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]
        return float(np.mean(roi))
    except Exception as e:
        print(f"Camera read error: {e}")
        return None


def connect_camera():
    print(f"Connecting to camera server at {CAMERA_IP}:{CAMERA_PORT}...")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        sock.connect((CAMERA_IP, CAMERA_PORT))
        sock.settimeout(10.0)
        print("Camera connected.")
        return sock
    except Exception as e:
        print(f"Camera unavailable ({e}). Continuing without camera.")
        return None


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

    # Scan outward from operating point
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
current_vals = deque()
intensity_vals = deque()
freq_base = None
t_start = None

# ─── Set up the plot ───
fig, (ax_freq, ax_piezo, ax_curr, ax_cam) = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
fig.canvas.manager.set_window_title("Auto Relock v2 — Ch%d + %s" % (CHANNEL, LASER_NAME))
fig.subplots_adjust(left=0.10, right=0.95, top=0.96, bottom=0.06, hspace=0.35)

# Frequency subplot (top)
freq_line, = ax_freq.plot([], [], 'b-o', markersize=3, linewidth=1.2)
ax_freq.set_ylabel("MHz")
ax_freq.set_title("Channel %d — Real-Time Frequency" % CHANNEL)
ax_freq.grid(True, alpha=0.3)
base_label = ax_freq.text(-0.01, 1.02, "", transform=ax_freq.transAxes, fontsize=10,
                          ha='left', va='bottom', fontweight='bold')
freq_band = None

# Piezo voltage subplot
piezo_line, = ax_piezo.plot([], [], 'm-o', markersize=3, linewidth=1.2)
ax_piezo.set_ylabel("Voltage (V)")
ax_piezo.set_title("%s — Piezo Voltage" % LASER_NAME)
ax_piezo.grid(True, alpha=0.3)

# Laser current subplot
curr_line, = ax_curr.plot([], [], 'c-o', markersize=3, linewidth=1.2)
ax_curr.set_ylabel("Current (mA)")
ax_curr.set_title("%s — Laser Current" % LASER_NAME)
ax_curr.grid(True, alpha=0.3)

# Camera ROI intensity subplot (bottom)
cam_line, = ax_cam.plot([], [], 'y-o', markersize=3, linewidth=1.2)
ax_cam.set_xlabel("Time (s elapsed)")
ax_cam.set_ylabel("ROI Mean")
ax_cam.set_title("Camera ROI Intensity")
ax_cam.grid(True, alpha=0.3)


def main():
    global freq_base, t_start, freq_band

    relock_attempts = 0
    mode_recoveries = 0

    load_wavemeter()

    print(f"Connecting to {LASER_NAME} at {TOPTICA_IP}...")
    toptica_conn = NetworkConnection(TOPTICA_IP, TOPTICA_PORT)
    dlc = DLCpro(toptica_conn)

    cam_sock = connect_camera()
    if cam_sock is None:
        ax_cam.set_title("Camera ROI Intensity (disconnected)")

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

        csvfile = open(OUTPUT_FILE, "w", newline="")
        writer = csv.writer(csvfile)
        writer.writerow(["timestamp_utc", "channel", "frequency_THz", "piezo_V", "current_mA", "temp_C", "roi_mean"])
        log_count = 0
        print(f"Logging to: {OUTPUT_FILE}")

        def update(frame):
            nonlocal relock_attempts, mode_recoveries, log_count
            global freq_base, freq_band

            now = datetime.now(timezone.utc)
            elapsed = (now - t_start).total_seconds()

            # ── Read wavemeter frequency ──
            freq = get_frequency()

            if freq is not None:
                if freq_base is None:
                    freq_base = int(freq * 10000) / 10000.0
                    base_label.set_text("+ %.4f THz" % freq_base)
                    lo = (FREQ_MIN - freq_base) * 1e6  # MHz
                    hi = (FREQ_MAX - freq_base) * 1e6  # MHz
                    freq_band = ax_freq.axhspan(lo, hi, color='green', alpha=0.15, label='Target window')
                    ax_freq.legend(loc='upper right', fontsize=8)

                offset_ghz = (freq - freq_base) * 1e6  # MHz
                times.append(elapsed)
                freq_offsets.append(offset_ghz)

            # ── Read Toptica piezo voltage + current ──
            try:
                piezo = dlc.laser1.dl.pc.voltage_act.get()
                current_mA = dlc.laser1.dl.cc.current_act.get()
            except Exception as e:
                piezo = None
                current_mA = None
                print(f"Toptica read error: {e}")

            if piezo is not None:
                piezo_vals.append(piezo)
            if current_mA is not None:
                current_vals.append(current_mA)

            # ── Camera ROI intensity ──
            roi_mean = None
            if cam_sock is not None:
                roi_mean = get_roi_intensity(cam_sock)
            if roi_mean is not None:
                intensity_vals.append(roi_mean)

            # ── Relock logic ──
            lock = dlc.laser1.dl.lock

            if freq is not None:
                current_voltage = dlc.laser1.dl.pc.voltage_act.get()
                lock_state = lock.state_txt.get()
                print(f"Freq: {freq:.6f} THz | Piezo: {current_voltage:.3f} V | Lock: {lock_state}")

                # Log to CSV
                now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                try:
                    temp_C = dlc.laser1.dl.tc.temp_act.get()
                except Exception:
                    temp_C = None
                writer.writerow([now_str, CHANNEL, f"{freq:.6f}",
                                 f"{current_voltage:.4f}" if current_voltage else "",
                                 f"{current_mA:.4f}" if current_mA is not None else "",
                                 f"{temp_C:.4f}" if temp_C is not None else "",
                                 f"{roi_mean:.2f}" if roi_mean is not None else ""])
                csvfile.flush()
                log_count += 1

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

            if len(current_vals) > 0:
                t_list = list(times)
                c_list = list(current_vals)
                min_len = min(len(t_list), len(c_list))
                curr_line.set_data(t_list[-min_len:], c_list[-min_len:])
                ax_curr.relim()
                ax_curr.autoscale_view()

            if len(intensity_vals) > 0:
                t_list = list(times)
                i_list = list(intensity_vals)
                min_len = min(len(t_list), len(i_list))
                cam_line.set_data(t_list[-min_len:], i_list[-min_len:])
                ax_cam.relim()
                ax_cam.autoscale_view()

            return freq_line, piezo_line, curr_line, cam_line

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
        if cam_sock is not None:
            try:
                cam_sock.close()
            except Exception:
                pass
        print("Exiting...")

if __name__ == "__main__":
    main()
