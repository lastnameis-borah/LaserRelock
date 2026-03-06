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

# Target Frequency Window (4 MHz)
FREQ_MIN = 434.829039
FREQ_MAX = 434.829041

# Safety Params
VOLTAGE_MIN = 25.0  # V
VOLTAGE_MAX = 35.0  # V
STEP_SIZE_V = 0.01  # 10mV 
DELAY = 0.2  # s

# Relock stability params
STABLE_READINGS = 3         # Consecutive in-window readings before re-locking
SETTLE_AFTER_LOCK = 3.0     # Seconds to wait after engaging lock
MAX_RELOCK_ATTEMPTS = 5     # Max attempts before backing off

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

# ─── Data buffers ───
times = deque()
freq_offsets = deque()
piezo_vals = deque()
freq_base = None
t_start = None

# ─── Set up the plot ───
# Layout: 2 rows x 1 col — frequency on top, current on bottom
fig, (ax_freq, ax_curr) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
fig.canvas.manager.set_window_title("Auto Relock — Ch%d + %s" % (CHANNEL, LASER_NAME))
fig.subplots_adjust(left=0.10, right=0.95, top=0.93, bottom=0.08, hspace=0.25)

# Frequency subplot (top)
freq_line, = ax_freq.plot([], [], 'b-o', markersize=3, linewidth=1.2)
ax_freq.set_ylabel("GHz")
ax_freq.set_title("Channel %d — Real-Time Frequency" % CHANNEL)
ax_freq.grid(True, alpha=0.3)
base_label = ax_freq.text(-0.01, 1.02, "", transform=ax_freq.transAxes, fontsize=10,
                          ha='left', va='bottom', fontweight='bold')

# Frequency target window shading (will be set once freq_base is known)
freq_band = None

# Piezo voltage subplot (bottom)
piezo_line, = ax_curr.plot([], [], 'm-o', markersize=3, linewidth=1.2)
ax_curr.set_xlabel("Time (s elapsed)")
ax_curr.set_ylabel("Voltage (V)")
ax_curr.set_title("%s — Piezo Voltage" % LASER_NAME)
ax_curr.grid(True, alpha=0.3)


def main():
    global freq_base, t_start, freq_band

    relock_attempts = 0  # Track consecutive relock failures

    load_wavemeter()
    
    print(f"Connecting to {LASER_NAME} at {TOPTICA_IP}...")
    toptica_conn = NetworkConnection(TOPTICA_IP, TOPTICA_PORT)
    dlc = DLCpro(toptica_conn)

    try:
        dlc.open()
        print(f"Connected to {LASER_NAME} (SN: {dlc.serial_number.get()})")
        
        target_freq = (FREQ_MIN + FREQ_MAX) / 2.0
        print(f"Tracking Channel {CHANNEL} for window: {FREQ_MIN:.6f} to {FREQ_MAX:.6f} THz")
        print(f"Lockpoint target: {target_freq:.6f} THz")

        t_start = datetime.now(timezone.utc)

        def update(frame):
            global freq_base, freq_band

            now = datetime.now(timezone.utc)
            elapsed = (now - t_start).total_seconds()

            # ── Read wavemeter frequency ──
            freq = get_frequency()

            if freq is not None:
                if freq_base is None:
                    freq_base = int(freq * 1000) / 1000.0
                    base_label.set_text("+ %.3f THz" % freq_base)
                    # Add target window shading in GHz offset units
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
                    direction = "ABOVE" if freq > FREQ_MAX else "BELOW"
                    print(f"Frequency {direction} window.")

                    # Check if we've been failing too many times in a row
                    if relock_attempts >= MAX_RELOCK_ATTEMPTS:
                        print(f"\n[CRITICAL] {relock_attempts} consecutive relock failures.")
                        print("Laser may be stuck in a bad mode. Human intervention required.")
                        lock.lock_enabled.set(False)
                        print(f"Lock disabled. Piezo: {dlc.laser1.dl.pc.voltage_act.get():.3f} V")
                        plt.close('all')
                        input("Fix the laser manually, then press Enter to exit...")
                        sys.exit(1)

                    # Unlock if currently locked
                    if lock.lock_enabled.get():
                        print("  Unlocking laser...")
                        lock.lock_enabled.set(False)
                        time.sleep(DELAY)
                        print(f"  Lock state: {lock.state_txt.get()}")

                    # Adjust piezo voltage toward target
                    print("  Adjusting piezo voltage...")
                    while (freq > target_freq if direction == "ABOVE" else freq < target_freq):
                        step = -STEP_SIZE_V if direction == "ABOVE" else STEP_SIZE_V
                        new_voltage = current_voltage + step
                        human_intervention(new_voltage)
                        dlc.laser1.dl.pc.voltage_set.set(new_voltage)
                        current_voltage = new_voltage
                        print(f"  -> Set Voltage to {current_voltage:.3f}V")
                        time.sleep(DELAY)
                        freq_new = get_frequency()
                        if freq_new is not None:
                            freq = freq_new
                        print(f"  -> New Freq: {freq:.6f} THz")

                    # Verify frequency is stable before re-locking
                    print(f"  Reached target ({freq:.6f} THz). Verifying stability...")
                    stable_count = 0
                    for _ in range(STABLE_READINGS * 2):  # Allow some failed readings
                        time.sleep(DELAY)
                        f = get_frequency()
                        if f is not None and FREQ_MIN <= f <= FREQ_MAX:
                            stable_count += 1
                            print(f"  Stable reading {stable_count}/{STABLE_READINGS}: {f:.6f} THz")
                            if stable_count >= STABLE_READINGS:
                                break
                        else:
                            stable_count = 0
                            if f is not None:
                                print(f"  Unstable: {f:.6f} THz — resetting count")

                    if stable_count >= STABLE_READINGS:
                        # Re-lock 
                        print("  Re-locking...")
                        lock.lock_enabled.set(True)
                        time.sleep(SETTLE_AFTER_LOCK)

                        # Verify lock held
                        f_after = get_frequency()
                        lock_state = lock.state_txt.get()
                        if f_after is not None and FREQ_MIN <= f_after <= FREQ_MAX:
                            print(f"  ✓ Lock held! Freq: {f_after:.6f} THz | Lock: {lock_state}")
                            relock_attempts = 0
                        else:
                            relock_attempts += 1
                            freq_str = f"{f_after:.6f} THz" if f_after else "N/A"
                            print(f"  ✗ Lock caused freq jump! Freq: {freq_str} | Lock: {lock_state}")
                            print(f"  Relock attempt {relock_attempts}/{MAX_RELOCK_ATTEMPTS}")
                    else:
                        relock_attempts += 1
                        print(f"  Could not stabilize. Attempt {relock_attempts}/{MAX_RELOCK_ATTEMPTS}")

                else:
                    # Frequency in window — reset failure counter
                    relock_attempts = 0

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
                ax_curr.relim()
                ax_curr.autoscale_view()

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
