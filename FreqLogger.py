import wlmData
import wlmConst
import sys
import csv
import ctypes
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime, timezone
from collections import deque
from toptica.lasersdk.dlcpro.v2_0_3 import DLCpro, NetworkConnection

# ─── Configuration ───
DLL_PATH = "/usr/lib/libwlmData.so"
CHANNEL = 3                    # Wavemeter channel to read
POLL_INTERVAL = 0.5            # Seconds
OUTPUT_FILE = "freq_log_ch%d_%s.csv" % (CHANNEL, datetime.now().strftime("%Y%m%d_%H%M%S"))

# Toptica testbed689 laser
TOPTICA_IP = "172.29.13.247"
TOPTICA_PORT = 1998
LASER_NAME = "testbed689"

# ─── Load wavemeter DLL ───
try:
    wlmData.LoadDLL(DLL_PATH)
except:
    sys.exit("Error: Couldn't find DLL on path %s." % DLL_PATH)

if wlmData.dll.GetWLMCount(0) == 0:
    sys.exit("Error: No running wlmServer instance found.")

ver = [wlmData.dll.GetWLMVersion(i) for i in range(4)]
print("WLM Version: [%s.%s.%s.%s]" % tuple(ver))

# Enable pattern output
wlmData.dll.SetPattern(wlmConst.cSignal1Interferometers, wlmConst.cPatternEnable)
wlmData.dll.SetPattern(wlmConst.cSignal1WideInterferometer, wlmConst.cPatternEnable)

# ─── Connect to Toptica laser ───
print("Connecting to %s at %s..." % (LASER_NAME, TOPTICA_IP))
toptica_conn = NetworkConnection(TOPTICA_IP, TOPTICA_PORT)
dlc = DLCpro(toptica_conn)
dlc.open()
print("Connected to %s (SN: %s, FW: %s)" % (LASER_NAME, dlc.serial_number.get(), dlc.fw_ver.get()))

print("Logging Channel %d frequency + %s piezo/current" % (CHANNEL, LASER_NAME))
print("Saving to: %s" % OUTPUT_FILE)
print("Poll interval: %.1f s" % POLL_INTERVAL)
print("Close the plot window or press Ctrl+C to stop.\n")


def get_frequency_status(freq):
    """Convert frequency value to a status string if it's an error code."""
    if freq == wlmConst.ErrWlmMissing:
        return "WLM inactive"
    elif freq == wlmConst.ErrNoSignal:
        return "No Signal"
    elif freq == wlmConst.ErrBadSignal:
        return "Bad Signal"
    elif freq == wlmConst.ErrLowSignal:
        return "Low Signal"
    elif freq == wlmConst.ErrBigSignal:
        return "High Signal"
    elif freq == wlmConst.ErrOutOfRange:
        return "Out of Range"
    elif freq <= 0:
        return "Error code: %d" % int(freq)
    return None


def get_pattern_data(channel, index):
    """Read interferometer pattern data for a channel and pattern index."""
    item_count = wlmData.dll.GetPatternItemCount(index)
    item_size = wlmData.dll.GetPatternItemSize(index)
    if item_count <= 0 or item_size <= 0:
        return None

    ptr = wlmData.dll.GetPatternNum(channel, index)
    if not ptr:
        return None

    if item_size == 2:
        arr = (ctypes.c_int16 * item_count).from_address(ptr)
    else:
        arr = (ctypes.c_int32 * item_count).from_address(ptr)

    return list(arr)


# ─── Data buffers ───
times = deque()
freqs = deque()
freq_offsets = deque()
piezo_vals = deque()
current_vals = deque()
count = 0
freq_base = None

# ─── Set up the plot ───
# Layout: 2 rows x 4 cols
#   Row 0: frequency (2 cols)    | piezo voltage (2 cols)
#   Row 1: fine int. | wide int. | laser current (2 cols)
fig = plt.figure(figsize=(16, 8))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 1], hspace=0.4, wspace=0.35)
fig.canvas.manager.set_window_title("Wavemeter Ch%d + %s — Live Monitor" % (CHANNEL, LASER_NAME))

ax_freq  = fig.add_subplot(gs[0, 0:2])  # top left, 2 cols
ax_piezo = fig.add_subplot(gs[0, 2:4])  # top right, 2 cols
ax_pat1  = fig.add_subplot(gs[1, 0])    # bottom far left
ax_pat2  = fig.add_subplot(gs[1, 1])    # bottom mid left
ax_curr  = fig.add_subplot(gs[1, 2:4])  # bottom right, 2 cols

# Frequency subplot
freq_line, = ax_freq.plot([], [], 'b-o', markersize=3, linewidth=1.2)
ax_freq.set_xlabel("Time (s elapsed)")
ax_freq.set_ylabel("GHz")
ax_freq.set_title("Channel %d — Real-Time Frequency" % CHANNEL)
ax_freq.grid(True, alpha=0.3)
base_label = ax_freq.text(-0.01, 1.02, "", transform=ax_freq.transAxes, fontsize=10,
                          ha='left', va='bottom', fontweight='bold')

# Piezo voltage subplot
piezo_line, = ax_piezo.plot([], [], 'm-o', markersize=3, linewidth=1.2)
ax_piezo.set_xlabel("Time (s elapsed)")
ax_piezo.set_ylabel("Voltage (V)")
ax_piezo.set_title("%s — Piezo Voltage" % LASER_NAME)
ax_piezo.grid(True, alpha=0.3)

# Fine interferometer subplot
pat1_line, = ax_pat1.plot([], [], 'r-', linewidth=0.8)
ax_pat1.set_xlabel("Pixel")
ax_pat1.set_ylabel("Amplitude")
ax_pat1.set_title("Fine Interferometer")
ax_pat1.grid(True, alpha=0.3)

# Wide interferometer subplot
pat2_line, = ax_pat2.plot([], [], 'g-', linewidth=0.8)
ax_pat2.set_xlabel("Pixel")
ax_pat2.set_ylabel("Amplitude")
ax_pat2.set_title("Wide Interferometer")
ax_pat2.grid(True, alpha=0.3)

fig.subplots_adjust(left=0.06, right=0.97, top=0.95, bottom=0.06)

# Laser current subplot
curr_line, = ax_curr.plot([], [], 'c-o', markersize=3, linewidth=1.2)
ax_curr.set_xlabel("Time (s elapsed)")
ax_curr.set_ylabel("Current (mA)")
ax_curr.set_title("%s — Laser Current" % LASER_NAME)
ax_curr.grid(True, alpha=0.3)

# Open CSV file
csvfile = open(OUTPUT_FILE, "w", newline="")
writer = csv.writer(csvfile)
writer.writerow(["timestamp_utc", "channel", "frequency_THz", "piezo_V", "current_mA"])
t_start = datetime.now(timezone.utc)


def update(frame):
    """Called by FuncAnimation each interval."""
    global count, freq_base

    now = datetime.now(timezone.utc)
    now_str = now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    elapsed = (now - t_start).total_seconds()

    # ── Wavemeter frequency ──
    freq = wlmData.dll.GetFrequencyNum(CHANNEL, 0.0)
    status = get_frequency_status(freq)
    freq_str = ""

    if status is None:
        if freq_base is None:
            freq_base = int(freq * 1000) / 1000.0
            base_label.set_text("+ %.3f THz" % freq_base)

        offset_ghz = (freq - freq_base) * 1000.0
        times.append(elapsed)
        freqs.append(freq)
        freq_offsets.append(offset_ghz)
        freq_str = "%.6f" % freq

    # ── Toptica laser data ──
    try:
        piezo = dlc.laser1.dl.pc.voltage_act.get()
        current = dlc.laser1.dl.cc.current_act.get()
    except Exception as e:
        piezo = None
        current = None
        print("Toptica read error: %s" % e)

    if piezo is not None:
        piezo_vals.append(piezo)
    if current is not None:
        current_vals.append(current)

    # ── CSV logging ──
    writer.writerow([now_str, CHANNEL, freq_str,
                     "%.4f" % piezo if piezo is not None else "",
                     "%.4f" % current if current is not None else ""])
    csvfile.flush()
    count += 1

    # ── Update frequency plot ──
    if len(times) > 0:
        freq_line.set_data(list(times), list(freq_offsets))
        ax_freq.relim()
        ax_freq.autoscale_view()

    # ── Update interferometer patterns ──
    pat1 = get_pattern_data(CHANNEL, wlmConst.cSignal1Interferometers)
    if pat1:
        pat1_line.set_data(range(len(pat1)), pat1)
        ax_pat1.relim()
        ax_pat1.autoscale_view()

    pat2 = get_pattern_data(CHANNEL, wlmConst.cSignal1WideInterferometer)
    if pat2:
        pat2_line.set_data(range(len(pat2)), pat2)
        ax_pat2.relim()
        ax_pat2.autoscale_view()

    # ── Update Toptica plots ──
    # Use same times list (same elapsed timestamps)
    t_list = list(times)
    if len(piezo_vals) > 0:
        # Align to latest N points matching times length
        p_list = list(piezo_vals)
        min_len = min(len(t_list), len(p_list))
        piezo_line.set_data(t_list[-min_len:], p_list[-min_len:])
        ax_piezo.relim()
        ax_piezo.autoscale_view()

    if len(current_vals) > 0:
        c_list = list(current_vals)
        min_len = min(len(t_list), len(c_list))
        curr_line.set_data(t_list[-min_len:], c_list[-min_len:])
        ax_curr.relim()
        ax_curr.autoscale_view()

    return freq_line, pat1_line, pat2_line, piezo_line, curr_line


ani = animation.FuncAnimation(fig, update, interval=int(POLL_INTERVAL * 1000), blit=False, cache_frame_data=False)

try:
    plt.show()
except KeyboardInterrupt:
    pass
finally:
    csvfile.close()
    dlc.close()
    print("\nStopped. %d readings saved to '%s'." % (count, OUTPUT_FILE))
