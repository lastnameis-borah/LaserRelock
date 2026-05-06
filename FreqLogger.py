import wlmData
import wlmConst
import sys
import csv
import ctypes
import socket
import struct
import numpy as np
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
        print("Camera read error: %s" % e)
        return None

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

# ─── Connect to camera server (optional) ───
cam_sock = None
print("Connecting to camera server at %s:%d..." % (CAMERA_IP, CAMERA_PORT))
try:
    cam_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    cam_sock.settimeout(5.0)
    cam_sock.connect((CAMERA_IP, CAMERA_PORT))
    cam_sock.settimeout(10.0)
    print("Camera connected.")
except Exception as e:
    print("Camera unavailable (%s). Continuing without camera." % e)
    if cam_sock is not None:
        try:
            cam_sock.close()
        except Exception:
            pass
    cam_sock = None

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
intensity_vals = deque()
count = 0
freq_base = None

# ─── Set up the plot ───
# Layout: 3 rows x 4 cols
#   Row 0: frequency (2 cols)    | piezo voltage (2 cols)
#   Row 1: fine int. | wide int. | laser current (2 cols)
#   Row 2: camera ROI intensity (4 cols)
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1], hspace=0.45, wspace=0.35)
fig.canvas.manager.set_window_title("Wavemeter Ch%d + %s — Live Monitor" % (CHANNEL, LASER_NAME))

ax_freq  = fig.add_subplot(gs[0, 0:2])  # top left, 2 cols
ax_piezo = fig.add_subplot(gs[0, 2:4])  # top right, 2 cols
ax_pat1  = fig.add_subplot(gs[1, 0])    # mid far left
ax_pat2  = fig.add_subplot(gs[1, 1])    # mid mid left
ax_curr  = fig.add_subplot(gs[1, 2:4])  # mid right, 2 cols
ax_cam   = fig.add_subplot(gs[2, 0:4])  # bottom, full width

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

# Camera ROI intensity subplot
cam_line, = ax_cam.plot([], [], 'y-o', markersize=3, linewidth=1.2)
ax_cam.set_xlabel("Time (s elapsed)")
ax_cam.set_ylabel("ROI Mean")
ax_cam.set_title("Camera ROI Intensity%s"
                 % ("" if cam_sock is not None else " (disconnected)"))
ax_cam.grid(True, alpha=0.3)

# Open CSV file
csvfile = open(OUTPUT_FILE, "w", newline="")
writer = csv.writer(csvfile)
writer.writerow(["timestamp_utc", "channel", "frequency_THz", "piezo_V", "current_mA", "roi_mean"])
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

    # ── Camera ROI intensity ──
    roi_mean = None
    if cam_sock is not None:
        roi_mean = get_roi_intensity(cam_sock)
    if roi_mean is not None:
        intensity_vals.append(roi_mean)

    # ── CSV logging ──
    writer.writerow([now_str, CHANNEL, freq_str,
                     "%.4f" % piezo if piezo is not None else "",
                     "%.4f" % current if current is not None else "",
                     "%.2f" % roi_mean if roi_mean is not None else ""])
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

    if len(intensity_vals) > 0:
        i_list = list(intensity_vals)
        min_len = min(len(t_list), len(i_list))
        cam_line.set_data(t_list[-min_len:], i_list[-min_len:])
        ax_cam.relim()
        ax_cam.autoscale_view()

    return freq_line, pat1_line, pat2_line, piezo_line, curr_line, cam_line


ani = animation.FuncAnimation(fig, update, interval=int(POLL_INTERVAL * 1000), blit=False, cache_frame_data=False)

try:
    plt.show()
except KeyboardInterrupt:
    pass
finally:
    csvfile.close()
    dlc.close()
    if cam_sock is not None:
        try:
            cam_sock.close()
        except Exception:
            pass
    print("\nStopped. %d readings saved to '%s'." % (count, OUTPUT_FILE))
