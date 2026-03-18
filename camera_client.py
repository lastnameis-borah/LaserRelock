"""
Camera Client — Thorlabs DCC1545M Viewer
==========================================
Run this on the Linux client PC.
Connects to camera_server.py running on the server PC.

Displays the live camera feed with intensity statistics,
useful for finding cavity transmission peaks.
"""

import socket
import struct
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import csv
from datetime import datetime, timezone

# ─── Configuration ───
# Set this to the IP of your server PC (where camera_server.py is running)
SERVER_IP = "172.29.13.140"   # IP of the windows PC
SERVER_PORT = 5556
POLL_INTERVAL = 200           # ms between frame requests

# ROI for intensity calculation (x1, y1, x2, y2)
ROI_X1, ROI_Y1 = 395, 430
ROI_X2, ROI_Y2 = 595, 630


def recv_exact(sock, n):
    """Receive exactly n bytes from socket."""
    data = b''
    while len(data) < n:
        chunk = sock.recv(min(n - len(data), 65536))
        if not chunk:
            raise ConnectionError("Server disconnected")
        data += chunk
    return data


def main():
    if "XXX" in SERVER_IP:
        print("ERROR: Update SERVER_IP in camera_client.py to your server PC's IP address.")
        sys.exit(1)

    print(f"Connecting to camera server at {SERVER_IP}:{SERVER_PORT}...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5.0)
    try:
        sock.connect((SERVER_IP, SERVER_PORT))
    except (ConnectionRefusedError, TimeoutError):
        print(f"Could not connect. Is camera_server.py running on {SERVER_IP}?")
        sys.exit(1)

    sock.settimeout(10.0)
    print("Connected!\n")

    # ─── Set up plot ───
    fig, (ax_img, ax_hist) = plt.subplots(1, 2, figsize=(20, 6),
                                           gridspec_kw={'width_ratios': [1, 1]})
    fig.canvas.manager.set_window_title("Thorlabs DCC1545M — Cavity Transmission")
    fig.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.08, wspace=0.15)

    img_display = None
    cbar = None

    # Intensity history for time trace
    intensity_history = []

    # CSV logging
    csv_filename = f"camera-{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    csv_file = open(csv_filename, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["timestamp_utc", "roi_mean", "roi_peak", "roi_total"])
    log_count = 0
    print(f"Logging to: {csv_filename}")

    def update(frame_num):
        nonlocal img_display, cbar, log_count

        try:
            # Request a frame
            sock.sendall(b'FRAME')

            # Receive header
            header = recv_exact(sock, 8)
            h, w = struct.unpack('>II', header)

            if h == 0 or w == 0:
                return

            # Receive pixel data
            data = recv_exact(sock, h * w)
            frame = np.frombuffer(data, dtype=np.uint8).reshape(h, w)

        except (ConnectionError, TimeoutError) as e:
            print(f"Connection error: {e}")
            plt.close('all')
            return

        # ── ROI intensity statistics ──
        roi = frame[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]
        total = np.sum(roi.astype(np.float64))
        peak = int(np.max(roi))
        mean = float(np.mean(roi))

        intensity_history.append(mean)

        # Log to CSV
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        csv_writer.writerow([now_str, f"{mean:.2f}", peak, f"{total:.0f}"])
        csv_file.flush()
        log_count += 1

        # ── Update camera image ──
        ax_img.clear()
        img_display = ax_img.imshow(frame, cmap='inferno', vmin=0, vmax=255,
                                     aspect='equal')
        # Draw ROI rectangle
        roi_rect = plt.Rectangle((ROI_X1, ROI_Y1), ROI_X2 - ROI_X1, ROI_Y2 - ROI_Y1,
                                  linewidth=1.5, edgecolor='lime', facecolor='none')
        ax_img.add_patch(roi_rect)
        ax_img.set_title(f"ROI Mean: {mean:.1f}  |  Peak: {peak}  |  Total: {total:.0f}",
                         fontsize=11)
        ax_img.set_xlabel("Pixel X")
        ax_img.set_ylabel("Pixel Y")

        # ── Update intensity trace ──
        ax_hist.clear()
        ax_hist.plot(intensity_history, 'c-', linewidth=1.2)
        ax_hist.set_xlabel("Frame #")
        ax_hist.set_ylabel("Mean Intensity")
        ax_hist.set_title("Intensity Trace")
        ax_hist.grid(True, alpha=0.3)
        if len(intensity_history) > 1:
            ax_hist.set_xlim(0, max(len(intensity_history), 10))

    ani = animation.FuncAnimation(fig, update, interval=POLL_INTERVAL,
                                   blit=False, cache_frame_data=False)
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        csv_file.close()
        print(f"\n{log_count} readings saved to 'camera.csv'.")
        sock.close()
        print("Disconnected.")


if __name__ == '__main__':
    main()
