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

# ─── Configuration ───
# Set this to the IP of your server PC (where camera_server.py is running)
SERVER_IP = "172.29.13.XXX"   # <-- UPDATE THIS
SERVER_PORT = 5556
POLL_INTERVAL = 200           # ms between frame requests


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
    fig, (ax_img, ax_hist) = plt.subplots(1, 2, figsize=(14, 6),
                                           gridspec_kw={'width_ratios': [3, 1]})
    fig.canvas.manager.set_window_title("Thorlabs DCC1545M — Cavity Transmission")
    fig.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.08, wspace=0.15)

    img_display = None
    cbar = None

    # Intensity history for time trace
    intensity_history = []
    max_history = 200

    def update(frame_num):
        nonlocal img_display, cbar

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

        # ── Intensity statistics ──
        total = np.sum(frame.astype(np.float64))
        peak = int(np.max(frame))
        mean = float(np.mean(frame))

        intensity_history.append(mean)
        if len(intensity_history) > max_history:
            intensity_history.pop(0)

        # ── Update camera image ──
        ax_img.clear()
        img_display = ax_img.imshow(frame, cmap='inferno', vmin=0, vmax=255,
                                     aspect='equal')
        ax_img.set_title(f"Mean: {mean:.1f}  |  Peak: {peak}  |  Total: {total:.0f}",
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
        sock.close()
        print("Disconnected.")


if __name__ == '__main__':
    main()
