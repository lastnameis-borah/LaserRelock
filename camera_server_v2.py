"""
Camera Server v2 for Thorlabs DCC1545M
========================================
Run this on the Windows server PC where the camera is USB-connected.
Replaces ThorCam — shows a local display AND serves frames to remote clients.

Uses the uc480 driver directly via ctypes (no pyueye needed).
Only requires: numpy, matplotlib

Usage:
    python camera_server_v2.py

The local window shows the live camera feed with intensity stats.
Remote clients connect on port 5556 to receive frames.
"""

import ctypes
import socket
import struct
import sys
import threading
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ─── Camera specs (DCC1545M) ───
WIDTH = 1280
HEIGHT = 1024

# ─── uc480 constants ───
IS_SUCCESS = 0
IS_CM_MONO8 = 6
IS_WAIT = 0x0001
IS_SET_DM_DIB = 1

# ─── Server config ───
HOST = '0.0.0.0'
PORT = 5556
DISPLAY_INTERVAL = 100  # ms between local display updates

# ─── Load uc480 DLL ───
try:
    dll = ctypes.cdll.LoadLibrary("uc480_64")
except OSError:
    try:
        dll = ctypes.cdll.LoadLibrary("uc480")
    except OSError:
        sys.exit("Error: Could not load uc480 DLL. Is the Thorlabs DCx driver installed?")

print("uc480 DLL loaded successfully.")

# ─── Set up function signatures for 64-bit safety ───
HIDS = ctypes.c_uint32

dll.is_InitCamera.argtypes = [ctypes.POINTER(HIDS), ctypes.c_void_p]
dll.is_InitCamera.restype = ctypes.c_int

dll.is_SetDisplayMode.argtypes = [HIDS, ctypes.c_int]
dll.is_SetDisplayMode.restype = ctypes.c_int

dll.is_SetColorMode.argtypes = [HIDS, ctypes.c_int]
dll.is_SetColorMode.restype = ctypes.c_int

dll.is_AllocImageMem.argtypes = [HIDS, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                  ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_int)]
dll.is_AllocImageMem.restype = ctypes.c_int

dll.is_SetImageMem.argtypes = [HIDS, ctypes.c_char_p, ctypes.c_int]
dll.is_SetImageMem.restype = ctypes.c_int

dll.is_FreezeVideo.argtypes = [HIDS, ctypes.c_int]
dll.is_FreezeVideo.restype = ctypes.c_int

dll.is_CopyImageMem.argtypes = [HIDS, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p]
dll.is_CopyImageMem.restype = ctypes.c_int

dll.is_FreeImageMem.argtypes = [HIDS, ctypes.c_char_p, ctypes.c_int]
dll.is_FreeImageMem.restype = ctypes.c_int

dll.is_ExitCamera.argtypes = [HIDS]
dll.is_ExitCamera.restype = ctypes.c_int


def init_camera():
    """Initialize the DCC1545M camera."""
    hCam = HIDS(0)
    ret = dll.is_InitCamera(ctypes.byref(hCam), None)
    if ret != IS_SUCCESS:
        sys.exit(f"Camera init failed (code {ret}). Is the camera connected? Is ThorCam closed?")

    dll.is_SetDisplayMode(hCam, IS_SET_DM_DIB)
    dll.is_SetColorMode(hCam, IS_CM_MONO8)

    mem_ptr = ctypes.c_char_p()
    mem_id = ctypes.c_int()
    ret = dll.is_AllocImageMem(hCam, WIDTH, HEIGHT, 8,
                                ctypes.byref(mem_ptr), ctypes.byref(mem_id))
    if ret != IS_SUCCESS:
        sys.exit(f"Memory allocation failed (code {ret})")

    ret = dll.is_SetImageMem(hCam, mem_ptr, mem_id)
    if ret != IS_SUCCESS:
        sys.exit(f"Set image memory failed (code {ret})")

    print(f"Camera initialized: {WIDTH}x{HEIGHT} Mono8")
    return hCam, mem_ptr, mem_id


def capture_frame(hCam, mem_ptr, mem_id):
    """Capture a single frame and return as numpy array."""
    ret = dll.is_FreezeVideo(hCam, IS_WAIT)
    if ret != IS_SUCCESS:
        return None

    buf = (ctypes.c_char * (WIDTH * HEIGHT))()
    ret = dll.is_CopyImageMem(hCam, mem_ptr, mem_id, buf)
    if ret != IS_SUCCESS:
        return None

    return np.frombuffer(buf, dtype=np.uint8).reshape(HEIGHT, WIDTH)


# ─── Shared state for server thread ───
latest_frame = None
frame_lock = threading.Lock()
server_running = True


def server_thread():
    """Background TCP server that serves frames to remote clients."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.settimeout(1.0)  # Allow periodic check of server_running
    server.bind((HOST, PORT))
    server.listen(2)

    print(f"Remote server listening on port {PORT}")

    while server_running:
        try:
            conn, addr = server.accept()
            print(f"Remote client connected: {addr}")
            try:
                while server_running:
                    req = conn.recv(5)
                    if not req or req != b'FRAME':
                        break

                    with frame_lock:
                        frame = latest_frame.copy() if latest_frame is not None else None

                    if frame is None:
                        conn.sendall(struct.pack('>II', 0, 0))
                        continue

                    h, w = frame.shape
                    header = struct.pack('>II', h, w)
                    conn.sendall(header + frame.tobytes())

            except (ConnectionResetError, BrokenPipeError):
                pass
            finally:
                conn.close()
                print(f"Remote client {addr} disconnected")

        except socket.timeout:
            continue
        except OSError:
            break

    server.close()
    print("Remote server stopped.")


def main():
    global latest_frame, server_running

    hCam, mem_ptr, mem_id = init_camera()

    # Start TCP server in background thread
    srv_thread = threading.Thread(target=server_thread, daemon=True)
    srv_thread.start()

    # ─── Set up local display ───
    fig, ax = plt.subplots(figsize=(12, 9))
    fig.canvas.manager.set_window_title("DCC1545M Camera Server — Cavity Transmission")
    fig.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.05)

    img_display = [None]  # Use list for mutability in nested function

    def update(frame_num):
        global latest_frame

        frame = capture_frame(hCam, mem_ptr, mem_id)
        if frame is None:
            return

        # Share with server thread
        with frame_lock:
            latest_frame = frame.copy()

        # Intensity stats
        mean_val = float(np.mean(frame))
        peak_val = int(np.max(frame))
        total_val = float(np.sum(frame.astype(np.float64)))

        # Update display
        if img_display[0] is None:
            img_display[0] = ax.imshow(frame, cmap='gray', vmin=0, vmax=255, aspect='equal')
            plt.colorbar(img_display[0], ax=ax, label='Intensity', shrink=0.8)
        else:
            img_display[0].set_data(frame)

        ax.set_title(f"Mean: {mean_val:.1f}  |  Peak: {peak_val}  |  Total: {total_val:.0f}",
                     fontsize=12)

    ani = animation.FuncAnimation(fig, update, interval=DISPLAY_INTERVAL,
                                   blit=False, cache_frame_data=False)

    print("Local display running. Close the window to stop.\n")

    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        server_running = False
        srv_thread.join(timeout=3)
        dll.is_FreeImageMem(hCam, mem_ptr, mem_id)
        dll.is_ExitCamera(hCam)
        print("Camera released. Exiting.")


if __name__ == '__main__':
    main()
