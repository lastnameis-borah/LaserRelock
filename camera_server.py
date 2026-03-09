"""
Camera Server for Thorlabs DCC1545M
====================================
Run this on the Windows server PC where the camera is USB-connected.

Uses the uc480 driver directly via ctypes — no pyueye needed.
Only requires: numpy (pip install numpy)

The uc480_64.dll is already in C:\\Windows\\System32 from the
Thorlabs DCx Camera Support installation.
"""

import ctypes
import socket
import struct
import sys
import numpy as np

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
    hCam = HIDS(0)  # 0 = first available camera
    ret = dll.is_InitCamera(ctypes.byref(hCam), None)
    if ret != IS_SUCCESS:
        sys.exit(f"Camera init failed (code {ret}). Is the camera connected? Is ThorCam using it?")

    # DIB mode = capture to memory (no display window needed)
    dll.is_SetDisplayMode(hCam, IS_SET_DM_DIB)

    # Monochrome 8-bit
    dll.is_SetColorMode(hCam, IS_CM_MONO8)

    # Allocate image memory
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

    # Copy pixel data to our buffer
    buf = (ctypes.c_char * (WIDTH * HEIGHT))()
    ret = dll.is_CopyImageMem(hCam, mem_ptr, mem_id, buf)
    if ret != IS_SUCCESS:
        return None

    frame = np.frombuffer(buf, dtype=np.uint8).reshape(HEIGHT, WIDTH)
    return frame


def main():
    hCam, mem_ptr, mem_id = init_camera()

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(1)

    print(f"Camera server listening on port {PORT}")
    print("Waiting for client connection...\n")

    try:
        while True:
            conn, addr = server.accept()
            print(f"Client connected: {addr}")
            try:
                while True:
                    # Wait for frame request
                    req = conn.recv(5)
                    if not req or req != b'FRAME':
                        break

                    # Capture and send frame
                    frame = capture_frame(hCam, mem_ptr, mem_id)
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
                print(f"Client {addr} disconnected")

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        dll.is_FreeImageMem(hCam, mem_ptr, mem_id)
        dll.is_ExitCamera(hCam)
        server.close()
        print("Camera released.")


if __name__ == '__main__':
    main()
