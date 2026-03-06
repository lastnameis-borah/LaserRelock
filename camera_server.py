"""
Camera Server for Thorlabs DCC1545M
====================================
Run this on the Windows server PC where the camera is USB-connected.

Requirements:
    pip install pyueye numpy

Note: This should work alongside ThorCam (the IDS uEye daemon
supports shared access). If you get an init error, try closing ThorCam.
"""

import socket
import struct
import sys
import numpy as np
from pyueye import ueye

# ─── Configuration ───
HOST = '0.0.0.0'       # Listen on all interfaces
PORT = 5556             # Camera server port


def init_camera():
    """Initialize the DCC1545M camera and return handles."""
    hCam = ueye.HIDS(0)  # 0 = first available camera
    ret = ueye.is_InitCamera(hCam, None)
    if ret != ueye.IS_SUCCESS:
        sys.exit(f"Camera init failed (error code {ret}). Is ThorCam closed?")

    # Get sensor info
    sInfo = ueye.SENSORINFO()
    ueye.is_GetSensorInfo(hCam, sInfo)
    width = sInfo.nMaxWidth
    height = sInfo.nMaxHeight

    # Set monochrome 8-bit mode
    ueye.is_SetColorMode(hCam, ueye.IS_CM_MONO8)

    # Allocate image memory
    mem_ptr = ueye.c_mem_p()
    mem_id = ueye.INT()
    ret = ueye.is_AllocImageMem(hCam, width, height, 8, mem_ptr, mem_id)
    if ret != ueye.IS_SUCCESS:
        sys.exit(f"Memory allocation failed (error code {ret})")

    ueye.is_SetImageMem(hCam, mem_ptr, mem_id)

    print(f"Camera initialized: {width.value}x{height.value} Mono8")
    return hCam, mem_ptr, mem_id, width, height


def capture_frame(hCam, mem_ptr, width, height):
    """Capture a single frame and return as numpy array."""
    ret = ueye.is_FreezeVideo(hCam, ueye.IS_WAIT)
    if ret != ueye.IS_SUCCESS:
        return None

    array = ueye.get_data(mem_ptr, width, height, 8, copy=True)
    frame = np.reshape(array, (height.value, width.value))
    return frame


def main():
    hCam, mem_ptr, mem_id, width, height = init_camera()

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
                    # Wait for frame request from client
                    req = conn.recv(5)
                    if not req or req != b'FRAME':
                        break

                    # Capture frame
                    frame = capture_frame(hCam, mem_ptr, width, height)
                    if frame is None:
                        # Send zero-size to indicate error
                        conn.sendall(struct.pack('>II', 0, 0))
                        continue

                    # Send: height (4 bytes) + width (4 bytes) + raw pixel data
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
        ueye.is_StopLiveVideo(hCam, ueye.IS_FORCE_VIDEO_STOP)
        ueye.is_FreeImageMem(hCam, mem_ptr, mem_id)
        ueye.is_ExitCamera(hCam)
        server.close()
        print("Camera released.")


if __name__ == '__main__':
    main()
