import sys
import wlmData
import wlmConst

# ─── Configuration ───
DLL_PATH = "/usr/lib/libwlmData.so"
CHANNEL = 8
TARGET_FREQ = 434.829040  # 689nm cavity

# Load wavemeter
try:
    wlmData.LoadDLL(DLL_PATH)
except:
    sys.exit(f"Error: Couldn't find DLL on path {DLL_PATH}.")

if wlmData.dll.GetWLMCount(0) == 0:
    sys.exit("Error: No running wlmServer instance found.")

# Read current frequency before calibration
freq_before = wlmData.dll.GetFrequencyNum(CHANNEL, 0.0)
if freq_before <= 0:
    sys.exit(f"Error: No valid frequency on channel {CHANNEL}.")

print(f"Channel {CHANNEL} before calibration: {freq_before:.6f} THz")
print(f"Target frequency:                     {TARGET_FREQ:.6f} THz")
print(f"Offset:                               {(freq_before - TARGET_FREQ)*1e6:.1f} MHz")

# Calibrate
print(f"\nCalibrating channel {CHANNEL} to {TARGET_FREQ:.6f} THz...")
result = wlmData.dll.Calibration(wlmConst.cOther, wlmConst.cReturnFrequency, TARGET_FREQ, CHANNEL)

if result == wlmConst.ResERR_NoErr:
    print("Calibration successful!")
else:
    print(f"Calibration returned code: {result}")

# Read frequency after calibration
freq_after = wlmData.dll.GetFrequencyNum(CHANNEL, 0.0)
if freq_after > 0:
    print(f"\nChannel {CHANNEL} after calibration:  {freq_after:.6f} THz")
    print(f"Residual offset:                      {(freq_after - TARGET_FREQ)*1e6:.1f} MHz")

print("\nDone.")
