import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Load data
df = pd.read_csv("ltm-17march.csv", parse_dates=["timestamp_utc"])

# Calculate elapsed time in hours
t0 = df["timestamp_utc"].iloc[0]
df["elapsed_h"] = (df["timestamp_utc"] - t0).dt.total_seconds() / 3600

# Frequency offset in MHz from base
freq_base = int(df["frequency_THz"].iloc[0] * 10000) / 10000.0
df["freq_MHz"] = (df["frequency_THz"] - freq_base) * 1e6

# ─── Plot ───
fig, (ax_freq, ax_piezo, ax_curr, ax_temp) = plt.subplots(4, 1, figsize=(20, 15), sharex=True)
fig.canvas.manager.set_window_title("LTM Log")
fig.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.06, hspace=0.25)

# Frequency
ax_freq.plot(df["elapsed_h"], df["freq_MHz"], 'b-', linewidth=0.5)
ax_freq.set_ylabel("MHz")
ax_freq.set_title(f"Frequency (+ {freq_base:.4f} THz)")
ax_freq.grid(True, alpha=0.3)

# Piezo voltage
ax_piezo.plot(df["elapsed_h"], df["piezo_V"], 'm-', linewidth=0.5)
ax_piezo.set_ylabel("Voltage (V)")
ax_piezo.set_title("Piezo Voltage")
ax_piezo.grid(True, alpha=0.3)

# Current
ax_curr.plot(df["elapsed_h"], df["current_mA"], 'c-', linewidth=0.5)
ax_curr.set_ylabel("Current (mA)")
ax_curr.set_title("Laser Current")
ax_curr.grid(True, alpha=0.3)

# Temperature
ax_temp.plot(df["elapsed_h"], df["temp_C"], 'r-', linewidth=0.5)
ax_temp.set_xlabel("Time (hours)")
ax_temp.set_ylabel("Temp (°C)")
ax_temp.set_title("Laser Temperature")
ax_temp.grid(True, alpha=0.3)

# ax_piezo.set_xlim(0, 22)
    
plt.show()