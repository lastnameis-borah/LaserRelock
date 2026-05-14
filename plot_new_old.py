import pandas as pd
import matplotlib.pyplot as plt

# Load merged dataset
df = pd.read_csv("relock_log/new_vs_old.csv", parse_dates=["timestamp_utc"])

# Elapsed time in hours from the start of the merged data
t0 = df["timestamp_utc"].iloc[0]
df["elapsed_h"] = (df["timestamp_utc"] - t0).dt.total_seconds() / 3600

# Boundary between ml_lock and relock_v2 segments
boundary_h = df.loc[df["source"] == "ml_lock", "elapsed_h"].max()

REMOVE = [(9.5849, 9.5862)]
mask = ~pd.concat(
    [(df["elapsed_h"] >= lo) & (df["elapsed_h"] <= hi) for lo, hi in REMOVE],
    axis=1,
).any(axis=1)
df = df[mask]

# Frequency offset in MHz
freq_base = int(df["frequency_THz"].iloc[0] * 10000) / 10000.0
df["freq_MHz"] = (df["frequency_THz"] - freq_base) * 1e6
df.loc[df["elapsed_h"] < 9.5849, "piezo_V"] += 0.15

# ─── Plot ───
fig, (ax_freq, ax_piezo, ax_curr) = plt.subplots(3, 1, figsize=(20, 12), sharex=True)
fig.canvas.manager.set_window_title("Merged Log: ml_lock → relock_v2")
fig.subplots_adjust(left=0.08, right=0.95, top=0.93, bottom=0.07, hspace=0.3)
# fig.suptitle(
#     f"CH3 — last 25 h ml_lock  +  first 25 h relock_v2\n"
#     f"Start: {t0.strftime('%Y-%m-%d %H:%M')} UTC",
#     fontsize=11,
# )

# Frequency
ax_freq.plot(df["elapsed_h"], df["freq_MHz"], "b-", linewidth=0.4)
ax_freq.set_ylabel("MHz")
ax_freq.set_title(f"Frequency (+ {freq_base:.4f} THz)")
ax_freq.grid(True, alpha=0.3)

# Piezo voltage
ax_piezo.plot(df["elapsed_h"], df["piezo_V"], "m-", linewidth=0.4)
ax_piezo.set_ylabel("Voltage (V)")
ax_piezo.set_title("Piezo Voltage")
ax_piezo.grid(True, alpha=0.3)

# Current
ax_curr.plot(df["elapsed_h"], df["current_mA"], "c-", linewidth=0.4)
ax_curr.set_xlabel("Time (hours from start)")
ax_curr.set_ylabel("Current (mA)")
ax_curr.set_title("Laser Current")
ax_curr.grid(True, alpha=0.3)

# Dotted red vertical line at the segment boundary on all axes
for ax in (ax_freq, ax_piezo, ax_curr):
    ax.axvline(boundary_h, color="red", linestyle=":", linewidth=1.5, zorder=5)


# ax_piezo.set_xlim(9.5849, 9.5862)


plt.show()
