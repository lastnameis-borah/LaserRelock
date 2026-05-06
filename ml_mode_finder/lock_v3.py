"""
ML-assisted auto-relock with piezo centering (v3)

New in v3:
  - Loads current_search scan data to build an I*(V) curve — the current
    that places the laser on target frequency at each piezo voltage.
  - While locked, a slow centering loop nudges the current to keep the
    piezo near the centre of its MHF plateau, reducing mode-hop frequency.

Run: python3 -m ml_mode_finder.lock_v3
"""
import csv
import glob
import json
import os
import sys
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import wlmData
from toptica.lasersdk.dlcpro.v2_0_3 import DLCpro, NetworkConnection

from ml_mode_finder.find_mode import find, FREQ_TARGET, FREQ_WINDOW, I_MIN, I_MAX

# ─── Hardware ────────────────────────────────────────────────────────────────
DLL_PATH     = "/usr/lib/libwlmData.so"
CHANNEL      = 3
TOPTICA_IP   = "172.29.13.247"
TOPTICA_PORT = 1998
LASER_NAME   = "testbed689"

# ─── Lock window ─────────────────────────────────────────────────────────────
FREQ_MIN = FREQ_TARGET - FREQ_WINDOW
FREQ_MAX = FREQ_TARGET + FREQ_WINDOW

# ─── Timing ──────────────────────────────────────────────────────────────────
SETTLE_AFTER_SET  = 0.3
DELAY             = 0.2
SETTLE_AFTER_LOCK = 5.0

# ─── Piezo sweep ─────────────────────────────────────────────────────────────
STEP_COARSE_V     = 0.1    # 100 mV far from target
STEP_FINE_V       = 0.01   # 10 mV near target
COARSE_THRESH_MHZ = 50.0
STABLE_READINGS   = 5
VOLTAGE_MIN       = 20.0
VOLTAGE_MAX       = 60.0

# ─── Relock ──────────────────────────────────────────────────────────────────
MAX_RELOCK_ATTEMPTS = 3
LARGE_HOP_THZ       = 0.005  # 5 GHz — skip bounded sweep beyond this

# ─── Piezo centering ─────────────────────────────────────────────────────────
CURRENT_SEARCH_DIR     = "/home/artiq/LaserRelock/current_search"
RECENT_SCANS_CS        = 3     # most recent current_search scans to use
CENTERING_INTERVAL_S   = 30.0  # how often to check centering while locked
CENTERING_MIN_LOCK_S   = 60.0  # don't center until locked this long
CENTERING_DEADBAND_V   = 1.5   # V — skip nudge if piezo within this of centre
CENTERING_GAIN         = 0.5   # fraction of full correction to apply per step
CENTERING_MAX_NUDGE_MA = 0.15  # mA — hard cap per nudge

# ─── Output ──────────────────────────────────────────────────────────────────
LOG_DIR = "/home/artiq/LaserRelock/relock_log"


# ═════════════════════════════════════════════════════════════════════════════
# I*(V) map built from current_search scans
# ═════════════════════════════════════════════════════════════════════════════

class CurrentSearchMap:
    """
    Sparse I*(V) curve: interpolates the current that places the laser on
    FREQ_TARGET at each piezo voltage, built from recent current_search JSONs.
    """

    def __init__(self, search_dir=CURRENT_SEARCH_DIR, recent_scans=RECENT_SCANS_CS):
        json_files = sorted(glob.glob(os.path.join(search_dir, "search_*.json")))
        if not json_files:
            raise FileNotFoundError(f"No current_search JSONs in {search_dir}")
        json_files = json_files[-recent_scans:]

        by_piezo = defaultdict(list)
        for jf in json_files:
            with open(jf) as f:
                data = json.load(f)
            for cand in data.get("all_candidates", []):
                by_piezo[round(cand["piezo_V"], 4)].append(cand["current_mA"])

        if not by_piezo:
            raise ValueError("No candidates found in current_search JSONs")

        vs  = np.array(sorted(by_piezo))
        is_ = np.array([float(np.median(by_piezo[v])) for v in vs])
        self.piezo_vals   = vs
        self.current_vals = is_
        self.n_scans      = len(json_files)
        print(f"[CENTER] I*(V) curve: {len(vs)} piezo points from "
              f"{self.n_scans} scan(s)  "
              f"V={vs[0]:.1f}–{vs[-1]:.1f} V  "
              f"I={is_.min():.2f}–{is_.max():.2f} mA")

    def get_target_current(self, piezo_V):
        """Interpolated target current at piezo_V. Returns None if out of range."""
        v = float(piezo_V)
        if v < self.piezo_vals[0] or v > self.piezo_vals[-1]:
            return None
        return float(np.interp(v, self.piezo_vals, self.current_vals))


# ═════════════════════════════════════════════════════════════════════════════
# Shared state (worker thread writes, animation thread reads)
# ═════════════════════════════════════════════════════════════════════════════

_buf = {
    'times':   deque(),
    'freq':    deque(),    # MHz offset from FREQ_TARGET
    'piezo':   deque(),    # V
    'current': deque(),    # mA
    'status':  ['Starting up...'],
    't0':      [0.0],
}


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def get_frequency():
    f = wlmData.dll.GetFrequencyNum(CHANNEL, 0.0)
    return None if f <= 0 else f

def _voltage_safety(v):
    if v < VOLTAGE_MIN or v > VOLTAGE_MAX:
        print(f"\n[CRITICAL] Voltage {v:.3f} V out of bounds. Exiting.")
        os._exit(1)

def _record(writer, csvfile, elapsed, freq, piezo, current_mA, temp_c,
            lock_state, in_win):
    _buf['times'].append(elapsed)
    _buf['freq'].append((freq - FREQ_TARGET) * 1e6)
    _buf['piezo'].append(piezo)
    _buf['current'].append(current_mA)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    writer.writerow([ts, CHANNEL, f"{freq:.6f}",
                     f"{piezo:.4f}", f"{current_mA:.4f}",
                     f"{temp_c:.4f}" if temp_c is not None else "",
                     lock_state, int(in_win)])
    csvfile.flush()


# ═════════════════════════════════════════════════════════════════════════════
# Piezo sweep + lock
# ═════════════════════════════════════════════════════════════════════════════

def piezo_sweep_and_lock(dlc, writer, csvfile, mhf_min=None, mhf_max=None):
    """
    Sweep piezo toward target freq, check stability, engage lock.
    Returns True on success.
    """
    target = (FREQ_MIN + FREQ_MAX) / 2.0
    lock   = dlc.laser1.dl.lock

    v_lo = mhf_min if mhf_min is not None else VOLTAGE_MIN
    v_hi = mhf_max if mhf_max is not None else VOLTAGE_MAX

    freq = get_frequency()
    if freq is None:
        return False

    if lock.lock_enabled.get():
        lock.lock_enabled.set(False)
        time.sleep(DELAY)

    voltage    = dlc.laser1.dl.pc.voltage_act.get()
    current_mA = dlc.laser1.dl.cc.current_act.get()
    try:
        temp_c = dlc.laser1.dl.tc.temp_act.get()
    except Exception:
        temp_c = None

    if freq > FREQ_MAX:
        direction = "ABOVE"
    elif freq < FREQ_MIN:
        direction = "BELOW"
    else:
        direction = None

    if direction:
        sign = -1 if direction == "ABOVE" else 1
        _buf['status'][0] = (f"Sweeping {'down' if sign < 0 else 'up'} "
                             f"({(freq - target)*1e6:+.0f} MHz)...")
        print(f"  [SWEEP] Freq {direction} target "
              f"({(freq - target)*1e6:+.1f} MHz). "
              f"MHF bounds {v_lo:.2f}–{v_hi:.2f} V")
        while freq > FREQ_MAX if direction == "ABOVE" else freq < FREQ_MIN:
            offset_mhz = abs(freq - target) * 1e6
            step = sign * (STEP_COARSE_V if offset_mhz > COARSE_THRESH_MHZ
                           else STEP_FINE_V)
            new_v = voltage + step
            if not (v_lo <= new_v <= v_hi):
                print(f"  [SWEEP] Hit MHF bound at {new_v:.3f} V "
                      f"(region {v_lo:.2f}–{v_hi:.2f} V). Giving up.")
                return False
            _voltage_safety(new_v)
            dlc.laser1.dl.pc.voltage_set.set(new_v)
            voltage = new_v
            time.sleep(DELAY)
            f_new = get_frequency()
            if f_new is not None:
                freq = f_new
            elapsed = time.monotonic() - _buf['t0'][0]
            in_win  = FREQ_MIN <= freq <= FREQ_MAX
            _record(writer, csvfile, elapsed, freq, voltage, current_mA,
                    temp_c, "sweeping", in_win)
            print(f"    V={voltage:.3f}V  f={freq:.6f} THz "
                  f"({(freq - target)*1e6:+.1f} MHz)  "
                  f"step={step*1e3:.0f}mV")

    _buf['status'][0] = "Checking stability..."
    print(f"  [SWEEP] Checking stability ({STABLE_READINGS} reads)...")
    stable = 0
    for _ in range(STABLE_READINGS * 2):
        time.sleep(DELAY)
        f = get_frequency()
        if f is not None and FREQ_MIN <= f <= FREQ_MAX:
            stable += 1
            if stable >= STABLE_READINGS:
                break
        else:
            stable = 0

    if stable < STABLE_READINGS:
        print("  [SWEEP] Stability check failed.")
        return False

    print("  [SWEEP] Stable. Engaging lock...")
    _buf['status'][0] = "Engaging lock..."
    lock.lock_enabled.set(True)
    time.sleep(SETTLE_AFTER_LOCK)

    f_after    = get_frequency()
    lock_state = lock.state_txt.get()
    if f_after is not None and FREQ_MIN <= f_after <= FREQ_MAX:
        print(f"  ✓ Locked — f={f_after:.6f} THz  [{lock_state}]")
        _buf['status'][0] = "Locked"
        return True
    print(f"  ✗ Lock failed — f={f_after}  [{lock_state}]")
    return False


# ═════════════════════════════════════════════════════════════════════════════
# ML acquire — single candidate attempt
# ═════════════════════════════════════════════════════════════════════════════

def try_candidate(dlc, writer, csvfile, cand, ci, total):
    print(f"\n[ML] Candidate {ci+1}/{total}: "
          f"I={cand['current_mA']:.2f} mA, V={cand['piezo_V']:.2f} V "
          f"(MHF {cand['mhf_piezo_min_V']:.2f}–{cand['mhf_piezo_max_V']:.2f} V)")
    _buf['status'][0] = (f"ML → {cand['current_mA']:.2f} mA / "
                         f"{cand['piezo_V']:.2f} V")

    lock = dlc.laser1.dl.lock
    if lock.lock_enabled.get():
        lock.lock_enabled.set(False)
        time.sleep(0.3)

    dlc.laser1.dl.cc.current_set.set(float(cand["current_mA"]))
    dlc.laser1.dl.pc.voltage_set.set(float(cand["piezo_V"]))

    target_v = float(cand["piezo_V"])
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        time.sleep(0.2)
        if abs(dlc.laser1.dl.pc.voltage_act.get() - target_v) < 0.5:
            break
    time.sleep(SETTLE_AFTER_SET)

    init_freq = get_frequency()
    if init_freq is not None:
        if init_freq > FREQ_TARGET:
            headroom = target_v - cand["mhf_piezo_min_V"]
        else:
            headroom = cand["mhf_piezo_max_V"] - target_v
        if headroom < 1.0:
            print(f"  Skipping: only {headroom:.2f} V headroom "
                  f"(offset {(init_freq - FREQ_TARGET)*1e6:+.0f} MHz).")
            return False

    return piezo_sweep_and_lock(dlc, writer, csvfile,
                                mhf_min=cand["mhf_piezo_min_V"],
                                mhf_max=cand["mhf_piezo_max_V"])


# ═════════════════════════════════════════════════════════════════════════════
# Worker thread  (all hardware I/O lives here)
# ═════════════════════════════════════════════════════════════════════════════

def _worker(dlc, writer, csvfile, cs_map):
    _buf['t0'][0] = time.monotonic()

    state = {'cands': [], 'cand_idx': 0}

    def _fetch():
        _buf['status'][0] = "ML predict..."
        print("\n[ML] Fetching candidates from consensus map...")
        _, new_cands = find()
        if new_cands:
            print(f"[ML] {len(new_cands)} candidates.")
        else:
            print("[ML] No candidates found.")
            _buf['status'][0] = "ML: no candidates"
        state['cands'] = new_cands or []
        state['cand_idx'] = 0

    def _acquire(reset=False):
        if reset:
            state['cand_idx'] = 0
        while True:
            if state['cand_idx'] >= len(state['cands']):
                _fetch()
                if not state['cands']:
                    time.sleep(5.0)
                    continue
            ci = state['cand_idx']
            state['cand_idx'] += 1
            if try_candidate(dlc, writer, csvfile,
                             state['cands'][ci], ci, len(state['cands'])):
                return state['cands'][ci]

    # ── Centering helper ──────────────────────────────────────────────────────

    def _apply_centering(locked_cand, current_mA, piezo):
        """
        Nudge current toward I*(MHF_centre) to pull the piezo back to the
        middle of the plateau.  The wavemeter PID then naturally recentres.
        Returns True if a nudge was applied.
        """
        if cs_map is None or locked_cand is None:
            return False

        mhf_center = ((locked_cand["mhf_piezo_min_V"]
                       + locked_cand["mhf_piezo_max_V"]) / 2.0)
        error_V = piezo - mhf_center

        if abs(error_V) <= CENTERING_DEADBAND_V:
            return False

        target_I = cs_map.get_target_current(mhf_center)
        if target_I is None:
            print(f"  [CENTER] No I*(V) data at V={mhf_center:.2f} V — skipping.")
            return False

        delta_I = target_I - current_mA
        nudge   = float(np.clip(delta_I * CENTERING_GAIN,
                                -CENTERING_MAX_NUDGE_MA, CENTERING_MAX_NUDGE_MA))
        if abs(nudge) < 0.005:
            return False

        new_I = float(np.clip(current_mA + nudge, I_MIN, I_MAX))
        print(f"\n[CENTER] Piezo {piezo:.2f} V  centre {mhf_center:.2f} V  "
              f"error {error_V:+.2f} V\n"
              f"         I: {current_mA:.4f} → {new_I:.4f} mA  "
              f"(nudge {nudge:+.4f} mA)")
        _buf['status'][0] = f"Centering: I→{new_I:.3f} mA"
        dlc.laser1.dl.cc.current_set.set(new_I)
        return True

    # ── Initial acquire ───────────────────────────────────────────────────────

    _fetch()
    locked_cand      = _acquire()
    relock_attempts  = 0
    lock_acquired_at = time.monotonic()
    last_center_t    = time.monotonic()

    # ── Main monitor loop ─────────────────────────────────────────────────────

    while True:
        time.sleep(DELAY)
        freq = get_frequency()

        try:
            piezo      = dlc.laser1.dl.pc.voltage_act.get()
            current_mA = dlc.laser1.dl.cc.current_act.get()
            temp_c     = dlc.laser1.dl.tc.temp_act.get()
            lock_state = dlc.laser1.dl.lock.state_txt.get()
            lock_on    = dlc.laser1.dl.lock.lock_enabled.get()
        except Exception as e:
            print(f"[WORKER] Toptica error: {e}")
            continue

        if freq is None:
            continue

        elapsed = time.monotonic() - _buf['t0'][0]
        in_win  = FREQ_MIN <= freq <= FREQ_MAX
        _record(writer, csvfile, elapsed, freq, piezo, current_mA,
                temp_c, lock_state, in_win)

        offset_mhz = (freq - FREQ_TARGET) * 1e6
        print(f"t={elapsed:7.1f}s | f={freq:.6f} ({offset_mhz:+7.2f} MHz) | "
              f"V={piezo:.3f} | I={current_mA:.3f} mA | {lock_state}")

        if in_win and lock_on:
            relock_attempts  = 0
            _buf['status'][0] = "Locked"

            now = time.monotonic()
            if (now - lock_acquired_at >= CENTERING_MIN_LOCK_S
                    and now - last_center_t >= CENTERING_INTERVAL_S):
                _apply_centering(locked_cand, current_mA, piezo)
                last_center_t = now

        else:
            in_mhf = (locked_cand is not None
                      and locked_cand["mhf_piezo_min_V"] <= piezo
                      <= locked_cand["mhf_piezo_max_V"])
            large_hop = abs(freq - FREQ_TARGET) > LARGE_HOP_THZ

            if in_mhf and not large_hop and relock_attempts < MAX_RELOCK_ATTEMPTS:
                relock_attempts += 1
                _buf['status'][0] = (f"Relock {relock_attempts}/"
                                     f"{MAX_RELOCK_ATTEMPTS}...")
                print(f"\n[RELOCK] In MHF ({locked_cand['mhf_piezo_min_V']:.2f}–"
                      f"{locked_cand['mhf_piezo_max_V']:.2f} V). "
                      f"Attempt {relock_attempts}/{MAX_RELOCK_ATTEMPTS}.")
                if piezo_sweep_and_lock(dlc, writer, csvfile,
                                        mhf_min=locked_cand["mhf_piezo_min_V"],
                                        mhf_max=locked_cand["mhf_piezo_max_V"]):
                    relock_attempts  = 0
                    lock_acquired_at = time.monotonic()
            else:
                reason = (f"large mode hop ({(freq-FREQ_TARGET)*1e6:+.0f} MHz)"
                          if large_hop else
                          "outside MHF" if not in_mhf else
                          f"{MAX_RELOCK_ATTEMPTS} relock attempts failed")
                print(f"\n[RELOCK] {reason} — restarting from candidate 1.")
                _buf['status'][0] = "ML relock..."
                relock_attempts  = 0
                locked_cand      = _acquire(reset=True)
                lock_acquired_at = time.monotonic()


# ═════════════════════════════════════════════════════════════════════════════
# Plot  (animation reads from shared buffers, never touches hardware)
# ═════════════════════════════════════════════════════════════════════════════

def _start_plot():
    fig, (ax_f, ax_p, ax_c) = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    fig.canvas.manager.set_window_title(
        f"ML Lock v3 — Ch{CHANNEL}  {LASER_NAME}"
    )
    fig.subplots_adjust(left=0.09, right=0.96, top=0.94, bottom=0.07, hspace=0.35)

    f_line, = ax_f.plot([], [], 'b-', linewidth=1.0)
    ax_f.axhspan(-FREQ_WINDOW * 1e6, FREQ_WINDOW * 1e6,
                 color='green', alpha=0.15, label='Target window')
    ax_f.axhline(0, color='green', linestyle='--', alpha=0.5)
    ax_f.set_ylabel("Freq offset (MHz)")
    ax_f.set_title(f"Ch{CHANNEL} — offset from {FREQ_TARGET:.6f} THz")
    ax_f.grid(True, alpha=0.3)
    ax_f.legend(loc='upper right', fontsize=8)
    status_text = ax_f.text(
        0.01, 0.97, _buf['status'][0],
        transform=ax_f.transAxes, fontsize=9, va='top',
        color='darkblue', fontweight='bold',
    )

    p_line, = ax_p.plot([], [], 'm-', linewidth=1.0)
    ax_p.set_ylabel("Piezo (V)")
    ax_p.set_title("Piezo voltage")
    ax_p.grid(True, alpha=0.3)

    c_line, = ax_c.plot([], [], 'c-', linewidth=1.0)
    ax_c.set_ylabel("Current (mA)")
    ax_c.set_xlabel("Time elapsed (s)")
    ax_c.set_title("Diode current")
    ax_c.grid(True, alpha=0.3)

    def update(_frame):
        status_text.set_text(_buf['status'][0])
        t = list(_buf['times'])
        if not t:
            return f_line, p_line, c_line, status_text

        f_line.set_data(t, list(_buf['freq']))
        ax_f.relim(); ax_f.autoscale_view()

        n = min(len(t), len(_buf['piezo']))
        p_line.set_data(t[-n:], list(_buf['piezo'])[-n:])
        ax_p.relim(); ax_p.autoscale_view()

        n = min(len(t), len(_buf['current']))
        c_line.set_data(t[-n:], list(_buf['current'])[-n:])
        ax_c.relim(); ax_c.autoscale_view()

        return f_line, p_line, c_line, status_text

    fig._ani = animation.FuncAnimation(
        fig, update, interval=int(DELAY * 1000),
        blit=False, cache_frame_data=False,
    )
    plt.show()


def main():
    try:
        wlmData.LoadDLL(DLL_PATH)
    except Exception:
        sys.exit(f"Couldn't load wavemeter DLL: {DLL_PATH}")
    if wlmData.dll.GetWLMCount(0) == 0:
        sys.exit("No wlmServer instance found")

    try:
        cs_map = CurrentSearchMap()
    except (FileNotFoundError, ValueError) as e:
        print(f"[CENTER] Warning: {e}. Centering disabled.")
        cs_map = None

    print(f"Connecting to {LASER_NAME} at {TOPTICA_IP}:{TOPTICA_PORT}...")
    dlc = DLCpro(NetworkConnection(TOPTICA_IP, TOPTICA_PORT))

    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(
        LOG_DIR,
        f"ml_lock_ch{CHANNEL}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    )
    csvfile = open(log_path, "w", newline="")
    writer  = csv.writer(csvfile)
    writer.writerow(["timestamp_utc", "channel", "frequency_THz", "piezo_V",
                     "current_mA", "temp_C", "lock_state", "in_window"])
    print(f"Logging to: {log_path}")

    try:
        dlc.open()
        print(f"Connected (SN: {dlc.serial_number.get()})")

        t = threading.Thread(
            target=_worker, args=(dlc, writer, csvfile, cs_map), daemon=True
        )
        t.start()

        _start_plot()

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        try:
            csvfile.close()
        except Exception:
            pass
        try:
            dlc.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()