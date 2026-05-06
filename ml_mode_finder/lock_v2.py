"""
ML-assisted auto-relock

Run: python3 -m ml_mode_finder.lock_v2
"""
import csv
import os
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import wlmData
from toptica.lasersdk.dlcpro.v2_0_3 import DLCpro, NetworkConnection

from ml_mode_finder.find_mode import find, FREQ_TARGET, FREQ_WINDOW

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
STEP_COARSE_V      = 0.1   # 100 mV when out of COARSE_THRESH_MHZ from FREQ_TARGET
STEP_FINE_V        = 0.01   # 10 mV when within COARSE_THRESH_MHZ from FREQ_TARGET
COARSE_THRESH_MHZ  = 50.0  # switch to fine steps within this offset
STABLE_READINGS    = 5
VOLTAGE_MIN        = 20.0
VOLTAGE_MAX        = 60.0

# ─── Relock ──────────────────────────────────────────────────────────────────
MAX_RELOCK_ATTEMPTS = 3
LARGE_HOP_THZ = 0.005  # 5 GHz — offset beyond which a piezo sweep is futile

# ─── Output ──────────────────────────────────────────────────────────────────
LOG_DIR = "/home/artiq/LaserRelock/relock_log"


# ═════════════════════════════════════════════════════════════════════════════
# Shared state (worker thread writes, animation thread reads)
# ═════════════════════════════════════════════════════════════════════════════
# deque appends/pops are thread-safe under the GIL
_buf = {
    'times':   deque(),
    'freq':    deque(),    # MHz offset from FREQ_TARGET
    'piezo':   deque(),    # V
    'current': deque(),    # mA
    'status':  ['Starting up...'],  # single-element list so threads can mutate it
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
    """Append one sample to the plot buffers and CSV."""
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
# Piezo sweep + lock attempts
# ═════════════════════════════════════════════════════════════════════════════

def piezo_sweep_and_lock(dlc, writer, csvfile, mhf_min=None, mhf_max=None):
    """
    Sweep piezo toward target freq (10 mV steps), verify stability,
    engage lock.  Appends each step to shared buffers.
    Stops if the next step would leave [mhf_min, mhf_max].
    Returns True on success.
    """
    target = (FREQ_MIN + FREQ_MAX) / 2.0
    lock   = dlc.laser1.dl.lock

    # Fall back to hardware limits if no MHF bounds provided
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
                      f"(region {v_lo:.2f}–{v_hi:.2f} V). Giving up on this candidate.")
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
    """
    Set laser to one (I,V) candidate and sweep piezo to target freq.
    Returns True on success.
    """
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

def _worker(dlc, writer, csvfile):
    _buf['t0'][0] = time.monotonic()

    # Persistent candidate list — survives unlock events.
    # find() is only re-run when every candidate has been exhausted.
    state = {'cands': [], 'cand_idx': 0}

    def _fetch():
        """Run find() and reset the candidate list to index 0."""
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
        """
        Try candidates in order until one locks; return the locked candidate.
        If reset=True, restart from candidate 1 before trying.
        Calls find() again only when the whole list is exhausted.
        """
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

    # Initial lock
    _fetch()
    locked_cand = _acquire()
    relock_attempts = 0

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

        if not in_win or not lock_on:
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
                    relock_attempts = 0
            else:
                reason = (f"large mode hop ({(freq-FREQ_TARGET)*1e6:+.0f} MHz)"
                          if large_hop else
                          "outside MHF" if not in_mhf else
                          f"{MAX_RELOCK_ATTEMPTS} relock attempts failed")
                print(f"\n[RELOCK] {reason} — restarting from candidate 1 "
                      f"({len(state['cands'])} in list).")
                _buf['status'][0] = "ML relock from cand 1..."
                relock_attempts = 0
                locked_cand = _acquire(reset=True)
        else:
            relock_attempts = 0
            _buf['status'][0] = "Locked"


# ═════════════════════════════════════════════════════════════════════════════
# Plot  (animation reads from shared buffers, never touches hardware)
# ═════════════════════════════════════════════════════════════════════════════

def _start_plot():
    fig, (ax_f, ax_p, ax_c) = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    fig.canvas.manager.set_window_title(
        f"ML Lock v2 — Ch{CHANNEL}  {LASER_NAME}"
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

        # Start hardware worker in background
        t = threading.Thread(
            target=_worker, args=(dlc, writer, csvfile), daemon=True
        )
        t.start()

        # Open plot immediately — blocks until window is closed
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
