# LaserRelock — Technical Documentation

## Overview

This project implements an autonomous frequency-locking and auto-relocking system for a 689 nm diode laser (Toptica DLC Pro) used in a strontium atomic physics experiment. The laser must be maintained at a precise frequency (434.829035 THz) continuously, and must recover automatically when it unlocks due to mode hops or environmental perturbations.

The frequency reference is a HighFinesse wavemeter. The laser is locked by the wavemeter's internal PID controller, which actuates the laser's piezo voltage to hold the measured frequency at the target. The software layer described here operates around this PID: it finds good operating points, sweeps the laser onto target before engaging the PID lock, and monitors continuously to detect and recover from unlock events.

---

## Hardware

| Component | Details |
|-----------|---------|
| Laser | Toptica DLC Pro, 689 nm diode, LASER_NAME = `testbed689` |
| Wavemeter | HighFinesse WS-U, interface via `wlmData.dll` / `libwlmData.so` |
| Wavemeter channel | Channel 3 |
| Control interface | Toptica SDK (`lasersdk.dlcpro.v2_0_3`) over TCP, IP `172.29.13.247:1998` |
| Control axes | Piezo voltage (V) — fast axis, PID-controlled; Diode current (mA) — slow axis, manually adjusted |

**Key laser parameters:**
- Operating current range: 89.5–92.0 mA (below 89.5 mA the laser intensity is too low for useful operation)
- Piezo voltage range: 20–60 V
- Target frequency: 434.829035 THz (updated from 434.829040 after wavemeter drift)
- Lock window: ±3 MHz around target

**Toptica SDK paths used:**
```
dlc.laser1.dl.cc.current_set / current_act   — diode current
dlc.laser1.dl.pc.voltage_set / voltage_act   — piezo voltage
dlc.laser1.dl.lock.lock_enabled              — enable/disable PID lock
dlc.laser1.dl.lock.state_txt                 — lock status string
dlc.laser1.dl.tc.temp_act                    — diode temperature
```

**Wavemeter API:**
```python
wlmData.dll.GetFrequencyNum(channel, 0.0)   # returns freq in THz; ≤0 means no signal
```

---

## Repository Structure

```
LaserRelock/
├── auto_relock.py             # v1: simple piezo sweep, no ML
├── auto_relock_v2.py          # v2: improved relock logic
├── auto_relock_v3.py          # v3: mode_search integration
├── auto_relock_v4.py          # v4: reads latest.json from mode_search
│
├── mode_search.py             # 2D scan: outer=current, inner=piezo
├── current_search.py          # 2D scan: outer=piezo, inner=current
├── plot_mode_search.py        # Visualise mode_search results
├── plot_relock_log.py         # Visualise relock CSV logs
├── analyze_mode_stability.py  # Post-hoc stability analysis
│
├── FreqLogger.py              # Standalone frequency logger
├── check_lock.py              # Quick lock status check
├── calibrate_wm.py            # Wavemeter calibration utility
│
├── mode_search/               # Scan data (CSV, JSON, PNG per run)
├── current_search/            # Scan data (CSV, JSON, PNG per run)
├── relock_log/                # Lock session logs (CSV)
│
└── ml_mode_finder/
    ├── train.py               # Build kNN index from mode_search data
    ├── find_mode.py           # kNN consensus map + candidate finder
    ├── lock.py                # First ML lock script
    ├── lock_v2.py             # Production ML lock with adaptive sweep
    ├── lock_v3.py             # v2 + slow current centering loop
    └── checkpoints/
        └── scan_index.npz     # Trained kNN index
```

---

## Stage 1 — Manual Piezo Sweep Locking (auto_relock v1–v4)

### What these do

The early scripts (`auto_relock.py` through `auto_relock_v4.py`) implement a straightforward relock strategy with no machine learning:

1. Monitor wavemeter frequency on channel 3
2. If frequency leaves the ±3 MHz window around target, disable the PID lock
3. Step the piezo voltage toward the target frequency (10 mV steps, 200 ms settle per step)
4. Once the frequency re-enters the window, confirm stability (3–5 consecutive in-window readings), then re-engage the PID lock
5. If the piezo hits its safety bounds without finding the target, stop and alert

### Limitations discovered

- **No knowledge of mode structure:** the sweep walks the piezo blindly; if the laser has mode-hopped to a completely different longitudinal mode, it can never reach the target frequency by adjusting piezo alone — the current needs to change
- **Voltage bounds were arbitrary:** safety limits (e.g. 25–35 V) were not informed by where mode-hop-free regions actually exist
- **No persistence across large hops:** after a large mode hop, the script had no recovery strategy beyond exhausting the voltage range

---

## Stage 2 — Mode Structure Mapping

To overcome the blind-sweep problem, two complementary 2D scan scripts were developed to characterise the laser's mode structure.

### mode_search.py

**What it does:** Outer loop steps through **current** values (89.0–92.0 mA, 0.1 mA steps). Inner loop sweeps **piezo voltage** (20–60 V, 0.1 V steps) at each current. Records frequency at each (current, piezo) point.

**What it finds:** At each fixed current, it detects **mode-hop-free (MHF) regions** in the piezo axis — contiguous voltage ranges where the frequency changes smoothly without a >1 GHz jump. For MHF regions that span the target frequency, it records the operating point as a candidate.

**Output per run:** `mode_search/search_YYYYMMDD_HHMMSS.{csv,json,png}`

The JSON stores the best candidate (highest MHF width) and all candidates found. A `latest.json` symlink is kept up to date.

**Why this scan matters:** It reveals the 2D landscape of (current, piezo) → frequency and identifies where mode-hop-free operation exists. This data is the training input for the ML model.

### current_search.py

**What it does:** The inverse scan. Outer loop steps through **piezo voltages** (30–60 V, 0.5 V steps). Inner loop sweeps **current** (89.0–92.0 mA, 0.1 mA steps) at each piezo. Detects MHF regions in the **current axis**.

**What it finds:** For each fixed piezo voltage, it identifies stable current ranges and interpolates the exact current where the target frequency is crossed. Candidates are stored as `(piezo_V, current_mA)` pairs — the I*(V) curve.

**Output per run:** `current_search/search_YYYYMMDD_HHMMSS.{csv,json,png}`

**Why this scan matters:** It provides the complementary axis of the mode structure. Critically, the I*(V) curve it produces is used in `lock_v3.py` to recenter the piezo within its MHF plateau (see Stage 4).

---

## Stage 3 — ML-Assisted Locking (lock_v2.py)

### The core problem with deterministic locking

A single mode_search scan gives one operating point, but the laser's mode structure **drifts week to week** — MHF region boundaries can shift by >1 V. A point that worked last week may be in a mode-hop boundary today. Using only the most recent scan is brittle; using all historical scans uniformly is wrong because old data is stale.

### kNN consensus map

`ml_mode_finder/train.py` aggregates all `mode_search/search_*.csv` files into a single numpy archive (`scan_index.npz`) containing all (current_mA, piezo_V, frequency_THz) observations with their scan IDs.

`ml_mode_finder/find_mode.py` implements a **k-nearest-neighbour consensus predictor** (`ConsensusMap`):

- **Temporal filtering:** uses only the most recent N scans (`RECENT_SCANS = 5`) — old data is discarded to track laser drift
- **Normalised distance:** current and piezo axes are normalised to [0, 1] before computing distances, so both axes contribute equally
- **Prediction:** for a query point (I, V), finds the k=20 nearest historical observations and returns their **median frequency** plus a **stability flag** (IQR < 1.5 GHz)

The IQR stability criterion is key: near a mode-hop boundary, different historical scans land in different modes, producing high IQR → flagged as unstable. Inside a stable MHF plateau, all neighbours agree → low IQR → flagged stable.

### Candidate finding (find_mode.py)

`find()` builds a 2D grid over (current, piezo) space and queries the `ConsensusMap` at every grid point. It then:

1. For each current row, walks along the piezo axis and identifies **stable, smooth segments** (consecutive stable cells with <1 GHz inter-cell jumps) — these are the MHF regions
2. Filters segments by width: `2.0 V ≤ width ≤ 15.0 V` (the upper bound removes kNN smoothing artefacts that produce falsely wide "regions" spanning the full 20–60 V range)
3. Finds segments that **cross the target frequency** and interpolates the exact piezo voltage
4. Returns all valid candidates sorted by (current ascending, piezo ascending)

**Why sort by current ascending?** Lower current generally means less thermal load; the lowest viable current is preferred.

**Current minimum set to 89.5 mA** (not 89.0 mA) because below 89.5 mA the laser intensity is insufficient for reliable wavemeter readings.

### lock_v2.py — system architecture

The script runs two threads:

| Thread | Role |
|--------|------|
| Worker thread | All hardware I/O: frequency reading, piezo/current control, lock engagement, CSV logging |
| Main thread | matplotlib animation for live plotting; blocks on `plt.show()` |

Data flows from worker → main via `_buf` (shared dict of `deque`s). `deque` append/pop is atomic under Python's GIL, so no explicit locking is needed.

### Acquisition flow

```
main()
 └─ _worker()
     ├─ _fetch()          ← runs find(), stores candidate list in state{}
     └─ _acquire()        ← iterates candidates until one locks
         └─ try_candidate()
             ├─ set current + piezo to candidate values
             ├─ wait for piezo to settle (polls voltage_act)
             ├─ headroom check: skip if <1 V room to sweep toward target
             └─ piezo_sweep_and_lock()
                 ├─ determine sweep direction (freq ABOVE or BELOW target)
                 ├─ adaptive step: 100 mV coarse (>50 MHz offset), 10 mV fine
                 ├─ abort if sweep hits MHF bounds
                 ├─ stability check: 5 consecutive in-window readings
                 └─ engage lock, verify with post-lock frequency read
```

**Candidate list persistence:** `find()` is expensive (kNN grid query). The candidate list is computed once and stored in `state{}`. On an unlock event, the script first tries **candidate 1** from the existing list before trying subsequent candidates. `find()` is only re-run when the entire list has been exhausted. This prevents redundant predictions on every relock.

### Unlock detection and recovery

The monitor loop runs at 200 ms intervals. On each iteration it checks `in_win = FREQ_MIN ≤ freq ≤ FREQ_MAX` and `lock_on`. If either is false:

**Type 1 — normal unlock (drift, piezo still in MHF):**
- Condition: `in_mhf = True` and `|freq - target| ≤ 5 GHz`
- Action: bounded piezo sweep within the known MHF region, up to `MAX_RELOCK_ATTEMPTS = 3` times
- If all 3 attempts fail → escalate to Type 2

**Type 2 — mode hop (outside MHF or large offset):**
- Condition: `not in_mhf` OR `|freq - target| > 5 GHz (LARGE_HOP_THZ)`
- Action: restart from candidate 1 in the current list (reset index, don't re-run find)
- Only runs find() again if all candidates fail

The 5 GHz `LARGE_HOP_THZ` guard prevents wasting time on a bounded sweep when the laser is many GHz away — it jumps straight to a full re-acquisition.

### CSV log format

```
timestamp_utc, channel, frequency_THz, piezo_V, current_mA, temp_C, lock_state, in_window
```

Written on every 200 ms monitor cycle and every sweep step. Used for post-hoc analysis with `plot_relock_log.py`.

### Live plot

Three subplots updated at 200 ms via `FuncAnimation`:
- Frequency offset from target (MHz)
- Piezo voltage (V)
- Diode current (mA)

Status text overlay shows current state (Locked / Sweeping / ML predict / Centering).

---

## Stage 4 — Piezo Centering (lock_v3.py)

### The problem

Even when locked, the wavemeter PID slowly drifts the piezo voltage to compensate for laser aging, thermal drift, and environmental perturbations. If the piezo walks toward the **edge** of the MHF plateau and crosses it, the laser mode-hops — a Type 2 unlock. This is avoidable if the piezo can be kept near the **centre** of the plateau.

### The slow outer loop

`lock_v3.py` adds a current centering loop that runs inside the worker thread every 30 seconds (after 60 seconds of continuous lock). The strategy exploits the complementary relationship between piezo and current:

- The wavemeter PID holds **frequency = target** by adjusting piezo
- Changing the **current** shifts the entire frequency landscape
- The PID responds to a current change by moving the piezo to compensate
- Net effect: changing current causes the steady-state piezo position to shift

This means: if we set the current to the value where the target frequency is hit at the **MHF centre voltage**, the PID will naturally drive the piezo to that centre.

### The I*(V) curve

At startup, `CurrentSearchMap` loads the 3 most recent `current_search/search_*.json` files. From the `all_candidates` list in each JSON (each entry is a `(piezo_V, current_mA)` pair where the target frequency is hit), it builds a lookup table keyed by piezo voltage. Multiple scans at the same piezo voltage are averaged (median).

The result is I*(V): a sparse curve that maps any piezo voltage to the current needed to hit the target frequency there. `np.interp` handles interpolation between scan points.

### Centering decision logic

```python
mhf_center = (mhf_piezo_min + mhf_piezo_max) / 2
error_V    = piezo_act - mhf_center

if |error_V| > CENTERING_DEADBAND_V (1.5 V):
    target_I = I*(mhf_center)          # from CurrentSearchMap
    delta_I  = target_I - current_act
    nudge    = clip(delta_I × 0.5, ±0.15 mA)
    apply nudge to current_set
```

The gain of 0.5 means 50% of the correction is applied per step, preventing overshoot. The ±0.15 mA hard cap prevents large sudden current changes that could disturb the lock. The 1.5 V deadband prevents unnecessary nudges when the piezo is already near centre.

### Parameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `CENTERING_INTERVAL_S` | 30 s | How often to check centering |
| `CENTERING_MIN_LOCK_S` | 60 s | Don't center until locked this long (avoid disturbing fresh locks) |
| `CENTERING_DEADBAND_V` | 1.5 V | Minimum piezo error to trigger a nudge |
| `CENTERING_GAIN` | 0.5 | Fraction of full correction per step |
| `CENTERING_MAX_NUDGE_MA` | 0.15 mA | Hard cap per nudge |
| `RECENT_SCANS_CS` | 3 | Number of recent current_search scans to use |

If no `current_search/` JSON files are found, `cs_map = None` and centering is silently disabled — the script falls back to identical behaviour as `lock_v2.py`.

---

## Wavemeter Drift

The HighFinesse wavemeter is calibrated against an ultrastable reference laser. When this reference laser unlocks (which happens periodically), the wavemeter loses its calibration reference and drifts — observed as ~5 MHz shift over a weekend. This manifests as the locked laser frequency gradually moving away from the true target.

**Current mitigation:** manually update `FREQ_TARGET` in `find_mode.py` and `current_search.py` when drift is observed (e.g. 434.829040 → 434.829035).

**Planned mitigation:** monitor the reference laser's wavemeter channel; when it returns to its known frequency, trigger `wlmData.dll.Calibration()` to recalibrate automatically.

---

## Data Flow Summary

```
mode_search.py          current_search.py
      │                        │
      ▼                        ▼
mode_search/*.csv       current_search/*.json
      │                        │
      ▼                        ▼
ml_mode_finder/         CurrentSearchMap
train.py                (I*(V) curve)
      │                        │
      ▼                        │
scan_index.npz                 │
      │                        │
      ▼                        │
ConsensusMap (kNN)             │
      │                        │
      ▼                        ▼
find_mode.py:find()     lock_v3.py centering loop
      │
      ▼
candidate list [(I,V,MHF_bounds), ...]
      │
      ▼
lock_v3.py:_acquire()
      │
      ▼
piezo_sweep_and_lock()
      │
      ▼
lock engaged → monitor loop → relock on unlock
```

---

## Running the System

### One-time setup

1. **Run mode_search** to map the laser's mode structure:
   ```bash
   python mode_search.py
   ```

2. **Run current_search** to build the I*(V) curve for centering:
   ```bash
   python current_search.py
   ```

3. **Train the kNN model** on all accumulated mode_search scans:
   ```bash
   python -m ml_mode_finder.train
   ```

### Starting the lock

```bash
python3 -m ml_mode_finder.lock_v3
```

At startup this will:
- Load the wavemeter DLL
- Load the I*(V) curve from the 3 most recent current_search JSONs
- Connect to the Toptica DLC Pro
- Run `find()` to predict the best (current, piezo) operating points
- Attempt acquisition through the candidate list
- Enter the monitor loop once locked
- Begin the centering loop after 60 s of continuous lock

### Periodic maintenance

| Task | Frequency | Command |
|------|-----------|---------|
| New mode_search scan | Weekly or after laser service | `python mode_search.py` |
| New current_search scan | After mode_search, or after large drift | `python current_search.py` |
| Retrain kNN model | After new mode_search scans | `python -m ml_mode_finder.train` |
| Update FREQ_TARGET | When wavemeter drifts | Edit `FREQ_TARGET` in `find_mode.py` and `current_search.py` |

### Visualising log data

```bash
python plot_relock_log.py
```

Edit the `REMOVE` list at the top of `plot_relock_log.py` to mask known-bad time segments (e.g. manual interventions) before plotting.

---

## Key Design Decisions

**Why kNN and not a neural network?**
Near mode-hop boundaries, the (I, V) → frequency map is multi-valued across scans (the boundary position drifts). A smooth MLP would average across modes and produce meaningless predictions in those regions. kNN preserves the multi-modality: boundary queries produce high IQR (flagged unstable), interior queries produce low IQR (reliable). See comment in `train.py`.

**Why filter to the 5 most recent scans?**
The laser's mode structure drifts week to week. Using all historical scans would allow stale data to outvote recent data. Using only the most recent 5 scans gives sufficient statistics while staying within the laser's current operating regime.

**Why bound the piezo sweep within MHF regions?**
An unbounded sweep can walk the piezo across a mode-hop boundary, causing the frequency to jump discontinuously. Bounding within the predicted MHF region ensures the sweep stays on the same longitudinal mode throughout.

**Why not re-run find() on every unlock?**
`find()` takes several seconds (kNN grid query over the full (I, V) space). On a simple drift unlock, the laser is still in the same mode — candidate 1 from the last prediction is almost always still valid. Re-running find() would add latency and is unnecessary. It only re-runs when the full candidate list is exhausted.

**Why use current as the centering actuator rather than temperature?**
Current responds in ~100 ms; temperature responds in tens of minutes. For a feedback loop checking every 30 s, current is fast enough to correct for drift while being slow compared to the PID timescale. Temperature would be more thermally stable but far too slow to track within a lock session.
