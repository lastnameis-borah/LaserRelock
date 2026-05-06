"""
Pick the lowest-current, lowest-piezo MHF region that crosses target frequency.

Uses a kNN consensus map built from historical mode_search scans:
  - For each query (I, V): find k nearest historical observations,
    return median freq plus a "stable" flag (= small IQR).
  - On a fine grid: find per-current segments that are stable AND smooth
    (no inter-cell freq jumps > MODE_HOP_THRESHOLD), then pick the one
    that crosses target freq with smallest current (tie-break: piezo).

Run: python -m ml_mode_finder.find_mode
"""
import os

import numpy as np
from scipy.spatial import cKDTree

INDEX_PATH = "/home/artiq/LaserRelock/ml_mode_finder/checkpoints/scan_index.npz"

# Hardware ranges
I_MIN, I_MAX = 89.5, 92.0
V_MIN, V_MAX = 20.0, 60.0

# Target frequency
FREQ_TARGET = 434.829040
FREQ_WINDOW = 0.000005

# kNN consensus
RECENT_SCANS = 5             # use only the latest N scans (laser drifts week-to-week)
K_NEIGHBORS = 20
STABILITY_IQR_THZ = 0.0015   # 1.5 GHz — even MHF cells drift ~1 GHz across recent scans

# MHF region detection on consensus grid
GRID_CURRENT_STEP = 0.1
GRID_PIEZO_STEP = 0.1
MODE_HOP_THRESHOLD_THZ = 0.001  # 1 GHz between adjacent cells = mode hop
MIN_MHF_WIDTH_V = 2.0
MAX_MHF_WIDTH_V = 15.0  # wider than this is likely a kNN artifact


class ConsensusMap:
    """kNN-based predictor over historical mode_search data.

    Filters to the most recent `recent_scans` scans because the laser's
    mode structure drifts week-to-week (boundaries can shift by >1 V).
    """

    def __init__(self, path=INDEX_PATH, recent_scans=RECENT_SCANS):
        if not os.path.exists(path):
            raise SystemExit(
                f"No index at {path}. Run python -m ml_mode_finder.train first."
            )
        d = np.load(path, allow_pickle=True)
        scan_id = d["scan_id"]
        all_ids = np.unique(scan_id)
        if recent_scans is not None and recent_scans < len(all_ids):
            keep_ids = all_ids[-recent_scans:]
            mask = np.isin(scan_id, keep_ids)
        else:
            keep_ids = all_ids
            mask = np.ones(len(scan_id), dtype=bool)

        self.curr = d["current_mA"][mask]
        self.piezo = d["piezo_V"][mask]
        self.freq = d["frequency_THz"][mask]
        self.n_scans_total = int(len(all_ids))
        self.n_scans_used = int(len(keep_ids))
        self.scan_names = [str(s) for s in d["scan_names"][keep_ids]]

        # Normalize each axis to [0, 1] so distances are comparable
        self.i_range = I_MAX - I_MIN
        self.v_range = V_MAX - V_MIN
        i_norm = (self.curr - I_MIN) / self.i_range
        v_norm = (self.piezo - V_MIN) / self.v_range
        self.tree = cKDTree(np.stack([i_norm, v_norm], axis=1))

    def predict(self, I_q, V_q, k=K_NEIGHBORS, iqr_threshold=STABILITY_IQR_THZ):
        """Predict (median_freq, stable_flag) for arrays of queries."""
        I_q = np.asarray(I_q, dtype=np.float64)
        V_q = np.asarray(V_q, dtype=np.float64)
        I_qn = (I_q - I_MIN) / self.i_range
        V_qn = (V_q - V_MIN) / self.v_range
        Q = np.stack([I_qn.ravel(), V_qn.ravel()], axis=1)
        _, idx = self.tree.query(Q, k=k)
        nf = self.freq[idx]  # (N, k)
        median = np.median(nf, axis=1)
        q1 = np.quantile(nf, 0.25, axis=1)
        q3 = np.quantile(nf, 0.75, axis=1)
        iqr = q3 - q1
        stable = iqr < iqr_threshold
        return median.reshape(I_q.shape), stable.reshape(I_q.shape)


def find_target_regions(freq_grid, stable_grid, currents, piezos,
                         freq_target=FREQ_TARGET,
                         hop_threshold=MODE_HOP_THRESHOLD_THZ,
                         min_width_V=MIN_MHF_WIDTH_V,
                         max_width_V=MAX_MHF_WIDTH_V):
    """Find per-current MHF segments crossing target. Returns list of dicts."""
    candidates = []
    for i, cur in enumerate(currents):
        f_line = freq_grid[i]
        s_line = stable_grid[i]
        # An "MHF cell" is stable AND has a smooth jump to its neighbor
        diffs = np.abs(np.diff(f_line))
        smooth = diffs < hop_threshold  # length len(f_line)-1
        # Cell j belongs to a smooth segment if it's stable and (smooth[j-1] or smooth[j])
        ok = s_line.copy()
        ok[1:] &= np.concatenate([[True], smooth])[1:] | s_line[1:] * False
        # Simpler: walk through, accumulating runs where consecutive cells
        # are stable AND inter-cell jump < threshold
        runs = []
        run_start = None
        for j in range(len(f_line)):
            if not s_line[j]:
                if run_start is not None and j - 1 - run_start >= 1:
                    runs.append((run_start, j - 1))
                run_start = None
                continue
            if run_start is None:
                run_start = j
            elif diffs[j - 1] >= hop_threshold:
                if j - 1 - run_start >= 1:
                    runs.append((run_start, j - 1))
                run_start = j
        if run_start is not None and len(f_line) - 1 - run_start >= 1:
            runs.append((run_start, len(f_line) - 1))

        for s, e in runs:
            p_start, p_end = piezos[s], piezos[e]
            width = p_end - p_start
            if width < min_width_V or width > max_width_V:
                continue
            seg_p = piezos[s:e + 1]
            seg_f = f_line[s:e + 1]
            if not (seg_f.min() <= freq_target <= seg_f.max()):
                continue
            d = seg_f - freq_target
            target_piezo = None
            for k in range(len(d) - 1):
                if d[k] * d[k + 1] <= 0:
                    p0, p1 = seg_p[k], seg_p[k + 1]
                    f0, f1 = seg_f[k], seg_f[k + 1]
                    if f1 != f0:
                        target_piezo = float(
                            p0 + (freq_target - f0) * (p1 - p0) / (f1 - f0)
                        )
                    else:
                        target_piezo = float(0.5 * (p0 + p1))
                    break
            if target_piezo is None:
                continue
            candidates.append({
                "current_mA": round(float(cur), 4),
                "piezo_V": round(target_piezo, 4),
                "mhf_width_V": round(float(width), 4),
                "mhf_piezo_min_V": round(float(p_start), 4),
                "mhf_piezo_max_V": round(float(p_end), 4),
            })
    return candidates


def find(model=None,
         current_step=GRID_CURRENT_STEP, piezo_step=GRID_PIEZO_STEP,
         freq_target=FREQ_TARGET):
    if model is None:
        model = ConsensusMap()
    currents = np.arange(I_MIN, I_MAX + 1e-6, current_step)
    piezos = np.arange(V_MIN, V_MAX + 1e-6, piezo_step)
    I_grid, V_grid = np.meshgrid(currents, piezos, indexing="ij")
    freq_grid, stable_grid = model.predict(I_grid, V_grid)
    cands = find_target_regions(freq_grid, stable_grid, currents, piezos,
                                 freq_target=freq_target)
    cands.sort(key=lambda c: (c["current_mA"], c["piezo_V"]))
    best = cands[0] if cands else None
    return best, cands


def main():
    model = ConsensusMap()
    print(f"Consensus from {model.n_scans_used} of {model.n_scans_total} scans "
          f"({len(model.freq):,} points)")
    print(f"  Scans used: {', '.join(model.scan_names)}")
    best, cands = find(model)
    if best is None:
        print("No target-crossing MHF region found in consensus map.")
        return
    print(f"\n{'='*60}")
    print("Selected operating point (lowest current -> lowest piezo)")
    print(f"{'='*60}")
    print(f"  Target freq:  {FREQ_TARGET:.6f} THz")
    print(f"  Current:      {best['current_mA']:.2f} mA")
    print(f"  Piezo:        {best['piezo_V']:.2f} V")
    print(f"  MHF region:   {best['mhf_piezo_min_V']:.2f} – "
          f"{best['mhf_piezo_max_V']:.2f} V "
          f"(width {best['mhf_width_V']:.2f} V)")
    print(f"\n  Total candidates: {len(cands)}")
    if len(cands) > 1:
        print(f"  Top 5 (current asc, piezo asc):")
        for c in cands[:5]:
            print(f"    I={c['current_mA']:.2f} mA  V={c['piezo_V']:.2f} V  "
                  f"MHF[{c['mhf_piezo_min_V']:.1f}, "
                  f"{c['mhf_piezo_max_V']:.1f}] (w={c['mhf_width_V']:.1f})")


if __name__ == "__main__":
    main()
