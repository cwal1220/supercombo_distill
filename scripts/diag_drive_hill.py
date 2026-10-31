"""Plan and lane shape on hills in a recorded drive (its own ModelState): elevation of the road at 25-35 m from the
model's ego lane lines' height (camera height - z, z down; > 0 the road ahead rises), then per elevation bin how bent
the plan is (rms of plan y minus a straight line through its 5-40 m points), the plan's lateral range at 40 m against
the lane centre, and the lane width ratio 30 m / 10 m. x forward, y right.

usage: .venv/bin/python scripts/diag_drive_hill.py <route dir> ..."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "Documents/ax630/edgepilot/tools/model"))
import recording_reader as rr  # noqa: E402

X = rr.X_IDXS
I10, I30 = np.argmin(abs(X - 10)), np.argmin(abs(X - 30))
EL = (X >= 25) & (X <= 35)
for route in map(Path, sys.argv[1:]):
    ms, cs = [], []
    for f in [f for f in rr.route_event_files(route) if not f.name.startswith("._")]:
        p = rr.chunk_states(f, rr.RECORD_MODEL_STATE, rr.RECORD_CONTROL_STATE)
        ms.append(p[rr.RECORD_MODEL_STATE][1].copy())
        cs.append(p[rr.RECORD_CONTROL_STATE][1].copy())
    M, C = np.concatenate(ms), np.concatenate(cs)
    ci = np.clip(np.searchsorted(C["timestamp_ns"], M["model_timestamp_ns"]) - 1, 0, len(C) - 1)
    v = C[ci]["ego_speed_kph"] / 3.6
    lines = (M["lane_probabilities"][:, 1] > .5) & (M["lane_probabilities"][:, 2] > .5)
    h = M["camera_height_m"]
    z = (M["lanes"][:, 1, :, 1] + M["lanes"][:, 2, :, 1]) / 2
    elev = h - z[:, EL].mean(1)
    plan = M["plan"]
    bent, end = [], []
    for k in range(len(M)):
        px, py = plan[k, :, 0], plan[k, :, 1]
        xs = np.linspace(5, 40, 15)
        if px[-1] < 40:
            bent.append(np.nan); end.append(np.nan); continue
        yq = np.interp(xs, px, py)
        a, b = np.polyfit(xs, yq, 1)
        bent.append(np.sqrt(np.mean((yq - (a * xs + b)) ** 2)))
        end.append(np.interp(40, px, py))
    bent, end = np.array(bent), np.array(end)
    l1, l2 = M["lanes"][:, 1, :, 0], M["lanes"][:, 2, :, 0]
    wr = (l2[:, I30] - l1[:, I30]) / (l2[:, I10] - l1[:, I10])
    ctr40 = (np.array([np.interp(40, X, a) for a in l1]) + np.array([np.interp(40, X, a) for a in l2])) / 2
    mov = (v > 8) & lines & (M["valid"] == 1)
    print(f"{route.name}: moving with lines {mov.sum()} frames")
    print("elev (m)       frames  plan bend rms 5-40 m (cm)  |plan-centre| at 40 m (cm)  lane width 30/10 m")
    for lo, hi in ((-9, -.6), (-.6, -.3), (-.3, -.1), (-.1, .1), (.1, .3), (.3, .6), (.6, 9)):
        m = mov & (elev >= lo) & (elev < hi) & np.isfinite(bent)
        if m.sum() < 30:
            continue
        print(f"[{lo:+.1f},{hi:+.1f})  {m.sum():6d}   {np.median(bent[m]) * 100:6.1f} (p90 {np.percentile(bent[m], 90) * 100:5.1f})"
              f"      {np.median(np.abs(end[m] - ctr40[m])) * 100:6.1f}            {np.median(wr[m]):.3f}")
