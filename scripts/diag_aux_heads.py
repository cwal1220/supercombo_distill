"""The outputs edgepilot uses besides plan / lanes / lead, for a student as the NPU runs it (S8) against the big
teacher, next to the shipped model's fp32 core on the same frames (labels' `small`): pose (online calibration from
the direction of travel, locationd), road_transform height, road edges, lead std / v, desire_state and gas_press.
Moving frames (v > 5 m/s) of the given units.

  calib pitch / yaw  -atan2(pose z, x), atan2(pose y, x) per frame (what calibration_online.cc averages): the mean
                     (bias, mrad) against the teacher's and the mean |difference|
  pose rot           |yaw rate (rot z)| difference, mrad/s; trans std y: median exp(log std) (rpy_certain gate)
  road height        road_transform z (m), mean and |difference|
  road edges         |y - teacher y| of both edges at 6-40 m (cm), median std
  desire_state       argmax agreement with the teacher and mean total-variation distance
usage: .venv/bin/python scripts/diag_aux_heads.py <ckpt> <unit> ..."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_student as E  # noqa: E402
from train import BIG  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OPL = {"pose": (87, 99), "road_transform": (105, 117), "road_edges": (653, 917), "lead": (917, 1061), "meta": (0, 55),
       "desire_state": (2566, 2574)}
X = np.array([192.0 * (i / 32) ** 2 for i in range(33)])
NEAR = (X >= 6) & (X <= 40)
ck = sys.argv[1]
acc = {k: {"student": [], "shipped": [], "teacher": []} for k in ("pitch", "yaw", "rotz", "tstd", "height", "edge", "estd", "ds", "dsarg", "gas")}
for u in sys.argv[2:]:
    d = ROOT / "data/ep_label" / u
    lab = np.load(d / "labels.npz")
    n = len(lab["big"])
    even = np.arange(4, n - n % 2, 2)
    sel = lab["v_ego"][even] > 5
    raw = E.run_student(Path(ck), d, lab, even, "mps", True, raw=True)
    src = {"student": raw,
           "shipped": {k: lab["small"][even, a:b] for k, (a, b) in OPL.items()},
           "teacher": {k: lab["big"][even, a:b] for k, (a, b) in BIG.items() if k in OPL}}
    for nm, r in src.items():
        p = np.asarray(r["pose"])[sel]
        acc["pitch"][nm].append(-np.arctan2(p[:, 2], p[:, 0]))
        acc["yaw"][nm].append(np.arctan2(p[:, 1], p[:, 0]))
        acc["rotz"][nm].append(p[:, 5])
        acc["tstd"][nm].append(np.exp(p[:, 7]))
        acc["height"][nm].append(np.asarray(r["road_transform"])[sel][:, 2])
        e = np.asarray(r["road_edges"])[sel]
        acc["edge"][nm].append(e[:, :132].reshape(-1, 2, 33, 2)[:, :, NEAR, 0])
        acc["estd"][nm].append(np.exp(e[:, 132:].reshape(-1, 2, 33, 2)[:, :, 0, 0]))
        ds = np.asarray(r["desire_state"])[sel]
        ds = np.exp(ds - ds.max(1, keepdims=True)); ds /= ds.sum(1, keepdims=True)
        acc["ds"][nm].append(ds)
        m = np.asarray(r["meta"])[sel]
        acc["gas"][nm].append(1 / (1 + np.exp(-m[:, 31 + 4])))
    print(u, "done", flush=True)
A = {k: {nm: np.concatenate(v) for nm, v in d.items()} for k, d in acc.items() if d["teacher"]}
t = lambda k: A[k]["teacher"]
print(f"{len(t('pitch'))} moving frames, {len(sys.argv) - 2} units; student {ck}")
print(f"{'':28s} {'student':>10s} {'shipped':>10s} {'teacher':>10s}")
for k, name, s in (("pitch", "calib pitch mean (mrad)", 1e3), ("yaw", "calib yaw mean (mrad)", 1e3), ("height", "road height mean (m)", 1)):
    print(f"{name:28s} " + " ".join(f"{np.mean(A[k][nm]) * s:10.3f}" for nm in ("student", "shipped", "teacher")))
for k, name, s in (("pitch", "|pitch - teacher| (mrad)", 1e3), ("yaw", "|yaw - teacher| (mrad)", 1e3), ("rotz", "|yaw rate - teacher| (mrad/s)", 1e3),
                   ("height", "|height - teacher| (cm)", 100), ("edge", "|edge y - teacher| (cm)", 100), ("gas", "|gas2 - teacher|", 1)):
    print(f"{name:28s} " + " ".join(f"{np.nanmean(np.abs(A[k][nm] - t(k))) * s:10.3f}" for nm in ("student", "shipped")))
print(f"{'trans std y median':28s} " + " ".join(f"{np.median(A['tstd'][nm]):10.4f}" for nm in ("student", "shipped", "teacher")))
print(f"{'edge std median (m)':28s} " + " ".join(f"{np.median(A['estd'][nm]):10.4f}" for nm in ("student", "shipped", "teacher")))
tv = lambda nm: 0.5 * np.abs(A["ds"][nm] - t("ds")).sum(1)
arg = lambda nm: (A["ds"][nm].argmax(1) == t("ds").argmax(1)).mean() * 100
lc = t("ds").argmax(1) != 0
print(f"{'desire_state TV distance':28s} " + " ".join(f"{tv(nm).mean():10.4f}" for nm in ("student", "shipped")))
print(f"{'desire argmax agree (%)':28s} " + " ".join(f"{arg(nm):10.2f}" for nm in ("student", "shipped")) + f"   (teacher non-none frames {lc.sum()})")
