"""Lanes on hills: how a student's lane lines (as the NPU runs it, S8), the shipped model's (recorded), the big
teacher's and LD's (the lanes reference, projected onto a flat road at the camera height: ld_teacher.to_road) behave
with the road's relative elevation ahead.

  elevation   the teacher's lane-line height at 25-35 m (lines 1, 2): camera height - z; > 0 the road ahead rises
              relative to the car's own road plane (uphill ahead / sag), < 0 it falls (downhill ahead / crest)
  width ratio lane width (line 2 - line 1) at 30 m / at 10 m; ~1 on a straight road of constant width. A flat-road
              projection of a rising road puts its points too far: the width at 30 m grows by about h / (h - elev)
  jitter      |frame-to-frame change| of lines 1, 2 y at 30 m (cm, 0.1 s apart), moving frames
  vs LD/teach |y - LD y| and |y - teacher y| of lines 1, 2 at 6-40 m (cm)
  vs LDfix    |y - LD y| with LD put on the teacher's road surface (lane_geom.slope_fix), the reference train.py
              --ld-slope-fix trains on; the last line: all moving frames

The student's per-frame outputs are cached in outputs/diag_lane/<name>/<unit>.npz.

usage: .venv/bin/python scripts/diag_lane_grade.py --ckpt ckpt/mac_lrc/ema.pt --name lrc <unit> ..."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lane_geom import slope_fix  # noqa: E402
from train import BIG  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
X = np.array([192.0 * (i / 32) ** 2 for i in range(33)])
I10, I30 = int(np.argmin(abs(X - 10))), int(np.argmin(abs(X - 30)))
NEAR = (X >= 6) & (X <= 40)
ELEV = (X >= 25) & (X <= 35)


def student(ckpt: Path, unit: str, cache: Path) -> dict:
    if cache.exists():
        return dict(np.load(cache))
    import torch
    import viz_compare as V
    d = ROOT / "data/ep_label" / unit
    lab = np.load(d / "labels.npz")
    n = len(lab["big"])
    even = np.arange(4, n - n % 2, 2)
    s = V.run_student(ckpt, d, lab, even)
    out = {"even": even, "lanes": s["lanes"], "lane_prob": s["lane_prob"], "plan": s["plan"]}
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **out)
    del torch
    return out


def frame_table(unit: str, s: dict) -> dict:
    lab = np.load(ROOT / "data/ep_label" / unit / "labels.npz")
    ev = s["even"]
    a, b = BIG["lanes"]
    t = lab["big"][ev, a:b][:, :264].reshape(-1, 4, 33, 2)
    h = lab["mount"][ev, 1]
    elev = h - t[:, 1:3][:, :, ELEV, 1].mean((1, 2))
    q = {"student": s["lanes"][..., 0], "shipped": lab["rec_lanes"][ev][..., 0], "teacher": t[..., 0],
         "LD": lab["ld_lanes"][ev], "LDfix": slope_fix(lab["ld_lanes"][ev], t[..., 1], h, lab["mount"][ev, 0])}
    prob = {"student": s["lane_prob"], "shipped": lab["rec_lane_probabilities"][ev], "teacher": np.ones((len(ev), 4)),
            "LD": lab["ld_lane_prob"][ev], "LDfix": lab["ld_lane_prob"][ev]}
    return {"elev": elev, "v": lab["v_ego"][ev], "q": q, "prob": prob, "ok": np.isfinite(lab["rec_lanes"][ev][:, 0, 0, 0])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("units", nargs="+")
    args = ap.parse_args()
    rows = [frame_table(u, student(args.ckpt, u, ROOT / f"outputs/diag_lane/{args.name}/{u}.npz")) for u in args.units]
    cat = lambda f: np.concatenate([f(r) for r in rows])
    elev, v, ok = cat(lambda r: r["elev"]), cat(lambda r: r["v"]), cat(lambda r: r["ok"])
    names = ("student", "shipped", "teacher", "LD")
    Q = {k: cat(lambda r: r["q"][k]) for k in names + ("LDfix",)}
    # frame-to-frame jitter within each unit
    J = {k: cat(lambda r: np.r_[np.nan, np.abs(np.diff(r["q"][k][:, 1:3, I30], axis=0)).mean(1)] * 100) for k in names}
    mov = (v > 5) & ok
    bins = [(-9, -0.6), (-0.6, -0.3), (-0.3, -0.1), (-0.1, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 9)]
    print(f"{args.name}: {len(args.units)} units, {mov.sum()} moving frames; elevation of the road at 25-35 m (teacher)")
    print("elev (m)      frames | width 30m/10m: student shipped teacher LD | jitter@30m cm: student shipped teacher LD"
          " | |y-LD| 6-40m cm: student shipped | |y-teacher| cm: student shipped LD | |y-LDfix| cm: student shipped teacher")
    dif = lambda a, b, m: np.nanmean(np.abs(Q[a][m][:, 1:3][:, :, NEAR] - Q[b][m][:, 1:3][:, :, NEAR])) * 100
    for lo, hi in bins:
        m = mov & (elev >= lo) & (elev < hi)
        if m.sum() < 30:
            continue
        wr = {k: np.nanmedian((Q[k][m, 2, I30] - Q[k][m, 1, I30]) / (Q[k][m, 2, I10] - Q[k][m, 1, I10])) for k in names}
        jit = {k: np.nanmedian(J[k][m]) for k in names}
        print(f"[{lo:+.1f},{hi:+.1f}) {m.sum():7d} | " + " ".join(f"{wr[k]:.3f}" for k in names)
              + " | " + " ".join(f"{jit[k]:5.1f}" for k in names)
              + f" | {dif('student', 'LD', m):5.1f} {dif('shipped', 'LD', m):5.1f} | {dif('student', 'teacher', m):5.1f}"
              f" {dif('shipped', 'teacher', m):5.1f} {dif('LD', 'teacher', m):5.1f} | {dif('student', 'LDfix', m):5.1f}"
              f" {dif('shipped', 'LDfix', m):5.1f} {dif('teacher', 'LDfix', m):5.1f}")
    print(f"all moving: |y-LD| student {dif('student', 'LD', mov):.1f} shipped {dif('shipped', 'LD', mov):.1f}"
          f" teacher {dif('teacher', 'LD', mov):.1f} | |y-LDfix| student {dif('student', 'LDfix', mov):.1f}"
          f" shipped {dif('shipped', 'LDfix', mov):.1f} teacher {dif('teacher', 'LDfix', mov):.1f}"
          f" | jitter@30m student {np.nanmedian(J['student'][mov]):.1f} shipped {np.nanmedian(J['shipped'][mov]):.1f}")


if __name__ == "__main__":
    main()
