"""Lateral behaviour of a recorded edgepilot drive, from its own ModelState and ControlState (no teacher): where the
car sat in its lane and where the model's plan pointed relative to the model's own ego lane lines, split by
laneless / lane mode and engagement. Coordinates as recorded: x forward, y RIGHT, z down (so left = negative y).

  car offset     lane centre at x = 0 (mean of ego lines 1, 2): > 0 the lane centre is right of the car = the car sits
                 left of centre
  plan - centre  plan y minus the lane centre at 10 / 20 / 30 m (plan y interpolated at those x): < 0 the plan aims
                 left of the lane centre
Frames: v > 5 m/s, both ego lines with probability > 0.5 for the offsets.

usage: .venv/bin/python scripts/diag_drive_lateral.py <route dir> [--edgepilot ~/Documents/ax630/edgepilot]"""
import argparse
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("route", type=Path)
    ap.add_argument("--edgepilot", type=Path, default=Path.home() / "Documents/ax630/edgepilot")
    args = ap.parse_args()
    sys.path.insert(0, str(args.edgepilot / "tools/model"))
    import recording_reader as rr
    ms, cs = [], []
    # 외장 디스크의 AppleDouble(._*) 파일은 건너뛴다
    for f in [f for f in rr.route_event_files(args.route) if not f.name.startswith("._")]:
        part = rr.chunk_states(f, rr.RECORD_MODEL_STATE, rr.RECORD_CONTROL_STATE)
        ms.append(part[rr.RECORD_MODEL_STATE][1].copy())
        cs.append(part[rr.RECORD_CONTROL_STATE][1].copy())
    M, C = np.concatenate(ms), np.concatenate(cs)
    # 각 모델 프레임에 가장 가까운(이전) 제어 상태
    ci = np.clip(np.searchsorted(C["timestamp_ns"], M["model_timestamp_ns"]) - 1, 0, len(C) - 1)
    c = C[ci]
    v = c["ego_speed_kph"] / 3.6
    laneless = (c["hud_flags"] & 1) == 1
    active = c["active"] == 1
    X = rr.X_IDXS
    l1, l2 = M["lanes"][:, 1, :, 0], M["lanes"][:, 2, :, 0]
    p1, p2 = M["lane_probabilities"][:, 1], M["lane_probabilities"][:, 2]
    lines = (p1 > .5) & (p2 > .5)
    centre = (l1 + l2) / 2
    plan = M["plan"]
    def at(xq):
        py = np.array([np.interp(xq, px, py) for px, py in zip(plan[:, :, 0], plan[:, :, 1])])
        cy = np.array([np.interp(xq, X, cc) for cc in centre])
        return py - cy
    mov = (v > 5) & (M["valid"] == 1)
    print(f"{args.route.name}: {len(M)} model frames, moving {mov.sum()}, laneless {np.mean(laneless[mov]) * 100:.0f}% of moving, "
          f"active {np.mean(active[mov]) * 100:.0f}%, both ego lines {np.mean(lines[mov]) * 100:.0f}%")
    off = {x: at(x) for x in (10.0, 20.0, 30.0)}
    car = centre[:, 0]
    width = (l2 - l1)[:, 0]
    print("subset                      frames   car offset (cm, >0 car left)   plan-centre 10/20/30 m (cm, <0 left)   lane width (m)")
    for name, m in (("all moving, lines", mov & lines), ("active, laneless", mov & lines & active & laneless),
                    ("active, lane mode", mov & lines & active & ~laneless), ("not active", mov & lines & ~active)):
        if m.sum() < 20:
            print(f"{name:26s} {m.sum():7d}")
            continue
        print(f"{name:26s} {m.sum():7d}   mean {np.mean(car[m]) * 100:+6.1f}  p10/p90 {np.percentile(car[m], 10) * 100:+6.1f} "
              f"{np.percentile(car[m], 90) * 100:+6.1f}      " + " ".join(f"{np.mean(off[x][m]) * 100:+6.1f}" for x in off) +
              f"        {np.median(width[m]):.2f}")
    # 차선을 넘은 순간: 자차 중심이 차선선 밖(|offset| > 폭/2)
    out = mov & lines & active & (np.abs(car) > width / 2)
    print(f"active frames with the car centre outside its ego lines: {out.sum()} "
          f"(left side {np.sum(out & (car > 0))}, right side {np.sum(out & (car < 0))})")
    Path("outputs/diag_drive").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(Path("outputs/diag_drive") / f"lateral_{args.route.name}.npz", t=M["model_timestamp_ns"], v=v, laneless=laneless,
                        active=active, car=car, width=width, p1=p1, p2=p2, off10=off[10.0], off20=off[20.0], off30=off[30.0])


if __name__ == "__main__":
    main()
