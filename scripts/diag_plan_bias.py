"""Signed lateral bias of a student's plan against the big teacher (S8, as the NPU runs it), next to the shipped
model's fp32 core (labels' `small`) on the same frames, and how much the student's plan moves when traffic_convention
is flipped to left-hand traffic ([0, 1], what --flip trains mirrored samples with). y > 0 is left.

Frames: moving (v > 5 m/s), split by LD's ego lane-line probability (both lines 1, 2 > 0.5 = lines, both < 0.3 =
no lines) and by the teacher's turning (|y at 4 s| > 2 m). Plan y at 1, 2, 3, 4 s (interpolated on T_IDXS), cm.

usage: .venv/bin/python scripts/diag_plan_bias.py <ckpt> <unit> ..."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_student as E  # noqa: E402
from student import HEADS, POLICY_HEADS, load_student, quant_sim  # noqa: E402
from train import BIG  # noqa: E402
from zframes import load_frames  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TS = (1.0, 2.0, 3.0, 4.0)


def plan_y(plan990):
    p = plan990[:, :495].reshape(-1, 33, 15)[..., 1]
    return np.stack([[np.interp(t, E.T_IDXS, r) for t in TS] for r in p])


def run(m, d, lab, even, dev, tc):
    frames = load_frames(d)
    pulse = lab["pulse"]
    hist = np.zeros(((len(lab["big"]) + 1) // 2, 512), np.float32)
    plans = []
    with torch.no_grad():
        for t in even:
            j = t // 2
            road = torch.from_numpy(np.concatenate([frames[j - 2, 0], frames[j, 0]])[None]).to(dev).float()
            wide = torch.from_numpy(np.concatenate([frames[j - 2, 1], frames[j, 1]])[None]).to(dev).float()
            des = np.zeros((100, 8), np.float32)
            lo = max(0, t - 99)
            des[100 - (t + 1 - lo):] = pulse[lo:t + 1]
            idx = np.arange(t - 96, t, 4) // 2
            fb = np.zeros((24, 512), np.float32)
            fb[idx >= 0] = hist[idx[idx >= 0]]
            o = m(road, wide, torch.from_numpy(des.reshape(25, 4, 8).max(1)[None]).to(dev), torch.from_numpy(fb[None]).to(dev),
                  torch.tensor([tc], device=dev))
            hist[j] = o["out_hidden"][0].cpu().numpy()
            plans.append(m.cpu_heads["plan"](o["hx_plan"])[0].cpu().numpy())
    return plan_y(np.stack(plans))


def main():
    ck = sys.argv[1]
    m = load_student(torch.load(ck, map_location="cpu"))
    quant_sim(m.fuse())
    dev = "mps"
    m = m.to(dev).eval()
    acc = {k: [] for k in ("s", "sL", "ship", "t", "lines", "nolines", "turn")}
    for u in sys.argv[2:]:
        d = ROOT / "data/ep_label" / u
        lab = np.load(d / "labels.npz")
        n = len(lab["big"])
        even = np.arange(4, n - n % 2, 2)
        sel = lab["v_ego"][even] > 5
        a, b = BIG["plan"]
        t = plan_y(lab["big"][even, a:b])
        ship = plan_y(np.nan_to_num(lab["small"][even, 1576:2566], nan=np.nan))
        s = run(m, d, lab, even, dev, [1.0, 0.0])
        sL = run(m, d, lab, even, dev, [0.0, 1.0])
        lp = lab["ld_lane_prob"][even]
        for k, v in (("s", s), ("sL", sL), ("ship", ship), ("t", t), ("lines", (lp[:, 1] > .5) & (lp[:, 2] > .5)),
                     ("nolines", (lp[:, 1] < .3) & (lp[:, 2] < .3)), ("turn", np.abs(t[:, 3]) > 2.0)):
            acc[k].append(v[sel])
        print(u, "done", flush=True)
    A = {k: np.concatenate(v) for k, v in acc.items()}
    print(f"{ck}: {len(A['t'])} moving frames; plan y - teacher y (cm, + = left) at {TS} s: mean (signed) | mean |.|")
    for name, m_ in (("all", np.ones(len(A["t"]), bool)), ("lane lines", A["lines"]), ("no lane lines", A["nolines"]),
                     ("straight, no lines", A["nolines"] & ~A["turn"]), ("turning", A["turn"])):
        row = f"{name:20s} n={m_.sum():6d} "
        for k, lbl in (("s", "student"), ("ship", "shipped"), ("sL", "student L-traffic")):
            dd = (A[k][m_] - A["t"][m_]) * 100
            ok = np.isfinite(dd).all(1)
            row += f"| {lbl} " + " ".join(f"{x:+6.1f}" for x in dd[ok].mean(0)) + " / " + " ".join(f"{x:5.1f}" for x in np.abs(dd[ok]).mean(0)) + " "
        print(row)
    dl = (A["sL"] - A["s"]) * 100
    print("student plan moves when traffic_convention is flipped (cm, mean signed / mean |.|):",
          " ".join(f"{x:+.1f}" for x in dl.mean(0)), "/", " ".join(f"{x:.1f}" for x in np.abs(dl).mean(0)))


if __name__ == "__main__":
    main()
