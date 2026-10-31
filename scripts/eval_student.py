"""Compare models on held-out edgepilot routes: the board's shipped model (recorded), the small model (fp32 host),
students (fp32, closed loop on their own hidden states) and the big teacher.

References: the big teacher for the plan, lead, meta and desire; LD (inner line edges) for the lanes. Frames are
the even ones (the warped views ep_label.py kept) with v > 5 m/s.

  laneless lat accel   openpilot get_curvature_from_plan (action_t 0.3 s) x v^2, |model - teacher|
  long accel           get_accel_from_plan (0.3 s), |model - teacher| (not recorded for the board's model)
  plan y 2 s           |model - teacher| of the plan's lateral position at 2 s
  ego lanes vs LD      |model - LD| of lanes 1 and 2 at 6-40 m where LD saw them; also lane-width difference
  lead x               |model - teacher| where the teacher's lead probability > 0.5 (routes weighted by moving
                       frames as the other metrics; lead_x_lf weights them by those lead frames)
  gas press            |model - teacher| of the gas-press probability at 2 s
  false lead           % of the moving frames without a teacher lead (prob <= 0.5) where the model has one (> 0.5)
  vx0                  |model - teacher| of the plan's current speed, cm/s (not recorded for the board's model)
  lanes / width ldfix  ego lanes vs LD as above, with LD on the teacher's road surface instead of a flat road
                       (lane_geom.slope_fix): LD's flat projection widens a rising road's lanes and narrows a falling one's

--s8: the students as the NPU runs them (fused, S8 per-channel weights: student.quant_sim()), rows marked (S8).

usage: scripts/py scripts/eval_student.py [--ckpt ckpt/run1/last.pt ...] [--routes r1,r2] [--s8]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from student import HEADS, POLICY_HEADS, load_student, quant_sim  # noqa: E402
from lane_geom import slope_fix  # noqa: E402
from train import BIG, EVAL_ROUTES, default_device  # noqa: E402
from zframes import load_frames  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
T_IDXS = np.array([10.0 * (i / 32) ** 2 for i in range(33)])
X_IDXS = np.array([192.0 * (i / 32) ** 2 for i in range(33)])
LANE_X = (X_IDXS >= 6) & (X_IDXS <= 40)
OP = {"meta": (0, 55), "lanes": (117, 645), "lane_prob": (645, 653), "lead": (917, 1061), "lead_prob": (1061, 1064),
      "plan": (1576, 2566), "desire_state": (2566, 2574)}


def parse_raw(raw: dict) -> dict:
    """Raw head outputs (name -> [N, width]) to the quantities compared."""
    n = len(raw["plan"])
    plan = raw["plan"][:, :495].reshape(n, 33, 15)
    lanes = raw["lanes"][:, :264].reshape(n, 4, 33, 2)[..., 0]
    lead = raw["lead"][:, :72].reshape(n, 3, 6, 4)
    sig = lambda x: 1 / (1 + np.exp(-x))
    return {"plan_xyz": plan[..., 0:3], "plan_v": plan[..., 3], "plan_a": plan[..., 6], "yaw": plan[..., 11],
            "yaw_rate": plan[..., 14], "lanes": lanes, "lane_prob": sig(raw["lane_prob"][:, 1::2]),
            "lead_x": lead[:, 0, 0, 0], "lead_prob": sig(raw["lead_prob"][:, 0]),
            "gas2": sig(raw["meta"][:, 31 + 4])}


def from_layout(y: np.ndarray, layout: dict) -> dict:
    return parse_raw({k: y[:, a:b] for k, (a, b) in layout.items()})


def recorded(lab) -> dict:
    return {"plan_xyz": lab["rec_plan"], "yaw": lab["rec_plan_yaw"], "yaw_rate": lab["rec_plan_yaw_rate"],
            "lanes": lab["rec_lanes"][..., 0], "lane_prob": lab["rec_lane_probabilities"],
            "lead_x": lab["rec_lead_x"], "lead_prob": lab["rec_lead_probability"], "gas2": lab["rec_gas_press_probs"][:, 1]}


def lat_acc(q, v):
    psi = np.array([np.interp(0.3, T_IDXS, y) for y in q["yaw"]])
    vv = np.clip(v, 1.0, None)
    return (2 * psi / (vv * 0.3) - q["yaw_rate"][:, 0] / vv) * v ** 2


def long_acc(q):
    if "plan_v" not in q:
        return None
    vt = np.array([np.interp(0.3, T_IDXS, y) for y in q["plan_v"]])
    return 2 * (vt - q["plan_v"][:, 0]) / 0.3 - q["plan_a"][:, 0]


@torch.no_grad()
def run_student(ckpt: Path, route_dir: Path, lab, even, dev, s8=False, raw=False):
    m = load_student(torch.load(ckpt, map_location="cpu"))
    if s8:
        quant_sim(m.fuse())
    m = m.to(dev)
    frames = load_frames(route_dir)
    pulse = lab["pulse"]
    hist = np.zeros(((len(lab["big"]) + 1) // 2, 512), np.float32)   # hidden per kept frame, closed loop
    outs = {k: [] for k in [*HEADS, *POLICY_HEADS]}
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
              torch.tensor([[1.0, 0.0]], device=dev))
        hist[j] = o["out_hidden"][0].cpu().numpy()
        y = m.outputs(o)
        for k in outs:
            outs[k].append(y[k][0].cpu().numpy())
    outs = {k: np.stack(v) for k, v in outs.items()}
    return outs if raw else parse_raw(outs)


def compare(q: dict, ref: dict, ld_lanes, v, sel, ld_fix=None) -> dict:
    r = {}
    la, lr = lat_acc(q, v), lat_acc(ref, v)
    r["lat_acc"] = np.nanmean(np.abs(la - lr)[sel])
    a, ar = long_acc(q), long_acc(ref)
    r["long_acc"] = np.nanmean(np.abs(a - ar)[sel]) if a is not None else np.nan
    y2 = lambda x: np.array([np.interp(2.0, T_IDXS, p[:, 1]) for p in x["plan_xyz"]])
    r["plan_y2s_cm"] = np.nanmean(np.abs(y2(q) - y2(ref))[sel]) * 100
    d = []
    w = []
    for i in (1, 2):
        ok = np.isfinite(ld_lanes[:, i]) & sel[:, None] & LANE_X[None]
        d.append(np.abs(q["lanes"][:, i] - ld_lanes[:, i])[ok])
    r["lanes_vs_ld_cm"] = np.mean(np.concatenate(d)) * 100
    both = np.isfinite(ld_lanes[:, 1]) & np.isfinite(ld_lanes[:, 2]) & sel[:, None] & LANE_X[None]
    r["width_vs_ld_cm"] = np.mean(np.abs((q["lanes"][:, 2] - q["lanes"][:, 1]) - (ld_lanes[:, 2] - ld_lanes[:, 1]))[both]) * 100
    if ld_fix is not None:   # the same against LD on the teacher's road surface (lane_geom.slope_fix; LD assumes a flat road)
        ok = np.isfinite(ld_fix[:, 1:3]) & sel[:, None, None] & LANE_X[None, None]
        r["lanes_ldfix_cm"] = np.mean(np.abs(q["lanes"][:, 1:3] - ld_fix[:, 1:3])[ok]) * 100
        both = np.isfinite(ld_fix[:, 1]) & np.isfinite(ld_fix[:, 2]) & sel[:, None] & LANE_X[None]
        r["width_ldfix_cm"] = np.mean(np.abs((q["lanes"][:, 2] - q["lanes"][:, 1]) - (ld_fix[:, 2] - ld_fix[:, 1]))[both]) * 100
    lead = (ref["lead_prob"] > 0.5) & sel
    r["lead_x_m"] = np.nanmean(np.abs(q["lead_x"] - ref["lead_x"])[lead])
    r["gas2"] = np.nanmean(np.abs(q["gas2"] - ref["gas2"])[sel])
    r["lead_x_lf"], r["n_lead"] = r["lead_x_m"], lead.sum()
    nolead = (ref["lead_prob"] <= 0.5) & sel
    r["false_lead_pct"], r["n_nolead"] = ((q["lead_prob"] > 0.5) & nolead).sum() / max(nolead.sum(), 1) * 100, nolead.sum()
    r["vx0_cms"] = np.nanmean(np.abs(q["plan_v"][:, 0] - ref["plan_v"][:, 0])[sel]) * 100 if "plan_v" in q else np.nan
    return r


def student_name(c: Path, s8=False) -> str:
    """Table row of a checkpoint: its run directory, plus the file stem unless it is last.pt (ep1.pt, ... of one run)."""
    return f"student {c.parent.name}" + ("" if c.stem == "last" else f"/{c.stem}") + (" (S8)" if s8 else "")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "data/ep_label")
    ap.add_argument("--ckpt", type=Path, nargs="*", default=[])
    ap.add_argument("--routes", default=",".join(EVAL_ROUTES))
    ap.add_argument("--device", default=default_device(), help="cuda, mps (Apple GPU) or cpu")
    ap.add_argument("--tb", action=argparse.BooleanOptionalAction, default=True,
                    help="write the students' metrics to <ckpt dir>/tb (step = epoch)")
    ap.add_argument("--s8", action="store_true", help="evaluate the students with S8 weights (NPU quantization)")
    ap.add_argument("--ensemble", action="store_true",
                    help="also a row for the mean of the students' raw outputs (each in its own closed loop)")
    args = ap.parse_args()
    rows = {}
    units = []
    for rname in args.routes.split(","):
        found = sorted(d for d in args.data.glob(rname + "*") if (d / "labels.npz").exists())
        if not found:
            print(f"{rname}: not labelled yet")
        units += found
    for d in units:
        lab = np.load(d / "labels.npz")
        n = len(lab["big"])
        even = np.arange(4, n - n % 2, 2)
        v = lab["v_ego"][even]
        sel = v > 5.0
        ld = lab["ld_lanes"][even]
        big = from_layout(lab["big"][even], BIG)
        a, b = BIG["lanes"]
        ld_fix = slope_fix(ld, lab["big"][even, a:b][:, :264].reshape(-1, 4, 33, 2)[..., 1], lab["mount"][even, 1], lab["mount"][even, 0])
        models = {"board NPU (shipped)": {k: x[even] for k, x in recorded(lab).items()},
                  "small fp32": from_layout(lab["small"][even], OP),
                  "big teacher": big}
        raws = []
        for c in args.ckpt:
            raws.append(run_student(c, d, lab, even, args.device, args.s8, raw=True))
            models[student_name(c, args.s8)] = parse_raw(raws[-1])
        if args.ensemble and len(raws) > 1:
            models[f"ensemble of {len(raws)}" + (" (S8)" if args.s8 else "")] = parse_raw({k: np.mean([r[k] for r in raws], 0) for k in raws[0]})
        for name, q in models.items():
            rows.setdefault(name, []).append((sel.sum(), compare(q, big, ld, v, sel, ld_fix)))
    keys = ["lat_acc", "long_acc", "plan_y2s_cm", "lanes_vs_ld_cm", "width_vs_ld_cm", "lead_x_m", "gas2",
            "lead_x_lf", "false_lead_pct", "vx0_cms", "lanes_ldfix_cm", "width_ldfix_cm"]
    weight = {"lead_x_lf": "n_lead", "false_lead_pct": "n_nolead"}   # the rest: moving frames
    print(f"{'model':28s} " + " ".join(f"{k:>15s}" for k in keys))
    table = {}
    for name, rs in rows.items():
        table[name] = []
        for k in keys:   # mean over the routes weighted by frames; NaN when the model has no such value (board long_acc)
            ok = [(r[weight[k]] if k in weight else n, r[k]) for n, r in rs if np.isfinite(r[k])]
            table[name].append(sum(n * v for n, v in ok) / sum(n for n, _ in ok) if ok else np.nan)
        print(f"{name:28s} " + " ".join(f"{x:15.4f}" for x in table[name]))
    if args.tb:
        write_tb(args.ckpt, table, keys, args.s8)


def write_tb(ckpts, table, keys, s8=False):
    """Each student's metrics into its checkpoint's TensorBoard (<ckpt dir>/tb) at its epoch, with the shipped
    board model and the small model in the same charts for reference."""
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        print("tensorboard not installed: metrics not written to TensorBoard")
        return
    for c in ckpts:
        ep = int(torch.load(c, map_location="cpu").get("epoch", 0)) + 1
        tb = SummaryWriter(str(c.parent / "tb"))
        for i, k in enumerate(keys):
            vals = {"student_s8" if s8 else "student": table[student_name(c, s8)][i], "board_npu": table["board NPU (shipped)"][i],
                    "small_fp32": table["small fp32"][i]}
            tb.add_scalars(f"eval/{k}", {n: v for n, v in vals.items() if np.isfinite(v)}, ep)
        tb.close()


if __name__ == "__main__":
    main()
