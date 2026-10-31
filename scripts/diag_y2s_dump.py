"""Per-frame plan and lead outputs (closed loop) of one student checkpoint on some units, plus the teacher, small and
shipped plans and leads and the frame context, saved to one npz per unit for slicing the plan_y2s and lead_x errors
afterwards (diag_y2s.py, diag_lead.py). --s8: the student as the NPU runs it (eval_student --s8).

usage: scripts/diag_y2s_dump.py <ckpt> outputs/diag_y2s/dump/<name> <unit dir> ... [--nohist] [--s8]"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import eval_student as E  # noqa: E402
from student import HEADS, POLICY_HEADS, load_student, quant_sim  # noqa: E402
from train import BIG  # noqa: E402
from zframes import load_frames  # noqa: E402

args = [a for a in sys.argv[1:] if not a.startswith("--")]
nohist = "--nohist" in sys.argv
s8 = "--s8" in sys.argv
ckpt, out, units = Path(args[0]), Path(args[1]), [Path(u) for u in args[2:]]
out.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(4)


@torch.no_grad()
def run(route_dir, lab, even):
    m = load_student(torch.load(ckpt, map_location="cpu"))
    if s8:
        quant_sim(m.fuse())
    frames = load_frames(route_dir)
    pulse = lab["pulse"]
    hist = np.zeros(((len(lab["big"]) + 1) // 2, 512), np.float32)
    outs = {k: [] for k in [*HEADS, *POLICY_HEADS]}
    for t in even:
        j = t // 2
        road = torch.from_numpy(np.concatenate([frames[j - 2, 0], frames[j, 0]])[None]).float()
        wide = torch.from_numpy(np.concatenate([frames[j - 2, 1], frames[j, 1]])[None]).float()
        des = np.zeros((100, 8), np.float32)
        lo = max(0, t - 99)
        des[100 - (t + 1 - lo):] = pulse[lo:t + 1]
        idx = np.arange(t - 96, t, 4) // 2
        fb = np.zeros((24, 512), np.float32)
        if not nohist:
            fb[idx >= 0] = hist[idx[idx >= 0]]
        o = m(road, wide, torch.from_numpy(des.reshape(25, 4, 8).max(1)[None]), torch.from_numpy(fb[None]),
              torch.tensor([[1.0, 0.0]]))
        hist[j] = o["out_hidden"][0].numpy()
        y = m.outputs(o)
        for k in outs:
            outs[k].append(y[k][0].numpy())
    return {k: np.stack(v) for k, v in outs.items()}


for d in units:
    lab = np.load(d / "labels.npz")
    n = len(lab["big"])
    even = np.arange(4, n - n % 2, 2)
    raw = run(d, lab, even)
    s = E.parse_raw(raw)
    big = lab["big"][even]
    a, b = BIG["plan"]
    tplan = big[:, a:b]
    # slimmed training units have no small model outputs (slim_labels.py): NaN in its place
    small = E.from_layout(lab["small"][even] if "small" in lab.files else np.full((len(even), max(b for _, b in E.OP.values())), np.nan, np.float32), E.OP)
    rec = {k: x[even] for k, x in E.recorded(lab).items()}
    t = E.from_layout(big, BIG)
    a, b = BIG["lead"]
    np.savez_compressed(out / f"{d.name}.npz", even=even, v=lab["v_ego"][even], yaw_rate=lab["yaw_rate"][even],
                        desire=lab["desire"][even], pulse=lab["pulse"][even], engaged=lab["engaged"][even],
                        seg=lab["seg"][even], t_mu=tplan[:, :495].reshape(-1, 33, 15),
                        t_logstd=tplan[:, 495:].reshape(-1, 33, 15),
                        t_desire=big[:, slice(*BIG["desire_state"])],
                        s_plan=raw["plan"][:, :495].reshape(-1, 33, 15), s_desire=raw["desire_state"],
                        small_xyz=small["plan_xyz"], rec_xyz=rec["plan_xyz"],
                        s_lat=E.lat_acc(s, lab["v_ego"][even]), t_lat=E.lat_acc(E.from_layout(big, BIG), lab["v_ego"][even]),
                        small_lat=E.lat_acc(small, lab["v_ego"][even]), rec_lat=E.lat_acc(rec, lab["v_ego"][even]),
                        s_lead=raw["lead"][:, :72].reshape(-1, 3, 6, 4), s_lead_prob=s["lead_prob"],
                        t_lead=big[:, a:a + 72].reshape(-1, 3, 6, 4), t_lead_logstd=big[:, a + 72:b].reshape(-1, 3, 6, 4),
                        t_lead_prob=t["lead_prob"], small_lead_x=small["lead_x"], small_lead_prob=small["lead_prob"],
                        rec_lead_x=rec["lead_x"], rec_lead_prob=rec["lead_prob"],
                        rec_lead_v=lab["rec_lead_velocity"][even])
    print(d.name, len(even), "saved", flush=True)
