"""Per-element normalised plan error (|student - teacher| / teacher std) of a checkpoint, closed loop on one route:
which plan components and horizons dominate the loss, and how many sit past the loss cap.

usage: .venv/bin/python scripts/diag_plan.py ckpt/run/last.pt data/ep_label/<unit> 1500   (CPU, closed loop)
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from student import HEADS, POLICY_HEADS, Student  # noqa: E402
from train import BIG  # noqa: E402

ckpt, route, nfr = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
torch.set_num_threads(4)
ck = torch.load(ckpt, map_location="cpu")
m = Student(**ck.get("arch", {})).eval()
m.load_state_dict(ck["model"])
lab = np.load(route / "labels.npz")
frames = np.load(route / "frames.npy", mmap_mode="r")
pulse = lab["pulse"]
n = len(lab["big"])
even = np.arange(4, min(n - n % 2, 4 + 2 * nfr), 2)
hist = np.zeros(((n + 1) // 2, 512), np.float32)
outs = {"plan": [], "road_transform": [], "lead": [], "pose": []}
with torch.no_grad():
    for t in even:
        j = t // 2
        road = torch.from_numpy(np.concatenate([frames[j - 2, 0], frames[j, 0]])[None]).float()
        wide = torch.from_numpy(np.concatenate([frames[j - 2, 1], frames[j, 1]])[None]).float()
        des = np.zeros((100, 8), np.float32)
        lo = max(0, t - 99)
        des[100 - (t + 1 - lo):] = pulse[lo:t + 1]
        idx = np.arange(t - 96, t, 4) // 2
        fb = np.zeros((24, 512), np.float32)
        fb[idx >= 0] = hist[idx[idx >= 0]]
        o = m(road, wide, torch.from_numpy(des.reshape(25, 4, 8).max(1)[None]), torch.from_numpy(fb[None]),
              torch.tensor([[1.0, 0.0]]))
        hist[j] = o["out_hidden"][0].numpy()
        y = m.outputs(o)
        for k in outs:
            outs[k].append(y[k][0].numpy())
v = lab["v_ego"][even]
sel = v > 5
names = ["x", "y", "z", "vx", "vy", "vz", "ax", "ay", "az", "roll", "pitch", "yaw", "droll", "dpitch", "dyaw"]
for k in outs:
    s = np.stack(outs[k])[sel]
    a, b = BIG[k]
    tg = lab["big"][even][sel][:, a:b]
    half = tg.shape[1] // 2
    err = np.abs(s[:, :half] - tg[:, :half]) / np.exp(np.clip(tg[:, half:], -7, 5))
    capped = np.minimum(err, 20)
    print(f"== {k}: loss-mean part {capped.mean():.3f}, at cap {np.mean(err >= 20) * 100:.1f}% of elements, "
          f"logstd part {0.1 * np.abs(s[:, half:] - tg[:, half:]).mean():.3f}")
    if k == "plan":
        e = capped.reshape(-1, 33, 15)
        ec = (err >= 20).reshape(-1, 33, 15)
        sd = np.exp(np.clip(tg[:, half:], -7, 5)).reshape(-1, 33, 15)
        print("comp   mean_err  cap%   err@t0  err@2s(i=14)  err@t32  teacher_std_median")
        for c in range(15):
            print(f"{names[c]:6s} {e[:, :, c].mean():8.3f} {ec[:, :, c].mean() * 100:5.1f} {e[:, 0, c].mean():8.3f} "
                  f"{e[:, 14, c].mean():12.3f} {e[:, 32, c].mean():8.3f}  {np.median(sd[:, :, c]):.4f}")
    elif k == "road_transform":
        e = capped
        print("elements:", " ".join(f"{x:.2f}" for x in e.mean(0)), "| cap%:",
              " ".join(f"{x * 100:.0f}" for x in (err >= 20).mean(0)))
