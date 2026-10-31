"""Does the student read motion from its frame pair? Closed loop with the older frame of each pair taken 0, 0.2
(normal) or 0.4 s back: mean predicted speed vx0, |yaw rate r0| and the plan_y2s error, against the teacher.

usage: .venv/bin/python scripts/diag_motion.py <ckpt> <unit name> ..."""
import sys
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from student import Student  # noqa: E402
from train import BIG  # noqa: E402
T = np.array([10.0 * (i / 32) ** 2 for i in range(33)])
torch.set_num_threads(3)
ck = torch.load(sys.argv[1], map_location="cpu")
m = Student(**ck.get("arch", {})).eval()
m.load_state_dict(ck["model"])
for u in sys.argv[2:]:
    d = ROOT / "data/ep_label" / u
    lab = np.load(d / "labels.npz")
    frames = np.load(d / "frames.npy", mmap_mode="r")
    n = len(lab["big"])
    even = np.arange(8, min(n - n % 2, 8 + 2 * 2000), 2)
    a, b = BIG["plan"]
    tp = lab["big"][even][:, a:b][:, :495].reshape(-1, 33, 15)
    v = lab["v_ego"][even]
    sel = v > 5
    ty2 = np.array([np.interp(2.0, T, q[:, 1]) for q in tp])
    print(f"{u}: car v {v[sel].mean():.2f} m/s, teacher vx0 {tp[sel, 0, 3].mean():.2f}, teacher |r0| {np.abs(tp[sel, 0, 14]).mean()*1000:.2f} mrad/s")
    for lag in (0, 2, 4):
        hist = np.zeros(((n + 1) // 2, 512), np.float32)
        out = []
        with torch.no_grad():
            for t in even:
                j = t // 2
                road = torch.from_numpy(np.concatenate([frames[j - lag, 0], frames[j, 0]])[None]).float()
                wide = torch.from_numpy(np.concatenate([frames[j - lag, 1], frames[j, 1]])[None]).float()
                des = np.zeros((100, 8), np.float32)
                lo = max(0, t - 99)
                des[100 - (t + 1 - lo):] = lab["pulse"][lo:t + 1]
                idx = np.arange(t - 96, t, 4) // 2
                fb = np.zeros((24, 512), np.float32)
                fb[idx >= 0] = hist[idx[idx >= 0]]
                o = m(road, wide, torch.from_numpy(des.reshape(25, 4, 8).max(1)[None]), torch.from_numpy(fb[None]),
                      torch.tensor([[1.0, 0.0]]))
                hist[j] = o["out_hidden"][0].numpy()
                out.append(m.outputs(o)["plan"][0, :495].numpy().reshape(33, 15))
        p = np.stack(out)
        y2 = np.array([np.interp(2.0, T, q[:, 1]) for q in p])
        print(f"  pair gap {lag * 0.1:.1f} s: vx0 {p[sel, 0, 3].mean():6.2f} m/s  |r0| {np.abs(p[sel, 0, 14]).mean()*1000:5.2f} mrad/s  "
              f"vx0 err {np.abs(p[sel, 0, 3] - tp[sel, 0, 3]).mean():.2f}  plan_y2s {np.abs(y2 - ty2)[sel].mean()*100:.2f} cm", flush=True)
