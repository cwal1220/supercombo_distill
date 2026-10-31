"""Does a student read motion? Its pose head (current frame pair only, S8) for pairs (j-g, j), g = 0, 1, 2 (normal), 4 kept
frames, against the teacher pose of the normal pair. A model that measures motion scales 0, 0.5, 1, 2.

usage: .venv/bin/python scripts/diag_gap.py <unit> [ckpt, default ckpt/mac_base_qat3/last.pt]"""
import sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from student import load_student, quant_sim
from train import BIG
from zframes import load_frames
torch.set_num_threads(4)
unit = ROOT / "data/ep_label" / sys.argv[1]
ckpt = sys.argv[2] if len(sys.argv) > 2 else ROOT / "ckpt/mac_base_qat3/last.pt"
m = quant_sim(load_student(torch.load(ckpt, map_location="cpu")).fuse())
lab = np.load(unit / "labels.npz"); frames = load_frames(unit)
n = len(lab["big"]); even = np.arange(8, n - n % 2, 2)
even = even[lab["v_ego"][even] > 5][::5]
a, _ = BIG["pose"]; tp = lab["big"][even][:, a:a + 6]
res = {}
with torch.no_grad():
    for g in (0, 1, 2, 4):
        out = []
        for i in range(0, len(even), 64):
            js = even[i:i + 64] // 2
            r = torch.from_numpy(np.stack([np.concatenate([frames[j - g, 0], frames[j, 0]]) for j in js])).float() * m.in_scale
            w = torch.from_numpy(np.stack([np.concatenate([frames[j - g, 1], frames[j, 1]]) for j in js])).float() * m.in_scale
            f = m.neck(m.backbone(torch.cat([r, w])))
            feat = m.vis_fc(torch.cat([f[:len(js)].flatten(1), f[len(js):].flatten(1)], 1))
            out.append(m.cpu_heads["pose"](m.heads["pose"](m.vision_trunk(feat)))[:, :6].numpy())
        res[g] = np.concatenate(out)
print(sys.argv[1], f"{len(even)} frames; teacher vx {tp[:, 0].mean():.2f} m/s |yaw| {np.abs(tp[:, 5]).mean()*1000:.1f} mrad/s")
for g, p in res.items():
    k = np.sum(p[:, 0] * tp[:, 0]) / np.sum(tp[:, 0] ** 2)
    ky = np.sum(p[:, 5] * tp[:, 5]) / np.sum(tp[:, 5] ** 2)
    print(f"  gap {g/2:.1f}x: pose vx mean {p[:, 0].mean():.2f} (fit {k:.2f} x teacher), yaw fit {ky:.2f} x teacher, |vx err| vs teacher {np.abs(p[:,0]-tp[:,0]).mean():.2f}")
