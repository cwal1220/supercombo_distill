"""Grow a trained R18 student's backbone by ImageNet ResNet-18 layer4 (arch r18_layers 4, in place of the stride-2
RepConv after layer3) without losing what the heads learned: layer4 (pretrained) and a new neck learn to reproduce the
old neck's output on training frames (MSE, photometric augmentation, everything else frozen and copied), so the new
student starts close to the old function and then trains end to end with the extra pretrained depth (FitNets-style
hint training, Romero et al. 2015).

The old checkpoint may be a QAT one (BN folded into the convs): the new student folds layer4's ImageNet BN the same way.

usage: .venv/bin/python scripts/mimic.py --ckpt ckpt/mac_lr/init_h64_v192.pt --out ckpt/mac_l4/init.pt [--steps 3000]
"""
import argparse
import sys
import threading
import queue
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from student import R18Backbone, Student, load_student  # noqa: E402
from train import EVAL_ROUTES, augment, default_device  # noqa: E402
from zframes import load_frames  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def batches(units, pairs, seed):
    """Endless [2 * pairs, 12, 128, 256] batches: the road then the wide frame pairs (kept frames j-2, j) of `pairs`
    random frames, a quarter from each of four random units."""
    rng = np.random.default_rng(seed)
    frames = {}
    while True:
        road, wide = [], []
        for u in rng.choice(len(units), 4, replace=len(units) < 4):
            if u not in frames:
                frames[u] = load_frames(units[u])
            fr = frames[u]
            for j in rng.integers(2, len(fr), pairs // 4):
                road.append(np.concatenate([fr[j - 2, 0], fr[j, 0]]))
                wide.append(np.concatenate([fr[j - 2, 1], fr[j, 1]]))
        yield np.stack(road + wide)


def prefetch(gen, n=4):
    q = queue.Queue(n)
    threading.Thread(target=lambda: [q.put(x) for x in gen], daemon=True).start()
    while True:
        yield q.get()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32, help="frame pairs per step (x2 cameras)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default=default_device())
    a = ap.parse_args()
    dev = a.device
    ck = torch.load(a.ckpt, map_location="cpu")
    old = load_student(ck).to(dev)
    arch = {**ck["arch"], "r18_layers": 4}
    new = Student(**arch, pretrained=True)                       # ImageNet layer4
    if ck.get("qat"):
        new.backbone.fuse_bn()
    st = new.state_dict()
    copied = [k for k in st if k in ck["model"] and ck["model"][k].shape == st[k].shape]
    st.update({k: ck["model"][k] for k in copied})
    new.load_state_dict(st)
    fresh = [k for k in st if k not in copied]
    print(f"copied {len(copied)} tensors; new: {sorted({k.rsplit('.', 2)[0] for k in fresh})}", flush=True)
    new = new.to(dev).train()
    params = list(new.backbone.layers[3].parameters()) + list(new.neck.parameters())
    for p in new.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
    data = ROOT / "data/ep_label"
    units = sorted(d for d in data.iterdir() if (d / "labels.npz").exists() and not any(d.name.startswith(r) for r in EVAL_ROUTES))
    held = [u for u in units if u.name.count("__") < 2][-3:]       # held-out real units for the final check
    train_units = [u for u in units if u not in held]
    gen = prefetch(batches(train_units, a.batch, 0))
    t0, run = time.time(), 0.0
    for step in range(a.steps):
        x = torch.from_numpy(next(gen)).to(dev).float()
        x = augment(x) * new.in_scale
        with torch.no_grad():
            tgt = old.neck(old.backbone(x))
        with torch.autocast(dev, dtype=torch.bfloat16):
            y = new.neck(new.backbone(x))
        loss = F.mse_loss(y.float(), tgt) / tgt.pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        run = 0.98 * run + 0.02 * loss.item() if step else loss.item()
        if step % 100 == 0 or step == a.steps - 1:
            print(f"step {step}: relative mse {run:.4f} ({time.time() - t0:.0f} s)", flush=True)
    new.eval()
    with torch.no_grad():
        g = batches(held, 64, 1)
        e = []
        for _ in range(10):
            x = torch.from_numpy(next(g)).to(dev).float() * new.in_scale
            tgt = old.neck(old.backbone(x))
            e.append((F.mse_loss(new.neck(new.backbone(x)), tgt) / tgt.pow(2).mean()).item())
    print(f"held-out relative mse {np.mean(e):.4f}", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": {k: v.cpu() for k, v in new.state_dict().items()}, "arch": arch, "epoch": ck.get("epoch"),
                "fused": False, "qat": ck.get("qat"), "mimic_from": str(a.ckpt)}, a.out)
    print(f"saved {a.out}", flush=True)


if __name__ == "__main__":
    main()
