"""The student's training step in MLX (Apple GPU), to compare with PyTorch MPS (bench_train.py) on a Mac.

The same network as student.py (RepVGG backbone in its training form: 3x3+BN, 1x1+BN, identity BN; neck, the
vision MLPs, the temporal MLP and every head with its CPU linear layer), in MLX's NHWC layout, with the same
stand-in loss (mean square of every output) and AdamW, on synthetic batches. MLX has no autocast: --dtype bf16 /
fp16 casts the whole model (pure low-precision training), fp32 keeps it in fp32.

usage: python scripts/bench_mlx.py [--arch '{"widths":[48,96,192,384,768],"depths":[1,3,6,9,3]}'] [--batch 64]
"""
from __future__ import annotations

import argparse
import json
import time
from functools import partial

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

HEADS = {"meta": (64, 55), "desire_pred": (32, 32), "pose": (32, 12), "wide_from_device": (32, 6),
         "road_transform": (32, 12), "lanes": (64, 528), "lane_prob": (16, 8), "road_edges": (32, 264),
         "lead": (64, 144), "lead_prob": (16, 3)}
POLICY_HEADS = {"plan": (256, 990), "desire_state": (32, 8)}


class RepConv(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c3, self.b3 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm(cout)
        self.c1, self.b1 = nn.Conv2d(cin, cout, 1, stride, 0, bias=False), nn.BatchNorm(cout)
        self.has_idn = cin == cout and stride == 1
        if self.has_idn:
            self.idn = nn.BatchNorm(cout)

    def __call__(self, x):
        y = self.b3(self.c3(x)) + self.b1(self.c1(x))
        if self.has_idn:
            y = y + self.idn(x)
        return nn.relu(y)


def mlp(cin, cout, hidden=None, depth=1):
    hidden = hidden or cout
    layers, c = [], cin
    for _ in range(depth - 1):
        layers += [nn.Linear(c, hidden), nn.ReLU()]
        c = hidden
    return nn.Sequential(*layers, nn.Linear(c, cout), nn.ReLU())


class Student(nn.Module):
    def __init__(self, widths=(32, 48, 96, 192, 384), depths=(1, 2, 3, 4, 2), strides=(2, 2, 2, 2, 2), neck_ch=64,
                 feat=1024, hidden=512, policy_tokens=9, policy_width=512, head_width=512):
        super().__init__()
        layers, cin = [], 12
        for i, (w, d) in enumerate(zip(widths, depths)):
            layers.append(RepConv(cin, w, strides[i]))
            layers += [RepConv(w, w) for _ in range(d - 1)]
            cin = w
        self.body = layers
        s = 1
        for x in strides:
            s *= x
        hw = (128 // s) * (256 // s)
        self.neck = nn.Conv2d(cin, neck_ch, 1)
        self.vis_fc = mlp(2 * neck_ch * hw, feat)
        self.summarizer = nn.Sequential(nn.Linear(feat, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.vision_trunk = mlp(feat, head_width)
        self.heads = {k: mlp(head_width, w) for k, (w, _) in HEADS.items()}
        self.policy_tokens = policy_tokens
        self.desire_enc = mlp(25 * 8, 64)
        self.traffic_enc = mlp(2, 16)
        self.temporal = mlp(policy_tokens * hidden + 64 + 16, policy_width, policy_width, depth=2)
        self.policy_heads = {k: mlp(policy_width, w, 256, depth=2) for k, (w, _) in POLICY_HEADS.items()}
        self.cpu_heads = {k: nn.Linear(w, o) for k, (w, o) in {**HEADS, **POLICY_HEADS}.items()}

    def features(self, x):   # x: [2B, 128, 256, 12] float (road images then wide)
        for m in self.body:
            x = m(x)
        f = nn.relu(self.neck(x))
        n = f.shape[0] // 2
        return self.vis_fc(mx.concatenate([f[:n].reshape(n, -1), f[n:].reshape(n, -1)], 1))

    def __call__(self, x, desire, feat_buf, traffic):
        b = self.features(x)
        h = self.summarizer(b)
        hidden = h / mx.sqrt((h * h).sum(-1, keepdims=True) + 1e-12)
        trunk = self.vision_trunk(b)
        hx = {k: m(trunk) for k, m in self.heads.items()}
        hist = feat_buf[:, -(self.policy_tokens - 1):].reshape(feat_buf.shape[0], -1)
        p = self.temporal(mx.concatenate([hist, hidden, self.desire_enc(desire.reshape(desire.shape[0], -1)),
                                          self.traffic_enc(traffic)], 1))
        for k, m in self.policy_heads.items():
            hx[k] = m(p)
        return {k: self.cpu_heads[k](v) for k, v in hx.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", default="{}")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--samples", type=int, default=400_000)
    ap.add_argument("--frames", type=int, default=484_000)
    args = ap.parse_args()
    dt = {"bf16": mx.bfloat16, "fp16": mx.float16, "fp32": mx.float32}[args.dtype]
    B = args.batch
    model = Student(**json.loads(args.arch))
    model.set_dtype(dt)
    mx.eval(model.parameters())
    opt = optim.AdamW(learning_rate=1e-4)
    imgs = mx.random.randint(0, 255, (2 * B, 128, 256, 12)).astype(mx.uint8)
    desire = mx.zeros((B, 25, 8), dt)
    feat = mx.random.normal((B, 24, 512)).astype(dt)
    traffic = mx.broadcast_to(mx.array([[1.0, 0.0]], dt), (B, 2))

    def loss_fn(m, x, d, f, t):
        out = m(x.astype(dt) * (1 / 64), d, f, t)
        return sum((v.astype(mx.float32) ** 2).mean() for v in out.values())

    grad_fn = nn.value_and_grad(model, loss_fn)
    state = [model.state, opt.state]

    def step(x, d, f, t):
        loss, g = grad_fn(model, x, d, f, t)
        opt.update(model, g)
        return loss

    if not args.no_compile:
        step = partial(mx.compile, inputs=state, outputs=state)(step)
    for _ in range(3):
        mx.eval(step(imgs, desire, feat, traffic), state)
    t0 = time.time()
    for _ in range(args.steps):
        mx.eval(step(imgs, desire, feat, traffic), state)
    train_sps = args.steps * B / (time.time() - t0)

    model.eval()
    x128 = mx.random.randint(0, 255, (128, 128, 256, 12)).astype(mx.uint8)

    def fwd(x):
        return model.summarizer(model.features(x.astype(dt) * (1 / 64)))

    if not args.no_compile:
        fwd = mx.compile(fwd)
    mx.eval(fwd(x128))
    t0 = time.time()
    for _ in range(5):
        mx.eval(fwd(x128))
    refresh_fps = 5 * 64 / (time.time() - t0)
    epoch = args.samples / train_sps + args.frames / refresh_fps
    print(f"mlx {mx.__version__} {args.dtype} batch {B}{'' if not args.no_compile else ' no-compile'}: train "
          f"{train_sps:.0f} samples/s, refresh {refresh_fps:.0f} frames/s, epoch ~{epoch / 60:.0f} min")


if __name__ == "__main__":
    main()
