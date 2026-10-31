"""Training throughput of a student on synthetic batches, to compare machines (CUDA, Apple MPS, CPU).

Times train.py's step (batched_forward, a stand-in loss over every head, backward, AdamW) and the bank refresh's
forward pass, then projects one epoch: --samples training samples plus --frames bank-refresh frames.

usage: python scripts/bench_train.py --device mps --arch '{"widths":[48,96,192,384,768],"depths":[1,3,6,9,3]}' --batch 32
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from student import Student  # noqa: E402
from train import batched_forward  # noqa: E402


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--arch", default="{}")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--channels-last", action="store_true", help="NHWC memory format for the conv backbone")
    ap.add_argument("--compile", action="store_true", help="torch.compile the backbone and neck")
    ap.add_argument("--samples", type=int, default=400_000, help="training samples per epoch")
    ap.add_argument("--frames", type=int, default=484_000, help="frames the bank refresh runs per epoch")
    args = ap.parse_args()
    dev, B = args.device, args.batch
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[args.dtype]
    model = Student(**json.loads(args.arch)).to(dev)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
        body = model.backbone.forward
        model.backbone.forward = lambda x: body(x.contiguous(memory_format=torch.channels_last))
    if args.compile:
        model.backbone = torch.compile(model.backbone)
        model.neck = torch.compile(model.neck)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    b = {"road": torch.randint(0, 255, (B, 12, 128, 256), dtype=torch.uint8),
         "wide": torch.randint(0, 255, (B, 12, 128, 256), dtype=torch.uint8),
         "desire": torch.zeros(B, 25, 8), "feat": F.normalize(torch.randn(B, 24, 512), dim=-1)}

    def step():
        with torch.autocast(dev, dtype=dtype, enabled=dtype is not None):
            out, _ = batched_forward(model, b, dev)
        loss = sum(v.float().pow(2).mean() for v in out.values())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    for _ in range(3):
        step()
    sync(dev)
    t0 = time.time()
    for _ in range(args.steps):
        step()
    sync(dev)
    train_sps = args.steps * B / (time.time() - t0)

    model.eval()
    x = torch.randint(0, 255, (128, 12, 128, 256), dtype=torch.uint8)

    @torch.no_grad()
    def refresh():
        with torch.autocast(dev, dtype=dtype, enabled=dtype is not None):
            xx = x.to(dev).float() * model.in_scale
            f = model.neck(model.backbone(xx))
            n = len(xx) // 2
            feat = model.vis_fc(torch.cat([f[:n].flatten(1), f[n:].flatten(1)], 1))
            model.summarizer(feat)

    refresh()
    sync(dev)
    t0 = time.time()
    for _ in range(5):
        refresh()
    sync(dev)
    refresh_fps = 5 * 64 / (time.time() - t0)
    epoch = args.samples / train_sps + args.frames / refresh_fps
    opts = "".join(f" {o}" for o, on in (("channels_last", args.channels_last), ("compile", args.compile)) if on)
    print(f"torch {torch.__version__} {dev} {args.dtype} batch {B}{opts}: train {train_sps:.0f} samples/s, refresh {refresh_fps:.0f} frames/s, "
          f"epoch ~{epoch / 60:.0f} min (compute only, {args.samples} samples + {args.frames} refresh frames)")


if __name__ == "__main__":
    main()
