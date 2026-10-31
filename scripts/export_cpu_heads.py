"""Write a student's CPU heads (the fp32 last linear layer of every head) as the .heads file edgepilot's modeld reads
next to the axmodel (edgepilot src/model/cpu_heads.h): "EPHEADS1", u32 head count, then per head char name[32]
(NUL-padded), u32 in, u32 out, f32 W[out][in] row-major, f32 b[out], all little-endian. The heads come from the same
checkpoint the axmodel was built from (quant_build.py --ckpt), so the far-lead calibration's CPU part (lead_cal.py:
the lead_prob weights scaled by a, the feature's weight -b1, the bias + b0) is in them.

--check <board npz>: reads the file back with numpy and compares it, on every frame of a board_eval.py run
(<build>/board_<unit>.npz: the NPU's hx_* features), with the checkpoint's own CPU heads in torch.

usage: .venv/bin/python scripts/export_cpu_heads.py <ckpt> <out.heads> [--check <board npz>]"""
import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from student import HEADS, POLICY_HEADS, load_student  # noqa: E402

MAGIC = b"EPHEADS1"


def write(m, path: Path):
    with open(path, "wb") as f:
        f.write(MAGIC + struct.pack("<I", len(m.cpu_heads)))
        for name in [*HEADS, *POLICY_HEADS]:
            lin = m.cpu_heads[name]
            w = lin.weight.detach().float().numpy()
            b = lin.bias.detach().float().numpy()
            assert w.shape == (b.shape[0], (HEADS | POLICY_HEADS)[name][0]), (name, w.shape)
            f.write(name.encode().ljust(32, b"\0") + struct.pack("<II", w.shape[1], w.shape[0]))
            f.write(np.ascontiguousarray(w, "<f4").tobytes() + np.ascontiguousarray(b, "<f4").tobytes())


def read(path: Path) -> dict:
    buf = path.read_bytes()
    assert buf[:8] == MAGIC
    n, = struct.unpack_from("<I", buf, 8)
    off, heads = 12, {}
    for _ in range(n):
        name = buf[off:off + 32].rstrip(b"\0").decode()
        i, o = struct.unpack_from("<II", buf, off + 32)
        off += 40
        w = np.frombuffer(buf, "<f4", o * i, off).reshape(o, i)
        off += 4 * o * i
        b = np.frombuffer(buf, "<f4", o, off)
        off += 4 * o
        heads[name] = (w, b)
    assert off == len(buf)
    return heads


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--check", type=Path, help="a board_eval.py output npz to compare on")
    args = ap.parse_args()
    m = load_student(torch.load(args.ckpt, map_location="cpu")).eval()
    write(m, args.out)
    heads = read(args.out)
    print(f"wrote {args.out} ({args.out.stat().st_size} bytes): " +
          ", ".join(f"{k} {w.shape[1]}->{w.shape[0]}" for k, (w, _) in heads.items()))
    if args.check:
        z = np.load(args.check)
        with torch.no_grad():
            ref = m.outputs({k: torch.from_numpy(z[k]) for k in z.files if k.startswith("hx_")})
        for k, (w, b) in heads.items():
            x = z["hx_" + k].astype(np.float64)
            y = x @ w.T.astype(np.float64) + b
            r = ref[k].numpy()
            print(f"  {k:17s} max |file - torch| {np.abs(y - r).max():.2e}  (|torch| max {np.abs(r).max():.2f}, {len(x)} frames)")


if __name__ == "__main__":
    main()
