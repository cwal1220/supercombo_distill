"""Closed-loop run of a Pulsar2-quantized student on eval units, to check eval_student.py --s8 (the PyTorch S8
simulation) against Pulsar2's own numerics.

The model is quant_build.py's QuantAxModel (<build>/quant/quant_axmodel.onnx) run by `pulsar2 run --mode Reference`,
which matched the compiled m.axmodel's simulation to 1e-4 on calibration samples (and runs at ~0.1 s a frame against
~20 s for the compiled model under Rosetta). The closed loop takes two passes, since the hidden state depends on the
frames only: pass 1 gives every frame's hidden (and the vision heads), pass 2 the policy with features_buffer built
from the pass-1 hidden as eval_student.py does. Frames go through docker in chunks (float32 inputs, 3.1 MB a frame).

Prints eval_student's metrics on each unit for fp32, the S8 simulation and Pulsar2, and saves the raw outputs in
<build>/p2eval_<unit>.npz (--reuse: units that already have one are not run again). With several units it also prints
the frame-weighted means over them, with the shipped model's recorded board outputs, as eval_student.py does.

usage: .venv/bin/python scripts/quant_eval_p2.py --ckpt ckpt/mac_base_flip8/last.pt --build outputs/quant_flip8 \
           --units 2026-10-25--22-42-26-175__015 [--chunk 400]
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_student as E  # noqa: E402
from student import HEADS, POLICY_HEADS, load_student  # noqa: E402
from train import BIG  # noqa: E402
from zframes import load_frames  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUTS = ["out_hidden"] + ["hx_" + k for k in [*HEADS, *POLICY_HEADS]]
WIDTH = {"out_hidden": 512, **{"hx_" + k: w for k, (w, _) in {**HEADS, **POLICY_HEADS}.items()}}


def inputs(frames, lab, t, hist):
    j = t // 2
    des = np.zeros((100, 8), np.float32)
    lo = max(0, t - 99)
    des[100 - (t + 1 - lo):] = lab["pulse"][lo:t + 1]
    idx = np.arange(t - 96, t, 4) // 2
    fb = np.zeros((24, 512), np.float32)
    fb[idx >= 0] = hist[idx[idx >= 0]]
    return {"input_imgs": np.concatenate([frames[j - 2, 0], frames[j, 0]])[None].astype(np.float32),
            "big_input_imgs": np.concatenate([frames[j - 2, 1], frames[j, 1]])[None].astype(np.float32),
            "desire": des.reshape(25, 4, 8).max(1)[None], "features_buffer": fb[None],
            "traffic_convention": np.array([[1.0, 0.0]], np.float32)}


def p2_pass(build: Path, frames, lab, even, hist, chunk: int) -> dict[str, np.ndarray]:
    """pulsar2 run (Reference) over the frames `even`, features_buffer from hist; outputs name -> [N, width]."""
    work = build / "p2eval_tmp"
    res = {o: [] for o in OUTS}
    for s in range(0, len(even), chunk):
        shutil.rmtree(work, ignore_errors=True)
        ts = even[s:s + chunk]
        for i, t in enumerate(ts):
            d = work / "in" / str(i)
            d.mkdir(parents=True)
            for k, a in inputs(frames, lab, t, hist).items():
                a.tofile(d / f"{k}.bin")
        (work / "list.txt").write_text("".join(f"{i}\n" for i in range(len(ts))))
        rel = work.relative_to(ROOT)
        r = subprocess.run(["docker", "run", "--rm", "--platform", "linux/amd64", "--entrypoint", "/bin/bash",
                            "-v", f"{ROOT}:/data/supercombo_distill", "-w", "/data/supercombo_distill", "pulsar2:6.0-lite", "-c",
                            f"pulsar2 run --model {build.relative_to(ROOT)}/quant/quant_axmodel.onnx --input_dir {rel}/in "
                            f"--output_dir {rel}/out --list {rel}/list.txt --mode Reference --target_hardware AX620E"],
                           capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(r.stdout[-2000:] + r.stderr[-2000:])
        for o in OUTS:
            res[o] += [np.fromfile(work / "out" / str(i) / f"{o}.bin", np.float32) for i in range(len(ts))]
        print(f"    {s + len(ts)}/{len(even)}", flush=True)
    shutil.rmtree(work, ignore_errors=True)
    return {o: np.stack(v).reshape(len(even), WIDTH[o]) for o, v in res.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--build", type=Path, required=True)
    ap.add_argument("--units", nargs="+", required=True)
    ap.add_argument("--data", type=Path, default=ROOT / "data/ep_label")
    ap.add_argument("--chunk", type=int, default=400)
    ap.add_argument("--reuse", action="store_true", help="take <build>/p2eval_<unit>.npz where it exists")
    args = ap.parse_args()
    build = args.build.resolve()
    m = load_student(torch.load(args.ckpt, map_location="cpu"))
    keys = ["lat_acc", "long_acc", "plan_y2s_cm", "lanes_vs_ld_cm", "width_vs_ld_cm", "lead_x_m", "gas2",
            "lead_x_lf", "false_lead_pct"]
    weight = {"lead_x_lf": "n_lead", "false_lead_pct": "n_nolead"}   # the rest: moving frames
    total = {}
    for u in args.units:
        d = args.data / u
        lab = np.load(d / "labels.npz")
        n = len(lab["big"])
        even = np.arange(4, n - n % 2, 2)
        saved = build / f"p2eval_{u}.npz"
        if args.reuse and saved.exists():
            z = np.load(saved)
            assert np.array_equal(z["even"], even), u
            hx = {o: z[o] for o in OUTS}
            print(f"{u}: Pulsar2 outputs from {saved.name}", flush=True)
        else:
            frames = load_frames(d)
            hist = np.zeros(((n + 1) // 2, 512), np.float32)
            print(f"{u}: pass 1 (hidden), {len(even)} frames", flush=True)
            p1 = p2_pass(build, frames, lab, even, hist, args.chunk)
            hist[even // 2] = p1["out_hidden"]
            print(f"{u}: pass 2 (policy)", flush=True)
            p2 = p2_pass(build, frames, lab, even, hist, args.chunk)
            hx = {o: p1[o] for o in OUTS}
            for k in [*POLICY_HEADS, *m.temporal_heads]:   # the heads that see the history: from pass 2
                hx["hx_" + k] = p2["hx_" + k]
            np.savez_compressed(saved, even=even, **hx)
        with torch.no_grad():
            raw = {k: v.numpy() for k, v in m.outputs({o: torch.from_numpy(v) for o, v in hx.items()}).items()}
        v = lab["v_ego"][even]
        sel = v > 5.0
        big = E.from_layout(lab["big"][even], BIG)
        ld = lab["ld_lanes"][even]
        rows = {"board NPU (shipped)": {k: x[even] for k, x in E.recorded(lab).items()},
                "fp32": E.run_student(args.ckpt, d, lab, even, "cpu"),
                "S8 sim (eval_student --s8)": E.run_student(args.ckpt, d, lab, even, "cpu", s8=True),
                "Pulsar2 (Reference)": E.parse_raw(raw)}
        print(f"{'model':28s} " + " ".join(f"{k:>15s}" for k in keys))
        for name, q in rows.items():
            r = E.compare(q, big, ld, v, sel)
            total.setdefault(name, []).append((sel.sum(), r))
            print(f"{name:28s} " + " ".join(f"{r[k]:15.4f}" for k in keys), flush=True)
    if len(args.units) > 1:
        print(f"mean over {len(args.units)} units (frame-weighted)")
        print(f"{'model':28s} " + " ".join(f"{k:>15s}" for k in keys))
        for name, rs in total.items():
            row = []
            for k in keys:
                ok = [(r[weight[k]] if k in weight else n, r[k]) for n, r in rs if np.isfinite(r[k])]
                row.append(sum(n * v for n, v in ok) / sum(n for n, _ in ok) if ok else np.nan)
            print(f"{name:28s} " + " ".join(f"{x:15.4f}" for x in row), flush=True)


if __name__ == "__main__":
    main()
