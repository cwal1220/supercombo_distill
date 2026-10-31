"""Measure a Pulsar2-built student on the MaixCAM2: NPU latency (ax_run_model) and the closed-loop outputs on eval
units (board_run.py, pyaxengine), compared with fp32, the S8 simulation (eval_student --s8) and, when
<build>/p2eval_<unit>.npz exists (quant_eval_p2.py), Pulsar2's Reference run.

edgepilot.service holds the NPU, so the board part stops it and starts it again on exit (also on failure).
SSH is password auth: the password comes from the SSHPASS environment variable (never a file) through a temporary
SSH_ASKPASS helper that echoes it. Board files go to /root/distill (model as <name>.axmodel, units/<unit>/ with
frames.npy and pulse.npy, uploaded once). Writes <build>/board_<unit>.npz and logs/board_<name>_ax_run_model.txt.

usage: SSHPASS=<board password> .venv/bin/python scripts/board_eval.py --ckpt ckpt/mac_base_flip8/last.pt \
           --build outputs/quant_flip8 --name flip8 --units 2026-10-25--22-42-26-175__015 [--repeat 500]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_student as E  # noqa: E402
from quant_eval_p2 import OUTS  # noqa: E402
from student import POLICY_HEADS, load_student  # noqa: E402
from train import BIG  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/root/distill"


class Board:
    def __init__(self, host: str):
        if "SSHPASS" not in os.environ:
            sys.exit("set SSHPASS to the board password")
        self.host = host
        self.tmp = tempfile.TemporaryDirectory()
        ask = Path(self.tmp.name) / "askpass.sh"
        ask.write_text('#!/bin/sh\necho "$SSHPASS"\n')
        ask.chmod(0o700)
        self.env = {**os.environ, "SSH_ASKPASS": str(ask), "SSH_ASKPASS_REQUIRE": "force"}

    def ssh(self, cmd: str, check=True) -> str:
        r = subprocess.run(["ssh", "-o", "ConnectTimeout=10", self.host, cmd], env=self.env, stdin=subprocess.DEVNULL,
                           capture_output=True, text=True)
        if check and r.returncode:
            raise RuntimeError(f"{cmd}\n{r.stdout[-3000:]}{r.stderr[-3000:]}")
        return r.stdout + r.stderr

    def put(self, src: Path, dst: str):
        subprocess.run(["scp", "-q", str(src), f"{self.host}:{dst}"], env=self.env, stdin=subprocess.DEVNULL, check=True)

    def get(self, src: str, dst: Path):
        subprocess.run(["scp", "-q", f"{self.host}:{src}", str(dst)], env=self.env, stdin=subprocess.DEVNULL, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--build", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--units", nargs="+", required=True)
    ap.add_argument("--data", type=Path, default=ROOT / "data/ep_label")
    ap.add_argument("--host", default="root@192.168.219.107")
    ap.add_argument("--repeat", type=int, default=500)
    args = ap.parse_args()
    b = Board(args.host)
    model = f"{REMOTE}/{args.name}.axmodel"
    b.ssh(f"mkdir -p {REMOTE}/units")
    b.put(args.build / "m.axmodel", model)
    b.put(Path(__file__).with_name("board_run.py"), f"{REMOTE}/board_run.py")
    for u in args.units:
        have = b.ssh(f"test -f {REMOTE}/units/{u}/pulse.npy && echo yes", check=False).strip()
        if have != "yes":
            print(f"{u}: uploading frames", flush=True)
            b.ssh(f"mkdir -p {REMOTE}/units/{u}")
            b.put(args.data / u / "frames.npy", f"{REMOTE}/units/{u}/frames.npy")
            pulse = Path(b.tmp.name) / "pulse.npy"
            np.save(pulse, np.load(args.data / u / "labels.npz")["pulse"])
            b.put(pulse, f"{REMOTE}/units/{u}/pulse.npy")

    runs = "; ".join(f"python3 board_run.py {model} units/{u} out_{args.name}_{u}.npz" for u in args.units)
    print("board: edgepilot.service stopped for the run", flush=True)
    out = b.ssh(f"cd {REMOTE} && trap 'systemctl start edgepilot.service' EXIT && systemctl stop edgepilot.service && "
                f"sleep 2 && ax_run_model -m {model} -w 20 -r {args.repeat} && {runs}")
    print("board: edgepilot.service", b.ssh("systemctl is-active edgepilot.service", check=False).strip())
    log = ROOT / f"logs/board_{args.name}_ax_run_model.txt"
    log.write_text(out)
    print("\n".join(l for l in out.splitlines() if any(k in l for k in ("min", "max", "avg", "median", "frames"))))

    m = load_student(torch.load(args.ckpt, map_location="cpu"))
    keys = ["lat_acc", "long_acc", "plan_y2s_cm", "lanes_vs_ld_cm", "width_vs_ld_cm", "lead_x_m", "gas2"]
    for u in args.units:
        d = args.data / u
        res = args.build / f"board_{u}.npz"
        b.get(f"{REMOTE}/out_{args.name}_{u}.npz", res)
        r = np.load(res)
        even = r["even"]
        lab = np.load(d / "labels.npz")
        v = lab["v_ego"][even]
        sel = v > 5.0
        big = E.from_layout(lab["big"][even], BIG)
        ld = lab["ld_lanes"][even]
        rows = {"fp32": E.run_student(args.ckpt, d, lab, even, "cpu"),
                "S8 sim (eval_student --s8)": E.run_student(args.ckpt, d, lab, even, "cpu", s8=True)}
        p2 = args.build / f"p2eval_{u}.npz"
        for name, f in [("Pulsar2 (Reference)", p2), ("board NPU", res)]:
            if f.exists():
                hx = np.load(f)
                with torch.no_grad():
                    raw = m.outputs({o: torch.from_numpy(hx[o]) for o in OUTS})
                rows[name] = E.parse_raw({k: x.numpy() for k, x in raw.items()})
        print(f"{u} ({len(even)} frames)")
        if p2.exists():
            q = np.load(p2)
            for o in ["out_hidden", *("hx_" + k for k in POLICY_HEADS)]:
                rel = np.linalg.norm(r[o] - q[o]) / np.linalg.norm(q[o])
                print(f"  {o}: board vs Pulsar2 rel err {rel:.4f}")
        print(f"{'model':28s} " + " ".join(f"{k:>15s}" for k in keys))
        for name, q in rows.items():
            c = E.compare(q, big, ld, v, sel)
            print(f"{name:28s} " + " ".join(f"{c[k]:15.4f}" for k in keys), flush=True)


if __name__ == "__main__":
    main()
