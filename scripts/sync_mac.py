"""Copy what training and evaluation need to another machine (the MacBook) over ssh, one labelled unit at a time.

Sent: scripts/, configs/, data/dedup/dedup.json and every finished unit of data/ep_label (frames_idx.npy,
frames.npy, labels.npz without ld_raw, which no training or evaluation script reads; it stays here). Each unit
goes as tar | pigz -1 | ssh | tar -xz and is marked complete on the remote side only after it is extracted, so
a rerun (after an interruption, or once more chunks are labelled) sends only what is missing. The receiving side
runs under caffeinate so the Mac does not sleep mid-transfer.

usage: SSHPASS=... scripts/py scripts/sync_mac.py chan@192.168.219.111 [--dst Documents/supercombo_distill]
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "data/mac_stage"
DROP = ("ld_raw",)


def ssh(host, cmd, **kw):
    return subprocess.run(["sshpass", "-e", "ssh", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=30", host, cmd],
                          **kw)


def send(host, src_dir: Path, names: list, dst_dir: str, marker: str | None = None):
    """tar the names (relative to src_dir, symlinks followed) into dst_dir on the host."""
    done = f" && touch {marker}" if marker else ""
    tar = subprocess.Popen(["tar", "-chf", "-", "-C", str(src_dir), *names], stdout=subprocess.PIPE)
    gz = subprocess.Popen(["pigz", "-1", "-p", "6"], stdin=tar.stdout, stdout=subprocess.PIPE)
    tar.stdout.close()
    r = ssh(host, f"mkdir -p {dst_dir} && caffeinate -i tar -xzf - -C {dst_dir}{done}", stdin=gz.stdout)
    gz.stdout.close()
    if tar.wait() or gz.wait() or r.returncode:
        raise RuntimeError(f"transfer of {names} failed")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("host")
    ap.add_argument("--dst", default="Documents/supercombo_distill", help="directory on the host (relative to its home)")
    args = ap.parse_args()
    host, dst = args.host, args.dst
    send(host, ROOT, ["scripts", "configs", "data/dedup/dedup.json"], dst)
    have = set(ssh(host, f"cd {dst}/data/ep_label 2>/dev/null && ls -a */.complete 2>/dev/null", capture_output=True,
                   text=True).stdout.split())
    have = {p.split("/")[0] for p in have}
    units = sorted(d for d in (ROOT / "data/ep_label").iterdir()
                   if (d / "labels.npz").exists() and (d / "frames.npy").exists())
    todo = [d for d in units if d.name not in have]
    size = sum((d / "frames.npy").stat().st_size + (d / "labels.npz").stat().st_size for d in todo)
    print(f"{len(units)} units here, {len(have)} already there, sending {len(todo)} ({size / 1e9:.0f} GB raw)", flush=True)
    t0, sent = time.time(), 0
    for d in todo:
        st = STAGE / d.name
        shutil.rmtree(st, ignore_errors=True)
        st.mkdir(parents=True)
        lab = np.load(d / "labels.npz")
        np.savez(st / "labels.npz", **{k: lab[k] for k in lab.files if k not in DROP})
        for f in ("frames_idx.npy", "frames.npy"):
            (st / f).symlink_to(d / f)
        # labels.npz last: on the remote side its presence follows the frames
        send(host, STAGE, [f"{d.name}/frames_idx.npy", f"{d.name}/frames.npy", f"{d.name}/labels.npz"],
             f"{dst}/data/ep_label", marker=f"{dst}/data/ep_label/{d.name}/.complete")
        shutil.rmtree(st)
        sent += (d / "frames.npy").stat().st_size + (d / "labels.npz").stat().st_size
        dt = time.time() - t0
        print(f"{d.name}: done, {sent / 1e9:.1f}/{size / 1e9:.0f} GB in {dt / 60:.0f} min "
              f"({sent / dt / 1e6:.0f} MB/s raw)", flush=True)
    print("all sent")


if __name__ == "__main__":
    sys.exit(main())
