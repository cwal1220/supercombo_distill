"""TensorBoard scalars from a train.log, for runs started before train.py wrote TensorBoard itself.

train.log prints, every 200 steps, the current batch loss and each head's mean since the epoch started; this
turns those running means back into per-window means (loss/<head>) and writes them with lr and samples/s to
<log dir>/tb (exact for the first epoch; later epochs start their windows at the previous epoch's last printed step).
--follow keeps reading as the log grows and stops after the run's last "epoch N done" line.

usage: python scripts/log2tb.py ckpt/scale/s05_base/train.log [--follow]
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

STEP = re.compile(r"ep (\d+) step (\d+)/(\d+) loss (\S+) (.*) \| (\d+) samples/s lr (\S+)")
DONE = re.compile(r"epoch (\d+) done in (\d+) s, bank refreshed in (\d+) s")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", type=Path)
    ap.add_argument("--follow", action="store_true")
    args = ap.parse_args()
    tb = SummaryWriter(str(args.log.parent / "tb"), flush_secs=10)
    prev, ep_start, last_step, total = {}, 0, 0, None   # prev: (cnt, running means) at the last line of this epoch
    f = open(args.log)
    while True:
        line = f.readline()
        if not line:
            if not args.follow:
                break
            time.sleep(10)
            continue
        m = STEP.search(line)
        if m:
            ep, step, total = int(m[1]), int(m[2]), int(m[3])
            if ep != prev.get("ep"):
                prev, ep_start = {"ep": ep, "cnt": 0, "mean": {}}, last_step
            cnt = step - ep_start
            toks = m[5].split()
            means = {toks[i]: float(toks[i + 1]) for i in range(0, len(toks), 2)}
            for k, v in means.items():   # window mean = (sum now - sum at the previous line) / steps between
                w = (v * cnt - prev["mean"].get(k, 0.0) * prev["cnt"]) / (cnt - prev["cnt"])
                tb.add_scalar(f"loss/{k}", w, step)
            tb.add_scalar("loss/total_batch", float(m[4]), step)
            tb.add_scalar("train/samples_per_s", float(m[6]), step)
            tb.add_scalar("train/lr", float(m[7]), step)
            prev.update(cnt=cnt, mean=means)
            last_step = step
            continue
        m = DONE.search(line)
        if m:
            tb.add_scalar("epoch/minutes", int(m[2]) / 60, int(m[1]))
            tb.add_scalar("epoch/bank_refresh_s", int(m[3]), int(m[1]))
            tb.flush()
            if total and last_step + 200 > total:    # the last epoch is done
                break
    tb.close()


if __name__ == "__main__":
    main()
