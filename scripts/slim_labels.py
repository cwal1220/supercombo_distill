"""Move what no training or evaluation reads out of labels.npz of Mac-labelled units, to the external drive: ld_raw (LD's
raw outputs; the PC never sent them for the units labelled there) and, outside the eval routes, the all-NaN small
(ep_label.py --no-small). Per unit the moved arrays go to <offload>/<unit>.npz first and are read back and compared;
only then is labels.npz rewritten (uncompressed, train.npz_mmap maps its members) under a temporary name and swapped in,
so readers that already mapped the old file keep it. Restore: add the arrays of <offload>/<unit>.npz back.

usage: .venv/bin/python scripts/slim_labels.py [data/ep_label] [offload dir]
"""
import os
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import EVAL_ROUTES  # noqa: E402

root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/ep_label")
off = Path(sys.argv[2] if len(sys.argv) > 2 else "/Volumes/Backup/supercombo_distill_offload/labels_extra")
off.mkdir(parents=True, exist_ok=True)
saved = 0
for d in sorted(root.iterdir()):
    f = d / "labels.npz"
    if not f.exists():
        continue
    with zipfile.ZipFile(f) as z:
        names = [i.filename[:-4] for i in z.infolist()]
    if "ld_raw" not in names:
        continue
    ev = any(d.name.startswith(e) for e in EVAL_ROUTES)
    move = ["ld_raw"] + ([] if ev or "small" not in names else ["small"])
    before = f.stat().st_size
    with np.load(f) as src:
        moved = {k: src[k] for k in move}
        keep = {k: src[k] for k in names if k not in move}
    dst = off / f"{d.name}.npz"
    np.savez(dst, **moved)
    with np.load(dst) as chk:                       # the offloaded copy must read back identical
        for k, v in moved.items():
            assert np.array_equal(chk[k], v, equal_nan=True), (d.name, k)
    tmp = d / "labels.slim.npz"
    np.savez(tmp, **keep)
    os.replace(tmp, f)
    saved += before - f.stat().st_size
    print(f"{d.name}: {before / 1e9:.2f} -> {f.stat().st_size / 1e9:.2f} GB ({', '.join(move)} -> {dst})", flush=True)
print(f"saved {saved / 1e9:.1f} GB")
