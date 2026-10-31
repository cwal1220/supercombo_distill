"""False and missed leads of a dumped student against the shipped model (frames with v > 5, lead_prob at t = 0,
threshold 0.5 against the teacher's): the rates by teacher probability band, and what a constant shift of the
student's lead_prob logit (a bias change of the fp32 CPU head) trades between the two.

usage: .venv/bin/python scripts/diag_falselead.py <dump name> [unit prefix ...]"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "outputs/diag_y2s/dump"
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))

name = sys.argv[1]
files = sorted((D / name).glob("*.npz"))
cat = lambda k: np.concatenate([np.load(f)[k] for f in files])
v, t, s, rec = cat("v"), cat("t_lead_prob"), cat("s_lead_prob"), cat("rec_lead_prob")
t, s = (x[:, 0] if x.ndim > 1 else x for x in (t, s))
sel = v > 5
t, s, rec = t[sel], s[sel], rec[sel]
lead, nolead = t > 0.5, t <= 0.5
print(f"{len(files)} units, {sel.sum()} moving frames, teacher lead in {lead.mean() * 100:.1f}%")
rate = lambda q: (((q > 0.5) & nolead).sum() / nolead.sum() * 100, ((q <= 0.5) & lead).sum() / lead.sum() * 100)
print(f"{'model':24s} {'false %':>8s} {'missed %':>9s}")
print(f"{'shipped':24s} {rate(rec)[0]:8.2f} {rate(rec)[1]:9.2f}")
for b in (0.0, -0.25, -0.5, -0.75, -1.0, -1.5):
    q = 1 / (1 + np.exp(-(logit(s) + b)))
    print(f"{f'student logit {b:+.2f}':24s} {rate(q)[0]:8.2f} {rate(q)[1]:9.2f}")
print("\nfalse leads by teacher probability band (share of the student's / shipped false leads)")
for lo, hi in ((0, 0.1), (0.1, 0.3), (0.3, 0.5)):
    m = (t > lo) & (t <= hi)
    print(f"  teacher ({lo:.1f}, {hi:.1f}]: {m.sum():6d} frames, student {((s > 0.5) & m).sum() / max(((s > 0.5) & nolead).sum(), 1) * 100:5.1f}%, "
          f"shipped {((rec > 0.5) & m).sum() / max(((rec > 0.5) & nolead).sum(), 1) * 100:5.1f}%")
