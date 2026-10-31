"""Lead detection quality of dumped students (scripts/diag_y2s_dump.py) against the shipped model, on frames with
v > 5 and the teacher's lead_prob > 0.5 as the label (t = 0):
  - ROC AUC (threshold-free detection quality),
  - false / missed leads at 0.5,
  - false leads by the model's own lead distance (0-40, 40-60, > 60 m),
  - leave-one-route-out operating point: for each route the student's logit shift that gives the shipped model's
    false-lead rate on the other routes, applied to the held-out route (an honest estimate of a calibrated deployment).

usage: .venv/bin/python scripts/lead_quality.py <dump name> ..."""
import collections
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "outputs/diag_y2s/dump"
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
first = lambda x: x[:, 0] if x.ndim > 1 else x


def auc(y, s):
    o = np.argsort(s, kind="stable")
    r = np.empty(len(s))
    r[o] = np.arange(1, len(s) + 1)
    n1 = y.sum()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))


def load(name):
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for f in sorted((D / name).glob("*.npz")):
        z = np.load(f)
        v = (z["v"] > 5) & np.isfinite(z["rec_lead_prob"])
        r = by[f.stem[:24]]
        r["t"].append(first(z["t_lead_prob"])[v])
        r["s"].append(logit(first(z["s_lead_prob"])[v]))
        r["r"].append(z["rec_lead_prob"][v])
        r["sx"].append(z["s_lead"][v][:, 0, 0, 0])
        r["rx"].append(z["rec_lead_x"][v])
    return {k: {a: np.concatenate(b) for a, b in r.items()} for k, r in by.items()}


def main():
    bs = np.round(np.arange(-2.0, 0.51, 0.05), 2)
    for name in sys.argv[1:]:
        R = load(name)
        A = {k: np.concatenate([R[r][k] for r in R]) for k in ("t", "s", "r", "sx", "rx")}
        y, nl = A["t"] > 0.5, A["t"] <= 0.5
        pct = lambda m, d: m.sum() / d.sum() * 100
        print(f"{name} ({len(R)} routes, {len(y)} frames)")
        print(f"  AUC            student {auc(y, A['s']):.4f}   shipped {auc(y, A['r']):.4f}")
        print(f"  at 0.5         student false {pct((A['s'] > 0) & nl, nl):.2f}% missed {pct((A['s'] <= 0) & y, y):.2f}%   "
              f"shipped false {pct((A['r'] > .5) & nl, nl):.2f}% missed {pct((A['r'] <= .5) & y, y):.2f}%")
        for lo, hi in ((0, 40), (40, 60), (60, 1e9)):
            ms, mr = (A["sx"] >= lo) & (A["sx"] < hi), (A["rx"] >= lo) & (A["rx"] < hi)
            print(f"  false, own x {lo:g}-{hi if hi < 1e9 else '':} m".ljust(26) +
                  f"student {pct((A['s'] > 0) & nl & ms, nl):.2f}%   shipped {pct((A['r'] > .5) & nl & mr, nl):.2f}%")
        fp = mi = n0 = n1 = 0
        for held in R:
            t = np.concatenate([R[k]["t"] for k in R if k != held])
            s = np.concatenate([R[k]["s"] for k in R if k != held])
            r = np.concatenate([R[k]["r"] for k in R if k != held])
            target = ((r > .5) & (t <= .5)).sum() / (t <= .5).sum()
            b = max(b for b in bs if (((s + b) > 0) & (t <= .5)).sum() / (t <= .5).sum() <= target)
            h = R[held]
            fp += (((h["s"] + b) > 0) & (h["t"] <= .5)).sum()
            mi += (((h["s"] + b) <= 0) & (h["t"] > .5)).sum()
            n0, n1 = n0 + (h["t"] <= .5).sum(), n1 + (h["t"] > .5).sum()
        print(f"  route-out cal. student false {fp / n0 * 100:.2f}% missed {mi / n1 * 100:.2f}%")


if __name__ == "__main__":
    main()
