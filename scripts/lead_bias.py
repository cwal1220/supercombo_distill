"""Choose the student's lead_prob operating point on training drives, not on the eval routes: the logit shift b at which
the student's false-lead rate on the training units equals the shipped model's there (both against the teacher, frames
with v > 5), then report false / missed leads on the eval routes at b. --apply writes a checkpoint whose fp32 CPU
head cpu_heads.lead_prob has its bias shifted by b (all three horizons), which is all a deployment needs to change.

usage: .venv/bin/python scripts/lead_bias.py <train dump> <eval dump> [--apply <ckpt> <out ckpt>]"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "outputs/diag_y2s/dump"
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))


def load(name):
    fs = sorted((D / name).glob("*.npz"))
    cat = lambda k: np.concatenate([np.load(f)[k] for f in fs])
    first = lambda x: x[:, 0] if x.ndim > 1 else x
    v, t, s, r = cat("v"), first(cat("t_lead_prob")), first(cat("s_lead_prob")), cat("rec_lead_prob")
    m = (v > 5) & np.isfinite(r)
    return t[m], s[m], r[m], len(fs)


def rates(t, q):
    nolead, lead = t <= 0.5, t > 0.5
    return ((q > 0.5) & nolead).sum() / nolead.sum() * 100, ((q <= 0.5) & lead).sum() / lead.sum() * 100


def main():
    tr, ev = sys.argv[1], sys.argv[2]
    t, s, r, n = load(tr)
    ship_fp = rates(t, r)[0]
    shifted = lambda s, b: 1 / (1 + np.exp(-(logit(s) + b)))
    bs = np.round(np.arange(-2.0, 0.51, 0.05), 2)
    fp = np.array([rates(t, shifted(s, b))[0] for b in bs])
    b = float(bs[fp <= ship_fp].max())
    print(f"train dump {tr} ({n} units, {len(t)} frames): shipped false {ship_fp:.2f}% / missed {rates(t, r)[1]:.2f}%, "
          f"student {rates(t, s)[0]:.2f}% / {rates(t, s)[1]:.2f}% -> logit shift {b:+.2f} gives {rates(t, shifted(s, b))[0]:.2f}% / {rates(t, shifted(s, b))[1]:.2f}%")
    t, s, r, n = load(ev)
    print(f"eval dump {ev} ({n} units, {len(t)} frames): shipped {rates(t, r)[0]:.2f}% / {rates(t, r)[1]:.2f}%, student {rates(t, s)[0]:.2f}% / "
          f"{rates(t, s)[1]:.2f}%, student shifted {b:+.2f}: {rates(t, shifted(s, b))[0]:.2f}% / {rates(t, shifted(s, b))[1]:.2f}% (false / missed)")
    if "--apply" in sys.argv:
        import torch
        i = sys.argv.index("--apply")
        ck = torch.load(sys.argv[i + 1], map_location="cpu")
        ck["model"]["cpu_heads.lead_prob.bias"] = ck["model"]["cpu_heads.lead_prob.bias"] + b
        ck["lead_bias"] = b
        torch.save(ck, sys.argv[i + 2])
        print(f"saved {sys.argv[i + 2]} (cpu_heads.lead_prob.bias {b:+.2f})")


if __name__ == "__main__":
    main()
