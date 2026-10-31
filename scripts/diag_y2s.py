"""Slice the plan_y2s error (|model - teacher| of plan y at 2 s, cm, frames with v > 5) of the dumped students and
the small / shipped models by route, speed, curve, lane change, horizon, time in route, and relate it to the
current-motion (t0) estimates. Reads the dumps of scripts/diag_y2s_dump.py in outputs/diag_y2s/dump/<name>.

usage: .venv/bin/python scripts/diag_y2s.py [dump names, default base5 aug10 aug10_nohist]"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
S = ROOT / "outputs/diag_y2s"
sys.path.insert(0, str(ROOT / "scripts"))
from eval_student import OP  # noqa: E402

T = np.array([10.0 * (i / 32) ** 2 for i in range(33)])
EVAL = ["2026-10-25--22-42-26-175__000", "2026-10-25--22-42-26-175__015", "2026-10-05--00-07-29-810__000",
        "2026-10-05--16-28-44-410__000", "2026-10-05--16-28-44-410__015", "2026-09-27--02-44-40-679__000",
        "2026-09-27--02-44-40-679__015"]
TRAIN = ["2026-10-04--12-32-26-070__000", "2026-10-10--11-16-58-871__000", "2026-10-10--17-34-10-953__000"]
at = lambda p, t, c=1: np.array([np.interp(t, T, q[:, c]) for q in p])


def load(units, students):
    """Concatenated per-frame arrays: models' y at several horizons and t0 motion, context."""
    D = {}
    for u in units:
        lab = np.load(ROOT / "data/ep_label" / u / "labels.npz")
        z0 = np.load(S / "dump" / students[0] / f"{u}.npz")
        even = z0["even"]
        a, b = OP["plan"]
        small = lab["small"][even][:, a:b][:, :495].reshape(-1, 33, 15)
        plans = {"teacher": z0["t_mu"], "small": small}
        for s in students:
            f = S / "dump" / s / f"{u}.npz"
            if f.exists():
                plans[s] = np.load(f)["s_plan"]
        rows = {"unit": np.full(len(even), u), "v": z0["v"], "yaw_rate": z0["yaw_rate"], "desire": z0["desire"],
                "tsec": (even - even[0]) / 20.0, "t_sig_y2": np.exp(at(z0["t_logstd"], 2.0)),
                "shipped_y": np.stack([at(lab["rec_plan"][even], t) for t in (0.5, 1, 2, 4)], 1),
                "shipped_r0": lab["rec_plan_yaw_rate"][even][:, 0]}
        for k, p in plans.items():
            rows[f"{k}_y"] = np.stack([at(p, t) for t in (0.5, 1, 2, 4)], 1)
            rows[f"{k}_r0"] = p[:, 0, 14]     # yaw rate now
            rows[f"{k}_vx0"] = p[:, 0, 3]
            rows[f"{k}_vy0"] = p[:, 0, 4]
        for k, x in rows.items():
            D.setdefault(k, []).append(x)
    return {k: np.concatenate(v) for k, v in D.items()}


def table(D, models, title):
    sel = D["v"] > 5
    ty = D["teacher_y"]
    err = {m: np.abs(D[f"{m}_y"][:, 2] - ty[:, 2]) * 100 for m in models}
    print(f"\n#### {title}  ({sel.sum()} frames)")
    hdr = f"{'slice':34s} {'n%':>5s} " + " ".join(f"{m:>12s}" for m in models)
    print(hdr)
    def row(name, mask):
        mask = mask & sel
        print(f"{name:34s} {mask.sum() / sel.sum() * 100:5.1f} " + " ".join(f"{err[m][mask].mean():12.2f}" for m in models))
    row("all", sel)
    for u in np.unique(D["unit"]):
        row("unit " + u[:20] + u[-5:], D["unit"] == u)
    for lo, hi in [(5, 10), (10, 20), (20, 30), (30, 99)]:
        row(f"v {lo}-{hi} m/s", (D["v"] >= lo) & (D["v"] < hi))
    tyabs = np.abs(ty[:, 2])
    for lo, hi in [(0, .1), (.1, .3), (.3, 1), (1, 3), (3, 99)]:
        row(f"|teacher y2s| {lo}-{hi} m", (tyabs >= lo) & (tyabs < hi))
    row("desire != none (lane change etc.)", D["desire"] != 0)
    row("desire == none", D["desire"] == 0)
    for lo, hi in [(0, 5), (5, 20), (20, 1e9)]:
        row(f"time in unit {lo}-{hi if hi < 1e8 else 'end'} s", (D["tsec"] >= lo) & (D["tsec"] < hi))
    yr = np.abs(D["yaw_rate"])
    for lo, hi in [(0, .01), (.01, .05), (.05, 9)]:
        row(f"|car yaw rate| {lo}-{hi} rad/s", (yr >= lo) & (yr < hi))
    print("-- horizon (cm):")
    for i, t in enumerate((0.5, 1, 2, 4)):
        print(f"{'y @ ' + str(t) + ' s':34s} {'':5s} " +
              " ".join(f"{np.abs(D[f'{m}_y'][:, i] - ty[:, i])[sel].mean() * 100:12.2f}" for m in models))
    print("-- error shape:")
    print(f"{'signed mean (cm)':34s} {'':5s} " + " ".join(f"{(D[f'{m}_y'][:, 2] - ty[:, 2])[sel].mean() * 100:12.2f}" for m in models))
    print(f"{'median |err| (cm)':34s} {'':5s} " + " ".join(f"{np.median(err[m][sel]):12.2f}" for m in models))
    print(f"{'p90 |err| (cm)':34s} {'':5s} " + " ".join(f"{np.percentile(err[m][sel], 90):12.2f}" for m in models))
    print(f"{'share of error in worst 10% frames':34s} {'':5s} " +
          " ".join(f"{np.sort(err[m][sel])[-sel.sum() // 10:].sum() / err[m][sel].sum() * 100:11.1f}%" for m in models))
    print(f"{'slope on teacher y2s (|y2s| > 0.3)':34s} {'':5s} " + " ".join(
        f"{np.sum((D[f'{m}_y'][:, 2] * ty[:, 2])[sel & (np.abs(ty[:, 2]) > .3)]) / np.sum((ty[:, 2] ** 2)[sel & (np.abs(ty[:, 2]) > .3)]):12.3f}"
        for m in models))
    print(f"{'|err| / teacher sigma':34s} {'':5s} " + " ".join(f"{(err[m] / 100 / D['t_sig_y2'])[sel].mean():12.2f}" for m in models))
    # jitter: frame-to-frame change of y2s (10 Hz) vs teacher; slow part of the error = 2 s moving average
    print(f"{'|d y2s / frame| (cm), teacher ' + format(np.abs(np.diff(ty[:, 2]))[sel[1:]].mean() * 100, '.2f'):34s} {'':5s} " +
          " ".join(f"{np.abs(np.diff(D[f'{m}_y'][:, 2]))[sel[1:]].mean() * 100:12.2f}" for m in models))
    k = np.ones(21) / 21
    for name, f in [("slow error (2 s avg) |.| (cm)", lambda e: np.convolve(e, k, "same")),
                    ("fast error (residual) |.| (cm)", lambda e: e - np.convolve(e, k, "same"))]:
        print(f"{name:34s} {'':5s} " + " ".join(f"{np.abs(f(D[f'{m}_y'][:, 2] - ty[:, 2]))[sel].mean() * 100:12.2f}" for m in models))
    print("-- current motion (t0) error vs teacher:")
    for c, sc, unit in [("r0", 1000, "mrad/s"), ("vx0", 100, "cm/s"), ("vy0", 100, "cm/s")]:
        ms = [m for m in models if f"{m}_{c}" in D]
        print(f"{c + ' |err| (' + unit + ')':34s} {'':5s} " + " ".join(
            f"{np.abs(D[f'{m}_{c}'] - D[f'teacher_{c}'])[sel].mean() * sc:12.2f}" if m in ms else f"{'-':>12s}" for m in models))
    # how much of the y2s error a t0 yaw-rate error alone would make: y(2s) ~ r0 * v * t^2 / 2
    for m in models:
        if f"{m}_r0" not in D:
            continue
        pred = (D[f"{m}_r0"] - D["teacher_r0"]) * D["v"] * 2.0
        e = D[f"{m}_y"][:, 2] - ty[:, 2]
        c = np.corrcoef(pred[sel], e[sel])[0, 1]
        print(f"  {m:12s} corr(y2s err, r0 err * v * 2) = {c:.2f}")


if __name__ == "__main__":
    students = sys.argv[1:] or ["base5", "aug10", "aug10_nohist"]
    D = load(EVAL, students)
    table(D, ["shipped", "small", *students], "EVAL routes")
    tr = [s for s in students if all((S / "dump" / s / f"{u}.npz").exists() for u in TRAIN)]
    D = load(TRAIN, tr)
    table(D, ["shipped", "small", *tr], "TRAIN routes (3)")
