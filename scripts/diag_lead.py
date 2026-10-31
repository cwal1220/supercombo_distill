"""Slice the lead_x error (|model - teacher| of the current lead's distance, frames with v > 5 and teacher lead
probability > 0.5, as eval_student.py) of a dumped student (diag_y2s_dump.py, with lead fields) and the small / shipped
models by route, distance, relative speed, lead probability, lead age, and look at the lead-probability agreement.

usage: .venv/bin/python scripts/diag_lead.py [dump name, default qat3_s8]"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from diag_y2s import EVAL, S, TRAIN  # noqa: E402

MODELS = ["shipped", "small", "student"]


def load(units, name):
    D = {}
    for u in units:
        z = np.load(S / "dump" / name / f"{u}.npz")
        tp = z["t_lead_prob"]
        on = tp > 0.5
        # frames since the teacher's lead appeared (prob crossed 0.5), 10 Hz
        age = np.zeros(len(on))
        for i in range(1, len(on)):
            age[i] = age[i - 1] + 1 if on[i] and on[i - 1] else 0
        rows = {"unit": np.full(len(tp), u), "v": z["v"], "t_prob": tp, "age": age,
                "t_x": z["t_lead"][:, 0, 0, 0], "t_y": z["t_lead"][:, 0, 0, 1], "t_vrel": z["t_lead"][:, 0, 0, 2] - z["v"],
                "t_sig": np.exp(z["t_lead_logstd"][:, 0, 0, 0]),
                "student_x": z["s_lead"][:, 0, 0, 0], "student_prob": z["s_lead_prob"],
                "small_x": z["small_lead_x"], "small_prob": z["small_lead_prob"],
                "shipped_x": z["rec_lead_x"], "shipped_prob": z["rec_lead_prob"]}
        for k, x in rows.items():
            D.setdefault(k, []).append(x)
    return {k: np.concatenate(v) for k, v in D.items()}


def table(D, title):
    sel = (D["v"] > 5) & (D["t_prob"] > 0.5)
    err = {m: np.abs(D[f"{m}_x"] - D["t_x"]) for m in MODELS}
    print(f"\n#### {title}  ({sel.sum()} lead frames of {(D['v'] > 5).sum()} moving)")
    print(f"{'slice':36s} {'n%':>5s} " + " ".join(f"{m:>10s}" for m in MODELS))

    def row(name, mask):
        mask = mask & sel
        if mask.sum() == 0:
            return
        print(f"{name:36s} {mask.sum() / sel.sum() * 100:5.1f} " + " ".join(f"{np.nanmean(err[m][mask]):10.2f}" for m in MODELS))

    row("all", sel)
    for u in np.unique(D["unit"]):
        row("unit " + u[:20] + u[-5:], D["unit"] == u)
    for lo, hi in [(0, 20), (20, 40), (40, 60), (60, 80), (80, 999)]:
        row(f"teacher x {lo}-{hi} m", (D["t_x"] >= lo) & (D["t_x"] < hi))
    for lo, hi in [(-99, -5), (-5, -1), (-1, 1), (1, 99)]:
        row(f"teacher v_rel {lo}-{hi} m/s", (D["t_vrel"] >= lo) & (D["t_vrel"] < hi))
    for lo, hi in [(0.5, 0.8), (0.8, 0.95), (0.95, 1.01)]:
        row(f"teacher prob {lo}-{hi}", (D["t_prob"] >= lo) & (D["t_prob"] < hi))
    row("student prob < 0.5 (missed)", D["student_prob"] < 0.5)
    row("student prob >= 0.5", D["student_prob"] >= 0.5)
    for lo, hi in [(0, 10), (10, 50), (50, 1e9)]:
        row(f"lead age {lo}-{hi if hi < 1e8 else 'end'} frames", (D["age"] >= lo) & (D["age"] < hi))
    for lo, hi in [(5, 15), (15, 25), (25, 99)]:
        row(f"v {lo}-{hi} m/s", (D["v"] >= lo) & (D["v"] < hi))
    print("-- error shape:")
    line = lambda name, f: print(f"{name:36s} {'':5s} " + " ".join(f"{f(m):10.2f}" for m in MODELS))
    line("signed mean (m)", lambda m: np.nanmean((D[f"{m}_x"] - D["t_x"])[sel]))
    line("median |err| (m)", lambda m: np.nanmedian(err[m][sel]))
    line("p90 |err| (m)", lambda m: np.nanpercentile(err[m][sel], 90))
    line("share of error in worst 10% (%)", lambda m: np.sort(err[m][sel])[-sel.sum() // 10:].sum() / np.nansum(err[m][sel]) * 100)
    line("mean |err| / teacher x (%)", lambda m: np.nanmean((err[m] / np.maximum(D["t_x"], 1))[sel]) * 100)
    line("mean |err| / teacher sigma", lambda m: np.nanmean((err[m] / D["t_sig"])[sel]))
    k = np.ones(21) / 21
    line("slow error (2 s avg) |.| (m)", lambda m: np.nanmean(np.abs(np.convolve(np.nan_to_num(D[f"{m}_x"] - D["t_x"]), k, "same"))[sel]))
    print("-- lead probability vs the teacher's (moving frames):")
    mv = D["v"] > 5
    tp = D["t_prob"] > 0.5
    line("miss rate (teacher on, model off) %", lambda m: ((D[f"{m}_prob"] < 0.5) & tp & mv).sum() / (tp & mv).sum() * 100)
    line("false lead (teacher off, model on) %", lambda m: ((D[f"{m}_prob"] >= 0.5) & ~tp & mv).sum() / (~tp & mv).sum() * 100)
    line("mean |prob - teacher prob|", lambda m: np.nanmean(np.abs(D[f"{m}_prob"] - D["t_prob"])[mv]))


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "qat3_s8"
    table(load(EVAL, name), f"EVAL routes, student {name}")
    table(load(TRAIN, name), f"TRAIN routes (3), student {name}")
