"""Far-lead calibration of the student's lead_prob, fitted on drives that neither training nor the eval used: logit +=
b0 - b1 * relu(x - x0) / 20 with x the student's own lead distance (t = 0). The student's extra false leads against
the shipped model sit beyond 60 m (lead_quality.py), where one global shift would also drop real leads.
(x0, b0, b1): on the calibration dumps, the lowest missed-lead rate with false leads <= the shipped model's there
(both against the teacher, frames with v > 5). Then false / missed on the eval dumps: shipped, student, the global
shift alone (b1 = 0) and the far-lead calibration, all fitted on the calibration dumps only.

--apply writes a checkpoint that does it in the model, with the same outputs: the student's arch gets lead_cal
(student.Student), a 64 -> 16 linear on the lead features whose only nonzero row writes relu((x - x0) / 20) into a
lead_prob feature that is dead (zero on all the build's calibration samples, its own pre-activation then forced
negative), and the CPU head cpu_heads.lead_prob gets weight -b1 on that feature and bias + b0 (all three horizons).

--logistic (2026-10-31): logit' = a s + b0 - b1 relu(x - x0) / 20 with x0 80, a (a temperature) and b1 fitted by logistic
regression on the teacher's probabilities (the whole conditional probability, smooth), only b0 then set for false leads
<= the shipped model's with the fewest missed. The grid fit above picks its operating point on the fitting routes'
exact counts and overshoots on unseen ones (LRF leave-one-route-out 4.52 / 11.06 %); the logistic fit holds up
(4.81 / 10.13 %, shipped 5.06 / 10.59). The CPU head is scaled by a (all three horizons).

--params x0 b0 b1 uses those values instead of the fit (a student fine-tuned on the calibration drives takes its
starting point's values). --loro: the whole fit cross-validated on the eval dump's routes (each route with the values
fitted on the others), the honest estimate when the calibration dump is the eval dump itself.

usage: .venv/bin/python scripts/lead_cal.py <calibration dump> <eval dump> [--params x0 b0 b1] [--loro] [--logistic]
                                            [--apply <ckpt> <build dir> <out ckpt>]"""
import io
import sys
import tarfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import lead_quality as L  # noqa: E402

S = 20.0


def pool(name):
    R = L.load(name)
    return {k: np.concatenate([R[r][k] for r in R]) for k in ("t", "s", "r", "sx")}, len(R)


def rates(A, adj):
    p, nl, y = (A["s"] + adj) > 0, A["t"] <= .5, A["t"] > .5
    return (p & nl).sum() / nl.sum() * 100, (~p & y).sum() / y.sum() * 100


def adj(A, x0, b0, b1, a=1.0):
    """The calibrated logit minus the student's own."""
    return (a - 1.0) * A["s"] + b0 - b1 * np.maximum(A["sx"] - x0, 0) / S


def fit_logistic(A, x0=80.0):
    """--logistic: (x0, b0, b1, a), a and b1 by logistic regression on the teacher's probabilities, then b0 the
    shift with false leads <= the shipped model's (on A) and the fewest missed."""
    import torch
    X = torch.from_numpy(np.stack([A["s"], np.maximum(A["sx"] - x0, 0) / S], 1).astype(np.float64))
    t = torch.from_numpy(A["t"].astype(np.float64))
    w = torch.tensor([1.0, 0.0], dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(X @ w + b, t)
        loss.backward()
        return loss
    opt.step(closure)
    a, b1, b = float(w[0]), -float(w[1]), float(b)
    target = ((A["r"] > .5) & (A["t"] <= .5)).sum() / (A["t"] <= .5).sum() * 100
    best = None
    for c in np.arange(-3.0, 3.001, 0.02):
        f, m = rates(A, adj(A, x0, b + c, b1, a))
        if f <= target and (best is None or m < best[0]):
            best = (m, b + c)
    return float(x0), round(float(best[1]), 3), round(b1, 3), round(a, 3)   # plain floats: the ckpt loads weights_only


def fit(A, far=True):
    target = ((A["r"] > .5) & (A["t"] <= .5)).sum() / (A["t"] <= .5).sum() * 100
    best = None
    for x0 in ((40, 60, 80) if far else (0,)):
        for b1 in (np.arange(0, 4.01, 0.25) if far else (0.0,)):
            for b0 in np.round(np.arange(-2.0, 1.51, 0.05), 2):
                f, m = rates(A, b0 - b1 * np.maximum(A["sx"] - x0, 0) / S)
                if f <= target and (best is None or m < best[0]):
                    best = (m, x0, float(b0), float(b1))
    return best[1:]


def dead_channel(s, build):
    """The lead_prob feature that is zero on all of the build's calibration samples with the least CPU weight."""
    import torch
    names = ("input_imgs", "big_input_imgs", "desire", "features_buffer", "traffic_convention")
    X = {}
    for k in names:
        with tarfile.open(f"{build}/calib/{k}.tar") as t:
            X[k] = [np.load(io.BytesIO(t.extractfile(m).read())) for m in sorted(t.getmembers(), key=lambda m: m.name)]
    with torch.no_grad():
        h = np.concatenate([s(*[torch.from_numpy(X[k][i]) for k in names])["hx_lead_prob"].numpy() for i in range(len(X["desire"]))])
    dead = np.flatnonzero((h > 0).sum(0) == 0)
    assert len(dead), "no dead lead_prob feature"
    w = s.cpu_heads["lead_prob"].weight.detach().abs().max(0).values.numpy()
    return int(dead[np.argmin(w[dead])]), len(h), dead.tolist()


def main():
    cal, ev = sys.argv[1], sys.argv[2]
    C, nc = pool(cal)
    E, ne = pool(ev)
    logistic = "--logistic" in sys.argv
    x0, b0, b1, a = fit_logistic(C) if logistic else (*fit(C), 1.0)
    _, g0, _ = fit(C, far=False)
    print(f"fitted on {cal}{' (logistic)' if logistic else ''}: far-lead x0 {x0} b0 {b0:+.2f} b1 {b1:.2f} a {a:.3f};"
          f" global shift {g0:+.2f}")
    if "--params" in sys.argv:   # another student's fit (e.g. the one this student was fine-tuned from)
        i = sys.argv.index("--params")
        x0, b0, b1 = (float(a) for a in sys.argv[i + 1:i + 4])
        a = 1.0
        print(f"using --params: far-lead x0 {x0:g} b0 {b0:+.2f} b1 {b1:.2f}")
    sh = lambda A: (((A["r"] > .5) & (A["t"] <= .5)).sum() / (A["t"] <= .5).sum() * 100,
                    ((A["r"] <= .5) & (A["t"] > .5)).sum() / (A["t"] > .5).sum() * 100)
    for nm, A, n in ((cal, C, nc), (ev, E, ne)):
        print(f"{nm} ({n} routes, {len(A['t'])} frames), false / missed %:")
        print(f"  shipped              {sh(A)[0]:6.2f} {sh(A)[1]:6.2f}")
        print(f"  student              {rates(A, 0)[0]:6.2f} {rates(A, 0)[1]:6.2f}")
        print(f"  global shift         {rates(A, g0)[0]:6.2f} {rates(A, g0)[1]:6.2f}")
        f, m = rates(A, adj(A, x0, b0, b1, a))
        print(f"  far-lead calibration {f:6.2f} {m:6.2f}")
    if "--loro" in sys.argv:   # the whole fit (x0 too) cross-validated on the eval routes: fit on the others, apply to each
        R = L.load(ev)
        fp = mi = n0 = n1 = 0
        for held in R:
            T = {k: np.concatenate([R[r][k] for r in R if r != held]) for k in ("t", "s", "r", "sx")}
            fx0, fb0, fb1, fa = fit_logistic(T) if logistic else (*fit(T), 1.0)
            h = R[held]
            p = (h["s"] + adj(h, fx0, fb0, fb1, fa)) > 0
            fp, mi = fp + (p & (h["t"] <= .5)).sum(), mi + (~p & (h["t"] > .5)).sum()
            n0, n1 = n0 + (h["t"] <= .5).sum(), n1 + (h["t"] > .5).sum()
            print(f"  without {held}: x0 {fx0} b0 {fb0:+.2f} b1 {fb1:.2f} a {fa:.3f}")
        print(f"{ev} leave-one-route-out far-lead calibration: false {fp / n0 * 100:.2f}% missed {mi / n1 * 100:.2f}%")
    if "--apply" in sys.argv:
        import torch
        from student import load_student
        i = sys.argv.index("--apply")
        ck = torch.load(sys.argv[i + 1], map_location="cpu", weights_only=False)
        s = load_student(ck)
        j, n, dead = dead_channel(s, sys.argv[i + 2])
        print(f"dead lead_prob features {dead} on {n} calibration samples; using {j}")
        ck["arch"] = {**ck.get("arch", {}), "lead_cal": True}
        c = load_student({**ck, "model": {**ck["model"], "lead_cal.weight": torch.zeros(16, 64), "lead_cal.bias": torch.zeros(16)}})
        with torch.no_grad():
            last = c.heads["lead_prob"][-2]                                  # the feature's own pre-activation: never > 0
            last.weight[j] = 0
            last.bias[j] = -1
            wx, bx = c.cpu_heads["lead"].weight[0], c.cpu_heads["lead"].bias[0]   # t = 0, step 0, x
            c.lead_cal.weight.zero_()
            c.lead_cal.bias.zero_()
            c.lead_cal.weight[j] = wx / S
            c.lead_cal.bias[j] = (bx - x0) / S
            c.cpu_heads["lead_prob"].weight *= a                             # the temperature (1 unless --logistic)
            c.cpu_heads["lead_prob"].bias *= a
            c.cpu_heads["lead_prob"].weight[:, j] = -b1
            c.cpu_heads["lead_prob"].bias += b0
        ck["model"] = c.state_dict()
        ck["lead_cal"] = {"x0": x0, "b0": b0, "b1": b1, "a": a, "feature": j, "fitted_on": cal, "logistic": logistic}
        torch.save(ck, sys.argv[i + 3])
        print(f"saved {sys.argv[i + 3]}")


if __name__ == "__main__":
    main()
