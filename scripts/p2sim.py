"""Fast Pulsar2 simulation on the CPU: the student with S8 weights (student.quant_sim) exported to ONNX, with every
tensor Pulsar2 quantizes fake-quantized as Pulsar2 does it: U16 with the scale and zero point of the build
(<build>/quant/quant_axmodel.onnx: graph inputs, every Conv / FC / Add / LpNorm / Concat output), i.e.
clip(round(x / s) + zp, 0, 65535). Runs eval units in eval_student's closed loop (two passes like quant_eval_p2.py)
with onnxruntime, a minute a unit against ~15 for `pulsar2 run`, and prints the metrics for
  S8 weights only (no activation quantization; = eval_student --s8),
  S8 + U16 (all of Pulsar2's activation quantizers),
  S8 + U16 without the quantizers whose tensor names contain a --drop substring (one row each, to find the culprits),
and with --p2 the saved Pulsar2 Reference outputs (<build>/p2eval_<unit>.npz) where they exist.

usage: .venv/bin/python scripts/p2sim.py --ckpt ckpt/mac_lra/last.pt --build outputs/quant_lra \
           --units 2026-10-25--22-42-26-175__015 [--drop Concat_2 Div_output ...] [--p2] [--dump lra_ext_npu]"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import helper, numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_student as E  # noqa: E402
from quant_eval_p2 import OUTS, inputs  # noqa: E402
from student import POLICY_HEADS, export_onnx, load_student, quant_sim  # noqa: E402
from train import BIG  # noqa: E402
from zframes import load_frames  # noqa: E402
from lane_geom import slope_fix  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["lat_acc", "long_acc", "plan_y2s_cm", "lanes_vs_ld_cm", "width_vs_ld_cm", "lead_x_m", "gas2", "false_lead_pct",
        "lanes_ldfix_cm", "width_ldfix_cm"]


def p2_quant_params(build: Path) -> dict[str, tuple[float, int]]:
    """Float-graph tensor name -> (scale, zero point) of every U16 quantizer in a Pulsar2 build."""
    q = onnx.load(build / "quant/quant_axmodel.onnx", load_external_data=False)
    out = {}
    for n in q.graph.node:
        a = {x.name: helper.get_attribute_value(x) for x in n.attribute}
        if "output_scales" not in a or a.get("output_dtype") != b"U16":
            continue
        name = n.output[0]
        if name.endswith("_QuantizeLinear_out_link_var"):         # a graph input
            name = name[:-len("_QuantizeLinear_out_link_var")]
        elif "_Ax_Generated_Var_" in name:                        # a graph output (hx_*)
            name = name.split("_Ax_Generated_Var_")[0]
        out[name] = (float(a["output_scales"][0]), int(a["output_zeropoints"][0]))
    return out


def with_fake_quant(model: onnx.ModelProto, params: dict[str, tuple[float, int]]) -> onnx.ModelProto:
    """A copy of the float graph with x -> (clip(round(x / s) + zp, 0, 65535) - zp) * s on each named tensor."""
    m = onnx.ModelProto()
    m.CopyFrom(model)
    g = m.graph
    outputs = {o.name for o in g.output}
    inputs_ = {i.name for i in g.input}
    new_nodes = []
    for t, (s, zp) in params.items():
        raw = t + "__raw" if t in outputs else t
        fq = t if t in outputs else t + "__fq"
        if t in outputs:            # the producer writes raw, the quantizer writes the graph output's name
            for n in g.node:
                for i, o in enumerate(n.output):
                    if o == t:
                        n.output[i] = raw
        else:                       # consumers read the quantized tensor
            for n in g.node:
                for i, x in enumerate(n.input):
                    if x == t:
                        n.input[i] = fq
        c = lambda nm, v: numpy_helper.from_array(np.array(v, np.float32), f"{t}__{nm}")
        g.initializer.extend([c("s", s), c("zp", float(zp)), c("lo", 0.0), c("hi", 65535.0)])
        p = f"{t}__"
        new_nodes += [helper.make_node("Div", [raw, p + "s"], [p + "d"]),
                      helper.make_node("Round", [p + "d"], [p + "r"]),
                      helper.make_node("Add", [p + "r", p + "zp"], [p + "a"]),
                      helper.make_node("Clip", [p + "a", p + "lo", p + "hi"], [p + "c"]),
                      helper.make_node("Sub", [p + "c", p + "zp"], [p + "z"]),
                      helper.make_node("Mul", [p + "z", p + "s"], [fq])]
        assert t in inputs_ or t in outputs or any(t in n.output for n in g.node), t
    g.node.extend(new_nodes)
    # topological order: quantizers right after their producers (inputs first)
    order, have = [], set(inputs_) | {i.name for i in g.initializer}
    pending = list(g.node)
    while pending:
        rest = []
        for n in pending:
            if all(x in have or x == "" for x in n.input):
                order.append(n)
                have.update(n.output)
            else:
                rest.append(n)
        assert len(rest) < len(pending), "graph has a cycle"
        pending = rest
    del g.node[:]
    g.node.extend(order)
    return m


def closed_loop(sess: ort.InferenceSession, frames, lab, even, temporal_heads=()) -> dict[str, np.ndarray]:
    n = len(lab["big"])
    names = [o.name for o in sess.get_outputs()]
    run = lambda hist: [dict(zip(names, sess.run(None, inputs(frames, lab, t, hist)))) for t in even]
    hist = np.zeros(((n + 1) // 2, 512), np.float32)
    p1 = run(hist)
    hist[even // 2] = np.concatenate([o["out_hidden"] for o in p1])
    p2 = run(hist)
    hx = {o: np.concatenate([r[o] for r in p1]) for o in OUTS}
    for k in [*POLICY_HEADS, *temporal_heads]:   # the heads that see the history: from pass 2
        hx["hx_" + k] = np.concatenate([r["hx_" + k] for r in p2])
    return hx


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--build", type=Path, required=True)
    ap.add_argument("--units", nargs="+", required=True)
    ap.add_argument("--data", type=Path, default=ROOT / "data/ep_label")
    ap.add_argument("--drop", nargs="*", default=[], help="also a row without the quantizers matching each substring")
    ap.add_argument("--p2", action="store_true", help="also the saved Pulsar2 Reference outputs")
    ap.add_argument("--no-s8", action="store_true", help="skip the S8-weights-only row (half the time)")
    ap.add_argument("--dump", help="save the S8 + U16 row's lead outputs per frame to outputs/diag_y2s/dump/<name> "
                                   "(for lead_quality.py, lead_bias.py)")
    args = ap.parse_args()
    ck = torch.load(args.ckpt, map_location="cpu")
    m = load_student(ck)
    s8 = quant_sim(load_student(ck).fuse())
    params = p2_quant_params(args.build)
    with tempfile.TemporaryDirectory() as tmp:
        export_onnx(s8, f"{tmp}/s8.onnx")
        base = onnx.load(f"{tmp}/s8.onnx")
    variants = {"S8 + U16 (Pulsar2 quantizers)": params} if args.no_s8 else {"S8 weights only": {}, "S8 + U16 (Pulsar2 quantizers)": params}
    for d in args.drop:
        variants[f"  without *{d}*"] = {k: v for k, v in params.items() if d not in k}
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    sess = {k: ort.InferenceSession(with_fake_quant(base, p).SerializeToString(), so, providers=["CPUExecutionProvider"])
            for k, p in variants.items()}
    total = {}
    for u in args.units:
        d = args.data / u
        lab = np.load(d / "labels.npz")
        frames = load_frames(d)
        n = len(lab["big"])
        even = np.arange(4, n - n % 2, 2)
        v = lab["v_ego"][even]
        sel = v > 5.0
        big = E.from_layout(lab["big"][even], BIG)
        ld = lab["ld_lanes"][even]
        a, b = BIG["lanes"]
        ld_fix = slope_fix(ld, lab["big"][even, a:b][:, :264].reshape(-1, 4, 33, 2)[..., 1], lab["mount"][even, 1], lab["mount"][even, 0])
        rows = {"board NPU (shipped)": {k: x[even] for k, x in E.recorded(lab).items()}}
        with torch.no_grad():
            for name, se in sess.items():
                hx = closed_loop(se, frames, lab, even, m.temporal_heads)
                raw = {k: x.numpy() for k, x in m.outputs({o: torch.from_numpy(a) for o, a in hx.items()}).items()}
                rows[name] = E.parse_raw(raw)
                if args.dump and name.startswith("S8 + U16"):   # per frame, in diag_y2s_dump.py's lead fields
                    out = ROOT / "outputs/diag_y2s/dump" / args.dump
                    out.mkdir(parents=True, exist_ok=True)
                    a, b = BIG["lead"]
                    np.savez_compressed(out / f"{u}.npz", even=even, v=v, s_lead=raw["lead"][:, :72].reshape(-1, 3, 6, 4),
                                        s_lead_prob=rows[name]["lead_prob"], t_lead=lab["big"][even][:, a:a + 72].reshape(-1, 3, 6, 4),
                                        t_lead_prob=big["lead_prob"], rec_lead_x=rows["board NPU (shipped)"]["lead_x"],
                                        rec_lead_prob=rows["board NPU (shipped)"]["lead_prob"])
            saved = args.build / f"p2eval_{u}.npz"
            if args.p2 and saved.exists():
                z = np.load(saved)
                rows["Pulsar2 (saved; lead w/o history before 10-31)"] = E.parse_raw({k: x.numpy() for k, x in m.outputs({o: torch.from_numpy(z[o]) for o in OUTS}).items()})
        print(f"{u} ({len(even)} frames)")
        print(f"{'model':32s} " + " ".join(f"{k:>15s}" for k in KEYS))
        for name, q in rows.items():
            r = E.compare(q, big, ld, v, sel, ld_fix)
            total.setdefault(name, []).append((sel.sum(), r))
            print(f"{name:32s} " + " ".join(f"{r[k]:15.4f}" for k in KEYS), flush=True)
    if len(args.units) > 1:
        weight = {"false_lead_pct": "n_nolead"}
        print(f"mean over {len(args.units)} units (frame-weighted)")
        print(f"{'model':32s} " + " ".join(f"{k:>15s}" for k in KEYS))
        for name, rs in total.items():
            row = []
            for k in KEYS:
                ok = [(r[weight[k]] if k in weight else n, r[k]) for n, r in rs if np.isfinite(r[k])]
                row.append(sum(n * x for n, x in ok) / sum(n for n, _ in ok) if ok else np.nan)
            print(f"{name:32s} " + " ".join(f"{x:15.4f}" for x in row), flush=True)


if __name__ == "__main__":
    main()
