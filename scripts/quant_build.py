"""Quantize a student checkpoint for the AX630C NPU: ONNX export (fused, CPU heads split off), a calibration set
from real training frames, and a Pulsar2 6.0 AX620E build (NPU1, U16 activations, S8 per-channel weights, MinMax),
the same settings as the shipped core.

Calibration: --n samples from training units (the eval routes are excluded), --per-unit random even frames per
unit at t >= 100 (moving and standing alike); --extremes: frames chosen so that every tensor's range is covered
(extreme_picks(); random frames left desire_enc clipped on 23 % of the eval frames and the plan under-steered in
sharp turns on the NPU). Each sample's features_buffer holds the checkpoint's own hidden states of
frames t-96 ... t-4 (fp32; the hidden depends on the frames only), desire the pooled pulses as on the device,
traffic_convention [1, 0]. One .npy per sample in <out>/calib/<input>.tar, float32 in the input's shape.

Pulsar2 runs in docker (image pulsar2:6.0-lite, amd64; on the Mac via colima with Rosetta) with the project mounted
at /data/supercombo_distill, as /home/chan/pulsar2/work is on the PC. Writes:
  models/quant/<name>.onnx, configs/quant_<name>.json, <out> = outputs/quant_<name>/ (calib/, the build, m.axmodel),
  logs/quant_<name>.log

usage: .venv/bin/python scripts/quant_build.py --ckpt ckpt/mac_base_flip8/last.pt --name flip8 [--n 256] [--no-build]
"""
from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from student import Student, export_onnx, load_student  # noqa: E402
from train import EVAL_ROUTES, default_device  # noqa: E402
from zframes import load_frames  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TENSORS = ["input_imgs", "big_input_imgs", "desire", "features_buffer", "traffic_convention"]
IMAGE = "pulsar2:6.0-lite"


def load(ckpt: Path) -> Student:
    return load_student(torch.load(ckpt, map_location="cpu"))


@torch.no_grad()
def hidden(m: Student, frames, js, dev) -> np.ndarray:
    """out_hidden of kept frames js (road/wide of kept frames j-2 and j)."""
    out = []
    for i in range(0, len(js), 32):
        jj = js[i:i + 32]
        road = torch.from_numpy(np.stack([np.concatenate([frames[j - 2, 0], frames[j, 0]]) for j in jj])).to(dev).float()
        wide = torch.from_numpy(np.stack([np.concatenate([frames[j - 2, 1], frames[j, 1]]) for j in jj])).to(dev).float()
        b = torch.cat([m.features(road[k:k + 1], wide[k:k + 1]) for k in range(len(jj))])
        out.append(torch.nn.functional.normalize(m.summarizer(b), dim=-1).cpu().numpy())
    return np.concatenate(out)


def sample(m: Student, frames, lab, t: int, dev) -> dict[str, np.ndarray]:
    """The model inputs at even frame t (features_buffer: the checkpoint's own hidden of t-96 ... t-4)."""
    j = t // 2
    idx = np.arange(t - 96, t, 4) // 2
    des = np.zeros((100, 8), np.float32)
    des[100 - (t + 1 - max(0, t - 99)):] = lab["pulse"][max(0, t - 99):t + 1]
    return {"features_buffer": hidden(m, frames, idx, dev)[None],
            "input_imgs": np.concatenate([frames[j - 2, 0], frames[j, 0]])[None].astype(np.float32),
            "big_input_imgs": np.concatenate([frames[j - 2, 1], frames[j, 1]])[None].astype(np.float32),
            "desire": des.reshape(25, 4, 8).max(1)[None], "traffic_convention": np.array([[1.0, 0.0]], np.float32)}


def train_units(data: Path) -> list[Path]:
    return sorted(d for d in data.iterdir() if (d / "labels.npz").exists() and not any(d.name.startswith(r) for r in EVAL_ROUTES)
                  and d.name.count("__") < 2)      # real drives only, not the augmented replays (<route>__<seg>__<tag>)


def calib(m: Student, data: Path, n: int, per_unit: int, seed: int, dev, picks=None) -> dict[str, list[np.ndarray]]:
    """picks: {unit name: [t, ...]} to use instead of per_unit random frames of shuffled units."""
    rng = np.random.default_rng(seed)
    units = train_units(data)
    rng.shuffle(units)
    if picks is not None:
        units = [d for d in units if d.name in picks]
    m = m.to(dev)
    S = {t: [] for t in TENSORS}
    for d in units:
        if len(S["input_imgs"]) >= n:
            break
        lab = np.load(d / "labels.npz")
        frames = load_frames(d)
        nfr = len(lab["pulse"])
        cand = np.arange(100, nfr - nfr % 2, 2)
        if len(cand) == 0:
            continue
        ts = picks[d.name] if picks is not None else rng.choice(cand, min(per_unit, len(cand), n - len(S["input_imgs"])), replace=False)
        for t in np.sort(ts):
            for k, a in sample(m, frames, lab, int(t), dev).items():
                S[k].append(a)
        print(f"  {d.name}: {len(S['input_imgs'])} samples", flush=True)
    return S


def extreme_picks(ckpt: Path, m: Student, data: Path, n: int, per_unit: int, seed: int, dev) -> dict[str, list[int]]:
    """Calibration frames that cover the activations' range (MinMax takes each tensor's range from the calibration set,
    and Pulsar2 clips everything outside it). Candidates: per_unit random frames and per_unit frames with a desire
    pulse in the last 5 s or a turn (|teacher plan y at 2 s| > 1 m) from every real training unit. The S8 model
    (student.quant_sim) runs on each with every float tensor of its ONNX graph as an output; the 2 candidates with the
    largest max and the 2 with the smallest min of each tensor are kept, the rest of the n drawn at random."""
    import onnx
    import onnxruntime as ort
    import tempfile
    from student import quant_sim
    rng = np.random.default_rng(seed)
    s8 = quant_sim(load(ckpt).fuse())
    with tempfile.TemporaryDirectory() as tmp:
        export_onnx(s8, f"{tmp}/s8.onnx")
        g = onnx.shape_inference.infer_shapes(onnx.load(f"{tmp}/s8.onnx"))
    have = {o.name for o in g.graph.output}
    for vi in g.graph.value_info:
        if vi.type.tensor_type.elem_type == onnx.TensorProto.FLOAT and vi.name not in have:
            g.graph.output.append(vi)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    sess = ort.InferenceSession(g.SerializeToString(), so, providers=["CPUExecutionProvider"])
    names = [o.name for o in sess.get_outputs()]
    cands, hi, lo = [], [], []
    m = m.to(dev)
    T = np.array([10.0 * (i / 32) ** 2 for i in range(33)])
    for d in train_units(data):
        lab = np.load(d / "labels.npz")
        frames = load_frames(d)
        nfr = len(lab["pulse"])
        cand = np.arange(100, nfr - nfr % 2, 2)
        if len(cand) == 0:
            continue
        y2 = np.array([np.interp(2.0, T, p) for p in lab["big"][cand, 917:917 + 495].reshape(-1, 33, 15)[:, :, 1]])
        act = np.array([lab["pulse"][max(0, t - 99):t + 1, 1:].any() for t in cand]) | (np.abs(y2) > 1.0)
        ts = list(rng.choice(cand, min(per_unit, len(cand)), replace=False))
        if act.any():
            ts += list(rng.choice(cand[act], min(per_unit, act.sum()), replace=False))
        for t in sorted(set(int(t) for t in ts)):
            o = sess.run(None, sample(m, frames, lab, t, dev))
            cands.append((d.name, t))
            hi.append([float(a.max()) for a in o])
            lo.append([float(a.min()) for a in o])
        print(f"  {d.name}: {len(cands)} candidates", flush=True)
    hi, lo = np.array(hi), np.array(lo)
    keep = []
    for rank in (0, 1):                 # every tensor's most extreme candidate first, then the runners-up
        for k in range(len(names)):
            ext = [np.argsort(-hi[:, k])[rank]] + ([np.argsort(lo[:, k])[rank]] if lo[:, k].min() < 0 else [])
            keep += [int(i) for i in ext if i not in keep]
    keep = keep[:n]
    rest = [i for i in range(len(cands)) if i not in set(keep)]
    keep += list(rng.choice(rest, max(0, min(n - len(keep), len(rest))), replace=False))
    print(f"extremes: {len(cands)} candidates, {len(names)} tensors -> {len(keep)} frames", flush=True)
    picks = {}
    for i in keep:
        picks.setdefault(cands[i][0], []).append(cands[i][1])
    return picks


def write_tar(path: Path, arrays: list[np.ndarray]):
    with tarfile.open(path, "w") as tf:
        for i, a in enumerate(arrays):
            buf = io.BytesIO()
            np.save(buf, a)
            info = tarfile.TarInfo(f"{i:04d}.npy")
            info.size = buf.tell()
            buf.seek(0)
            tf.addfile(info, buf)


def config(name: str, out_rel: str, precision_analysis: bool) -> dict:
    d = "/data/supercombo_distill"
    return {"model_type": "ONNX", "npu_mode": "NPU1", "target_hardware": "AX620E",
            "input": f"{d}/models/quant/{name}.onnx", "output_dir": f"{d}/{out_rel}", "output_name": "m.axmodel",
            "quant": {"input_configs": [{"tensor_name": t, "calibration_dataset": f"{d}/{out_rel}/calib/{t}.tar",
                                         "calibration_size": -1, "calibration_format": "Numpy"} for t in TENSORS],
                      "calibration_method": "MinMax", "precision_analysis": precision_analysis,
                      "precision_analysis_method": "EndToEnd",
                      "layer_configs": [{"start_tensor_names": ["DEFAULT"], "end_tensor_names": ["DEFAULT"], "data_type": "U16"}]},
            "input_processors": [{"tensor_name": t, "src_dtype": "U8" if t.endswith("imgs") else "FP32"} for t in TENSORS],
            "compiler": {"check": 0, "npu_perf": True}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", type=Path, default=ROOT / "data/ep_label")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--per-unit", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--extremes", action="store_true",
                    help="calibration frames chosen to cover every tensor's range (extreme_picks()) instead of at random")
    ap.add_argument("--precision-analysis", action="store_true", help="Pulsar2 per-layer cosine vs fp32 (slower build)")
    ap.add_argument("--no-build", action="store_true")
    ap.add_argument("--device", default=default_device())
    args = ap.parse_args()
    out_rel = f"outputs/quant_{args.name}"
    out = ROOT / out_rel
    (out / "calib").mkdir(parents=True, exist_ok=True)
    (ROOT / "models/quant").mkdir(parents=True, exist_ok=True)

    m = load(args.ckpt)
    t0 = time.time()
    picks = extreme_picks(args.ckpt, m, args.data, args.n, args.per_unit, args.seed, args.device) if args.extremes else None
    S = calib(m, args.data, args.n, args.per_unit, args.seed, args.device, picks)
    for t, arrs in S.items():
        write_tar(out / "calib" / f"{t}.tar", arrs)
    print(f"calibration: {len(S['input_imgs'])} samples ({time.time() - t0:.0f} s)", flush=True)
    export_onnx(load(args.ckpt), str(ROOT / f"models/quant/{args.name}.onnx"))
    cfg = ROOT / f"configs/quant_{args.name}.json"
    cfg.write_text(json.dumps(config(args.name, out_rel, args.precision_analysis), indent=1))
    if args.no_build:
        return
    log = ROOT / f"logs/quant_{args.name}.log"
    t0 = time.time()
    with open(log, "w") as f:
        rc = subprocess.run(["docker", "run", "--rm", "--platform", "linux/amd64", "--entrypoint", "/bin/bash",
                             "-v", f"{ROOT}:/data/supercombo_distill", "-w", "/data/supercombo_distill", IMAGE, "-c",
                             f"pulsar2 build --config /data/supercombo_distill/configs/{cfg.name}"],
                            stdout=f, stderr=subprocess.STDOUT).returncode
    cyc = [l for l in log.read_text().splitlines() if "max_cycle" in l]
    print(f"pulsar2 build: exit {rc} ({time.time() - t0:.0f} s) {cyc[-1].strip() if cyc else ''}  log {log}")


if __name__ == "__main__":
    main()
