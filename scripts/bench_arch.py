"""Compile student variants for the AX630C NPU (AX620E, NPU1, U16) and print MACs and Pulsar2 max_cycle.

Weights are random and calibration is dummy: only the schedule length is wanted. ms is estimated from the
current edgepilot core (9.12M cycles, measured 15.5 ms on the board).

usage: .venv/bin/python scripts/bench_arch.py <name>=<json kwargs> [...]
"""
import json
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from student import Student, count_macs, export_onnx  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MS_PER_CYCLE = 15.5 / 9_115_185
TENSORS = ["input_imgs", "big_input_imgs", "desire", "features_buffer", "traffic_convention"]

for spec in sys.argv[1:]:
    name, kw = spec.split("=", 1)
    s = Student(**json.loads(kw)).eval()
    s.fuse()
    macs = count_macs(s)
    onnx_path = ROOT / "models" / f"arch_{name}.onnx"
    export_onnx(s, str(onnx_path))
    cfg = {"model_type": "ONNX", "npu_mode": "NPU1", "target_hardware": "AX620E",
           "input": f"/data/supercombo_distill/models/{onnx_path.name}", "output_dir": f"/data/supercombo_distill/outputs/arch_{name}",
           "output_name": "m.axmodel",
           "quant": {"input_configs": [{"tensor_name": t, "calibration_dataset": f"/data/supercombo_distill/calib/dummy/{t}.tar",
                                        "calibration_size": -1, "calibration_format": "Numpy"} for t in TENSORS],
                     "calibration_method": "MinMax", "precision_analysis": False,
                     "layer_configs": [{"start_tensor_names": ["DEFAULT"], "end_tensor_names": ["DEFAULT"], "data_type": "U16"}]},
           "input_processors": [{"tensor_name": t, "src_dtype": "U8" if t.endswith("imgs") else "FP32"} for t in TENSORS],
           "compiler": {"check": 0, "npu_perf": True}}
    (ROOT / "configs" / f"arch_{name}.json").write_text(json.dumps(cfg, indent=1))
    r = subprocess.run([str(ROOT / "scripts/p2build.sh"), f"arch_{name}"], capture_output=True, text=True)
    line = r.stdout.strip()
    cyc = int(line.split("max_cycle =")[-1].replace(",", "")) if "max_cycle" in line else None
    params = sum(p.numel() for p in s.parameters()) - sum(p.numel() for p in s.cpu_heads.parameters())
    tot = sum(macs.values())
    print(f"{name:12s} params(NPU) {params / 1e6:5.1f}M  MACs {tot / 1e9:5.2f}G (backbone {macs.get('backbone', 0) / 1e9:.2f}G)  "
          f"max_cycle {cyc if cyc is None else f'{cyc / 1e6:.2f}M'}  est {cyc * MS_PER_CYCLE if cyc else float('nan'):.1f} ms  {kw}",
          flush=True)
