"""Run openpilot's small and big driving models over an edgepilot (MaixCAM2) route, next to what the board's
shipped NPU model recorded, and save everything per frame for comparison.

Per frame of a run of consecutive segments:
  - the recorded ModelState of the board (the shipped U16 axmodel, parsed by edgepilot): plan, lanes, lane
    probabilities, road edges, lead, desire state;
  - the released small model (openpilot master driving_supercombo.onnx, fp16, queues in the graph) and the big
    model (big_driving_supercombo.onnx, same date) run on the host GPU in closed loop on the same frames,
    warped like the board (edgepilot tools/model/model_warp.py, MaixCAM2 intrinsics, the recorded per-frame
    calibration and camera mount) with the recorded desire;
  - the car's own motion (ControlState speed, Localization yaw rate) and the stock camera's lanes (LKA14).

usage: ep_eval.py <route_dir> <out_dir> [--segs 000,001,...] [--every 1]
Writes <out_dir>/<route>__<first seg>.npz.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "ep"))
import recording_reader as rr  # noqa: E402
from model_warp import WarpPair  # noqa: E402

ROOT = HERE.parent
LKA14_ADDRESS, CAMERA_BUS = 0x484, 2


def ort_session(path):
    """A CUDA onnxruntime session; run through scripts/py so the CUDA 13 libraries are on the library path.
    On the Mac (no CUDA) a CPU session with ORT_THREADS intra-op threads (default: ORT's choice; the big model is
    ~370 ms a frame there)."""
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    if sys.platform == "darwin":
        so.intra_op_num_threads = int(os.environ.get("ORT_THREADS", "0"))
        units = os.environ.get("ORT_COREML")      # e.g. CPUAndNeuralEngine: CoreML (compiled once into a cache)
        if units:
            cache = Path(__file__).resolve().parents[1] / "outputs/coreml_cache" / units
            opts = {"ModelFormat": "MLProgram", "MLComputeUnits": units, "ModelCacheDirectory": str(cache)}
            return ort.InferenceSession(str(path), so, providers=[("CoreMLExecutionProvider", opts), "CPUExecutionProvider"])
        return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    sess = ort.InferenceSession(str(path), so, providers=["CUDAExecutionProvider"])
    if sess.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError("onnxruntime did not get CUDA: run through scripts/py")
    return sess


class Stateful:
    """A released openpilot driving ONNX with its history queues in the graph."""

    def __init__(self, path):
        self.sess = ort_session(path)
        self.reset()
        self.state_out = [o.name for o in self.sess.get_outputs()][1:]

    def reset(self):
        """Empty history queues (a new stream)."""
        self.state = {i.name: np.zeros(i.shape, np.uint8 if "uint8" in i.type else np.float32)
                      for i in self.sess.get_inputs() if i.name.startswith("state_")}

    def step(self, road6, wide6, pulse):
        feed = {"new_img": np.stack([road6, wide6]), "desire": pulse,
                "traffic_convention": np.array([[1.0, 0.0]], np.float32),
                "action_t": np.array([[0.3, 0.3]], np.float32), **self.state}
        out, *nxt = self.sess.run(None, feed)
        self.state = {k.replace("next_", ""): v for k, v in zip(self.state_out, nxt)}
        return out.reshape(-1)


def recorded(route_dir: Path):
    parts_cs, parts_ms, parts_loc, lka_t, lka_raw = [], [], [], [], []
    for chunk in rr.iter_states(route_dir, rr.RECORD_CONTROL_STATE, rr.RECORD_MODEL_STATE, rr.RECORD_LOCALIZATION,
                                rr.RECORD_CAN_RX):
        _, c = chunk[rr.RECORD_CONTROL_STATE]
        _, m = chunk[rr.RECORD_MODEL_STATE]
        _, loc = chunk[rr.RECORD_LOCALIZATION]
        parts_cs.append(c)
        parts_ms.append(m)
        parts_loc.append(loc)
        log_t, can = chunk[rr.RECORD_CAN_RX]
        sel = (can["address"] == LKA14_ADDRESS) & (can["src"] == CAMERA_BUS)
        lka_t.append(np.asarray(log_t)[sel])
        lka_raw.append(can["data"][sel][:, :8].copy().view("<u8").reshape(-1))
    cat = lambda xs: np.concatenate([x for x in xs if len(x)]) if any(len(x) for x in xs) else np.zeros(0)
    return cat(parts_cs), cat(parts_ms), cat(parts_loc), cat(lka_t), cat(lka_raw)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("route", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--segs")
    ap.add_argument("--small", default=str(ROOT / "models/driving_supercombo.onnx"))
    ap.add_argument("--big", default=str(ROOT / "models/big_driving_supercombo.onnx"))
    args = ap.parse_args()

    cs, ms, loc, lka_t, lka_raw = recorded(args.route)
    seg_dirs = [args.route / "segments" / s for s in args.segs.split(",")] if args.segs else \
        sorted(d for d in (args.route / "segments").iterdir() if (d / "frames.bin").exists() and rr.segment_video(d))
    segments = [rr.read_segment_index(d) for d in seg_dirs]
    ms_t = ms["capture_timestamp_ns"].astype(np.int64)
    order = np.argsort(ms_t)
    ms, ms_t = ms[order], ms_t[order]
    cs_t = cs["timestamp_ns"].astype(np.int64)

    small, big = Stateful(args.small), Stateful(args.big)
    warp = WarpPair(1280, 720)
    prev = np.zeros(8, np.float32)
    rows = {k: [] for k in ("t_ns", "small", "big", "rec_idx", "desire", "v_ego")}
    t0 = time.time()
    for n, (rec, y, u, v) in enumerate(rr.decode_route_yuv(segments)):
        t = int(rec["capture_timestamp_ns"])
        j = int(np.clip(np.searchsorted(ms_t, t), 0, len(ms_t) - 1))
        if j > 0 and abs(ms_t[j - 1] - t) < abs(ms_t[j] - t):
            j -= 1
        m = ms[j]
        # the board warps with the calibration and mount its ModelState records; round to keep the map cache
        rpy = np.round(np.array([m["calibration"]["roll"], m["calibration"]["pitch"], m["calibration"]["yaw"]], np.float64), 4)
        warp.set_calibration(rpy, round(float(m["camera_offset_m"]), 3), round(float(m["camera_height_m"]), 3))
        med, sbig = warp.warp(y, u, v)
        k = int(np.clip(np.searchsorted(cs_t, t) - 1, 0, len(cs_t) - 1))
        d = int(cs["desire"][k])
        cur = np.zeros(8, np.float32)
        if d > 0:
            cur[d] = 1.0
        pulse = np.where(cur - prev > 0.99, cur, 0.0).astype(np.float32)
        prev = cur
        rows["small"].append(small.step(med, sbig, pulse).astype(np.float32))
        rows["big"].append(big.step(med, sbig, pulse)[:2066].astype(np.float32))
        rows["t_ns"].append(t)
        rows["rec_idx"].append(j if abs(int(ms_t[j]) - t) < 30_000_000 else -1)
        rows["desire"].append(d)
        rows["v_ego"].append(float(cs["ego_speed_kph"][k]) / 3.6)
        if n % 1200 == 0:
            print(f"{args.route.name}: frame {n}, {time.time() - t0:.0f} s", flush=True)
    rec_idx = np.array(rows["rec_idx"])
    out = {k: np.array(v) for k, v in rows.items()}
    keep = ("plan", "lanes", "lane_probabilities", "lane_stds", "road_edges", "desire_state", "plan_yaw",
            "plan_yaw_rate", "camera_offset_m", "camera_height_m", "gas_press_probs")
    for f in keep:
        a = ms[f][np.clip(rec_idx, 0, None)]
        out["rec_" + f] = np.where((rec_idx >= 0).reshape((-1,) + (1,) * (a.ndim - 1)), a, np.nan).astype(np.float32)
    for f in ("valid", "probability", "x", "y", "velocity"):
        out["rec_lead_" + f] = ms["lead"][f][np.clip(rec_idx, 0, None)]
    out["rec_calib_rpy"] = np.stack([ms["calibration"][k][np.clip(rec_idx, 0, None)] for k in ("roll", "pitch", "yaw")], 1)
    out["loc_t_ns"] = loc["timestamp_ns"].astype(np.int64)
    out["loc_yaw_rate"] = loc["angular_velocity_calib"][:, 2]
    out["loc_vel"] = loc["velocity_device"]
    out["cs_t_ns"] = cs_t
    out["cs_v_ego"] = cs["ego_speed_kph"] / 3.6
    out["cs_engaged"] = cs["engaged"]
    out["lka_t_ns"] = np.asarray(lka_t, np.int64)
    out["lka_raw"] = np.asarray(lka_raw, np.uint64)
    args.out.mkdir(parents=True, exist_ok=True)
    dst = args.out / f"{args.route.name}__{seg_dirs[0].name}.npz"
    np.savez_compressed(dst, **out)
    print(f"{dst}: {len(rec_idx)} frames, {(rec_idx >= 0).mean() * 100:.1f}% matched to a ModelState, "
          f"{time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
