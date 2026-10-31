"""Label an edgepilot (MaixCAM2) route for distillation and evaluation.

For every frame of the route's video (consecutive segments decoded as one stream):
  - warp it like the board (edgepilot model_warp.py: MaixCAM2 intrinsics, the frame's recorded calibration and
    camera mount) into the two 6x128x256 model views;
  - run in closed loop, with the recorded desire pulses: the big model (teacher, outputs[:2066] without its
    16384-float hidden state) and the small model edgepilot ships (fp32 host reference, 2576 floats);
  - run the LD lane model on the 1280x640 band under the top 80 rows (ld_teacher.py) and keep its raw outputs and
    its lines projected to the road in openpilot's lane layout;
  - keep the board's recorded ModelState (the shipped U16 NPU model) and the car's speed and yaw rate.
Every second frame's warped views are written for training (10 Hz; t-4 of an even frame is even).

Routes are labelled in chunks of up to CHUNK_SEGS consecutive segments, each saved on its own
(<out>/<route>__<first seg>/) so a crash loses at most one chunk; a rerun labels only the segments no finished
chunk covers. Segments dedup.py marks as repeats (driven again in the same conditions) are not labelled; a chunk
after a skipped segment warms up on it. Output per chunk:
  labels.npz      per frame: t_ns, seg, frame index, big [N,2066], small [N,2576], ld_raw [N,2071],
                  ld_lanes [N,4,33] (y of each line's inner edge, NaN outside what LD saw), ld_lanes_centre
                  (the same at the paint centre), ld_lane_prob [N,4], ld_edges [N,2,33],
                  desire, v_ego, yaw_rate, engaged, recorded NPU fields (rec_*), calib rpy, mount
  frames.npy      uint8 [M,2,6,128,256] warped views of the kept frames; frames_idx.npy their frame numbers

Augmented replays for consistent distillation (--tag, the teacher labels the very views the student will see):
--rpy-deg r p y adds to the recorded calibration, --offset m moves the virtual camera sideways (both for the warp and the
LD projection), --rate k plays the video k times faster (every k-th frame through the models, as if consecutive). The
chunks go to <route>__<seg>__<tag>; chunks without a tag ignore tagged ones and the other way round.

--no-small skips the small model (its labels are NaN): it only gives the eval routes' reference row, and on the Mac
(CPU onnxruntime, ep_eval.ort_session) every model counts (big ~370 ms a frame; with ORT_COREML=CPUAndNeuralEngine the
teacher runs on the Neural Engine at ~80 ms, see big_driving_supercombo_coreml.onnx below).

usage: scripts/py scripts/ep_label.py <route_dir> <out_dir> [--also <route_dir> ...] [--no-small]   (Mac: .venv/bin/python)
"""
import argparse
import json
import os
import zlib
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "ep"))
sys.path.insert(0, str(HERE))
import recording_reader as rr  # noqa: E402
from ep_eval import Stateful, recorded  # noqa: E402
from ld_teacher import LDModel, roi_input, to_road, resample  # noqa: E402
from model_warp import WarpPair, maixcam2_intrinsics  # noqa: E402

ROOT = HERE.parent
LD_ROI = (0, 80, 1280, 640)
LD_TO_OP = {4: 0, 0: 1, 1: 2, 5: 3}   # LD line position -> openpilot lane_lines index
LD_EDGE_TO_OP = {0: 0, 1: 1}         # LD near left / near right boundary -> openpilot road_edges index


LEFT_OF_CAR = (0, 4)   # LD lines whose inner (lane-facing) edge is on their right in the image


def ld_layout(lines, k, rpy, height, offset):
    """LD lines projected to the road: lanes at their inner edge (the edge facing the lane they bound), the
    same lanes at the paint centre, line probabilities, and the near road edges."""
    lanes = np.full((4, 33), np.nan, np.float32)
    centre = np.full((4, 33), np.nan, np.float32)
    prob = np.zeros(4, np.float32)
    edges = np.full((2, 33), np.nan, np.float32)
    for L in lines:
        v = L["y"] * LD_ROI[3] + LD_ROI[1]
        if L["kind"] == "line" and L["pos"] in LD_TO_OP:
            i = LD_TO_OP[L["pos"]]
            side = 1.0 if L["pos"] in LEFT_OF_CAR else -1.0
            for dst_, xn in ((lanes, L["x"] + side * L["half"]), (centre, L["x"])):
                x, y = to_road(xn * LD_ROI[2] + LD_ROI[0], v, k, rpy, height, offset)
                dst_[i] = resample(x, y)
            prob[i] = L["valid"]
        elif L["kind"] == "boundary" and L["pos"] in LD_EDGE_TO_OP:
            x, y = to_road(L["x"] * LD_ROI[2] + LD_ROI[0], v, k, rpy, height, offset)
            edges[LD_EDGE_TO_OP[L["pos"]]] = resample(x, y)
    return lanes, centre, prob, edges


CHUNK_SEGS = 15      # segments (60 s each) labelled and saved together: a crash loses at most this much
WARMUP_FRAMES = 200  # the models run this many frames of the segment before a chunk to rebuild their history


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("route", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--dedup", type=Path, default=ROOT / "data/dedup/dedup.json",
                    help="dedup.py's decisions: its segments_not_labelled are skipped")
    ap.add_argument("--no-small", action="store_true", help="skip the small model (labels NaN; training does not use it)")
    ap.add_argument("--tag", default="", help="augmented replay: output chunks <route>__<seg>__<tag>")
    ap.add_argument("--rpy-deg", type=float, nargs=3, default=(0.0, 0.0, 0.0), help="calibration roll pitch yaw offset (deg)")
    ap.add_argument("--offset", type=float, default=0.0, help="virtual camera moved right by this (m)")
    ap.add_argument("--rate", type=int, default=1, help="play the video this many times faster")
    ap.add_argument("--rand-rpy-deg", type=float, nargs=3, default=None,
                    help="per chunk, add a uniform draw in +-these (deg) to roll pitch yaw (seeded by route and chunk)")
    ap.add_argument("--rand-offset", type=float, default=0.0, help="per chunk, a uniform camera offset in +-this (m)")
    ap.add_argument("--zstd", action="store_true", help="compress each chunk's frames right away (zframes.py)")
    ap.add_argument("--desire-inject", type=float, default=0.0,
                    help="counterfactual desires: about every this many seconds of driving (v > 8 m/s, no recorded "
                         "desire) a lane change left or right is requested for 3 s; the teacher plans it, and the "
                         "injected desire is what the labels record")
    ap.add_argument("--also", type=Path, nargs="*", default=[], help="more routes, labelled in the same process (the models load once)")
    args = ap.parse_args()
    assert args.tag or (args.rpy_deg == (0.0, 0.0, 0.0) and args.offset == 0.0 and args.rate == 1 and args.rand_rpy_deg is None
                        and args.rand_offset == 0.0), "augmentations need --tag"
    for route in [args.route, *args.also]:
        try:
            label_route(route, args)
        except Exception:                      # one bad route must not stop the others
            import traceback
            traceback.print_exc()
            print(f"{route.name}: FAILED", flush=True)


def label_route(route: Path, args):
    import shutil
    args.route = route
    free = shutil.disk_usage(args.out).free / 1e9
    if free < 8:
        print(f"{route.name}: skipped, only {free:.1f} GB free", flush=True)
        return
    if (args.out / args.route.name / "labels.npz").exists():   # labelled whole before chunking existed
        print(f"{args.out / args.route.name}: exists")
        return
    skip = set()
    if args.dedup.exists():
        skip = {s for r, s in json.load(open(args.dedup))["segments_not_labelled"] if r == args.route.name}
    seg_dirs = sorted(d for d in (args.route / "segments").iterdir() if (d / "frames.bin").exists() and rr.segment_video(d))
    covered = set()          # segments in chunks labelled before (they keep their dropped seconds; train.py masks them)
    for d in args.out.glob(args.route.name + "__*"):
        if (d / "labels.npz").exists() and (d.name.endswith("__" + args.tag) if args.tag else d.name.count("__") == 1):
            covered |= {int(s) for s in np.unique(np.load(d / "labels.npz")["seg"])}
    todo = [i for i, d in enumerate(seg_dirs) if d.name not in skip and int(d.name) not in covered]
    chunks = []              # runs of consecutive segments, at most CHUNK_SEGS each
    for i in todo:
        if chunks and chunks[-1][-1] == i - 1 and len(chunks[-1]) < CHUNK_SEGS:
            chunks[-1].append(i)
        else:
            chunks.append([i])
    print(f"{args.route.name}: {len(seg_dirs)} segments, {len(covered)} labelled before, {len(skip)} skipped "
          f"(repeats), {len(todo)} to label in {len(chunks)} chunks", flush=True)
    if not chunks:
        return
    rec = recorded(args.route)
    for c in chunks:
        name = f"{args.route.name}__{seg_dirs[c[0]].name}"
        rng = np.random.default_rng(zlib.crc32((name + args.tag).encode()))
        rpy = np.array(args.rpy_deg) + (rng.uniform(-1, 1, 3) * np.array(args.rand_rpy_deg) if args.rand_rpy_deg else 0.0)
        off = args.offset + rng.uniform(-1, 1) * args.rand_offset
        if args.tag:
            print(f"{name}__{args.tag}: rpy offset {np.round(rpy, 3)} deg, camera offset {off:+.3f} m, rate {args.rate}", flush=True)
        label_chunk(args.route, rec, [seg_dirs[i] for i in c], seg_dirs[c[0] - 1] if c[0] > 0 else None,
                    args.out / (name + (f"__{args.tag}" if args.tag else "")),
                    small=not args.no_small, rpy_delta=np.radians(rpy), offset_delta=off, rate=args.rate,
                    desire_inject=args.desire_inject, seed=zlib.crc32((name + args.tag + "desire").encode()))
        if args.zstd:
            from zframes import convert
            convert(args.out / (name + (f"__{args.tag}" if args.tag else "")))


_MODELS = {}


def models(small: bool):
    """The small model (or None), the teacher and LD, loaded once per process (a CoreML load takes minutes); the
    driving models' history queues emptied."""
    if not _MODELS:
        # on the Mac with ORT_COREML (e.g. CPUAndNeuralEngine, ~45 ms a frame): the teacher with its negative Gather
        # indices made positive, which CoreML otherwise gets wrong (plan, lead, desire_state off by 100 %+; within 1 % then)
        _MODELS["big"] = Stateful(ROOT / ("models/big_driving_supercombo_coreml.onnx" if os.environ.get("ORT_COREML") else
                                          "models/big_driving_supercombo.onnx"))
        _MODELS["ld"] = LDModel()
    if small and "small" not in _MODELS:
        _MODELS["small"] = Stateful(ROOT / "models/driving_supercombo.onnx")
    for k in ("small", "big"):
        if k in _MODELS:
            _MODELS[k].reset()
    return _MODELS.get("small") if small else None, _MODELS["big"], _MODELS["ld"]


def label_chunk(route: Path, rec, seg_dirs, warm_dir, dst: Path, small=True, rpy_delta=(0.0, 0.0, 0.0), offset_delta=0.0,
                rate=1, desire_inject=0.0, seed=0):
    """Label the consecutive segments seg_dirs into dst; warm_dir (the segment before) only rebuilds the models'
    history queues: its last WARMUP_FRAMES frames run through the models and are not saved. rpy_delta (rad),
    offset_delta (m), rate: an augmented replay (main())."""
    import cv2
    cs, ms, loc, _, _ = rec
    dst.mkdir(parents=True, exist_ok=True)
    for f in dst.glob("*.tmp.npy"):
        f.unlink()
    segments = [rr.read_segment_index(d) for d in seg_dirs]
    warm = [rr.read_segment_index(warm_dir)] if warm_dir is not None else []
    n_warm = len(warm[0].frames) if warm else 0
    n_total = (sum(len(s.frames) for s in segments) + rate - 1) // rate      # frames through the models
    ms_t = ms["capture_timestamp_ns"].astype(np.int64)
    order = np.argsort(ms_t)
    ms, ms_t = ms[order], ms_t[order]
    cs_t = cs["timestamp_ns"].astype(np.int64)
    loc_t = loc["timestamp_ns"].astype(np.int64) if len(loc) else np.zeros(0, np.int64)   # some routes have no Localization
    fx, fy, cx, cy = maixcam2_intrinsics(1280, 720)
    k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])

    small, big, ld = models(small)
    warp = WarpPair(1280, 720)
    frames = np.lib.format.open_memmap(dst / "frames.tmp.npy", mode="w+", dtype=np.uint8,
                                       shape=((n_total + 1) // 2, 2, 6, 128, 256))
    kept = []
    rows = {k_: [] for k_ in ("t_ns", "big", "small", "ld_raw", "ld_lanes", "ld_lanes_centre", "ld_lane_prob", "ld_edges",
                             "rec_idx", "desire", "pulse", "v_ego", "yaw_rate", "engaged", "rpy", "mount")}
    seg_of = np.concatenate([np.full(len(s.frames), int(d.name)) for s, d in zip(segments, seg_dirs)])[::rate]
    prev = np.zeros(8, np.float32)
    rng, inj_d, inj_left = np.random.default_rng(seed), 0, 0
    t0 = time.time()
    for g, (frec, y, u, v) in enumerate(rr.decode_route_yuv(warm + segments)):
        n = g - n_warm                                   # frame number inside the chunk
        if n < -WARMUP_FRAMES * rate or n % rate:
            continue
        n //= rate                                       # the frame's number in the (replayed) stream
        t = int(frec["capture_timestamp_ns"])
        j = int(np.clip(np.searchsorted(ms_t, t), 0, len(ms_t) - 1))
        if j > 0 and abs(ms_t[j - 1] - t) < abs(ms_t[j] - t):
            j -= 1
        m = ms[j]
        rpy = np.round(np.array([m["calibration"]["roll"], m["calibration"]["pitch"], m["calibration"]["yaw"]],
                                np.float64) + rpy_delta, 4)
        mount = (round(float(m["camera_offset_m"]) + offset_delta, 3), round(float(m["camera_height_m"]), 3))
        warp.set_calibration(rpy, *mount)
        med, sbig = warp.warp(y, u, v)
        kc = int(np.clip(np.searchsorted(cs_t, t) - 1, 0, len(cs_t) - 1))
        d = int(cs["desire"][kc])
        if desire_inject > 0:            # counterfactual lane-change requests (--desire-inject)
            if inj_left > 0:
                d, inj_left = (inj_d if d == 0 else d), inj_left - 1
            elif d == 0 and float(cs["ego_speed_kph"][kc]) > 8 * 3.6 and rng.random() < 1.0 / (desire_inject * 20 / rate):
                inj_d, inj_left = int(rng.choice([3, 4])), int(3 * 20 / rate)
                d = inj_d
        cur = np.zeros(8, np.float32)
        if d > 0:
            cur[d] = 1.0
        pulse = np.where(cur - prev > 0.99, cur, 0.0).astype(np.float32)
        prev = cur
        y_small = small.step(med, sbig, pulse).astype(np.float32) if small else np.full(2576, np.nan, np.float32)
        y_big = big.step(med, sbig, pulse)[:2066].astype(np.float32)
        if n < 0:
            continue
        rows["small"].append(y_small)
        rows["big"].append(y_big)
        bgr = cv2.cvtColor(np.concatenate([y.reshape(-1), u.reshape(-1), v.reshape(-1)]).reshape(1080, 1280),
                           cv2.COLOR_YUV2BGR_I420)
        raw = ld.run(roi_input(bgr, LD_ROI))
        rows["ld_raw"].append(np.concatenate([raw[name] for name in ld.names]).astype(np.float32))
        lanes, centre, prob, edges = ld_layout(ld.decode(raw), k, rpy, mount[1], mount[0])
        rows["ld_lanes"].append(lanes)
        rows["ld_lanes_centre"].append(centre)
        rows["ld_lane_prob"].append(prob)
        rows["ld_edges"].append(edges)
        rows["t_ns"].append(t)
        rows["rec_idx"].append(j if abs(int(ms_t[j]) - t) < 30_000_000 else -1)
        rows["desire"].append(d)
        rows["pulse"].append(pulse)
        rows["v_ego"].append(float(cs["ego_speed_kph"][kc]) / 3.6)
        rows["engaged"].append(int(cs["engaged"][kc]))
        kl = int(np.clip(np.searchsorted(loc_t, t), 0, len(loc_t) - 1))
        rows["yaw_rate"].append(float(loc["angular_velocity_calib"][kl, 2]) if len(loc_t) else np.nan)
        rows["rpy"].append(rpy.astype(np.float32))
        rows["mount"].append(np.array(mount, np.float32))
        if n % 2 == 0:
            frames[n // 2, 0], frames[n // 2, 1] = med, sbig
            kept.append(n)
        if n % 2400 == 0:
            print(f"{dst.name}: {n}/{n_total} frames, {time.time() - t0:.0f} s", flush=True)
    frames.flush()
    del frames
    out = {k_: np.array(v_) for k_, v_ in rows.items()}
    out["seg"] = seg_of[:len(out["t_ns"])]
    rec_idx = out["rec_idx"]
    sel = np.clip(rec_idx, 0, None)
    for f in ("plan", "lanes", "lane_probabilities", "road_edges", "desire_state", "plan_yaw", "plan_yaw_rate",
              "gas_press_probs"):
        a = ms[f][sel].astype(np.float32)
        out["rec_" + f] = np.where((rec_idx >= 0).reshape((-1,) + (1,) * (a.ndim - 1)), a, np.nan)
    for f in ("probability", "x", "y", "velocity"):
        out["rec_lead_" + f] = ms["lead"][f][sel].astype(np.float32)
    np.save(dst / "frames_idx.npy", np.array(kept))
    (dst / "frames.tmp.npy").rename(dst / "frames.npy")
    np.savez(dst / "labels.tmp.npz", **out)
    (dst / "labels.tmp.npz").rename(dst / "labels.npz")      # labels.npz last: its presence marks a finished chunk
    print(f"{dst}: {len(rec_idx)} frames ({len(kept)} warped kept), {(rec_idx >= 0).mean() * 100:.1f}% matched, "
          f"{time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
