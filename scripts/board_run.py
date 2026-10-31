"""Closed-loop run of a quantized student on the MaixCAM2 NPU (pyaxengine). Runs on the board, for board_eval.py.

Same inputs as quant_eval_p2.py (road/wide of kept frames j-2 and j, the last 100 desire pulses pooled 4:1,
features_buffer from the hidden of frames t-96 ... t-4, traffic_convention [1, 0]), but in one pass: frame t's hidden
is stored before the frames that read it. Saves every output tensor per frame and the run() time.

usage (board): python3 board_run.py <model.axmodel> <unit dir: frames.npy, pulse.npy> <out.npz>
"""
import sys
import time

import axengine as ax
import numpy as np

model, unit, out = sys.argv[1:4]
s = ax.InferenceSession(model)
names = [o.name for o in s.get_outputs()]
frames = np.load(f"{unit}/frames.npy", mmap_mode="r")
pulse = np.load(f"{unit}/pulse.npy")
n = len(pulse)
even = np.arange(4, n - n % 2, 2)
hist = np.zeros(((n + 1) // 2, 512), np.float32)
res = {k: [] for k in names}
ms = []
for t in even:
    j = t // 2
    des = np.zeros((100, 8), np.float32)
    lo = max(0, t - 99)
    des[100 - (t + 1 - lo):] = pulse[lo:t + 1]
    idx = np.arange(t - 96, t, 4) // 2
    fb = np.zeros((24, 512), np.float32)
    fb[idx >= 0] = hist[idx[idx >= 0]]
    feed = {"input_imgs": np.ascontiguousarray(np.concatenate([frames[j - 2, 0], frames[j, 0]])[None]),
            "big_input_imgs": np.ascontiguousarray(np.concatenate([frames[j - 2, 1], frames[j, 1]])[None]),
            "desire": des.reshape(25, 4, 8).max(1)[None], "features_buffer": fb[None],
            "traffic_convention": np.array([[1.0, 0.0]], np.float32)}
    t0 = time.perf_counter()
    o = s.run(None, feed)
    ms.append((time.perf_counter() - t0) * 1e3)
    for k, v in zip(names, o):
        res[k].append(np.asarray(v, np.float32).reshape(-1))
    hist[j] = res["out_hidden"][-1]
np.savez(out, even=even, ms=np.array(ms), **{k: np.stack(v) for k, v in res.items()})
print(f"{len(even)} frames, run() {np.median(ms):.2f} ms median, {np.mean(ms):.2f} mean")
